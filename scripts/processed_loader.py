#!/usr/bin/env python
"""Read the processed layer -- with the supervision rules enforced, not documented.

    from scripts.processed_loader import iter_rows, reason_code_rows

    for row in reason_code_rows("factify2", split="train", purpose="train"):
        ...

``iter_rows`` is the plain reader. ``reason_code_rows`` is the only sanctioned
way to feed a reason-code head, and it refuses three things:

1. a dataset whose label scheme carries no reason information (every row
   ``unmapped``) -- asked for, it raises rather than returning nothing, so a
   misconfigured run fails at the first call instead of training on zero rows;
2. any row whose reason_code is ``unmapped`` or null -- never yielded, for
   training or for evaluation;
3. any ``evaluation_only`` row when the purpose is training -- AVerImaTeC and
   VERITE may score a model, never fit one.

Every row it yields has passed ``check_reason_row``, which raises on a
violation; the filter and the check are the same rule applied twice, so a
future edit to one cannot quietly disable the other.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import PROCESSED  # noqa: E402
from scripts import reason_codes  # noqa: E402

PURPOSES = ("train", "evaluate")


class SupervisionError(ValueError):
    """A row or dataset that must not supervise the reason-code head."""


def records_path(dataset: str, root: Path | None = None) -> Path:
    return (root or PROCESSED / "records") / f"{dataset}.jsonl.gz"


def iter_rows(dataset: str, *, split: str | None = None,
              root: Path | None = None) -> Iterator[dict[str, Any]]:
    with gzip.open(records_path(dataset, root), "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split is None or row["split"] == split:
                yield row


def check_reason_row(row: dict[str, Any], purpose: str) -> dict[str, Any]:
    """Raise unless this row may supervise the reason-code head for ``purpose``."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, not {purpose!r}")
    code = row.get("reason_code")
    if code == reason_codes.UNMAPPED:
        raise SupervisionError(
            f"{row['dataset']}/{row['record_id']}: reason_code 'unmapped' -- the "
            "dataset's labels carry no reason information; never train or evaluate on it")
    if not reason_codes.is_supervised(code):
        raise SupervisionError(
            f"{row['dataset']}/{row['record_id']}: reason_code {code!r} is not a "
            f"supervised code {reason_codes.SUPERVISED_CODES}")
    if purpose == "evaluate" and row.get("weak_supervision"):
        raise SupervisionError(
            f"{row['dataset']}/{row['record_id']}: weak (synthetic) supervision may train "
            "but never evaluate; VERITE out-of-context is the media_mismatch evaluation set")
    if purpose == "train" and row.get("evaluation_only"):
        raise SupervisionError(
            f"{row['dataset']}/{row['record_id']}: evaluation-only corpus used for training")
    if purpose == "train" and row.get("split") != "train":
        raise SupervisionError(
            f"{row['dataset']}/{row['record_id']}: split {row.get('split')!r} used for training")
    return row


def reason_code_rows(dataset: str, *, split: str, purpose: str,
                     root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Rows that may train (purpose='train') or evaluate a reason-code head."""
    if dataset in reason_codes.UNMAPPED_DATASETS:
        raise SupervisionError(
            f"{dataset}: its label scheme carries no reason information; every row is "
            "'unmapped' and none may supervise the reason-code head")
    if purpose == "train" and reason_codes.evaluation_only(dataset):
        raise SupervisionError(
            f"{dataset} is evaluation-only ({reason_codes.EVALUATION_ONLY[dataset]})")
    for row in iter_rows(dataset, split=split, root=root):
        if not reason_codes.is_supervised(row.get("reason_code")):
            continue
        if purpose == "evaluate" and row.get("weak_supervision"):
            continue
        yield check_reason_row(row, purpose)
