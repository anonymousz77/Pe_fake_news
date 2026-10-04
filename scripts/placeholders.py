#!/usr/bin/env python
"""The placeholder registry: images that decode but carry no evidence.

A fetch that validates on status code has not validated anything, and one that
validates on *decodability* has only moved the goalposts: imgur's "this image
does not exist" bitmap is a perfectly well-formed PNG. 2,899 copies of it sat
in fakeddit counted as recovered images, every one of them in label 4.

So the third gate is content identity. ``data/placeholders.yaml`` lists sha256
digests of bytes we have seen and judged evidence-free, and anything matching
is a MISSING IMAGE -- at fetch time (rejected) and in every coverage number
(not counted).

Statuses are in the YAML. ``generic_stock`` is recorded but deliberately does
NOT count as missing: a photograph of a real building is a real photograph,
and the line between "topic illustration" and "evidence" is a judgement we
would rather leave visible than bake into a number.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import CROSS_CORPUS_YAML, PLACEHOLDERS_YAML  # noqa: E402

#: Statuses that mean "we do not have this image".
MISSING_STATUSES = frozenset({"placeholder", "furniture", "blank", "verdict"})

#: ``verdict``: the image shows the fact-checker's ruling -- a PolitiFact
#: Truth-O-Meter, a FALSE / MISLEADING stamp, a struck-through screenshot.
#: Whatever else it depicts, a model shown it is shown the label, so it counts
#: as missing. Unlike furniture it is not content-free; it is answer-bearing.

#: Recorded, but not counted as missing.
ADVISORY_STATUSES = frozenset({"generic_stock"})


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    """The registry keyed by sha256. Cached; lazy, like the dataset register."""
    import yaml

    if not PLACEHOLDERS_YAML.is_file():
        return {}
    with PLACEHOLDERS_YAML.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    entries = document.get("entries") or []
    out: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        digest = entry.get("sha256")
        if not digest:
            raise ValueError(f"{PLACEHOLDERS_YAML}: entry {index} has no sha256")
        status = entry.get("status")
        if status not in MISSING_STATUSES | ADVISORY_STATUSES:
            raise ValueError(
                f"{PLACEHOLDERS_YAML}: entry {index} has unknown status {status!r}"
            )
        if digest in out:
            raise ValueError(f"{PLACEHOLDERS_YAML}: duplicate sha256 {digest}")
        out[digest] = entry
    return out


def registry_version() -> int:
    import yaml

    if not PLACEHOLDERS_YAML.is_file():
        return 0
    with PLACEHOLDERS_YAML.open(encoding="utf-8") as handle:
        return int((yaml.safe_load(handle) or {}).get("version", 0))


@lru_cache(maxsize=1)
def missing_hashes() -> frozenset[str]:
    """Digests that count as a missing image."""
    return frozenset(h for h, e in registry().items()
                     if e["status"] in MISSING_STATUSES)


def is_missing(digest: str | None) -> bool:
    """True if these bytes are registered as carrying no evidence."""
    return bool(digest) and digest in missing_hashes()


def describe(digest: str) -> str:
    entry = registry().get(digest)
    return "" if entry is None else f"{entry['status']}: {entry['description']}"


# --------------------------------------------------------------------------
# cross-corpus duplicates
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def cross_corpus_pairs() -> tuple[dict[str, Any], ...]:
    """Records that are byte-identical across two corpora."""
    import yaml

    if not CROSS_CORPUS_YAML.is_file():
        return ()
    with CROSS_CORPUS_YAML.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    return tuple(document.get("pairs") or [])


def excluded_keys(dataset: str, *, against: str) -> frozenset[str]:
    """Processed record_ids in ``dataset`` that duplicate something in ``against``
    (``key``, the hydrate item key, for entries written before version 2).

    An experiment that trains on one corpus and evaluates on the other must
    drop these, or it is scoring itself on images it trained on.
    """
    out: set[str] = set()
    for pair in cross_corpus_pairs():
        members = {m["dataset"]: m for m in pair.get("members", [])}
        if dataset in members and against in members:
            m = members[dataset]
            out.update(m.get("record_ids") or [m["key"]])
    return frozenset(out)


def assert_no_cross_corpus_leak(dataset: str, keys, *, against: str) -> None:
    """Raise if any evaluation key duplicates a record in ``against``.

    Enforced rather than commented: a note in a README does not stop a run.
    """
    leaked = sorted(set(keys) & excluded_keys(dataset, against=against))
    if leaked:
        raise ValueError(
            f"cross-corpus leakage: {len(leaked)} {dataset} record(s) are "
            f"byte-identical to {against} records and must be excluded when "
            f"{against} is on the other side of the split: {leaked}"
        )


# --------------------------------------------------------------------------
# regenerating the cross-corpus list from the processed layer
# --------------------------------------------------------------------------


def find_cross_corpus() -> list[dict[str, Any]]:
    """Every sha256 that a processed record of two or more corpora points at.

    Usable images only (a registry match is already a missing image). Each
    member carries the processed ``record_ids`` that reference the bytes, so an
    experiment can drop them by the id it trains and evaluates on.
    """
    from collections import defaultdict

    from scripts import images
    from scripts.processed_loader import iter_rows, records_path

    sha_of: dict[str, tuple[str, int]] = {}
    for name in images.IMAGE_DATASETS:
        for path, row in images.load_index(name).items():
            if row.get("sha256"):
                sha_of[path] = (row["sha256"], row.get("bytes") or 0)
    members: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"record_ids": set(), "paths": set()}))
    for name in images.IMAGE_DATASETS:
        if not records_path(name).is_file():
            continue
        for row in iter_rows(name):
            for p in row.get("image_paths") or []:
                if p in sha_of:
                    m = members[sha_of[p][0]][name]
                    m["record_ids"].add(row["record_id"])
                    m["paths"].add(p)
    size = {h: b for h, b in sha_of.values()}
    # Paths only where the register says the dataset may be redistributed: a
    # path can carry the label (VERITE's true_N / false_N). record_ids, which
    # the leak guard needs, are label-free for every dataset.
    import yaml

    from configs.paths import DATA

    doc = yaml.safe_load((DATA / "sources.yaml").read_text(encoding="utf-8"))
    entries = doc["datasets"] if isinstance(doc, dict) else doc
    redistributable = {e["name"]: bool(e.get("redistributable")) for e in entries}
    out = []
    for h, by_ds in sorted(members.items()):
        if len(by_ds) < 2:
            continue
        out.append({"sha256": h, "bytes": size.get(h), "kind": "shared_source_image",
                    "datasets": sorted(by_ds),
                    "members": [{"dataset": ds, "record_ids": sorted(m["record_ids"]),
                                 **({"paths": sorted(m["paths"])} if redistributable.get(ds) else {})}
                                for ds, m in sorted(by_ds.items())]})
    return out


def write_cross_corpus(pairs: list[dict[str, Any]]) -> None:
    import datetime

    import yaml

    old = {}
    if CROSS_CORPUS_YAML.is_file():
        old = yaml.safe_load(CROSS_CORPUS_YAML.read_text(encoding="utf-8")) or {}
    header = (
        "# Images that are byte-identical across two or more corpora.\n"
        "#\n"
        "# Generated by `python scripts/placeholders.py cross-corpus` from the processed\n"
        "# layer: every sha256 that usable image references of records in two or more\n"
        "# corpora point at. A shared photograph is genuine -- corpora built from the same\n"
        "# fact-checked events share images -- and it is a leak the moment one corpus is\n"
        "# used for training and the other for evaluation: the model is scored on pixels\n"
        "# it has already seen. Registry matches are missing images and are not listed.\n"
        "#\n"
        "# Splits keep every such group in one split (scripts/split_all.py, live from the\n"
        "# image index); this file is what a CROSS-corpus experiment drops, enforced by\n"
        "# scripts/placeholders.assert_no_cross_corpus_leak(), keyed by processed record_id.\n\n")
    doc = {"version": int(old.get("version", 0)) + 1,
           "updated": datetime.date.today().isoformat(), "pairs": pairs}
    CROSS_CORPUS_YAML.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                                         width=120), encoding="utf-8")
    cross_corpus_pairs.cache_clear()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from collections import Counter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cross-corpus", help="regenerate data/cross_corpus_duplicates.yaml")
    c.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    pairs = find_cross_corpus()
    combos = Counter(" + ".join(p["datasets"]) for p in pairs)
    records = Counter()
    for p in pairs:
        for m in p["members"]:
            records[m["dataset"]] += len(m["record_ids"])
    print(json.dumps({"hashes": len(pairs), "by_corpus_combination": dict(combos.most_common()),
                      "records_involved": dict(records)}, indent=2))
    if not args.dry_run:
        write_cross_corpus(pairs)
        print(f"-> {CROSS_CORPUS_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
