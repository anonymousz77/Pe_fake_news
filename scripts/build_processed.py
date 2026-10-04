#!/usr/bin/env python
"""Build the modelling-ready layer: raw -> interim -> processed, one schema.

    python scripts/build_processed.py interim --all          # raw -> interim records
    python scripts/build_processed.py processed --reverify   # interim -> processed

Stage ``interim`` reads data/raw through scripts/records.py and writes one
normalised record per line to data/interim/records/<dataset>.jsonl.gz. Stage
``processed`` reads ONLY those files plus the image index, so the derivation
is strictly raw -> interim -> processed.

Every row of the processed layer has the same fields (docs/processed_schema.md):

    dataset, record_id, text, evidence_text, label, reason_code,
    modality_evidence, evaluation_only, split, image_paths, images_expected,
    images_usable, image_status, modality, usable, usable_reason, meta

``reason_code`` and ``modality_evidence`` are derived from the label by
scripts/reason_codes.py; a label with no clean mapping gets null and is listed
in the report rather than forced.

An image is in ``image_paths`` only if it passed all three gates AT BUILD TIME:
``--reverify`` re-decodes every image file before resolving (the default
reuses index rows whose size and mtime are unchanged). Splits come from
scripts/split_all.py, which keeps every duplicate group of five or more copies,
and every cross-corpus duplicate, inside one split, and refuses otherwise.

Outputs:
    data/processed/records/<dataset>.jsonl.gz   rows (gitignored)
    data/processed/splits/<dataset>_<split>.csv record_id,split ONLY (published)
    data/reports/processed.json                 aggregates (published)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROCESSED, PROJECT_ROOT, REPORTS, SPLITS  # noqa: E402
from scripts import images, reason_codes, recover, split_all  # noqa: E402
from scripts.confound import domain_of  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402
from scripts.placeholders import MISSING_STATUSES  # noqa: E402
from scripts.hydrate import Item  # noqa: E402
from scripts.records import LOADERS  # noqa: E402

INTERIM_RECORDS = INTERIM / "records"
PROCESSED_RECORDS = PROCESSED / "records"

#: Records the register says must not enter the pipeline but that cannot be
#: excluded by path: m4fc's test rows sit in the same file as its train rows.
EXCLUDED_SPLITS = {"m4fc": {"test"}}

#: Image roles that form an optional SET rather than a required slot. MOCHEG's
#: evidence images are whatever the fact-check article carried; one that the
#: registry removes (a logo, a placeholder, the verdict graphic itself) or that
#: is provably the wrong image is EXCLUDED from the set, not a missing piece of
#: the record. A claim / document / pair / post image is a required slot: if it
#: is anything but usable, the record is missing an image.
OPTIONAL_ROLES = frozenset({"evidence"})
REMOVED_STATUSES = MISSING_STATUSES | {images.WRONG_IMAGE}


# --------------------------------------------------------------------------
# stage 1: raw -> interim
# --------------------------------------------------------------------------


def write_interim(name: str) -> int:
    INTERIM_RECORDS.mkdir(parents=True, exist_ok=True)
    path = INTERIM_RECORDS / f"{name}.jsonl.gz"
    tmp = path.with_suffix(".gz.part")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for record in LOADERS[name]():
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_interim(name: str) -> Iterator[dict[str, Any]]:
    with gzip.open(INTERIM_RECORDS / f"{name}.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


# --------------------------------------------------------------------------
# stage 2: interim -> processed
# --------------------------------------------------------------------------


def load_dead(name: str) -> dict[str, str]:
    """key -> terminal reason, from the dead-images manifest if it exists."""
    path = recover.RECOVERY / f"{name}_dead.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = row["terminal_reason"]
    return out


class Resolver:
    """Image reference -> (status, path, sha256), from the gate-checked index."""

    def __init__(self, names: list[str]):
        self.index = {n: images.load_index(n) for n in names}
        self.recorded = {n: recover.ledger_paths(n) for n in names if n in recover.DATASETS}
        self.matches = {n: recover.load_matches(n) for n in names if n in recover.DATASETS}
        self.dead = {n: load_dead(n) for n in names}

    def resolve(self, name: str, ref: dict[str, Any], label: str | None
                ) -> tuple[str, str | None, str | None]:
        idx = self.index.get(name, {})
        if ref["kind"] == "file":
            row = idx.get(ref["ref"])
            status = images.status(row) if row else "not_on_disk"
            if status == images.USABLE:
                return status, ref["ref"], row["sha256"]
            return status, None, None
        item = Item(ref["ref"], ref.get("url") or "", label or "")
        path = recover.resolve_image(name, item, idx, self.recorded.get(name),
                                     self.matches.get(name))
        if path is not None:
            relpath = images.rel(path)
            row = idx.get(relpath) or (self.matches.get(name, {}).get(ref["ref"]) or {}).get("row")
            return images.USABLE, relpath, (row or {}).get("sha256")
        reason = self.dead.get(name, {}).get(ref["ref"])
        if reason is None and not str(ref.get("url") or "").startswith("http"):
            # the dataset's own URL is malformed (m4fc ships one "Nhttps://..."):
            # never fetchable, so never in the crawl's dead manifest
            reason = "malformed_url_in_dataset"
        return "dead:" + (reason or "missing"), None, None


def missing_images(statuses: list[dict[str, str]]) -> int:
    """Required images that are not usable (an optional-set member that was
    removed is excluded from its set, not missing)."""
    return sum(1 for s in statuses if s["status"] != images.USABLE
               and not (s["role"] in OPTIONAL_ROLES and s["status"] in REMOVED_STATUSES))


def modality(text: str | None, usable_images: int) -> str:
    has_text = bool((text or "").strip())
    if has_text and usable_images:
        return "text+image"
    if has_text:
        return "text_only"
    return "image_only" if usable_images else "none"


def build(names: list[str], reverify: bool, workers: int) -> dict[str, Any]:
    started = utcnow()
    image_names = [n for n in names if n in images.IMAGE_DATASETS]
    for n in image_names:
        images.build_index(n, workers=workers, force=reverify,
                           log=lambda *a: print(*a, flush=True))
    verified_at = utcnow()
    resolver = Resolver(names)
    copies: Counter = Counter()
    for n in images.IMAGE_DATASETS:
        for row in (resolver.index.get(n) or images.load_index(n)).values():
            if images.is_usable(row):
                copies[row["sha256"]] += 1

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    split_inputs: dict[tuple[str, str], split_all.SplitInput] = {}
    excluded = Counter()
    for name in names:
        for rec in read_interim(name):
            if rec["official_split"] in EXCLUDED_SPLITS.get(name, set()):
                excluded[name] += 1
                continue
            paths, hashes, statuses, domains = [], [], [], set()
            for ref in rec["images"]:
                status, path, digest = resolver.resolve(name, ref, rec["label"])
                statuses.append({"role": ref["role"], "status": status})
                if path:
                    paths.append(path)
                    if digest:
                        hashes.append(digest)
                if name == "factify2" and ref.get("url"):
                    domains.add(domain_of(ref["url"]))
            # The author's split survives here even where split_all overrides it
            # (averimatec is evaluation-only: every row test, train/val kept).
            meta = {**(rec.get("meta") or {}), "official_split": rec["official_split"]}
            reasons = []
            if not rec["label"]:
                reasons.append("no_label")
            if meta.get("label_conflict"):
                reasons.append("label_conflict")
            if not (rec["text"] or "").strip():
                reasons.append("no_text")
            missing = missing_images(statuses)
            if missing:
                reasons.append(f"images_missing:{missing}")
            key = (name, rec["record_id"])
            rows[key] = {
                "dataset": name, "record_id": rec["record_id"], "text": rec["text"],
                "evidence_text": rec.get("evidence_text"), "label": rec["label"],
                "reason_code": reason_codes.reason_code(name, rec["label"], meta),
                "modality_evidence": reason_codes.modality_evidence(name, rec["label"]),
                "evaluation_only": reason_codes.evaluation_only(name),
                "weak_supervision": reason_codes.weak_supervision(
                    name, reason_codes.reason_code(name, rec["label"], meta)),
                "image_paths": paths, "images_expected": len(rec["images"]),
                "images_usable": len(paths), "image_status": statuses,
                "modality": modality(rec["text"], len(paths)),
                "usable": not reasons, "usable_reason": ";".join(reasons) or None,
                "meta": meta,
            }
            split_inputs[key] = split_all.SplitInput(
                rec["label"], rec["official_split"], hashes, domains,
                text_key=(rec["text"] if name in split_all.TEXT_GROUPED_DATASETS else None))

    placement, split_report = split_all.assign_all(split_inputs, copies)
    for key, split in placement.items():
        rows[key]["split"] = split

    PROCESSED_RECORDS.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (ds, _), row in rows.items():
        by_ds[ds].append(row)
    report: dict[str, Any] = {"built_at": utcnow(), "started_at": started,
                              "images_verified_at": verified_at, "reverified": reverify,
                              "excluded_by_register": dict(excluded),
                              "splits": split_report, "datasets": {}}
    for ds, ds_rows in sorted(by_ds.items()):
        ds_rows.sort(key=lambda r: r["record_id"])
        path = PROCESSED_RECORDS / f"{ds}.jsonl.gz"
        tmp = path.with_suffix(".gz.part")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for row in ds_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        for split in ("train", "val", "test"):
            # record_id and split ONLY: labels, text and URLs stay unpublished.
            with (SPLITS / f"{ds}_{split}.csv").open("w", encoding="utf-8", newline="") as h:
                h.write("record_id,split\n")
                for row in ds_rows:
                    if row["split"] == split:
                        rid = row["record_id"]
                        h.write(f"\"{rid}\",{split}\n" if ("," in rid or '"' in rid) else f"{rid},{split}\n")
        report["datasets"][ds] = summarise(ds_rows)
    report["reason_codes"] = reason_codes.report(
        {ds: Counter(r["label"] for r in ds_rows) for ds, ds_rows in by_ds.items()})
    report["modality_evidence"] = {
        ds: dict(Counter(str(r["modality_evidence"]) for r in ds_rows))
        for ds, ds_rows in sorted(by_ds.items())}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "processed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def summarise(ds_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_class: dict[str, Counter] = defaultdict(Counter)
    for row in ds_rows:
        c = per_class[str(row["label"])]
        c["rows"] += 1
        c["usable"] += row["usable"]
        c["images_expected"] += row["images_expected"]
        c["images_usable"] += row["images_usable"]
        c[f"split_{row['split']}"] += 1
        c[f"reason_{row['reason_code']}"] += 1
        c[f"modality_{row['modality']}"] += 1
        for s in row["image_status"]:
            if s["status"] != images.USABLE:
                c[f"image_{s['status']}"] += 1
    out = {}
    for label, c in sorted(per_class.items()):
        d = dict(sorted(c.items()))
        d["image_coverage"] = round(c["images_usable"] / c["images_expected"], 4) \
            if c["images_expected"] else None
        out[label] = d
    total = Counter()
    for c in per_class.values():
        total.update(c)
    return {"total": dict(sorted(total.items())), "per_class": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("interim")
    s2 = sub.add_parser("processed")
    for p in (s1, s2):
        p.add_argument("--dataset", action="append", dest="datasets", choices=sorted(LOADERS))
        p.add_argument("--all", action="store_true")
    s2.add_argument("--reverify", action="store_true",
                    help="re-decode every image file before resolving (the build-time check)")
    s2.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    names = sorted(LOADERS) if args.all or not args.datasets else args.datasets
    t = time.time()
    if args.stage == "interim":
        for name in names:
            print(f"[{name}] {write_interim(name):,} records -> data/interim/records/", flush=True)
    else:
        report = build(names, args.reverify, args.workers)
        for ds, info in report["datasets"].items():
            t_ = info["total"]
            print(f"[{ds}] rows {t_.get('rows', 0):,}  usable {t_.get('usable', 0):,}  "
                  f"images {t_.get('images_usable', 0):,}/{t_.get('images_expected', 0):,}")
        print(json.dumps(report["splits"].get("moved_by_reconciliation"), indent=1))
    print(f"done in {time.time() - t:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
