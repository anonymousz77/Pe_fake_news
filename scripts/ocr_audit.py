#!/usr/bin/env python
"""Read the text in single-copy images and flag error cards.

    python scripts/ocr_audit.py --dataset factify2 --dataset fakeddit
    python scripts/ocr_audit.py --dataset verite --limit 50     # smoke test
    python scripts/ocr_audit.py --report

The duplicate-group review catches a placeholder the moment a host serves it
twice. A placeholder served ONCE has no duplicate to give it away, so the only
signal left is what it says: "image unavailable", "404 not found", "this image
license has expired". This reads every single-copy usable image with EasyOCR
and records the text; images whose text matches ERROR_WORDS are flagged for
review. Flagging is not registering -- "error" also appears on real protest
signs and screenshots -- so flagged images still go through visual review.

Resumable: one row per file in data/interim/ocr/<dataset>.jsonl, keyed by
sha256; a re-run skips what it has read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT  # noqa: E402
from scripts import images  # noqa: E402

OCR_DIR = INTERIM / "ocr"

#: The words the brief names, plus the forms hosts actually use for them.
ERROR_WORDS = re.compile(
    r"\b(unavailable|not\s+available|no\s+longer\s+available|not\s+found|404|403|"
    r"error|expired|removed|forbidden|deleted|does\s+not\s+exist|no\s+image|"
    r"image\s+not)\b", re.I)

MAX_SIDE = 640


def single_copy_rows(name: str) -> list[dict]:
    """Usable images whose sha256 occurs exactly once across every index."""
    counts: Counter = Counter()
    for ds in images.IMAGE_DATASETS:
        for row in images.load_index(ds).values():
            if row.get("sha256"):
                counts[row["sha256"]] += 1
    return [r for r in images.load_index(name).values()
            if images.is_usable(r) and counts[r["sha256"]] == 1]


def few_copy_rows(name: str, low: int = 2, high: int = 4) -> list[dict]:
    """One usable file per sha256 with ``low``..``high`` copies, owned by the
    first dataset (IMAGE_DATASETS order) that holds it. Groups of five or more
    were reviewed by eye (data/placeholders.yaml v2 and v7); these sat between
    that review and the single-copy pass."""
    counts: Counter = Counter()
    owner: dict[str, tuple[str, dict]] = {}
    for ds in images.IMAGE_DATASETS:
        for row in images.load_index(ds).values():
            if row.get("sha256") and images.is_usable(row):
                counts[row["sha256"]] += 1
                owner.setdefault(row["sha256"], (ds, row))
    return [row for h, (ds, row) in owner.items() if ds == name and low <= counts[h] <= high]


def load_image(relpath: str):
    import io
    import zipfile

    from PIL import Image

    if "::" in relpath:
        archive, member = relpath.split("::", 1)
        with zipfile.ZipFile(PROJECT_ROOT / archive) as zf:
            img = Image.open(io.BytesIO(zf.read(member)))
    else:
        img = Image.open(PROJECT_ROOT / relpath)
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def run(name: str, limit: int | None = None, few_copies: bool = False) -> dict:
    import numpy as np
    import easyocr

    out_path = OCR_DIR / f"{name}.jsonl"
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["sha256"])
    rows = few_copy_rows(name) if few_copies else single_copy_rows(name)
    todo = [r for r in rows if r["sha256"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{name}] {len(done):,} already read, {len(todo):,} to read", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    flagged = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(todo, 1):
            try:
                text = " ".join(reader.readtext(np.asarray(load_image(row["path"])),
                                                detail=0, paragraph=True))
                error = None
            except Exception as exc:  # noqa: BLE001 -- record and move on
                text, error = "", f"{type(exc).__name__}: {exc}"[:200]
            match = ERROR_WORDS.findall(text)
            flagged += bool(match)
            handle.write(json.dumps({"sha256": row["sha256"], "path": row["path"],
                                     "text": text[:500], "matched": sorted(
                                         {m.lower() for m in match}),
                                     "error": error}, ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                handle.flush()
                print(f"  {n:,}/{len(todo):,}  flagged {flagged:,}", flush=True)
    return {"read": len(todo), "flagged_this_run": flagged}


def report() -> dict:
    out = {}
    for path in sorted(OCR_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[path.stem] = {"read": len(rows),
                          "with_text": sum(1 for r in rows if r["text"].strip()),
                          "flagged": sum(1 for r in rows if r["matched"]),
                          "errors": sum(1 for r in rows if r["error"]),
                          "by_word": dict(Counter(w for r in rows for w in r["matched"]).most_common())}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", dest="datasets",
                        choices=images.IMAGE_DATASETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--few-copies", action="store_true",
                        help="read images with 2-4 copies (one per hash) instead of single copies")
    args = parser.parse_args(argv)
    if args.report:
        print(json.dumps(report(), indent=2))
        return 0
    if not args.datasets:
        parser.error("--dataset or --report is required")
    for name in args.datasets:
        print(json.dumps(run(name, args.limit, args.few_copies)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
