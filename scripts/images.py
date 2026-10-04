#!/usr/bin/env python
"""Which image files on disk are usable: the three gates, applied to files.

    python scripts/images.py index --dataset factify2      # incremental
    python scripts/images.py index --all --workers 12
    python scripts/images.py summary --all

A file existing is a fact about the transport. Whether it is an image of
anything is a fact about its bytes, and every coverage number in this project
has to be about the second. So every image file gets one row in
``data/interim/image_index/<dataset>.jsonl``:

    path, bytes, mtime_ns, sha256, format, width, height, mode,
    verifies, loads, flat, error

``verifies`` is PIL's structural check, which is what the fetch gate runs.
``loads`` is a full decode, which catches what ``verify()`` does not: a JPEG cut
off half way through passes ``verify()`` and fails the moment anything reads
its pixels. A file is USABLE only if it loads and its sha256 is not in the
placeholder registry (``data/placeholders.yaml``).

The index is incremental. A row is reused when path, size and mtime are all
unchanged, so re-running after a recovery pass rereads only new files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import FINGERPRINT_REJECTIONS_YAML, INTERIM, PROJECT_ROOT, raw_dir  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
                            ".tif", ".tiff"})

#: Datasets whose images are files on disk. averimatec ships its images inside
#: one zip, indexed member by member as ``images.zip::<member>``.
IMAGE_DATASETS = ("verite", "factify2", "fakeddit", "averimatec", "m4fc", "mocheg", "dgm4")

INDEX_DIR = INTERIM / "image_index"

#: Status of a file, in the order the gates are applied.
USABLE = "usable"
UNDECODABLE = "undecodable"


#: Datasets whose images ship inside zip archives, indexed member by member
#: as ``<archive>::<member>``. Globs are relative to the dataset's raw_dir().
ZIP_SOURCES: dict[str, tuple[str, ...]] = {
    "averimatec": ("images.zip",),
    "dgm4": ("origin/*.zip", "manipulation/*.zip"),
}


def zip_archives(name: str) -> list[Path]:
    root = raw_dir(name)
    return sorted(p for g in ZIP_SOURCES.get(name, ()) for p in root.glob(g) if p.is_file())


def image_roots(name: str) -> list[Path]:
    """Directories holding a dataset's image files (every provenance source)."""
    root = raw_dir(name)
    if name in ZIP_SOURCES:
        return []
    if name == "mocheg":
        # the image-bearing release, streamed out of its 74.3 GB archive
        return [p for p in [root / "with_tweet_2023_03" / "images"] if p.is_dir()]
    if name == "verite":
        base = root / "data" / "VERITE"
        return sorted(p for p in base.glob("images*") if p.is_dir())
    media = root / "media"
    return sorted(p for p in media.glob("images*") if p.is_dir())


def image_files(name: str) -> Iterator[Path]:
    """Every file in an image directory, whatever its name says.

    Selection by extension would trust the name over the bytes: MOCHEG holds
    20 ``.jfif`` files and 2 named ``*.jpg%0D%0A`` (a CRLF leaked into the
    source URL), all valid JPEGs that an extension filter silently skipped.
    The gates decide what is an image; only our own temp files are excluded.
    """
    for directory in image_roots(name):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.endswith(".part"):
                yield path


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# --------------------------------------------------------------------------
# examining one file
# --------------------------------------------------------------------------


def examine_bytes(body: bytes) -> dict[str, Any]:
    """Hash, structural check and full decode of one image's bytes."""
    from PIL import Image

    row: dict[str, Any] = {
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "format": None, "width": None, "height": None, "mode": None,
        "verifies": False, "loads": False, "flat": None, "error": None,
    }
    if not body:
        row["error"] = "empty_body"
        return row
    try:
        with Image.open(io.BytesIO(body)) as img:
            row["format"], row["mode"] = img.format, img.mode
            row["width"], row["height"] = img.size
            img.verify()
        row["verifies"] = True
    except Exception as exc:  # noqa: BLE001 -- the reason IS the measurement
        row["error"] = f"verify:{type(exc).__name__}"
        row["head"] = body[:80].decode("utf-8", "replace")
        return row
    try:
        # verify() leaves the image unusable; reopen for the full decode.
        with Image.open(io.BytesIO(body)) as img:
            img.load()
            extrema = img.convert("RGB").getextrema()
        row["loads"] = True
        row["flat"] = all(lo == hi for lo, hi in extrema)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"load:{type(exc).__name__}"
    return row


def _examine_path(args: tuple[str, int, int]) -> dict[str, Any]:
    relpath, size, mtime_ns = args
    path = PROJECT_ROOT / relpath
    try:
        body = path.read_bytes()
    except OSError as exc:
        return {"path": relpath, "bytes": size, "mtime_ns": mtime_ns,
                "sha256": None, "verifies": False, "loads": False,
                "error": f"read:{type(exc).__name__}"}
    row = examine_bytes(body)
    row.update(path=relpath, mtime_ns=mtime_ns)
    return row


def _zip_rows(archive_rel: str) -> list[dict[str, Any]]:
    """Every image member of one archive, examined in memory. One per process."""
    archive = PROJECT_ROOT / archive_rel
    stamp = archive.stat().st_mtime_ns
    out = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            row = examine_bytes(zf.read(info))
            row.update(path=f"{archive_rel}::{info.filename}", mtime_ns=stamp)
            out.append(row)
    return out


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


def index_path(name: str) -> Path:
    return INDEX_DIR / f"{name}.jsonl"


def load_index(name: str) -> dict[str, dict[str, Any]]:
    """path -> row. Empty if the dataset has never been indexed."""
    path = index_path(name)
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["path"]] = row
    return out


def build_index(name: str, workers: int = 8, log=print,
                force: bool = False) -> dict[str, dict[str, Any]]:
    """Examine every image file of ``name``, reusing unchanged rows.

    ``force`` re-examines everything -- the build-time verification, so that
    a file counted usable in the processed layer was decoded during the build
    that counted it, not on some earlier day.
    """
    previous = {} if force else load_index(name)
    rows: dict[str, dict[str, Any]] = {}
    if name in ZIP_SOURCES:
        todo_zips = []
        for archive in zip_archives(name):
            key = rel(archive)
            stamp = archive.stat().st_mtime_ns
            cached = [r for p, r in previous.items()
                      if p.startswith(key + "::") and r.get("mtime_ns") == stamp]
            if cached:
                rows.update({r["path"]: r for r in cached})
            else:
                todo_zips.append(key)
        log(f"  [{name}] {len(rows):,} members unchanged, {len(todo_zips)} archive(s) to examine")
        if todo_zips:
            with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo_zips)))) as pool:
                for archive_rows in pool.map(_zip_rows, todo_zips):
                    rows.update({r["path"]: r for r in archive_rows})
    else:
        todo: list[tuple[str, int, int]] = []
        for path in image_files(name):
            st = path.stat()
            key = rel(path)
            old = previous.get(key)
            if old and old.get("bytes") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
                rows[key] = old
            else:
                todo.append((key, st.st_size, st.st_mtime_ns))
        log(f"  [{name}] {len(rows):,} unchanged, {len(todo):,} to examine")
        if todo:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                for done, row in enumerate(pool.map(_examine_path, todo, chunksize=64), 1):
                    rows[row["path"]] = row
                    if done % 20000 == 0:
                        log(f"    {done:,}/{len(todo):,}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = index_path(name).with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], sort_keys=True) + "\n")
    os.replace(tmp, index_path(name))
    return rows


# --------------------------------------------------------------------------
# the gates, applied to an index row
# --------------------------------------------------------------------------


WRONG_IMAGE = "wrong_image"


@lru_cache(maxsize=1)
def fingerprint_rejections() -> dict[tuple[str, str], dict[str, Any]]:
    """(dataset, sha256) -> entry, from data/fingerprint_rejections.yaml.

    Keyed by the BYTES within one dataset, never by hash alone across corpora:
    the same meme may be genuine content in another corpus. Not keyed by path,
    because a VERITE path (true_N / false_N) is its label; entries carry the
    record rows and a label-free file alias instead (records.verite_alias).
    """
    import yaml

    if not FINGERPRINT_REJECTIONS_YAML.is_file():
        return {}
    with FINGERPRINT_REJECTIONS_YAML.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    return {(e["dataset"], e["sha256"]): e for e in doc.get("rejections") or []}


def dataset_of(path: str) -> str | None:
    """``data/raw/<dataset>/...`` -> dataset."""
    parts = (path or "").split("/")
    return parts[2] if len(parts) > 2 and parts[:2] == ["data", "raw"] else None


def status(row: dict[str, Any] | None) -> str:
    """USABLE, UNDECODABLE, the registry status (placeholder/furniture/blank),
    or WRONG_IMAGE for a file listed in data/fingerprint_rejections.yaml.

    ``None`` (no row) is reported as UNDECODABLE: a file nobody has examined has
    not passed the gates, and an unexamined file is not counted as an image.
    """
    from scripts.placeholders import registry, MISSING_STATUSES

    if not row or not row.get("loads"):
        return UNDECODABLE
    entry = registry().get(row.get("sha256") or "")
    if entry and entry["status"] in MISSING_STATUSES:
        return entry["status"]
    if (dataset_of(row.get("path") or ""), row.get("sha256")) in fingerprint_rejections():
        return WRONG_IMAGE
    return USABLE


def is_usable(row: dict[str, Any] | None) -> bool:
    return status(row) == USABLE


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        s = status(row)
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))


def rejected_dir(name: str) -> Path:
    return INTERIM / f"{name}_rejected"


def quarantine_undecodable(name: str, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Move every file that does not decode out of data/raw, never deleting it.

    The same treatment verite's six HTML error pages got on 2026-09-21: the
    file goes to data/interim/<name>_rejected/ under its own name, and
    ``_why.json`` there records where it came from, how big it was, its
    sha256, the decoder's complaint and its first bytes. Nothing is destroyed;
    the evidence for why the gate exists is kept.
    """
    import shutil

    index = load_index(name)
    bad = [row for row in index.values()
           if not row.get("loads") and "::" not in row["path"]]
    if dry_run or not bad:
        return bad
    target = rejected_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    why_path = target / "_why.json"
    why = json.loads(why_path.read_text(encoding="utf-8")) if why_path.is_file() else {
        "reason": ("Files that do not decode as images, written with an image "
                   "name by a hydration pass that accepted any non-empty HTTP "
                   "200. Quarantined, not deleted."),
        "files": []}
    moved = []
    for row in sorted(bad, key=lambda r: r["path"]):
        source = PROJECT_ROOT / row["path"]
        if not source.is_file():
            continue
        dest = target / source.name
        n = 2
        while dest.exists():
            dest = target / f"{source.stem}.{n}{source.suffix}"
            n += 1
        shutil.move(str(source), str(dest))
        entry = {"file": dest.name, "from": row["path"], "bytes": row.get("bytes"),
                 "sha256": row.get("sha256"), "error": row.get("error"),
                 "head": row.get("head"), "moved_on": utcnow()[:10]}
        why["files"].append(entry)
        moved.append(entry)
    why_path.write_text(json.dumps(why, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    build_index(name, workers=2, log=lambda *_: None)
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("quarantine", help="move undecodable files to interim/<name>_rejected/")
    q.add_argument("--dataset", action="append", dest="datasets", required=True,
                   choices=IMAGE_DATASETS)
    q.add_argument("--dry-run", action="store_true")
    for command in ("index", "summary"):
        p = sub.add_parser(command)
        target = p.add_mutually_exclusive_group(required=True)
        target.add_argument("--dataset", action="append", dest="datasets",
                            choices=IMAGE_DATASETS)
        target.add_argument("--all", action="store_true")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--force", action="store_true",
                       help="re-examine every file, ignoring cached rows")
    args = parser.parse_args(argv)
    if args.command == "quarantine":
        for name in args.datasets:
            moved = quarantine_undecodable(name, dry_run=args.dry_run)
            verb = "would move" if args.dry_run else "moved"
            print(f"[{name}] {verb} {len(moved)} undecodable file(s) -> "
                  f"{rejected_dir(name).relative_to(PROJECT_ROOT).as_posix()}/")
        return 0
    names = list(IMAGE_DATASETS) if args.all else args.datasets
    for name in names:
        rows = (build_index(name, args.workers, force=args.force)
                if args.command == "index" else load_index(name))
        print(f"[{name}] {len(rows):,} files  {summarise(rows.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
