#!/usr/bin/env python
"""Stream a remote .tar.gz and keep only the members we need, under a cap.

    python scripts/stream_archive.py --dataset mocheg --url URL \
        --subdir with_tweet_2023_03 --keep "mocheg/images/*" \
        --keep-small-mb 100 --store-cap-gb 35 --transfer-cap-gb 80

Some archives are larger than the pass may store. MOCHEG's image-bearing
release is one 74.3 GB gzip; the images in it are roughly a third of that.
Downloading the whole file and extracting afterwards would store 74 GB on its
own, so instead the gzip is decompressed as it arrives, each tar member is
written or skipped as it passes, and nothing but the kept members ever lands.

Every member -- kept or not -- is listed in
``data/interim/<dataset>_archive_members.jsonl`` with its size, so what was
skipped is on record rather than merely absent.

Resume is per member: a re-run streams from the start again (a single gzip
stream cannot be entered mid-way) but writes nothing that is already on disk
at the right size. The transfer cap counts the re-stream, so a resume is a
budget decision.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, raw_dir  # noqa: E402
from scripts.fetchlib import FetchError, log_event, make_session, utcnow  # noqa: E402

GB = 1024**3


class Counting:
    """File-like over the raw HTTP stream, counting compressed bytes."""

    def __init__(self, raw, cap_bytes: int):
        self.raw = raw
        self.cap = cap_bytes
        self.n = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.n += len(chunk)
        if self.n > self.cap:
            raise FetchError(f"transfer cap reached at {self.n / GB:.2f} GB")
        return chunk


def safe_member_path(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise FetchError(f"unsafe archive member path {name!r}")
    return p


#: Characters a Windows filename cannot hold, plus '%' so the mapping inverts.
_WINDOWS_UNSAFE = set('<>:"\\|?*%')


def windows_safe(part: str) -> str:
    """Percent-encode what NTFS refuses; urllib.parse.unquote() inverts it.

    MOCHEG names files after their source URLs, so a member can be called
    ``00455-02446-01-1*IiDLaDux.png``. Stored as ``1%2AIiDLaDux.png``, and the
    member listing records both names so nothing has to guess.
    """
    out = "".join(f"%{ord(c):02X}" if c in _WINDOWS_UNSAFE or ord(c) < 32 else c
                  for c in part)
    if out.endswith((".", " ")):
        out = out[:-1] + f"%{ord(out[-1]):02X}"
    return out


def wanted(name: str, size: int, keep: list[str], keep_small: int) -> bool:
    if any(fnmatch.fnmatch(name, pattern) for pattern in keep):
        return True
    return size <= keep_small and not any(
        part in ("images", "videos", "video", "img") for part in PurePosixPath(name).parts)


def stream(name: str, url: str, subdir: str, keep: list[str], keep_small: int,
           store_cap: int, transfer_cap: int, strip: int = 1) -> dict:
    dest_root = raw_dir(name) / subdir
    listing_path = INTERIM / f"{name}_archive_members.jsonl"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    session = make_session(pool_size=1)
    started = time.time()
    kept = skipped = kept_bytes = skipped_bytes = resumed = 0
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        if response.status_code != 200:
            raise FetchError(f"{url}: HTTP {response.status_code}")
        counting = Counting(response.raw, transfer_cap)
        with tarfile.open(fileobj=counting, mode="r|gz") as archive, \
                listing_path.open("w", encoding="utf-8") as listing:
            for member in archive:
                rel = safe_member_path(member.name)
                take = member.isfile() and wanted(member.name, member.size, keep, keep_small)
                stored = PurePosixPath(*[windows_safe(p) for p in rel.parts[strip:]]) \
                    if len(rel.parts) > strip else None
                listing.write(json.dumps({
                    "name": member.name, "size": member.size,
                    "type": member.type.decode(), "kept": take,
                    "stored_as": stored.as_posix() if take and stored else None}) + "\n")
                if not take:
                    if member.isfile():
                        skipped += 1
                        skipped_bytes += member.size
                    continue
                out = dest_root.joinpath(*stored.parts)
                if out.is_file() and out.stat().st_size == member.size:
                    resumed += 1
                    kept += 1
                    kept_bytes += member.size
                    continue
                if kept_bytes + member.size > store_cap:
                    raise FetchError(
                        f"store cap: keeping {member.name} would take this dataset to "
                        f"{(kept_bytes + member.size) / GB:.2f} GB")
                source = archive.extractfile(member)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".part")
                try:
                    with tmp.open("wb") as handle:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                    tmp.replace(out)
                finally:
                    # Our own half-written temp file, never data: an abort
                    # mid-member must not leave a .part in data/raw.
                    tmp.unlink(missing_ok=True)
                kept += 1
                kept_bytes += member.size
                if kept % 5000 == 0:
                    print(f"  {utcnow()}  kept {kept:,} ({kept_bytes / GB:.2f} GB)  "
                          f"skipped {skipped:,} ({skipped_bytes / GB:.2f} GB)  "
                          f"transferred {counting.n / GB:.2f} GB", flush=True)
    result = {"kept_files": kept, "kept_gb": round(kept_bytes / GB, 4),
              "resumed_files": resumed, "skipped_files": skipped,
              "skipped_gb": round(skipped_bytes / GB, 4),
              "transferred_gb": round(counting.n / GB, 4),
              "duration_s": round(time.time() - started, 1)}
    log_event(dataset=name, method="stream_archive", url=url, bytes=counting.n,
              sha256=None, duration_s=result["duration_s"], status="ok",
              subdir=subdir, keep=keep, **{k: v for k, v in result.items()
                                           if k != "duration_s"})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--keep", action="append", default=[])
    parser.add_argument("--keep-small-mb", type=float, default=0.0)
    parser.add_argument("--store-cap-gb", type=float, required=True)
    parser.add_argument("--transfer-cap-gb", type=float, required=True)
    args = parser.parse_args(argv)
    raw_dir(args.dataset)  # KeyError for a name not in the register
    try:
        result = stream(args.dataset, args.url, args.subdir, args.keep,
                        int(args.keep_small_mb * 1024**2), int(args.store_cap_gb * GB),
                        int(args.transfer_cap_gb * GB))
    except FetchError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
