#!/usr/bin/env python
"""Refuse to publish anything that must not leave this machine.

    git add -A && python scripts/prepush_check.py

Every dataset in the register is ``redistributable: false``. AVeriTeC is
CC-BY-NC, Factify2 arrives password-protected behind a shared-task
registration, and Fakeddit's restriction is unresolved. So the rule this
enforces is simple and keyed off the register rather than anyone's memory:

    published per-record files carry IDENTIFIERS AND SPLIT ASSIGNMENT ONLY.

Aggregate statistics are fine and are the point of publishing at all — a
domain's purity and count are facts about a public web property, not somebody's
annotation, and they are what make the reported results checkable. A per-record
label is the annotation the licence exists to protect.

Three refusals, any one of which stops the push:

1. **Data bytes.** Anything under ``data/raw/``, or any image, archive or
   columnar-data extension.
2. **Per-record labels.** A staged CSV/JSONL whose columns name a label, or a
   staged JSON that is a *list of records* carrying one. A JSON that is a
   mapping is an aggregate report and passes.
3. **Gate credentials.** The Factify2 archive passwords, wherever they appear —
   including in files that merely mention them — so removing them from source
   cannot be quietly undone.

Exit code 0 means the staged set is safe to push. Anything else refuses, names
every offending path, and says why.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import PROJECT_ROOT, dataset_names, dataset  # noqa: E402
from scripts.hydrate import LABEL_COLUMNS  # noqa: E402

#: Extensions that are data rather than description.
FORBIDDEN_SUFFIXES = {
    ".zip", ".7z", ".tar", ".gz", ".tgz",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    ".parquet", ".npy", ".npz", ".pkl", ".pt", ".bin", ".arrow",
}

#: Path prefixes that hold data, never description.
FORBIDDEN_PREFIXES = ("data/raw/", "data/bible/", "data/interim/")

#: Column / key names that carry an annotation. Built from the same table
#: hydrate.py uses to find label columns, so the two cannot drift apart.
LABEL_FIELDS = {c.lower() for cols in LABEL_COLUMNS.values() for c in cols} | {
    "label", "labels", "category", "class", "truthfulness",
    "cleaned_truthfulness", "2_way_label", "3_way_label", "6_way_label",
    "verdict", "rating", "gold_label",
}

#: Columns naming a fetchable location for gated content.
URL_FIELDS = {"url", "image_url", "claim_image", "document_image",
              "source_url", "wayback_url"}

#: SHA-256 of each gate credential. Stored as digests, never as text: an
#: earlier version held the passwords as literal substrings and promptly
#: flagged ITSELF -- a checker that cannot be committed is no checker, and one
#: that publishes most of the secret in order to detect it is worse.
CREDENTIAL_SHA256 = {
    "225b39d8105ed7ed8e58fda9d032f536c952811bb797600dd94f3196b8274d2e",
    "73bb607439c8d9c0c004d3418d194b32607c21ebdc7857f3e786be911badc43e",
}

#: Candidate secret-shaped tokens. Passwords here contain '@', so the class is
#: wider than a bare word.
_TOKEN = re.compile(r"[A-Za-z0-9@._-]{8,64}")


def contains_credential(text: str) -> bool:
    """True if any token in `text` hashes to a known gate credential."""
    return any(hashlib.sha256(t.encode("utf-8")).hexdigest() in CREDENTIAL_SHA256
               for t in _TOKEN.findall(text))

TEXT_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".py",
                 ".yaml", ".yml", ".ini", ".cfg", ".toml"}


class Refusal(Exception):
    """The staged set may not be published."""


def staged_files(repo: Path) -> list[str]:
    """Paths staged for commit, POSIX-style and repo-relative."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise Refusal(f"git failed: {out.stderr.strip()}")
    return [line.strip().replace("\\", "/") for line in out.stdout.splitlines()
            if line.strip()]


def gated_datasets() -> set[str]:
    """Every register entry that may not be redistributed."""
    return {n for n in dataset_names()
            if dataset(n).get("redistributable") is not True}


def _dataset_of(path: str, gated: set[str]) -> str | None:
    stem = Path(path).name.lower()
    for name in gated:
        if name in stem or f"/{name}/" in path.lower():
            return name
    return None


def _header_of(path: Path) -> list[str]:
    """Column names of a CSV/TSV, or [] if it cannot be read as one."""
    try:
        with path.open(encoding="utf-8", newline="", errors="replace") as fh:
            line = fh.readline()
    except OSError:
        return []
    delimiter = "\t" if line.count("\t") > line.count(",") else ","
    try:
        return next(csv.reader(io.StringIO(line), delimiter=delimiter), [])
    except csv.Error:
        return []


#: Zero-byte placeholders that hold a directory's shape in git. They are
#: tracked deliberately and carry nothing, so refusing them would be a false
#: positive -- and a guard that cries wolf is a guard people switch off.
PLACEHOLDERS = {".gitkeep"}


def check_path(path: str, repo: Path | None = None) -> list[str]:
    """Refusals arising from where a file is and what type it is."""
    problems = []
    lower = path.lower()
    if Path(path).name in PLACEHOLDERS:
        full = (repo / path) if repo is not None else None
        # The exemption is for an EMPTY placeholder, not for the name: a file
        # called .gitkeep with content in it is still content.
        if full is None or not full.is_file() or full.stat().st_size == 0:
            return []
    if lower.startswith(FORBIDDEN_PREFIXES):
        problems.append(f"lives under a data directory ({path.split('/')[1]}/)")
    suffix = Path(lower).suffix
    if suffix in FORBIDDEN_SUFFIXES:
        problems.append(f"is a data/media file ({suffix})")
    return problems


def check_contents(repo: Path, path: str, gated: set[str]) -> list[str]:
    """Refusals arising from what is inside a file."""
    full = repo / path
    if not full.is_file():
        return []
    problems: list[str] = []
    suffix = full.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        try:
            blob = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            blob = ""
        if contains_credential(blob):
            problems.append(
                "contains a gate credential (a Factify2 archive password). "
                "Those are handed out on accepting the shared-task terms; "
                "publishing them defeats the registration gate")

    owner = _dataset_of(path, gated)
    if owner is None:
        return problems

    if suffix in {".csv", ".tsv"}:
        header = [c.strip().lower() for c in _header_of(full)]
        bad = sorted(set(header) & LABEL_FIELDS)
        if bad:
            problems.append(
                f"is a per-record file for the non-redistributable dataset "
                f"{owner!r} and carries label column(s) {bad}. Publish "
                "record_id and split assignment only")
        urls = sorted(set(header) & URL_FIELDS)
        if urls:
            problems.append(
                f"carries per-record URL column(s) {urls} for {owner!r}")

    elif suffix in {".jsonl", ".ndjson"}:
        try:
            first = next((l for l in full.read_text(encoding="utf-8",
                                                    errors="replace").splitlines()
                          if l.strip()), "")
            record = json.loads(first) if first else {}
        except (OSError, json.JSONDecodeError):
            record = {}
        if isinstance(record, dict):
            keys = {k.lower() for k in record}
            bad = sorted(keys & LABEL_FIELDS)
            if bad:
                problems.append(
                    f"is a per-record JSONL for {owner!r} carrying label key(s) "
                    f"{bad}")
            urls = sorted(keys & URL_FIELDS)
            if urls:
                problems.append(
                    f"is a per-record JSONL for {owner!r} carrying URL key(s) "
                    f"{urls}")

    elif suffix == ".json":
        try:
            payload = json.loads(full.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return problems
        # A mapping is an aggregate report and is exactly what we want to
        # publish. A list of records is per-record data wearing a .json suffix.
        if isinstance(payload, list):
            labelled = [r for r in payload[:200]
                        if isinstance(r, dict) and {k.lower() for k in r} & LABEL_FIELDS]
            if labelled:
                problems.append(
                    f"is a JSON LIST of per-record objects for {owner!r} carrying "
                    "label keys — aggregate reports are mappings, not lists")
    return problems


def check(repo: Path, paths: Iterable[str]) -> dict[str, list[str]]:
    gated = gated_datasets()
    found: dict[str, list[str]] = {}
    for path in paths:
        problems = check_path(path, repo) + check_contents(repo, path, gated)
        if problems:
            found[path] = problems
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=str(PROJECT_ROOT))
    parser.add_argument("--path", action="append", dest="paths", metavar="PATH",
                        help="check these paths instead of the git index")
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    try:
        paths = args.paths or staged_files(repo)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"pre-push check: {len(paths):,} staged path(s)")
    gated = sorted(gated_datasets())
    print(f"  non-redistributable datasets: {len(gated)} "
          f"({', '.join(gated[:4])}{', ...' if len(gated) > 4 else ''})")

    problems = check(repo, paths)
    if not problems:
        print("\nOK — nothing staged carries data bytes, per-record labels, "
              "or a gate credential.")
        return 0

    print(f"\nREFUSING TO PUSH — {len(problems)} offending path(s):",
          file=sys.stderr)
    for path, reasons in sorted(problems.items()):
        print(f"\n  {path}", file=sys.stderr)
        for reason in reasons:
            print(f"      - {reason}", file=sys.stderr)
    print("\nUnstage these, or reduce them to identifiers and split assignment, "
          "before pushing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
