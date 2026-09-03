#!/usr/bin/env python
"""Integrity manifest and register audit for the Pe_Fake_News_Dec data layer.

Three subcommands, all offline:

    python scripts/manifest.py audit            # register self-check, no data needed
    python scripts/manifest.py build [--dataset NAME ...]
    python scripts/manifest.py verify [--dataset NAME ...]

``audit`` validates data/sources.yaml itself: required fields, the
include/forbidden leak guard, the disk budget, and — if raw data is present —
whether any file forbidden by the register has actually landed on disk.

``build`` hashes everything under data/raw/<name> for each registered dataset
that exists, and writes data/MANIFEST.sha256. ``verify`` recomputes and reports
missing, changed, and unrecorded files, exiting non-zero on any mismatch.

Paths in the manifest are POSIX-style and relative to PROJECT_ROOT, so the file
is stable across machines and across the PE_FAKE_NEWS_ROOT indirection.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import (  # noqa: E402
    MANIFEST,
    PROJECT_ROOT,
    RAW,
    SOURCES_YAML,
    dataset,
    dataset_names,
    free_gb,
    raw_dir,
)

#: Hard ceiling on the sum of est_gb over enabled datasets. The register also
#: declares budget_gb; audit() insists the two agree.
BUDGET_GB: float = 120.0

#: Fields every register entry must declare, explicitly, even as null.
#: forbidden_patterns is required even when empty — an absent key means
#: "nobody thought about leakage here", which is exactly the state this
#: project refuses to be in. `expected` and `repo_type` follow the same rule:
#: an explicit null says "unknown", an omission says nothing at all.
#:
#: include_patterns is deliberately NOT here — it is optional, and its absence
#: means "take everything". See entry_leak_problems().
REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "method",
    "url",
    "repo_type",
    "licence",
    "redistributable",
    "expected",
    "est_gb",
    "enabled",
    "forbidden_patterns",
)

#: Recognised acquisition methods. `zenodo` and `gdrive` are separate from
#: `direct` because both need an API call to enumerate a record or folder
#: before anything can be selected, guarded or downloaded.
METHODS: frozenset[str] = frozenset(
    {"hf", "git", "direct", "kaggle", "zenodo", "gdrive"}
)

_CHUNK = 1024 * 1024


# --------------------------------------------------------------------------
# pattern algebra
# --------------------------------------------------------------------------


def normalise(pattern: str) -> str:
    """Canonical form of a register glob: POSIX separators, no './' prefix."""
    cleaned = pattern.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def patterns_overlap(a: str, b: str) -> bool:
    """True if globs ``a`` and ``b`` can ever select the same path.

    Deliberately symmetric and deliberately over-eager: a false positive costs
    someone a rewritten glob, a false negative costs the project its test-set
    integrity.

    Three ways two patterns can collide:

    1. they are literally equal;
    2. one matches the other as a glob (``fnmatch``'s ``*`` spans ``/``, so
       ``data_store/knowledge_store/*`` matches
       ``data_store/knowledge_store/test/*``);
    3. the literal directory prefix of one contains the other — the case with
       no wildcard to catch, e.g. include ``data_store/knowledge_store/``
       against forbidden ``data_store/knowledge_store/test/*``.
    """
    a, b = normalise(a), normalise(b)
    if a == b:
        return True
    if fnmatch(a, b) or fnmatch(b, a):
        return True

    a_dir = a.rstrip("*").rstrip("/")
    b_dir = b.rstrip("*").rstrip("/")
    if a_dir and b_dir:
        if a_dir == b_dir:
            return True
        if (a_dir + "/").startswith(b_dir + "/"):
            return True
        if (b_dir + "/").startswith(a_dir + "/"):
            return True
    return False


def matches_any(relpath: str, patterns: Iterable[str]) -> bool:
    """True if ``relpath`` matches any glob in ``patterns``."""
    target = normalise(relpath)
    return any(fnmatch(target, normalise(p)) for p in patterns)


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def _entries() -> list[dict[str, Any]]:
    """Every register entry, in file order. Triggers the lazy register load."""
    return [dataset(name) for name in dataset_names()]


def entry_leak_problems(entry: dict[str, Any]) -> list[str]:
    """Leak-guard problems for a single register entry.

    Split out of audit() so it can be unit-tested against synthetic entries —
    a guard that only ever runs on the real register is a guard nobody can
    prove still works.

    An entry with no ``include_patterns`` takes everything, so it carries an
    implicit ``"*"``. That implicit include reaches anything
    ``forbidden_patterns`` is trying to hold back, which makes "no includes
    plus some forbidden globs" a contradiction rather than a safe default.
    """
    name = entry.get("name", "<unnamed>")
    forbidden = entry.get("forbidden_patterns")
    if forbidden is None:
        return []  # reported separately as a missing required field

    includes = entry.get("include_patterns") or []
    implicit = not includes
    effective = includes or (["*"] if forbidden else [])

    problems: list[str] = []
    for inc in effective:
        for bad in forbidden:
            if patterns_overlap(inc, bad):
                detail = (
                    " (implicit '*': the entry declares no include_patterns, "
                    "so it would take everything, forbidden globs included)"
                    if implicit
                    else ""
                )
                problems.append(
                    f"{name}: include_patterns entry {inc!r} overlaps "
                    f"forbidden_patterns entry {bad!r}{detail}"
                )
    return problems


def audit() -> list[str]:
    """Validate the register. Returns a list of problems (empty means clean)."""
    problems: list[str] = []
    entries = _entries()

    for entry in entries:
        name = entry.get("name", "<unnamed>")
        for field in REQUIRED_FIELDS:
            if field not in entry:
                problems.append(f"{name}: missing required field '{field}'")

        method = entry.get("method")
        if method is not None and method not in METHODS:
            problems.append(
                f"{name}: unknown method {method!r} "
                f"(expected one of {sorted(METHODS)})"
            )

        problems.extend(entry_leak_problems(entry))

    enabled = [e for e in entries if e.get("enabled")]
    total = sum(float(e.get("est_gb", 0)) for e in enabled)
    declared = _budget_gb()
    if declared != BUDGET_GB:
        problems.append(
            f"register declares budget_gb={declared} but manifest.py enforces "
            f"{BUDGET_GB}; reconcile them deliberately"
        )
    if total > BUDGET_GB:
        problems.append(
            f"enabled datasets estimate {total:.2f} GB, over the "
            f"{BUDGET_GB:.0f} GB budget"
        )

    # Filesystem-level leak check: has anything forbidden actually landed?
    for entry in entries:
        forbidden = entry.get("forbidden_patterns") or []
        if not forbidden:
            continue
        root = RAW / entry["name"]
        if not root.is_dir():
            continue
        for path in _walk(root):
            rel = path.relative_to(root).as_posix()
            if matches_any(rel, forbidden):
                problems.append(
                    f"{entry['name']}: forbidden file present on disk: {rel}"
                )

    return problems


def _budget_gb() -> float:
    import yaml

    with SOURCES_YAML.open(encoding="utf-8") as handle:
        return float(yaml.safe_load(handle).get("budget_gb", BUDGET_GB))


def summarise() -> str:
    entries = _entries()
    enabled = [e for e in entries if e.get("enabled")]
    total = sum(float(e.get("est_gb", 0)) for e in enabled)
    lines = [
        f"register: {SOURCES_YAML}",
        f"datasets: {len(entries)} ({len(enabled)} enabled)",
        f"budget:   {total:.2f} / {BUDGET_GB:.0f} GB estimated",
        f"free:     {free_gb():.1f} GB on the volume holding {PROJECT_ROOT}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


#: Directories excluded from the manifest and from tree hashing -- client
#: bookkeeping, not data. `.git` comes from sparse clones (FETCH_HEAD and
#: index change on every fetch without the data changing); `.cache` is what
#: huggingface_hub writes beside a local_dir snapshot; `__pycache__` is
#: generated bytecode -- pytest imported a vendored test suite out of
#: data/raw/mocheg/_aux/ and left a .pyc there, which then read as a change
#: to immutable raw data. Hashing any of them would make the integrity
#: record permanently unstable.
EXCLUDED_DIRS: frozenset[str] = frozenset({".git", ".cache", "__pycache__"})


def data_files(root: Path) -> Iterator[Path]:
    """Every data file under ``root``, sorted, excluding EXCLUDED_DIRS.

    The single definition of "what counts as data" — shared with fetchlib so
    the integrity manifest and the .done completion markers can never disagree
    about which files a dataset consists of.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and not EXCLUDED_DIRS.intersection(path.parts):
            yield path


#: Back-compat alias for the internal call sites.
_walk = data_files


def sha256(path: Path, retries: int = 2, pause: float = 0.5) -> str:
    """SHA-256 of a file, retrying briefly on a transient OS error.

    On Windows an antivirus scanner or a just-closed write handle can make a
    file momentarily unopenable (errno 22, "Invalid argument") even though it
    is perfectly readable a second later. Retrying twice costs nothing and
    avoids failing a 142,000-file scan over a file that is actually fine.
    """
    last: OSError | None = None
    for attempt in range(retries + 1):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as exc:
            last = exc
            if attempt < retries:
                time.sleep(pause)
    raise last  # type: ignore[misc]


def _targets(selected: list[str] | None) -> list[str]:
    known = dataset_names()
    if not selected:
        return list(known)
    for name in selected:
        raw_dir(name)  # KeyError with a helpful message if the name is bogus
    return selected


def scan(selected: list[str] | None = None,
         errors: list[tuple[str, str]] | None = None) -> dict[str, str]:
    """Hash every present raw file. Maps PROJECT_ROOT-relative path -> sha256.

    If ``errors`` is given, a file that cannot be read is appended to it as
    (relpath, message) and the scan continues; otherwise the OSError
    propagates.

    That option exists because of a real incident: one transient errno 22 on a
    single JPEG aborted `manifest.update(["fakeddit"])` and left all 142,434
    of that dataset's files unrecorded, with nothing but one line in a log to
    show for it. Abandoning an entire manifest over one unreadable byte is
    worse than recording the other 142,433 and saying loudly which one failed.
    """
    found: dict[str, str] = {}
    for name in _targets(selected):
        root = raw_dir(name)
        if not root.is_dir():
            continue
        for path in _walk(root):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            try:
                found[rel] = sha256(path)
            except OSError as exc:
                if errors is None:
                    raise
                errors.append((rel, f"{type(exc).__name__}: {exc}"))
    return dict(sorted(found.items()))


def read_manifest() -> dict[str, str]:
    if not MANIFEST.is_file():
        return {}
    recorded: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            raise ValueError(f"{MANIFEST}: malformed line: {line!r}")
        recorded[rel] = digest
    return recorded


def write_manifest(entries: dict[str, str]) -> None:
    lines = [
        "# Pe_Fake_News_Dec raw-data integrity manifest",
        "# sha256  <path relative to PROJECT_ROOT>",
        "# regenerate: python scripts/manifest.py build",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in entries.items()]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_prefixes(selected: Sequence[str]) -> tuple[str, ...]:
    """PROJECT_ROOT-relative path prefixes for the given datasets."""
    return tuple(
        raw_dir(n).relative_to(PROJECT_ROOT).as_posix() + "/" for n in selected
    )


def update(selected: Sequence[str]) -> dict[str, str]:
    """Rehash only `selected` and merge into the existing manifest.

    `build` with no arguments rehashes every byte under data/raw. Calling that
    after each dataset completes would rehash the whole corpus once per
    dataset -- tens of minutes of pure waste on a 66 GB tree. This replaces
    just the selected datasets' rows and leaves every other row untouched.
    """
    merged = read_manifest()
    prefixes = _dataset_prefixes(selected)
    errors: list[tuple[str, str]] = []
    fresh = scan(list(selected), errors=errors)
    for rel in [k for k in merged if k.startswith(prefixes)]:
        del merged[rel]
    merged.update(fresh)
    ordered = dict(sorted(merged.items()))
    write_manifest(ordered)
    if errors:
        print(f"  WARNING: {len(errors)} file(s) could not be hashed and are "
              "NOT in the manifest:", file=sys.stderr)
        for rel, message in errors[:5]:
            print(f"    {rel}  ({message})", file=sys.stderr)
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more", file=sys.stderr)
        print("  'manifest.py verify' will report them as UNTRACKED.",
              file=sys.stderr)
    return ordered


def verify(selected: list[str] | None = None) -> list[str]:
    """Compare disk against the manifest. Returns a list of discrepancies."""
    if not MANIFEST.is_file():
        return [f"no manifest at {MANIFEST}; run 'manifest.py build' first"]
    # An existing manifest with no entries is a valid state: nothing acquired
    # yet. Only an absent file means "you never ran build".
    recorded = read_manifest()

    if selected:
        prefixes = _dataset_prefixes(selected)
        recorded = {k: v for k, v in recorded.items() if k.startswith(prefixes)}

    found = scan(selected)
    problems: list[str] = []
    for rel, digest in recorded.items():
        if rel not in found:
            problems.append(f"MISSING  {rel}")
        elif found[rel] != digest:
            problems.append(f"CHANGED  {rel}")
    for rel in found:
        if rel not in recorded:
            problems.append(f"UNTRACKED {rel}")
    return problems


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("audit", "validate the register and check for forbidden files on disk"),
        ("build", "hash raw data and write the manifest"),
        ("verify", "check raw data against the manifest"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name != "audit":
            p.add_argument(
                "--dataset",
                action="append",
                dest="datasets",
                metavar="NAME",
                help="restrict to this dataset (repeatable)",
            )

    args = parser.parse_args(argv)
    selected = getattr(args, "datasets", None)

    if args.command == "audit":
        print(summarise())
        problems = audit()
        if problems:
            print(f"\n{len(problems)} problem(s):", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("\nregister OK")
        return 0

    if args.command == "build":
        if selected:
            entries = update(selected)
            print(
                f"updated {', '.join(selected)}; manifest now holds "
                f"{len(entries)} entries"
            )
        else:
            errors: list[tuple[str, str]] = []
            entries = scan(None, errors=errors)
            write_manifest(entries)
            print(f"wrote {len(entries)} entries to {MANIFEST}")
            if errors:
                print(f"WARNING: {len(errors)} unreadable file(s) omitted:",
                      file=sys.stderr)
                for rel, message in errors[:5]:
                    print(f"  {rel}  ({message})", file=sys.stderr)
                return 1
        return 0

    problems = verify(selected)
    if problems:
        print(f"{len(problems)} discrepancy(ies):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
