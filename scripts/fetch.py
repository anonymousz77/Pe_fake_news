#!/usr/bin/env python
"""Network acquisition for Pe_Fake_News_Dec.

    python scripts/fetch.py --dry-run --all
    python scripts/fetch.py --dataset liar --yes
    python scripts/fetch.py --dataset averitec --accept-licence --yes
    python scripts/fetch.py --all --sample-size 50000 --yes

Every fetch passes three gates before a byte moves: the volume must have
1.5x the dataset's estimate free, the projected committed total (raw on disk +
everything still promised + 30 GB for the evidence store + 20 GB for working
directories) must stay under 140 GB, and an enumeration of what the fetch
*would* retrieve must contain nothing the register forbids.

``--dry-run`` is entirely offline: estimates come from the register, and the
process exits before opening a socket.

Raw data is immutable. Downloads land in ``data/raw/<name>/`` and nothing here
rewrites them afterwards; derived artefacts (the Fakeddit sample index,
completion markers) go to ``data/interim/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT, dataset, raw_dir  # noqa: E402
from scripts import manifest  # noqa: E402
from scripts.fetchlib import (  # noqa: E402
    GB,
    CredentialError,
    DiskBudgetError,
    FetchError,
    ForbiddenPathError,
    LicenceNotAcceptedError,
    allow_patterns_for,
    check_free_space,
    check_projection,
    check_selection_size,
    completion_status,
    download_resumable,
    enabled_names,
    est_gb,
    guard_or_abort,
    log_event,
    make_session,
    project_disk,
    reconcile_estimate,
    select_included,
    stratified_indices,
    write_marker,
)

#: Fakeddit: images are sampled, not taken wholesale. The seed is recorded in
#: the .done marker and the fetch log so the sample is reproducible.
FAKEDDIT_SAMPLE_SIZE = 150_000
FAKEDDIT_SEED = 20260901
FAKEDDIT_LABEL_COLUMN = "6_way_label"

#: Factify2's archives are password-protected, and those passwords are the
#: registration gate: they are handed out on accepting the shared-task terms.
#: They are therefore NEVER stored in this file -- committing them to a public
#: repository would defeat the gate as surely as republishing the data. Set:
#:
#:     FACTIFY2_ZIP_PASSWORD       (train + val, factify2.zip)
#:     FACTIFY2_TEST_ZIP_PASSWORD  (factify2test.zip)
#:
#: Keyed on the archive filenames actually in the Drive folder (verified by
#: listing it), not on split names -- train and val share one factify2.zip.
FACTIFY_PASSWORD_ENV = {
    "factify2.zip": "FACTIFY2_ZIP_PASSWORD",
    "factify2test.zip": "FACTIFY2_TEST_ZIP_PASSWORD",
}


def factify_passwords() -> dict[str, str]:
    """Archive passwords from the environment. Raises if any are missing."""
    found, missing = {}, []
    for archive, var in FACTIFY_PASSWORD_ENV.items():
        value = os.environ.get(var)
        if value:
            found[archive] = value
        else:
            missing.append(f"{var} (for {archive})")
    if missing:
        raise CredentialError(
            "factify2: archive password(s) not set: " + ", ".join(missing) + ".\n"
            "  These are handed out on accepting the shared-task terms at\n"
            "  https://aiisc.ai/defactify2/ — obtain them yourself; they are\n"
            "  deliberately not stored in this repository, because publishing\n"
            "  them would defeat the registration gate the dataset relies on.\n"
            "  Then: set FACTIFY2_ZIP_PASSWORD and FACTIFY2_TEST_ZIP_PASSWORD."
        )
    return found

#: Datasets that may not be fetched without --accept-licence.
LICENCE_GATED: dict[str, str] = {
    "averitec": (
        "AVeriTeC is distributed under CC-BY-NC-4.0 (Attribution-NonCommercial).\n"
        "  You may share and adapt it for NON-COMMERCIAL purposes, with\n"
        "  attribution. Commercial use is not permitted under this licence.\n"
        "  Source: https://huggingface.co/chenxwh/AVeriTeC"
    ),
}

ARCHIVE_SUFFIXES = (".zip", ".7z", ".tar", ".tar.gz", ".tgz")


@dataclass
class Context:
    """Everything the handlers need from the command line."""

    yes: bool = False
    accept_licence: bool = False
    sample_size: int = FAKEDDIT_SAMPLE_SIZE
    seed: int = FAKEDDIT_SEED


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def require_licence(entry: dict[str, Any], ctx: Context) -> None:
    name = entry["name"]
    text = LICENCE_GATED.get(name)
    if text is None:
        return
    print(f"\n  LICENCE — {name} ({entry.get('licence')})")
    print("  " + "-" * 66)
    for line in text.splitlines():
        print(f"  {line}")
    print("  " + "-" * 66)
    if not ctx.accept_licence:
        raise LicenceNotAcceptedError(
            f"{name} is licence-gated. Re-run with --accept-licence to confirm "
            "you accept the terms printed above."
        )
    print("  accepted via --accept-licence\n")
    log_event(
        event="licence_accepted",
        dataset=name,
        licence=entry.get("licence"),
        method=entry.get("method"),
    )


def kaggle_token_source() -> str:
    """Where Kaggle credentials will come from, or raise naming every option.

    kaggle 2.2.4 no longer reads ~/.kaggle/kaggle.json first; its own resolution
    order is KAGGLE_API_TOKEN, then ~/.kaggle/access_token(.txt), then OAuth.
    Verified against the installed kagglesdk, not from memory.
    """
    home = Path.home()
    candidates = [
        ("KAGGLE_API_TOKEN env var", bool(os.environ.get("KAGGLE_API_TOKEN"))),
        ("~/.kaggle/access_token", (home / ".kaggle" / "access_token").is_file()),
        ("~/.kaggle/access_token.txt", (home / ".kaggle" / "access_token.txt").is_file()),
        ("KAGGLE_USERNAME + KAGGLE_KEY env vars",
         bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))),
        ("~/.kaggle/kaggle.json (legacy)", (home / ".kaggle" / "kaggle.json").is_file()),
    ]
    for label, present in candidates:
        if present:
            return label
    raise CredentialError(
        "Kaggle credentials not found. kaggle 2.2.4 accepts any of:\n"
        "    1. run `kaggle auth login` (OAuth, nothing to manage)\n"
        "    2. export KAGGLE_API_TOKEN=<token from kaggle.com/settings/api>\n"
        "    3. save that token to ~/.kaggle/access_token\n"
        "    4. legacy ~/.kaggle/kaggle.json with username + key\n"
        "  Note: kaggle.json is no longer the primary location in this version."
    )


# --------------------------------------------------------------------------
# method handlers
# --------------------------------------------------------------------------


def fetch_hf(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """HuggingFace snapshot, filtered and guarded before anything downloads."""
    from huggingface_hub import snapshot_download

    repo_id = entry["url"]
    repo_type = entry.get("repo_type") or "model"
    allow = allow_patterns_for(entry)
    forbidden = entry.get("forbidden_patterns") or []

    # Enumerate with allow_patterns ONLY. Applying ignore_patterns here too
    # would silently strip forbidden files from the plan, so a widened include
    # that reaches held-out evidence would look clean. Guard what the include
    # selects, then download with both filters as defence in depth.
    plan = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow or None,
        dry_run=True,
    )
    names = [f.filename for f in plan]
    guard_or_abort(entry, names)

    selected_bytes = sum(f.file_size or 0 for f in plan)
    selected_gb, warning = check_selection_size(entry["name"], selected_bytes)
    print(f"    {len(names)} file(s), {selected_gb:.2f} GB selected from {repo_id}")
    if warning:
        print(f"    NOTE: {warning}")

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow or None,
        ignore_patterns=forbidden or None,
        local_dir=dest,
        max_workers=8,
    )
    return {"files": len(names), "selected_gb": round(selected_gb, 3)}


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise FetchError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def fetch_git(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """Shallow, blobless sparse clone.

    ``--filter=blob:none`` means the fetch brings trees but no file contents,
    so the guard can enumerate the whole repository before a single byte of
    data is materialised.
    """
    url = entry["url"]
    if not url.startswith(("http://", "https://", "git@")):
        url = f"https://{url}"
    patterns = allow_patterns_for(entry)

    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _run_git(["init", "--quiet"], dest)
        _run_git(["remote", "add", "origin", url], dest)

    _run_git(["fetch", "--depth", "1", "--filter=blob:none", "origin", "HEAD"], dest)
    listing = _run_git(["ls-tree", "-r", "--name-only", "FETCH_HEAD"], dest)
    all_paths = [line.strip() for line in listing.splitlines() if line.strip()]

    # Guard the selection, not the whole listing: a repository may legitimately
    # contain a held-out split that include_patterns already excludes. What
    # matters is whether this fetch would RETRIEVE one.
    wanted = select_included(entry, all_paths)
    guard_or_abort(entry, wanted)
    if not wanted:
        raise FetchError(
            f"{entry['name']}: include_patterns matched none of the "
            f"{len(all_paths)} files in {url}. Check the patterns against the "
            "repository layout."
        )

    _run_git(["sparse-checkout", "init", "--no-cone"], dest)
    _run_git(["sparse-checkout", "set", "--no-cone", *(patterns or ["/*"])], dest)
    _run_git(["checkout", "--quiet", "FETCH_HEAD"], dest)

    landed = [
        p.relative_to(dest).as_posix()
        for p in dest.rglob("*")
        if p.is_file() and ".git" not in p.parts
    ]
    guard_or_abort(entry, landed)  # belt and braces: what actually landed
    return {"files": len(landed), "repo_files": len(all_paths)}


def _zenodo_files(record_url: str, session) -> list[tuple[str, str]]:
    """(filename, download_url) for a Zenodo record."""
    record_id = record_url.rstrip("/").split("/")[-1]
    api = f"https://zenodo.org/api/records/{record_id}"
    response = session.get(api, timeout=30)
    response.raise_for_status()
    payload = response.json()
    files = []
    for item in payload.get("files", []):
        name = item.get("key") or item.get("filename")
        link = (item.get("links") or {}).get("self") or item.get("links", {}).get("download")
        if name and link:
            files.append((name, link))
    return files


def _extract_archive(
    entry: dict[str, Any], archive: Path, dest: Path, password: str | None = None
) -> list[str]:
    """Enumerate members, guard them, then extract only the included ones."""
    suffix = "".join(archive.suffixes[-2:]).lower()
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            wanted = select_included(entry, names) or names
            guard_or_abort(entry, wanted)
            try:
                zf.extractall(dest, members=wanted,
                              pwd=password.encode() if password else None)
            except RuntimeError as exc:
                raise FetchError(
                    f"{archive.name}: could not extract ({exc}). If this zip uses "
                    "AES encryption, Python's zipfile only supports legacy "
                    "ZipCrypto and a pyzipper dependency would be needed."
                ) from exc
    elif archive.suffix.lower() == ".7z":
        import py7zr

        with py7zr.SevenZipFile(archive, mode="r", password=password) as sz:
            names = sz.getnames()
            wanted = select_included(entry, names) or names
            guard_or_abort(entry, wanted)
            sz.extractall(path=dest)
        with py7zr.SevenZipFile(archive, mode="r", password=password) as sz:
            if sz.testzip() is not None:
                raise FetchError(f"{archive.name}: archive failed integrity check")
    elif suffix in (".tar.gz", ".tgz") or archive.suffix.lower() == ".tar":
        with tarfile.open(archive) as tf:
            names = [m.name for m in tf.getmembers() if m.isfile()]
            guard_or_abort(entry, names)
            tf.extractall(dest, filter="data")
            wanted = names
    else:
        return []
    return wanted


def fetch_direct(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """Ranged, resumable HTTP download, extracting archives after guarding."""
    session = make_session()
    url = entry["url"]
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    if "zenodo.org/record" in url:
        targets = _zenodo_files(url, session)
        if not targets:
            raise FetchError(f"{entry['name']}: Zenodo record listed no files ({url})")
    else:
        targets = [(url.rstrip("/").split("/")[-1], url)]

    guard_or_abort(entry, [name for name, _ in targets])

    dest.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    extracted: list[str] = []
    for name, link in targets:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"    have {name}")
        else:
            print(f"    get  {name}")
            size, _ = download_resumable(session, link, target)
            total_bytes += size
        if name.lower().endswith(ARCHIVE_SUFFIXES):
            extracted += _extract_archive(entry, target, dest)

    return {"files": len(targets), "extracted": len(extracted), "bytes": total_bytes}


def _zenodo_record(url: str, session) -> dict[str, Any]:
    record_id = url.rstrip("/").split("/")[-1]
    response = session.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_zenodo(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """Zenodo record: enumerate via the API, guard, then download the selection."""
    session = make_session()
    url = entry["url"]
    if not url.startswith("http"):
        url = f"https://{url}"

    record = _zenodo_record(url, session)
    files = [
        (f.get("key"), (f.get("links") or {}).get("self"), f.get("size") or 0)
        for f in record.get("files", [])
    ]
    files = [(k, link, size) for k, link, size in files if k and link]
    if not files:
        raise FetchError(f"{entry['name']}: Zenodo record {url} lists no files")

    wanted = select_included(entry, [k for k, _, _ in files])
    guard_or_abort(entry, wanted)
    selected = [(k, link, size) for k, link, size in files if k in wanted]

    selected_gb, warning = check_selection_size(
        entry["name"], sum(size for _, _, size in selected)
    )
    licence = (record.get("metadata") or {}).get("license") or {}
    print(f"    {len(selected)} file(s), {selected_gb:.3f} GB from Zenodo "
          f"(record licence: {licence.get('id', 'unstated')})")
    if warning:
        print(f"    NOTE: {warning}")

    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    for key, link, _size in selected:
        target = dest / key
        if target.exists() and target.stat().st_size > 0:
            print(f"    have {key}")
        else:
            print(f"    get  {key}")
            download_resumable(session, link, target)
        if key.lower().endswith(ARCHIVE_SUFFIXES):
            extracted += _extract_archive(entry, target, dest)

    return {
        "files": len(selected),
        "extracted": len(extracted),
        "zenodo_licence": licence.get("id"),
        "record_title": (record.get("metadata") or {}).get("title"),
    }


def fetch_gdrive(entry: dict[str, Any], dest: Path, ctx: Context,
                 passwords: dict[str, str] | None = None) -> dict[str, Any]:
    """Google Drive folder: enumerate, guard the selection, fetch file by file.

    ``download_folder`` would pull the entire folder, so selected files are
    downloaded individually by id. That is what makes include_patterns real
    here rather than advisory -- Fakeddit's folder contains a held-out test TSV
    that must never land on disk.
    """
    import gdown

    url = entry["url"]
    if not url.startswith("http"):
        url = f"https://{url}"

    listing = gdown.download_folder(url=url, skip_download=True, quiet=True,
                                    use_cookies=False) or []
    available = [f.path.replace("\\", "/") for f in listing]
    by_path = {f.path.replace("\\", "/"): f for f in listing}

    wanted = select_included(entry, available)
    guard_or_abort(entry, wanted)
    if not wanted:
        raise FetchError(
            f"{entry['name']}: include_patterns matched none of the "
            f"{len(available)} files in the Drive folder:\n    "
            + "\n    ".join(available[:20])
        )

    skipped = [p for p in available if p not in wanted]
    print(f"    {len(wanted)} of {len(available)} file(s) selected"
          + (f"; not fetching {len(skipped)}" if skipped else ""))

    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    for rel in wanted:
        target = dest / Path(rel).name
        if target.exists() and target.stat().st_size > 0:
            print(f"    have {target.name}")
        else:
            print(f"    get  {target.name}")
            gdown.download(id=by_path[rel].id, output=str(target),
                           quiet=False, resume=True)
        if target.name.lower().endswith(ARCHIVE_SUFFIXES):
            password = (passwords or {}).get(target.name)
            members = _extract_archive(entry, target, dest, password=password)
            if not members:
                raise FetchError(
                    f"{target.name}: extracted no members - not deleting it"
                )
            extracted += members
            print(f"      {len(members)} member(s) OK")

    return {"files": len(wanted), "available": len(available),
            "not_fetched": skipped, "extracted": len(extracted)}


def fetch_kaggle(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """Kaggle dataset download. Credentials checked before the noisy import."""
    source = kaggle_token_source()
    print(f"    credentials: {source}")

    import kaggle  # noqa: F401  (prints an auth banner at import; keep it lazy)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    slug = entry["url"]
    listed = api.dataset_list_files(slug)
    names = [f.name for f in getattr(listed, "files", [])]
    if names:
        guard_or_abort(entry, names)

    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

    landed = [p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()]
    guard_or_abort(entry, landed)
    return {"files": len(landed), "listed": len(names), "credentials": source}


# --------------------------------------------------------------------------
# dataset-specific handlers
# --------------------------------------------------------------------------


def fetch_factify2(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """Drive folder, then extract each archive with its own password.

    Archives are kept until their members extract, so a failed extraction never
    destroys the thing you would need to retry.
    """
    stats = fetch_gdrive(entry, dest, ctx, passwords=factify_passwords())
    print("    images are NOT in these archives - run hydrate.py for them")
    return {**stats, "note": "images come from URLs via hydrate.py"}


def fetch_fakeddit(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any]:
    """v2.0 TSVs from Drive, then a seeded stratified image sample.

    The GitHub repo holds only image_downloader.py; its README points the TSVs
    at a Drive folder, which also contains the held-out test TSVs that
    forbidden_patterns keeps out.
    """
    stats = fetch_gdrive(entry, dest, ctx)

    tsvs = sorted(dest.rglob("*.tsv"))
    if not tsvs:
        print("    WARNING: no .tsv files landed; skipping sampling")
        return {**stats, "sampled": 0, "note": "no tsv files found to sample"}

    rows: list[dict[str, str]] = []
    for tsv in tsvs:
        if "test" in tsv.name.lower():
            continue  # held out; the guard already refuses it
        with tsv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("image_url"):
                    rows.append(
                        {
                            "id": row.get("id", ""),
                            "label": row.get(FAKEDDIT_LABEL_COLUMN, ""),
                            "image_url": row["image_url"],
                            "source_tsv": tsv.name,
                        }
                    )

    if not rows:
        raise FetchError(
            f"fakeddit: no rows with an image_url and a {FAKEDDIT_LABEL_COLUMN} "
            "column were found. Sampling cannot be stratified without the label, "
            "and an unstratified sample would silently break class balance."
        )

    indices = stratified_indices([r["label"] for r in rows], ctx.sample_size, ctx.seed)
    sample = [rows[i] for i in indices]

    out_dir = INTERIM / "fakeddit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sample_seed{ctx.seed}_n{len(sample)}.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "image_url", "source_tsv"])
        writer.writeheader()
        writer.writerows(sample)

    counts: dict[str, int] = {}
    for row in sample:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"    sampled {len(sample)} of {len(rows)} rows -> {out.name}")
    print(f"    per class: {dict(sorted(counts.items()))}")

    return {
        **stats,
        "candidate_rows": len(rows),
        "sampled": len(sample),
        "sample_seed": ctx.seed,
        "sample_size": ctx.sample_size,
        "sample_index": str(out.relative_to(PROJECT_ROOT)),
        "per_class": dict(sorted(counts.items())),
    }


def fetch_aux(entry: dict[str, Any], dest: Path, ctx: Context) -> dict[str, Any] | None:
    """Fetch an entry's secondary source into ``<raw>/_aux/``.

    Some corpora split data and the code needed to use it across two hosts --
    mocheg's data is on Zenodo while the tweet-id hydration script lives on
    GitHub. The aux source is fetched with its own method but shares the
    entry's forbidden_patterns, and never its include_patterns: those describe
    the primary source's layout, not the auxiliary repository's.
    """
    aux_url = entry.get("aux_url")
    if not aux_url:
        return None
    aux_method = entry.get("aux_method") or "git"
    aux_entry = {
        **entry,
        "name": f"{entry['name']}:aux",
        "method": aux_method,
        "url": aux_url,
        "include_patterns": None,
    }
    aux_dest = dest / "_aux"
    print(f"    aux: {aux_method} {aux_url}")
    handler = METHOD_HANDLERS.get(aux_method)
    if handler is None:
        raise FetchError(f"{entry['name']}: unknown aux_method {aux_method!r}")
    stats = handler(aux_entry, aux_dest, ctx)
    landed = [p.relative_to(aux_dest).as_posix()
              for p in aux_dest.rglob("*") if p.is_file()]
    guard_or_abort(entry, landed)
    return {"method": aux_method, "url": aux_url, **stats}


METHOD_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "hf": fetch_hf,
    "git": fetch_git,
    "direct": fetch_direct,
    "kaggle": fetch_kaggle,
    "zenodo": fetch_zenodo,
    "gdrive": fetch_gdrive,
}

DATASET_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "factify2": fetch_factify2,
    "fakeddit": fetch_fakeddit,
}


def handler_for(entry: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    if entry["name"] in DATASET_HANDLERS:
        return DATASET_HANDLERS[entry["name"]]
    try:
        return METHOD_HANDLERS[entry["method"]]
    except KeyError:
        raise FetchError(f"{entry['name']}: no handler for method {entry['method']!r}")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def fetch_one(name: str, ctx: Context, pending: list[str]) -> str:
    """Fetch one dataset. Returns a short status word."""
    entry = dataset(name)
    dest = raw_dir(name)

    print(f"\n[{name}] method={entry['method']} est={est_gb(name):.2f} GB")

    skip, reason = completion_status(name, dest)
    if skip:
        print(f"  SKIP — {reason}")
        return "skipped"
    if reason != "no .done marker":
        print(f"  {reason}")

    check_projection(pending)
    available = check_free_space(name)
    print(f"  disk OK — {available:.1f} GB free, "
          f"need {est_gb(name) * 1.5:.2f} GB")

    require_licence(entry, ctx)

    started = time.time()
    try:
        stats = handler_for(entry)(entry, dest, ctx)
        aux = fetch_aux(entry, dest, ctx)
        if aux:
            stats = {**stats, "aux": aux}
    except BaseException as exc:  # third-party clients raise their own types
        if not isinstance(exc, KeyboardInterrupt):
            log_event(dataset=name, method=entry["method"], url=entry["url"],
                      status="aborted", error_type=type(exc).__name__,
                      error=str(exc).splitlines()[0][:500])
        raise
    duration = round(time.time() - started, 2)

    marker = write_marker(name, dest, **{k: v for k, v in stats.items()
                                        if isinstance(v, (int, float, str, list, dict))})
    actual_gb, estimate_note = reconcile_estimate(name, marker["bytes"])
    log_event(
        dataset=name,
        method=entry["method"],
        url=entry["url"],
        bytes=marker["bytes"],
        sha256=marker["content_hash"],
        duration_s=duration,
        status="ok",
        # Nested, not splatted: a handler returning its own "bytes" key would
        # otherwise collide with the record's own field.
        est_gb=est_gb(name),
        actual_gb=round(actual_gb, 4),
        estimate_note=estimate_note,
        stats={k: v for k, v in stats.items()
               if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
    )
    print(f"  done - {marker['files']} files, {actual_gb:.3f} GB, {duration}s")
    if estimate_note:
        # Undershoot is the dangerous direction: it trips no disk guard and
        # looks exactly like success.
        flag = "WARNING" if actual_gb < est_gb(name) else "NOTE"
        print(f"  {flag}: {estimate_note}")

    manifest.update([name])
    print("  manifest updated")
    return "fetched" if not estimate_note else "fetched (check est_gb)"


def print_plan(names: list[str]) -> None:
    print("\nFetch plan")
    print(f"  {'dataset':<14} {'method':<8} {'est_gb':>8}  {'bytes':>16}  state")
    total = 0.0
    for name in names:
        entry = dataset(name)
        gb = est_gb(name)
        total += gb
        skip, _ = completion_status(name, raw_dir(name))
        print(f"  {name:<14} {entry['method']:<8} {gb:>8.2f}  "
              f"{int(gb * GB):>16,}  {'complete' if skip else 'pending'}")
    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<14} {'':<8} {total:>8.2f}  {int(total * GB):>16,}")
    print()
    print(project_disk([n for n in names
                        if not completion_status(n, raw_dir(n))[0]]).render())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", action="append", dest="datasets", metavar="NAME",
                        help="fetch this dataset (repeatable)")
    target.add_argument("--all", action="store_true", help="fetch every enabled dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without touching the network")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--accept-licence", action="store_true",
                        help="accept the licence terms of licence-gated datasets")
    parser.add_argument("--sample-size", type=int, default=FAKEDDIT_SAMPLE_SIZE,
                        help=f"Fakeddit image sample size (default {FAKEDDIT_SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=FAKEDDIT_SEED,
                        help=f"sampling seed (default {FAKEDDIT_SEED})")
    args = parser.parse_args(argv)

    if args.all:
        names = enabled_names()
    else:
        names = []
        for name in args.datasets:
            entry = dataset(name)  # KeyError names the unknown dataset
            if not entry.get("enabled"):
                print(f"  skipping {name}: enabled: false in the register")
                continue
            names.append(name)

    if not names:
        print("Nothing to fetch.")
        return 0

    print_plan(names)

    if args.dry_run:
        print("\n--dry-run: no network access was made.")
        if args.sample_size != FAKEDDIT_SAMPLE_SIZE:
            print(f"  NOTE: --sample-size {args.sample_size} differs from the "
                  f"default {FAKEDDIT_SAMPLE_SIZE}; fakeddit's est_gb of "
                  f"{est_gb('fakeddit'):.2f} assumes the default and must be "
                  "recomputed.")
        return 0

    try:
        check_projection(names)
    except DiskBudgetError as exc:
        print(f"\nREFUSED\n{exc}", file=sys.stderr)
        return 2

    if not args.yes:
        reply = input(f"Fetch {len(names)} dataset(s)? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    ctx = Context(yes=args.yes, accept_licence=args.accept_licence,
                  sample_size=args.sample_size, seed=args.seed)

    remaining = list(names)
    results: dict[str, str] = {}
    failures = 0
    for name in names:
        remaining.remove(name)
        try:
            results[name] = fetch_one(name, ctx, [name] + remaining)
        except FetchError as exc:
            failures += 1
            results[name] = "FAILED"
            print(f"\n  ABORTED — {name}\n{exc}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print(f"\n  interrupted during {name}; partial data left resumable")
            return 130
        except Exception as exc:
            # gdown, huggingface_hub and kaggle raise their own exception types.
            # Catching only FetchError meant a gdown.DownloadError on factify2
            # ended the run and the seven datasets after it never ran.
            failures += 1
            results[name] = f"FAILED ({type(exc).__name__})"
            print(f"\n  ABORTED {name}: {type(exc).__name__}: {exc}\n",
                  file=sys.stderr)

    print("\nSummary")
    for name in names:
        print(f"  {name:<14} {results.get(name, 'not reached')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
