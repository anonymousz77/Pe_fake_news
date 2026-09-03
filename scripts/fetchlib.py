"""Shared acquisition machinery for fetch.py and hydrate.py.

Everything here is offline-testable: the disk model, the forbidden-path guard,
the ``.done`` marker protocol, the backoff policy and the sampler are all pure
functions or thin wrappers over injectable I/O. The network lives in the
callers.

Three invariants this module exists to enforce:

1. **Never fill the disk.** Every fetch is gated twice — once on the volume's
   actual free space (1.5x the dataset's estimate) and once on a projected
   committed total that includes reserves for the evidence store and working
   directories.
2. **Never retrieve a forbidden path.** The register's ``forbidden_patterns``
   are checked against an enumeration of what a fetch *would* pull, before a
   byte is transferred. A match aborts the whole dataset.
3. **Never lie about what happened.** Every fetch appends to
   ``data/FETCH_LOG.jsonl`` and flushes immediately, so an interrupted run
   still leaves an honest record.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import (  # noqa: E402
    DATA,
    INTERIM,
    PROJECT_ROOT,
    RAW,
    REPORTS,
    dataset,
    dataset_names,
    free_gb,
    raw_dir,
)
from scripts.manifest import (  # noqa: E402
    data_files,
    matches_any,
    normalise,
    sha256,
)

GB: int = 1024**3

#: Space held back for data/bible/, the evidence store. It is not in the
#: register — it is derived later — so nothing else would account for it.
BIBLE_RESERVE_GB: float = 30.0

#: Space held back for data/interim and data/processed.
WORKING_RESERVE_GB: float = 20.0

#: Ceiling on the projected committed total. Below the machine's hard 160 GB
#: with deliberate headroom: a projection that only just fits is a projection
#: that fails on the first bad estimate.
PROJECTED_CAP_GB: float = 140.0

#: A fetch needs this multiple of est_gb free before it may start. Downloads
#: need room for archives plus their extracted contents at the same time.
FREE_SPACE_FACTOR: float = 1.5

FETCH_LOG: Path = DATA / "FETCH_LOG.jsonl"

#: Completion markers live under interim, never under data/raw — raw must stay
#: exactly what was downloaded so the integrity manifest never hashes our own
#: bookkeeping.
FETCH_STATE: Path = INTERIM / "_fetch_state"

#: Retried with exponential backoff.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Never retried. A 404 does not become a 200 by asking again, and hammering
#: dead URLs is how a hydration run takes six hours to fail.
TERMINAL_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 410, 451})

_CHUNK = 1024 * 1024


class FetchError(RuntimeError):
    """Base class for acquisition failures that should abort cleanly."""


class ForbiddenPathError(FetchError):
    """A fetch would have retrieved a path the register forbids."""


class DiskBudgetError(FetchError):
    """A fetch would have exceeded the free-space or projected-disk budget."""


class CredentialError(FetchError):
    """A required credential is missing."""


class LicenceNotAcceptedError(FetchError):
    """A licence-gated dataset was requested without --accept-licence."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# register helpers
# --------------------------------------------------------------------------


def enabled_names() -> list[str]:
    """Register order, enabled entries only."""
    return [n for n in dataset_names() if dataset(n).get("enabled")]


def est_gb(name: str) -> float:
    return float(dataset(name).get("est_gb", 0.0))


def resolve_allow_patterns(entry: dict[str, Any]) -> list[str]:
    """The include globs for an entry. Empty means 'take everything'."""
    return list(entry.get("include_patterns") or [])


#: A resolved selection larger than this multiple of est_gb means the fetch
#: scope is wrong, not that the estimate was slightly off.
SELECTION_ALARM_FACTOR: float = 3.0


def require_allow_patterns(entry: dict[str, Any]) -> list[str]:
    """Allow patterns for an entry where an unfiltered fetch is a contradiction.

    Applied to any entry that declares forbidden_patterns. The register already
    says an entry with no include_patterns may not forbid anything, because
    "take everything" reaches the forbidden globs too -- this is that same rule
    enforced at fetch time, where the consequence is real bytes on disk.

    averitec is the case that matters: its full knowledge store is hundreds of
    GB against ~165 GB free, so a missing filter is not "a wider download", it
    is a disk-filling event.
    """
    patterns = resolve_allow_patterns(entry)
    if not patterns:
        raise DiskBudgetError(
            f"{entry['name']}: refusing an unfiltered {entry.get('method')} fetch. "
            "This entry declares forbidden_patterns but no include_patterns, so "
            "it would pull the entire repository -- forbidden paths included. "
            "Declare include_patterns in data/sources.yaml before fetching it."
        )
    return patterns


def allow_patterns_for(entry: dict[str, Any]) -> list[str]:
    """Include globs, required when the entry forbids anything, else optional.

    An entry with no forbidden_patterns (averimatec) may legitimately take a
    whole repository; an entry that holds something back (averitec) may not.
    """
    if entry.get("forbidden_patterns"):
        return require_allow_patterns(entry)
    return resolve_allow_patterns(entry)


def check_selection_size(
    name: str, selection_bytes: int, available_gb: float | None = None
) -> tuple[float, str | None]:
    """Compare what a fetch actually resolved to against its estimate.

    This is the guard that catches a scope error the pattern rules cannot: an
    include list that looks reasonable but selects a hundred times more data
    than the register promised. Returns (selected_gb, warning_or_None).
    """
    selected_gb = selection_bytes / GB
    estimate = est_gb(name)
    available = free_gb() if available_gb is None else available_gb

    if selected_gb > available:
        raise DiskBudgetError(
            f"{name}: the resolved fetch selects {selected_gb:.2f} GB but only "
            f"{available:.2f} GB is free. Refusing to start a download that "
            "cannot finish."
        )
    if estimate and selected_gb > estimate * SELECTION_ALARM_FACTOR:
        raise DiskBudgetError(
            f"{name}: the resolved fetch selects {selected_gb:.2f} GB, more than "
            f"{SELECTION_ALARM_FACTOR:g}x its {estimate:.2f} GB estimate. That is "
            "a wrong fetch scope, not a stale estimate -- check include_patterns "
            "before overriding."
        )
    if estimate and selected_gb > estimate:
        return selected_gb, (
            f"selection is {selected_gb:.2f} GB against a {estimate:.2f} GB "
            "estimate; update est_gb in data/sources.yaml once this lands"
        )
    return selected_gb, None


# --------------------------------------------------------------------------
# the forbidden-path guard
# --------------------------------------------------------------------------


def find_forbidden(entry: dict[str, Any], paths: Iterable[str]) -> list[tuple[str, str]]:
    """Every (path, pattern) pair where a planned path hits a forbidden glob."""
    forbidden = entry.get("forbidden_patterns") or []
    if not forbidden:
        return []
    hits: list[tuple[str, str]] = []
    for path in paths:
        for pattern in forbidden:
            if matches_any(path, [pattern]):
                hits.append((normalise(path), normalise(pattern)))
    return hits


def guard_or_abort(entry: dict[str, Any], paths: Iterable[str]) -> None:
    """Abort the whole dataset if any planned path is forbidden.

    Aborting the dataset rather than skipping the offending file is
    deliberate: if a fetch method is reaching for held-out evidence, the
    fetch plan is wrong, and quietly downloading the rest would hide that.
    """
    hits = find_forbidden(entry, paths)
    if not hits:
        return
    shown = hits[:10]
    lines = [f"    {path}   <- forbidden by {pattern!r}" for path, pattern in shown]
    if len(hits) > len(shown):
        lines.append(f"    ... and {len(hits) - len(shown)} more")
    raise ForbiddenPathError(
        f"{entry['name']}: ABORTING THE WHOLE DATASET — the fetch plan reaches "
        f"{len(hits)} path(s) that data/sources.yaml forbids:\n"
        + "\n".join(lines)
        + "\n  Held-out evidence must never enter data/raw. Fix the fetch scope "
        "or change forbidden_patterns deliberately."
    )


def select_included(entry: dict[str, Any], paths: Iterable[str]) -> list[str]:
    """Paths an entry's include_patterns actually select (empty = all)."""
    patterns = resolve_allow_patterns(entry)
    if not patterns:
        return list(paths)
    return [p for p in paths if matches_any(p, patterns)]


# --------------------------------------------------------------------------
# the disk model
# --------------------------------------------------------------------------


def dir_bytes(root: Path) -> int:
    """Bytes of data under root, excluding VCS bookkeeping."""
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in data_files(root))


@dataclass(frozen=True)
class DiskProjection:
    """Committed disk: what is on disk plus everything already promised."""

    raw_now_gb: float
    remaining_gb: float
    pending: tuple[str, ...]
    bible_reserve_gb: float = BIBLE_RESERVE_GB
    working_reserve_gb: float = WORKING_RESERVE_GB
    cap_gb: float = PROJECTED_CAP_GB

    @property
    def total_gb(self) -> float:
        return (
            self.raw_now_gb
            + self.remaining_gb
            + self.bible_reserve_gb
            + self.working_reserve_gb
        )

    @property
    def over_by_gb(self) -> float:
        return max(0.0, self.total_gb - self.cap_gb)

    @property
    def ok(self) -> bool:
        return self.total_gb <= self.cap_gb

    def render(self) -> str:
        return "\n".join(
            [
                f"  raw on disk now:     {self.raw_now_gb:8.2f} GB",
                f"  remaining to fetch:  {self.remaining_gb:8.2f} GB"
                f"  ({len(self.pending)} dataset(s))",
                f"  data/bible reserve:  {self.bible_reserve_gb:8.2f} GB",
                f"  interim/processed:   {self.working_reserve_gb:8.2f} GB",
                f"  {'-' * 34}",
                f"  projected committed: {self.total_gb:8.2f} GB"
                f"  / {self.cap_gb:.0f} GB cap",
            ]
        )


def remaining_gb(name: str) -> float:
    """How much of a dataset's estimate is still to be downloaded.

    Anything already on disk is counted in raw_now_gb, so charging the full
    est_gb again would double-count it. Before this, the projection grew as
    data landed and would eventually refuse a fetch that fits perfectly well.
    """
    try:
        root = raw_dir(name)
    except KeyError:
        return est_gb(name)  # not in the register: nothing can be on disk for it
    on_disk = dir_bytes(root) / GB
    return max(0.0, est_gb(name) - on_disk)


def project_disk(pending: Sequence[str]) -> DiskProjection:
    return DiskProjection(
        raw_now_gb=dir_bytes(RAW) / GB,
        remaining_gb=sum(remaining_gb(n) for n in pending),
        pending=tuple(pending),
    )


def _cut_advice(projection: DiskProjection) -> str:
    """Which datasets to cut to get back under the cap, largest first."""
    ranked = sorted(projection.pending, key=remaining_gb, reverse=True)
    running = projection.total_gb
    cuts: list[str] = []
    for name in ranked:
        saved = remaining_gb(name)
        running -= saved
        cuts.append(f"    cut {name:<14} -{saved:6.2f} GB -> {running:7.2f} GB")
        if running <= projection.cap_gb:
            break
    if not cuts:
        return "    (nothing pending to cut — the reserves alone exceed the cap)"
    if running > projection.cap_gb:
        cuts.append("    ...still over even with every pending dataset cut.")
    return "\n".join(cuts)


def check_projection(pending: Sequence[str]) -> DiskProjection:
    """Refuse to start if the committed total would pass the cap."""
    projection = project_disk(pending)
    if projection.ok:
        return projection
    raise DiskBudgetError(
        f"Projected committed disk exceeds the {projection.cap_gb:.0f} GB cap "
        f"by {projection.over_by_gb:.2f} GB.\n"
        + projection.render()
        + "\n  Cut one of these rather than proceeding:\n"
        + _cut_advice(projection)
    )


def check_free_space(name: str, factor: float = FREE_SPACE_FACTOR) -> float:
    """Refuse to start unless the volume has factor x est_gb free."""
    needed = est_gb(name) * factor
    available = free_gb()
    if available < needed:
        raise DiskBudgetError(
            f"{name}: not enough free space. Need {needed:.2f} GB "
            f"({factor}x its {est_gb(name):.2f} GB estimate), "
            f"have {available:.2f} GB on the volume holding {PROJECT_ROOT}.\n"
            "  Free space, lower the estimate if it is wrong, or skip this dataset."
        )
    return available


#: A completed fetch smaller than this fraction of its estimate is reported as
#: a likely scope error. Undershoot is the more dangerous direction: an
#: oversized fetch announces itself by filling the disk, while a fetch that
#: silently selects nothing looks exactly like success.
UNDERSHOOT_FRACTION: float = 0.25


def reconcile_estimate(name: str, actual_bytes: int) -> tuple[float, str | None]:
    """Compare a completed fetch against its est_gb. Returns (gb, warning).

    Runs after every fetch regardless of method, so it catches what the
    pre-fetch selection check cannot: include_patterns that match nothing.
    That failure produced an averitec fetch of 12 MB against a 12 GB estimate
    and reported success, because a fetch far UNDER its estimate trips no disk
    guard at all.
    """
    actual_gb = actual_bytes / GB
    estimate = est_gb(name)
    if not estimate:
        return actual_gb, None
    fraction = actual_gb / estimate
    if fraction < UNDERSHOOT_FRACTION:
        return actual_gb, (
            f"fetched {actual_gb:.3f} GB against a {estimate:.2f} GB estimate "
            f"({fraction:.1%}). Either include_patterns match less than intended "
            f"or est_gb is wrong -- check which before trusting this dataset."
        )
    if fraction > 1.0:
        return actual_gb, (
            f"fetched {actual_gb:.2f} GB against a {estimate:.2f} GB estimate; "
            "update est_gb in data/sources.yaml with the measured figure"
        )
    return actual_gb, None


# --------------------------------------------------------------------------
# .done markers
# --------------------------------------------------------------------------


def tree_hash(root: Path) -> tuple[str, int, int]:
    """Content hash over a directory: (sha256, file_count, total_bytes).

    Hashes the sorted ``relpath:sha256`` lines rather than the raw bytes, so a
    renamed file changes the tree hash even when its content did not.
    """
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in data_files(root):
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}:{sha256(path)}\n".encode("utf-8"))
        files += 1
        total += path.stat().st_size
    return digest.hexdigest(), files, total


def marker_path(name: str) -> Path:
    return FETCH_STATE / f"{name}.done"


def read_marker(name: str) -> dict[str, Any] | None:
    path = marker_path(name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def scope_fingerprint(entry: dict[str, Any]) -> str:
    """Stable hash of the patterns that decide WHAT a fetch retrieves.

    Stored in the .done marker so that changing include_patterns or
    forbidden_patterns invalidates a completed fetch. Without it a marker only
    proves the bytes on disk are intact -- not that they are the bytes the
    register now asks for -- and a corrected glob would be silently ignored in
    favour of stale data.
    """
    payload = json.dumps(
        {
            "include": sorted(entry.get("include_patterns") or []),
            "forbidden": sorted(entry.get("forbidden_patterns") or []),
            "method": entry.get("method"),
            "url": entry.get("url"),
            "aux_url": entry.get("aux_url"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_marker(name: str, root: Path, **extra: Any) -> dict[str, Any]:
    """Record a completed fetch.

    Handler stats are nested under "stats" rather than splatted in: a handler
    returning its own "files" or "bytes" would otherwise overwrite the values
    measured from the tree, and the marker would describe the handler's idea of
    the fetch instead of what is actually on disk.
    """
    content_hash, files, total = tree_hash(root)
    marker = {
        "dataset": name,
        "content_hash": content_hash,
        "files": files,
        "bytes": total,
        "completed_at": utcnow(),
        "scope": scope_fingerprint(dataset(name)) if name in dataset_names() else None,
        "stats": dict(extra),
    }
    FETCH_STATE.mkdir(parents=True, exist_ok=True)
    marker_path(name).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def completion_status(name: str, root: Path) -> tuple[bool, str]:
    """(skip, reason) — is this dataset already complete and intact?"""
    marker = read_marker(name)
    if marker is None:
        return False, "no .done marker"
    if not root.is_dir():
        return False, f".done marker present but {root} is missing"
    if name in dataset_names():
        expected_scope = scope_fingerprint(dataset(name))
        recorded_scope = marker.get("scope")
        if recorded_scope != expected_scope:
            # A marker with no scope predates this check; treat it as unproven
            # rather than trusted. Re-fetching costs bandwidth once; trusting
            # it could keep data the register no longer asks for indefinitely.
            why = ("was written before scope tracking existed"
                   if recorded_scope is None
                   else "include/forbidden patterns, method or url differ")
            return False, (
                f"FETCH SCOPE CHANGED ({why}). The data on disk may be intact "
                "but is not provably what the register asks for. Re-fetching."
            )

    current, files, _ = tree_hash(root)
    if current != marker.get("content_hash"):
        return False, (
            f".done marker content hash MISMATCH — {root} changed since it was "
            f"written ({marker.get('files')} files recorded, {files} on disk). "
            "Re-fetching."
        )
    return True, f"complete: {marker['files']} files, {marker['bytes'] / GB:.2f} GB"


# --------------------------------------------------------------------------
# the fetch log
# --------------------------------------------------------------------------


def log_event(**fields: Any) -> dict[str, Any]:
    """Append one JSON object to data/FETCH_LOG.jsonl and flush immediately."""
    event = {"timestamp": utcnow(), **fields}
    FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FETCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=False) + "\n")
        handle.flush()
    return event


def read_log() -> list[dict[str, Any]]:
    if not FETCH_LOG.is_file():
        return []
    events = []
    for line in FETCH_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


# --------------------------------------------------------------------------
# http: backoff and resumable download
# --------------------------------------------------------------------------


def should_retry(status: int | None) -> bool:
    """None means a transport error (timeout, reset) — those are retryable."""
    if status is None:
        return True
    if status in TERMINAL_STATUS:
        return False
    return status in RETRYABLE_STATUS


def backoff_delays(
    retries: int, base: float = 0.5, cap: float = 30.0, rng: random.Random | None = None
) -> list[float]:
    """Exponential backoff with full jitter, capped."""
    rng = rng or random.Random()
    delays = []
    for attempt in range(retries):
        ceiling = min(cap, base * (2**attempt))
        delays.append(rng.uniform(0, ceiling))
    return delays


def make_session(pool_size: int = 16):
    """A requests Session sized to match the worker pool."""
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Pe_Fake_News_Dec/1.0 (research)"})
    return session


def download_resumable(
    session,
    url: str,
    dest: Path,
    timeout: float = 30.0,
    retries: int = 4,
    sleeper=time.sleep,
    rng: random.Random | None = None,
) -> tuple[int, str]:
    """Download to ``dest``, resuming a partial ``.part`` file via HTTP Range.

    Returns (bytes_written, sha256). The file is only moved into place once the
    transfer completes, so an interrupted run never leaves a truncated file
    that looks finished.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    delays = backoff_delays(retries, rng=rng)

    last_error: str = "no attempt made"
    for attempt in range(retries + 1):
        already = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={already}-"} if already else {}
        try:
            response = session.get(url, stream=True, timeout=timeout, headers=headers)
            status = response.status_code
            if status in (200, 206):
                mode = "ab" if (status == 206 and already) else "wb"
                if mode == "wb":
                    already = 0
                with part.open(mode) as handle:
                    for chunk in response.iter_content(_CHUNK):
                        if chunk:
                            handle.write(chunk)
                digest = sha256(part)
                size = part.stat().st_size
                part.replace(dest)
                return size, digest
            last_error = f"HTTP {status}"
            if not should_retry(status):
                raise FetchError(f"{url}: {last_error} (not retryable)")
        except FetchError:
            raise
        except Exception as exc:  # transport-level: timeout, reset, DNS
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            sleeper(delays[attempt])

    raise FetchError(f"{url}: giving up after {retries + 1} attempts — {last_error}")


# --------------------------------------------------------------------------
# seeded stratified sampling
# --------------------------------------------------------------------------


def stratified_indices(labels: Sequence[Any], size: int, seed: int) -> list[int]:
    """Indices of a reproducible stratified sample preserving class balance.

    Works on the label column alone rather than whole rows, so sampling
    150,000 of Fakeddit's ~1M rows costs one column in memory, not the corpus.

    Deterministic given ``seed``: groups are visited in sorted label order and
    each group's indices are already sorted, so file or dict ordering cannot
    leak into the result. Allocation uses largest-remainder, so the sample sums
    to exactly ``size`` (or to len(labels) when that is smaller).
    """
    total = len(labels)
    if size >= total:
        return list(range(total))
    if size <= 0:
        return []

    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)

    names = sorted(groups)
    exact = {name: len(groups[name]) * size / total for name in names}
    take = {name: int(exact[name]) for name in names}

    # Largest remainder, ties broken by label name so it stays deterministic.
    shortfall = size - sum(take.values())
    ranked = sorted(names, key=lambda name: (-(exact[name] - take[name]), name))
    for name in ranked[:shortfall]:
        take[name] += 1

    rng = random.Random(seed)
    chosen: list[int] = []
    for name in names:
        indices = groups[name]
        chosen.extend(rng.sample(indices, min(take[name], len(indices))))
    return sorted(chosen)


def stratified_sample(
    rows: Sequence[dict[str, Any]], label_key: str, size: int, seed: int
) -> list[dict[str, Any]]:
    """Row-level convenience wrapper over :func:`stratified_indices`."""
    labels = [row[label_key] for row in rows]
    return [rows[i] for i in stratified_indices(labels, size, seed)]


# --------------------------------------------------------------------------
# hydration reporting
# --------------------------------------------------------------------------


@dataclass
class HydrationReport:
    """Recovery statistics for a URL-hydration run.

    ``recovery_rate_per_class`` is mandatory, not decorative. If media loss
    concentrates in one label the class balance is silently broken, and an
    overall rate of 0.85 hides a class that recovered 0.20. A report without
    the per-class breakdown is worse than no report, so serialisation refuses.
    """

    dataset: str
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    per_class_attempted: dict[str, int] = field(default_factory=dict)
    per_class_succeeded: dict[str, int] = field(default_factory=dict)
    #: How many recovered files came from the origin host vs an archive copy.
    #: Kept separate because an archived snapshot is not the same artefact the
    #: dataset authors fetched, and averaging them would hide that.
    per_source: dict[str, int] = field(default_factory=dict)
    label_field: str | None = None
    seed: int | None = None
    concurrency: int | None = None
    started_at: str = field(default_factory=utcnow)
    finished_at: str | None = None
    notes: str | None = None

    @property
    def recovery_rate(self) -> float:
        return round(self.succeeded / self.attempted, 4) if self.attempted else 0.0

    @property
    def recovery_rate_per_class(self) -> dict[str, float]:
        return {
            label: round(self.per_class_succeeded.get(label, 0) / attempted, 4)
            for label, attempted in sorted(self.per_class_attempted.items())
            if attempted
        }

    def record(self, label: str, ok: bool, reason: str | None = None) -> None:
        self.attempted += 1
        self.per_class_attempted[label] = self.per_class_attempted.get(label, 0) + 1
        if ok:
            self.succeeded += 1
            self.per_class_succeeded[label] = self.per_class_succeeded.get(label, 0) + 1
        else:
            self.failed += 1
            key = reason or "unknown"
            self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        if not self.per_class_attempted:
            raise FetchError(
                f"{self.dataset}: refusing to write a hydration report with no "
                "per-class breakdown. recovery_rate_per_class is the field that "
                "reveals a broken class balance; a report without it would hide "
                "the failure it exists to surface. Locate the label column "
                "before hydrating."
            )
        return {
            "dataset": self.dataset,
            "label_field": self.label_field,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "recovery_rate": self.recovery_rate,
            "recovery_rate_per_class": self.recovery_rate_per_class,
            "per_class_attempted": dict(sorted(self.per_class_attempted.items())),
            "per_class_succeeded": dict(sorted(self.per_class_succeeded.items())),
            "recovered_by_source": dict(sorted(self.per_source.items())),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "seed": self.seed,
            "concurrency": self.concurrency,
            "started_at": self.started_at,
            "finished_at": self.finished_at or utcnow(),
            "notes": self.notes,
        }

    def write(self) -> Path:
        payload = self.to_dict()  # raises before touching disk if incomplete
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / f"hydration_{self.dataset}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def render(self) -> str:
        lines = [
            f"  attempted {self.attempted}, succeeded {self.succeeded}, "
            f"failed {self.failed}",
            f"  overall recovery: {self.recovery_rate:.1%}",
            "  per class:",
        ]
        for label, rate in self.recovery_rate_per_class.items():
            attempted = self.per_class_attempted[label]
            flag = "   <-- LOW" if rate < 0.5 else ""
            lines.append(f"    {label:<20} {rate:6.1%}  of {attempted}{flag}")
        if self.failure_reasons:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.failure_reasons.items()))
            lines.append(f"  failures: {reasons}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# retry policies, polite throttling, and archival fallback
# --------------------------------------------------------------------------

#: Headers that make a request look like a browser fetching an image. Some
#: hosts serve images only with a plausible Referer (hotlink protection) or
#: reject unknown user agents outright. This is not evasion of a paywall or a
#: login -- it is asking for a public image the way a browser would.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class RetryPolicy:
    """Which HTTP statuses are worth trying again.

    Separated from the module-level sets so a retry pass can treat 403 as
    worth another attempt with better headers, while the ordinary pass keeps
    treating it as final. 404 and 410 are terminal under every policy: a
    deleted resource does not come back because it was asked twice.
    """

    retryable: frozenset[int] = RETRYABLE_STATUS
    terminal: frozenset[int] = TERMINAL_STATUS
    name: str = "default"

    def should_retry(self, status: int | None) -> bool:
        if status is None:  # transport error: timeout, reset, DNS
            return True
        if status in self.terminal:
            return False
        return status in self.retryable


DEFAULT_POLICY = RetryPolicy()

#: Used by --retry-failed. 403 and 406 move from terminal to retryable,
#: because with browser headers they often turn into a 200. 404 and 410 stay
#: terminal, as does 401 (authentication is not a header trick).
UNBLOCK_POLICY = RetryPolicy(
    retryable=RETRYABLE_STATUS | {403, 406},
    terminal=TERMINAL_STATUS - {403, 406},
    name="unblock",
)


def unblock_headers(url: str) -> dict[str, str]:
    """Browser headers plus a Referer pointing at the URL's own origin."""
    parsed = urlparse(url)
    headers = dict(BROWSER_HEADERS)
    if parsed.scheme and parsed.netloc:
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return headers


class HostThrottle:
    """Space out request *starts* per hostname, leaving hosts independent.

    A global rate limit would punish the long tail of one-request domains for
    the sins of the one host we are hammering. Keyed per host, a slow crawl of
    web.archive.org does not slow down anything else.
    """

    def __init__(self, delay: float, clock=time.monotonic, sleeper=time.sleep):
        self.delay = delay
        self._clock = clock
        self._sleeper = sleeper
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}

    def _lock_for(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def wait(self, url: str) -> float:
        """Block until this host may be contacted again. Returns seconds slept."""
        if self.delay <= 0:
            return 0.0
        host = (urlparse(url).hostname or "").lower()
        slept = 0.0
        with self._lock_for(host):
            now = self._clock()
            last = self._last.get(host)
            if last is not None:
                gap = self.delay - (now - last)
                if gap > 0:
                    self._sleeper(gap)
                    slept = gap
                    now = self._clock()
            self._last[host] = now
        return slept


class HostCircuitBreaker:
    """Stop asking a host that has clearly made up its mind.

    snopes.com refused all 5,814 requests in the first pass. Retrying each of
    them with the full retry budget would mean ~29,000 further requests to a
    host behind a blanket Cloudflare block -- wasteful, slow, and rude enough
    to earn an IP ban. After `threshold` consecutive failures a host is marked
    open and its remaining URLs are recorded as blocked rather than attempted.

    A single success closes the breaker again, so a host that is merely flaky
    is not written off.
    """

    def __init__(self, threshold: int = 25):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._consecutive: dict[str, int] = {}
        self._blocked: dict[str, int] = {}

    @staticmethod
    def _host(url: str) -> str:
        return (urlparse(url).hostname or "").lower()

    def is_open(self, url: str) -> bool:
        with self._lock:
            return self._host(url) in self._blocked

    def record(self, url: str, ok: bool) -> None:
        host = self._host(url)
        with self._lock:
            if ok:
                self._consecutive.pop(host, None)
                self._blocked.pop(host, None)
                return
            n = self._consecutive.get(host, 0) + 1
            self._consecutive[host] = n
            if n >= self.threshold and host not in self._blocked:
                self._blocked[host] = n

    def blocked_hosts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._blocked)


# --------------------------------------------------------------------------
# wayback machine
# --------------------------------------------------------------------------

WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
WAYBACK_HOST = "web.archive.org"


def wayback_raw_url(timestamp: str, url: str) -> str:
    """The URL for the archived bytes themselves.

    The ``id_`` suffix asks for the original unmodified resource rather than
    the rewritten page the Wayback UI serves -- without it an image request
    can come back as HTML.
    """
    return f"https://{WAYBACK_HOST}/web/{timestamp}id_/{url}"


def wayback_snapshot(session, url: str, timeout: float = 30.0) -> tuple[str, str] | None:
    """Closest available snapshot as (timestamp, raw_url), or None."""
    try:
        response = session.get(WAYBACK_AVAILABLE, params={"url": url}, timeout=timeout)
        if response.status_code != 200:
            return None
        closest = ((response.json() or {}).get("archived_snapshots") or {}).get("closest")
    except Exception:
        return None
    if not closest or not closest.get("available"):
        return None
    timestamp = closest.get("timestamp")
    if not timestamp:
        return None
    return timestamp, wayback_raw_url(timestamp, url)


# --------------------------------------------------------------------------
# per-file media provenance
# --------------------------------------------------------------------------

#: Where a media file came from. An archived copy may differ from what the
#: dataset authors fetched, so the two are never recorded as the same thing.
ORIGIN = "origin"
WAYBACK = "wayback"


class ProvenanceLedger:
    """Append-only record of where each hydrated media file came from."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, **record: Any) -> dict[str, Any]:
        entry = {"fetched_at": utcnow(), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
        return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def counts_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.read():
            source = entry.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts


class FailureLedger:
    """Which URLs failed, and why -- so a retry can skip the truly dead ones."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, failures: list[dict[str, Any]]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in failures:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return self.path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def terminal_keys(self, terminal_reasons=("http_404", "http_410")) -> set[str]:
        """Keys whose failure will not be fixed by asking again."""
        return {
            r["key"] for r in self.read()
            if r.get("reason") in terminal_reasons and "key" in r
        }
