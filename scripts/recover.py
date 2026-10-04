#!/usr/bin/env python
"""Exhaust image recovery, one route at a time, and say why the rest are dead.

    python scripts/recover.py run --dataset verite --dataset factify2 --dataset fakeddit
    python scripts/recover.py run --all --limit 20          # smoke test
    python scripts/recover.py status --all                  # per route, per class
    python scripts/recover.py dead --all                    # terminal manifests

Every missing image is tried along a fixed sequence of routes, stopping at the
first success:

    origin     the dataset's own URL again, fresh session, a different user agent
    wayback    the Internet Archive's CDX index, then the archived bytes
    alternate  another URL for the same image that the dataset itself carries
    match      (offline, scripts/match_images.py) a file already on disk

"Missing" is decided by the gates, not by the filesystem: a file that does not
decode, or whose sha256 is in data/placeholders.yaml, is a missing image.

Politeness is a hard constraint, not a default: ONE worker, at most one request
per HOST_DELAY seconds per host. At 12 workers this project measured 53.3%
errors; at one request per five seconds, 6.0%. Hosts are interleaved -- while
one host's five seconds run down the worker talks to another -- so the wall
clock is set by the busiest host, not by the sum. Redirects are followed by
hand so every hop is throttled too.

Every HTTP request is appended to that dataset's provenance ledger
(data/raw/<name>/media/provenance.jsonl) with URL, host, HTTP status, outcome
and reason, tagged with PASS_ID. The ledger is also the resume state: a route
that has a ``route_result`` row for this pass is not repeated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence
from urllib.parse import quote, unquote, urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT, raw_dir  # noqa: E402
from scripts import images  # noqa: E402
from scripts.fetchlib import (  # noqa: E402
    ALTERNATE,
    ORIGIN,
    ORIGIN_RETRY,
    WAYBACK,
    log_event,
    utcnow,
)
from scripts.hydrate import (  # noqa: E402
    RESOLVERS,
    Item,
    destination,
    looks_like_an_image,
    not_a_placeholder,
    provenance_ledger,
)

#: Tags every ledger row this pass writes. The resume state is keyed on it.
PASS_ID = "closeout-2026-10"

DATASETS = ("verite", "factify2", "fakeddit", "m4fc")

#: Seconds between two request STARTS to the same host. Not configurable down.
HOST_DELAY = 5.0

NETWORK_ROUTES = ("origin", "wayback", "alternate")

RECOVERY = INTERIM / "recovery"

#: A different browser from the 2026-09 retry (Chrome 124) and from the first
#: pass (the project's own research UA). A fresh requests.Session per run means
#: no cookies carry over either.
FRESH_HEADERS: dict[str, str] = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) "
                   "Gecko/20100101 Firefox/131.0"),
    "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-GB,en;q=0.7",
    "DNT": "1",
}

#: The Internet Archive asks automated clients to identify themselves.
ARCHIVE_HEADERS: dict[str, str] = {
    "User-Agent": "Pe_Fake_News_Dec/1.0 (research; archival image recovery; single worker)",
}

CDX = "https://web.archive.org/cdx/search/cdx"
TRANSIENT = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
MAX_HOPS = 6
MAX_BODY = 40 * 1024 * 1024
TIMEOUT = (15.0, 40.0)
#: Distinct archived contents tried per URL before the archive route gives up.
MAX_SNAPSHOTS = 3
#: Directories on these hosts are looked up in the CDX index once per
#: directory instead of once per URL: 4,691 snopes URLs sit in 144 directories.
BATCH_HOSTS = frozenset({"www.snopes.com", "media.snopes.com"})
BATCH_LIMIT = 150000


# --------------------------------------------------------------------------
# units of work
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    dataset: str
    item: Item


@dataclass
class Unit:
    """One distinct URL and every dataset item that points at it.

    Fetched once; the result fans out. Factify2 alone has 1,717 placeholder
    items behind 894 distinct URLs.
    """

    url: str
    members: list[Member] = field(default_factory=list)

    @property
    def host(self) -> str:
        return host_of(self.url)


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def ledger_paths(name: str) -> dict[str, list[str]]:
    """key -> every path a successful fetch for that key was written to."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in provenance_ledger(name).read():
        if row.get("outcome") == "ok" and row.get("path") and row.get("event") != "attempt":
            out[row["key"]].append(row["path"])
    return out


def resolve_image(name: str, item: Item, index: dict[str, dict[str, Any]],
                  recorded: dict[str, list[str]] | None = None,
                  matches: dict[str, dict[str, Any]] | None = None) -> Path | None:
    """The first USABLE file for this item, from any source, or None.

    Order: origin, origin retry, archive, alternate, then any other path the
    ledger recorded, then an offline cross-dataset match. Usable means it
    passed all three gates (see scripts/images.py), not that it exists.
    """
    candidates: list[Path] = [destination(name, item, s)
                              for s in (ORIGIN, ORIGIN_RETRY, WAYBACK, ALTERNATE)]
    for relpath in (recorded or {}).get(item.key, []):
        p = raw_dir(name) / relpath
        if p not in candidates:
            candidates.append(p)
    for path in candidates:
        # The index is built from the files on disk immediately before any
        # caller resolves against it, so a row means the file was there and
        # was examined; no second stat() per candidate.
        if images.is_usable(index.get(images.rel(path))):
            return path
    match = (matches or {}).get(item.key)
    if match and images.is_usable(match.get("row")):
        return PROJECT_ROOT / match["path"]
    return None


def load_matches(name: str) -> dict[str, dict[str, Any]]:
    path = RECOVERY / f"{name}_matches.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = row
    return out


def missing_items(name: str, index: dict[str, dict[str, Any]] | None = None
                  ) -> tuple[list[Item], list[Item]]:
    """(all items, distinct items with no usable file)."""
    resolver = RESOLVERS[name]
    items, _label_col, _extra = resolver(None) if name == "fakeddit" else resolver()
    index = images.load_index(name) if index is None else index
    recorded = ledger_paths(name)
    matches = load_matches(name)
    seen: set[str] = set()
    missing = []
    for item in items:
        if item.key in seen:
            continue
        seen.add(item.key)
        if resolve_image(name, item, index, recorded, matches) is None:
            missing.append(item)
    return items, missing


def build_units(per_dataset: dict[str, list[Item]]) -> list[Unit]:
    by_url: dict[str, Unit] = {}
    for name in DATASETS:
        for item in per_dataset.get(name, []):
            unit = by_url.setdefault(item.url, Unit(item.url))
            unit.members.append(Member(name, item))
    return list(by_url.values())


# --------------------------------------------------------------------------
# requests and the single worker
# --------------------------------------------------------------------------


@dataclass
class Req:
    url: str | None
    headers: dict[str, str] | None = None
    not_before: float = 0.0  # monotonic; a url of None is a pure wait


@dataclass
class Resp:
    url: str
    status: int | None
    headers: dict[str, str]
    body: bytes
    error: str | None
    elapsed: float


def classify_exception(exc: BaseException) -> str:
    """A transport failure, named so DNS can be told apart from a refusal."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(s in text for s in ("getaddrinfo", "name or service not known",
                               "nameresolution", "failed to resolve",
                               "no address associated", "nodename nor servname")):
        return "dns_failure"
    if "ssl" in text or "certificate" in text:
        return "ssl_error"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "toomanyredirects" in text:
        return "too_many_redirects"
    if "connection" in text or "reset" in text or "refused" in text:
        return "connection_error"
    return type(exc).__name__.lower()


class Worker:
    """Exactly one request in flight, and per-host spacing of request starts.

    Generators yield a Req and receive a Resp. The worker always serves the
    generator whose host becomes available soonest, so a five-second wait on
    one host is spent talking to others.
    """

    def __init__(self, session, delay: float = HOST_DELAY, clock=time.monotonic,
                 sleeper=time.sleep, transport: Callable[[Req], Resp] | None = None,
                 budget_bytes: int | None = None):
        if delay < HOST_DELAY and transport is None:
            raise ValueError(f"host delay below {HOST_DELAY}s is not permitted")
        self.session = session
        self.delay = delay
        self.clock = clock
        self.sleep = sleeper
        self.transport = transport or self._http
        self.next_ok: dict[str, float] = {}
        self.queues: dict[str, deque] = defaultdict(deque)
        self.waiting: list[tuple[float, int, Generator]] = []
        self.requests = 0
        self.bytes_in = 0
        self.budget_bytes = budget_bytes
        self._seq = 0

    def _http(self, req: Req) -> Resp:
        start = time.monotonic()
        try:
            r = self.session.get(req.url, headers=req.headers, timeout=TIMEOUT,
                                 allow_redirects=False, stream=True)
            chunks, size = [], 0
            for chunk in r.iter_content(64 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_BODY:
                    r.close()
                    return Resp(req.url, r.status_code, dict(r.headers), b"",
                                "body_too_large", time.monotonic() - start)
            return Resp(req.url, r.status_code, {k.lower(): v for k, v in r.headers.items()},
                        b"".join(chunks), None, time.monotonic() - start)
        except Exception as exc:  # noqa: BLE001 -- the failure is the datum
            return Resp(req.url, None, {}, b"", classify_exception(exc),
                        time.monotonic() - start)

    def _park(self, gen: Generator, req: Req) -> None:
        self._seq += 1
        if req.url is None:
            self.waiting.append((req.not_before, self._seq, gen))
        else:
            self.queues[host_of(req.url)].append((req, gen))

    def start(self, gen: Generator) -> None:
        try:
            self._park(gen, next(gen))
        except StopIteration:
            pass

    def _advance(self, gen: Generator, value: Any) -> None:
        try:
            self._park(gen, gen.send(value))
        except StopIteration:
            pass

    def run(self, progress: Callable[[], None] | None = None,
            every: int = 200) -> None:
        while True:
            now = self.clock()
            # wake pure waits whose time has come
            if self.waiting:
                due = [w for w in self.waiting if w[0] <= now]
                if due:
                    self.waiting = [w for w in self.waiting if w[0] > now]
                    for _t, _s, gen in sorted(due):
                        self._advance(gen, None)
                    continue
            ready = None
            for host, queue in self.queues.items():
                if not queue:
                    continue
                req, _gen = queue[0]
                t = max(self.next_ok.get(host, 0.0), req.not_before)
                if ready is None or t < ready[0]:
                    ready = (t, host)
            if ready is None:
                if not self.waiting:
                    return
                self.sleep(max(0.0, min(w[0] for w in self.waiting) - now))
                continue
            t, host = ready
            if self.waiting:
                t_wait = min(w[0] for w in self.waiting)
                if t_wait < t:
                    self.sleep(max(0.0, t_wait - now))
                    continue
            if t > now:
                self.sleep(t - now)
            req, gen = self.queues[host].popleft()
            if not self.queues[host]:
                del self.queues[host]
            self.next_ok[host] = self.clock() + self.delay
            if self.budget_bytes is not None and self.bytes_in > self.budget_bytes:
                raise BudgetExceeded(f"transferred {self.bytes_in:,} bytes")
            resp = self.transport(req)
            self.requests += 1
            self.bytes_in += len(resp.body)
            if progress is not None and self.requests % every == 0:
                progress()
            self._advance(gen, resp)


class BudgetExceeded(RuntimeError):
    pass


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


class Log:
    """Appends every attempt and every route result to the dataset ledgers."""

    def __init__(self):
        self.ledgers = {name: provenance_ledger(name) for name in DATASETS}

    def attempt(self, unit: Unit, route: str, resp: Resp, outcome: str,
                reason: str | None, **extra: Any) -> None:
        for m in unit.members:
            self.ledgers[m.dataset].append(
                event="attempt", **{"pass": PASS_ID}, route=route, key=m.item.key,
                label=m.item.label, url=resp.url, item_url=unit.url,
                host=host_of(resp.url), http_status=resp.status, outcome=outcome,
                attempt_reason=reason, bytes=len(resp.body),
                elapsed_s=round(resp.elapsed, 3), **extra)

    def result(self, member: Member, route: str, ok: bool, reason: str | None,
               status: int | None, path: str | None, **extra: Any) -> None:
        self.ledgers[member.dataset].append(
            event="route_result", **{"pass": PASS_ID}, route=route,
            key=member.item.key, label=member.item.label, url=member.item.url,
            host=host_of(member.item.url), http_status=status,
            outcome="ok" if ok else "failed", attempt_reason=reason, path=path,
            source={"origin": ORIGIN_RETRY, "wayback": WAYBACK,
                    "alternate": ALTERNATE}.get(route), **extra)


def done_routes(name: str) -> dict[str, dict[str, dict[str, Any]]]:
    """key -> route -> last route_result row of THIS pass."""
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in provenance_ledger(name).read():
        if row.get("event") == "route_result" and row.get("pass") == PASS_ID:
            out[row["key"]][row["route"]] = row
    return out


# --------------------------------------------------------------------------
# HTTP helpers, as generators
# --------------------------------------------------------------------------


def gate(body: bytes) -> str | None:
    """The second and third gates. None means the bytes are a real image."""
    return looks_like_an_image(body) or not_a_placeholder(body)


def get(url: str, headers: dict[str, str], log: Log, unit: Unit, route: str,
        clock=time.monotonic, retries: int = 2) -> Generator[Req, Resp, tuple[Resp, list[str]]]:
    """GET with throttled manual redirects and bounded transient retries.

    Returns (final response, redirect chain). Every hop and every retry is a
    logged attempt.
    """
    chain: list[str] = []
    attempt = 0
    current = url
    hops = 0
    not_before = 0.0
    while True:
        h = dict(headers)
        if "Referer" not in h and route != "wayback":
            p = urlparse(current)
            if p.scheme and p.netloc:
                h["Referer"] = f"{p.scheme}://{p.netloc}/"
        resp: Resp = yield Req(current, h, not_before)
        if resp.status in REDIRECTS and resp.headers.get("location") and hops < MAX_HOPS:
            target = urljoin(current, resp.headers["location"])
            log.attempt(unit, route, resp, "redirect", f"http_{resp.status}", location=target)
            chain.append(target)
            current, hops, not_before = target, hops + 1, 0.0
            continue
        transient = resp.status is None and resp.error not in ("dns_failure",) \
            or resp.status in TRANSIENT
        if transient and attempt < retries:
            wait = 30.0 * (4 ** attempt)
            if resp.status == 429:
                try:
                    wait = max(wait, min(900.0, float(resp.headers.get("retry-after", 0))))
                except ValueError:
                    pass
            reason = resp.error or f"http_{resp.status}"
            log.attempt(unit, route, resp, "retrying", reason, retry_in_s=wait)
            attempt += 1
            not_before = clock() + wait
            continue
        return resp, chain


def removed_redirect(chain: Sequence[str]) -> bool:
    """imgur and friends answer a deleted image with a redirect to a stock one."""
    return any(re.search(r"/removed\.(png|jpg)|/404|not[-_]?found", c, re.I) for c in chain)


@dataclass
class Outcome:
    ok: bool
    reason: str | None
    status: int | None
    body: bytes = b""
    extra: dict[str, Any] = field(default_factory=dict)


def fetch_live(unit: Unit, url: str, route: str, log: Log,
               clock=time.monotonic) -> Generator[Req, Resp, Outcome]:
    resp, chain = yield from get(url, FRESH_HEADERS, log, unit, route, clock)
    if resp.status == 200 and not resp.error:
        rejected = gate(resp.body)
        if rejected:
            log.attempt(unit, route, resp, "rejected", rejected)
            if removed_redirect(chain):
                rejected = f"{rejected}+redirect_to_removed"
            return Outcome(False, rejected, 200, extra={"redirects": chain})
        log.attempt(unit, route, resp, "ok", None)
        return Outcome(True, None, 200, resp.body, {"redirects": chain, "final_url": resp.url})
    reason = resp.error or f"http_{resp.status}"
    log.attempt(unit, route, resp, "failed", reason)
    if removed_redirect(chain):
        reason += "+redirect_to_removed"
    return Outcome(False, reason, resp.status, extra={"redirects": chain})


# --------------------------------------------------------------------------
# the Internet Archive
# --------------------------------------------------------------------------


def strip_scheme(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc}{p.path}"


def path_key(url: str) -> str:
    """host+path, lower-cased host, %-decoded, no scheme, no query."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{unquote(p.path)}"


def parse_cdx(body: bytes) -> list[dict[str, str]]:
    """CDX JSON output -> list of rows keyed by the header row."""
    if not body.strip():
        return []
    data = json.loads(body)
    if not data:
        return []
    header, rows = data[0], data[1:]
    return [dict(zip(header, r)) for r in rows]


def cdx_url(target: str, *, prefix: bool, limit: int = 200) -> str:
    params = [
        ("url", strip_scheme(target)),
        ("output", "json"),
        ("fl", "timestamp,original,statuscode,mimetype,digest"),
        ("filter", "statuscode:200"),
        ("collapse", "digest"),
        ("limit", str(limit)),
    ]
    if prefix:
        params.append(("matchType", "prefix"))
    # '%' is safe: a dataset URL that is already percent-encoded must reach the
    # index as written, not double-encoded into a URL nobody ever captured.
    return CDX + "?" + "&".join(f"{k}={quote(v, safe=':/,%')}" for k, v in params)


def snapshot_url(row: dict[str, str]) -> str:
    return f"https://web.archive.org/web/{row['timestamp']}id_/{row['original']}"


def choose_snapshots(url: str, rows: Iterable[dict[str, str]],
                     prefer_year: int | None = None) -> list[dict[str, str]]:
    """Captures of THIS resource, best first, one per distinct content.

    Same host and path as the dataset URL (scheme and www. ignored); a capture
    with the dataset's own query string first, then any other query variant
    of the same path, then nearest to the year the dataset was built.
    """
    want = path_key(url)
    want_query = urlparse(url).query
    same = [r for r in rows if path_key(r.get("original", "")) == want]
    same = [r for r in same if not r.get("mimetype", "").startswith("text/")]

    def rank(r: dict[str, str]) -> tuple[int, int]:
        exact = 0 if urlparse(r["original"]).query == want_query else 1
        year = int(r["timestamp"][:4]) if r.get("timestamp", "")[:4].isdigit() else 0
        return exact, abs(year - prefer_year) if prefer_year else 0

    seen: set[str] = set()
    out = []
    for r in sorted(same, key=rank):
        if r.get("digest") in seen:
            continue
        seen.add(r.get("digest"))
        out.append(r)
    return out[:MAX_SNAPSHOTS]


DATASET_YEAR = {"verite": 2022, "factify2": 2022, "fakeddit": 2019, "m4fc": 2023}


class CdxCache:
    """Directory-level CDX lookups, shared by every URL in the directory.

    Persisted under data/interim/recovery/cdx_batch/ so that a restarted run
    does not ask the archive for a directory it has already listed.
    """

    def __init__(self, root: Path | None = None):
        self.rows: dict[str, list[dict[str, str]]] = {}
        self.state: dict[str, str] = {}  # dir -> pending | done | failed
        self.root = root
        if root is not None and root.is_dir():
            for path in root.glob("*.json"):
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.rows[saved["directory"]] = saved["rows"]
                self.state[saved["directory"]] = "done"

    def save(self, directory: str, rows: list[dict[str, str]]) -> None:
        self.rows[directory] = rows
        self.state[directory] = "done"
        if self.root is not None:
            import hashlib
            self.root.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha1(directory.encode("utf-8")).hexdigest()[:16]
            (self.root / f"{name}.json").write_text(json.dumps(
                {"directory": directory, "fetched_at": utcnow(), "rows": rows}),
                encoding="utf-8")

    @staticmethod
    def directory(url: str) -> str | None:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        parts = p.path.split("/")
        if host not in BATCH_HOSTS or len(parts) < 4:
            return None
        return f"{host}{'/'.join(parts[:-1])}/"


def snopes_equivalent(url: str) -> str | None:
    """www.snopes.com/tachyon/YYYY/MM/f -> media.snopes.com/YYYY/MM/f.

    Snopes' 'tachyon' path is its image CDN in front of the media host, so
    the two name the same file. Not an alternate the dataset carries: used
    only as a last archive lookup and recorded as ``via_equivalent``.
    """
    p = urlparse(url)
    m = re.match(r"^/tachyon/(\d{4}/\d{2}/[^/]+)$", p.path)
    if (p.hostname or "").lower() == "www.snopes.com" and m:
        return f"https://media.snopes.com/{m.group(1)}"
    return None


def wayback_lookup(unit: Unit, target: str, log: Log, cache: CdxCache,
                   clock=time.monotonic, route: str = "wayback"
                   ) -> Generator[Req, Resp, tuple[list[dict[str, str]], str | None]]:
    """(candidate captures, error). error is set only when the INDEX failed:
    'no snapshot' must never be recorded for a lookup that did not happen."""
    directory = cache.directory(target)
    if directory is not None:
        while cache.state.get(directory) == "pending":
            yield Req(None, not_before=clock() + 10.0)
        if directory not in cache.state:
            cache.state[directory] = "pending"
            resp, _ = yield from get(cdx_url(directory, prefix=True, limit=BATCH_LIMIT),
                                     ARCHIVE_HEADERS, log, unit, route, clock)
            try:
                rows = parse_cdx(resp.body) if resp.status == 200 else None
            except ValueError:
                rows = None
            if rows is None or len(rows) >= BATCH_LIMIT:
                cache.state[directory] = "failed"
                log.attempt(unit, route, resp, "failed",
                            "cdx_batch_unusable", directory=directory, lookup="cdx_batch")
            else:
                cache.save(directory, rows)
                log.attempt(unit, route, resp, "ok", None, directory=directory,
                            cdx_rows=len(rows), lookup="cdx_batch")
        if cache.state.get(directory) == "done":
            return cache.rows[directory], None
        # a failed batch falls through to the per-URL lookup below
    # per-URL lookup: a prefix match on host+path finds every query variant
    # of this resource in one request; choose_snapshots() then keeps only
    # captures of exactly this path.
    resp, _ = yield from get(cdx_url(target.split("?")[0], prefix=True),
                             ARCHIVE_HEADERS, log, unit, route, clock)
    if resp.status != 200:
        reason = resp.error or f"cdx_http_{resp.status}"
        log.attempt(unit, route, resp, "failed", reason, lookup="cdx")
        return [], reason
    try:
        rows = parse_cdx(resp.body)
    except ValueError:
        log.attempt(unit, route, resp, "failed", "cdx_unparseable", lookup="cdx")
        return [], "cdx_unparseable"
    log.attempt(unit, route, resp, "ok" if rows else "empty", None,
                cdx_rows=len(rows), lookup="cdx")
    return rows, None


def fetch_wayback(unit: Unit, log: Log, cache: CdxCache, year: int | None,
                  clock=time.monotonic, route: str = "wayback"
                  ) -> Generator[Req, Resp, Outcome]:
    targets = [(unit.url, False)]
    equivalent = snopes_equivalent(unit.url)
    if equivalent:
        targets.append((equivalent, True))
    tried = 0
    index_errors = []
    last_reason = "no_snapshot"
    for target, via_equivalent in targets:
        rows, error = yield from wayback_lookup(unit, target, log, cache, clock, route)
        if error:
            index_errors.append(error)
            continue
        for snap in choose_snapshots(target, rows, year):
            tried += 1
            resp, chain = yield from get(snapshot_url(snap), ARCHIVE_HEADERS, log,
                                         unit, route, clock)
            meta = {"wayback_timestamp": snap["timestamp"], "wayback_original": snap["original"],
                    "wayback_url": snapshot_url(snap), "via_equivalent": via_equivalent}
            if resp.status == 200 and not resp.error:
                rejected = gate(resp.body)
                if not rejected:
                    log.attempt(unit, route, resp, "ok", None, **meta)
                    return Outcome(True, None, 200, resp.body, meta)
                log.attempt(unit, route, resp, "rejected", rejected, **meta)
                last_reason = f"snapshot_{rejected}"
            else:
                reason = resp.error or f"http_{resp.status}"
                log.attempt(unit, route, resp, "failed", reason, **meta)
                last_reason = f"snapshot_{reason}"
    if tried == 0 and index_errors:
        # The index could not be read. That is not the same as "no snapshot".
        return Outcome(False, f"cdx_unavailable:{index_errors[-1]}", None)
    return Outcome(False, "no_snapshot" if tried == 0 else last_reason, None,
                   extra={"snapshots_tried": tried})


# --------------------------------------------------------------------------
# alternates the datasets themselves carry
# --------------------------------------------------------------------------

_MEDIAPROXY = re.compile(r"^https?://mediaproxy\.snopes\.com/width/\d+/(https?://.+)$")


def declared_alternates(name: str, item: Item) -> list[tuple[str, str]]:
    """(url, kind) for other addresses of this image found in the dataset.

    Only verite carries any: its snopes URLs are often wrapped in snopes'
    resizing proxy, and the wrapped URL is written out in full inside the
    field. factify2 and fakeddit carry exactly one image URL per image.
    """
    out = []
    m = _MEDIAPROXY.match(item.url)
    if m:
        out.append((m.group(1), "mediaproxy_inner_url"))
    if name == "m4fc":
        archived = _m4fc_wayback().get(item.key)
        if archived:
            out.append((archived, "authors_wayback_url"))
    return out


_WAYBACK_PAGE = re.compile(r"^(https?://web\.archive\.org/web/\d{14})(?:[a-z_]{2,3})?/(.+)$")


def raw_snapshot(url: str) -> str:
    """A Wayback replay URL rewritten to ask for the original bytes (id_)."""
    m = _WAYBACK_PAGE.match(url)
    return f"{m.group(1)}id_/{m.group(2)}" if m else url


@lru_cache(maxsize=1)
def _m4fc_wayback() -> dict[str, str]:
    """M4FC's own archived copy of each image, as the authors recorded it."""
    path = raw_dir("m4fc") / "data" / "M4FC.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["image_path"]: raw_snapshot(r["wayback_image_url"]) for r in rows
            if str(r.get("wayback_image_url") or "").startswith("http")}


# --------------------------------------------------------------------------
# writing a recovered file
# --------------------------------------------------------------------------


def free_destination(name: str, item: Item, source: str) -> Path:
    """Where a recovered file goes. Never overwrites anything in data/raw.

    An origin retry whose origin path is already occupied -- by a registered
    placeholder, say -- goes to images_retry/. A second file for an occupied
    path gets a numbered stem rather than replacing the first.
    """
    if source == ORIGIN_RETRY:
        first = destination(name, item, ORIGIN)
        if not first.exists():
            return first
    path = destination(name, item, source)
    n = 2
    while path.exists():
        path = path.with_name(f"{path.stem.split('.v')[0]}.v{n}{path.suffix}")
        n += 1
    return path


def write_file(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)


# --------------------------------------------------------------------------
# the pipeline for one URL
# --------------------------------------------------------------------------


@dataclass
class Tally:
    by_route: Counter = field(default_factory=Counter)
    stored_bytes: int = 0


def _route_settled(row: dict[str, Any] | None) -> bool:
    """A route result that a re-run should not repeat."""
    if not row:
        return False
    reason = str(row.get("attempt_reason") or "")
    return not reason.startswith("cdx_unavailable") or reason.endswith("cdx_http_404")


def _still_usable(m: Member, row: dict[str, Any] | None,
                  index: dict[str, dict[str, dict]]) -> bool:
    if not row or row.get("outcome") != "ok" or not row.get("path"):
        return False
    return images.is_usable(index[m.dataset].get(images.rel(raw_dir(m.dataset) / row["path"])))


def pipeline(unit: Unit, routes: Sequence[str], done: dict[str, dict], log: Log,
             cache: CdxCache, tally: Tally, index: dict[str, dict[str, dict]],
             clock=time.monotonic) -> Generator[Req, Resp, None]:
    """Try each route in order for one URL; stop at the first success."""
    pending = list(unit.members)
    for route in routes:
        # members that already have a result for this route (resume) skip it --
        # unless that result was the archive index being unreachable, which
        # says nothing about the image and is retried on every run
        todo = [m for m in pending if not _route_settled(done[m.dataset].get(m.item.key, {}).get(route))]
        # a prior success counts only while its file is still usable: bytes
        # registered later, or rejected against a shipped fingerprint, leave
        # the item pending for the routes after this one (not this one again:
        # it would return the same bytes)
        prior_ok = [m for m in pending
                    if _still_usable(m, done[m.dataset].get(m.item.key, {}).get(route), index)]
        pending = [m for m in pending if m not in prior_ok]
        if not pending:
            return
        if not todo:
            continue
        sub = Unit(unit.url, todo)
        year = max(DATASET_YEAR.get(m.dataset, 0) for m in todo) or None
        if route == "origin":
            outcome = yield from fetch_live(sub, unit.url, "origin", log, clock)
            source = ORIGIN_RETRY
        elif route == "wayback":
            outcome = yield from fetch_wayback(sub, log, cache, year, clock)
            source = WAYBACK
        elif route == "alternate":
            alts = {u: k for m in todo for u, k in declared_alternates(m.dataset, m.item)}
            if not alts:
                for m in todo:
                    log.result(m, route, False, "no_alternate_in_metadata", None, None)
                continue
            outcome = Outcome(False, "alternates_failed", None)
            for alt_url, kind in alts.items():
                outcome = yield from fetch_live(sub, alt_url, "alternate", log, clock)
                if not outcome.ok and kind != "authors_wayback_url":
                    live_reason = outcome.reason
                    archived = yield from fetch_wayback(Unit(alt_url, todo), log, cache,
                                                        year, clock, route="alternate")
                    outcome = archived if archived.ok else Outcome(
                        False, f"live:{live_reason}|archive:{archived.reason}", None)
                if outcome.ok:
                    outcome.extra.update(alternate_url=alt_url, alternate_kind=kind)
                    break
            source = ALTERNATE
        else:
            raise ValueError(f"unknown route {route!r}")

        # A success fills every item behind this URL, including any that a
        # previous run had already failed on this route.
        for m in (pending if outcome.ok else todo):
            if outcome.ok:
                path = free_destination(m.dataset, m.item, source)
                write_file(path, outcome.body)
                if path == destination(m.dataset, m.item, ORIGIN):
                    # the first-ever origin fetch (m4fc), or a retry into a
                    # path the first pass left empty: either way, origin bytes
                    outcome.extra["source_dir"] = ORIGIN
                row = images.examine_bytes(outcome.body)
                row.update(path=images.rel(path), mtime_ns=path.stat().st_mtime_ns)
                index[m.dataset][row["path"]] = row
                tally.stored_bytes += len(outcome.body)
                tally.by_route[(m.dataset, route)] += 1
                log.result(m, route, True, None, outcome.status,
                           path.relative_to(raw_dir(m.dataset)).as_posix(),
                           sha256=row["sha256"], **outcome.extra)
            else:
                log.result(m, route, False, outcome.reason, outcome.status, None,
                           **outcome.extra)
        if outcome.ok:
            return


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(names: Sequence[str], routes: Sequence[str], limit: int | None = None,
        budget_gb: float = 120.0, session=None) -> dict[str, Any]:
    from scripts.fetchlib import make_session

    # Refresh from disk first: a run killed from outside never reaches its
    # closing build_index(), and an index that has not seen the files that run
    # recovered would report them missing again. Only new or changed files are
    # read, so this costs a directory walk, not a re-hash.
    index = {n: (images.build_index(n, workers=2, log=lambda *_: None) if n in names
                 else images.load_index(n)) for n in DATASETS}
    per_dataset: dict[str, list[Item]] = {}
    for name in names:
        _all, missing = missing_items(name, index[name])
        per_dataset[name] = missing
        print(f"[{name}] {len(missing):,} distinct images with no usable file")
    units = build_units(per_dataset)
    if limit:
        units = units[:limit]
    done = {n: done_routes(n) for n in DATASETS}
    log = Log()
    cache = CdxCache(RECOVERY / "cdx_batch")
    tally = Tally()
    session = session or make_session(pool_size=2)
    session.headers.clear()
    worker = Worker(session, budget_bytes=int(budget_gb * 1024**3))
    hosts = Counter(u.host for u in units)
    print(f"{len(units):,} distinct URLs over {len(hosts):,} hosts; busiest: "
          f"{hosts.most_common(5)}")
    print(f"routes: {' -> '.join(routes)}; one worker; {HOST_DELAY:.0f}s per host")
    started = time.time()

    def progress() -> None:
        elapsed = time.time() - started
        print(f"  {utcnow()}  requests {worker.requests:,}  in {worker.bytes_in / 1e6:,.1f} MB"
              f"  recovered {dict(tally.by_route)}  queued hosts {len(worker.queues)}"
              f"  {elapsed / 3600:.2f} h", flush=True)

    for unit in units:
        worker.start(pipeline(unit, routes, done, log, cache, tally, index))
    try:
        worker.run(progress=progress, every=250)
    finally:
        progress()
        for name in names:
            images.build_index(name, workers=4, log=lambda *_: None)
        log_event(dataset=",".join(names), method=f"recover:{PASS_ID}",
                  url=f"<{len(units)} urls>", bytes=worker.bytes_in, sha256=None,
                  duration_s=round(time.time() - started, 1), status="ok",
                  requests=worker.requests, recovered={f"{d}:{r}": n for (d, r), n
                                                       in tally.by_route.items()})
    return {"requests": worker.requests, "bytes_in": worker.bytes_in,
            "stored_bytes": tally.stored_bytes,
            "recovered": {f"{d}:{r}": n for (d, r), n in tally.by_route.items()}}


# --------------------------------------------------------------------------
# the residue: terminal reasons and the dead-images manifests
# --------------------------------------------------------------------------


def origin_category(reason: str | None, status: int | None) -> str:
    """Why the dataset's own URL cannot serve the image, in words a reader can count."""
    r = (reason or "").lower()
    if not r:
        return "not_attempted"
    if "redirect_to_removed" in r:
        return "removed_by_host"
    if r.startswith("placeholder_image"):
        return "host_serves_placeholder"
    if r in ("not_an_image", "empty_body") or r.startswith(("undecodable", "verify:", "load:")):
        return "host_serves_non_image"
    if r == "dns_failure":
        return "dns_failure"
    if r in ("timeout", "connection_error", "ssl_error", "too_many_redirects", "body_too_large"):
        return f"unreachable_{r}"
    if status in (404, 410):
        return f"gone_{status}"
    if status in (401, 403, 451):
        return f"blocked_{status}"
    if status is not None and 500 <= status < 600:
        return "server_error_5xx"
    if status is not None:
        return f"refused_http_{status}"
    return f"other_{r[:40]}"


def archive_category(reason: str | None) -> str:
    r = (reason or "")
    if not r:
        return "not_attempted"
    if r == "no_snapshot":
        return "no_snapshot"
    if r.startswith("cdx_unavailable") and r.endswith("cdx_http_404"):
        # The index ANSWERED, with a refusal for this host: 47 M4FC images on one
        # object-storage host got it on every pass over three days. Unlike a 5xx
        # or a timeout it is not transient, so it closes the row.
        return "archive_refuses_host_404"
    if r.startswith("cdx_unavailable"):
        return "archive_index_unavailable"
    if "placeholder" in r:
        return "snapshots_are_placeholders"
    if "not_an_image" in r or "undecodable" in r:
        return "snapshots_not_images"
    return "snapshots_failed"


def dead_manifest(name: str, index: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every image of ``name`` still without a usable file, with its terminal reason."""
    index = images.load_index(name) if index is None else index
    _items, missing = missing_items(name, index)
    results = done_routes(name)
    matches = load_matches(name)

    def fetched(row: dict[str, Any]) -> str | None:
        """A route that DID fetch bytes the gates later refused, and why."""
        if row.get("outcome") != "ok" or not row.get("path"):
            return None
        return images.status(index.get(images.rel(raw_dir(name) / row["path"])))

    rows = []
    for item in missing:
        if item.key in matches:
            continue
        routes = results.get(item.key, {})
        o, w, a = routes.get("origin", {}), routes.get("wayback", {}), routes.get("alternate", {})
        o_cat = (f"fetched_{fetched(o)}" if fetched(o)
                 else origin_category(o.get("attempt_reason"), o.get("http_status")))
        w_cat = (f"snapshot_{fetched(w)}" if fetched(w)
                 else archive_category(w.get("attempt_reason")))
        alt = a.get("attempt_reason") or "not_attempted"
        rows.append({
            "key": item.key, "label": item.label, "url": item.url, "host": host_of(item.url),
            "origin": o_cat, "origin_http_status": o.get("http_status"), "archive": w_cat,
            "alternate": "none_in_metadata" if alt == "no_alternate_in_metadata" else alt,
            "match": "no_match",
            "terminal_reason": f"{o_cat}+{w_cat}",
            "closed": "not_attempted" not in (o_cat, w_cat) and w_cat != "archive_index_unavailable",
        })
    return rows


def write_dead(names: Sequence[str]) -> dict[str, Any]:
    from configs.paths import DATA, REPORTS

    dead_dir = DATA / "dead"
    dead_dir.mkdir(parents=True, exist_ok=True)
    RECOVERY.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"pass": PASS_ID, "written_at": utcnow(), "datasets": {}}
    for name in names:
        rows = dead_manifest(name)
        with (RECOVERY / f"{name}_dead.jsonl").open("w", encoding="utf-8") as h:
            for row in rows:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Published manifest: the record key and the reason, nothing else -- no
        # URL, no label (both belong to the dataset's licence, not to us).
        # VERITE's image keys ARE labels (true_N / false_N), so its manifest is
        # keyed by record_id (row index), one line per row; the image key stays
        # in the private jsonl above and in records.VERITE_PRIVATE_MAP.
        published = [(row["key"], row) for row in rows]
        if name == "verite":
            from scripts.records import verite_rows_by_image, write_verite_private_map

            write_verite_private_map()
            by_image = verite_rows_by_image()
            published = [(rid, row) for row in rows
                         for rid in by_image.get(Path(row["key"]).stem, [])]
        with (dead_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as h:
            h.write("key,terminal_reason,origin,archive,alternate\n")
            order = (lambda kr: int(kr[0])) if name == "verite" else (lambda kr: kr[0])
            for key, row in sorted(published, key=order):
                key = str(key).replace('"', '""')
                h.write(f"\"{key}\",{row['terminal_reason']},{row['origin']},"
                        f"{row['archive']},{row['alternate']}\n")
        summary["datasets"][name] = {
            "dead": len(rows),
            "not_closed": sum(1 for r in rows if not r["closed"]),
            "by_terminal_reason": dict(Counter(r["terminal_reason"] for r in rows).most_common()),
            "by_origin": dict(Counter(r["origin"] for r in rows).most_common()),
            "by_archive": dict(Counter(r["archive"] for r in rows).most_common()),
            "by_class": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
            "top_hosts": dict(Counter(r["host"] for r in rows).most_common(25)),
        }
    (REPORTS / "dead_images.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dead", help="write the terminal dead-images manifests")
    d.add_argument("--dataset", action="append", dest="datasets", choices=DATASETS)
    p = sub.add_parser("run", help="try every missing image along the routes")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", action="append", dest="datasets", choices=DATASETS)
    target.add_argument("--all", action="store_true")
    p.add_argument("--routes", default=",".join(NETWORK_ROUTES))
    p.add_argument("--limit", type=int, default=None, help="first N URLs only (smoke test)")
    p.add_argument("--budget-gb", type=float, default=120.0,
                   help="abort if this run has transferred more than this")
    args = parser.parse_args(argv)
    if args.command == "dead":
        summary = write_dead(args.datasets or list(DATASETS))
        for name, info in summary["datasets"].items():
            print(f"[{name}] dead {info['dead']:,}  not closed {info['not_closed']:,}  "
                  f"{list(info['by_terminal_reason'].items())[:4]}")
        return 0
    names = list(DATASETS) if args.all else args.datasets
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    for r in routes:
        if r not in NETWORK_ROUTES:
            parser.error(f"unknown route {r!r}; network routes are {NETWORK_ROUTES}")
    if [r for r in NETWORK_ROUTES if r in routes] != routes:
        parser.error(f"routes must keep the order {NETWORK_ROUTES}")
    result = run(names, routes, limit=args.limit, budget_gb=args.budget_gb)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
