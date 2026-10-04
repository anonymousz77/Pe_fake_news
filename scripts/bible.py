#!/usr/bin/env python
"""Build the evidence store: a corpus of DATED documents for time-filtered retrieval.

    python scripts/bible.py --source wikinews
    python scripts/bible.py --source enrich  [--dry-run]
    python scripts/bible.py --source ccnews  [--warc-limit 40]
    python scripts/bible.py --requarantine   [--dry-run]
    python scripts/bible.py --index
    python scripts/bible.py --report

A claim dated D may only be checked against evidence published before D. That is
the whole point of the store, and it has one hard consequence:

    A DOCUMENT WITH NO PUBLICATION DATE CANNOT BE TIME-FILTERED, SO IT CANNOT BE
    USED. Undated documents are quarantined to data/bible/undated/, counted, and
    NEVER indexed.

A date that cannot be one does not count as having a date, and this is the same
rule rather than a second one. A document stamped ``1970-01-01`` satisfies
``published_at <= :as_of`` for EVERY claim, so a page whose real date is unknown
gets served as evidence for all of them -- including claims it postdates. So
``make_document`` refuses anything outside 1900..today, and refuses the sentinel
values that sit inside that window (the Unix epoch, 32-bit ``INT_MIN``). The
refused value is kept as ``raw_date_rejected``. ``--requarantine`` applies the
same rule to documents already in the store.

``http_last_modified`` deliberately does NOT set ``published_at``. It is the
server's file mtime, not a publication date: in a 70-URL probe only 15 pages
returned one and 9 of those 15 were dated *today*, for claims from 2018-2020.
Treating it as a publication date would silently misdate most of the store, so
it is recorded as provenance and the document is quarantined.

Budgets are hard: 30 GB stored, 60 GB transferred. Both are refused at the cap,
and cumulative transfer is written to the report so the spend is visible rather
than inferred.
"""

from __future__ import annotations

import argparse
import bz2
import collections
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import BIBLE, INTERIM, PROJECT_ROOT, REPORTS, raw_dir  # noqa: E402
from scripts.fetchlib import (  # noqa: E402
    GB,
    FetchError,
    backoff_delays,
    HostCircuitBreaker,
    HostThrottle,
    make_session,
    unblock_headers,
    utcnow,
)
from scripts.verify import iso_date  # noqa: E402

DOCS = BIBLE / "docs"
UNDATED = BIBLE / "undated"
INDEX = BIBLE / "index"
STATE = BIBLE / "_state.json"

#: Hard caps. Both refuse rather than truncate.
STORE_CAP_GB = 30.0
TRANSFER_CAP_GB = 60.0
DEFAULT_WARC_LIMIT = 40

#: Transfer is charged and persisted in slices this big, so an interrupted run
#: leaves at most this much unaccounted rather than a whole gigabyte.
_CHARGE_EVERY = 64 * 1024 * 1024

#: Month -> WARC files, chosen to maximise the share of our claims having
#: CC-NEWS coverage within the 12 months BEFORE them. Derived from the measured
#: averitec+averimatec claim-date distribution (6,623 dated claims); these 11
#: months give 99.2% of claims recent pre-claim coverage, and the 0.8% that
#: remain are almost all pre-2016 claims that CC-NEWS cannot reach at all.
CCNEWS_ALLOCATION: dict[str, int] = {
    # CC-NEWS begins 2016-08 -- probed, not assumed: every month before it
    # returns 404 for warc.paths.gz. 75 claims (1.1%) predate the crawl plus
    # its 12-month window and can never be covered; the other 98.9% are.
    # WARCs are weighted by how many claims each month actually serves, and
    # every month here was confirmed to list at least its share.
    "2016-08": 1,    # serves 5 claims
    "2016-10": 1,    # 31
    "2017-07": 1,    # 4
    "2018-02": 2,    # 122
    "2019-02": 8,    # 1,476
    "2019-10": 1,    # 105
    "2020-03": 12,   # 2,462
    "2021-01": 2,    # 217
    "2021-07": 4,    # 652
    "2022-07": 7,    # 1,423
    "2023-04": 1,    # 51
}

DATE_SOURCES = ("explicit_field", "http_last_modified", "implausible_date",
                "meta_published", "jsonld", "url_path", "none")

#: A publication date outside this window is not a date. It is a parse failure
#: wearing one, and it is a hole in the time filter rather than untidy data: a
#: document stamped 1970-01-01 satisfies `published_at <= :as_of` for every
#: claim ever made, so a page whose real date is unknown -- and might well be
#: AFTER the claim -- is served as evidence for all of them. That is precisely
#: the hindsight leakage the quarantine rule exists to prevent, so such
#: documents are quarantined exactly as an undated one is.
#:
#: The floor is 1900. The store's earliest genuine document is a New York Times
#: archive page from 1970-04-17 and nothing here reaches the 19th century, so
#: nothing real is lost. The ceiling is today, because nothing can have been
#: published tomorrow. Note what the floor does NOT catch on its own: the epoch
#: (1970) sits well inside it. See SENTINEL_DATES.
PLAUSIBLE_MIN = "1900-01-01"

#: Two year bands that are real dates in another calendar rather than junk, and
#: are therefore NOT quarantined. Solar Hijri (Jalali) runs about 621 years
#: behind the Gregorian year, so 1300-1450 covers 1921-2071; the Buddhist era
#: runs 543 ahead, so 2400-2600 covers 1857-2057. They are counted and reported
#: as a known issue instead. Converting them is a separate job and a wrong
#: conversion is worse than a flagged one -- but note the consequence, which is
#: recorded rather than hidden: until they are converted a Jalali-dated document
#: carries the same admit-everywhere property described above.
JALALI_YEARS = (1300, 1450)
BUDDHIST_YEARS = (2400, 2600)

#: Values that are not dates but the residue of a failed timestamp conversion.
#: They sit INSIDE the plausible window, which is why the window alone is not
#: enough: a 1900 floor lets every one of them through, and they are 650 of the
#: 742 documents this rule removes.
#:
#: 1970-01-01 is the Unix epoch -- a timestamp of 0, i.e. "no timestamp" -- and
#: the two neighbouring days are the same zero read in a negative or positive
#: UTC offset. Measured in this store: 575 documents on 1970-01-01 and 75 on
#: 1969-12-31, in a corpus whose real content is 2016-2023. A single genuine
#: article on that day is possible; 650 of them sharing two dates is a sentinel.
#: 1901-12-13/14 is the same failure in 32-bit signed seconds, at INT_MIN.
#:
#: Isolated old dates are NOT included and must not be. Every pre-1996 date in
#: this store that is not one of these values was checked by hand and is a real
#: archival article whose URL carries the same date -- NYT 1970, CSMonitor 1982,
#: the 1995 Congressional Record. Quarantining those would destroy correct data
#: to tidy a table.
SENTINEL_DATES = frozenset({
    "1969-12-31", "1970-01-01", "1970-01-02",   # Unix epoch, any UTC offset
    "1901-12-13", "1901-12-14",                 # 32-bit signed INT_MIN
})

#: Verdicts from `classify_date` that mean "do not admit to the index".
QUARANTINE_VERDICTS = ("sentinel", "out_of_window")

_JSONLD = re.compile(r'"datePublished"\s*:\s*"([^"]{4,40})"', re.I)
_META = re.compile(
    r'<meta[^>]+(?:article:published_time|datePublished|pubdate)[^>]+content=["\']([^"\']{4,40})', re.I)
_META_REV = re.compile(
    r'<meta[^>]+content=["\']([^"\']{4,40})["\'][^>]*(?:article:published_time|datePublished)', re.I)
_TIME = re.compile(r'<time[^>]+datetime=["\']([^"\']{4,40})', re.I)
_URL_DASH = re.compile(r"/(\d{4})-(\d{2})-(\d{2})(?:/|\b)")
_URL_SLASH = re.compile(r"/(\d{4})/(\d{2})/(\d{2})(?:/|\b)")
_WAYBACK = re.compile(r"/web/(\d{4})(\d{2})(\d{2})\d*/")


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@dataclass
class Document:
    """One record in the store. `published_at is None` means quarantined."""

    doc_id: str
    url: str
    title: str
    text: str
    published_at: str | None
    date_source: str
    date_confidence: str          # "high" | "low" | "none"
    source_domain: str
    language: str
    fetched_at: str
    content_sha256: str
    #: The parsed date that was refused for being outside the plausible window.
    #: Kept so a later calendar fix or parser fix has something to work from --
    #: quarantining must not destroy the evidence of why.
    raw_date_rejected: str | None = None
    images: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1) + "\n"


def doc_id_for(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "unknown").lower()
    return host[4:] if host.startswith("www.") else host


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _in_band(year: int, band: tuple[int, int]) -> bool:
    return band[0] <= year <= band[1]


def classify_date(published_at: str | None, today: str | None = None
                  ) -> tuple[str | None, str]:
    """(date to keep or None, verdict) for one parsed publication date.

    Verdicts:

        ok              a plausible Gregorian date; keep it
        non_gregorian   a real date in another calendar; keep it, flag it
        sentinel        a failed timestamp conversion; quarantine
        out_of_window   before 1900 or after today; quarantine

    The two refusals are named separately because they are different findings.
    `out_of_window` says the value cannot be a publication date at all;
    `sentinel` says it is a specific value that means "no value", and lands
    inside the window, which is why the window alone would miss it.

    The calendar bands are tested FIRST, because both fall outside the window --
    a Jalali year is below the floor, a Buddhist-era year above the ceiling --
    and they are the one case where being outside it is not evidence of a failed
    parse.
    """
    if not published_at:
        return None, "ok"          # already undated; not this function's job
    today = today or today_utc()
    try:
        year = int(str(published_at)[:4])
    except (TypeError, ValueError):
        return None, "out_of_window"
    if _in_band(year, JALALI_YEARS) or _in_band(year, BUDDHIST_YEARS):
        return published_at, "non_gregorian"
    if published_at in SENTINEL_DATES:
        return None, "sentinel"
    if published_at < PLAUSIBLE_MIN or published_at > today:
        return None, "out_of_window"
    return published_at, "ok"


def make_document(url: str, title: str, text: str, published_at: str | None,
                  date_source: str, date_confidence: str, *,
                  language: str = "en", images: Iterable[str] = (),
                  provenance: dict[str, Any] | None = None,
                  today: str | None = None) -> Document:
    """Build one record, refusing a date that cannot be one.

    The plausibility window is enforced HERE rather than in each extractor,
    because every source -- Wikinews' explicit field, enrich's publisher
    metadata, CC-NEWS' WARC stamp -- funnels through this one function, and a
    check that lives in three places is a check that will one day live in two.
    """
    body = (title or "") + "\n" + (text or "")
    provenance = dict(provenance or {})
    raw_date_rejected = None
    _kept, verdict = classify_date(published_at, today)
    if verdict in QUARANTINE_VERDICTS:
        raw_date_rejected = published_at
        provenance["date_rejected_because"] = verdict
        published_at, date_source, date_confidence = None, "implausible_date", "none"
    elif verdict == "non_gregorian":
        provenance["calendar"] = "non_gregorian"
    return Document(
        doc_id=doc_id_for(url), url=url, title=(title or "").strip(),
        text=(text or "").strip(), published_at=published_at,
        date_source=date_source, date_confidence=date_confidence,
        source_domain=domain_of(url), language=language, fetched_at=utcnow(),
        content_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        raw_date_rejected=raw_date_rejected,
        images=list(images), provenance=provenance)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def _valid(y: str, m: str, d: str) -> str | None:
    try:
        datetime(int(y), int(m), int(d))
    except ValueError:
        return None
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def url_date(url: str | None) -> tuple[str | None, str]:
    """(date, date_source) inferred from a URL, or (None, "none").

    Order is deliberate. A publisher filing path -- /2019/05/03/ -- is the
    publisher's own convention and normally IS the publication date. A wayback
    capture stamp -- /web/20210412093000/ -- is when someone archived the page,
    which is an upper bound on publication and usually a bad one: measured over
    570 sampled AVeriTeC archive URLs, the inner filing path predates its claim
    87.8% of the time while the capture stamp postdates it 95.4% of the time.
    So the inner path is tried first, and the capture only when nothing else is
    available -- where it is still sound, because a capture before the claim
    guarantees the content existed before the claim.
    """
    if not url:
        return None, "none"
    for pat, src in ((_URL_DASH, "url_path"), (_URL_SLASH, "url_path"),
                     (_WAYBACK, "wayback_capture")):
        m = pat.search(url)
        if m:
            got = _valid(*m.groups())
            if got:
                return got, src
    return None, "none"


def date_from_url(url: str | None) -> str | None:
    """Just the date from `url_date`, for callers that do not need provenance."""
    return url_date(url)[0]


def extract_date(html: str, headers: dict[str, str] | None,
                 url: str) -> tuple[str | None, str, str]:
    """(published_at, date_source, date_confidence).

    Order matters, and so does what is *refused*: a publisher's structured
    metadata is an assertion, a URL path is an inference, and Last-Modified is
    neither -- it is when the file changed on disk. Only the first two produce a
    usable date; Last-Modified is recorded and then quarantined.
    """
    for pat, src in ((_JSONLD, "jsonld"), (_META, "meta_published"),
                     (_META_REV, "meta_published"), (_TIME, "meta_published")):
        m = pat.search(html or "")
        if m:
            got = iso_date(m.group(1))
            if got:
                return got, src, "high"
    got, src = url_date(url)
    if got:
        return got, src, "low"
    if headers and headers.get("Last-Modified"):
        # Recorded for provenance, but NOT a publication date -> quarantined.
        return None, "http_last_modified", "none"
    return None, "none", "none"


# --------------------------------------------------------------------------
# budgets and state
# --------------------------------------------------------------------------


class BudgetError(FetchError):
    """A hard cap would be exceeded."""


def load_state() -> dict[str, Any]:
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"transfer_bytes": 0, "sources": {}, "started_at": utcnow()}


def save_state(state: dict[str, Any]) -> None:
    BIBLE.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def store_bytes() -> int:
    """Bytes held by the store. Called before every run, so it must stay cheap.

    `os.walk` over `DirEntry.stat()` rather than `Path.rglob`: the size comes
    back with the directory entry instead of costing a second syscall and a
    Path object per file, which at a store of this size is the difference
    between seconds and minutes.
    """
    total = 0
    for folder in (DOCS, UNDATED, INDEX):
        if not folder.is_dir():
            continue
        for root, _dirs, _files in os.walk(folder):
            for entry in os.scandir(root):
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    return total


class Budget:
    """Hard caps on stored bytes and transferred bytes. Refuses, never truncates."""

    def __init__(self, state: dict[str, Any],
                 store_cap_gb: float = STORE_CAP_GB,
                 transfer_cap_gb: float = TRANSFER_CAP_GB):
        self.state = state
        self.store_cap = store_cap_gb * GB
        self.transfer_cap = transfer_cap_gb * GB
        self._store = store_bytes()

    @property
    def transferred(self) -> int:
        return int(self.state.get("transfer_bytes", 0))

    def spend(self, nbytes: int) -> None:
        if self.transferred + nbytes > self.transfer_cap:
            raise BudgetError(
                f"transfer cap reached: {self.transferred/GB:.2f} GB spent of "
                f"{self.transfer_cap/GB:.0f} GB, and this would add "
                f"{nbytes/GB:.2f} GB. Raise --transfer-cap deliberately or "
                "narrow the source.")
        self.state["transfer_bytes"] = self.transferred + nbytes

    def store(self, nbytes: int) -> None:
        if self._store + nbytes > self.store_cap:
            raise BudgetError(
                f"store cap reached: {self._store/GB:.2f} GB on disk of "
                f"{self.store_cap/GB:.0f} GB. Raise --store-cap deliberately.")
        self._store += nbytes

    def render(self) -> str:
        return (f"  store    {self._store/GB:7.3f} / {self.store_cap/GB:.0f} GB"
                f"    transfer {self.transferred/GB:7.3f} / "
                f"{self.transfer_cap/GB:.0f} GB")


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_document(doc: Document, budget: Budget) -> str:
    """Write to docs/ or undated/. Returns 'indexed' | 'quarantined' | 'skipped'."""
    dated = doc.published_at is not None
    folder = DOCS if dated else UNDATED
    path = folder / f"{doc.doc_id}.json"
    if path.exists():
        return "skipped"
    payload = doc.to_json().encode("utf-8")
    budget.store(len(payload))
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return "indexed" if dated else "quarantined"


def requarantine(dry_run: bool = False, today: str | None = None
                 ) -> dict[str, Any]:
    """Re-apply the plausibility window to documents already in the store.

    `make_document` stops the next implausible date at the door. This is for the
    ones already inside, and it exists as its own command rather than as a step
    of `--index` because moving a document between `docs/` and `undated/` is a
    change to the STORE, and the store is not something an index build may edit
    as a side effect.

    Order matters on each move: the quarantined copy is written first and the
    dated copy removed second, so an interruption leaves the document in both
    places rather than in neither. Re-running then finishes the job -- the
    reverse order could lose a document outright.
    """
    today = today or today_utc()
    stats: collections.Counter = collections.Counter()
    moved_years: collections.Counter = collections.Counter()
    non_gregorian: collections.Counter = collections.Counter()

    for d in read_documents(DOCS):
        pub, did = d.get("published_at"), d.get("doc_id")
        stats["scanned"] += 1
        if not did:
            stats["no_doc_id"] += 1
            continue
        _kept, verdict = classify_date(pub, today)
        if verdict == "non_gregorian":
            non_gregorian[str(pub)[:4]] += 1
            stats["non_gregorian_kept"] += 1
            continue
        if verdict not in QUARANTINE_VERDICTS:
            continue
        moved_years[str(pub)[:4]] += 1
        stats["quarantined"] += 1
        stats[f"quarantined_{verdict}"] += 1
        if dry_run:
            continue
        d["raw_date_rejected"] = pub
        d["published_at"] = None
        d["date_source"] = "implausible_date"
        d["date_confidence"] = "none"
        prov = d.setdefault("provenance", {}) or {}
        prov["requarantined_at"] = utcnow()
        prov["date_rejected_because"] = verdict
        prov["requarantine_reason"] = (
            f"published_at {pub!r}: {verdict} "
            f"(window {PLAUSIBLE_MIN}..{today})")
        d["provenance"] = prov
        UNDATED.mkdir(parents=True, exist_ok=True)
        dest = UNDATED / f"{did}.json"
        tmp = dest.with_suffix(".json.part")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        tmp.replace(dest)
        (DOCS / f"{did}.json").unlink(missing_ok=True)

    return {**dict(stats), "window": {"min": PLAUSIBLE_MIN, "max": today},
            "quarantined_by_year": dict(moved_years.most_common()),
            "non_gregorian_by_year": dict(non_gregorian.most_common())}


def already_have(url: str) -> bool:
    did = doc_id_for(url)
    return (DOCS / f"{did}.json").exists() or (UNDATED / f"{did}.json").exists()


def _pick_parser() -> str:
    try:
        import lxml  # noqa: F401
    except ImportError:
        return "html.parser"
    return "lxml"


#: Chosen once at import; recorded in the report so a corpus built with a
#: different parser is identifiable rather than silently mixed.
_PARSER = _pick_parser()

MIN_PAYLOAD_BYTES = 1500      # below this a page cannot yield 200 chars of prose
_HTML_SNIFF = 4096            # window searched for a markup marker


def looks_like_an_article(payload: bytes) -> bool:
    """Cheap reject BEFORE parsing, applied to the PAYLOAD not the WARC record.

    A full bs4 parse of a record the 200-character text rule will discard anyway
    is the largest single cost in a CC-NEWS WARC. Deciding on raw bytes costs
    microseconds instead of milliseconds.

    Two things this got wrong once and must not again:

    * A WARC response record begins with HTTP headers. Sniffing the record body
      reads headers, never markup -- that version rejected 93% of records and
      lost 522 of 550 real documents.
    * Real pages do not start at `<html`. They open with BOMs, doctypes,
      comments and conditional blocks, so the window is 4 KB and several
      markers count.
    """
    if len(payload) < MIN_PAYLOAD_BYTES:
        return False
    head = payload[:_HTML_SNIFF].lower()
    return (b"<html" in head or b"<!doctype" in head or b"<head" in head
            or b"<body" in head or b"<meta" in head or b"<div" in head
            or b"<title" in head or b"<p>" in head)


_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


#: Blocks whose text is chrome, not article. bs4 drops these with decompose(),
#: which is where nearly all of its cost is; the same job by regex is ~15x
#: cheaper and, measured over 400 real crawl records, keeps 100% of bs4's
#: article words at the median.
_BOILERPLATE = re.compile(
    r"(?is)<(script|style|nav|footer|header|aside|form|noscript|svg)\b.*?</\1>")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_tags(html: str) -> tuple[str, str]:
    """(title, text) without a parser. The floor every other path falls back to.

    Also the default extractor for bulk sources: a WARC holds ~60k pages and at
    bs4's 56 ms each that is ~56 minutes per WARC against ~4 minutes here.
    """
    html = html or ""
    m = _TITLE_TAG.search(html)
    title = _WS.sub(" ", _TAG.sub(" ", m.group(1))).strip() if m else ""
    body = _BOILERPLATE.sub(" ", html)
    return title, _WS.sub(" ", _TAG.sub(" ", body)).strip()


class _CountingReader(io.RawIOBase):
    """Wraps a stream and tallies bytes actually read from it.

    Transfer budgets must be charged for what crosses the network. A WARC is
    gzipped, so measuring after decompression overstates it by roughly 4x.
    """

    def __init__(self, raw):
        self._raw = raw
        self.count = 0

    def read(self, size=-1):
        chunk = self._raw.read(size)
        self.count += len(chunk)
        return chunk

    def readable(self):
        return True


def html_to_text(html: str, extractor: str = "bs4") -> tuple[str, str]:
    """(title, text) from HTML.

    `extractor="fast"` uses the regex path -- 15x cheaper and what bulk sources
    use. `extractor="bs4"` parses properly and is the default for per-URL work,
    where a few milliseconds per page is irrelevant.

    Never raises. Real-world pages include ones bs4 refuses outright -- binary
    bytes inside a marked section, for instance -- and a builder streaming
    thousands of documents cannot afford to die on one of them. A rejected page
    falls back to `_strip_tags`, so it still yields text rather than vanishing.
    """
    if extractor == "fast":
        return _strip_tags(html)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _strip_tags(html)
    try:
        # lxml is a C parser and roughly 5x html.parser, which is the whole
        # cost of a CC-NEWS WARC. html.parser stays as the fallback so the
        # module still works if lxml is not installed.
        soup = BeautifulSoup(html, _PARSER)
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return _strip_tags(html)
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")
    parts = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])]
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if len(text) < 200:                      # fall back to the whole body
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title, text


# --------------------------------------------------------------------------
# source D: wikinews
# --------------------------------------------------------------------------

WIKINEWS_DUMP = "https://dumps.wikimedia.org/enwikinews/{date}/enwikinews-{date}-pages-articles.xml.bz2"
_WN_DATE_TPL = re.compile(r"\{\{\s*date\s*\|\s*([^}|]+?)\s*\}\}", re.I)
_WN_DATE_CAT = re.compile(r"\[\[Category:(\w+ \d{1,2}, \d{4})\]\]", re.I)
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August",
     "September","October","November","December"], 1)}


def _wikinews_date(wikitext: str) -> str | None:
    for pat in (_WN_DATE_TPL, _WN_DATE_CAT):
        m = pat.search(wikitext or "")
        if not m:
            continue
        raw = m.group(1).strip()
        mm = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", raw)
        if mm and mm.group(1).lower() in _MONTHS:
            return _valid(mm.group(3), f"{_MONTHS[mm.group(1).lower()]:02d}",
                          mm.group(2).zfill(2))
        got = iso_date(raw)
        if got:
            return got
    return None


def _strip_wikitext(text: str) -> str:
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_wikinews(budget: Budget, state: dict, dry_run: bool,
                   dump_date: str = "20260901") -> dict[str, Any]:
    """Wikinews: 40 MB, CC BY-SA 4.0 + GFDL, natively dated, never blocked.

    This is the fixture the retrieval tests run against, so the suite never
    depends on gated or blockable content.
    """
    url = WIKINEWS_DUMP.format(date=dump_date)
    local = BIBLE / "_src" / f"enwikinews-{dump_date}-pages-articles.xml.bz2"
    stats = collections.Counter()

    if not local.exists():
        if dry_run:
            print(f"  would download {url}")
            return {"dry_run": True}
        local.parent.mkdir(parents=True, exist_ok=True)
        session = make_session(2)
        print(f"  downloading {url}")
        r = session.get(url, timeout=120, stream=True)
        r.raise_for_status()
        tmp = local.with_suffix(".part")
        n = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk); n += len(chunk)
        budget.spend(n)
        tmp.replace(local)
        print(f"  {n/1024**2:.1f} MB")

    page_re = re.compile(r"<page>(.*?)</page>", re.S)
    title_re = re.compile(r"<title>(.*?)</title>", re.S)
    text_re = re.compile(r'<text[^>]*>(.*?)</text>', re.S)
    ns_re = re.compile(r"<ns>(\d+)</ns>")

    buf = ""
    with bz2.open(local, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            buf += line
            if "</page>" not in buf:
                continue
            for raw in page_re.findall(buf):
                stats["pages"] += 1
                ns = ns_re.search(raw)
                if ns is None or ns.group(1) != "0":
                    stats["skipped_non_article"] += 1   # talk, category, template
                    continue
                title = (title_re.search(raw).group(1) if title_re.search(raw) else "").strip()
                wt = (text_re.search(raw).group(1) if text_re.search(raw) else "")
                published = _wikinews_date(wt)
                text = _strip_wikitext(wt)
                if len(text) < 120:
                    stats["skipped_stub"] += 1
                    continue
                page_url = ("https://en.wikinews.org/wiki/"
                            + title.replace(" ", "_"))
                if already_have(page_url):
                    stats["skipped_existing"] += 1
                    continue
                doc = make_document(
                    page_url, title, text, published,
                    "explicit_field" if published else "none",
                    "high" if published else "none",
                    provenance={"source": "wikinews", "dump": dump_date,
                                "licence": "CC BY-SA 4.0 + GFDL"})
                if dry_run:
                    stats["would_write"] += 1
                else:
                    stats[write_document(doc, budget)] += 1
            buf = buf[buf.rfind("</page>") + 7:]
    return dict(stats)


# --------------------------------------------------------------------------
# source A: enrichment of bundled evidence
# --------------------------------------------------------------------------

#: mocheg is deliberately absent. It ships NO claim-date field, so nothing in it
#: can be placed on a timeline relative to its claim -- 43,148 evidence URLs,
#: 78% of all bundled evidence, that time-filtered retrieval cannot use. That is
#: a property of the dataset, not of this pipeline.
ENRICH_DATASETS = {
    "averitec": (["data/train.json", "data/dev.json", "data/test.json"], "claim_date"),
    "averimatec": (["train.json", "val.json"], "date"),
}


def enrich_targets() -> list[tuple[str, str, str]]:
    """(dataset, claim_date, evidence_url) for evidence whose claim is dated."""
    out = []
    for name, (files, field_) in ENRICH_DATASETS.items():
        for rel in files:
            p = raw_dir(name) / rel
            if not p.is_file():
                continue
            for rec in json.loads(p.read_text(encoding="utf-8")):
                cd = iso_date(rec.get(field_))
                if not cd:
                    continue
                for q in (rec.get("questions") or []):
                    for a in (q.get("answers") or []):
                        u = a.get("source_url")
                        if u:
                            out.append((name, cd, u))
    return out


def build_enrich(budget: Budget, state: dict, dry_run: bool,
                 limit: int | None = None) -> dict[str, Any]:
    targets = enrich_targets()
    seen, uniq = set(), []
    for ds, cd, u in targets:
        if u not in seen:
            seen.add(u); uniq.append((ds, cd, u))
    if limit:
        uniq = uniq[:limit]
    stats = collections.Counter()
    stats["evidence_urls"] = len(uniq)
    if dry_run:
        doms = collections.Counter(domain_of(u) for _, _, u in uniq)
        stats["distinct_domains"] = len(doms)
        stats["already_have"] = sum(1 for _, _, u in uniq if already_have(u))
        print(f"  {len(uniq):,} distinct evidence URLs, {len(doms):,} domains")
        print(f"  mocheg excluded: no claim-date field (43,148 URLs)")
        return dict(stats)

    session = make_session(8)
    general = HostThrottle(1.0)
    archive = HostThrottle(5.0)      # wayback is fragile; pace it far more gently
    breaker = HostCircuitBreaker(25)

    for i, (ds, claim_date, url) in enumerate(uniq, 1):
        if already_have(url):
            stats["skipped_existing"] += 1
            continue
        dom = domain_of(url)
        if breaker.is_open(url):
            stats["host_circuit_open"] += 1
            continue
        (archive if "archive" in dom else general).wait(url)
        html, headers, nbytes = "", {}, 0
        try:
            r = session.get(url, headers=unblock_headers(url), timeout=25, stream=True)
            if r.status_code == 200:
                raw = r.raw.read(400_000, decode_content=True)
                nbytes = len(raw)
                html = raw.decode("utf-8", "replace")
                headers = dict(r.headers)
                breaker.record(url, True)
            else:
                stats[f"http_{r.status_code}"] += 1
                breaker.record(url, False)
                continue
        except Exception as exc:
            stats[type(exc).__name__.lower()[:24]] += 1
            breaker.record(url, False)
            continue
        try:
            budget.spend(nbytes)
        except BudgetError:
            stats["stopped_on_transfer_cap"] += 1
            break
        try:
            published, src, conf = extract_date(html, headers, url)
            title, text = html_to_text(html)
            doc = make_document(url, title, text, published, src, conf,
                                provenance={"source": "enrich", "dataset": ds,
                                            "claim_date": claim_date,
                                            "capture_bound": src == "wayback_capture",
                                            "extractor": "bs4"})
            stats[write_document(doc, budget)] += 1
        except BudgetError:
            # A hard cap stops the run; it is not a per-document failure.
            stats["stopped_on_store_cap"] += 1
            break
        except Exception as exc:
            stats[f"unprocessable_{type(exc).__name__.lower()[:20]}"] += 1
            continue
        stats[f"date_{src}"] += 1
        if i % 200 == 0:
            print(f"  {i}/{len(uniq)}  indexed={stats['indexed']} "
                  f"quarantined={stats['quarantined']}", flush=True)
    return dict(stats)


# --------------------------------------------------------------------------
# source B: CC-NEWS
# --------------------------------------------------------------------------

CC_BASE = "https://data.commoncrawl.org"


class MonthUnavailable(FetchError):
    """A month's WARC listing could not be read.

    `permanent` separates "the crawl has no such month" from "the server had a
    bad minute". The two look identical at the call site and must not: one is a
    fact about CC-NEWS, the other is a reason to try again.
    """

    def __init__(self, message: str, *, permanent: bool):
        super().__init__(message)
        self.permanent = permanent


#: `make_session`'s argument is a connection-pool size, not a retry count -- it
#: mounts no Retry at all -- so this listing got exactly one attempt. A single
#: 503 therefore dropped a whole month, and the builder's skip-and-continue
#: turned that into silence: the first resume lost 2020-03 and 2021-01 that way,
#: 14 of 40 WARCs, including the month serving more claims than any other.
_WARC_PATHS_ATTEMPTS = 6


def warc_paths(session, month: str, attempts: int = _WARC_PATHS_ATTEMPTS,
               sleeper=time.sleep, rng=None) -> list[str]:
    """The WARC paths listed for `month`, retried through transient failures."""
    y, m = month.split("-")
    url = f"{CC_BASE}/crawl-data/CC-NEWS/{y}/{m}/warc.paths.gz"
    delays = backoff_delays(max(0, attempts - 1), base=2.0, cap=60.0, rng=rng)
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            r = session.get(url, timeout=90)
        except Exception as exc:                       # connection-level failure
            last = f"{type(exc).__name__}: {exc}"[:120]
        else:
            if r.status_code == 200:
                return gzip.decompress(r.content).decode().split()
            if r.status_code == 404:
                # CC-NEWS begins 2016-08; earlier months answer 404 forever.
                raise MonthUnavailable(
                    f"CC-NEWS {month}: HTTP 404 -- the crawl has no such month",
                    permanent=True)
            last = f"HTTP {r.status_code}"
        if attempt < attempts - 1:
            sleeper(delays[attempt])
    raise MonthUnavailable(
        f"CC-NEWS {month}: {last} after {attempts} attempts", permanent=False)


def iter_warc_records(stream: io.BufferedReader) -> Iterator[tuple[dict, bytes]]:
    """Yield (headers, body) for each WARC record in a decompressed stream."""
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if not line.startswith(b"WARC/"):
            continue
        headers: dict[str, str] = {}
        while True:
            hl = stream.readline()
            if not hl or hl in (b"\r\n", b"\n"):
                break
            if b":" in hl:
                k, _, v = hl.decode("utf-8", "replace").partition(":")
                headers[k.strip()] = v.strip()
        length = int(headers.get("Content-Length", 0))
        body = stream.read(length)
        stream.readline(); stream.readline()
        yield headers, body


def build_ccnews(budget: Budget, state: dict, dry_run: bool,
                 warc_limit: int = DEFAULT_WARC_LIMIT,
                 allocation: dict[str, int] | None = None,
                 extractor: str = "fast") -> dict[str, Any]:
    alloc = dict(allocation or CCNEWS_ALLOCATION)
    scale = warc_limit / max(1, sum(alloc.values()))
    if scale != 1.0:
        alloc = {m: max(1, round(n * scale)) for m, n in alloc.items()}
    stats = collections.Counter()
    stats["months"] = len(alloc)
    print(f"  allocation across {len(alloc)} months, "
          f"{sum(alloc.values())} WARC files (~1 GB each):")
    for m, n in sorted(alloc.items()):
        print(f"    {m}  {n:>2}")
    if dry_run:
        stats["would_transfer_gb"] = sum(alloc.values())
        return dict(stats)

    session = make_session(4)
    done = state["sources"].setdefault("ccnews", {}).setdefault("warcs_done", [])
    for month, want in sorted(alloc.items()):
        y, m = month.split("-")
        month_done = sum(1 for p in done if f"/CC-NEWS/{y}/{m}/" in p)
        try:
            paths = warc_paths(session, month)
        except MonthUnavailable as exc:
            # Name the cost in WARCs, not just in months. A month is worth
            # between 1 and 12 of them, so "1 month unavailable" understates a
            # 12-WARC loss by an order of magnitude and reads like a rounding
            # error in the stats line.
            lost = max(0, want - month_done)
            print(f"  {exc}  [{lost} WARC(s) not fetched]")
            stats["month_unavailable"] += 1
            stats["months_absent" if exc.permanent
                  else "months_transiently_unavailable"] += 1
            stats["warcs_unfetched"] += lost
            continue
        # `want` is this month's share of the TOTAL --warc-limit, so WARCs
        # already taken for it count against it. Dropping them from the
        # candidate list only prevents a refetch -- the quota itself refilled,
        # so resuming a 40-WARC build fetched 40 MORE (42 in all) and gave a
        # second WARC to whichever months had already run, which are the ones
        # that ran first because they are cheapest, not because they matter most.
        chosen = [p for p in paths
                  if p not in done][:max(0, want - month_done)]
        for path in chosen:
            print(f"  {month}: {path.split('/')[-1]}", flush=True)
            try:
                budget.spend(0)
                r = session.get(f"{CC_BASE}/{path}", timeout=300, stream=True)
                if r.status_code != 200:
                    stats[f"warc_http_{r.status_code}"] += 1
                    continue
                moved = 0
                charged = 0
                wire = _CountingReader(r.raw)   # bytes off the socket, not after gunzip
                gz = gzip.GzipFile(fileobj=wire)
                buf = io.BufferedReader(gz, buffer_size=1 << 20)
                for headers, body in iter_warc_records(buf):
                    # Charge the ledger as the bytes move, not once the WARC
                    # finishes. Charging on completion means an interrupted
                    # WARC's bytes -- up to a gigabyte that really crossed the
                    # network -- are never recorded, so the ledger under-reports
                    # in exactly the case where it is being consulted.
                    if wire.count - charged >= _CHARGE_EVERY:
                        budget.spend(wire.count - charged)
                        charged = wire.count
                        save_state(state)
                    if headers.get("WARC-Type") != "response":
                        continue
                    moved += len(body)
                    url = headers.get("WARC-Target-URI", "")
                    if not url or already_have(url):
                        stats["skipped_existing"] += 1
                        continue
                    head, _, payload = body.partition(b"\r\n\r\n")
                    if not looks_like_an_article(payload):
                        stats["skipped_before_parse"] += 1
                        continue
                    http_headers = {}
                    for hl in head.decode("utf-8", "replace").splitlines()[1:]:
                        if ":" in hl:
                            k, _, v = hl.partition(":")
                            http_headers[k.strip()] = v.strip()
                    html = payload.decode("utf-8", "replace")
                    published, src, conf = extract_date(html, http_headers, url)
                    if published is None:
                        warc_date = headers.get("WARC-Date", "")
                        got = iso_date(warc_date[:10])
                        if got:
                            # Crawl time is LATER than publication: a sound
                            # conservative bound, never an assertion -- and not
                            # a url_path, which is the publisher's own filing
                            # convention. Kept distinct so the report cannot
                            # average a weak bound into a strong signal.
                            published, src, conf = got, "warc_date", "low"
                    title, text = html_to_text(html, extractor)
                    if len(text) < 200:
                        stats["skipped_short"] += 1
                        continue
                    doc = make_document(
                        url, title, text[:20000], published, src, conf,
                        provenance={"source": "ccnews", "month": month,
                                    "warc": path.split("/")[-1],
                                    "warc_date": headers.get("WARC-Date"),
                                    "extractor": extractor,
                                    "licence": "Common Crawl ToU; page content "
                                               "copyright its publisher"})
                    try:
                        stats[write_document(doc, budget)] += 1
                    except BudgetError as exc:
                        print(f"  {exc}")
                        stats["stopped_on_store_cap"] += 1
                        save_state(state)
                        return dict(stats)
                    stats[f"date_{src}"] += 1
                budget.spend(wire.count - charged)
                done.append(path)
                save_state(state)
                print(f"    {wire.count/1024**2:.0f} MB transferred, "
                      f"{moved/1024**2:.0f} MB scanned; "
                      f"indexed={stats['indexed']} quarantined={stats['quarantined']}")
            except BudgetError as exc:
                print(f"  {exc}")
                stats["stopped_on_transfer_cap"] += 1
                save_state(state)
                return dict(stats)
            except Exception as exc:
                stats[f"warc_{type(exc).__name__.lower()[:20]}"] += 1

    if stats["warcs_unfetched"]:
        print("")
        print(f"  WARNING: {stats['warcs_unfetched']} of {sum(alloc.values())} "
              f"allocated WARCs were NOT fetched because "
              f"{stats['month_unavailable']} month(s) could not be listed. "
              f"Transiently-unavailable months are worth re-running; "
              f"permanently absent ones are not.")
    return dict(stats)


# --------------------------------------------------------------------------
# reading the store back
# --------------------------------------------------------------------------

#: Walking the store is open()-bound, not CPU-bound. Measured over 4,000 real
#: documents on this machine: 1 thread 139 files/s, 16 threads 987 -- 7.1x,
#: because each read blocks in a syscall that releases the GIL. Both --index and
#: --report walk the whole store, so at its size that is the difference between
#: a quarter-hour and most of two hours, twice.
_READ_THREADS = 16

#: 4096 was too many. `ThreadPoolExecutor.map` submits every item in the chunk
#: at once, so a chunk's worth of parsed documents can sit completed-but-unread
#: at the same time. On a 16 GB machine already holding the file cache for a
#: 1.37M-file walk, that was enough to get the report killed for memory twice.
_READ_CHUNK = 1024


def read_documents(folder: Path, strip_text: bool = False
                   ) -> Iterator[dict[str, Any]]:
    """Every parseable document in `folder`, in filename order.

    A file that will not parse is skipped, not fatal. A build killed mid-write
    leaves a truncated document behind -- `write_document` writes to `.part` and
    renames precisely so that cannot happen, but the report must survive one
    anyway rather than lose the other million.

    `strip_text` drops the body before the document is yielded. The report reads
    dates, sources and domains and never touches the text, but the text is
    almost all of the bytes -- up to 20,000 characters per document -- so
    carrying it through the walk costs most of the memory for none of the
    answers. Callers that REWRITE a document must not use it.
    """
    if not folder.is_dir():
        return
    names = sorted(e.name for e in os.scandir(folder) if e.name.endswith(".json"))

    def _read(name: str) -> dict[str, Any] | None:
        try:
            doc = json.loads((folder / name).read_bytes())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if strip_text and isinstance(doc, dict):
            doc.pop("text", None)
        return doc

    # Chunked rather than one map() over the whole list: at store scale that
    # would allocate a future per document before reading any of them.
    with ThreadPoolExecutor(_READ_THREADS) as pool:
        for i in range(0, len(names), _READ_CHUNK):
            for doc in pool.map(_read, names[i:i + _READ_CHUNK]):
                if doc is not None:
                    yield doc


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def build_index() -> dict[str, Any]:
    """FTS5 for BM25 and a separate temporal table, joined at query time.

    Both are SQLite, which ships with Python -- no new dependency, and `bm25()`
    is built into FTS5. Keeping the date in its own database and JOINing means
    the time filter is part of the query rather than something applied to
    results afterwards.
    """
    INDEX.mkdir(parents=True, exist_ok=True)
    bm25_db, temporal_db = INDEX / "bm25.sqlite", INDEX / "temporal.sqlite"
    for p in (bm25_db, temporal_db):
        if p.exists():
            p.unlink()

    bm = sqlite3.connect(bm25_db)
    bm.execute("CREATE VIRTUAL TABLE docs_fts USING fts5("
               "doc_id UNINDEXED, title, text, tokenize='porter unicode61')")
    tm = sqlite3.connect(temporal_db)
    tm.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, published_at TEXT "
               "NOT NULL, source_domain TEXT, url TEXT)")

    n = 0
    for d in read_documents(DOCS):      # NOT strip_text: the body IS the index
        if not d.get("published_at"):
            continue          # belt and braces: docs/ should never hold these
        bm.execute("INSERT INTO docs_fts (doc_id, title, text) VALUES (?,?,?)",
                   (d["doc_id"], d.get("title", ""), d.get("text", "")))
        tm.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)",
                   (d["doc_id"], d["published_at"], d.get("source_domain"),
                    d.get("url")))
        n += 1
    tm.execute("CREATE INDEX idx_published_at ON docs(published_at)")
    bm.commit(); tm.commit(); bm.close(); tm.close()
    return {"indexed": n,
            "bm25_bytes": bm25_db.stat().st_size,
            "temporal_bytes": temporal_db.stat().st_size}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


#: The scan's fields, in the shape `report` already expects a document to have,
#: so the report can read a scan line or a real document without caring which.
_SCAN_FIELDS = ("published_at", "date_source", "date_confidence", "source_domain",
                "raw_date_rejected")


def scan_path(folder: Path) -> Path:
    return INTERIM / f"bible_scan_{folder.name}.jsonl"


def _scan_slim(d: dict[str, Any]) -> dict[str, Any]:
    prov = d.get("provenance") or {}
    out = {k: d.get(k) for k in _SCAN_FIELDS}
    out["provenance"] = {"source": prov.get("source"),
                         "claim_date": prov.get("claim_date")}
    return out


def scan_store(folder: Path, chunk: int | None = None) -> dict[str, Any]:
    """Extract the report's few fields into a compact JSONL, resumably.

    The report needs a date, a source, a confidence and a domain per document.
    Getting them means reading all 8.6 GB of the store, and on a 16 GB machine
    that fills the file cache until the OS reports a few hundred MB free -- this
    report was killed for memory three times while using 217 MB itself. Free
    memory recovers as soon as the process exits, so the fix is not to use less
    memory but to touch less of the store per process.

    So: an append-only JSONL of ~90 bytes per document, written `chunk` at a
    time. Each invocation exits cleanly and the next resumes where it stopped,
    and afterwards every report reads 120 MB instead of 8.6 GB.

    Resume position is the line count, which is why the walk is in sorted
    filename order and the file is opened for append. A run killed mid-write can
    leave a partial last line, so the file is truncated back to its last
    newline before anything is added.
    """
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = scan_path(folder)
    names = sorted(e.name for e in os.scandir(folder)
                   if e.name.endswith(".json")) if folder.is_dir() else []

    done = 0
    if out.is_file():
        data = out.read_bytes()
        cut = data.rfind(b"\n") + 1          # drop a half-written final line
        if cut != len(data):
            with open(out, "r+b") as fh:
                fh.truncate(cut)
            data = data[:cut]
        done = data.count(b"\n")
        del data
    if done > len(names):                    # the store shrank under the scan
        out.unlink()
        done = 0

    todo = names[done:]
    if chunk:
        todo = todo[:chunk]

    def _read(name: str) -> dict[str, Any] | None:
        try:
            return _scan_slim(json.loads((folder / name).read_bytes()))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None

    written = 0
    with open(out, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(_READ_THREADS) as pool:
            for i in range(0, len(todo), _READ_CHUNK):
                for rec in pool.map(_read, todo[i:i + _READ_CHUNK]):
                    fh.write(json.dumps(rec or {}, ensure_ascii=False) + "\n")
                    written += 1
                fh.flush()
    return {"folder": folder.name, "total": len(names), "scanned": done + written,
            "written_now": written, "done": done + written >= len(names),
            "path": str(out)}


def read_scan(folder: Path) -> Iterator[dict[str, Any]] | None:
    """The scan for `folder` if it is complete and current, else None.

    "Current" is a line count matching the file count. That is deliberately a
    weak check -- it catches a truncated or stale scan, not an edited document --
    and the scan is a derived artefact in `data/interim/`, so the answer to any
    doubt is to delete it and scan again.
    """
    path = scan_path(folder)
    if not path.is_file() or not folder.is_dir():
        return None
    files = sum(1 for e in os.scandir(folder) if e.name.endswith(".json"))
    with open(path, "rb") as fh:
        lines = sum(1 for _ in fh)
    if lines != files:
        return None

    def _iter() -> Iterator[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec:
                    yield rec

    return _iter()


def documents_for_report(folder: Path) -> Iterator[dict[str, Any]]:
    """The scan if there is a usable one, otherwise the store itself."""
    scanned = read_scan(folder)
    return scanned if scanned is not None else read_documents(folder,
                                                              strip_text=True)


def report(state: dict[str, Any]) -> dict[str, Any]:
    by_source: collections.Counter = collections.Counter()
    by_conf: collections.Counter = collections.Counter()
    by_domain: collections.Counter = collections.Counter()
    by_origin: collections.Counter = collections.Counter()
    # date provenance differs completely between sources; pooling it describes
    # none of them, so keep the cross-tab.
    src_by_origin: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    conf_by_origin: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    predate: collections.Counter = collections.Counter()
    by_year: collections.Counter = collections.Counter()
    # Running min/max rather than a list. The list held every published_at in
    # the store -- 1.37M strings, and growing with it -- to compute two values,
    # and a report is run on the same machine that is holding the store. It was
    # killed for memory once; that is a cheap thing to make impossible.
    date_min: str | None = None
    date_max: str | None = None
    n = 0
    for d in documents_for_report(DOCS):
        n += 1
        prov = d.get("provenance") or {}
        origin = prov.get("source", "?")
        src = d.get("date_source", "?")
        by_source[src] += 1
        by_conf[d.get("date_confidence", "?")] += 1
        by_domain[d.get("source_domain", "?")] += 1
        by_origin[origin] += 1
        src_by_origin[origin][src] += 1
        conf_by_origin[origin][d.get("date_confidence", "?")] += 1
        if d.get("published_at"):
            pub = d["published_at"]
            if date_min is None or pub < date_min:
                date_min = pub
            if date_max is None or pub > date_max:
                date_max = pub
            by_year[d["published_at"][:4]] += 1
            # Only enrich documents have a claim to be compared against.
            claim = prov.get("claim_date")
            if claim:
                predate["predates" if d["published_at"] <= claim
                        else "postdates"] += 1

    quarantined = collections.Counter()
    quarantined_by_origin: collections.Counter = collections.Counter()
    q = 0
    for d in documents_for_report(UNDATED):
        q += 1
        quarantined[d.get("date_source", "?")] += 1
        quarantined_by_origin[(d.get("provenance") or {}).get("source", "?")] += 1

    # `date_source` above covers the INDEXED half only, and `none` exists only in
    # the quarantined half, so neither answers "what dated this store". The two
    # sets are disjoint by construction -- a document is in docs/ or undated/,
    # never both -- so the sum is the whole store.
    whole_store = collections.Counter(by_source) + collections.Counter(quarantined)

    # Source A's denominator is the URLs it was ASKED for, not the documents it
    # got back: a yield reported over what survived fetching would divide by its
    # own success rate. enrich_targets() needs the raw corpora on disk, so a
    # missing tree reports null rather than a wrong number.
    enrich_yield: dict[str, Any] | None
    try:
        enrich_urls = len({u for _, _, u in enrich_targets()})
    except Exception as exc:            # raw corpora absent or unreadable
        enrich_yield = {"evidence_urls": None,
                        "unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        got_high = conf_by_origin["enrich"]["high"]
        enrich_yield = {
            "evidence_urls": enrich_urls,
            "indexed": by_origin["enrich"],
            "quarantined": quarantined_by_origin["enrich"],
            "high_confidence": got_high,
            "high_confidence_date_sources":
                {k: v for k, v in src_by_origin["enrich"].items()
                 if k in ("jsonld", "meta_published")},
            "high_confidence_rate_over_urls":
                round(got_high / enrich_urls, 4) if enrich_urls else None,
            "indexed_rate_over_urls":
                round(by_origin["enrich"] / enrich_urls, 4) if enrich_urls else None,
        }

    idx = {p.name: p.stat().st_size for p in INDEX.glob("*.sqlite")} if INDEX.is_dir() else {}
    out = {
        "generated_at": utcnow(),
        "html_parser": _PARSER,
        "documents_indexed": n,
        "documents_quarantined": q,
        "quarantine_rate": round(q / max(1, n + q), 4),
        "quarantined_by_reason": dict(quarantined.most_common()),
        "quarantined_by_origin": dict(quarantined_by_origin.most_common()),
        "date_source": dict(by_source.most_common()),
        "date_source_whole_store": dict(whole_store.most_common()),
        "date_confidence": dict(by_conf.most_common()),
        "by_origin": dict(by_origin.most_common()),
        "date_source_by_origin": {o: dict(c.most_common())
                                  for o, c in sorted(src_by_origin.items())},
        "date_confidence_by_origin": {o: dict(c.most_common())
                                      for o, c in sorted(conf_by_origin.items())},
        # Of enrich documents that carry a claim date: did the evidence exist
        # before the claim it was collected for? Documents that postdate their
        # own claim stay in the store -- they are valid evidence for other
        # claims -- but they can never be retrieved for this one.
        "enrich_vs_claim": dict(predate.most_common()),
        "enrich_predate_rate": (round(predate["predates"] / sum(predate.values()), 4)
                                if predate else None),
        "date_range": {"min": date_min, "max": date_max} if date_min else None,
        "plausibility_window": {"min": PLAUSIBLE_MIN, "max": today_utc()},
        # Left in the store on purpose, and reported so that "left alone" is a
        # visible decision rather than an omission. See JALALI_YEARS.
        "non_gregorian_calendar": {
            y: c for y, c in sorted(by_year.items())
            if _in_band(int(y), JALALI_YEARS) or _in_band(int(y), BUDDHIST_YEARS)},
        "by_year": dict(sorted(by_year.items())),
        "enrich_yield": enrich_yield,
        "domains": {"distinct": len(by_domain),
                    "top": dict(by_domain.most_common(25))},
        "index_bytes": idx,
        "store_bytes": store_bytes(),
        "store_gb": round(store_bytes() / GB, 4),
        "transfer_bytes": state.get("transfer_bytes", 0),
        "transfer_gb": round(state.get("transfer_bytes", 0) / GB, 4),
        "caps": {"store_gb": STORE_CAP_GB, "transfer_gb": TRANSFER_CAP_GB},
        "excluded": {
            "mocheg": "43,148 evidence URLs (78% of all bundled evidence) cannot "
                      "enter the store: the corpus ships no claim-date field, so "
                      "nothing in it can be placed on a timeline relative to its "
                      "claim. A property of the dataset, not of this pipeline.",
            "http_last_modified": "recorded as provenance but never used as a "
                                  "publication date; such documents are quarantined.",
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "bible.json").write_text(json.dumps(out, indent=2) + "\n",
                                        encoding="utf-8")
    return out


def render_report(r: dict[str, Any]) -> str:
    lines = [
        f"  documents indexed    : {r['documents_indexed']:,}",
        f"  quarantined (undated): {r['documents_quarantined']:,} "
        f"({r['quarantine_rate']:.1%})",
        f"  date source (indexed): {r['date_source']}",
        f"  date source (store)  : {r['date_source_whole_store']}",
        f"  date confidence      : {r['date_confidence']}",
        f"  by origin  (indexed) : {r['by_origin']}",
        f"  quarantine by origin : {r['quarantined_by_origin']}",
    ]
    ey = r.get("enrich_yield") or {}
    if ey.get("evidence_urls"):
        lines.append(
            f"  enrich yield         : {ey['high_confidence']:,} high-confidence "
            f"of {ey['evidence_urls']:,} URLs "
            f"({ey['high_confidence_rate_over_urls']:.1%}); "
            f"{ey['indexed']:,} indexed ({ey['indexed_rate_over_urls']:.1%})")
    if r["enrich_predate_rate"] is not None:
        lines.append(f"  enrich vs its claim  : {r['enrich_vs_claim']} "
                     f"-> {r['enrich_predate_rate']:.1%} predate")
    if r["date_range"]:
        lines.append(f"  date range           : {r['date_range']['min']} .. "
                     f"{r['date_range']['max']}")
        lines.append("  by year              : "
                     + "  ".join(f"{y}:{c:,}" for y, c in r["by_year"].items()))
    lines += [
        f"  distinct domains     : {r['domains']['distinct']:,}",
        f"  index                : {r['index_bytes']}",
        f"  store                : {r['store_gb']:.3f} / {r['caps']['store_gb']:.0f} GB",
        f"  transfer             : {r['transfer_gb']:.3f} / "
        f"{r['caps']['transfer_gb']:.0f} GB",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=("wikinews", "enrich", "ccnews"))
    p.add_argument("--index", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--scan", action="store_true",
                   help="extract the report's fields to data/interim/ in "
                        "resumable chunks; run until it says done")
    p.add_argument("--scan-chunk", type=int, default=250_000,
                   help="documents per --scan invocation (default 250,000)")
    p.add_argument("--requarantine", action="store_true",
                   help="re-apply the plausibility window to the existing "
                        "store; move implausibly-dated documents to undated/")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--warc-limit", type=int, default=DEFAULT_WARC_LIMIT)
    p.add_argument("--extractor", choices=("fast", "bs4"), default="fast",
                   help="ccnews text extraction: fast regex (14x, default) or bs4")
    p.add_argument("--limit", type=int, default=None,
                   help="enrich: cap the number of evidence URLs")
    p.add_argument("--store-cap", type=float, default=STORE_CAP_GB)
    p.add_argument("--transfer-cap", type=float, default=TRANSFER_CAP_GB)
    args = p.parse_args(argv)

    if not (args.source or args.index or args.report or args.requarantine
            or args.scan):
        p.error("nothing to do: pass --source, --requarantine, --scan, "
                "--index or --report")

    BIBLE.mkdir(parents=True, exist_ok=True)
    state = load_state()
    budget = Budget(state, args.store_cap, args.transfer_cap)

    try:
        if args.source:
            print(f"\n[bible] source={args.source}"
                  f"{'  (dry run)' if args.dry_run else ''}")
            print(budget.render())
            builders = {"wikinews": lambda: build_wikinews(budget, state, args.dry_run),
                        "enrich": lambda: build_enrich(budget, state, args.dry_run,
                                                       args.limit),
                        "ccnews": lambda: build_ccnews(budget, state, args.dry_run,
                                                       args.warc_limit,
                                                       extractor=args.extractor)}
            stats = builders[args.source]()
            if not args.dry_run:      # a dry run has no side effects, state included
                save_state(state)
            print(f"  {stats}")
            print(budget.render())

        if args.requarantine:
            print("\n[bible] re-applying the plausibility window"
                  f"{'  (dry run)' if args.dry_run else ''}")
            rq = requarantine(dry_run=args.dry_run)
            print(f"  window {rq['window']['min']} .. {rq['window']['max']}")
            print(f"  scanned {rq.get('scanned', 0):,}, "
                  f"quarantined {rq.get('quarantined', 0):,}, "
                  f"non-Gregorian kept {rq.get('non_gregorian_kept', 0):,}")
            print(f"  quarantined by year: {rq['quarantined_by_year']}")
            print(f"  non-Gregorian by year: {rq['non_gregorian_by_year']}")
            if rq.get("quarantined") and not args.dry_run:
                print("  the index is now stale -- rebuild it with --index")

        if args.scan:
            print("\n[bible] scanning the store for the report")
            all_done = True
            for folder in (DOCS, UNDATED):
                s = scan_store(folder, chunk=args.scan_chunk)
                print(f"  {s['folder']:<9} {s['scanned']:>9,} / {s['total']:,}"
                      f"  (+{s['written_now']:,})"
                      f"  {'done' if s['done'] else 'MORE TO DO'}")
                all_done &= s["done"]
            if not all_done:
                print("  not finished -- run --scan again; it resumes where it "
                      "stopped")
                return 3

        if args.index:
            print("\n[bible] building index")
            print(f"  {build_index()}")

        if args.report:
            print("\n[bible] report")
            r = report(state)
            print(render_report(r))
            print(f"  report -> {(REPORTS/'bible.json').relative_to(PROJECT_ROOT)}")
    except BudgetError as exc:
        # A refusal still records what was spent before it -- except in a dry
        # run, which spent nothing and must leave no trace.
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        if not args.dry_run:
            save_state(state)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
