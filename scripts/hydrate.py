#!/usr/bin/env python
"""Fetch the URL-referenced media that the corpora only ship as links.

    python scripts/hydrate.py --dataset fakeddit --dry-run
    python scripts/hydrate.py --dataset factify2 --concurrency 16
    python scripts/hydrate.py --all

Three datasets ship indices rather than content: Factify2's images, Fakeddit's
sampled images, and MOCHEG's ID-only tweets. This script resolves them, with a
concurrency cap, exponential backoff, per-URL timeouts, and no retry on a 404 —
a dead URL does not become live by being asked again, and hammering 150,000 of
them is how a hydration run takes six hours to fail.

Every run writes ``data/reports/hydration_<dataset>.json`` containing an
overall recovery rate **and a per-class breakdown**. The per-class figure is the
point: if loss concentrates in one label the class balance is silently broken,
and an overall 85% hides a class that recovered 20%. If the label column cannot
be located this script aborts rather than writing a report without it.

Re-running is safe: anything already on disk is skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT, REPORTS, raw_dir  # noqa: E402
from scripts import manifest  # noqa: E402
from scripts.fetchlib import (  # noqa: E402
    DEFAULT_POLICY,
    ORIGIN,
    SOURCE_DIRS,
    UNBLOCK_POLICY,
    WAYBACK,
    FailureLedger,
    FetchError,
    HostCircuitBreaker,
    HostThrottle,
    HydrationReport,
    ProvenanceLedger,
    RetryPolicy,
    backoff_delays,
    ChainedThrottle,
    completion_status,
    GlobalThrottle,
    log_event,
    make_session,
    read_marker,
    unblock_headers,
    wayback_snapshot,
    write_marker,
)

HYDRATABLE = ("factify2", "fakeddit", "mocheg", "verite", "m4fc")

#: Tried in order; the first column present wins. If none is present the
#: resolver aborts and prints what the file actually contains — guessing a
#: label column would produce a per-class breakdown that is quietly wrong.
LABEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "verite": ("label",),
    "fakeddit": ("label", "6_way_label", "3_way_label", "2_way_label"),
    "factify2": ("Category", "category", "class", "label", "Label"),
    "mocheg": ("cleaned_truthfulness", "truthfulness", "label", "Label"),
    "m4fc": ("verdict_coarse",),
}

URL_COLUMNS: dict[str, tuple[str, ...]] = {
    "fakeddit": ("image_url",),
    "factify2": ("claim_image", "document_image", "claim_image_url",
                 "document_image_url", "claim_img", "document_img"),
}

TWEET_ID_COLUMNS = ("tweet_id", "tweetid", "id", "twitter_id")

#: Public syndication endpoint. Expected to have a low success rate since the
#: API closed in 2023 — that loss is measured and reported, not hidden.
TWEET_ENDPOINT = "https://cdn.syndication.twimg.com/tweet-result"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

#: Datasets whose media are images, so a 200 body that does not decode is a
#: failed fetch rather than a file. mocheg is absent on purpose: it fetches
#: tweet JSON, which would never decode as an image.
IMAGE_DATASETS = frozenset({"verite", "factify2", "fakeddit", "m4fc"})


@dataclass(frozen=True)
class Item:
    """One thing to fetch: a stable key, a URL, and the class it belongs to.

    ``record`` names the dataset ROW this item belongs to. Factify2 puts two
    images on every row (claim + document), so a complete-cases design keeps a
    row only when both arrived -- and that cannot be counted without knowing
    which items share a row. Defaults to the key for one-image-per-row corpora.
    """

    key: str
    url: str
    label: str
    record: str = ""

    @property
    def record_id(self) -> str:
        return self.record or self.key


# --------------------------------------------------------------------------
# column resolution — abort rather than guess
# --------------------------------------------------------------------------


def pick_column(name: str, header: Sequence[str], candidates: Sequence[str],
                kind: str, source: Path) -> str:
    for candidate in candidates:
        if candidate in header:
            return candidate
    raise FetchError(
        f"{name}: could not find a {kind} column in {source.name}.\n"
        f"  tried: {list(candidates)}\n"
        f"  found: {list(header)}\n"
        f"  Refusing to continue: without the label column the per-class "
        f"recovery breakdown is impossible, and a hydration report without it "
        f"would hide exactly the failure it exists to surface."
    )


#: MOCHEG stores whole scraped articles in single cells, well past Python's
#: 128 KB default. Raised once at import rather than per-read.
csv.field_size_limit(64 * 1024 * 1024)


def sniff_delimiter(path: Path) -> str:
    """Detect the real delimiter from the header line.

    Factify2 ships tab-separated data in files named .csv, so trusting the
    extension parses the whole header as one column and every label lookup
    fails. Counting separators in the header is cruder than csv.Sniffer and far
    harder to fool on quoted free text.
    """
    with path.open(encoding="utf-8", newline="", errors="replace") as handle:
        header = handle.readline()
    return "	" if header.count("	") > header.count(",") else ","


def read_rows(path: Path, delimiter: str | None = None) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = delimiter or sniff_delimiter(path)
    with path.open(encoding="utf-8", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = [h for h in (reader.fieldnames or []) if h]
        return header, list(reader)


# --------------------------------------------------------------------------
# per-dataset resolvers
# --------------------------------------------------------------------------


def resolve_fakeddit(seed: int | None = None) -> tuple[list[Item], str, dict[str, Any]]:
    """Items come from the sample index fetch.py wrote, never the full TSV."""
    sample_dir = INTERIM / "fakeddit"
    samples = sorted(sample_dir.glob("sample_seed*_n*.csv"))
    if not samples:
        raise FetchError(
            "fakeddit: no sample index found in data/interim/fakeddit/. "
            "Run `python scripts/fetch.py --dataset fakeddit` first — hydrating "
            "the full ~1M image set would blow the disk budget."
        )
    chosen = samples[-1]
    if seed is not None:
        match = [p for p in samples if f"seed{seed}_" in p.name]
        if not match:
            raise FetchError(
                f"fakeddit: no sample index for seed {seed}; have "
                f"{[p.name for p in samples]}"
            )
        chosen = match[-1]

    header, rows = read_rows(chosen)
    label_col = pick_column("fakeddit", header, LABEL_COLUMNS["fakeddit"], "label", chosen)
    items = [
        Item(key=row.get("id") or f"row{i}", url=row["image_url"], label=row[label_col])
        for i, row in enumerate(rows)
        if row.get("image_url")
    ]
    parsed_seed = chosen.stem.split("seed")[1].split("_")[0]
    return items, label_col, {"sample_index": chosen.name, "seed": int(parsed_seed)}


def resolve_factify2() -> tuple[list[Item], str, dict[str, Any]]:
    """Both image columns per row; each becomes its own fetchable item."""
    root = raw_dir("factify2")
    csvs = [p for p in sorted(root.rglob("*.csv")) if "test" not in p.name.lower()]
    if not csvs:
        raise FetchError(
            "factify2: no train/val CSVs under data/raw/factify2/. "
            "Run `python scripts/fetch.py --dataset factify2` first."
        )

    items: list[Item] = []
    label_col = ""
    for path in csvs:
        header, rows = read_rows(path)
        label_col = pick_column("factify2", header, LABEL_COLUMNS["factify2"],
                                "label", path)
        url_cols = [c for c in URL_COLUMNS["factify2"] if c in header]
        if not url_cols:
            raise FetchError(
                f"factify2: no image-URL column in {path.name}.\n"
                f"  tried: {list(URL_COLUMNS['factify2'])}\n  found: {header}"
            )
        for index, row in enumerate(rows):
            for column in url_cols:
                url = (row.get(column) or "").strip()
                if url.startswith("http"):
                    row_id = f"{path.stem}_{row.get('Id') or row.get('id') or index}"
                    items.append(Item(key=f"{row_id}_{column}", url=url,
                                      label=row[label_col], record=row_id))
    return items, label_col, {"csv_files": [p.name for p in csvs]}


def resolve_verite() -> tuple[list[Item], str, dict[str, Any]]:
    """VERITE ships captions and labels but no images; fetch them from source.

    The images are addressed by VERITE.csv as ``images/true_N.jpg`` and
    ``images/false_N.jpg``, and the URLs live in VERITE_articles.csv keyed by
    the same N: ``true_url`` for the true image, ``false_url`` for the
    out-of-context one.

    Items are emitted **per row**, not per image, so per-class recovery means
    "rows whose image arrived". That matters here because one image serves two
    classes: `true` and `miscaptioned` share the same ``true_N.jpg``, differing
    only in caption. Their recovery rates are therefore identical by
    construction, not by coincidence.
    """
    root = raw_dir("verite")
    csv_path = next(root.rglob("VERITE.csv"), None)
    articles_path = next(root.rglob("VERITE_articles.csv"), None)
    if csv_path is None or articles_path is None:
        raise FetchError(
            "verite: VERITE.csv and VERITE_articles.csv are both required and "
            "at least one is missing. Run `python scripts/fetch.py "
            "--dataset verite` first."
        )

    _ah, articles = read_rows(articles_path)
    urls: dict[str, str] = {}
    for row in articles:
        ident = str(row.get("id", "")).strip()
        if not ident:
            continue
        for column, stem in (("true_url", "true"), ("false_url", "false")):
            url = (row.get(column) or "").strip()
            if url.startswith("http"):
                urls[f"images/{stem}_{ident}.jpg"] = url

    header, rows = read_rows(csv_path)
    label_col = pick_column("verite", header, LABEL_COLUMNS["verite"],
                            "label", csv_path)
    items: list[Item] = []
    unmapped = 0
    for index, row in enumerate(rows):
        path = (row.get("image_path") or "").strip()
        url = urls.get(path)
        if not url:
            unmapped += 1
            continue
        items.append(Item(key=path, url=url, label=row[label_col],
                          record=path))
    if not items:
        raise FetchError(
            "verite: no image_path in VERITE.csv could be matched to a URL in "
            "VERITE_articles.csv. The two files may have diverged."
        )
    return items, label_col, {
        "rows": len(rows),
        "distinct_images": len(set(i.key for i in items)),
        "rows_without_a_url": unmapped,
        "note": "true and miscaptioned share one image per id, so their "
                "recovery rates are identical by construction",
    }


def resolve_mocheg() -> tuple[list[Item], str, dict[str, Any]]:
    """MOCHEG's ID-only tweets. Best-effort: the free API closed in 2023."""
    root = raw_dir("mocheg")
    candidates = [p for p in sorted(root.rglob("*.csv"))]
    for path in candidates:
        header, rows = read_rows(path)
        id_col = next((c for c in TWEET_ID_COLUMNS if c in header), None)
        if not id_col:
            continue
        label_col = pick_column("mocheg", header, LABEL_COLUMNS["mocheg"], "label", path)
        items = [
            Item(
                key=str(row[id_col]).strip(),
                url=f"{TWEET_ENDPOINT}?id={str(row[id_col]).strip()}&lang=en",
                label=row[label_col],
            )
            for row in rows
            if str(row.get(id_col) or "").strip().isdigit()
        ]
        if items:
            return items, label_col, {"source_csv": path.name, "id_column": id_col}

    raise FetchError(
        "mocheg: no CSV under data/raw/mocheg/ carries a tweet-id column "
        f"({list(TWEET_ID_COLUMNS)}). Run `python scripts/fetch.py --dataset "
        "mocheg` first, or point this resolver at the right file."
    )


def resolve_m4fc() -> tuple[list[Item], str, dict[str, Any]]:
    """M4FC: one image per entry, withheld by the authors and published as a URL.

    The key is the dataset's own ``image_path`` (unique across all 4,982
    entries); the label is ``verdict_coarse``, the field the authors use for
    verdict prediction. ``split`` is carried in the record id's metadata by
    the processed build, not here.
    """
    import json

    path = raw_dir("m4fc") / "data" / "M4FC.json"
    if not path.is_file():
        raise FetchError("m4fc: data/M4FC.json is missing. Run "
                         "`python scripts/fetch.py --dataset m4fc` first.")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows or "verdict_coarse" not in rows[0]:
        raise FetchError(f"m4fc: no verdict_coarse label in {path.name}; found "
                         f"{sorted(rows[0]) if rows else []}")
    items = [Item(key=r["image_path"], url=r["image_url"], label=r["verdict_coarse"],
                  record=r["image_path"])
             for r in rows if str(r.get("image_url") or "").startswith("http")]
    return items, "verdict_coarse", {"entries": len(rows),
                                     "with_url": len(items)}


RESOLVERS = {
    "verite": resolve_verite,
    "fakeddit": resolve_fakeddit,
    "factify2": resolve_factify2,
    "mocheg": resolve_mocheg,
    "m4fc": resolve_m4fc,
}


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def _filename(item: Item) -> str:
    suffix = Path(item.url.split("?")[0]).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in item.key)
    return f"{safe}{suffix}"


def destination(name: str, item: Item, source: str = ORIGIN) -> Path:
    """Where a media file lives, keyed by WHERE IT CAME FROM.

    Origin and archived copies go to different directories on purpose. A
    Wayback snapshot may differ from what the dataset authors fetched, so the
    integrity manifest must distinguish them by path -- not only in a sidecar
    ledger that downstream code might never read.
    """
    root = raw_dir(name) / "media"
    if name == "mocheg":
        return root / "tweets" / f"{item.key}.json"
    folder = SOURCE_DIRS[source]
    if name == "verite":
        # VERITE's shipped CSVs address images by a path relative to the CSV
        # itself ("images/<name>.jpg"), so they land where the corpus expects
        # rather than under media/ -- otherwise every file we fetch is invisible
        # to the dataset's own index.
        rel = item.key
        if source != ORIGIN:
            rel = rel.replace("images/", f"{folder}/", 1)
        return raw_dir(name) / "data" / "VERITE" / rel
    return root / folder / _filename(item)


def existing_destination(name: str, item: Item) -> Path | None:
    """The file for this item from either source, or None if absent."""
    for source in (ORIGIN, WAYBACK):
        path = destination(name, item, source)
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def provenance_ledger(name: str) -> ProvenanceLedger:
    return ProvenanceLedger(raw_dir(name) / "media" / "provenance.jsonl")


def failure_ledger(name: str) -> FailureLedger:
    return FailureLedger(REPORTS / f"hydration_{name}_failures.jsonl")


def not_a_placeholder(body: bytes) -> str | None:
    """None unless these exact bytes are a known evidence-free image.

    A SEPARATE gate from looks_like_an_image, because these decode perfectly.
    imgur's "this image does not exist" bitmap is a valid 161x81 PNG; 2,899
    copies of it were counted as recovered fakeddit images, all in label 4.
    Decodability cannot catch that. Only content identity can.

    Reported as a retryable failure: a host that served a placeholder once may
    serve the real image on the next request, and a permanent one simply fails
    the retry budget and is recorded as missing -- which it is.
    """
    from scripts.placeholders import describe, is_missing

    digest = hashlib.sha256(body).hexdigest()
    if is_missing(digest):
        return f"placeholder_image[{describe(digest).split(':')[0]}]"
    return None


def looks_like_an_image(body: bytes) -> str | None:
    """None if these bytes decode as an image, else a failure reason.

    A blocked host does not always say 403. Several of VERITE's sources answer
    200 with an HTML error page, and the fetcher happily wrote one as
    one VERITE image -- 195 bytes reading "Error. Page cannot be displayed".
    Five more were already on disk from an earlier pass, counted as recovered
    images for months. Accepting any non-empty 200 body means the recovery rate
    measures "hosts that answered", not "images we have", and it silently
    substitutes junk for data.

    Checked by decoding, not by sniffing Content-Type: the header is what the
    server claims, the pixels are what we got.
    """
    if not body:
        return "empty_body"
    from PIL import Image, UnidentifiedImageError  # lazy: not every dataset is images

    try:
        Image.open(io.BytesIO(body)).verify()
    except UnidentifiedImageError:
        return "not_an_image"
    except Exception as exc:
        return f"undecodable_image_{type(exc).__name__.lower()}"
    return None


def fetch_url(session, url: str, dest: Path, timeout: float, retries: int,
              rng: random.Random, *, policy: RetryPolicy = DEFAULT_POLICY,
              headers: dict[str, str] | None = None,
              throttle: HostThrottle | None = None,
              validator: Callable[[bytes], str | None] | None = None
              ) -> tuple[bool, str | None, int, int | None]:
    """(succeeded, failure_reason, bytes, last_http_status).

    The status is returned separately from the reason because a transport-level
    failure has no status at all -- a connection reset and an HTTP 503 are
    different events, and a log that renders both as a string cannot be counted
    by status later. ``None`` means the request never got as far as a response.

    Never retries a terminal status.
    """
    delays = backoff_delays(retries, rng=rng)
    reason = "unknown"
    status: int | None = None
    for attempt in range(retries + 1):
        if throttle is not None:
            throttle.wait(url)
        try:
            response = session.get(url, timeout=timeout, stream=True, headers=headers)
            status = response.status_code
            if status == 200:
                body = response.content
                rejected = validator(body) if validator is not None else (
                    "empty_body" if not body else None)
                if rejected:
                    # A 200 carrying the wrong thing is a failure, not a file.
                    # Retry it: some hosts serve an interstitial once and the
                    # real image on the next request.
                    reason = rejected
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.write_bytes(body)
                    tmp.replace(dest)
                    return True, None, len(body), status
            else:
                reason = f"http_{status}"
                if not policy.should_retry(status):
                    # 404 and friends: stop immediately
                    return False, reason, 0, status
        except Exception as exc:
            reason = type(exc).__name__.lower()
            status = None

        if attempt < retries:
            time.sleep(delays[attempt])
    return False, reason, 0, status


def fetch_item(session, item: Item, dest: Path, timeout: float, retries: int,
               rng: random.Random, **kwargs) -> tuple[bool, str | None]:
    """Backwards-compatible wrapper: skip what already exists, then fetch."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, None
    ok, reason, _bytes, _status = fetch_url(session, item.url, dest, timeout,
                                            retries, rng, **kwargs)
    return ok, reason


def fetch_via_wayback(session, item: Item, dest: Path, timeout: float,
                      retries: int, rng: random.Random,
                      throttle: HostThrottle | None = None,
                      validator: Callable[[bytes], str | None] | None = None
                      ) -> tuple[bool, str | None, dict[str, Any]]:
    """Fetch an archived copy. Returns (ok, reason, provenance_extra)."""
    if throttle is not None:
        throttle.wait("https://archive.org/")
    snapshot = wayback_snapshot(session, item.url, timeout=timeout)
    if snapshot is None:
        return False, "wayback_no_snapshot", {}
    timestamp, raw_url = snapshot
    ok, reason, _bytes, status = fetch_url(session, raw_url, dest, timeout,
                                           retries, rng, policy=UNBLOCK_POLICY,
                                           throttle=throttle, validator=validator)
    extra = {"wayback_timestamp": timestamp, "wayback_url": raw_url,
             "http_status": status}
    return ok, (None if ok else f"wayback_{reason}"), extra


def backfill_origin_provenance(name: str, items: list[Item]) -> int:
    """Record provenance for origin files fetched before the ledger existed.

    Without this the ledger would imply the 77,948 images already on disk have
    unknown provenance, when in fact they are all origin fetches. Cheap: no
    hashing, since data/MANIFEST.sha256 already carries the digests.
    """
    ledger = provenance_ledger(name)
    known = {entry.get("key") for entry in ledger.read()}
    added = 0
    for item in items:
        if item.key in known:
            continue
        path = destination(name, item, ORIGIN)
        if path.exists() and path.stat().st_size > 0:
            ledger.append(key=item.key, label=item.label, url=item.url,
                          source=ORIGIN, bytes=path.stat().st_size,
                          path=path.relative_to(raw_dir(name)).as_posix())
            added += 1
    return added


def cumulative_report(name: str, items: list[Item], label_col: str,
                      extra: dict[str, Any], **fields: Any) -> HydrationReport:
    """Recovery over EVERY item, counted from what is on disk.

    A retry pass touches only the failures, so scoring just that pass would
    report a recovery rate for a subset and call it the dataset's. The honest
    number is: of all N items, how many have a file now.
    """
    report = HydrationReport(dataset=name, label_field=label_col,
                             seed=extra.get("seed"), **fields)
    ledger_paths = {e.get("key"): e.get("source") for e in provenance_ledger(name).read()}
    for item in items:
        found = existing_destination(name, item)
        report.record(item.label, found is not None, "still_missing")
        if found is not None:
            src = ledger_paths.get(item.key, ORIGIN)
            report.per_source[src] = report.per_source.get(src, 0) + 1
    return report


def hydrate(name: str, concurrency: int, timeout: float, retries: int,
            seed: int | None, dry_run: bool, *, retry_failed: bool = False,
            use_wayback: bool = False, host_delay: float = 0.0,
            global_delay: float = 0.0,
            breaker_threshold: int = 0) -> HydrationReport:
    resolver = RESOLVERS[name]
    items, label_col, extra = resolver(seed) if name == "fakeddit" else resolver()

    mode = "wayback" if use_wayback else ("retry" if retry_failed else "first-pass")
    failures = failure_ledger(name)
    provenance = provenance_ledger(name)

    print(f"\n[{name}] {len(items)} item(s), label column {label_col!r}  mode={mode}")
    for key, value in extra.items():
        print(f"  {key}: {value}")

    if retry_failed or use_wayback:
        backfilled = backfill_origin_provenance(name, items)
        if backfilled:
            print(f"  provenance backfilled for {backfilled} existing origin file(s)")

    missing = [it for it in items if existing_destination(name, it) is None]
    print(f"  on disk: {len(items) - len(missing)}   missing: {len(missing)}")

    if retry_failed or use_wayback:
        dead = failures.terminal_keys()
        pending_items = [it for it in missing if it.key not in dead]
        if dead:
            print(f"  skipping {len(missing) - len(pending_items)} known 404/410 "
                  "(a deleted resource does not return)")
    else:
        pending_items = missing

    # One media file can serve several rows -- VERITE's true_N.jpg backs both a
    # `true` row and a `miscaptioned` one -- and fetch_url does not check for an
    # existing file, so an un-deduplicated list asks each host twice for the
    # same bytes. At one request every five seconds that is not a rounding
    # error. Per-class rates are unaffected: cumulative_report counts every
    # item from what is on disk, not from what this pass attempted.
    seen: set[str] = set()
    deduped = []
    for item in pending_items:
        if item.key in seen:
            continue
        seen.add(item.key)
        deduped.append(item)
    if len(deduped) != len(pending_items):
        print(f"  {len(pending_items)} pending row(s) -> {len(deduped)} distinct "
              "file(s); shared media fetched once")
    pending_items = deduped

    if dry_run:
        counts: dict[str, int] = {}
        for item in pending_items:
            counts[item.label] = counts.get(item.label, 0) + 1
        print(f"  would attempt {len(pending_items)}; per class: "
              f"{dict(sorted(counts.items()))}")
        print("  --dry-run: nothing fetched")
        return cumulative_report(name, items, label_col, extra,
                                 concurrency=concurrency,
                                 notes=f"dry run ({mode}); nothing fetched")

    if not pending_items:
        print("  nothing to do")
        report = cumulative_report(name, items, label_col, extra,
                                   concurrency=concurrency, notes=f"{mode}: no work")
        print(report.render())
        return report

    # Global first, then per-host: the per-host wait then happens INSIDE the
    # slot the global limiter granted, instead of stacking a second full delay.
    throttle = ChainedThrottle(
        GlobalThrottle(global_delay) if global_delay > 0 else None,
        HostThrottle(host_delay) if host_delay > 0 else None,
    )
    if not throttle.throttles:
        throttle = None
    breaker = HostCircuitBreaker(breaker_threshold) if breaker_threshold else None
    def _image_gate(body: bytes) -> str | None:
        # decodability first (cheap, and a non-image has no meaningful hash),
        # then content identity.
        return looks_like_an_image(body) or not_a_placeholder(body)

    validator = _image_gate if name in IMAGE_DATASETS else None
    policy = UNBLOCK_POLICY if (retry_failed or use_wayback) else DEFAULT_POLICY
    session = make_session(pool_size=max(concurrency, 1))
    rng = random.Random(0xC0FFEE)
    started = time.time()

    attempted = succeeded = 0
    round_failures: list[dict[str, Any]] = []

    def work(item: Item) -> tuple[Item, bool, str | None, dict[str, Any]]:
        if breaker is not None and not use_wayback and breaker.is_open(item.url):
            return item, False, "host_circuit_open", {"source": ORIGIN, "dest": None,
                                                      "http_status": None}
        if use_wayback:
            dest = destination(name, item, WAYBACK)
            ok, reason, meta = fetch_via_wayback(session, item, dest, timeout,
                                                 retries, rng, throttle, validator)
            return item, ok, reason, {**meta, "source": WAYBACK, "dest": dest}
        dest = destination(name, item, ORIGIN)
        headers = unblock_headers(item.url) if retry_failed else None
        ok, reason, _n, status = fetch_url(session, item.url, dest, timeout, retries,
                                           rng, policy=policy, headers=headers,
                                           throttle=throttle, validator=validator)
        return item, ok, reason, {"source": ORIGIN, "dest": dest,
                                  "http_status": status}

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        futures = [pool.submit(work, item) for item in pending_items]
        for future in as_completed(futures):
            try:
                item, ok, reason, meta = future.result()
            except Exception as exc:
                attempted += 1
                continue
            attempted += 1
            if breaker is not None and reason != "host_circuit_open":
                breaker.record(item.url, ok)
            # Every attempt is appended as it happens, successes and failures
            # alike. A ledger of successes only cannot answer "did we ask, and
            # what did the host say" -- which is the whole question when a
            # quarter of a corpus is missing. ProvenanceLedger.append flushes
            # per line, so an interrupted run keeps everything it learned.
            dest = meta.pop("dest", None)
            rest = {k: v for k, v in meta.items() if k != "dest"}
            if ok:
                succeeded += 1
                provenance.append(key=item.key, label=item.label, url=item.url,
                                  outcome="ok", attempt_reason=None,
                                  bytes=dest.stat().st_size,
                                  path=dest.relative_to(raw_dir(name)).as_posix(),
                                  **rest)
            else:
                provenance.append(key=item.key, label=item.label, url=item.url,
                                  outcome="failed", attempt_reason=reason,
                                  bytes=0, path=None, **rest)
                round_failures.append({"key": item.key, "label": item.label,
                                       "url": item.url, "reason": reason,
                                       "http_status": meta.get("http_status")})
            if attempted % 200 == 0 or attempted == len(pending_items):
                rate = succeeded / attempted if attempted else 0.0
                print(f"  {attempted}/{len(pending_items)}  this pass {rate:.1%}",
                      flush=True)

    duration = round(time.time() - started, 2)
    # Merge rather than overwrite: FailureLedger.write() opens "w", so writing
    # only this pass would erase what earlier passes established -- including
    # the 404/410s that terminal_keys() relies on to stop asking.
    merged = {r["key"]: r for r in failures.read() if "key" in r}
    merged.update({r["key"]: r for r in round_failures})
    recovered_now = {it.key for it in pending_items
                     if existing_destination(name, it) is not None}
    failures.write([r for k, r in merged.items() if k not in recovered_now])
    if breaker is not None and breaker.blocked_hosts():
        print("  hosts written off after consecutive refusals: "
              f"{breaker.blocked_hosts()}")

    report = cumulative_report(name, items, label_col, extra, concurrency=concurrency)
    report.notes = (
        f"{mode} pass: attempted {attempted}, recovered {succeeded}. "
        f"Rates below are cumulative over all {len(items)} items."
    )
    if name == "mocheg":
        report.notes += (
            " Tweet hydration is best-effort: X/Twitter closed free API access "
            "in 2023, so low recovery is expected and is measured, not assumed."
        )
    # Replace the placeholder "still_missing" tally with the real reasons this
    # pass observed, so the histogram says WHY rather than merely that.
    reasons: dict[str, int] = {}
    for record in round_failures:
        key = record["reason"] or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    unexplained = report.failed - len(round_failures)
    if unexplained > 0:
        reasons["not_attempted_this_pass"] = unexplained
    report.failure_reasons = dict(sorted(reasons.items()))

    print(report.render())
    if getattr(report, "per_source", None):
        print(f"  by provenance: {report.per_source}")
    path = report.write()
    print(f"  report -> {path.relative_to(PROJECT_ROOT)}")
    print(f"  failures -> {failures.path.relative_to(PROJECT_ROOT)} "
          f"({len(round_failures)})")

    log_event(dataset=name, method=f"hydrate:{mode}", url=f"<{len(pending_items)} urls>",
              bytes=None, sha256=None, duration_s=duration, status="ok",
              pass_attempted=attempted, pass_succeeded=succeeded,
              cumulative_recovery=report.recovery_rate,
              recovery_rate_per_class=report.recovery_rate_per_class)

    manifest.update([name])
    refresh_fetch_marker(name)
    print("  manifest updated")
    return report


def refresh_fetch_marker(name: str) -> None:
    """Re-stamp the fetch .done marker after hydration changes the tree.

    Hydration adds media under data/raw/<name>/, so the tree hash recorded at
    fetch time no longer matches. Without this, the next `fetch.py --all` sees
    a content-hash MISMATCH and re-downloads a dataset that is perfectly
    intact -- every single time anything is hydrated.
    """
    marker = read_marker(name)
    if marker is None:
        return
    write_marker(name, raw_dir(name), **(marker.get("stats") or {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", action="append", dest="datasets", metavar="NAME",
                        choices=HYDRATABLE, help="hydrate this dataset (repeatable)")
    target.add_argument("--all", action="store_true",
                        help="hydrate every dataset that ships URL-only media")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="parallel requests (default 16)")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="per-URL timeout in seconds (default 20)")
    parser.add_argument("--retries", type=int, default=4,
                        help="retries per URL; 404s are never retried (default 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="fakeddit: pick the sample index for this seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and count items without fetching")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-attempt only what is still missing, using browser "
                             "headers and a per-host delay; retries 403/406/429/5xx")
    parser.add_argument("--wayback", action="store_true",
                        help="fetch still-missing items from web.archive.org "
                             "instead of the origin (implies --retry-failed)")
    parser.add_argument("--breaker", type=int, default=25,
                        help="stop attempting a host after this many consecutive "
                             "failures during a retry pass (0 disables)")
    parser.add_argument("--host-delay", type=float, default=None,
                        help="minimum seconds between requests to the SAME host "
                             "(default 0 normally, 0.5 on retry, 1.0 on wayback)")
    parser.add_argument("--global-delay", type=float, default=0.0,
                        help="minimum seconds between ANY two requests, across "
                             "every host (default 0 = off). --host-delay alone "
                             "does not rate-limit a run whose URLs are spread "
                             "over many hosts")
    args = parser.parse_args(argv)

    names = list(HYDRATABLE) if args.all else args.datasets
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    retry_failed = args.retry_failed or args.wayback
    # Defaults chosen per mode: the first pass is a normal crawl, a retry is
    # deliberately gentler on hosts that already refused once, and the archive
    # gets ~1 req/s because it is a free service doing us a favour.
    if args.host_delay is not None:
        host_delay = args.host_delay
    elif args.wayback:
        host_delay = 1.0
    elif retry_failed:
        host_delay = 0.5
    else:
        host_delay = 0.0
    concurrency = args.concurrency
    if retry_failed and "--concurrency" not in (argv or sys.argv[1:]):
        concurrency = 1 if args.wayback else 4

    if args.global_delay < 0:
        parser.error("--global-delay must not be negative")

    if retry_failed or args.global_delay > 0:
        print(f"mode: {'wayback' if args.wayback else 'retry' if retry_failed else 'first-pass'}  "
              f"concurrency={concurrency}  host-delay={host_delay}s  "
              f"global-delay={args.global_delay}s")

    failures = 0
    for name in names:
        try:
            hydrate(name, concurrency, args.timeout, args.retries,
                    args.seed, args.dry_run, retry_failed=retry_failed,
                    use_wayback=args.wayback, host_delay=host_delay,
                    global_delay=args.global_delay,
                    breaker_threshold=args.breaker if retry_failed else 0)
        except FetchError as exc:
            failures += 1
            print(f"\n  SKIPPED — {name}\n{exc}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print(f"\n  interrupted during {name}; re-run to resume")
            return 130
        except Exception as exc:
            # A malformed CSV in one corpus must not abandon the others.
            failures += 1
            print(f"\n  SKIPPED {name}: {type(exc).__name__}: {exc}\n",
                  file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
