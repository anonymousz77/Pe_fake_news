"""Contract tests for the evidence store and time-filtered retrieval.

Offline: every test builds its own tiny index in a tmp_path. Nothing here
touches the network or the real store.

Two groups carry the weight, and they guard the same property from opposite
sides.

`test_a_future_dated_document_never_surfaces` plants a document published after
the claim and demands it never comes back at any `k`. A retrieval system that
leaks one such document invalidates every result computed with it, and the
failure is invisible in aggregate metrics.

The plausibility-window tests guard the other end. A document stamped
1970-01-01 is never *after* any claim, so it passes the filter for all of them —
which is the same leak wearing the opposite sign, because its real date is
unknown and may well be later than the claim it is being served to. Those tests
come in pairs: one that a value which cannot be a date is refused, and one that
a genuine old date is not, because a window tightened until the per-year table
looks tidy would delete real archival documents.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bible, redtest, retrieve  # noqa: E402


@pytest.fixture(autouse=True)
def _never_the_real_store(tmp_path_factory, monkeypatch):
    """No test may touch data/bible/, whether or not it asks for `store`.

    Three budget tests built a `Budget` without the fixture, so `store_bytes()`
    walked the real store. That was invisible while it held 122k documents and
    turned the suite into a multi-minute hang at 699k: the suite's runtime had
    quietly become a function of how much data happened to be on disk. Pointing
    the module at an empty directory by default makes that impossible rather
    than remembered.

    The one sanctioned exception is `test_the_red_test_holds_on_the_real_index`,
    which exists precisely to touch the real store. It reaches it through
    `scripts.redtest`, which resolves its own paths, so this fixture does not
    reach it -- and it is opt-in behind an environment variable so the default
    suite stays offline and fast.
    """
    root = tmp_path_factory.mktemp("empty_store")
    for name, path in (("BIBLE", root), ("DOCS", root / "docs"),
                       ("UNDATED", root / "undated"), ("INDEX", root / "index"),
                       ("STATE", root / "_state.json"),
                       # INTERIM too: the report scan is written there, and a
                       # test that wrote into the real one would leave a stale
                       # scan behind for the next real report to believe.
                       ("INTERIM", root / "interim")):
        monkeypatch.setattr(bible, name, path)


# --------------------------------------------------------------------------
# date extraction
# --------------------------------------------------------------------------


def test_jsonld_date_is_high_confidence():
    html = '<script type="application/ld+json">{"datePublished":"2020-03-04"}</script>'
    got, src, conf = bible.extract_date(html, {}, "https://x.example/a")
    assert (got, src, conf) == ("2020-03-04", "jsonld", "high")


def test_meta_published_is_high_confidence():
    html = '<meta property="article:published_time" content="2019-07-01T10:00:00Z">'
    got, src, conf = bible.extract_date(html, {}, "https://x.example/a")
    assert (got, src, conf) == ("2019-07-01", "meta_published", "high")


def test_url_path_date_is_low_confidence():
    got, src, conf = bible.extract_date("<html></html>", {},
                                        "https://x.example/2018/05/06/story")
    assert (got, src, conf) == ("2018-05-06", "url_path", "low")


def test_last_modified_never_becomes_a_publication_date():
    """The finding that drove this rule: 9 of 15 pages returning Last-Modified
    reported *today*, for claims from 2018-2020. It is a file mtime, not a
    publication date, so it records provenance and quarantines the document.
    """
    html = "<html><body>no dates here</body></html>"
    got, src, conf = bible.extract_date(
        html, {"Last-Modified": "Wed, 03 Sep 2026 11:00:00 GMT"},
        "https://x.example/no-date-in-path")
    assert got is None, "Last-Modified must not set published_at"
    assert src == "http_last_modified"
    assert conf == "none"


def test_a_page_with_nothing_yields_no_date():
    got, src, conf = bible.extract_date("<html></html>", {}, "https://x.example/a")
    assert (got, src, conf) == (None, "none", "none")


def test_wayback_capture_is_a_last_resort_bound():
    """A capture is later than publication, so it is a sound conservative bound.

    It is also a poor one -- 95.4% of sampled captures postdate their claim --
    so it must never win over a publisher filing path in the same URL.
    """
    bare = "https://web.archive.org/web/20190412093000/https://n.example/story"
    assert bible.url_date(bare) == ("2019-04-12", "wayback_capture")

    with_path = ("https://web.archive.org/web/20210412093000/"
                 "https://n.example/2019/05/03/story")
    assert bible.url_date(with_path) == ("2019-05-03", "url_path"), (
        "the archive capture stamp outranked the publisher's own filing date")


def test_a_publisher_filing_path_is_low_confidence_url_path():
    assert bible.url_date("https://n.example/2018-05-06/story") == (
        "2018-05-06", "url_path")


# --------------------------------------------------------------------------
# writing and quarantine
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    docs, undated, index = tmp_path / "docs", tmp_path / "undated", tmp_path / "index"
    monkeypatch.setattr(bible, "BIBLE", tmp_path)
    monkeypatch.setattr(bible, "DOCS", docs)
    monkeypatch.setattr(bible, "UNDATED", undated)
    monkeypatch.setattr(bible, "INDEX", index)
    monkeypatch.setattr(bible, "STATE", tmp_path / "_state.json")
    monkeypatch.setattr(bible, "INTERIM", tmp_path / "interim")
    return tmp_path


def _budget(store_gb=30.0, transfer_gb=60.0):
    return bible.Budget({"transfer_bytes": 0, "sources": {}}, store_gb, transfer_gb)


def test_a_dated_document_is_stored_for_indexing(store):
    doc = bible.make_document("https://x.example/a", "T", "body text",
                              "2020-01-01", "jsonld", "high")
    assert bible.write_document(doc, _budget()) == "indexed"
    assert (bible.DOCS / f"{doc.doc_id}.json").is_file()
    assert not (bible.UNDATED / f"{doc.doc_id}.json").exists()


def test_an_undated_document_is_quarantined_not_stored(store):
    doc = bible.make_document("https://x.example/b", "T", "body text",
                              None, "http_last_modified", "none")
    assert bible.write_document(doc, _budget()) == "quarantined"
    assert (bible.UNDATED / f"{doc.doc_id}.json").is_file()
    assert not (bible.DOCS / f"{doc.doc_id}.json").exists()


# --------------------------------------------------------------------------
# the plausibility window
#
# An impossible date is not untidy data, it is a hole in the time filter. A
# document stamped 1970-01-01 satisfies `published_at <= :as_of` for every claim
# ever made, so a page whose real date is unknown is served as evidence for all
# of them -- including claims it actually postdates. That is the same hindsight
# leakage the quarantine rule exists to prevent, so it gets the same treatment.
# --------------------------------------------------------------------------

TODAY = "2026-09-11"


@pytest.mark.parametrize("value, verdict", [
    ("0001-01-01", "out_of_window"),     # a null field reaching strftime
    ("0001-11-30", "out_of_window"),
    ("1899-12-31", "out_of_window"),
    ("2026-09-12", "out_of_window"),     # tomorrow
    ("2121-01-01", "out_of_window"),
    ("not-a-date", "out_of_window"),
    ("1969-12-31", "sentinel"),          # the epoch in a negative UTC offset
    ("1970-01-01", "sentinel"),          # the epoch itself
    ("1970-01-02", "sentinel"),
    ("1901-12-14", "sentinel"),          # 32-bit signed INT_MIN
])
def test_a_date_that_cannot_be_one_is_refused(value, verdict):
    assert bible.classify_date(value, TODAY) == (None, verdict)


@pytest.mark.parametrize("value", [
    "1900-01-01",   # the floor is inclusive
    "1970-04-17",   # a real NYT archive piece, one day off the epoch cluster
    "1985-01-20",
    "1995-01-31",
    "2020-06-15",
    "2026-09-11",   # today is inclusive
])
def test_a_genuine_date_survives_the_window(value):
    """The guard against over-correction, and it is the important half.

    Every pre-1996 date in the real store that is not a sentinel was checked by
    hand and is a genuine archival article whose URL carries the same date. A
    window tightened until the per-year table looks neat would delete them.
    """
    assert bible.classify_date(value, TODAY) == (value, "ok")


@pytest.mark.parametrize("value", ["1395-06-03", "1398-12-11", "2561-02-01",
                                   "2562-02-02"])
def test_another_calendar_is_not_a_broken_date(value):
    """Jalali and Buddhist-era years are real dates, so they are not refused.

    They fall outside the window at both ends, so they are tested before it.
    Converting them is a separate job; they are flagged and counted instead.
    """
    assert bible.classify_date(value, TODAY) == (value, "non_gregorian")


def test_a_refused_date_is_kept_as_raw_date_rejected(store):
    """Quarantining must not destroy the evidence of why it quarantined."""
    doc = bible.make_document("https://x.example/epoch", "T", "body text",
                              "1970-01-01", "jsonld", "high", today=TODAY)
    assert doc.published_at is None
    assert doc.raw_date_rejected == "1970-01-01"
    assert doc.date_source == "implausible_date"
    assert doc.date_confidence == "none"
    assert doc.provenance["date_rejected_because"] == "sentinel"
    assert bible.write_document(doc, _budget()) == "quarantined"
    assert (bible.UNDATED / f"{doc.doc_id}.json").is_file()
    assert not (bible.DOCS / f"{doc.doc_id}.json").exists()


def test_the_window_is_enforced_for_every_source(store):
    """It lives in `make_document`, which all three sources funnel through.

    Wikinews' explicit field, enrich's publisher metadata and CC-NEWS' WARC
    stamp reach the store by different routes and none of them may bypass it.
    """
    for src, conf in (("explicit_field", "high"), ("warc_date", "low"),
                      ("jsonld", "high"), ("url_path", "low")):
        doc = bible.make_document(f"https://x.example/{src}", "T", "body",
                                  "1970-01-01", src, conf, today=TODAY)
        assert doc.published_at is None, f"{src} bypassed the window"
        assert doc.raw_date_rejected == "1970-01-01"


def test_requarantine_moves_only_what_it_should(store):
    """The migration for documents already in the store, both directions."""
    keep = bible.make_document("https://x.example/keep", "T", "body",
                               "2020-06-15", "jsonld", "high", today=TODAY)
    old = bible.make_document("https://x.example/old", "T", "body",
                              "1985-01-20", "url_path", "low", today=TODAY)
    jalali = bible.make_document("https://x.example/jalali", "T", "body",
                                 "1395-06-03", "jsonld", "high", today=TODAY)
    for doc in (keep, old, jalali):
        assert bible.write_document(doc, _budget()) == "indexed"
    # Written straight to docs/ as a pre-window build would have left it.
    bad = bible.make_document("https://x.example/bad", "T", "body",
                              "2020-01-01", "jsonld", "high", today=TODAY)
    bad.published_at = "1970-01-01"
    (bible.DOCS / f"{bad.doc_id}.json").write_text(bad.to_json(),
                                                   encoding="utf-8")

    r = bible.requarantine(today=TODAY)
    assert r["quarantined"] == 1
    assert r["non_gregorian_kept"] == 1
    assert (bible.UNDATED / f"{bad.doc_id}.json").is_file()
    assert not (bible.DOCS / f"{bad.doc_id}.json").exists()
    for doc in (keep, old, jalali):
        assert (bible.DOCS / f"{doc.doc_id}.json").is_file()
    moved = json.loads((bible.UNDATED / f"{bad.doc_id}.json").read_text(
        encoding="utf-8"))
    assert moved["published_at"] is None
    assert moved["raw_date_rejected"] == "1970-01-01"


def test_requarantine_is_idempotent(store):
    doc = bible.make_document("https://x.example/e", "T", "body",
                              "2020-01-01", "jsonld", "high", today=TODAY)
    doc.published_at = "0001-01-01"
    bible.DOCS.mkdir(parents=True, exist_ok=True)
    (bible.DOCS / f"{doc.doc_id}.json").write_text(doc.to_json(),
                                                   encoding="utf-8")
    assert bible.requarantine(today=TODAY)["quarantined"] == 1
    assert bible.requarantine(today=TODAY).get("quarantined", 0) == 0


def test_a_dry_run_requarantine_moves_nothing(store):
    doc = bible.make_document("https://x.example/f", "T", "body",
                              "2020-01-01", "jsonld", "high", today=TODAY)
    doc.published_at = "1970-01-01"
    bible.DOCS.mkdir(parents=True, exist_ok=True)
    (bible.DOCS / f"{doc.doc_id}.json").write_text(doc.to_json(),
                                                   encoding="utf-8")
    r = bible.requarantine(dry_run=True, today=TODAY)
    assert r["quarantined"] == 1
    assert (bible.DOCS / f"{doc.doc_id}.json").is_file()
    assert not (bible.UNDATED / f"{doc.doc_id}.json").exists()


# --------------------------------------------------------------------------
# the report scan
#
# The report needs four fields per document and getting them meant reading all
# 8.6 GB of the store. On a 16 GB machine that filled the file cache until the
# OS reported a few hundred MB free, and the report was killed for memory three
# times while itself using 217 MB. Free memory returns as soon as the process
# exits, so the scan reads the store in resumable slices and writes ~90 bytes
# per document; the report then reads 120 MB instead of 8.6 GB.
# --------------------------------------------------------------------------


def _plant_many(n, dated=True, prefix='s'):
    """n documents straight through make_document/write_document.

    Named apart from the `_plant` further down, which writes a document dict
    directly: that one exists to plant awkward shapes the builders would refuse,
    this one exists to exercise the real write path.
    """
    made = []
    for i in range(n):
        doc = bible.make_document(f"https://x.example/{prefix}{i}", f"T{i}", "body",
                                  "2020-01-01" if dated else None,
                                  "jsonld" if dated else "none",
                                  "high" if dated else "none",
                                  provenance={"source": "ccnews"}, today=TODAY)
        bible.write_document(doc, _budget())
        made.append(doc)
    return made


def test_a_scan_resumes_where_it_stopped(store):
    _plant_many(5)
    first = bible.scan_store(bible.DOCS, chunk=2)
    assert (first["scanned"], first["done"]) == (2, False)
    second = bible.scan_store(bible.DOCS, chunk=2)
    assert (second["scanned"], second["written_now"], second["done"]) == (4, 2, False)
    third = bible.scan_store(bible.DOCS, chunk=2)
    assert (third["scanned"], third["written_now"], third["done"]) == (5, 1, True)
    lines = bible.scan_path(bible.DOCS).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_a_scan_killed_mid_write_drops_its_partial_line(store):
    """A run killed between two writes leaves half a line. Resuming must not
    treat that half as a scanned document, or every later record is off by one.
    """
    _plant_many(4)
    bible.scan_store(bible.DOCS, chunk=2)
    path = bible.scan_path(bible.DOCS)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"published_at": "2020-01-0')      # killed here
    done = bible.scan_store(bible.DOCS)
    assert done["done"] and done["scanned"] == 4
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(json.loads(line) for line in lines)


def test_an_incomplete_scan_is_ignored_rather_than_believed(store):
    _plant_many(4)
    bible.scan_store(bible.DOCS, chunk=2)
    assert bible.read_scan(bible.DOCS) is None, "a half-written scan was used"
    bible.scan_store(bible.DOCS)
    assert bible.read_scan(bible.DOCS) is not None


def test_a_stale_scan_is_ignored_when_the_store_moves_on(store):
    _plant_many(3)
    bible.scan_store(bible.DOCS)
    assert bible.read_scan(bible.DOCS) is not None
    _plant_many(1, prefix='late')   # a document arrives after the scan
    assert bible.read_scan(bible.DOCS) is None


def test_the_scan_carries_every_field_the_report_reads(store):
    doc = bible.make_document("https://x.example/scanned", "T", "body",
                              "2020-01-01", "jsonld", "high",
                              provenance={"source": "enrich",
                                          "claim_date": "2021-01-01"},
                              today=TODAY)
    bible.write_document(doc, _budget())
    bible.scan_store(bible.DOCS)
    rec = next(iter(bible.read_scan(bible.DOCS)))
    assert rec["published_at"] == "2020-01-01"
    assert rec["date_source"] == "jsonld"
    assert rec["date_confidence"] == "high"
    assert rec["source_domain"] == "x.example"
    assert rec["provenance"] == {"source": "enrich", "claim_date": "2021-01-01"}


def test_the_report_agrees_with_itself_scanned_or_not(store, monkeypatch):
    """The scan is an optimisation, so it must not change a single number."""
    monkeypatch.setattr(bible, "REPORTS", store / "reports")
    monkeypatch.setattr(bible, "enrich_targets", lambda: [])
    _plant_many(6)
    _plant_many(2, dated=False)
    walked = bible.report({"transfer_bytes": 0, "sources": {}})
    for folder in (bible.DOCS, bible.UNDATED):
        bible.scan_store(folder)
    scanned = bible.report({"transfer_bytes": 0, "sources": {}})
    for key in ("documents_indexed", "documents_quarantined", "date_source",
                "date_source_whole_store", "date_confidence", "by_origin",
                "by_year", "date_range", "quarantined_by_reason"):
        assert walked[key] == scanned[key], f"the scan changed {key}"


def test_the_index_contains_the_body_not_just_the_title(store):
    """A term that appears ONLY in the body must be searchable.

    Nothing else here would notice an index built from empty text: the row
    counts would be right, the dates would be right, and every date test would
    pass. It would simply retrieve nothing. This was very nearly shipped -- a
    memory fix that drops the body for the report walk was applied to the index
    walk by mistake -- so the property is pinned rather than trusted.
    """
    doc = bible.make_document("https://x.example/body", "a plain headline",
                              "quokka sightings rose sharply", "2019-01-01",
                              "jsonld", "high", today=TODAY)
    assert bible.write_document(doc, _budget()) == "indexed"
    bible.build_index()
    con = retrieve.connect(bible.INDEX / "bm25.sqlite",
                           bible.INDEX / "temporal.sqlite")
    try:
        hits = retrieve.search("quokka", "2020-01-01", k=5, con=con)
    finally:
        con.close()
    assert [h.doc_id for h in hits] == [doc.doc_id], (
        "a body-only term did not match -- the index holds no text")


def test_an_implausibly_dated_document_never_reaches_the_index(store, tmp_path):
    """The end the whole rule exists for: it must not be retrievable.

    Planted as a document rather than as an index row, so the path under test is
    the real one -- make_document, write_document, build_index -- and the
    assertion is that a 1970-stamped page cannot be returned for a 2019 claim.
    """
    good = bible.make_document("https://x.example/good", "vaccine study",
                               "vaccine trial results", "2018-01-01",
                               "jsonld", "high", today=TODAY)
    epoch = bible.make_document("https://x.example/epoch", "vaccine study",
                                "vaccine trial results", "1970-01-01",
                                "jsonld", "high", today=TODAY)
    assert bible.write_document(good, _budget()) == "indexed"
    assert bible.write_document(epoch, _budget()) == "quarantined"
    bible.build_index()
    con = retrieve.connect(bible.INDEX / "bm25.sqlite",
                           bible.INDEX / "temporal.sqlite")
    try:
        hits = retrieve.search("vaccine", "2019-01-01", k=100, con=con)
    finally:
        con.close()
    ids = {h.doc_id for h in hits}
    assert good.doc_id in ids
    assert epoch.doc_id not in ids, "an epoch-stamped document was retrievable"


def test_writing_is_idempotent(store):
    doc = bible.make_document("https://x.example/c", "T", "x", "2020-01-01",
                              "jsonld", "high")
    assert bible.write_document(doc, _budget()) == "indexed"
    assert bible.write_document(doc, _budget()) == "skipped"


# --------------------------------------------------------------------------
# budgets
# --------------------------------------------------------------------------


def test_transfer_cap_refuses_rather_than_truncating(store):
    b = _budget(transfer_gb=0.000001)
    with pytest.raises(bible.BudgetError, match="transfer cap"):
        b.spend(10_000)


def test_store_cap_refuses(store):
    b = _budget(store_gb=0.0000001)
    doc = bible.make_document("https://x.example/d", "T", "x" * 5000,
                              "2020-01-01", "jsonld", "high")
    with pytest.raises(bible.BudgetError, match="store cap"):
        bible.write_document(doc, b)


def test_transfer_is_cumulative_across_calls(store):
    b = _budget()
    b.spend(1000); b.spend(2500)
    assert b.transferred == 3500
    assert b.state["transfer_bytes"] == 3500


# --------------------------------------------------------------------------
# retrieval: the rule that matters
# --------------------------------------------------------------------------


def _tiny_index(tmp_path, rows):
    """rows = [(doc_id, title, text, published_at | None)]"""
    index = tmp_path / "index"
    index.mkdir(parents=True, exist_ok=True)
    bm, tm = index / "bm25.sqlite", index / "temporal.sqlite"
    b = sqlite3.connect(bm)
    b.execute("CREATE VIRTUAL TABLE docs_fts USING fts5("
              "doc_id UNINDEXED, title, text, tokenize='porter unicode61')")
    t = sqlite3.connect(tm)
    t.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, published_at TEXT "
              "NOT NULL, source_domain TEXT, url TEXT)")
    for doc_id, title, text, pub in rows:
        b.execute("INSERT INTO docs_fts VALUES (?,?,?)", (doc_id, title, text))
        if pub is not None:
            t.execute("INSERT INTO docs VALUES (?,?,?,?)",
                      (doc_id, pub, "x.example", f"https://x.example/{doc_id}"))
    t.execute("CREATE INDEX idx_published_at ON docs(published_at)")
    b.commit(); t.commit(); b.close(); t.close()
    return bm, tm


def test_a_future_dated_document_never_surfaces(tmp_path):
    """Plant a document published AFTER the as_of date and demand it stay out.

    k is deliberately larger than the corpus, so the only thing that can keep
    the future document out is the date predicate itself.
    """
    bm, tm = _tiny_index(tmp_path, [
        ("past", "vaccine study", "vaccine trial results", "2019-12-31"),
        ("same", "vaccine study", "vaccine trial results", "2020-01-01"),
        ("future", "vaccine study", "vaccine trial results", "2020-01-02"),
        ("far_future", "vaccine study", "vaccine trial results", "2031-06-06"),
    ])
    con = retrieve.connect(bm, tm)
    try:
        hits = retrieve.search("vaccine", "2020-01-01", k=100, con=con)
    finally:
        con.close()
    ids = {h.doc_id for h in hits}
    assert "past" in ids and "same" in ids, "on-or-before must be inclusive"
    assert "future" not in ids, "a document published the NEXT DAY leaked"
    assert "far_future" not in ids
    assert all(h.published_at <= "2020-01-01" for h in hits)


def test_an_undated_document_never_surfaces(tmp_path):
    """It is absent from the temporal table, so the JOIN cannot reach it."""
    bm, tm = _tiny_index(tmp_path, [
        ("dated", "vaccine study", "vaccine trial results", "2019-01-01"),
        ("undated", "vaccine study", "vaccine trial results", None),
    ])
    con = retrieve.connect(bm, tm)
    try:
        hits = retrieve.search("vaccine", "2030-01-01", k=100, con=con)
    finally:
        con.close()
    ids = {h.doc_id for h in hits}
    assert ids == {"dated"}, f"undated document leaked: {ids}"


def test_the_filter_is_in_the_query_not_a_post_filter(tmp_path):
    """Ask for k=2 where 2 of 4 matches are in-range: both must come back.

    A post-filter would return the top-2 by score and then drop the ones that
    fail the date test, silently returning fewer than k.
    """
    bm, tm = _tiny_index(tmp_path, [
        ("f1", "vaccine", "vaccine vaccine vaccine", "2025-01-01"),
        ("f2", "vaccine", "vaccine vaccine vaccine", "2026-01-01"),
        ("p1", "vaccine", "vaccine", "2019-01-01"),
        ("p2", "vaccine", "vaccine", "2018-01-01"),
    ])
    con = retrieve.connect(bm, tm)
    try:
        hits = retrieve.search("vaccine", "2020-01-01", k=2, con=con)
    finally:
        con.close()
    assert len(hits) == 2, "k was degraded by filtering after ranking"
    assert {h.doc_id for h in hits} == {"p1", "p2"}


def test_sql_names_the_date_predicate():
    """Guard the property structurally, so a refactor cannot quietly drop it."""
    sql = retrieve.SEARCH_SQL.lower()
    assert "published_at <= :as_of" in sql
    assert "published_at is not null" in sql
    assert "join" in sql and "match" in sql


def test_a_malformed_as_of_is_refused(tmp_path):
    bm, tm = _tiny_index(tmp_path, [("a", "t", "vaccine", "2019-01-01")])
    con = retrieve.connect(bm, tm)
    try:
        for bad in ("2020", "01-01-2020", "", "yesterday"):
            with pytest.raises(retrieve.RetrievalError, match="ISO date"):
                retrieve.search("vaccine", bad, k=5, con=con)
    finally:
        con.close()


def test_query_text_cannot_inject_fts_syntax(tmp_path):
    bm, tm = _tiny_index(tmp_path, [("a", "t", "vaccine safety", "2019-01-01")])
    con = retrieve.connect(bm, tm)
    try:
        hits = retrieve.search('vaccine OR "* safety" NEAR/5', "2020-01-01",
                               k=5, con=con)
        assert isinstance(hits, list)
    finally:
        con.close()


def test_a_missing_index_says_how_to_build_it(tmp_path):
    with pytest.raises(retrieve.RetrievalError, match="bible.py --index"):
        retrieve.connect(tmp_path / "nope.sqlite", tmp_path / "nah.sqlite")


#: The real-index red-test takes minutes and needs the built store, so it is
#: opt-in. `pytest -q` must stay offline, data-free and fast; this is the one
#: test that is none of those things.
_REDTEST_ENV = "PE_FAKE_NEWS_REDTEST"


@pytest.mark.skipif(not os.environ.get(_REDTEST_ENV),
                    reason=f"set {_REDTEST_ENV}=1 to red-test the real index")
def test_the_red_test_holds_on_the_real_index():
    """The same predicate, against 1.4M rows instead of 4.

    Everything above proves the SQL is right on a fixture. This proves it is
    right on the store, which is a different claim: SQLite chooses different
    plans at different table sizes, and a `k` that exceeds a four-row corpus is
    dwarfed by a real one. `scripts/redtest.py` is the entry point and rolls its
    plants back in a `finally`; this only asserts on what it reports.
    """
    r = redtest.run()
    assert r["failures"] == [], r["failures"]
    assert r["index_rows_after"] == r["index_rows_before"], "rollback did not restore"
    assert r["ranking_fingerprint_unchanged"]
    for probe in r["probe_nonce"]:
        assert not probe["future_returned"] and not probe["null_returned"]


# --------------------------------------------------------------------------
# wikinews parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wikitext, expected", [
    ("{{date|March 4, 2020}} body", "2020-03-04"),
    ("{{ date | 4 March 2020 }} body", "2020-03-04"),
    ("text [[Category:July 9, 2016]] more", "2016-07-09"),
    ("no date at all", None),
])
def test_wikinews_dates(wikitext, expected):
    assert bible._wikinews_date(wikitext) == expected


def test_wikitext_is_stripped_to_readable_prose():
    raw = "{{date|March 4, 2020}} The [[United Nations|UN]] said '''this'''."
    out = bible._strip_wikitext(raw)
    assert "UN said this." in out
    assert "{{" not in out and "[[" not in out


# --------------------------------------------------------------------------
# the mocheg exclusion, pinned
# --------------------------------------------------------------------------


def test_mocheg_is_not_an_enrichment_source():
    """It ships no claim-date field, so nothing in it can be time-filtered.

    78% of all bundled evidence. Pinned as a test so re-adding it requires
    deliberately deleting this.
    """
    assert "mocheg" not in bible.ENRICH_DATASETS
    assert set(bible.ENRICH_DATASETS) == {"averitec", "averimatec"}


@pytest.mark.parametrize("value, expected", [
    ("2019-07-01T10:00:00Z", "2019-07-01"),
    ("2019-07-01T10:00:00+01:00", "2019-07-01"),
    ("2019-07-01 10:00:00", "2019-07-01"),
    ("2019-07-01", "2019-07-01"),
    ("2019-07-011", None),
])
def test_iso_datetimes_are_parsed(value, expected):
    """An ISO datetime has no word boundary before the 'T'.

    Requiring one rejected the commonest meta / JSON-LD format outright, which
    under-counted high-confidence dates in the source survey.
    """
    from scripts.verify import iso_date

    assert iso_date(value) == expected


# --------------------------------------------------------------------------
# the CC-NEWS allocation
# --------------------------------------------------------------------------


def test_no_allocated_month_predates_the_crawl():
    """CC-NEWS begins 2016-08. Earlier months 404.

    The first allocation asked for 2016-01 and 2016-04, which do not exist, and
    the builder skips an unavailable month rather than failing -- so 2 of 40
    WARCs would have vanished silently and the stated claim coverage would have
    been wrong with nothing in the output to say so.
    """
    assert min(bible.CCNEWS_ALLOCATION) >= "2016-08"


def test_the_allocation_spends_exactly_the_warc_budget():
    assert sum(bible.CCNEWS_ALLOCATION.values()) == bible.DEFAULT_WARC_LIMIT == 40


# --------------------------------------------------------------------------
# robustness: one bad page must not kill a multi-hour build
# --------------------------------------------------------------------------


def test_markup_bs4_rejects_still_yields_text():
    """The page shape that killed the enrich run at 4,400 of 8,482.

    bs4 raises ParserRejectedMarkup on binary bytes inside a marked section.
    A builder streaming thousands of documents cannot die on one of them, and
    the page should still contribute its text rather than vanish.
    """
    bad = "<html><title>Ok</title><body><![CDATA[k\x1e junk]]><p>real text</p></body></html>"
    title, text = bible.html_to_text(bad)
    assert title == "Ok"
    assert "real text" in text


@pytest.mark.parametrize("html", ["", "<html>", "<p>" * 500, "\x00\x1e\xff"])
def test_html_to_text_never_raises(html):
    title, text = bible.html_to_text(html)
    assert isinstance(title, str) and isinstance(text, str)


def test_capture_bound_marks_the_archive_stamp_not_a_filing_path():
    """These were inverted once: the flag was computed as `src == "url_path"`
    after url_path and wayback_capture had been split apart, so publisher
    filing dates were labelled archive bounds and vice versa.
    """
    import inspect

    src = inspect.getsource(bible.build_enrich)
    assert '"capture_bound": src == "wayback_capture"' in src


# --------------------------------------------------------------------------
# transfer accounting
# --------------------------------------------------------------------------


def test_transfer_counts_wire_bytes_not_decompressed_bytes():
    """A WARC is gzipped ~4x, and the budget is a NETWORK budget.

    Measured after gunzip, four real CC-NEWS WARCs reported 15.96 GB against a
    60 GB cap when 3.27 GB had actually moved. The run would have refused at
    roughly WARC 15 of 40, and the transfer figure in the report would have been
    wrong by the compression ratio.
    """
    import gzip
    import io

    payload = b"news article body. " * 200_000
    wire = gzip.compress(payload)
    reader = bible._CountingReader(io.BytesIO(wire))
    out = gzip.GzipFile(fileobj=reader).read()

    assert out == payload
    assert reader.count == len(wire), "did not tally the compressed stream"
    assert reader.count < len(payload), "counted decompressed bytes"


def test_the_warc_crawl_date_is_not_a_url_path():
    """It is a crawl timestamp: an upper bound, like wayback_capture.

    CC-NEWS quarantines nothing because every record falls back to this date, so
    mislabelling it would have made `url_path` the dominant date_source in the
    whole store while saying nothing true about publisher filing conventions.
    """
    import inspect

    src = inspect.getsource(bible.build_ccnews)
    assert 'published, src, conf = got, "warc_date", "low"' in src
    # `wire` counts the compressed stream; `moved` counts what came out of it.
    # Charging is incremental, so what must hold is that every spend is against
    # wire and none is against moved.
    assert "budget.spend(wire.count - charged)" in src
    assert "budget.spend(moved" not in src, "spends decompressed bytes"


# --------------------------------------------------------------------------
# extraction: throughput without losing the article
# --------------------------------------------------------------------------


PAGE = ("<html><head><title>Headline</title></head><body>"
        "<nav>Home About Contact</nav>"
        "<script>var x = 'tracking';</script>"
        "<p>The article body says something specific and checkable.</p>"
        "<footer>Copyright 2020 Example</footer></body></html>")


@pytest.mark.parametrize("extractor", ["fast", "bs4"])
def test_both_extractors_drop_boilerplate_and_keep_the_article(extractor):
    title, text = bible.html_to_text(PAGE, extractor)
    assert title == "Headline"
    assert "article body says something specific" in text
    for chrome in ("Home About Contact", "tracking", "Copyright 2020"):
        assert chrome not in text, f"{extractor} kept boilerplate: {chrome!r}"


def test_the_fast_extractor_is_the_default_for_bulk_but_not_per_url():
    """bs4 costs ~56 ms/page against ~4 ms; over a WARC's ~53k pages that is
    ~50 minutes versus ~4. Per-URL work keeps bs4, where the cost is irrelevant.
    """
    import inspect

    assert inspect.signature(bible.html_to_text).parameters[
        "extractor"].default == "bs4"
    assert inspect.signature(bible.build_ccnews).parameters[
        "extractor"].default == "fast"


def test_the_prefilter_tests_the_payload_not_the_record():
    """A WARC record body starts with HTTP headers.

    Sniffing the record instead of the payload rejected 93% of real records and
    lost 522 of 550 documents the parse path kept.
    """
    payload = PAGE.encode() + b"x" * 2000
    record = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + payload

    assert bible.looks_like_an_article(payload)
    assert not bible.looks_like_an_article(b"tiny")
    # the header block alone must not be mistaken for an article
    assert not bible.looks_like_an_article(record.partition(b"\r\n\r\n")[0])


# --------------------------------------------------------------------------
# resuming: --warc-limit is a TOTAL, not a per-run quantity
# --------------------------------------------------------------------------


class _FakeResponse:
    status_code = 503


class _RecordingSession:
    """Records every WARC URL asked for and refuses to serve any of them."""

    def __init__(self):
        self.asked: list[str] = []

    def get(self, url, **kw):
        self.asked.append(url)
        return _FakeResponse()


def _paths_for(month: str, n: int) -> list[str]:
    y, m = month.split("-")
    return [f"crawl-data/CC-NEWS/{y}/{m}/CC-NEWS-{y}{m}01{i:06d}.warc.gz"
            for i in range(n)]


def test_a_resumed_month_does_not_refill_its_quota(store, monkeypatch):
    """A completed WARC must spend its month's share, not just be skipped.

    Excluding done paths from the candidate list stops a refetch and nothing
    more: the slice `[:want]` then simply took the NEXT file, so resuming a
    2-of-40 build fetched 40 more -- 42 in all -- and handed the extra two to
    the months that had already run, which are the cheapest ones rather than
    the ones carrying the most claims.
    """
    month = "2019-02"
    available = _paths_for(month, 10)
    session = _RecordingSession()
    monkeypatch.setattr(bible, "make_session", lambda *a, **k: session)
    monkeypatch.setattr(bible, "warc_paths", lambda s, mo: available)

    state = {"transfer_bytes": 0,
             "sources": {"ccnews": {"warcs_done": available[:2]}}}
    bible.build_ccnews(bible.Budget(state), state, dry_run=False,
                       warc_limit=3, allocation={month: 3})

    assert len(session.asked) == 1, (
        f"asked for {len(session.asked)} WARCs; 2 of the 3 were already done")
    assert session.asked[0].endswith(available[2].split("/")[-1])


def test_a_month_already_complete_is_skipped_entirely(store, monkeypatch):
    month = "2016-08"
    available = _paths_for(month, 5)
    session = _RecordingSession()
    monkeypatch.setattr(bible, "make_session", lambda *a, **k: session)
    monkeypatch.setattr(bible, "warc_paths", lambda s, mo: available)

    state = {"transfer_bytes": 0,
             "sources": {"ccnews": {"warcs_done": available[:1]}}}
    bible.build_ccnews(bible.Budget(state), state, dry_run=False,
                       warc_limit=1, allocation={month: 1})

    assert session.asked == [], "a satisfied month asked for another WARC"


# --------------------------------------------------------------------------
# the report: what it must not average away
# --------------------------------------------------------------------------


def _plant(doc_id, published_at, date_source, confidence, origin, **prov):
    folder = bible.DOCS if published_at else bible.UNDATED
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{doc_id}.json").write_text(json.dumps({
        "doc_id": doc_id, "url": f"https://x.example/{doc_id}", "title": "t",
        "text": "body", "published_at": published_at,
        "date_source": date_source, "date_confidence": confidence,
        "source_domain": "x.example", "language": "en",
        "fetched_at": "2026-09-05T00:00:00+00:00", "content_sha256": "0" * 64,
        "images": [], "provenance": {"source": origin, **prov},
    }), encoding="utf-8")


@pytest.fixture
def planted_store(store, monkeypatch):
    monkeypatch.setattr(bible, "REPORTS", store / "reports")
    monkeypatch.setattr(bible, "enrich_targets",
                        lambda: [("averitec", "2020-01-01", f"https://e/{i}")
                                 for i in range(10)])
    _plant("w1", "2016-01-01", "explicit_field", "high", "wikinews")
    _plant("w2", "2016-06-02", "explicit_field", "high", "wikinews")
    _plant("w3", None, "none", "none", "wikinews")
    _plant("e1", "2019-01-01", "jsonld", "high", "enrich", claim_date="2019-06-01")
    _plant("e2", "2019-02-01", "meta_published", "high", "enrich",
           claim_date="2019-01-01")
    _plant("e3", "2019-03-01", "url_path", "low", "enrich", claim_date="2020-01-01")
    _plant("e4", None, "http_last_modified", "none", "enrich")
    _plant("c1", "2020-05-05", "warc_date", "low", "ccnews")
    return store


def test_the_whole_store_date_source_includes_the_quarantined_half(planted_store):
    """`none` exists only in undated/, so the indexed breakdown cannot show it.

    Reporting only the indexed half describes the documents that survived, which
    is the half that cannot show why the others did not.
    """
    r = bible.report({"transfer_bytes": 0})
    assert r["date_source"] == {"explicit_field": 2, "jsonld": 1,
                                "meta_published": 1, "url_path": 1,
                                "warc_date": 1}
    assert r["date_source_whole_store"]["none"] == 1
    assert r["date_source_whole_store"]["http_last_modified"] == 1
    assert (sum(r["date_source_whole_store"].values())
            == r["documents_indexed"] + r["documents_quarantined"] == 8)


def test_quarantine_is_attributed_to_its_source(planted_store):
    """A pooled quarantine count hides which source is failing to date."""
    r = bible.report({"transfer_bytes": 0})
    assert r["quarantined_by_origin"] == {"wikinews": 1, "enrich": 1}


def test_the_enrich_yield_divides_by_urls_asked_for_not_documents_got(planted_store):
    """3 of 10 URLs indexed, 2 of them high-confidence -> 20%, not 66.7%.

    Dividing by the documents that came back would divide by the fetch success
    rate and report a yield the source never achieved.
    """
    r = bible.report({"transfer_bytes": 0})
    ey = r["enrich_yield"]
    assert ey["evidence_urls"] == 10
    assert ey["indexed"] == 3 and ey["quarantined"] == 1
    assert ey["high_confidence"] == 2
    assert ey["high_confidence_rate_over_urls"] == 0.2
    assert ey["high_confidence_date_sources"] == {"jsonld": 1, "meta_published": 1}


def test_the_enrich_yield_is_null_rather_than_wrong_without_the_corpora(
        planted_store, monkeypatch):
    def boom():
        raise FileNotFoundError("data/raw/averitec/data/train.json")

    monkeypatch.setattr(bible, "enrich_targets", boom)
    ey = bible.report({"transfer_bytes": 0})["enrich_yield"]
    assert ey["evidence_urls"] is None
    assert "FileNotFoundError" in ey["unavailable"]


def test_per_year_coverage_is_reported(planted_store):
    r = bible.report({"transfer_bytes": 0})
    assert r["by_year"] == {"2016": 2, "2019": 3, "2020": 1}
    assert r["date_range"] == {"min": "2016-01-01", "max": "2020-05-05"}


def test_predating_its_own_claim_is_counted_only_for_enrich(planted_store):
    """Only enrich documents carry a claim to be compared against."""
    r = bible.report({"transfer_bytes": 0})
    assert r["enrich_vs_claim"] == {"predates": 2, "postdates": 1}
    assert r["enrich_predate_rate"] == 0.6667


def test_a_truncated_document_does_not_cost_the_whole_report(store):
    """One unparseable file must not take the other million with it."""
    bible.DOCS.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        _plant(f"d{i:03d}", "2020-01-01", "jsonld", "high", "ccnews")
    (bible.DOCS / "truncated.json").write_text('{"doc_id": "x", "publi',
                                               encoding="utf-8")
    got = list(bible.read_documents(bible.DOCS))
    assert len(got) == 50
    assert [d["doc_id"] for d in got] == sorted(d["doc_id"] for d in got), (
        "filename order was not preserved across the thread pool")


def test_read_documents_is_empty_for_a_folder_that_does_not_exist(store):
    assert list(bible.read_documents(bible.DOCS / "nope")) == []


# --------------------------------------------------------------------------
# a month must not be lost to one bad response
# --------------------------------------------------------------------------


class _StatusSession:
    """Answers with a scripted sequence of status codes."""

    def __init__(self, codes, body=b""):
        self.codes = list(codes)
        self.body = body
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        code = self.codes.pop(0) if self.codes else 200

        class R:
            status_code = code
            content = self.body
        return R()


def _paths_gz(paths):
    import gzip as _gz
    return _gz.compress("\n".join(paths).encode())


def test_a_transient_503_is_retried_not_treated_as_a_missing_month():
    """One unretried 503 cost 2020-03 and 2021-01 -- 14 of 40 WARCs.

    `make_session`'s argument is a connection-pool size, not a retry count, so
    this listing had exactly one attempt and the builder's skip-and-continue
    turned a bad minute into a permanently missing month.
    """
    wanted = ["crawl-data/CC-NEWS/2020/03/a.warc.gz"]
    session = _StatusSession([503, 503, 429], _paths_gz(wanted))
    got = bible.warc_paths(session, "2020-03", sleeper=lambda s: None)
    assert got == wanted
    assert session.calls == 4, "did not retry through the transient failures"


def test_a_404_month_is_permanent_and_not_retried():
    """CC-NEWS begins 2016-08; earlier months answer 404 forever."""
    session = _StatusSession([404])
    with pytest.raises(bible.MonthUnavailable) as exc:
        bible.warc_paths(session, "2016-01", sleeper=lambda s: None)
    assert exc.value.permanent is True
    assert session.calls == 1, "retried a month that does not exist"


def test_exhausted_retries_report_the_month_as_transient():
    session = _StatusSession([503] * 10)
    with pytest.raises(bible.MonthUnavailable) as exc:
        bible.warc_paths(session, "2020-03", attempts=3, sleeper=lambda s: None)
    assert exc.value.permanent is False
    assert session.calls == 3


def test_an_unavailable_month_reports_its_cost_in_warcs(store, monkeypatch):
    """'1 month unavailable' understates a 12-WARC loss by an order of magnitude."""
    def refuse(session, month, **kw):
        raise bible.MonthUnavailable(f"{month}: HTTP 503", permanent=False)

    monkeypatch.setattr(bible, "make_session", lambda *a, **k: _RecordingSession())
    monkeypatch.setattr(bible, "warc_paths", refuse)
    state = {"transfer_bytes": 0, "sources": {}}
    stats = bible.build_ccnews(bible.Budget(state), state, dry_run=False,
                               warc_limit=14,
                               allocation={"2020-03": 12, "2021-01": 2})
    assert stats["warcs_unfetched"] == 14
    assert stats["months_transiently_unavailable"] == 2
    assert stats.get("months_absent", 0) == 0


def test_an_already_fetched_warc_is_not_counted_as_lost(store, monkeypatch):
    """The shortfall is what is still owed, not the month's whole allocation."""
    def refuse(session, month, **kw):
        raise bible.MonthUnavailable(f"{month}: HTTP 503", permanent=False)

    monkeypatch.setattr(bible, "make_session", lambda *a, **k: _RecordingSession())
    monkeypatch.setattr(bible, "warc_paths", refuse)
    state = {"transfer_bytes": 0, "sources": {"ccnews": {"warcs_done": [
        "crawl-data/CC-NEWS/2020/03/done1.warc.gz",
        "crawl-data/CC-NEWS/2020/03/done2.warc.gz"]}}}
    stats = bible.build_ccnews(bible.Budget(state), state, dry_run=False,
                               warc_limit=12, allocation={"2020-03": 12})
    assert stats["warcs_unfetched"] == 10
