#!/usr/bin/env python
"""Red-team the retrieval rules against the REAL evidence-store index.

    python scripts/redtest.py

`tests/test_bible.py` already plants documents that must not surface -- but it
does so in a four-row temporary index. A rule that holds on four rows proves
nothing about a rule that has to hold on 1.4 million: the query planner picks
different plans at different table sizes, a `LIMIT` that is larger than a toy
corpus is smaller than a real one, and an `ORDER BY` over four rows never
spills to a temp b-tree. So this runs the same statement against the store as
built.

Two rules are under test (scripts/retrieve.py): a document may only come back
if it was published STRICTLY BEFORE the `before` date, and never if it is a
fact-checker's page (configs/factcheckers.yaml: its hosts, sections, documents,
and the path words that block on every host). Every plant carries a nonce that
occurs nowhere else in the store:

    date plants, on redtest.invalid
      prior            the day BEFORE `before`             must come back
      same_day         ON `before`                          must never come back
      future           the day AFTER                        must never come back
      null             absent from the temporal table       must never come back

    source plants, all dated the day before
      fc_domain        a host the list blocks whole         must never come back
      fc_subdomain     a subdomain of it                    must never come back
      fc_upper_case    that host in upper case              must never come back
      fc_wayback       a Wayback capture of that host       must never come back
      desk_section     inside a publisher's fact-check desk must never come back
      desk_upper_path  that section in upper case           must never come back
      desk_wayback     a Wayback capture of the section     must never come back
      desk_news        the same publisher outside the desk  must come back
      lookalike_prefix "not" + the blocked host             must come back
      lookalike_suffix the blocked host + ".redtest.invalid" must come back
      fc_listed_doc    a document the list names by id      must never come back

    path-word plants, on redtest-news.invalid (listed nowhere), the day before
      word_segment     a path word as a path segment        must never come back
      word_upper_case  another, in upper case               must never come back
      word_in_slug     a third, inside an article slug      must never come back
      word_wayback     a fourth, in a Wayback capture       must never come back
      word_news        the same host, no path word          must come back

desk_news, the look-alikes and word_news keep the rules honest in the other
direction: a rule that blocked them would be blocking by substring of the
host, or blocking a host for one page's path.

Every fact-check desk names a path word, so with the words on, the desk plants
are blocked twice and prove nothing about the desk rule. So the nonce probe
runs a second time with the words deleted, inside the same transaction: the
desk plants must still stay out (the desk rule alone), and the word plants must
come back (nothing but the words blocked them).

They are planted inside one `BEGIN IMMEDIATE` transaction that is **always**
rolled back, in a `finally`, so a failed assertion restores the index just as a
passing one does. Nothing is ever committed. `isolation_level=None` is set so
that Python's implicit-commit behaviour cannot slip a COMMIT in on its own.

Four probes, because each answers a question the others cannot:

1. **nonce** -- the nonce alone, swept over k from 1 to well past the corpus
   size. The candidate set is exactly the plants, so nothing but the rules can
   be doing the excluding. Run again without the path words, as above.
2. **bounded** -- a real term whose whole match set is counted first, with k set
   ten times larger. The LIMIT is provably not binding, so the plants are in the
   candidate set on merit and are excluded by the rules.
3. **parity** -- every row of the store, plants included: each rule's verdict
   in SQL (the predicates SEARCH_SQL uses) against its Python twin
   (`factcheckers.blocker`). Probe 4 judges real rows by the twin, so the twin
   has to be the SQL, both ways, on every row.
4. **whole corpus** -- k above the total row count against a near-universal
   term, streamed rather than materialised. Every returned row is also checked
   against the blocklist in Python: not one real document may come from a
   blocked source either.

Probe 4 runs `retrieve.SEARCH_SQL` directly rather than through
`retrieve.search`, which would build a list of ~1M `Hit` objects. It is the same
SQL string imported from the same module, so the contract under test is the real
one; only the row handling differs.

Every probe searches one word, and a one-word query is searched the same way by
`retrieve.search`'s default (a claim's 8 rarest words) and by `exact` (every
word): the probes test the rules under both.

The index is verified unchanged afterwards by four separate means: both row
counts, a `quick_check` on the temporal database, a ranking fingerprint -- the
top 50 doc_ids and scores of a fixed query, captured before planting and
compared after the rollback -- and the sha256 of both files, before and after.
The fingerprint catches a rollback that restored the row count but disturbed the
FTS segments; the hashes catch anything else. The connection's TEMP rules,
which the planted listed document and the deleted words changed, are counted
again after the rollback too.

Exit code 0 means both rules held. Anything else means they did not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import BIBLE, REPORTS  # noqa: E402
from scripts import factcheckers, retrieve  # noqa: E402

#: Mid-corpus rather than at either end, so that both "before" and "after" have
#: hundreds of thousands of real documents on their side of the line.
BEFORE = "2020-06-15"
PRIOR_DATE = "2020-06-14"
FUTURE_DATE = "2020-06-16"

#: Probe 2's term. Any indexed word does; this one matches tens of thousands of
#: documents, which is enough for the LIMIT to be meaningfully non-binding
#: without the probe costing what probe 3 costs.
BOUNDED_TERM = "hospital"

#: Probe 3's term. Chosen for breadth -- it has to reach most of the corpus for
#: "k larger than everything else" to mean anything.
CORPUS_TERM = "a"

#: The ranking fingerprint's query. Fixed, so the before and after are
#: comparable; unrelated to the plants, so planting cannot explain a difference.
FINGERPRINT_QUERY = "minister"
FINGERPRINT_DEPTH = 50

STAMP = "20200101000000"
MUST_RETURN = ("prior", "desk_news", "lookalike_prefix", "lookalike_suffix", "word_news")

#: The plants only the path words block, and the desk plants they also block.
WORD_PLANTS = ("word_segment", "word_upper_case", "word_in_slug", "word_wayback")
DESK_PLANTS = ("desk_section", "desk_upper_path", "desk_wayback")

#: The path words the word plants use, in plant order, where the list has them.
PLANT_WORDS = ("fact-check", "webqoof", "hoax-alert", "verificador")
WORD_HOST = "redtest-news.invalid"


def _count(con: sqlite3.Connection) -> tuple[int, int]:
    return (con.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0],
            con.execute("SELECT COUNT(*) FROM temporal.docs").fetchone()[0])


def _fingerprint(con: sqlite3.Connection) -> list[list[Any]]:
    """Top-N (doc_id, score) for a fixed query -- a ranking, not just a count."""
    hits = retrieve.search(FINGERPRINT_QUERY, BEFORE, k=FINGERPRINT_DEPTH, con=con)
    return [[h.doc_id, round(h.score, 6)] for h in hits]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def connect_rw(bm25_db: Path, temporal_db: Path) -> sqlite3.Connection:
    """Read-write, with the temporal database attached read-write too.

    `retrieve.connect` opens both read-only, which is right for retrieval and
    useless for planting. The ATTACH alias must stay `temporal`, because that is
    the name `retrieve.EVIDENCE_VIEWS` read, and the rules are installed exactly
    as `retrieve.connect` installs them.
    """
    con = sqlite3.connect(str(bm25_db), isolation_level=None)
    con.execute("ATTACH DATABASE ? AS temporal", (str(temporal_db),))
    con.row_factory = sqlite3.Row
    retrieve.install_rules(con)
    return con


def targets() -> tuple[str, tuple[str, str]]:
    """The blocked host and the publisher's desk the source plants use."""
    rows = retrieve.load_blocklist().sources
    whole = sorted(h for h, p in rows if p is None)
    desks = sorted((h, p) for h, p in rows if p is not None)
    if not whole or not desks:
        raise retrieve.RetrievalError("the blocklist needs a whole-host and a section entry "
                                      "for the red-test to test both")
    return whole[0], desks[0]


def plant_words() -> list[str]:
    """One path word per word plant: PLANT_WORDS where the list has them."""
    words = retrieve.load_blocklist().words
    chosen = [w for w in PLANT_WORDS if w in words] or list(words[:1])
    if any(w in WORD_HOST for w in words):
        raise retrieve.RetrievalError(f"a path word is in {WORD_HOST}: pick another plant host")
    return [chosen[i % len(chosen)] for i in range(len(WORD_PLANTS))]


def plants(nonce: str) -> dict[str, dict[str, Any]]:
    """name -> {published_at, source_domain, url, returns}."""
    host, (desk, section) = targets()
    w = plant_words()
    on = lambda d, u, when=PRIOR_DATE: {"published_at": when, "source_domain": d, "url": u}
    out = {
        "prior": on("redtest.invalid", f"https://redtest.invalid/{nonce}/prior"),
        "same_day": on("redtest.invalid", f"https://redtest.invalid/{nonce}/same", BEFORE),
        "future": on("redtest.invalid", f"https://redtest.invalid/{nonce}/future", FUTURE_DATE),
        "null": {"published_at": None, "source_domain": None, "url": None},
        "fc_domain": on(host, f"https://www.{host}/{nonce}"),
        "fc_subdomain": on(f"redtest.{host}", f"https://redtest.{host}/{nonce}"),
        "fc_upper_case": on(host.upper(), f"https://WWW.{host.upper()}/{nonce}"),
        "fc_wayback": on("web.archive.org",
                         f"https://web.archive.org/web/{STAMP}/https://www.{host}/{nonce}"),
        "desk_section": on(desk, f"https://www.{desk}{section}/{nonce}"),
        "desk_upper_path": on(desk, f"https://www.{desk}{section.upper()}/{nonce}"),
        "desk_wayback": on("web.archive.org",
                           f"https://web.archive.org/web/{STAMP}/https://www.{desk}{section}/{nonce}"),
        "desk_news": on(desk, f"https://www.{desk}/redtest-news/{nonce}"),
        "lookalike_prefix": on(f"not{host}", f"https://not{host}/{nonce}"),
        "lookalike_suffix": on(f"{host}.redtest.invalid", f"https://{host}.redtest.invalid/{nonce}"),
        "fc_listed_doc": on("redtest.invalid", f"https://redtest.invalid/{nonce}/listed"),
        "word_segment": on(WORD_HOST, f"https://{WORD_HOST}/{w[0]}/{nonce}"),
        "word_upper_case": on(WORD_HOST, f"https://{WORD_HOST}/{w[1].upper()}/{nonce}"),
        "word_in_slug": on(WORD_HOST, f"https://{WORD_HOST}/2020/06/viral-{w[2]}-{nonce}"),
        "word_wayback": on("web.archive.org",
                           f"https://web.archive.org/web/{STAMP}/https://{WORD_HOST}/{w[3]}/{nonce}"),
        "word_news": on(WORD_HOST, f"https://{WORD_HOST}/news/{nonce}"),
    }
    for name, p in out.items():
        p["returns"] = name in MUST_RETURN
    return out


def without_words(planted: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each plant must do with the path words deleted: the word plants come
    back, the desk plants still never do."""
    return {n: {**p, "returns": p["returns"] or n in WORD_PLANTS} for n, p in planted.items()}


def plant(con: sqlite3.Connection, nonce: str, planted: dict[str, dict[str, Any]]) -> None:
    """Insert the plants. The caller owns the transaction."""
    body = f"{nonce} {BOUNDED_TERM} {CORPUS_TERM} {FINGERPRINT_QUERY}"
    for name, p in planted.items():
        doc_id = f"{nonce}_{name}"
        con.execute("INSERT INTO docs_fts (doc_id, title, text) VALUES (?,?,?)",
                    (doc_id, f"{nonce} planted document", body))
        if p["published_at"] is not None:
            con.execute("INSERT INTO temporal.docs VALUES (?,?,?,?)",
                        (doc_id, p["published_at"], p["source_domain"], p["url"]))
    # a document the list names by id (a publisher's cited fact-check); the TEMP
    # table is inside the same transaction, so the rollback removes it too
    con.execute("INSERT INTO temp.blocked_doc VALUES (?)", (f"{nonce}_fc_listed_doc",))


def _judge(probe: str, k: int, ids: set[str], nonce: str, planted: dict[str, Any],
           latest: str | None, failures: list[str]) -> dict[str, Any]:
    leaked = sorted(n for n, p in planted.items() if not p["returns"] and f"{nonce}_{n}" in ids)
    missing = sorted(n for n, p in planted.items() if p["returns"] and f"{nonce}_{n}" not in ids)
    # Below the number that must come back, k itself may keep one out; past it,
    # every one of them could have fitted, so a missing one is a failure.
    if leaked:
        failures.append(f"{probe} k={k}: LEAKED {leaked}")
    if missing and k >= sum(p["returns"] for p in planted.values()):
        failures.append(f"{probe} k={k}: wrongly excluded {missing}")
    if latest is not None and latest >= BEFORE:
        failures.append(f"{probe} k={k}: returned a document dated {latest} >= {BEFORE}")
    return {"k": k, "leaked": leaked, "missing": missing, "latest_published_at": latest}


def probe_nonce(con: sqlite3.Connection, nonce: str, planted: dict[str, Any], ks: list[int],
                failures: list[str], label: str = "nonce") -> list[dict[str, Any]]:
    """Only the plants match, so only the rules can exclude anything."""
    out = []
    for k in ks:
        hits = retrieve.search(nonce, BEFORE, k=k, con=con)
        row = _judge(label, k, {h.doc_id for h in hits}, nonce, planted,
                     max((h.published_at for h in hits), default=None), failures)
        row["returned"] = len(hits)
        out.append(row)
    return out


def probe_bounded(con: sqlite3.Connection, nonce: str, planted: dict[str, Any],
                  failures: list[str]) -> dict[str, Any]:
    """Count the whole match set first, then ask for ten times that many."""
    total = con.execute("SELECT COUNT(*) FROM docs_fts WHERE docs_fts MATCH ?",
                        (f'"{BOUNDED_TERM}"',)).fetchone()[0]
    k = total * 10
    started = time.time()
    hits = retrieve.search(BOUNDED_TERM, BEFORE, k=k, con=con)
    row = _judge("bounded", k, {h.doc_id for h in hits}, nonce, planted,
                 max((h.published_at for h in hits), default=None), failures)
    row.update({"term": BOUNDED_TERM, "matches_ignoring_rules": total,
                "returned": len(hits), "seconds": round(time.time() - started, 1)})
    if len(hits) >= k:
        failures.append("bounded: the LIMIT was binding after all, so this probe "
                        "proves nothing")
    return row


#: Each rule's verdict on every row, by the predicates retrieve.SEARCH_SQL uses.
#: CASE, not AND: in a result column SQLite evaluates both sides of an AND, so
#: `wayback AND EXISTS (...)` tested every row's archived host against every
#: listed host, and the probe took 54 minutes (2026-10-29). In SEARCH_SQL's
#: WHERE the AND short-circuits; the verdicts are the same either way.
PARITY_SQL = f"""
SELECT  t.doc_id, t.source_domain, t.url,
        (EXISTS ({retrieve._LISTED.format(h="lower(t.source_domain)", p="t.path")})
         OR CASE WHEN t.wayback
                 THEN EXISTS ({retrieve._LISTED.format(h="t.archived_host", p="t.archived_path")})
                 ELSE 0 END)                                                      AS by_source,
        t.doc_id IN (SELECT doc_id FROM temp.blocked_doc)                           AS by_doc,
        EXISTS ({retrieve._WORD.format(t="t")})                                      AS by_word
FROM    temp.evidence AS t
"""


def probe_parity(con: sqlite3.Connection, failures: list[str]) -> dict[str, Any]:
    """SQL against its Python twin, rule by rule, on every row of the store.

    The twin reads the rules from the connection's TEMP tables, which hold the
    list plus the planted listed document."""
    listed = tuple((h, p) for h, p in con.execute("SELECT host, path FROM temp.blocked_source"))
    sources = factcheckers.blocker(retrieve.Blocklist(listed, (), ()))
    words = factcheckers.blocker(retrieve.Blocklist(
        (), (), tuple(w for (w,) in con.execute("SELECT word FROM temp.blocked_word"))))
    docs = {d for (d,) in con.execute("SELECT doc_id FROM temp.blocked_doc")}
    started = time.time()
    rows = differ = 0
    blocked = {"by_source": 0, "by_doc": 0, "by_word": 0}
    examples: list[dict[str, Any]] = []
    cur = con.execute(PARITY_SQL)
    for r in cur:
        rows += 1
        host, path = factcheckers.page(r["source_domain"], r["url"])
        twin = {"by_source": sources(r["doc_id"], host, path) is not None,
                "by_doc": r["doc_id"] in docs,
                "by_word": words(r["doc_id"], host, path) is not None}
        sql = {n: bool(r[n]) for n in twin}
        for n in blocked:
            blocked[n] += sql[n]
        if sql != twin:
            differ += 1
            if len(examples) < 5:
                examples.append({"doc_id": r["doc_id"], "sql": sql, "python": twin})
    cur.close()
    if differ:
        failures.append(f"parity: SQL and its Python twin disagree on {differ} rows")
    return {"rows": rows, "differ": differ, "sql_blocked": blocked, "examples": examples,
            "seconds": round(time.time() - started, 1)}


def probe_whole_corpus(con: sqlite3.Connection, nonce: str, planted: dict[str, Any], k: int,
                       failures: list[str]) -> dict[str, Any]:
    """k above the corpus size, streamed; every row also checked against the list."""
    why = factcheckers.blocker(retrieve.load_blocklist())
    names = {f"{nonce}_{n}" for n in planted}
    total = con.execute("SELECT COUNT(*) FROM docs_fts WHERE docs_fts MATCH ?",
                        (f'"{CORPUS_TERM}"',)).fetchone()[0]
    started = time.time()
    cur = con.execute(retrieve.SEARCH_SQL,
                      {"query": f'"{CORPUS_TERM}"', "before": BEFORE, "k": k})
    seen = blocked = 0
    latest: str | None = None
    found: set[str] = set()
    for r in cur:
        seen += 1
        if r["doc_id"] in names:
            found.add(r["doc_id"])
        elif why(r["doc_id"], *factcheckers.page(r["source_domain"], r["url"])):
            blocked += 1
        if latest is None or r["published_at"] > latest:
            latest = r["published_at"]
    cur.close()
    row = _judge("whole_corpus", k, found, nonce, planted, latest, failures)
    row.update({"term": CORPUS_TERM, "matches_ignoring_rules": total, "returned": seen,
                "blocked_sources_returned": blocked, "seconds": round(time.time() - started, 1)})
    if blocked:
        failures.append(f"whole_corpus: {blocked} real documents from blocked sources came back")
    if seen >= k:
        failures.append("whole_corpus: the LIMIT was binding, so k was not large "
                        "enough to return everything else")
    return row


def run(ks: list[int] | None = None, corpus_k: int = 2_000_000) -> dict[str, Any]:
    """Plant, probe, roll back, verify. Returns the report; never commits."""
    index = BIBLE / "index"
    bm25_db, temporal_db = index / "bm25.sqlite", index / "temporal.sqlite"
    for p in (bm25_db, temporal_db):
        if not p.is_file():
            raise retrieve.RetrievalError(
                f"index not found: {p}. Build it with "
                "`python scripts/bible.py --index`.")
    ks = ks or [1, 10, 100, 1_000, 100_000, 2_000_000]
    nonce = "redteam" + secrets.token_hex(6)
    planted = plants(nonce)
    failures: list[str] = []
    corpus_row: dict[str, Any] | None = None

    files = (bm25_db, temporal_db)
    before_hashes = {p.name: _sha256(p) for p in files}
    con = connect_rw(bm25_db, temporal_db)
    try:
        before_counts = _count(con)
        before_fp = _fingerprint(con)
        before_sizes = {p.name: p.stat().st_size for p in files}

        con.execute("BEGIN IMMEDIATE")
        try:
            plant(con, nonce, planted)
            planted_counts = _count(con)
            expected = (before_counts[0] + len(planted),
                        before_counts[1] + sum(p["published_at"] is not None
                                               for p in planted.values()))
            if planted_counts != expected:
                failures.append(
                    f"planting did not take: {before_counts} -> {planted_counts}")
            nonce_rows = probe_nonce(con, nonce, planted, ks, failures)
            bounded_row = probe_bounded(con, nonce, planted, failures)
            parity_row = probe_parity(con, failures)
            if corpus_k:
                corpus_row = probe_whole_corpus(con, nonce, planted, corpus_k, failures)
            # last, because it changes the rules: the desk rule on its own, and
            # the word plants held out by the words and nothing else
            con.execute("DELETE FROM temp.blocked_word")
            alone_rows = probe_nonce(con, nonce, without_words(planted), ks, failures,
                                     label="nonce_without_words")
        finally:
            con.execute("ROLLBACK")

        after_counts = _count(con)
        after_fp = _fingerprint(con)
        after_sizes = {p.name: p.stat().st_size for p in files}
        quick_check = con.execute("PRAGMA temporal.quick_check").fetchone()[0]
        rules = retrieve.load_blocklist()
        temp_rules = {"words": con.execute("SELECT count(*) FROM temp.blocked_word").fetchone()[0],
                      "doc_ids": con.execute("SELECT count(*) FROM temp.blocked_doc").fetchone()[0]}
    finally:
        con.close()
    after_hashes = {p.name: _sha256(p) for p in files}
    if temp_rules != {"words": len(rules.words), "doc_ids": len(rules.doc_ids)}:
        failures.append(f"rollback did not restore the connection's rules: {temp_rules}")

    if after_counts != before_counts:
        failures.append(f"rollback did not restore the row counts: "
                        f"{before_counts} -> {after_counts}")
    if after_fp != before_fp:
        failures.append("rollback restored the row count but changed the ranking")
    if quick_check != "ok":
        failures.append(f"temporal quick_check: {quick_check}")
    if after_hashes != before_hashes:
        failures.append("an index file is not byte-identical after the rollback")

    host, (desk, section) = targets()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "before": BEFORE,
        "rule": "published_at < before, and no blocked source or path word "
                "(configs/factcheckers.yaml)",
        "blocklist": {"rows": len(rules.sources), "doc_ids": len(rules.doc_ids),
                      "path_words": len(rules.words), "tested_host": host,
                      "tested_desk": f"{desk}{section}", "tested_words": plant_words(),
                      "word_host": WORD_HOST},
        "plants": {n: {k: v for k, v in p.items() if k != "url"} for n, p in planted.items()},
        "index_rows_before": {"docs_fts": before_counts[0],
                              "temporal_docs": before_counts[1]},
        "index_rows_after": {"docs_fts": after_counts[0],
                             "temporal_docs": after_counts[1]},
        "index_bytes_before": before_sizes,
        "index_bytes_after": after_sizes,
        "index_sha256_before": before_hashes,
        "index_sha256_after": after_hashes,
        "index_byte_identical": after_hashes == before_hashes,
        "ranking_fingerprint_unchanged": after_fp == before_fp,
        "fingerprint_depth": FINGERPRINT_DEPTH,
        "temporal_quick_check": quick_check,
        "connection_rules_after": temp_rules,
        "probe_nonce": nonce_rows,
        "probe_nonce_without_words": alone_rows,
        "probe_bounded": bounded_row,
        "probe_parity": parity_row,
        "probe_whole_corpus": corpus_row,
        "failures": failures,
        "passed": not failures,
    }


def render(r: dict[str, Any]) -> str:
    b = r["blocklist"]
    lines = [f"\nred-test  before {r['before']}  "
             f"({r['index_rows_before']['docs_fts']:,} indexed documents; blocklist "
             f"{b['rows']} rows, {b['doc_ids']} doc ids, {b.get('path_words', 0)} path words; "
             f"tested on {b['tested_host']}, {b['tested_desk']} and {b.get('word_host')} "
             f"with {', '.join(b.get('tested_words', []))})",
             "",
             f"  {'probe':<20}{'k':>10}{'returned':>11}  {'leaked':>7}{'missing':>9}  newest"]

    def _row(label: str, row: dict[str, Any], extra: str = "") -> str:
        return (f"  {label:<20}{row['k']:>10,}{row['returned']:>11,}  "
                f"{len(row['leaked']):>7}{len(row['missing']):>9}  "
                f"{row['latest_published_at']}{extra}")

    for row in r["probe_nonce"]:
        lines.append(_row("nonce", row))
    for row in r.get("probe_nonce_without_words") or []:
        lines.append(_row("nonce, no words", row))
    for name, label in (("probe_bounded", "bounded"),
                        ("probe_whole_corpus", "whole corpus")):
        row = r[name]
        if row:
            extra = f"   [{row['term']}, {row['seconds']}s"
            if "blocked_sources_returned" in row:
                extra += f", blocked sources returned {row['blocked_sources_returned']}"
            lines.append(_row(label, row, extra + "]"))
    par = r.get("probe_parity")
    if par:
        lines.append(f"  parity: {par['rows']:,} rows, SQL and Python twin differ on "
                     f"{par['differ']} (SQL blocks {par['sql_blocked']}; {par['seconds']}s)")
    restored = r["index_rows_after"] == r["index_rows_before"]
    lines += ["",
              f"  rows restored: {r['index_rows_after']['docs_fts']:,} / "
              f"{r['index_rows_after']['temporal_docs']:,}"
              f"  ({'unchanged' if restored else 'CHANGED'})",
              f"  ranking fingerprint (top {r['fingerprint_depth']}): "
              f"{'unchanged' if r['ranking_fingerprint_unchanged'] else 'CHANGED'}",
              f"  temporal quick_check: {r['temporal_quick_check']}",
              f"  index files byte-identical (sha256): "
              f"{'yes' if r['index_byte_identical'] else 'NO'}"]
    if r["passed"]:
        lines.append("\n  PASS - nothing dated on or after `before`, undated, from a blocked "
                     "source or with a path word surfaced at any k; the look-alikes all came "
                     "back; without the words, the desks still held.")
    else:
        lines.append("\n  FAIL")
        lines += [f"    {f}" for f in r["failures"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-k", type=int, default=2_000_000,
                   help="k for the whole-corpus probe; must exceed the row count")
    p.add_argument("--skip-corpus", action="store_true",
                   help="skip the whole-corpus probe -- it is the slow one")
    args = p.parse_args(argv)

    r = run(corpus_k=0 if args.skip_corpus else args.corpus_k)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "redtest.json"
    out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(render(r))
    print(f"  wrote {out}")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
