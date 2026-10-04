#!/usr/bin/env python
"""Red-team the time filter against the REAL evidence-store index.

    python scripts/redtest.py

`tests/test_bible.py` already plants a future-dated and a null-dated document
and demands neither surfaces -- but it does so in a four-row temporary index. A
filter that holds on four rows proves nothing about a filter that has to hold on
1.4 million: the query planner picks different plans at different table sizes, a
`LIMIT` that is larger than a toy corpus is smaller than a real one, and an
`ORDER BY` over four rows never spills to a temp b-tree. So this runs the same
predicate against the store as built.

Three documents are planted directly into `bm25.sqlite` and `temporal.sqlite`:

    plant_prior    published the day BEFORE as_of   -- must always come back
    plant_future   published the day AFTER  as_of   -- must never come back
    plant_null     absent from the temporal table   -- must never come back

They are planted inside one `BEGIN IMMEDIATE` transaction that is **always**
rolled back, in a `finally`, so a failed assertion restores the index just as a
passing one does. Nothing is ever committed. `isolation_level=None` is set so
that Python's implicit-commit behaviour cannot slip a COMMIT in on its own.

Three probes, because each answers a question the others cannot:

1. **nonce** -- a term that occurs in the three plants and nowhere else, swept
   over k from 1 to well past the corpus size. The candidate set is exactly the
   three plants, so nothing but the date predicate can be doing the excluding.
2. **bounded** -- a real term whose whole match set is counted first, with k set
   ten times larger. The LIMIT is provably not binding, so the plants are in the
   candidate set on merit and are excluded on their dates.
3. **whole corpus** -- k above the total row count against a near-universal
   term, streamed rather than materialised. This is the "k large enough to
   return everything else" case, and it is the one that costs real time.

Probe 3 runs `retrieve.SEARCH_SQL` directly rather than through
`retrieve.search`, which would build a list of ~1M `Hit` objects. It is the same
SQL string imported from the same module, so the contract under test is the real
one; only the row handling differs.

The index is verified unchanged afterwards by three separate means: both row
counts, a `quick_check` on the temporal database, and a ranking fingerprint --
the top 50 doc_ids and scores of a fixed query, captured before planting and
compared after the rollback. The fingerprint is what catches a rollback that
restored the row count but disturbed the FTS segments.

Exit code 0 means the time filter held. Anything else means it did not.
"""

from __future__ import annotations

import argparse
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
from scripts import retrieve  # noqa: E402

#: Mid-corpus rather than at either end, so that both "before" and "after" have
#: hundreds of thousands of real documents on their side of the line.
AS_OF = "2020-06-15"
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


def _count(con: sqlite3.Connection) -> tuple[int, int]:
    return (con.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0],
            con.execute("SELECT COUNT(*) FROM temporal.docs").fetchone()[0])


def _fingerprint(con: sqlite3.Connection) -> list[list[Any]]:
    """Top-N (doc_id, score) for a fixed query -- a ranking, not just a count."""
    hits = retrieve.search(FINGERPRINT_QUERY, AS_OF, k=FINGERPRINT_DEPTH, con=con)
    return [[h.doc_id, round(h.score, 6)] for h in hits]


def connect_rw(bm25_db: Path, temporal_db: Path) -> sqlite3.Connection:
    """Read-write, with the temporal database attached read-write too.

    `retrieve.connect` opens both read-only, which is right for retrieval and
    useless for planting. The ATTACH alias must stay `temporal`, because that is
    the name `retrieve.SEARCH_SQL` joins against.
    """
    con = sqlite3.connect(str(bm25_db), isolation_level=None)
    con.execute("ATTACH DATABASE ? AS temporal", (str(temporal_db),))
    con.row_factory = sqlite3.Row
    return con


def plant(con: sqlite3.Connection, nonce: str) -> None:
    """Insert the three plants. The caller owns the transaction."""
    body = f"{nonce} {BOUNDED_TERM} {CORPUS_TERM} {FINGERPRINT_QUERY}"
    for doc_id, published in ((f"{nonce}_prior", PRIOR_DATE),
                              (f"{nonce}_future", FUTURE_DATE),
                              (f"{nonce}_null", None)):
        con.execute("INSERT INTO docs_fts (doc_id, title, text) VALUES (?,?,?)",
                    (doc_id, f"{nonce} planted document", body))
        if published is not None:
            con.execute("INSERT INTO temporal.docs VALUES (?,?,?,?)",
                        (doc_id, published, "redtest.invalid",
                         f"https://redtest.invalid/{doc_id}"))


def _judge(probe: str, k: int, ids: set[str], nonce: str,
           latest: str | None, failures: list[str]) -> dict[str, Any]:
    prior, future, null = (f"{nonce}_prior", f"{nonce}_future", f"{nonce}_null")
    leaked = sorted({future, null} & ids)
    # k < 3 is the one case where the prior plant may legitimately be absent: a
    # real document can outrank it. Its absence is only a failure once k is
    # large enough that every match could have fitted.
    tolerate_missing = k < 3
    if leaked:
        failures.append(f"{probe} k={k}: LEAKED {leaked}")
    if prior not in ids and not tolerate_missing:
        failures.append(
            f"{probe} k={k}: the day-earlier document did not come back")
    if latest is not None and latest > AS_OF:
        failures.append(
            f"{probe} k={k}: returned a document dated {latest} > {AS_OF}")
    return {"k": k, "future_returned": future in ids, "null_returned": null in ids,
            "prior_returned": prior in ids, "latest_published_at": latest}


def probe_nonce(con: sqlite3.Connection, nonce: str, ks: list[int],
                failures: list[str]) -> list[dict[str, Any]]:
    """Only the plants match, so only the date predicate can exclude anything."""
    out = []
    for k in ks:
        hits = retrieve.search(nonce, AS_OF, k=k, con=con)
        ids = {h.doc_id for h in hits}
        latest = max((h.published_at for h in hits), default=None)
        row = _judge("nonce", k, ids, nonce, latest, failures)
        row["returned"] = len(hits)
        out.append(row)
    return out


def probe_bounded(con: sqlite3.Connection, nonce: str,
                  failures: list[str]) -> dict[str, Any]:
    """Count the whole match set first, then ask for ten times that many."""
    total = con.execute("SELECT COUNT(*) FROM docs_fts WHERE docs_fts MATCH ?",
                        (f'"{BOUNDED_TERM}"',)).fetchone()[0]
    k = total * 10
    started = time.time()
    hits = retrieve.search(BOUNDED_TERM, AS_OF, k=k, con=con)
    ids = {h.doc_id for h in hits}
    latest = max((h.published_at for h in hits), default=None)
    row = _judge("bounded", k, ids, nonce, latest, failures)
    row.update({"term": BOUNDED_TERM, "matches_ignoring_date": total,
                "returned": len(hits), "seconds": round(time.time() - started, 1)})
    if len(hits) >= k:
        failures.append("bounded: the LIMIT was binding after all, so this probe "
                        "proves nothing")
    return row


def probe_whole_corpus(con: sqlite3.Connection, nonce: str, k: int,
                       failures: list[str]) -> dict[str, Any]:
    """k above the corpus size, streamed so a million rows cost no memory."""
    prior, future, null = (f"{nonce}_prior", f"{nonce}_future", f"{nonce}_null")
    total = con.execute("SELECT COUNT(*) FROM docs_fts WHERE docs_fts MATCH ?",
                        (f'"{CORPUS_TERM}"',)).fetchone()[0]
    started = time.time()
    cur = con.execute(retrieve.SEARCH_SQL,
                      {"query": f'"{CORPUS_TERM}"', "as_of": AS_OF, "k": k})
    seen = 0
    latest: str | None = None
    found = {prior: False, future: False, null: False}
    for r in cur:
        seen += 1
        doc_id, published = r["doc_id"], r["published_at"]
        if doc_id in found:
            found[doc_id] = True
        if latest is None or published > latest:
            latest = published
    cur.close()
    row = _judge("whole_corpus", k,
                 {d for d, hit in found.items() if hit}, nonce, latest, failures)
    row.update({"term": CORPUS_TERM, "matches_ignoring_date": total,
                "returned": seen, "seconds": round(time.time() - started, 1)})
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
    failures: list[str] = []
    corpus_row: dict[str, Any] | None = None

    con = connect_rw(bm25_db, temporal_db)
    try:
        before_counts = _count(con)
        before_fp = _fingerprint(con)
        before_sizes = {p.name: p.stat().st_size for p in (bm25_db, temporal_db)}

        con.execute("BEGIN IMMEDIATE")
        try:
            plant(con, nonce)
            planted_counts = _count(con)
            if planted_counts != (before_counts[0] + 3, before_counts[1] + 2):
                failures.append(
                    f"planting did not take: {before_counts} -> {planted_counts}")
            nonce_rows = probe_nonce(con, nonce, ks, failures)
            bounded_row = probe_bounded(con, nonce, failures)
            if corpus_k:
                corpus_row = probe_whole_corpus(con, nonce, corpus_k, failures)
        finally:
            con.execute("ROLLBACK")

        after_counts = _count(con)
        after_fp = _fingerprint(con)
        after_sizes = {p.name: p.stat().st_size for p in (bm25_db, temporal_db)}
        quick_check = con.execute("PRAGMA temporal.quick_check").fetchone()[0]
    finally:
        con.close()

    if after_counts != before_counts:
        failures.append(f"rollback did not restore the row counts: "
                        f"{before_counts} -> {after_counts}")
    if after_fp != before_fp:
        failures.append("rollback restored the row count but changed the ranking")
    if quick_check != "ok":
        failures.append(f"temporal quick_check: {quick_check}")

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "as_of": AS_OF,
        "plants": {"prior": PRIOR_DATE, "future": FUTURE_DATE, "null": None},
        "index_rows_before": {"docs_fts": before_counts[0],
                              "temporal_docs": before_counts[1]},
        "index_rows_after": {"docs_fts": after_counts[0],
                             "temporal_docs": after_counts[1]},
        "index_bytes_before": before_sizes,
        "index_bytes_after": after_sizes,
        "ranking_fingerprint_unchanged": after_fp == before_fp,
        "fingerprint_depth": FINGERPRINT_DEPTH,
        "temporal_quick_check": quick_check,
        "probe_nonce": nonce_rows,
        "probe_bounded": bounded_row,
        "probe_whole_corpus": corpus_row,
        "failures": failures,
        "passed": not failures,
    }


def render(r: dict[str, Any]) -> str:
    lines = [f"\nred-test  as_of {r['as_of']}  "
             f"({r['index_rows_before']['docs_fts']:,} indexed documents)",
             "",
             f"  {'probe':<14}{'k':>10}{'returned':>11}  "
             f"{'future':>7}{'null':>6}{'prior':>7}  newest"]

    def _row(label: str, row: dict[str, Any], extra: str = "") -> str:
        return (f"  {label:<14}{row['k']:>10,}{row['returned']:>11,}  "
                f"{str(row['future_returned']):>7}{str(row['null_returned']):>6}"
                f"{str(row['prior_returned']):>7}  "
                f"{row['latest_published_at']}{extra}")

    for row in r["probe_nonce"]:
        lines.append(_row("nonce", row))
    for name, label in (("probe_bounded", "bounded"),
                        ("probe_whole_corpus", "whole corpus")):
        row = r[name]
        if row:
            lines.append(_row(label, row,
                              f"   [{row['term']}, {row['seconds']}s]"))
    restored = r["index_rows_after"] == r["index_rows_before"]
    lines += ["",
              f"  rows restored: {r['index_rows_after']['docs_fts']:,} / "
              f"{r['index_rows_after']['temporal_docs']:,}"
              f"  ({'unchanged' if restored else 'CHANGED'})",
              f"  ranking fingerprint (top {r['fingerprint_depth']}): "
              f"{'unchanged' if r['ranking_fingerprint_unchanged'] else 'CHANGED'}",
              f"  temporal quick_check: {r['temporal_quick_check']}"]
    if r["passed"]:
        lines.append(
            "\n  PASS - no future-dated or undated document surfaced at any k.")
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
