#!/usr/bin/env python
"""Retrieval speed on the real evidence store: profiled first, then options timed.

    python scripts/retrieval_bench.py identity          # sha256 of both index files
    python scripts/retrieval_bench.py profile           # where a query's time goes (3 queries)
    python scripts/retrieval_bench.py df                # exact df of every token the 200 queries use
    python scripts/retrieval_bench.py run --option E1   # one option over the 200 queries; resumable
    python scripts/retrieval_bench.py check --option L  # an option's setup on query 1; writes nothing
    python scripts/retrieval_bench.py session           # L and the four L+P options in one process; resumable
    python scripts/retrieval_bench.py default           # retrieve.search's default, its df counts timed; resumable
    python scripts/retrieval_bench.py compare           # data/reports/retrieval_bench.{md,json}

Measurement only. The 200 queries are the M4FC evidence run's
(data/interim/m4fc_audit/evidence.jsonl): the same claims and the same as-of
dates, so every option is timed on the queries that averaged 93.5 s, and the
run's own ledger is a fingerprint of the current results (each hit's domain and
date, in rank order, and the top score). Every connection opens both index
files read-only (`connect_ro`). Ledgers go to data/interim/retrieval/
(ignored); the tracked report holds aggregates only, never a record id: M4FC's
ids are article slugs that state the verdict.

The options were fixed before any overlap was seen:

    E0   retrieve.SEARCH_SQL as it stood on 2026-10-28           the baseline
    E1   ORDER BY f.rank: FTS5 sorts, the join stops at k        exact
    E1+  E1 with a 256 MiB page cache and a 16 GiB mmap          exact
    P1   E1 without the terms FTS5 scores at idf 1e-6            near-exact
    P2   E1 with only the terms in at most 1% of documents       changes results
    P3   E1 with the claim's entities and its 5 rarest terms     changes results
    P4   E1 with the claim's 8 rarest terms                      changes results
    L    retrieve.SEARCH_SQL as adopted: E1, strictly before,    the leak rules'
         and no fact-checker page                                own effect
    L+P2, L+P3, L+P4   L with P2's, P3's or P4's terms          production candidates

and one came later, after P4's pass had been seen (the person's request,
2026-10-29):

    P4b  P4 without the terms FTS5 scores at idf 1e-6            near P4
    L+P4b  L with P4b's terms                                    a production candidate

P4's slow tail is its claims with fewer than 8 distinct words outside half the
store: P4 then keeps words in half the store or more. P4b drops them, as P1
does.

Every option bounds the 200 queries by M4FC's fact-check date
(configs/as_of.yaml): inclusively for E and P, as the run did; strictly for L
and L+P.

`session` times the production candidates against each other (the person's
request, 2026-10-28): L, L+P2, L+P3, L+P4 and L+P4b in one process,
interleaved. Each query runs all five, in row i mod 10 of a Williams design (an
odd number of options takes the square and its mirror), so every option runs in
each position for 40 queries and directly after every other option 40 times.
Passes run hours apart differ by what the OS has cached
(E1's median was 2.58 s in one session and 9.6 s in another); interleaved, the
options share each moment of the session, and the order effect within a query
is measured rather than hidden: the first option to run a query reads its
terms' posting lists cold, the others may find them cached. `compare` reports
every option against the session's own L.

`default` times L+P4b as wired into retrieve.search (the person's decision,
2026-10-29). The session read every word's document frequency from
token_df.json, so its times leave out the counting production does per query;
this pass runs what search runs, in its two steps -- retrieve.query_terms (each
distinct word counted, the store's size, the pruning), then the statement --
and `compare` checks it against the session's L+P4b: the same terms and the
same top 20, ids, order and scores.

FTS5's bm25 clamps a term's idf to 1e-6 once it occurs in half the documents
or more, so dropping such a term moves a score by at most 1e-6 * (k1 + 1) =
2.2e-6 per term: that is P1's "near-exact". Against P4, P4b moves scores that
little, but a document that matched only through a dropped term no longer
matches: where fewer than k documents match its other terms, P4b returns fewer
than k. Every other P option changes which
documents come back, and is measured as such: overlap@20 and overlap@5 with the
current results, top-1 agreement, and -- because sameness is not quality -- how
many of the claim's named entities the top 5 contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT, REPORTS  # noqa: E402
from scripts import retrieve  # noqa: E402

OUT = INTERIM / "retrieval"
IDENTITY = OUT / "index_identity.json"
PROFILE = OUT / "profile.jsonl"
DF = OUT / "token_df.json"
REPORT_JSON = REPORTS / "retrieval_bench.json"
REPORT_MD = REPORTS / "retrieval_bench.md"

K = 20
TOP = 5
PROFILE_QUANTILES = (0.10, 0.50, 0.90)
P2_SHARE = 0.01
P3_RAREST = 5
P4_RAREST = 8
CACHE_KIB = 262_144            # 256 MiB, as a negative PRAGMA cache_size
MMAP_BYTES = 16 * 1024 ** 3

OPTIONS = ("E0", "E1", "E1+", "P1", "P2", "P3", "P4", "P4b", "L", "L+P2", "L+P3", "L+P4", "L+P4b")

#: The options that query with every word of the claim, as the evidence run
#: did. Every other option prunes terms, and needs the df table to do it.
EVERY_WORD = ("E0", "E1", "E1+", "L")

#: The production candidates, timed together by `session`.
SESSION = ("L", "L+P2", "L+P3", "L+P4", "L+P4b")
SESSION_LEDGER = OUT / "session.jsonl"

#: retrieve.search's default as wired, timed by `default`.
DEFAULT_LEDGER = OUT / "default.jsonl"

#: E0: retrieve.SEARCH_SQL as it stood on 2026-10-28, frozen here so that a
#: later change to the retrieval SQL cannot move the baseline it is measured
#: against.
E0_SQL = """
SELECT  f.doc_id     AS doc_id,
        f.title      AS title,
        t.published_at,
        t.source_domain,
        t.url,
        bm25(docs_fts) AS score
FROM    docs_fts AS f
JOIN    temporal.docs AS t ON t.doc_id = f.doc_id
WHERE   docs_fts MATCH :query
  AND   t.published_at IS NOT NULL
  AND   t.published_at <= :as_of
ORDER BY score
LIMIT   :k
"""

#: E1: the current contract with FTS5's own rank order. FTS5 consumes
#: `ORDER BY rank` (EXPLAIN shows INDEX 32 and no temp b-tree), so rows stream
#: out best first and the join and the date predicate run only until k pass.
#: CROSS JOIN pins docs_fts as the outer loop, which that streaming needs.
E1_SQL = """
SELECT  f.doc_id     AS doc_id,
        f.title      AS title,
        t.published_at,
        t.source_domain,
        t.url,
        f.rank       AS score
FROM    docs_fts AS f
CROSS JOIN temporal.docs AS t
WHERE   docs_fts MATCH :query
  AND   t.doc_id = f.doc_id
  AND   t.published_at IS NOT NULL
  AND   t.published_at <= :as_of
ORDER BY f.rank
LIMIT   :k
"""

#: The profile's parts, each one statement that stops where its part ends.
PART_SQL = {
    "match": "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH :query",
    "rank": ("SELECT rowid, bm25(docs_fts) AS s FROM docs_fts WHERE docs_fts MATCH :query "
             "ORDER BY s LIMIT :k"),
    "content": "SELECT count(f.doc_id) FROM docs_fts AS f WHERE docs_fts MATCH :query",
    "join": ("SELECT count(*) FROM docs_fts AS f JOIN temporal.docs AS t ON t.doc_id = f.doc_id "
             "WHERE docs_fts MATCH :query AND t.published_at IS NOT NULL "
             "AND t.published_at <= :as_of"),
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --------------------------------------------------------------------------
# the queries and their terms
# --------------------------------------------------------------------------


def load_queries() -> list[dict[str, Any]]:
    """The evidence run's 200 queries, in its order, with the claim from raw M4FC."""
    from scripts import m4fc_audit

    claims = {r["image_path"]: r["claim"] for r in m4fc_audit.raw_rows()}
    rows = read_jsonl(m4fc_audit.EVIDENCE)
    if len(rows) != m4fc_audit.EVIDENCE_N:
        raise SystemExit(f"{m4fc_audit.EVIDENCE} holds {len(rows)} queries, not "
                         f"{m4fc_audit.EVIDENCE_N}: finish the evidence run first")
    return [{"record_id": r["record_id"], "claim": claims[r["record_id"]], "as_of": r["as_of"],
             "india": r["india"], "entities": r["entities"], "ledger": r} for r in rows]


#: A query's words, their OR, and the pruning helpers are retrieve's own, so
#: every option chooses its terms with the functions production runs.
tokens, match_expr = retrieve.words, retrieve.match_expr
distinct, rarest, clamped = retrieve.distinct, retrieve.rarest, retrieve.clamped


def pruning(option: str) -> str | None:
    """The P rule an option selects its terms by: None for every word, "P2" for P2 and L+P2."""
    if option in EVERY_WORD:
        return None
    return option[2:] if option.startswith("L+") else option


def with_rules(option: str) -> bool:
    """L and L+P run retrieve.SEARCH_SQL on retrieve.connect(): both leak rules on."""
    return option == "L" or option.startswith("L+")


def select_terms(option: str, claim: str, entities: Sequence[dict[str, str]],
                 df: dict[str, int] | None, n_docs: int) -> tuple[list[str], bool]:
    """(terms, fell_back) for an option. EVERY_WORD options keep every word, repeats included."""
    words = tokens(claim)
    rule = pruning(option)
    if rule is None:
        return words, False
    if df is None:
        raise ValueError(f"{option} prunes terms by document frequency: run `df` first")
    option = rule
    if option == "P1":   # repeats kept, so it stays near-exact
        kept = [t for t in words if not clamped(t, df, n_docs)]
    elif option == "P2":
        kept = [t for t in distinct(words) if df[t.lower()] <= P2_SHARE * n_docs]
    elif option == "P3":
        ent = distinct(t for e in entities for t in tokens(e["text"]))
        keys = {t.lower() for t in ent}
        kept = distinct(ent + rarest([t for t in words if t.lower() not in keys], df, P3_RAREST))
    elif option == "P4":
        kept = rarest(words, df, P4_RAREST)
    elif option == "P4b":  # retrieve's default: P4's 8 less any clamped, with this fallback
        return retrieve.prune(words, df, n_docs)
    else:
        raise ValueError(option)
    if kept:
        return kept, False
    return rarest(words, df, 1), True


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


def index_files() -> list[Path]:
    return [retrieve.BM25_DB, retrieve.TEMPORAL_DB]


def index_stamp() -> dict[str, Any]:
    """Cheap identity: size and mtime. `identity` adds the sha256."""
    return {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in index_files()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def identity() -> int:
    started = time.perf_counter()
    now = {p.name: {**index_stamp()[p.name], "sha256": sha256(p)} for p in index_files()}
    took = round(time.perf_counter() - started, 1)
    if IDENTITY.is_file():
        base = json.loads(IDENTITY.read_text(encoding="utf-8"))["files"]
        same = all(base[n]["sha256"] == now[n]["sha256"] and base[n]["bytes"] == now[n]["bytes"]
                   for n in now)
        for n, v in now.items():
            print(f"  {n:<16} {v['bytes']:>15,} bytes  sha256 {v['sha256'][:16]}…  "
                  f"{'unchanged' if base[n]['sha256'] == v['sha256'] else 'CHANGED'}")
        print(f"  {'byte-identical to' if same else 'DIFFERS from'} the identity of "
              f"{json.loads(IDENTITY.read_text(encoding='utf-8'))['taken']} ({took} s)")
        return 0 if same else 1
    write_json(IDENTITY, {"taken": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": took, "files": now})
    for n, v in now.items():
        print(f"  {n:<16} {v['bytes']:>15,} bytes  sha256 {v['sha256']}")
    print(f"  wrote {IDENTITY.relative_to(PROJECT_ROOT).as_posix()} ({took} s)")
    return 0


def connect_ro():
    """Both index files read-only, attached as retrieve.connect() attached them on
    2026-10-28. Not retrieve.connect() itself: that now installs the fact-checker
    blocklist, and E0 and E1 measure the contract as it stood before it."""
    import sqlite3

    con = sqlite3.connect(f"file:{retrieve.BM25_DB}?mode=ro", uri=True)
    con.execute("ATTACH DATABASE ? AS temporal", (f"file:{retrieve.TEMPORAL_DB}?mode=ro",))
    con.row_factory = sqlite3.Row
    return con


def explain(con, sql: str, params: dict[str, Any]) -> list[str]:
    return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql, params)]


def timed(con, sql: str, params: dict[str, Any]) -> tuple[float, list[Any]]:
    start = time.perf_counter()
    rows = con.execute(sql, params).fetchall()
    return time.perf_counter() - start, rows


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def profile_picks(queries: Sequence[dict[str, Any]]) -> list[int]:
    """The queries at the 10th, 50th and 90th percentile of the run's search time."""
    order = sorted(range(len(queries)), key=lambda i: queries[i]["ledger"]["search_seconds"])
    return [order[round(q * (len(order) - 1))] for q in PROFILE_QUANTILES]


def profile(limit: int | None) -> int:
    queries = load_queries()
    picks = profile_picks(queries)[:limit]
    done = {r["record_id"] for r in read_jsonl(PROFILE)}
    con = connect_ro()
    probe = {"query": match_expr(tokens(queries[picks[0]]["claim"])),
             "as_of": queries[picks[0]]["as_of"], "k": K}
    plans = {"E0": explain(con, E0_SQL, probe), "E1": explain(con, E1_SQL, probe)}
    for name, plan in plans.items():
        print(f"  plan {name}: {' | '.join(plan)}")
    if any("TEMP B-TREE" in step for step in plans["E1"]):
        raise SystemExit("E1's plan sorts in a temp b-tree: FTS5 did not consume ORDER BY rank")
    try:
        with PROFILE.open("a", encoding="utf-8") as ledger:
            for i in picks:
                q = queries[i]
                if q["record_id"] in done:
                    continue
                p = {"query": match_expr(tokens(q["claim"])), "as_of": q["as_of"], "k": K}
                row: dict[str, Any] = {"record_id": q["record_id"], "position": i,
                                       "run_seconds": q["ledger"]["search_seconds"],
                                       "terms": len(tokens(q["claim"])), "plans": plans}
                for part in ("match", "rank", "content", "join"):
                    took, got = timed(con, PART_SQL[part], p)
                    row[f"{part}_seconds"] = round(took, 3)
                    if part != "rank":
                        row[f"{part}_rows"] = got[0][0]
                    print(f"    {i:>3} {part:<8} {took:7.1f} s", flush=True)
                # E0 as retrieve.search runs it, with its Python parts timed apart
                t0 = time.perf_counter()
                expr = retrieve._sanitise(q["claim"])
                t1 = time.perf_counter()
                rows = con.execute(E0_SQL, {**p, "query": expr}).fetchall()
                t2 = time.perf_counter()
                hits = [retrieve.Hit(r["doc_id"], r["title"], r["published_at"], r["source_domain"],
                                     r["url"], r["score"]) for r in rows]
                t3 = time.perf_counter()
                row.update({"e0_sql_seconds": round(t2 - t1, 3),
                            "e0_python_seconds": round((t1 - t0) + (t3 - t2), 6),
                            "e0_hits": [[h.doc_id, h.score] for h in hits]})
                print(f"    {i:>3} E0 full  {t2 - t1:7.1f} s", flush=True)
                took, e1 = timed(con, E1_SQL, p)
                row.update({"e1_seconds": round(took, 3),
                            "e1_hits": [[r["doc_id"], r["score"]] for r in e1]})
                print(f"    {i:>3} E1       {took:7.1f} s", flush=True)
                took, _ = timed(con, PART_SQL["match"], p)
                row["match_again_seconds"] = round(took, 3)
                ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
    finally:
        con.close()
    return 0


# --------------------------------------------------------------------------
# document frequency
# --------------------------------------------------------------------------


def document_frequencies() -> int:
    """Exact df for every distinct token the 200 queries use: count(*) per token."""
    queries = load_queries()
    vocab = sorted({t.lower() for q in queries for t in tokens(q["claim"])} |
                   {t.lower() for q in queries for e in q["entities"] for t in tokens(e["text"])})
    con = connect_ro()
    started = time.perf_counter()
    try:
        n_docs = con.execute("SELECT count(*) FROM temporal.docs").fetchone()[0]
        df, slow = {}, []
        for n, t in enumerate(vocab, 1):
            s = time.perf_counter()
            df[t] = con.execute(PART_SQL["match"], {"query": f'"{t}"'}).fetchone()[0]
            took = time.perf_counter() - s
            if took > 1:
                slow.append([t, round(took, 2)])
            if n % 250 == 0:
                print(f"    {n:,} of {len(vocab):,} tokens, {time.perf_counter() - started:.0f} s",
                      flush=True)
    finally:
        con.close()
    took = round(time.perf_counter() - started, 1)
    write_json(DF, {"index": index_stamp(), "n_docs": n_docs, "tokens": len(vocab), "seconds": took,
                    "over_1s": slow, "df": df})
    print(f"  {len(vocab):,} tokens in {took} s; wrote {DF.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


def load_df() -> tuple[dict[str, int], int]:
    d = json.loads(DF.read_text(encoding="utf-8"))
    if d["index"] != index_stamp():
        raise SystemExit(f"{DF} was computed on another index; run `df` again")
    return d["df"], d["n_docs"]


# --------------------------------------------------------------------------
# a pass over the 200 queries
# --------------------------------------------------------------------------


def ledger_path(option: str) -> Path:
    return OUT / f"run_{option.replace('+', 'plus')}.jsonl"


def sql_for(option: str) -> str:
    if option == "E0":
        return E0_SQL
    return retrieve.SEARCH_SQL if with_rules(option) else E1_SQL


def params_for(q: dict[str, Any], terms: Sequence[str]) -> dict[str, Any]:
    """One set for every option: E and P bound inclusively (:as_of), L strictly (:before)."""
    return {"query": match_expr(terms), "as_of": q["as_of"], "before": q["as_of"], "k": K}


def open_for(option: str):
    """The connection an option runs on: L and L+P through retrieve.connect(), with the rules."""
    if with_rules(option):
        return retrieve.connect()
    con = connect_ro()
    if option == "E1+":
        con.execute(f"PRAGMA cache_size = -{CACHE_KIB}")
        con.execute(f"PRAGMA mmap_size = {MMAP_BYTES}")
    return con


def rules_config(rules: retrieve.Blocklist) -> dict[str, Any]:
    return {"sources": [list(r) for r in rules.sources], "doc_ids": list(rules.doc_ids),
            "words": list(rules.words)}


def check(option: str) -> int:
    """What a pass does before its first timed query, on the first query: terms,
    connection, and the statement planned with its parameters bound. Writes nothing."""
    queries = load_queries()
    df, n_docs = (None, 0) if option in EVERY_WORD else load_df()
    q = queries[0]
    terms, _ = select_terms(option, q["claim"], q["entities"], df, n_docs)
    con = open_for(option)
    try:
        plan = explain(con, sql_for(option), params_for(q, terms))
    finally:
        con.close()
    if option != "E0" and any("TEMP B-TREE" in step for step in plan):
        raise SystemExit(f"{option}: the plan sorts in a temp b-tree")
    print(f"  {option}: ok -- {len(terms)} terms; {' | '.join(plan[:2])}")
    return 0


def run(option: str, limit: int | None) -> int:
    queries = load_queries()
    df, n_docs = (None, 0) if option in EVERY_WORD else load_df()
    sql = sql_for(option)
    config = {"option": option, "k": K, "sql": sql, "index": index_stamp(),
              "connection": ({"cache_size": -CACHE_KIB, "mmap_size": MMAP_BYTES}
                             if option == "E1+" else "default")}
    if with_rules(option):
        config["blocklist"] = rules_config(retrieve.load_blocklist())
    path = ledger_path(option)
    cfg = path.with_suffix(".config.json")
    if cfg.is_file() and json.loads(cfg.read_text(encoding="utf-8")) != config:
        raise SystemExit(f"{cfg} was written for another setting or index; refusing to mix ledgers")
    write_json(cfg, config)
    done = {r["record_id"] for r in read_jsonl(path)}
    todo = [q for q in queries if q["record_id"] not in done][:limit]
    print(f"  {option}: {len(done)} of {len(queries)} done, {len(todo)} to run", flush=True)
    con = open_for(option)
    took = []
    try:
        with path.open("a", encoding="utf-8") as ledger:
            for n, q in enumerate(todo, 1):
                terms, fell_back = select_terms(option, q["claim"], q["entities"], df, n_docs)
                start = time.perf_counter()
                rows = con.execute(sql, params_for(q, terms)).fetchall()
                secs = time.perf_counter() - start
                took.append(secs)
                ledger.write(json.dumps({
                    "record_id": q["record_id"], "seconds": round(secs, 4), "terms": terms,
                    "fell_back": fell_back, "hits": hit_rows(rows)}, ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                if n <= 3 or n % 20 == 0 or secs > 30:
                    print(f"    {len(done) + n:>3}/{len(queries)}  {secs:7.2f} s  terms {len(terms):>2}  "
                          f"hits {len(rows):>2}", flush=True)
    finally:
        con.close()
    if took:
        left = len(queries) - len(done) - len(todo)
        print(f"  median {statistics.median(took):.2f} s, mean {statistics.mean(took):.2f} s; "
              f"{left} left, about {left * statistics.mean(took) / 60:.0f} min")
    return 0


def hit_rows(rows: Sequence[Any]) -> list[list[Any]]:
    return [[r["doc_id"], r["score"], r["published_at"], r["source_domain"], r["url"]] for r in rows]


def williams(n: int) -> list[list[int]]:
    """A Williams design: each of n options in each position equally often, and
    directly after each other option equally often. An even n takes one square
    (n rows: each position once, each pair once); an odd n takes the square and
    its mirror (2n rows: each position twice, each pair twice)."""
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    square = [[(x + r) % n for x in first] for r in range(n)]
    return square if n % 2 == 0 else square + [row[::-1] for row in square]


def session_order(i: int, options: Sequence[str] = SESSION) -> list[str]:
    """The order query i runs the options in: row i of the Williams design, cyclically."""
    rows = williams(len(options))
    return [options[j] for j in rows[i % len(rows)]]


def session(limit: int | None, path: Path = SESSION_LEDGER) -> int:
    """SESSION over the 200 queries in one process, interleaved (session_order).

    One connection, as production has; one ledger line per query holding every
    option's results, so a query is written whole or not at all. Resumable;
    each line records the process that wrote it, so the report can say whether
    the session was one.
    """
    queries = load_queries()
    df, n_docs = load_df()
    config = {"options": list(SESSION), "order": "williams", "k": K, "sql": retrieve.SEARCH_SQL,
              "index": index_stamp(), "blocklist": rules_config(retrieve.load_blocklist()),
              "connection": "default"}
    cfg = path.with_suffix(".config.json")
    if cfg.is_file() and json.loads(cfg.read_text(encoding="utf-8")) != config:
        raise SystemExit(f"{cfg} was written for another setting or index; refusing to mix ledgers")
    write_json(cfg, config)
    done = {r["record_id"] for r in read_jsonl(path)}
    todo = [(i, q) for i, q in enumerate(queries) if q["record_id"] not in done][:limit]
    process = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}"
    print(f"  session {', '.join(SESSION)}: {len(done)} of {len(queries)} queries done, "
          f"{len(todo)} to run", flush=True)
    con = retrieve.connect()
    took: dict[str, list[float]] = {o: [] for o in SESSION}
    started = time.perf_counter()
    try:
        with path.open("a", encoding="utf-8") as ledger:
            for n, (i, q) in enumerate(todo, 1):
                order = session_order(i)
                runs = {}
                for option in order:
                    terms, fell_back = select_terms(option, q["claim"], q["entities"], df, n_docs)
                    start = time.perf_counter()
                    rows = con.execute(retrieve.SEARCH_SQL, params_for(q, terms)).fetchall()
                    secs = time.perf_counter() - start
                    took[option].append(secs)
                    runs[option] = {"seconds": round(secs, 4), "terms": terms,
                                    "fell_back": fell_back, "hits": hit_rows(rows)}
                ledger.write(json.dumps({"record_id": q["record_id"], "position": i, "order": order,
                                         "process": process, "runs": runs},
                                        ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                if n <= 2 or n % 10 == 0:
                    left = (len(todo) - n) * (time.perf_counter() - started) / n / 60
                    print(f"    {len(done) + n:>3}/{len(queries)}  "
                          + "  ".join(f"{o} {runs[o]['seconds']:6.2f} s" for o in SESSION)
                          + f"   about {left:.0f} min left", flush=True)
    finally:
        con.close()
    if took["L"]:
        print("  medians: " + ", ".join(f"{o} {statistics.median(v):.2f} s" for o, v in took.items()))
    return 0


def default_pass(limit: int | None, path: Path = DEFAULT_LEDGER) -> int:
    """retrieve.search's default over the 200 queries, its document frequencies counted.

    Timed in search's two steps, as search runs them: retrieve.query_terms,
    then the statement. One connection, as production has; one ledger line per
    query, recording the process that wrote it. Resumable.
    """
    queries = load_queries()
    config = {"k": K, "sql": retrieve.SEARCH_SQL, "df_sql": retrieve.DF_SQL,
              "rarest": retrieve.RAREST, "index": index_stamp(),
              "blocklist": rules_config(retrieve.load_blocklist()), "connection": "default"}
    cfg = path.with_suffix(".config.json")
    if cfg.is_file() and json.loads(cfg.read_text(encoding="utf-8")) != config:
        raise SystemExit(f"{cfg} was written for another setting or index; refusing to mix ledgers")
    write_json(cfg, config)
    done = {r["record_id"] for r in read_jsonl(path)}
    todo = [q for q in queries if q["record_id"] not in done][:limit]
    process = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}"
    print(f"  default: {len(done)} of {len(queries)} done, {len(todo)} to run", flush=True)
    con = retrieve.connect()
    took = []
    try:
        with path.open("a", encoding="utf-8") as ledger:
            for n, q in enumerate(todo, 1):
                start = time.perf_counter()
                terms, fell_back = retrieve.query_terms(con, q["claim"])
                counted = time.perf_counter()
                rows = con.execute(retrieve.SEARCH_SQL, {"query": retrieve.match_expr(terms),
                                                         "before": q["as_of"], "k": K}).fetchall()
                end = time.perf_counter()
                took.append(end - start)
                ledger.write(json.dumps({
                    "record_id": q["record_id"], "seconds": round(end - start, 4),
                    "count_seconds": round(counted - start, 4),
                    "sql_seconds": round(end - counted, 4), "terms": terms,
                    "fell_back": fell_back, "process": process, "hits": hit_rows(rows)},
                    ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                if n <= 3 or n % 20 == 0:
                    print(f"    {len(done) + n:>3}/{len(queries)}  {end - start:6.2f} s  "
                          f"(counting {counted - start:5.2f} s)  terms {len(terms):>2}  "
                          f"hits {len(rows):>2}", flush=True)
    finally:
        con.close()
    if took:
        print(f"  median {statistics.median(took):.2f} s, mean {statistics.mean(took):.2f} s")
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def overlap(a: Sequence[str], b: Sequence[str], k: int) -> float | None:
    """|top-k of a ∩ top-k of b| / k; None when the baseline has fewer than k."""
    if len(b) < k:
        return None
    return len(set(a[:k]) & set(b[:k])) / k


def tie_reordered(e1: Sequence[list[Any]], ledger: dict[str, Any]) -> bool:
    """True when E1 and the ledger differ only by order within equal-score runs."""
    mine = [(h[3], h[2]) for h in e1]
    theirs = list(zip(ledger["hit_domains"], ledger["hit_dates"]))
    if len(mine) != len(theirs):
        return False
    i = 0
    while i < len(mine):
        j = i
        while j + 1 < len(mine) and e1[j + 1][1] == e1[i][1]:
            j += 1
        if Counter(mine[i:j + 1]) != Counter(theirs[i:j + 1]):
            return False
        i = j + 1
    return True


def quantiles(xs: Sequence[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)
    pick = lambda q: s[min(len(s) - 1, round(q * (len(s) - 1)))]
    return {"n": len(s), "mean": round(statistics.mean(s), 3), "median": round(statistics.median(s), 3),
            "p90": round(pick(0.90), 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


SUBSETS = (("all", lambda q: True), ("india", lambda q: q["india"]))


def entity_counts(queries: Sequence[dict[str, Any]], runs: dict[str, dict[str, Any]],
                  texts: dict[str, str]) -> dict[str, int]:
    """Per query, how many of the claim's entities its top 5 contain (whole word, case-folded)."""
    from scripts import m4fc_audit

    out = {}
    for q in queries:
        top = [texts[h[0]] for h in runs[q["record_id"]]["hits"][:TOP]]
        out[q["record_id"]] = sum(1 for e in q["entities"] if m4fc_audit.found_in(e["text"], top))
    return out


def entity_matches(queries: Sequence[dict[str, Any]], runs: dict[str, dict[str, Any]],
                   texts: dict[str, str]) -> dict[str, Any]:
    got = entity_counts(queries, runs, texts)
    out = {}
    for name, keep in SUBSETS:
        sub = [q for q in queries if keep(q)]
        out[name] = {"queries": len(sub), "1+": sum(got[q["record_id"]] >= 1 for q in sub),
                     "2+": sum(got[q["record_id"]] >= 2 for q in sub)}
    return out


def entities_against(queries: Sequence[dict[str, Any]], runs: dict[str, dict[str, Any]],
                     base: dict[str, dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """Paired with the baseline, per subset and bar: queries that reach it only here
    (gained) and only in the baseline (lost). Sameness is not quality; this is the
    quality the two disagree on."""
    mine, theirs = entity_counts(queries, runs, texts), entity_counts(queries, base, texts)
    out: dict[str, Any] = {}
    for name, keep in SUBSETS:
        sub = [q["record_id"] for q in queries if keep(q)]
        out[name] = {f"{n}+": {"gained": sum(mine[r] >= n > theirs[r] for r in sub),
                               "lost": sum(theirs[r] >= n > mine[r] for r in sub)} for n in (1, 2)}
    return out


def measure(queries: Sequence[dict[str, Any]], r: dict[str, dict[str, Any]],
            base: dict[str, dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """One option's timings and results against a baseline run of the same queries."""
    ids = lambda row: [h[0] for h in row["hits"]]
    ov = {}
    for k in (K, TOP):
        got = [overlap(ids(r[q["record_id"]]), ids(base[q["record_id"]]), k) for q in queries]
        got = [x for x in got if x is not None]
        ov[k] = {"mean": round(statistics.mean(got), 4), "full": sum(x == 1 for x in got),
                 "n": len(got)}
    return {
        "seconds": quantiles([r[q["record_id"]]["seconds"] for q in queries]),
        "terms": quantiles([len(r[q["record_id"]]["terms"]) for q in queries]),
        "fell_back": sum(1 for q in queries if r[q["record_id"]]["fell_back"]),
        "fewer_than_k": sum(1 for q in queries if len(r[q["record_id"]]["hits"]) < K),
        "overlap_at_20": ov[K],
        "overlap_at_5": ov[TOP],
        "top1_same": sum(1 for q in queries if r[q["record_id"]]["hits"][:1] and
                         base[q["record_id"]]["hits"][:1] and
                         r[q["record_id"]]["hits"][0][0] == base[q["record_id"]]["hits"][0][0]),
        "entities_in_top5": entity_matches(queries, r, texts),
        "entities_against_baseline": entities_against(queries, r, base, texts),
    }


def load_session(queries: Sequence[dict[str, Any]],
                 path: Path = SESSION_LEDGER) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(rows by record id, {option: {record id: run}}); the runs only when every query is in."""
    rows = {r["record_id"]: r for r in read_jsonl(path)}
    if len(rows) != len(queries):
        return rows, {}
    return rows, {o: {rid: r["runs"][o] for rid, r in rows.items()} for o in SESSION}


def session_summary(queries: Sequence[dict[str, Any]], rows: dict[str, Any],
                    runs: dict[str, dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """Every session option against the session's own L, with its time by position:
    position 1 is the option that read the query's posting lists first, cold."""
    out: dict[str, Any] = {"processes": sorted({r["process"] for r in rows.values()}),
                           "order": "williams", "options": {}}
    for o in SESSION:
        m = measure(queries, runs[o], runs["L"], texts)
        by_pos: list[list[float]] = [[] for _ in SESSION]
        for r in rows.values():
            by_pos[r["order"].index(o)].append(r["runs"][o]["seconds"])
        m["seconds_by_position"] = [quantiles(p) for p in by_pos]
        out["options"][o] = m
    return out


def default_summary(queries: Sequence[dict[str, Any]], runs: dict[str, Any],
                    session_runs: dict[str, dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """The default as wired, against the session's L, with its two steps' times.
    The same terms and the same top 20 as the session's L+P4b (ids, order and
    scores) are what show the wiring is the option the session measured."""
    twin = session_runs["L+P4b"]
    hits = lambda row: [(h[0], h[1]) for h in row["hits"]]
    return {**measure(queries, runs, session_runs["L"], texts),
            "processes": sorted({r["process"] for r in runs.values()}),
            "count_seconds": quantiles([r["count_seconds"] for r in runs.values()]),
            "sql_seconds": quantiles([r["sql_seconds"] for r in runs.values()]),
            "same_terms_as_session": sum(runs[q["record_id"]]["terms"] == twin[q["record_id"]]["terms"]
                                         for q in queries),
            "same_hits_as_session": sum(hits(runs[q["record_id"]]) == hits(twin[q["record_id"]])
                                        for q in queries)}


def rules_summary(rules: dict[str, Any] | None) -> str:
    """An earlier pass's leak rules, from its config, in one line."""
    rules = rules or {}
    return (f"earlier: {len(rules.get('sources') or [])} host rows, "
            f"{len(rules.get('doc_ids') or [])} doc ids, {len(rules.get('words') or [])} path words")


def rules_added_effect(queries: Sequence[dict[str, Any]], before: dict[str, Any] | None,
                       after: dict[str, Any] | None, before_config: dict[str, Any] | None
                       ) -> dict[str, Any] | None:
    """What the rules added since L's own pass changed in L: that pass against the
    session's L. The pass's hits passed its own rules, so a hit that left and that
    today's rules block left by a rule added since. Every hit that left must be
    one, and every hit that arrived must rank at or after the old 20th; a query
    where both hold is explained by the added rules alone."""
    from scripts import factcheckers

    if before is None or after is None:
        return None
    rules = retrieve.load_blocklist()
    why = factcheckers.blocker(rules)
    now = rules_config(rules)
    then = (before_config or {}).get("blocklist") or {}
    counts, by_rule, gone_hosts = Counter(), Counter(), Counter()
    for q in queries:
        old, new = before[q["record_id"]]["hits"], after[q["record_id"]]["hits"]
        old_ids, new_ids = {h[0] for h in old}, {h[0] for h in new}
        gone = [h for h in old if h[0] not in new_ids]
        came = [h for h in new if h[0] not in old_ids]
        blocked = [(h, why(h[0], *factcheckers.page(h[3], h[4]))) for h in gone]
        blocked = [(h, rule) for h, rule in blocked if rule]
        worst = max((h[1] for h in old), default=None)
        counts["queries_changed"] += bool(gone or came)
        counts["hits_removed"] += len(gone)
        counts["removed_by_todays_rules"] += len(blocked)
        counts["queries_explained"] += (len(blocked) == len(gone)
                                        and all(worst is None or h[1] >= worst for h in came))
        by_rule.update(rule for _, rule in blocked)
        gone_hosts.update(factcheckers.page(h[3], h[4])[0] for h, _ in blocked)
    return {"queries": len(queries), **dict(counts), "removed_by_rule": dict(by_rule),
            "rules_since": {
                "added": {k: sum(x not in (then.get(k) or []) for x in v) for k, v in now.items()},
                "removed": {k: sum(x not in v for x in then.get(k) or []) for k, v in now.items()}},
            "removed_by_host": dict(gone_hosts.most_common())}


def compare() -> int:
    from scripts import m4fc_audit

    queries = load_queries()
    runs = {o: {r["record_id"]: r for r in read_jsonl(ledger_path(o))} for o in OPTIONS}
    runs = {o: r for o, r in runs.items() if len(r) == len(queries)}
    if "E1" not in runs:
        raise SystemExit("E1 has not run over all 200 queries: it is the baseline")
    base = runs["E1"]
    rows, session_runs = load_session(queries)
    default_rows = {r["record_id"]: r for r in read_jsonl(DEFAULT_LEDGER)}
    default_runs = default_rows if len(default_rows) == len(queries) else {}
    texts: dict[str, str] = {}
    for r in [*runs.values(), *session_runs.values(), default_runs]:
        for row in r.values():
            for h in row["hits"][:TOP]:
                if h[0] not in texts:
                    texts[h[0]] = m4fc_audit.doc_text(h[0]).casefold()
    configs = {o: json.loads(ledger_path(o).with_suffix(".config.json").read_text(encoding="utf-8"))
               for o in runs if ledger_path(o).with_suffix(".config.json").is_file()}

    # E1 against the current results ------------------------------------
    exact = ties = differ = top1 = 0
    for q in queries:
        e1, led = base[q["record_id"]]["hits"], q["ledger"]
        same_top = bool(e1) and round(e1[0][1], 5) == led["top1_bm25"]
        top1 += same_top
        seq = [(h[3], h[2]) for h in e1] == list(zip(led["hit_domains"], led["hit_dates"]))
        if seq and same_top:
            exact += 1
        elif same_top and tie_reordered(e1, led):
            ties += 1
        else:
            differ += 1
    prof = read_jsonl(PROFILE)
    e0_same = sum(1 for p in prof if p["e0_hits"] == p["e1_hits"])
    equivalence = {"queries": len(queries), "identical": exact, "ties_reordered_only": ties,
                   "differ": differ, "top1_score_equal": top1,
                   "profiled_e0_equals_e1": [e0_same, len(prof)]}

    # every separate pass against E1 ---------------------------------------
    # A ledger written under other leak rules than today's keeps its numbers
    # under a label that says so: L's pass of 2026-10-28 had no path words and
    # none of the hosts added by hand.
    current = rules_config(retrieve.load_blocklist())
    earlier = {o for o in runs if with_rules(o)
               and (configs.get(o, {}).get("blocklist") or {}) != current}
    options = {}
    for o, r in runs.items():
        rules = ("none" if not with_rules(o) else
                 rules_summary(configs.get(o, {}).get("blocklist")) if o in earlier else "current")
        options[f"{o} (earlier rules)" if o in earlier else o] = {
            **measure(queries, r, base, texts), "leak_rules": rules}
    session = session_summary(queries, rows, session_runs, texts) if session_runs else None
    out = {"generated": time.strftime("%Y-%m-%d"), "queries": len(queries), "k": K,
           "index": index_stamp(), "profile": profile_summary(prof),
           "e1_equivalence": equivalence, "options": options,
           "session": session,
           "session_queries_done": len(rows),
           "default": (default_summary(queries, default_runs, session_runs, texts)
                       if default_runs and session_runs else None),
           "default_queries_done": len(default_rows),
           "rules_added": rules_added_effect(queries, runs["L"] if "L" in earlier else None,
                                             session_runs.get("L"), configs.get("L")),
           "leak_rules_on_e1": leak_effect(queries, base, session_runs.get("L") or runs.get("L")),
           "df": ({k: v for k, v in json.loads(DF.read_text(encoding="utf-8")).items() if k != "df"}
                  if DF.is_file() else None)}
    write_json(REPORT_JSON, out)
    REPORT_MD.write_text(render(out), encoding="utf-8")
    print(render(out))
    print(f"  wrote {REPORT_JSON.relative_to(PROJECT_ROOT).as_posix()} and "
          f"{REPORT_MD.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


def leak_effect(queries: Sequence[dict[str, Any]], base: dict[str, Any],
                rules_run: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the leak rules take out of E1's top 20, and what E1 returned from
    M4FC fact-checkers' domains (the evidence run counted 7 queries). Hits are
    judged by today's rules (factcheckers.blocker, the SQL's twin)."""
    from scripts import factcheckers, m4fc_audit

    why = factcheckers.blocker(retrieve.load_blocklist())

    def blocked(hit: list[Any]) -> str | None:
        return why(hit[0], *factcheckers.page(hit[3], hit[4]))

    orgs = sorted({r["fc_org"] for r in m4fc_audit.raw_rows()})
    org_hits = Counter()
    for q in queries:
        mine = [h for h in base[q["record_id"]]["hits"] if m4fc_audit.on_domain(h[3], orgs)]
        org_hits["queries"] += bool(mine)
        org_hits["hits"] += len(mine)
        org_hits["fact_check_pages"] += sum(blocked(h) is not None for h in mine)
        org_hits["news_pages"] += sum(blocked(h) is None for h in mine)
        org_hits["own_org_queries"] += any(m4fc_audit.on_domain(h[3], [q["ledger"]["fc_org"]])
                                           for h in mine)
    out: dict[str, Any] = {"m4fc_org_hits_in_e1": dict(org_hits)}
    if rules_run is None:
        return out
    reasons, hosts = Counter(), Counter()
    changed = 0
    for q in queries:
        keep = {h[0] for h in rules_run[q["record_id"]]["hits"]}
        gone = [h for h in base[q["record_id"]]["hits"] if h[0] not in keep]
        changed += bool(gone)
        for h in gone:
            rule = blocked(h)
            if h[2] >= q["as_of"]:
                reasons["dated on the bound"] += 1
            elif rule:
                reasons[f"blocked: {rule}"] += 1
                hosts[factcheckers.page(h[3], h[4])[0]] += 1
            else:
                reasons["other"] += 1
    out.update({"queries_changed": changed, "hits_removed": dict(reasons),
                "removed_by_host": dict(hosts.most_common())})
    return out


def profile_summary(prof: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per profiled query, the parts as shares of the current search (no record id)."""
    out = []
    for p in prof:
        out.append({
            "run_seconds": p["run_seconds"], "terms": p["terms"],
            "matched": p["match_rows"], "pass_date": p["join_rows"],
            "match": p["match_seconds"],
            "ranking": round(p["rank_seconds"] - p["match_seconds"], 3),
            "content_fetch": round(p["content_seconds"] - p["match_seconds"], 3),
            "temporal_join": round(p["join_seconds"] - p["content_seconds"], 3),
            "e0_sql": p["e0_sql_seconds"], "e0_python": p["e0_python_seconds"],
            "e1": p["e1_seconds"], "match_again": p["match_again_seconds"],
            "e1_equals_e0": p["e0_hits"] == p["e1_hits"]})
    return out


def render(r: dict[str, Any]) -> str:
    L = [f"# Retrieval bench, {r['generated']}", "",
         f"{r['queries']} M4FC evidence queries, k = {r['k']}.", "", "## Profile", "",
         "| run (s) | terms | matched | pass date | match | ranking | content fetch | temporal join "
         "| E0 SQL | E0 Python | E1 | E1 = E0 |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for p in r["profile"]:
        L.append(f"| {p['run_seconds']:.1f} | {p['terms']} | {p['matched']:,} | {p['pass_date']:,} | "
                 f"{p['match']:.1f} | {p['ranking']:.1f} | {p['content_fetch']:.1f} | "
                 f"{p['temporal_join']:.1f} | {p['e0_sql']:.1f} | {p['e0_python']:.4f} | "
                 f"{p['e1']:.1f} | {p['e1_equals_e0']} |")
    e = r["e1_equivalence"]
    L += ["", "## E1 against the current results", "",
          f"identical {e['identical']}, tie order only {e['ties_reordered_only']}, differ {e['differ']} "
          f"(of {e['queries']}); top-1 score equal {e['top1_score_equal']}; profiled E0 = E1 "
          f"{e['profiled_e0_equals_e1'][0]} of {e['profiled_e0_equals_e1'][1]}.", "",
          "## Options", "",
          "| option | median s | p90 s | max s | terms (median) | overlap@20 | overlap@5 | top-1 same "
          "| ≥1 entity in top 5 | ≥2 | India ≥1 | India ≥2 | < 20 hits |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for o, v in r["options"].items():
        s, m = v["seconds"], v["entities_in_top5"]
        L.append(f"| {o} | {s['median']:.2f} | {s['p90']:.2f} | {s['max']:.2f} | {v['terms']['median']:.0f} | "
                 f"{v['overlap_at_20']['mean']:.3f} | {v['overlap_at_5']['mean']:.3f} | {v['top1_same']} | "
                 f"{m['all']['1+']} | {m['all']['2+']} | {m['india']['1+']} | {m['india']['2+']} | "
                 f"{v['fewer_than_k']} |")
    s = r.get("session")
    if s:
        n = len(s["processes"])
        first_n = min(v["seconds_by_position"][0]["n"] for v in s["options"].values())
        L += ["", "## The production candidates, timed together", "",
              f"{'One process' if n == 1 else f'{n} processes'} wrote the ledger. Each query ran "
              f"all {len(s['options'])} options in a Williams-design order, and every option is "
              f"measured against the session's own L. \"First\" is the median over the {first_n} "
              "queries an option ran first, reading the query's posting lists cold.", "",
              "| option | median s | mean s | p90 s | first: median s | overlap@20 | overlap@5 "
              "| top-1 same | ≥1 entity in top 5 | ≥2 | India ≥1 | India ≥2 | < 20 hits | fell back |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
              "| ---: | ---: |"]
        for o, v in s["options"].items():
            t, m, first = v["seconds"], v["entities_in_top5"], v["seconds_by_position"][0]
            L.append(f"| {o} | {t['median']:.2f} | {t['mean']:.2f} | {t['p90']:.2f} | "
                     f"{first['median']:.2f} | {v['overlap_at_20']['mean']:.3f} | "
                     f"{v['overlap_at_5']['mean']:.3f} | {v['top1_same']} | {m['all']['1+']} | "
                     f"{m['all']['2+']} | {m['india']['1+']} | {m['india']['2+']} | "
                     f"{v['fewer_than_k']} | {v['fell_back']} |")
        L += ["", "Entities against L, queries gained / lost:", ""]
        for o, v in s["options"].items():
            if o == "L":
                continue
            p = v["entities_against_baseline"]
            L.append(f"- {o}: ≥1 +{p['all']['1+']['gained']}/−{p['all']['1+']['lost']}, "
                     f"≥2 +{p['all']['2+']['gained']}/−{p['all']['2+']['lost']}; India ≥1 "
                     f"+{p['india']['1+']['gained']}/−{p['india']['1+']['lost']}, ≥2 "
                     f"+{p['india']['2+']['gained']}/−{p['india']['2+']['lost']}")
    elif r.get("session_queries_done"):
        L += ["", f"Session: {r['session_queries_done']} of {r['queries']} queries done."]
    d = r.get("default")
    if d:
        n = len(d["processes"])
        m, p = d["entities_in_top5"], d["entities_against_baseline"]
        L += ["", "## The default as wired: L+P4b, its counts included", "",
              f"{'One process' if n == 1 else f'{n} processes'} ran retrieve.search's default over "
              f"the {r['queries']} queries: each distinct word's document frequency counted, then "
              "the statement.", "",
              "| | median s | mean s | p90 s | max s |", "| --- | ---: | ---: | ---: | ---: |"]
        for name, key in (("whole query", "seconds"), ("counting", "count_seconds"),
                          ("statement", "sql_seconds")):
            t = d[key]
            L.append(f"| {name} | {t['median']:.2f} | {t['mean']:.2f} | {t['p90']:.2f} | {t['max']:.2f} |")
        L += ["", f"Against the session's L+P4b: the same terms on {d['same_terms_as_session']} of "
              f"{r['queries']} queries, the same top 20 (ids, order and scores) on "
              f"{d['same_hits_as_session']}. Against the session's L: overlap@20 "
              f"{d['overlap_at_20']['mean']:.3f}, overlap@5 {d['overlap_at_5']['mean']:.3f}, top-1 "
              f"same {d['top1_same']}; ≥1 entity in top 5 {m['all']['1+']}, ≥2 {m['all']['2+']}, "
              f"India ≥1 {m['india']['1+']}, ≥2 {m['india']['2+']} (≥1 +{p['all']['1+']['gained']}/"
              f"−{p['all']['1+']['lost']}, ≥2 +{p['all']['2+']['gained']}/−{p['all']['2+']['lost']}); "
              f"< 20 hits {d['fewer_than_k']}; fell back {d['fell_back']}."]
    elif r.get("default_queries_done"):
        L += ["", f"Default: {r['default_queries_done']} of {r['queries']} queries done."]
    w = r.get("rules_added")
    if w:
        L += ["", "## What the rules added since L's own pass changed in L", "",
              f"Rules added since: {w['rules_since']['added']}; removed: "
              f"{w['rules_since']['removed']}. {w['queries_changed']} of {w['queries']} queries "
              f"changed; {w['hits_removed']} hits left, {w['removed_by_todays_rules']} of them pages "
              f"today's rules block {w['removed_by_rule']}; {w['queries_explained']} queries "
              f"explained by the added rules alone; by host {w['removed_by_host']}."]
    le = r.get("leak_rules_on_e1")
    if le:
        L += ["", "## The leak rules on E1's results", "",
              f"M4FC fact-checkers' domains in E1's top 20: {le['m4fc_org_hits_in_e1']}."]
        if "hits_removed" in le:
            L.append(f"L changed {le['queries_changed']} of {r['queries']} queries; removed "
                     f"{le['hits_removed']}; by host {le['removed_by_host']}.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# leaks: a claim's own fact-check, reachable as its evidence?
# --------------------------------------------------------------------------

LEAK = OUT / "own_fact_checks.json"

#: The blocklist exactly as retrieve.SEARCH_SQL applies it, for one document.
BLOCKED_SQL = f"""
SELECT  t.doc_id
FROM    temp.evidence AS t
WHERE   t.doc_id = :doc_id
  AND   (EXISTS ({retrieve._LISTED.format(h="lower(t.source_domain)", p="t.path")})
         OR (t.wayback AND EXISTS ({retrieve._LISTED.format(h="t.archived_host", p="t.archived_path")}))
         OR t.doc_id IN (SELECT doc_id FROM temp.blocked_doc)
         OR EXISTS ({retrieve._WORD.format(t="t")}))
"""


def own_fact_checks() -> list[dict[str, Any]]:
    """Every claim whose own fact-check page is in the store, with that claim's bound.

    Read from every split, as the blocklist is: this measures the store, and a
    leak into test evidence is the one that matters most.
    """
    from scripts import factcheckers, m4fc_audit
    from configs.paths import raw_dir

    con = connect_ro()
    store: dict[str, tuple[str, str]] = {}
    for doc_id, domain, url, pub in con.execute(
            "SELECT doc_id, source_domain, url, published_at FROM temporal.docs"):
        store.setdefault(factcheckers.url_key(*factcheckers.host_path(url)), (doc_id, pub))
    con.close()
    claims = [("m4fc", r["split"], r.get("article_url"), {"fc_pub_date": r["fc_pub_date"]})
              for r in m4fc_audit.raw_rows()]
    for split in ("train", "dev"):
        rows = json.loads((raw_dir("averitec") / "data" / f"{split}.json").read_text(encoding="utf-8"))
        claims += [("averitec", split, r.get("fact_checking_article"),
                    {"claim_date": r.get("claim_date")}) for r in rows]
    for split in ("train", "val"):
        rows = json.loads((raw_dir("averimatec") / f"{split}.json").read_text(encoding="utf-8"))
        claims += [("averimatec", split, r.get("article"), {"date": r.get("date")}) for r in rows]
    out = []
    for dataset, split, url, meta in claims:
        if not url:
            continue
        hit = store.get(factcheckers.url_key(*factcheckers.host_path(url)))
        if not hit:
            continue
        try:
            bound, which = retrieve.evidence_bound(dataset, meta)
        except retrieve.RetrievalError:
            bound, which = None, None
        out.append({"dataset": dataset, "split": split, "doc_id": hit[0], "published_at": hit[1],
                    "bound": bound, "bound_is": which})
    return out


def leak() -> int:
    found = own_fact_checks()
    con = retrieve.connect()
    try:
        for f in found:
            f["blocked"] = con.execute(BLOCKED_SQL, {"doc_id": f["doc_id"]}).fetchone() is not None
    finally:
        con.close()
    table: dict[str, Counter] = {}
    for f in found:
        c = table.setdefault(f["dataset"], Counter())
        c["in_store"] += 1
        if f["bound"] is None:
            c["no_bound"] += 1
            continue
        on, before = f["published_at"] <= f["bound"], f["published_at"] < f["bound"]
        c["reachable_current"] += on
        c["reachable_strict_only"] += before
        c["reachable_blocklist_only"] += on and not f["blocked"]
        c["reachable_both"] += before and not f["blocked"]
        c["same_day"] += f["published_at"] == f["bound"]
        c["dated_before_bound"] += before
    out = {d: dict(c) for d, c in sorted(table.items())}
    write_json(LEAK, {"claims": found, "summary": out})
    for d, c in out.items():
        print(f"  {d:<11} {c}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity")
    sub.add_parser("leak")
    pr = sub.add_parser("profile")
    pr.add_argument("--limit", type=int)
    sub.add_parser("df")
    r = sub.add_parser("run")
    r.add_argument("--option", required=True, choices=OPTIONS)
    r.add_argument("--limit", type=int)
    c = sub.add_parser("check", help="an option's setup on the first query; runs no pass, writes nothing")
    c.add_argument("--option", required=True, choices=OPTIONS)
    s = sub.add_parser("session", help=f"{', '.join(SESSION)} in one process, interleaved; resumable")
    s.add_argument("--limit", type=int)
    d = sub.add_parser("default", help="retrieve.search's default over the 200 queries, its "
                                       "df counts timed; resumable")
    d.add_argument("--limit", type=int)
    sub.add_parser("compare")
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "identity":
        return identity()
    if a.cmd == "leak":
        return leak()
    if a.cmd == "profile":
        return profile(a.limit)
    if a.cmd == "df":
        return document_frequencies()
    if a.cmd == "check":
        return check(a.option)
    if a.cmd == "run":
        return run(a.option, a.limit)
    if a.cmd == "session":
        return session(a.limit)
    if a.cmd == "default":
        return default_pass(a.limit)
    return compare()


if __name__ == "__main__":
    raise SystemExit(main())
