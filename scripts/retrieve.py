#!/usr/bin/env python
"""Time-filtered retrieval over the evidence store.

    python scripts/retrieve.py --query "vaccine safety" --before 2020-01-01 -k 10
    python scripts/retrieve.py --query "vaccine safety" --before 2020-01-01 --exact

Two rules govern this module:

    A DOCUMENT MAY ONLY BE RETURNED IF IT WAS PUBLISHED STRICTLY BEFORE THE
    `before` DATE, AND NEVER IF IT IS A FACT-CHECKER'S PAGE.

Both are enforced **inside the SQL**, as predicates beside the MATCH -- not by
filtering results afterwards. The difference matters. A post-filter is a
promise that every future caller remembers to apply it, and it silently
degrades `k` (ask for 10, filter 4 away, return 6). Putting the predicates in
the query means a future-dated, undated or fact-checker document cannot enter
the candidate set at all, whatever the ranking does, and `k` still means k.

`before` is the claim's own date where its dataset carries one, and the
fact-check's publication date otherwise (`evidence_bound`, configs/as_of.yaml).
Strictly before, because a fact-check is dated the day it appears: on
2026-10-28 two M4FC claims had their own fact-check in the store, dated exactly
their fact-check date, and the inclusive bound used until then returned each
as evidence for its own claim.

Undated documents cannot appear because they are never indexed: the store
quarantines them to `data/bible/undated/`. The JOIN is the second line of
defence, and `published_at` is `NOT NULL` in the schema as the third.

The fact-checkers are configs/factcheckers.yaml, built by
scripts/factcheckers.py from the fact-checkers behind our datasets. `connect()`
loads them into TEMP tables, and defines the TEMP view the statement reads
dates and URLs through, so a connection opened any other way fails with "no
such table" instead of returning a fact-check. A listed host is blocked with
its subdomains, or only within its fact-check section (a news publisher's
desk), or -- for a publisher whose fact-checks share no section -- only in
the documents that are its cited fact-checks. And on every host, listed or
not, a page whose path carries a fact-check desk word (`path_words`:
fact-check, webqoof, verificador, ...) is blocked. A Wayback capture is judged
by the URL it archived: the store files 3,056 captures under web.archive.org.

The statement orders by FTS5's own `rank` (bm25 by default), which FTS5 sorts
itself: rows stream out best first, so the join, the date and the blocklist
run only until k rows pass. Ordered by `bm25(docs_fts)` instead, SQLite joined
every matching document -- for an OR of a claim's words, most of the corpus --
and sorted them all (scripts/retrieval_bench.py measures both). CROSS JOIN
keeps docs_fts the outer loop, which that streaming needs.

By default a query is searched by its 8 rarest distinct words, none of them in
half the store or more (`query_terms`: P4b in scripts/retrieval_bench.py,
adopted on 2026-10-29 by the person's decision). FTS5 scores a word in half the
store at idf 1e-6, so dropping one moves a score by at most 2.2e-6; the cut to
8 is what changes results, and docs/evidence_store.md has what it changed on
200 M4FC claims. Each word's document frequency is counted per query, as bm25
counts it. `exact` (`--exact`) searches every word, as L did: slower, and the
reference the default was measured against. Both modes run the one statement,
so both rules hold in either.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import AS_OF_YAML, BIBLE, FACTCHECKERS_YAML  # noqa: E402

INDEX = BIBLE / "index"
BM25_DB = INDEX / "bm25.sqlite"
TEMPORAL_DB = INDEX / "temporal.sqlite"


# --------------------------------------------------------------------------
# a page's host and path, in SQL -- scripts/factcheckers.py is the Python twin
# --------------------------------------------------------------------------


def _after_scheme(x: str) -> str:
    """x without its scheme; a '://' past the first 12 characters is not one."""
    return (f"(CASE WHEN instr(substr({x}, 1, 12), '://') > 0 "
            f"THEN substr({x}, instr({x}, '://') + 3) ELSE {x} END)")


def _first(x: str, chars: str) -> str:
    """The position of the first of `chars` in x, or one past its end."""
    for ch in chars[1:]:
        x = f"replace({x}, '{ch}', '{chars[0]}')"
    return f"instr({x} || '{chars[0]}', '{chars[0]}')"


def _host_of(rest: str) -> str:
    """The host of `host[:port][/path][?query][#fragment]`, without `www.` (bible.domain_of)."""
    h = f"substr({rest}, 1, {_first(rest, '/?#:')} - 1)"
    return f"(CASE WHEN {h} LIKE 'www.%' THEN substr({h}, 5) ELSE {h} END)"


def _path_of(rest: str) -> str:
    """The path of `host[/path][?query][#fragment]`, as urlsplit reads it; '/' when empty."""
    tail = f"substr({rest}, {_first(rest, '/?#')})"
    return (f"(CASE WHEN substr({tail}, 1, 1) = '/' "
            f"THEN substr({tail}, 1, {_first(tail, '?#')} - 1) ELSE '/' END)")


_URL = "lower(d.url)"
_AFTER_STAMP = f"substr({_URL}, instr({_URL}, '/web/') + 5)"      # "<stamp>/<archived url>"

#: Every indexed document with the host and path the blocklist is judged on.
#: A Wayback capture (one level) is judged by the URL it archived. Two views,
#: so that neither statement nests deeper than SQLite's parser allows; SQLite
#: flattens them into the search, which still finds its rows by doc_id.
EVIDENCE_VIEWS = (f"""
CREATE TEMP VIEW evidence_url AS
SELECT  d.doc_id, d.published_at, d.source_domain, d.url,
        (d.url LIKE 'http://web.archive.org/web/%'
         OR d.url LIKE 'https://web.archive.org/web/%')     AS wayback,
        {_after_scheme(_URL)}                                 AS rest,
        {_after_scheme(f"substr({_AFTER_STAMP}, instr({_AFTER_STAMP}, '/') + 1)")}
                                                              AS archived_rest
FROM    temporal.docs AS d
""", f"""
CREATE TEMP VIEW evidence AS
SELECT  doc_id, published_at, source_domain, url, wayback,
        {_path_of("rest")}            AS path,
        {_host_of("archived_rest")}   AS archived_host,
        {_path_of("archived_rest")}   AS archived_path
FROM    temp.evidence_url
""")

#: A host matches a listed host when it is that host or a subdomain of it, and
#: a path is in a section when it is the prefix or lies below it.
_LISTED = """
            SELECT 1 FROM temp.blocked_source AS b
            WHERE  ({h} = b.host OR substr({h}, -length(b.host) - 1) = '.' || b.host)
              AND  (b.path IS NULL OR {p} = b.path
                    OR substr({p}, 1, length(b.path) + 1) = b.path || '/')"""

#: A path word anywhere in the page's path: the archived URL's for a capture,
#: the same path the host rules judge (and factcheckers.page computes).
_WORD = """
            SELECT 1 FROM temp.blocked_word AS w
            WHERE  instr(CASE WHEN {t}.wayback THEN {t}.archived_path ELSE {t}.path END,
                         w.word) > 0"""

#: The whole retrieval contract, in one statement.
SEARCH_SQL = f"""
SELECT  f.doc_id     AS doc_id,
        f.title      AS title,
        t.published_at,
        t.source_domain,
        t.url,
        f.rank       AS score
FROM    docs_fts AS f
CROSS JOIN temp.evidence AS t
WHERE   docs_fts MATCH :query
  AND   t.doc_id = f.doc_id
  AND   t.published_at IS NOT NULL
  AND   t.published_at < :before
  AND   NOT EXISTS ({_LISTED.format(h="lower(t.source_domain)", p="t.path")})
  AND   NOT (t.wayback AND EXISTS ({_LISTED.format(h="t.archived_host", p="t.archived_path")}))
  AND   t.doc_id NOT IN (SELECT doc_id FROM temp.blocked_doc)
  AND   NOT EXISTS ({_WORD.format(t="t")})
ORDER BY f.rank
LIMIT   :k
"""


class RetrievalError(RuntimeError):
    """The index or a retrieval rule is missing or unusable."""


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    published_at: str
    source_domain: str
    url: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title,
                "published_at": self.published_at,
                "source_domain": self.source_domain, "url": self.url,
                "score": round(self.score, 5)}


# --------------------------------------------------------------------------
# the rules, from configs/
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Blocklist:
    """What retrieval may never return: hosts (or their fact-check sections),
    documents, and pages whose path carries a fact-check word."""
    sources: tuple[tuple[str, str | None], ...]   # (host, section prefix, or None for the host)
    doc_ids: tuple[str, ...]
    words: tuple[str, ...]


#: A path word shorter than this would block by accident ("fact" is in "factory").
MIN_WORD = 4


def load_blocklist(path: Path | None = None) -> Blocklist:
    """configs/factcheckers.yaml, checked; refuses a missing or empty list."""
    import yaml

    path = path or FACTCHECKERS_YAML
    if not path.is_file():
        raise RetrievalError(f"fact-checker blocklist not found: {path}. Build it with "
                             "`python scripts/factcheckers.py build`.")
    return blocklist_from(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path)


def blocklist_from(doc: dict[str, Any], path: Any) -> Blocklist:
    """The rules in one parsed list; refuses one that blocks no host or no path word."""
    rows: list[tuple[str, str | None]] = []
    docs: list[str] = []
    for e in doc.get("hosts") or []:
        host = e["host"]
        if host != host.lower() or host.startswith("www.") or "/" in host:
            raise RetrievalError(f"{path}: host {host!r} is not a bare lower-case host")
        if e["scope"] == "domain":
            rows.append((host, None))
        elif e["scope"] == "path":
            for p in e["paths"]:
                if not p.startswith("/") or p != p.lower() or p.endswith("/"):
                    raise RetrievalError(f"{path}: {host} section {p!r} is not a lower-case "
                                         "path prefix without a trailing slash")
                rows.append((host, p))
        elif e["scope"] == "docs":
            for d in e.get("doc_ids") or []:
                if len(d) != 64 or d.strip("0123456789abcdef"):
                    raise RetrievalError(f"{path}: {host} doc id {d!r} is not a sha256")
                docs.append(d)
        elif e["scope"] != "none":      # "none": listed, not blocked; only a person sets it
            raise RetrievalError(f"{path}: {host} has scope {e['scope']!r}")
    if not rows:
        raise RetrievalError(f"{path} blocks no host; refusing to retrieve without it")
    words = (doc.get("path_words") or {}).get("words") or []
    for w in words:
        if (not isinstance(w, str) or w != w.lower() or not w.isascii() or len(w) < MIN_WORD
                or any(c.isspace() or c in "/?#" for c in w)):
            raise RetrievalError(f"{path}: path word {w!r} is not a lower-case ASCII word of "
                                 f"{MIN_WORD}+ characters without '/', '?', '#' or spaces")
    if not words:
        raise RetrievalError(f"{path} blocks no path word; refusing to retrieve without them")
    return Blocklist(tuple(rows), tuple(sorted(set(docs))), tuple(dict.fromkeys(words)))


def install_rules(con: sqlite3.Connection, factcheckers: Path | None = None) -> None:
    """The TEMP tables and view SEARCH_SQL reads. Every connection that searches needs them."""
    rules = load_blocklist(factcheckers)
    con.execute("CREATE TEMP TABLE blocked_source (host TEXT NOT NULL, path TEXT)")
    con.execute("CREATE TEMP TABLE blocked_doc (doc_id TEXT PRIMARY KEY)")
    con.execute("CREATE TEMP TABLE blocked_word (word TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO temp.blocked_source VALUES (?, ?)", rules.sources)
    con.executemany("INSERT INTO temp.blocked_doc VALUES (?)", [(d,) for d in rules.doc_ids])
    con.executemany("INSERT INTO temp.blocked_word VALUES (?)", [(w,) for w in rules.words])
    for view in EVIDENCE_VIEWS:
        con.execute(view)
    if con.in_transaction:
        con.commit()


@lru_cache(maxsize=None)
def _as_of_rules(path: Path) -> dict[str, Any]:
    import yaml

    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("datasets") or {}


def _parse_date(value: Any, formats: list[str]) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    for fmt in formats:
        try:
            if fmt == "iso":
                got = datetime.strptime(text[:10], "%Y-%m-%d")
            elif fmt == "unix":
                got = datetime.fromtimestamp(float(text), tz=timezone.utc)
            else:
                got = datetime.strptime(text, fmt)
        except (ValueError, OverflowError, OSError):
            continue
        return got.strftime("%Y-%m-%d")
    return None


def evidence_bound(dataset: str, meta: dict[str, Any],
                   rules: Path | None = None) -> tuple[str, str]:
    """(date, "claim_date" | "fact_check_date") bounding a claim's evidence.

    The claim's own date where the dataset carries one, the fact-check's
    publication date only as a fallback (configs/as_of.yaml). A claim with
    neither raises: no date, no retrieval.
    """
    rule = _as_of_rules(rules or AS_OF_YAML).get(dataset)
    if rule is None:
        raise RetrievalError(f"{dataset!r} has no entry in {rules or AS_OF_YAML}")
    for which in ("claim_date", "fact_check_date"):
        spec = rule.get(which)
        if spec:
            got = _parse_date((meta or {}).get(spec["field"]), spec["formats"])
            if got:
                return got, which
    raise RetrievalError(f"{dataset}: no date bounds this claim's evidence, so it gets none")


# --------------------------------------------------------------------------
# the words a query is searched by
# --------------------------------------------------------------------------

#: The default searches a query by this many of its rarest distinct words.
RAREST = 8

#: A word's document frequency as bm25 counts it: the documents that match it.
DF_SQL = "SELECT count(*) FROM docs_fts WHERE docs_fts MATCH :query"


def words(query: str) -> list[str]:
    """A query's words, in order: anything but a letter, a digit or a space separates them."""
    return "".join(c if c.isalnum() or c.isspace() else " " for c in query).split()


def match_expr(terms: Sequence[str]) -> str:
    """Each word quoted, so query text cannot become FTS5 operator syntax, ORed."""
    return " OR ".join(f'"{t}"' for t in terms)


def distinct(terms: Iterable[str]) -> list[str]:
    """Each word once, case-folded, where it first occurs."""
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def rarest(terms: Sequence[str], df: dict[str, int], n: int) -> list[str]:
    """The n rarest distinct terms; ties keep the claim's order."""
    terms = distinct(terms)
    order = sorted(range(len(terms)), key=lambda i: (df[terms[i].lower()], i))
    return [terms[i] for i in sorted(order[:n])]


def clamped(term: str, df: dict[str, int], n_docs: int) -> bool:
    """FTS5 clamps idf to 1e-6 at df >= N/2: this store's N is odd, so in more than half."""
    return df[term.lower()] >= n_docs / 2


def prune(terms: Sequence[str], df: dict[str, int], n_docs: int) -> tuple[list[str], bool]:
    """(the words the default searches by, fell back): the RAREST rarest distinct
    words outside half the store, or the one rarest word when none is outside it."""
    kept = rarest([t for t in terms if not clamped(t, df, n_docs)], df, RAREST)
    return (kept, False) if kept else (rarest(terms, df, 1), True)


def frequencies(con: sqlite3.Connection, terms: Sequence[str]) -> tuple[dict[str, int], int]:
    """(each distinct word's document frequency, the documents in the store), counted now.

    The store is the temporal table: one row per indexed document, since an
    undated one is never indexed. On the store as built that is bm25's own N.
    """
    n_docs = con.execute("SELECT count(*) FROM temporal.docs").fetchone()[0]
    df = {t.lower(): con.execute(DF_SQL, {"query": match_expr([t.lower()])}).fetchone()[0]
          for t in distinct(terms)}
    return df, n_docs


def query_terms(con: sqlite3.Connection, query: str,
                exact: bool = False) -> tuple[list[str], bool]:
    """(the words `search` matches, fell back): every word when exact, else `prune`'s."""
    terms = words(query)
    if not terms:
        raise RetrievalError("empty query after sanitisation")
    if exact:
        return terms, False
    return prune(terms, *frequencies(con, terms))


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def connect(bm25_db: Path | None = None, temporal_db: Path | None = None,
            factcheckers: Path | None = None) -> sqlite3.Connection:
    """The BM25 index, the temporal table ATTACHed as `temporal`, and the rules installed."""
    bm25_db = bm25_db or BM25_DB
    temporal_db = temporal_db or TEMPORAL_DB
    for p in (bm25_db, temporal_db):
        if not p.is_file():
            raise RetrievalError(
                f"index not found: {p}. Build it with "
                "`python scripts/bible.py --index`.")
    con = sqlite3.connect(f"file:{bm25_db}?mode=ro", uri=True)
    con.execute("ATTACH DATABASE ? AS temporal", (f"file:{temporal_db}?mode=ro",))
    con.row_factory = sqlite3.Row
    install_rules(con, factcheckers)
    return con


def _sanitise(query: str) -> str:
    """Every word of the query, quoted so user text cannot become FTS5 operator syntax."""
    terms = words(query)
    if not terms:
        raise RetrievalError("empty query after sanitisation")
    return match_expr(terms)


def search(query: str, before: str, k: int = 10,
           con: sqlite3.Connection | None = None, exact: bool = False) -> list[Hit]:
    """Documents matching `query` published strictly before `before`, fact-checkers excluded.

    `before` is an ISO date (YYYY-MM-DD); it is compared lexically, which is
    exactly right for zero-padded ISO dates and needs no parsing. The query is
    searched by its RAREST rarest words outside half the store, or by every
    word when `exact` (`query_terms`).
    """
    return search_with_terms(query, before, k, con, exact)[0]


def search_with_terms(query: str, before: str, k: int = 10,
                      con: sqlite3.Connection | None = None, exact: bool = False
                      ) -> tuple[list[Hit], list[str], bool]:
    """`search`, and the words it searched by: (hits, terms, fell back). The
    words are counted once, for the statement and the caller alike."""
    if not (isinstance(before, str) and len(before) == 10
            and before[4] == before[7] == "-"):
        raise RetrievalError(
            f"before must be an ISO date YYYY-MM-DD, got {before!r}")
    own = con is None
    con = con or connect()
    try:
        terms, fell_back = query_terms(con, query, exact)
        rows = con.execute(SEARCH_SQL, {"query": match_expr(terms),
                                        "before": before, "k": int(k)}).fetchall()
    finally:
        if own:
            con.close()
    return [Hit(r["doc_id"], r["title"], r["published_at"], r["source_domain"],
                r["url"], r["score"]) for r in rows], terms, fell_back


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--query", required=True)
    p.add_argument("--before", required=True,
                   help="ISO date; only documents published strictly before it can be returned")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--exact", action="store_true",
                   help=f"search every word, not only the {RAREST} rarest outside half the store")
    args = p.parse_args(argv)

    try:
        hits = search(args.query, args.before, args.k, exact=args.exact)
    except RetrievalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    by = "every word" if args.exact else f"its {RAREST} rarest words outside half the store"
    print(f"\nquery {args.query!r}  before {args.before}  k={args.k}  by {by}")
    if not hits:
        print("  no documents published before that date match.")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"  {i:>2}. [{h.published_at}] {h.title[:72]}")
        print(f"      {h.source_domain}  score={h.score:.3f}")
    latest = max(h.published_at for h in hits)
    print(f"\n  newest result {latest} < before {args.before}: "
          f"{'OK' if latest < args.before else 'VIOLATION'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
