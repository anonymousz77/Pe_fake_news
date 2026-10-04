#!/usr/bin/env python
"""Time-filtered retrieval over the evidence store.

    python scripts/retrieve.py --query "vaccine safety" --as-of 2020-01-01 -k 10

One rule governs this module:

    A DOCUMENT MAY ONLY BE RETURNED IF IT WAS PUBLISHED ON OR BEFORE THE
    as_of DATE.

That rule is enforced **inside the SQL**, as a JOIN against the temporal table
with a `published_at <= :as_of` predicate — not by filtering results afterwards.
The difference matters. A post-filter is a promise that every future caller
remembers to apply it, and it silently degrades `k` (ask for 10, filter 4 away,
return 6). Putting the predicate in the query means an undated or future-dated
document cannot enter the candidate set at all, whatever the ranking does, and
`k` still means k.

Undated documents cannot appear because they are never indexed: the store
quarantines them to `data/bible/undated/`. The JOIN is the second line of
defence, and `published_at` is `NOT NULL` in the schema as the third.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import BIBLE  # noqa: E402

INDEX = BIBLE / "index"
BM25_DB = INDEX / "bm25.sqlite"
TEMPORAL_DB = INDEX / "temporal.sqlite"

#: The whole retrieval contract, in one statement. The date predicate sits in
#: the WHERE clause beside the MATCH, so it constrains the candidate set rather
#: than trimming the output.
SEARCH_SQL = """
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


class RetrievalError(RuntimeError):
    """The index is missing or unusable."""


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


def connect(bm25_db: Path | None = None, temporal_db: Path | None = None
            ) -> sqlite3.Connection:
    """Open the BM25 index with the temporal table ATTACHed as `temporal`."""
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
    return con


def _sanitise(query: str) -> str:
    """Quote each term so user text cannot become FTS5 operator syntax."""
    terms = [t for t in "".join(c if c.isalnum() or c.isspace() else " "
                                for c in query).split() if t]
    if not terms:
        raise RetrievalError("empty query after sanitisation")
    return " OR ".join(f'"{t}"' for t in terms)


def search(query: str, as_of_date: str, k: int = 10,
           con: sqlite3.Connection | None = None) -> list[Hit]:
    """Documents matching `query` that were published on or before `as_of_date`.

    `as_of_date` is an ISO date (YYYY-MM-DD); it is compared lexically, which is
    exactly right for zero-padded ISO dates and needs no parsing.
    """
    if not (isinstance(as_of_date, str) and len(as_of_date) == 10
            and as_of_date[4] == as_of_date[7] == "-"):
        raise RetrievalError(
            f"as_of_date must be an ISO date YYYY-MM-DD, got {as_of_date!r}")
    own = con is None
    con = con or connect()
    try:
        rows = con.execute(SEARCH_SQL, {"query": _sanitise(query),
                                        "as_of": as_of_date, "k": int(k)}).fetchall()
    finally:
        if own:
            con.close()
    return [Hit(r["doc_id"], r["title"], r["published_at"], r["source_domain"],
                r["url"], r["score"]) for r in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--query", required=True)
    p.add_argument("--as-of", required=True, dest="as_of",
                   help="ISO date; nothing published after it can be returned")
    p.add_argument("-k", type=int, default=10)
    args = p.parse_args(argv)

    try:
        hits = search(args.query, args.as_of, args.k)
    except RetrievalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nquery {args.query!r}  as_of {args.as_of}  k={args.k}")
    if not hits:
        print("  no documents published on or before that date match.")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"  {i:>2}. [{h.published_at}] {h.title[:72]}")
        print(f"      {h.source_domain}  score={h.score:.3f}")
    latest = max(h.published_at for h in hits)
    print(f"\n  newest result {latest} <= as_of {args.as_of}: "
          f"{'OK' if latest <= args.as_of else 'VIOLATION'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
