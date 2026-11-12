#!/usr/bin/env python
"""The fact-checkers behind our datasets, as a retrieval blocklist kept in configs.

    python scripts/factcheckers.py build      # configs/factcheckers.yaml, offline
    python scripts/factcheckers.py residual   # fact-check-looking store pages on unlisted hosts

A fact-checker's own page is the answer, not evidence: it states the verdict
the model is asked for. The store holds such pages -- 3 M4FC claims and 53
AVeriTeC train/dev claims have their own fact-check in it -- so retrieval
excludes them inside the SQL (`retrieve.SEARCH_SQL`), from this list.

Every split is read, TEST INCLUDED: the person's decision of 2026-10-28. A
domain list carries no label, and a fact-checker we miss leaks answers into
test evidence, which is worse than tailoring the list to test. Each host
records the datasets and splits that cite it, and `build` reports how many
come only from test. Nothing is fetched: a test file that is not on disk is
reported as absent.

Scope, per host:

* **domain** -- block the host and its subdomains. The default: every host
  with 50 store documents or fewer, every host named for checking
  (africacheck.org, factcheck.org), and every host whose store documents are
  nearly all inside its fact-check section.
* **path** -- block only the fact-check section (its path prefix), when the
  store holds more than 50 of the host's documents outside that section: a
  news publisher with a fact-check desk (indiatoday.in /fact-check,
  usatoday.com /story/news/factcheck). Blocking those publishers whole would
  drop thousands of news pages to remove a handful of fact-checks.
* **docs** -- a publisher with more than 50 store documents whose fact-checks
  share no section (reuters.com files them as /article/<id>, bbc.co.uk as
  /news/<number>): block only the store documents that ARE its cited
  fact-checks, by doc id, and mark the host `review` until a person decides
  (`DECIDED_DOCS`: the nine of 2026-10-28 stay docs-scoped, not blocked whole).
* **none** -- listed but not blocked. Never generated: only a person sets it.

A section is the first one to three path segments shared by at least 90% of
the host's fact-check URLs in our datasets, and only a prefix that names a
fact-check desk counts (`FACT_CHECK_WORDS`): BBC's fact-checks share `/news`,
which is all of BBC. Wayback captures are judged by the archived URL: the store
files 3,056 of them under `web.archive.org`.

**Path words, on every host** (`path_words`; the person's decision of
2026-10-28). A page whose lower-cased path contains any of `FACT_CHECK_WORDS`
is blocked whatever its host: a fact-check desk our datasets never cite
(Reuters /article/...factcheck..., CNN fact-check- pieces, TEGNA's VERIFY
stations). On 2026-10-28 that was 1,038 store pages on 588 hosts beyond what
the host rules block, some of them news about fake news: losing a few real
news pages is cheaper than a leak. The host name is not read: `residual`
reports the unlisted hosts whose name reads like a fact-checker.

**Added by hand** (`ADDED_BY_HAND`; the person's decision of 2026-10-29). Five
fact-checkers no dataset of ours cites, which only a host-name rule would
catch: `residual` found them on 2026-10-28. Each is blocked whole, whatever its
store pages or sections, and since nothing cites it, none comes only from test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import FACTCHECKERS_YAML, PROJECT_ROOT, raw_dir  # noqa: E402

STORE_FLOOR = 50       # store documents outside a section before a host is path-scoped
SECTION_SHARE = 0.90   # share of a host's fact-check URLs a section must cover

#: Factify2 has no fact-check URLs, only image URLs. Its fact-checkers are the
#: four hosts its hostname confound names (CLAUDE.md, Factify2 standing rules:
#: each 100% Refute), by site, plus any image host on a site the other
#: datasets list.
FACTIFY2_DOCUMENTED = {"factly.in": "factly.in", "boomlive.in": "boomlive.in",
                       "images.thequint.com": "thequint.com", "snopes.com": "snopes.com"}

#: Datasets that are one fact-checker's by construction, and where that is stated.
BY_CONSTRUCTION = {
    "liar": ({"politifact.com"}, "data/raw/liar/README: statements from the politifact.com API"),
    "fakenewsnet": ({"politifact.com", "gossipcop.com"},
                    "data/raw/fakenewsnet/dataset/{politifact,gossipcop}_{fake,real}.csv"),
}

#: Path words that name a fact-check desk: a section must carry one, and since
#: 2026-10-28 a page whose path carries one is blocked on every host (the
#: list's `path_words`). Matched as substrings: "verifica" also takes
#: "verification", which is part of the price the person accepted.
FACT_CHECK_WORDS = ("fact-check", "factcheck", "fact_check", "faktencheck", "fact-checking",
                    "hoax-alert", "hoax", "verificador", "verifica", "verify", "webqoof",
                    "fake-news", "reality-check", "antibulo", "faux-et-usage", "desintox",
                    "fakecheck", "viral-check", "debunk")

#: A listed host whose own name says it checks facts is a fact-checker, blocked
#: whole whatever its sections: africacheck.org's older reports sit outside
#: its /fact-checks section.
CHECKER_NAME_WORDS = ("fact", "check", "hoax", "verif", "truth")

#: Docs-scoped hosts a person has decided: they stay docs-scoped (only their
#: cited fact-checks are blocked), and `build` writes the decision where it
#: would write `review`. A docs-scoped host not named here is still flagged.
DECIDED_DOCS = {h: "the person, 2026-10-28: block only its cited fact-checks, not the host whole"
                for h in ("aap.com.au", "abc.net.au", "bbc.co.uk", "cbsnews.com", "king5.com",
                          "nytimes.com", "reuters.com", "thejournal.ie", "washingtonpost.com")}

PATH_WORDS_DECIDED = ("the person, 2026-10-28: block every page whose path carries one, on every "
                      "host; losing a few real news pages is cheaper than a leak")

#: Fact-checkers no dataset of ours cites, added by a person: `build` lists
#: each one whole (scope domain), whatever its store pages.
ADDED_BY_HAND = {h: ("the person, 2026-10-29: a fact-checker no dataset of ours cites, found by "
                     "name by `residual`; block the host whole")
                 for h in ("factcheck.aap.com.au", "smhoaxslayer.com", "turnbackhoax.id", "hoax.sk",
                           "myanmar.factcrescendo.com")}

WAYBACK = re.compile(r"^(?:https?://)?web\.archive\.org/web/[^/]+/", re.I)


# --------------------------------------------------------------------------
# hosts and paths, the same way the SQL computes them
# --------------------------------------------------------------------------


def unwrap(url: str, levels: int | None = None) -> str:
    """The archived URL inside a Wayback capture (every level, or `levels`); others unchanged.

    A dataset's citation is unwrapped fully. A store page is unwrapped one
    level, as retrieve.EVIDENCE_VIEWS do in SQL, so the two agree.
    """
    url = (url or "").strip()
    n = 0
    while WAYBACK.match(url) and (levels is None or n < levels):
        url = WAYBACK.sub("", url, count=1)
        n += 1
    return url if "://" in url[:12] else "http://" + url


#: SQLite's lower() folds ASCII only, so the twin does too.
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def host_path(url: str, levels: int | None = None) -> tuple[str, str]:
    """(host, path): host lower-cased without `www.` (bible.domain_of); path lower-cased."""
    u = urlsplit(unwrap(url, levels).translate(_ASCII_LOWER))
    host = u.netloc.split(":", 1)[0]          # up to a port, as the SQL cuts it
    host = host[4:] if host.startswith("www.") else host
    return host, u.path or "/"


def on_host(host: str, listed: str) -> bool:
    return host == listed or host.endswith("." + listed)


def section(paths: list[str], depth: int) -> tuple[str, float] | None:
    """The commonest prefix of `depth` path segments, and its share."""
    if not paths:
        return None
    heads = Counter("/" + "/".join(p.strip("/").split("/")[:depth]) for p in paths
                    if len(p.strip("/").split("/")) > depth)
    if not heads:
        return None
    head, n = heads.most_common(1)[0]
    return head, n / len(paths)


def under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


# --------------------------------------------------------------------------
# what each dataset says
# --------------------------------------------------------------------------


def _dict_rows(path: Path, delimiter: str | None = None) -> Iterator[dict[str, str]]:
    from scripts.records import _dict_rows as rows

    yield from rows(path, delimiter)


def suffixes(host: str) -> Iterator[str]:
    """The host and each parent domain: a.b.c, b.c (never the bare TLD)."""
    parts = host.split(".")
    for i in range(len(parts) - 1):
        yield ".".join(parts[i:])


def citations() -> tuple[list[tuple[str, str, str, str]], dict[str, Any]]:
    """[(host, dataset, split, path)] for every fact-check URL, and what was read."""
    out: list[tuple[str, str, str, str]] = []
    read: dict[str, Any] = {}

    def add(dataset: str, split: str, url: str | None) -> bool:
        if not url or not url.strip():
            return False
        host, path = host_path(url)
        if host:
            out.append((host, dataset, split, path))
        return bool(host)

    m4fc = json.loads((raw_dir("m4fc") / "data" / "M4FC.json").read_text(encoding="utf-8"))
    read["m4fc"] = dict(Counter(r["split"] for r in m4fc if add("m4fc", r["split"], r.get("article_url"))))
    read["m4fc_fc_org_differs_from_url_host"] = sum(
        1 for r in m4fc if not on_host(host_path(r["article_url"])[0], r["fc_org"].lower()))

    base = raw_dir("averitec") / "data"
    read["averitec"] = {}
    for split, name in (("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")):
        rows = json.loads((base / name).read_text(encoding="utf-8"))
        if not any("fact_checking_article" in r for r in rows):
            read["averitec"][split] = "no fact_checking_article field"
            continue
        read["averitec"][split] = sum(add("averitec", split, r.get("fact_checking_article")) for r in rows)

    base = raw_dir("averimatec")
    read["averimatec"] = {}
    for split, name in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
        if not (base / name).is_file():
            read["averimatec"][split] = "not on disk"
            continue
        rows = json.loads((base / name).read_text(encoding="utf-8"))
        read["averimatec"][split] = sum(add("averimatec", split, r.get("article")) for r in rows)

    base = raw_dir("mocheg") / "mocheg_back_up_without_image"
    read["mocheg"] = {}
    for split in ("train", "val", "test"):
        seen: set[str] = set()
        for row in _dict_rows(base / split / "Corpus2.csv", ","):
            url = (row.get("Snopes URL") or "").strip()
            if url and url not in seen:
                seen.add(url)
                add("mocheg", split, url)
        read["mocheg"][split] = len(seen)

    verite = raw_dir("verite") / "data" / "VERITE" / "VERITE_articles.csv"
    read["verite"] = {"all": sum(add("verite", "all", r.get("snopes_url"))
                                 for r in _dict_rows(verite, ","))}

    for name, (hosts, why) in BY_CONSTRUCTION.items():
        for h in sorted(hosts):
            out.append((h, name, "all", ""))
        read[name] = why
    return out, read


def factify2_hosts(listed: Iterable[str]) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    """Factify2's fact-checker image hosts, by site, from every split's CSV."""
    listed = set(listed)
    out, read = [], {}
    for path in sorted(raw_dir("factify2").rglob("*.csv")):
        split = path.stem.lower()
        n = 0
        for row in _dict_rows(path):
            for col in ("claim_image", "document_image"):
                url = (row.get(col) or "").strip()
                if not url.startswith("http"):
                    continue
                host = host_path(url)[0]
                site = FACTIFY2_DOCUMENTED.get(host) or next(
                    (s for s in listed if on_host(host, s)), None)
                if site:
                    out.append((site, "factify2", split, ""))
                    n += 1
        read[split] = n
    return out, read


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def page(source_domain: str | None, url: str | None) -> tuple[str, str]:
    """(host, path) of one store page, exactly as retrieve.EVIDENCE_VIEWS compute it."""
    if re.match(r"https?://web\.archive\.org/web/", url or "", re.I):
        return host_path(url or "", levels=1)
    return (source_domain or "").translate(_ASCII_LOWER), host_path(url or "", levels=0)[1]


def store_pages() -> list[tuple[str, str, str]]:
    """(doc_id, host, path) of every indexed document, Wayback captures by the archived URL."""
    from scripts import retrieve

    con = sqlite3.connect(f"file:{retrieve.TEMPORAL_DB}?mode=ro", uri=True)
    try:
        return [(i, *page(d, u)) for i, d, u in
                con.execute("SELECT doc_id, source_domain, url FROM docs")]
    finally:
        con.close()


def url_key(host: str, path: str) -> str:
    """One page, whatever its scheme, `www.`, query, case or trailing slash."""
    return host + path.rstrip("/")


def blocker(rules: Any) -> Callable[[str, str, str], str | None]:
    """retrieve.SEARCH_SQL's source rules in Python, for one page as `page()` gives it:
    `why(doc_id, host, path)` is "host", "section", "doc", "word" or None (returned).

    `rules` is a retrieve.Blocklist. The SQL also tests a Wayback capture's own
    host, web.archive.org, which the list never names, so the twin skips it.
    """
    listed: dict[str, list[str | None]] = defaultdict(list)
    for h, p in rules.sources:
        listed[h].append(p)
    docs, words = set(rules.doc_ids), tuple(rules.words)

    def why(doc_id: str, host: str, path: str) -> str | None:
        for s in suffixes(host):
            for p in listed.get(s, ()):
                if p is None:
                    return "host"
                if under(path, p):
                    return "section"
        if doc_id in docs:
            return "doc"
        if any(w in path for w in words):
            return "word"
        return None

    return why


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def desk_section(fc_paths: list[str]) -> tuple[str, float] | None:
    """The fact-check section: one to three segments, >= 90% of the URLs, naming a desk."""
    for depth in (1, 2, 3):
        got = section(fc_paths, depth)
        if got and got[1] >= SECTION_SHARE and any(w in got[0] for w in FACT_CHECK_WORDS):
            return got
    return None


def decide(host: str, fc_paths: list[str], pages: list[str],
           cited_docs: list[str] | None = None) -> dict[str, Any]:
    """Scope one host from its fact-check URL paths, its store pages' paths, and the
    store documents that are its cited fact-checks."""
    total = len(pages)
    whole = {"scope": "domain", "store_docs": total, "store_blocked": total}
    if total <= STORE_FLOOR or any(w in host for w in CHECKER_NAME_WORDS):
        return whole
    got = desk_section(fc_paths)
    if got is None:
        docs = sorted(set(cited_docs or []))
        return {"scope": "docs", "doc_ids": docs, "store_docs": total, "store_blocked": len(docs),
                "review": (f"{total} store documents and no fact-check section separates them: "
                           f"only its {len(docs)} cited fact-check(s) in the store are blocked")}
    prefix, share = got
    inside = sum(1 for p in pages if under(p, prefix))
    if total - inside <= STORE_FLOOR:
        return whole
    return {"scope": "path", "paths": [prefix], "section_share": round(share, 3),
            "store_docs": total, "store_blocked": inside}


def by_hand(host: str, pages: list[str]) -> dict[str, Any]:
    """A host a person added: blocked whole, whatever its store pages or sections."""
    return {"scope": "domain", "store_docs": len(pages), "store_blocked": len(pages),
            "added_by_hand": ADDED_BY_HAND[host]}


def only_from_test(entries: list[dict[str, Any]]) -> list[str]:
    """Hosts only test files cite. A host nothing cites (added by hand) is not one."""
    return sorted(e["host"] for e in entries
                  if e["cited_by"] and all(set(s) == {"test"} for s in e["cited_by"].values()))


def decided(entry: dict[str, Any]) -> dict[str, Any]:
    """A docs-scoped entry a person has decided carries the decision, not `review`."""
    if entry["scope"] != "docs" or entry["host"] not in DECIDED_DOCS:
        return entry
    out = {("decided" if k == "review" else k): v for k, v in entry.items()}
    out["decided"] = DECIDED_DOCS[entry["host"]]
    return out


def build() -> int:
    import yaml

    from scripts import retrieve

    started = time.perf_counter()
    cites, read = citations()
    listed = {c[0] for c in cites}
    f2, read["factify2"] = factify2_hosts(listed)
    cites += f2
    by_host: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    paths: dict[str, list[str]] = defaultdict(list)
    cited: dict[str, set[str]] = defaultdict(set)
    for host, dataset, split, path in cites:
        by_host[host][dataset].add(split)
        if path:
            paths[host].append(path)
            cited[host].add(url_key(host, path))
    pages: dict[str, list[str]] = defaultdict(list)
    cited_docs: dict[str, list[str]] = defaultdict(list)
    hosts = sorted(set(by_host) | set(ADDED_BY_HAND))
    known = set(hosts)
    store = store_pages()
    for doc_id, h, p in store:
        for s in suffixes(h):
            if s in known:
                pages[s].append(p)
                if url_key(h, p) in cited[s]:
                    cited_docs[s].append(doc_id)
    entries = []
    for h in hosts:
        scope = (by_hand(h, pages[h]) if h in ADDED_BY_HAND
                 else decide(h, paths[h], pages[h], cited_docs[h]))
        e = {"host": h, **scope,
             "cited_by": {d: sorted(s) for d, s in sorted(by_host.get(h, {}).items())},
             "fact_check_urls": len(paths[h]),
             "cited_fact_checks_in_store": len(set(cited_docs[h]))}
        entries.append(decided(e))
    only_test = only_from_test(entries)
    # what the rules block in the store, each page counted under the first rule
    # that takes it, in the order the twin tests them
    words = list(FACT_CHECK_WORDS)
    why = blocker(retrieve.blocklist_from({"hosts": entries, "path_words": {"words": words}},
                                          "the build"))
    by_rule: Counter = Counter()
    by_word: Counter = Counter()
    word_hosts: set[str] = set()
    for doc_id, h, p in store:
        rule = why(doc_id, h, p)
        by_rule[rule or "returned"] += 1
        if rule == "word":
            word_hosts.add(h)
            by_word.update(w for w in words if w in p)
    doc = {
        "version": 1,
        "built": time.strftime("%Y-%m-%d"),
        "generated_by": "python scripts/factcheckers.py build",
        "reads_test_splits": ("yes, by the person's decision of 2026-10-28: a domain list carries no "
                              "label, and a fact-checker we miss leaks answers into test evidence"),
        "rule": {"store_floor": STORE_FLOOR, "section_share": SECTION_SHARE,
                 "section_words": list(FACT_CHECK_WORDS),
                 "match": "the host or any subdomain; Wayback captures by the archived URL; "
                          "path prefixes on the lower-cased path"},
        "read": read,
        "counts": {"hosts": len(entries),
                   "domain": sum(e["scope"] == "domain" for e in entries),
                   "path": sum(e["scope"] == "path" for e in entries),
                   "docs": sum(e["scope"] == "docs" for e in entries),
                   "review": sum("review" in e for e in entries),
                   "decided": sum("decided" in e for e in entries),
                   "added_by_hand": sum("added_by_hand" in e for e in entries),
                   "only_from_test": len(only_test),
                   "store_docs_blocked": sum(by_rule[r] for r in ("host", "section", "doc", "word")),
                   "store_docs_blocked_by_rule": {r: by_rule[r] for r in ("host", "section", "doc",
                                                                          "word")}},
        "only_from_test": only_test,
        "path_words": {
            "words": words,
            "match": "a substring of the page's lower-cased path (a Wayback capture's: the "
                     "archived URL's), on every host; never the host, the query or the fragment",
            "decided": PATH_WORDS_DECIDED,
            "store_blocked": by_rule["word"],
            "store_blocked_hosts": len(word_hosts),
            "store_blocked_by_word": {w: by_word[w] for w in words},
        },
        "hosts": entries,
    }
    header = ("# Fact-checker blocklist for retrieval. Generated, then reviewed by a person.\n"
              "# Regenerate with `python scripts/factcheckers.py build`; read by retrieve.connect().\n")
    FACTCHECKERS_YAML.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                                         width=100), encoding="utf-8")
    c = doc["counts"]
    print(f"  {c['hosts']} hosts: {c['domain']} domain, {c['path']} path, {c['docs']} docs "
          f"({c['decided']} decided, {c['review']} for review); {c['added_by_hand']} added by "
          f"hand; {c['only_from_test']} only from "
          f"test; {c['store_docs_blocked']:,} store documents blocked {c['store_docs_blocked_by_rule']} "
          f"({time.perf_counter() - started:.0f} s)")
    print(f"  wrote {FACTCHECKERS_YAML.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


def residual() -> int:
    """What the path words take beyond the host rules, and what nothing takes: store
    pages on unlisted hosts whose NAME reads like a fact-checker (never blocked by name)."""
    from scripts import retrieve

    rules = retrieve.load_blocklist()
    why = blocker(rules)
    hosts_only = blocker(retrieve.Blocklist(rules.sources, rules.doc_ids, ()))
    listed = {h for h, _ in rules.sources}
    by_words: Counter = Counter()
    named: dict[str, Counter] = {"path word": Counter(), "checker word": Counter()}
    for doc_id, h, p in store_pages():
        if why(doc_id, h, p) is None:
            if any(s in listed for s in suffixes(h)):
                continue
            if any(w in h for w in FACT_CHECK_WORDS):
                named["path word"][h] += 1
            elif any(w in h for w in CHECKER_NAME_WORDS):
                named["checker word"][h] += 1
        elif hosts_only(doc_id, h, p) is None:
            by_words[h] += 1
    print(f"  the path words block {sum(by_words.values()):,} store pages on {len(by_words)} "
          f"hosts beyond the host rules; the most:")
    for h, n in by_words.most_common(15):
        print(f"    {n:>5}  {h}")
    for kind, c in named.items():
        print(f"  returned: {sum(c.values()):,} store pages on {len(c)} unlisted hosts whose name "
              f"carries a {kind} ({'FACT_CHECK_WORDS' if kind == 'path word' else 'CHECKER_NAME_WORDS'}"
              f"); the most:")
        for h, n in c.most_common(25):
            print(f"    {n:>5}  {h}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("build", "residual"))
    a = p.parse_args(argv)
    return build() if a.cmd == "build" else residual()


if __name__ == "__main__":
    raise SystemExit(main())
