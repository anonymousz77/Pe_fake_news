"""The retrieval contract's rules on fixtures: strictly before, and no fact-checker.

scripts/redtest.py proves the same on the real 1.4M-document index. These run in
milliseconds with no store on disk, which is what a contract test needs to be.
"""

from __future__ import annotations

import sqlite3

import pytest
import yaml

from configs.paths import AS_OF_YAML, FACTCHECKERS_YAML, dataset, dataset_names
from scripts import factcheckers, retrieve

#: The path words here are ones no other test URL carries, so each rule is
#: tested on its own: "/fact-checkers/x" must still show a section is a prefix.
BLOCKLIST = """
hosts:
  - {host: snopes.com, scope: domain}
  - {host: indiatoday.in, scope: path, paths: [/fact-check]}
  - {host: bbc.co.uk, scope: none}
  - {host: reuters.com, scope: docs, doc_ids: [ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff]}
path_words:
  words: [webqoof, hoax-alert, verificador]
"""

WORDS = "path_words:\n  words: [webqoof]\n"

LISTED_DOC = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

WAYBACK = "https://web.archive.org/web/20200101000000/"


@pytest.fixture
def blocklist(tmp_path):
    p = tmp_path / "factcheckers.yaml"
    p.write_text(BLOCKLIST, encoding="utf-8")
    return p


def _index(tmp_path, rows):
    """rows = [(doc_id, published_at | None, source_domain, url)]; every text says 'vaccine'."""
    index = tmp_path / "index"
    index.mkdir(parents=True, exist_ok=True)
    bm, tm = index / "bm25.sqlite", index / "temporal.sqlite"
    b = sqlite3.connect(bm)
    b.execute("CREATE VIRTUAL TABLE docs_fts USING fts5("
              "doc_id UNINDEXED, title, text, tokenize='porter unicode61')")
    t = sqlite3.connect(tm)
    t.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, published_at TEXT "
              "NOT NULL, source_domain TEXT, url TEXT)")
    for doc_id, pub, domain, url in rows:
        b.execute("INSERT INTO docs_fts VALUES (?,?,?)", (doc_id, "vaccine study", "vaccine trial"))
        if pub is not None:
            t.execute("INSERT INTO docs VALUES (?,?,?,?)", (doc_id, pub, domain, url))
    b.commit(); t.commit(); b.close(); t.close()
    return bm, tm


def _ids(tmp_path, rows, blocklist, before="2020-06-15", k=100):
    bm, tm = _index(tmp_path, rows)
    con = retrieve.connect(bm, tm, blocklist)
    try:
        return {h.doc_id for h in retrieve.search("vaccine", before, k=k, con=con)}
    finally:
        con.close()


def test_only_documents_strictly_before_the_bound_come_back(tmp_path, blocklist):
    ids = _ids(tmp_path, [
        ("prior", "2020-06-14", "x.example", "https://x.example/a"),
        ("same_day", "2020-06-15", "x.example", "https://x.example/b"),
        ("future", "2020-06-16", "x.example", "https://x.example/c"),
        ("null", None, None, None),
    ], blocklist)
    assert ids == {"prior"}, ids


def test_fact_checker_pages_never_come_back(tmp_path, blocklist):
    blocked = {
        "fc_domain": ("snopes.com", "https://www.snopes.com/fact-check/x/"),
        "fc_subdomain": ("media.snopes.com", "https://media.snopes.com/x"),
        "fc_upper_case": ("SNOPES.COM", "https://WWW.SNOPES.COM/x"),
        "fc_wayback": ("web.archive.org", WAYBACK + "https://www.snopes.com/fact-check/x/"),
        "fc_wayback_no_scheme": ("web.archive.org", WAYBACK + "www.snopes.com/x"),
        "fc_wayback_port": ("web.archive.org", WAYBACK + "http://snopes.com:80/x"),
        "desk_section": ("indiatoday.in", "https://www.indiatoday.in/fact-check/story/x"),
        "desk_exact": ("indiatoday.in", "https://www.indiatoday.in/fact-check"),
        "desk_query": ("indiatoday.in", "https://www.indiatoday.in/fact-check?page=2"),
        "desk_upper_path": ("indiatoday.in", "https://www.indiatoday.in/FACT-CHECK/story/x"),
        "desk_wayback": ("web.archive.org", WAYBACK + "https://www.indiatoday.in/fact-check/x"),
    }
    kept = {
        "desk_news": ("indiatoday.in", "https://www.indiatoday.in/india/story/x"),
        "desk_lookalike_path": ("indiatoday.in", "https://www.indiatoday.in/fact-checkers/x"),
        "lookalike_prefix": ("notsnopes.com", "https://notsnopes.com/x"),
        "lookalike_suffix": ("snopes.com.example.org", "https://snopes.com.example.org/x"),
        "listed_as_none_is_not_blocked": ("bbc.co.uk", "https://www.bbc.co.uk/news/x"),
        "wayback_of_news": ("web.archive.org", WAYBACK + "https://www.indiatoday.in/india/x"),
    }
    rows = [(n, "2020-06-01", d, u) for n, (d, u) in {**blocked, **kept}.items()]
    ids = _ids(tmp_path, rows, blocklist)
    assert not ids & set(blocked), f"fact-checker pages came back: {sorted(ids & set(blocked))}"
    assert ids == set(kept), f"wrongly blocked: {sorted(set(kept) - ids)}"


def test_a_path_word_blocks_on_every_host(tmp_path, blocklist):
    """The person's rule of 2026-10-28: a fact-check desk word in the path, any host."""
    blocked = {
        "word_segment": ("news.example", "https://news.example/webqoof/x"),
        "word_upper_case": ("news.example", "https://news.example/WEBQOOF/x"),
        "word_in_slug": ("news.example", "https://news.example/2020/06/viral-hoax-alert-x"),
        "word_wayback": ("web.archive.org", WAYBACK + "https://news.example/verificador/x"),
        "word_on_a_docs_host": ("reuters.com", "https://www.reuters.com/article/hoax-alert-x"),
        "word_on_a_none_host": ("bbc.co.uk", "https://www.bbc.co.uk/news/webqoof-x"),
    }
    kept = {
        "same_host_no_word": ("news.example", "https://news.example/news/x"),
        "word_in_the_query_only": ("news.example", "https://news.example/news/y?tag=webqoof"),
        "word_in_the_fragment_only": ("news.example", "https://news.example/news/z#webqoof"),
        "word_in_the_host_only": ("webqoof.example", "https://webqoof.example/news/x"),
        "wayback_word_in_archived_host_only": ("web.archive.org",
                                               WAYBACK + "https://webqoof.example/news/y"),
    }
    rows = [(n, "2020-06-01", d, u) for n, (d, u) in {**blocked, **kept}.items()]
    ids = _ids(tmp_path, rows, blocklist)
    assert not ids & set(blocked), f"path-word pages came back: {sorted(ids & set(blocked))}"
    assert ids == set(kept), f"wrongly blocked: {sorted(set(kept) - ids)}"


def test_the_python_twin_blocks_exactly_what_the_sql_blocks(tmp_path, blocklist):
    """factcheckers.blocker judges real rows in the red-test, so it must be the SQL."""
    urls = URLS + [("news.example", "https://news.example/webqoof/x"),
                   ("web.archive.org", WAYBACK + "https://news.example/verificador/x"),
                   ("web.archive.org", WAYBACK + "https://webqoof.example/x"),
                   ("indiatoday.in", "https://www.indiatoday.in/fact-checkers/x"),
                   ("reuters.com", "https://www.reuters.com/article/x")]
    rows = [(f"d{i}", "2020-01-01", d, u) for i, (d, u) in enumerate(urls)]
    rows.append((LISTED_DOC, "2020-01-01", "reuters.com", "https://www.reuters.com/article/y"))
    returned = _ids(tmp_path, rows, blocklist)
    why = factcheckers.blocker(retrieve.load_blocklist(blocklist))
    for doc_id, _, domain, url in rows:
        assert (doc_id in returned) == (why(doc_id, *factcheckers.page(domain, url)) is None), url


def test_a_document_the_list_names_never_comes_back(tmp_path, blocklist):
    """A publisher whose fact-checks share no section: only its cited fact-check goes."""
    ids = _ids(tmp_path, [
        (LISTED_DOC, "2020-06-01", "reuters.com", "https://www.reuters.com/article/idus1"),
        ("reuters_news", "2020-06-01", "reuters.com", "https://www.reuters.com/article/idus2"),
    ], blocklist)
    assert ids == {"reuters_news"}, ids


def test_k_still_means_k_with_the_blocklist(tmp_path, blocklist):
    """The rules sit in the query: blocked pages never cost a slot."""
    rows = [(f"fc{i}", "2020-06-01", "snopes.com", f"https://snopes.com/{i}") for i in range(5)]
    rows += [("ok1", "2020-06-01", "x.example", "https://x.example/1"),
             ("ok2", "2020-06-01", "x.example", "https://x.example/2")]
    assert _ids(tmp_path, rows, blocklist, k=2) == {"ok1", "ok2"}


def test_the_statement_streams_in_rank_order(tmp_path, blocklist):
    """FTS5 consumes ORDER BY rank: no temp b-tree, so the join stops at k."""
    bm, tm = _index(tmp_path, [("a", "2020-01-01", "x.example", "https://x.example/a")])
    con = retrieve.connect(bm, tm, blocklist)
    try:
        plan = [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + retrieve.SEARCH_SQL,
                                          {"query": '"vaccine"', "before": "2020-06-15", "k": 20})]
    finally:
        con.close()
    assert not any("TEMP B-TREE" in step for step in plan), plan
    assert any(step.startswith("SCAN f VIRTUAL TABLE INDEX 32") for step in plan), plan


def test_a_connection_without_the_rules_cannot_search(tmp_path):
    """Opened any other way, the statement fails instead of returning a fact-check."""
    bm, tm = _index(tmp_path, [("a", "2020-01-01", "snopes.com", "https://snopes.com/a")])
    con = sqlite3.connect(bm)
    con.execute("ATTACH DATABASE ? AS temporal", (str(tm),))
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            con.execute(retrieve.SEARCH_SQL, {"query": '"vaccine"', "before": "2020-06-15",
                                              "k": 5}).fetchall()
    finally:
        con.close()


def test_a_missing_or_empty_blocklist_is_refused(tmp_path):
    with pytest.raises(retrieve.RetrievalError, match="factcheckers.py build"):
        retrieve.load_blocklist(tmp_path / "absent.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("hosts:\n  - {host: bbc.co.uk, scope: none}\n" + WORDS, encoding="utf-8")
    with pytest.raises(retrieve.RetrievalError, match="blocks no host"):
        retrieve.load_blocklist(empty)
    for words in ("", "path_words:\n  words: []\n"):
        no_words = tmp_path / "no_words.yaml"
        no_words.write_text("hosts:\n  - {host: snopes.com, scope: domain}\n" + words,
                            encoding="utf-8")
        with pytest.raises(retrieve.RetrievalError, match="blocks no path word"):
            retrieve.load_blocklist(no_words)


@pytest.mark.parametrize("entry", [
    "{host: www.snopes.com, scope: domain}",
    "{host: Snopes.com, scope: domain}",
    "{host: indiatoday.in, scope: path, paths: [/fact-check/]}",
    "{host: indiatoday.in, scope: path, paths: [fact-check]}",
    "{host: snopes.com, scope: everything}",
    "{host: reuters.com, scope: docs, doc_ids: [not-a-sha256]}",
])
def test_a_malformed_blocklist_entry_is_refused(tmp_path, entry):
    p = tmp_path / "bad.yaml"
    p.write_text(f"hosts:\n  - {entry}\n" + WORDS, encoding="utf-8")
    with pytest.raises(retrieve.RetrievalError):
        retrieve.load_blocklist(p)


@pytest.mark.parametrize("word", ["Webqoof", "fac", "fact check", "/fact-check", "a?b#c",
                                  "vérifié", 7])
def test_a_malformed_path_word_is_refused(tmp_path, word):
    """A short or odd word would block by accident, and SQLite's lower() folds ASCII only."""
    p = tmp_path / "bad.yaml"
    p.write_text("hosts:\n  - {host: snopes.com, scope: domain}\n"
                 f"path_words:\n  words: [{word!r}]\n".replace("'", '"'), encoding="utf-8")
    with pytest.raises(retrieve.RetrievalError, match="path word"):
        retrieve.load_blocklist(p)


URLS = [
    ("snopes.com", "https://www.snopes.com/fact-check/x/?utm=1#top"),
    ("x.example", "http://x.example"),
    ("x.example", "https://x.example:8443/A/B"),
    ("x.example", "x.example/no-scheme/Path"),
    ("web.archive.org", WAYBACK + "https://www.Politifact.com/factchecks/2020/x/"),
    ("web.archive.org", WAYBACK + "www.africacheck.org/reports/x"),
    ("web.archive.org", "https://web.archive.org/web/2020id_/http://snopes.com:80/x?q=1"),
    ("web.archive.org", "https://web.archive.org/save/https://snopes.com/x"),
    ("indiatoday.in", "https://www.indiatoday.in/FACT-CHECK"),
    ("web.archive.org", WAYBACK + "https://www.npr.org?t=1623307033650"),
    ("web.archive.org", WAYBACK + "https://snopes.com#top"),
]


def test_the_sql_and_python_twins_agree(tmp_path, blocklist):
    """retrieve.EVIDENCE_VIEWS and factcheckers.page() must compute one host and path."""
    rows = [(f"d{i}", "2020-01-01", d, u) for i, (d, u) in enumerate(URLS)]
    bm, tm = _index(tmp_path, rows)
    con = retrieve.connect(bm, tm, blocklist)
    try:
        got = {r["doc_id"]: r for r in con.execute("SELECT * FROM temp.evidence")}
    finally:
        con.close()
    for i, (domain, url) in enumerate(URLS):
        r = got[f"d{i}"]
        sql = ((r["archived_host"], r["archived_path"]) if r["wayback"]
               else (domain.lower(), r["path"]))
        assert sql == factcheckers.page(domain, url), url


@pytest.mark.parametrize("name, meta, expected", [
    ("averitec", {"claim_date": "31-10-2020"}, ("2020-10-31", "claim_date")),
    ("averitec", {"claim_date": "9-7-2020"}, ("2020-07-09", "claim_date")),
    ("averimatec", {"date": "2023-03-17"}, ("2023-03-17", "claim_date")),
    ("fakeddit", {"created_utc": "1347846272.0"}, ("2012-09-17", "claim_date")),
    ("isot", {"date": "August 4, 2016 "}, ("2016-08-04", "claim_date")),
    ("isot", {"date": "Dec 31, 2017"}, ("2017-12-31", "claim_date")),
    ("isot", {"date": "19-Feb-18"}, ("2018-02-19", "claim_date")),
    ("m4fc", {"fc_pub_date": "2021-12-19 19:36:09"}, ("2021-12-19", "fact_check_date")),
    ("m4fc", {"fc_pub_date": "2020-11-28"}, ("2020-11-28", "fact_check_date")),
])
def test_evidence_bound_reads_each_datasets_date(name, meta, expected):
    assert retrieve.evidence_bound(name, meta) == expected


@pytest.mark.parametrize("name, meta", [
    ("averitec", {"claim_date": ""}),
    ("averitec", {}),
    ("isot", {"date": "https://100percentfedup.com/x/"}),
    ("liar", {"speaker": "someone"}),
    ("verite", {}),
    ("no_such_dataset", {}),
])
def test_no_date_means_no_evidence(name, meta):
    with pytest.raises(retrieve.RetrievalError):
        retrieve.evidence_bound(name, meta)


def test_the_fact_check_date_is_only_a_fallback(tmp_path):
    rules = tmp_path / "as_of.yaml"
    rules.write_text("datasets:\n  both:\n"
                     "    claim_date: {field: claim, formats: [iso]}\n"
                     "    fact_check_date: {field: fc, formats: [iso]}\n", encoding="utf-8")
    meta = {"claim": "2020-01-02", "fc": "2020-03-04"}
    assert retrieve.evidence_bound("both", meta, rules) == ("2020-01-02", "claim_date")
    assert retrieve.evidence_bound("both", {"fc": "2020-03-04"}, rules) == ("2020-03-04",
                                                                             "fact_check_date")


def test_as_of_rules_cover_exactly_the_enabled_datasets():
    rules = yaml.safe_load(AS_OF_YAML.read_text(encoding="utf-8"))["datasets"]
    enabled = {n for n in dataset_names() if dataset(n).get("enabled")}
    assert set(rules) == enabled
    for name, rule in rules.items():
        assert set(rule) >= {"claim_date", "fact_check_date"}, name


def test_the_shipped_blocklist_loads_and_says_it_reads_test_splits():
    doc = yaml.safe_load(FACTCHECKERS_YAML.read_text(encoding="utf-8"))
    assert "reads_test_splits" in doc
    rules = retrieve.load_blocklist()
    assert rules.sources and all(p is None or p.count("/") <= 3 for _, p in rules.sources)
    assert set(rules.words) == set(factcheckers.FACT_CHECK_WORDS), "the path words drifted"
    assert "decided" in doc["path_words"]
    for e in doc["hosts"]:
        if e["scope"] == "docs":   # a person has decided it, or it waits for one
            assert ("decided" in e) != ("review" in e), e["host"]
    by_hand = {e["host"]: e for e in doc["hosts"] if "added_by_hand" in e}
    assert set(by_hand) == set(factcheckers.ADDED_BY_HAND), "the hosts added by hand drifted"
    for host, e in by_hand.items():   # blocked whole, cited by nothing, so never "only from test"
        assert e["scope"] == "domain" and (host, None) in rules.sources and not e["cited_by"], host
        assert host not in doc["only_from_test"], host


# --------------------------------------------------------------------------
# the words a query is searched by: P4b by default, every word when exact
# --------------------------------------------------------------------------

#: Twenty documents, all with "the" (so in half the store or more); the i-th
#: word of WORDS in the first i+1 of them; the last eleven hold "the" alone.
WORDS_BY_DF = "alpha bravo charlie delta echo foxtrot golf hotel india".split()
QUERY = "the india alpha hotel bravo golf charlie foxtrot delta echo"


def _words_index(tmp_path):
    index = tmp_path / "words"
    index.mkdir(parents=True, exist_ok=True)
    bm, tm = index / "bm25.sqlite", index / "temporal.sqlite"
    b = sqlite3.connect(bm)
    b.execute("CREATE VIRTUAL TABLE docs_fts USING fts5("
              "doc_id UNINDEXED, title, text, tokenize='porter unicode61')")
    t = sqlite3.connect(tm)
    t.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, published_at TEXT "
              "NOT NULL, source_domain TEXT, url TEXT)")
    for d in range(20):
        text = " ".join(["the"] + [w for i, w in enumerate(WORDS_BY_DF) if d <= i])
        b.execute("INSERT INTO docs_fts VALUES (?,?,?)", (f"d{d}", "", text))
        t.execute("INSERT INTO docs VALUES (?,?,?,?)",
                  (f"d{d}", "2019-01-01", "news.example", f"https://news.example/{d}"))
    b.commit(); t.commit(); b.close(); t.close()
    return bm, tm


@pytest.fixture
def words_con(tmp_path, blocklist):
    con = retrieve.connect(*_words_index(tmp_path), blocklist)
    yield con
    con.close()


def test_the_counts_are_what_bm25_counts(words_con):
    df, n_docs = retrieve.frequencies(words_con, retrieve.words(QUERY + " The absent"))
    assert n_docs == 20
    assert df == {"the": 20, "absent": 0, **{w: i + 1 for i, w in enumerate(WORDS_BY_DF)}}


def test_the_default_searches_the_8_rarest_words_outside_half_the_store(words_con):
    terms, fell = retrieve.query_terms(words_con, QUERY)
    # "the" is in all 20 documents and "india" is the 9th rarest: both left out
    assert terms == ["alpha", "hotel", "bravo", "golf", "charlie", "foxtrot", "delta", "echo"]
    assert not fell
    assert retrieve.query_terms(words_con, QUERY, exact=True) == (QUERY.split(), False)
    ids = lambda **kw: {h.doc_id for h in retrieve.search(QUERY, "2020-01-01", k=100,
                                                          con=words_con, **kw)}
    assert ids() == {f"d{d}" for d in range(8)}          # the documents with a kept word
    assert ids(exact=True) == {f"d{d}" for d in range(20)}


def test_a_query_with_every_word_in_half_the_store_falls_back_to_its_rarest(words_con):
    assert retrieve.query_terms(words_con, "the THE") == (["the"], True)
    hits = retrieve.search("the", "2020-01-01", k=100, con=words_con)
    assert len(hits) == 20


def test_a_one_word_query_is_searched_the_same_both_ways(words_con):
    """Why the red-test, whose probes are one word each, covers the default and exact alike."""
    for word in ("the", "india", "alpha", "absent"):
        assert (retrieve.query_terms(words_con, word)[0]
                == retrieve.query_terms(words_con, word, exact=True)[0] == [word])


def test_search_is_query_terms_then_the_statement(words_con):
    """scripts/retrieval_bench.py times the default in these two steps."""
    for exact in (False, True):
        terms, _ = retrieve.query_terms(words_con, QUERY, exact)
        rows = words_con.execute(retrieve.SEARCH_SQL, {"query": retrieve.match_expr(terms),
                                                       "before": "2020-01-01", "k": 100}).fetchall()
        hits = retrieve.search(QUERY, "2020-01-01", k=100, con=words_con, exact=exact)
        assert [(h.doc_id, h.score) for h in hits] == [(r["doc_id"], r["score"]) for r in rows]


def test_an_empty_query_is_refused_both_ways(words_con):
    for exact in (False, True):
        with pytest.raises(retrieve.RetrievalError, match="empty query"):
            retrieve.search("?! --", "2020-01-01", con=words_con, exact=exact)


def test_the_cli_searches_every_word_only_when_asked(monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(retrieve, "search",
                        lambda query, before, k, exact=False: asked.append(exact) or [])
    assert retrieve.main(["--query", "x", "--before", "2020-01-01"]) == 0
    assert retrieve.main(["--query", "x", "--before", "2020-01-01", "--exact"]) == 0
    assert asked == [False, True]
    assert "every word" in capsys.readouterr().out
