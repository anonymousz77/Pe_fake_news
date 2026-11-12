"""The retrieval bench's pure parts: term selection, overlap, the equivalence check. Offline."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from scripts import retrieval_bench as b
from scripts import retrieve

N = 1000
DF = {"the": 900, "a": 600, "photo": 120, "shows": 40, "modi": 9, "kerala": 3, "flood": 15,
      "in": 700, "2018": 30}


def test_the_bench_builds_the_query_retrieve_builds():
    for claim in ("Photo shows Modi in Kerala, 2018!", "a/b c-d", "Ünïcode wörds"):
        assert b.match_expr(b.tokens(claim)) == retrieve._sanitise(claim)


def test_every_option_keeps_every_word_or_prunes_with_the_df_table():
    """L crashed on its first query (2026-10-28): it was neither an E option nor given df."""
    claim = "the photo shows Modi in Kerala"
    ents = [{"type": "PER", "text": "Modi"}]
    assert set(b.EVERY_WORD) <= set(b.OPTIONS) and set(b.SESSION) <= set(b.OPTIONS)
    for option in b.OPTIONS:
        if option in b.EVERY_WORD:
            assert b.select_terms(option, claim, [], None, 0) == (b.tokens(claim), False), option
        else:
            rule = b.pruning(option)
            assert rule in ("P1", "P2", "P3", "P4", "P4b"), f"{option} is neither EVERY_WORD nor pruned"
            with pytest.raises(ValueError, match="df"):
                b.select_terms(option, claim, ents, None, 0)
            assert b.select_terms(option, claim, ents, DF, N) == b.select_terms(rule, claim, ents,
                                                                                DF, N), option


def test_l_and_l_plus_options_run_with_the_leak_rules_and_nothing_else_does():
    assert [o for o in b.OPTIONS if b.with_rules(o)] == ["L", "L+P2", "L+P3", "L+P4", "L+P4b"]
    for option in b.OPTIONS:
        assert (b.sql_for(option) == b.retrieve.SEARCH_SQL) == b.with_rules(option), option


def test_every_option_gets_every_parameter_its_statement_names():
    params = b.params_for({"as_of": "2020-01-01"}, ["x"])
    for option in b.OPTIONS:
        named = set(re.findall(r":(\w+)", b.sql_for(option)))
        assert named and named <= set(params), (option, named - set(params))
    for option in b.SESSION:
        assert ":before" in b.sql_for(option) and ":as_of" not in b.sql_for(option), option
    assert ":as_of" in b.sql_for("E1") and ":as_of" in b.sql_for("E1+")


def test_a_williams_design_balances_position_and_what_ran_just_before():
    for n in (2, 3, 4, 5, 6):
        rows = b.williams(n)
        reps = 1 if n % 2 == 0 else 2      # an odd n takes the square and its mirror
        assert len(rows) == reps * n
        assert all(sorted(r) == list(range(n)) for r in rows)
        assert all(Counter(col) == {o: reps for o in range(n)} for col in zip(*rows))
        pairs = Counter((r[i], r[i + 1]) for r in rows for i in range(n - 1))
        assert len(pairs) == n * (n - 1) and set(pairs.values()) == {reps}


def test_the_session_balances_order_over_the_200_queries():
    n, cycle = len(b.SESSION), len(b.williams(len(b.SESSION)))
    assert 200 % cycle == 0, "the 200 queries must cover whole cycles of the design"
    orders = [b.session_order(i) for i in range(200)]
    for o in b.SESSION:
        assert Counter(order.index(o) for order in orders) == {p: 200 // n for p in range(n)}, o
        if o != "L":    # a pruned option reads its terms before L does, cold, for half
            assert sum(order.index(o) < order.index("L") for order in orders) == 100, o
    after = Counter((order[i], order[i + 1]) for order in orders for i in range(n - 1))
    assert len(after) == n * (n - 1) and set(after.values()) == {200 // n}


def _q(rid, entities, india=False):
    return {"record_id": rid, "entities": [{"type": "PER", "text": e} for e in entities],
            "india": india}


def test_entities_against_the_baseline_count_queries_gained_and_lost():
    qs = [_q("q1", ["Modi", "Kerala"], india=True), _q("q2", ["Biden"]), _q("q3", ["Paris"])]
    texts = {"a": "modi spoke in kerala", "b": "nothing here", "c": "biden", "d": "paris"}
    run = lambda *tops: {q["record_id"]: {"hits": [[d] for d in t]} for q, t in zip(qs, tops)}
    mine, base = run(["a"], ["b"], ["d"]), run(["b"], ["c"], ["d"])
    got = b.entities_against(qs, mine, base, texts)
    assert got["all"]["1+"] == {"gained": 1, "lost": 1}      # q1 gained, q2 lost; q3 both
    assert got["all"]["2+"] == {"gained": 1, "lost": 0}
    assert got["india"]["1+"] == {"gained": 1, "lost": 0}
    assert b.entity_matches(qs, mine, texts)["all"] == {"queries": 3, "1+": 2, "2+": 1}


def test_the_rules_added_since_ls_pass_explain_a_query_only_by_pages_they_block():
    page = lambda i, s, host, path: [i, s, "2020-01-01", host, f"https://{host}/{path}"]
    news = lambda i, s: page(f"n{i}", s, "news.example", i)
    qs = [{"record_id": "q1"}, {"record_id": "q2"}, {"record_id": "q3"}]
    before = {"q1": {"hits": [news(1, -9), page("w2", -8, "news.example", "fact-check/2"),
                              news(3, -7)]},
              "q2": {"hits": [news(4, -9), news(5, -8)]},
              "q3": {"hits": [page("h8", -9, "hoax.sk", "8"), news(9, -8)]}}
    after = {"q1": {"hits": [news(1, -9), news(3, -7), news(6, -6)]},    # a path-word page left
             "q2": {"hits": [news(4, -9), news(7, -8.5)]},               # a page no rule blocks left
             "q3": {"hits": [news(9, -8), news(10, -7)]}}                # a hand-added host's left
    got = b.rules_added_effect(qs, before, after, {"blocklist": {"sources": [], "doc_ids": []}})
    assert got["queries_changed"] == 3 and got["hits_removed"] == 3
    assert got["removed_by_todays_rules"] == 2 and got["queries_explained"] == 2
    assert got["removed_by_rule"] == {"word": 1, "host": 1}
    assert got["rules_since"]["added"]["words"] == len(b.retrieve.load_blocklist().words)


def test_e_options_keep_every_word_with_repeats():
    terms, fell = b.select_terms("E1", "the photo shows the flood", [], DF, N)
    assert terms == ["the", "photo", "shows", "the", "flood"] and not fell


def test_p1_drops_only_the_terms_fts5_scores_at_idf_one_millionth():
    # df >= N/2 clamps idf to 1e-6; repeats of every other term stay, so scores barely move
    terms, _ = b.select_terms("P1", "the photo shows a Modi photo in Kerala", [], DF, N)
    assert terms == ["photo", "shows", "Modi", "photo", "Kerala"]


def test_p2_keeps_terms_in_at_most_one_percent_of_documents_once_each():
    terms, _ = b.select_terms("P2", "the photo shows Modi Modi in Kerala", [], DF, N)
    assert terms == ["Modi", "Kerala"]


def test_p3_is_the_entities_and_the_five_rarest_others():
    ents = [{"type": "PER", "text": "Modi"}, {"type": "LOC", "text": "Kerala"}]
    terms, _ = b.select_terms("P3", "the photo shows Modi in Kerala flood 2018", ents, DF, N)
    # the commonest non-entity word ("the") is the one left out
    assert terms == ["Modi", "Kerala", "photo", "shows", "in", "flood", "2018"]


def test_p4_is_the_eight_rarest_in_claim_order():
    terms, _ = b.select_terms("P4", "the photo shows Modi in Kerala flood 2018 a", [], DF, N)
    # nine distinct words; the commonest ("the") is the one left out
    assert terms == ["photo", "shows", "Modi", "in", "Kerala", "flood", "2018", "a"]


def test_p4b_is_p4_less_the_terms_fts5_scores_at_idf_one_millionth():
    claim = "the photo shows Modi in Kerala flood 2018 a"
    p4, _ = b.select_terms("P4", claim, [], DF, N)
    terms, fell = b.select_terms("P4b", claim, [], DF, N)
    # "in" (700) and "a" (600) are in half the documents or more: P4 kept them, P4b does not
    assert terms == [t for t in p4 if DF[t.lower()] < N / 2] == [
        "photo", "shows", "Modi", "Kerala", "flood", "2018"] and not fell
    rare = {**DF, **{w: 5 + i for i, w in enumerate("one two three four five six seven eight".split())}}
    claim = "the one two three four five six seven eight shows"    # 8 rarer words: P4b is P4
    assert b.select_terms("P4b", claim, [], rare, N) == b.select_terms("P4", claim, [], rare, N)


def test_an_option_that_keeps_nothing_falls_back_to_the_rarest_word():
    for option in ("P2", "P4b"):
        terms, fell = b.select_terms(option, "the a in", [], DF, N)
        assert terms == ["a"] and fell, option


def test_the_benchs_p4b_is_what_retrieve_runs_by_default():
    """The session measured L+P4b; retrieve.search runs it. One function, not a copy."""
    assert (b.tokens, b.match_expr, b.rarest, b.clamped) == (
        retrieve.words, retrieve.match_expr, retrieve.rarest, retrieve.clamped)
    assert retrieve.RAREST == b.P4_RAREST
    for claim in ("the photo shows Modi in Kerala flood 2018 a", "the a in", "Modi", "the the"):
        assert b.select_terms("L+P4b", claim, [], DF, N) == retrieve.prune(b.tokens(claim), DF, N)


def test_the_default_is_checked_against_the_sessions_l_plus_p4b():
    """Same terms and same top 20 -- ids, order and scores -- or the wiring is not the option."""
    qs = [_q("q1", []), _q("q2", []), _q("q3", [])]
    top = lambda tag, score=-1.0: [[f"{tag}{i}", score - i, "2020-01-01", "x.org", ""] for i in range(20)]
    row = lambda terms, hits: {"seconds": 1.0, "terms": terms, "fell_back": False, "hits": hits}
    session = {o: {"q1": row(["a", "b"], top("d")), "q2": row(["c"], top("e")),
                   "q3": row(["f"], top("g"))} for o in b.SESSION}
    wired = {"q1": row(["a", "b"], top("d")),                 # the same
             "q2": row(["c"], top("e", -1.5)),                # the same ids, other scores
             "q3": row(["f", "h"], top("g"))}                 # other terms
    for r in wired.values():
        r.update({"count_seconds": 0.2, "sql_seconds": 0.8, "process": "p"})
    texts = {h[0]: "" for r in [*session["L"].values(), *wired.values()] for h in r["hits"]}
    got = b.default_summary(qs, wired, session, texts)
    assert got["same_terms_as_session"] == 2 and got["same_hits_as_session"] == 2
    assert got["processes"] == ["p"] and got["count_seconds"]["median"] == 0.2
    assert got["overlap_at_20"]["mean"] == 1.0 and got["top1_same"] == 3


def test_overlap_is_shared_ids_over_k_and_none_without_a_full_baseline():
    assert b.overlap(list("abcde"), list("abxyz"), 5) == 0.4
    assert b.overlap(list("abc"), list("ab"), 5) is None


def test_ties_may_reorder_but_nothing_else_may():
    hits = [["d1", -9.0, "2020-01-01", "x.org", ""], ["d2", -5.0, "2020-01-02", "y.org", ""],
            ["d3", -5.0, "2020-01-03", "z.org", ""]]
    swapped = {"hit_domains": ["x.org", "z.org", "y.org"],
               "hit_dates": ["2020-01-01", "2020-01-03", "2020-01-02"]}
    moved = {"hit_domains": ["y.org", "x.org", "z.org"],
             "hit_dates": ["2020-01-02", "2020-01-01", "2020-01-03"]}
    assert b.tie_reordered(hits, swapped)
    assert not b.tie_reordered(hits, moved)


def test_the_profile_picks_the_10th_50th_and_90th_percentile_queries():
    qs = [{"ledger": {"search_seconds": s}} for s in (5, 1, 4, 2, 3, 9, 8, 7, 6, 10, 11)]
    picks = b.profile_picks(qs)
    assert [qs[i]["ledger"]["search_seconds"] for i in picks] == [2, 6, 10]
