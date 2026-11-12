"""The fact-checker blocklist builder's rules: hosts, Wayback, sections, scope. Offline."""

from __future__ import annotations

import pytest

from scripts import factcheckers as fc

W = "https://web.archive.org/web/20210111003633/"


@pytest.mark.parametrize("url, expected", [
    ("https://www.Snopes.com/fact-check/x/?q=1#f", ("snopes.com", "/fact-check/x/")),
    ("http://media.snopes.com:8080/a", ("media.snopes.com", "/a")),
    ("snopes.com/no-scheme", ("snopes.com", "/no-scheme")),
    ("https://x.example", ("x.example", "/")),
    (W + "https://www.politifact.com/factchecks/2020/x/", ("politifact.com", "/factchecks/2020/x/")),
    (W + "www.africacheck.org/reports/x", ("africacheck.org", "/reports/x")),
    (W + W + "https://factly.in/x", ("factly.in", "/x")),
])
def test_a_citation_is_judged_by_the_url_it_archived(url, expected):
    assert fc.host_path(url) == expected


def test_a_store_page_unwraps_one_level_as_the_sql_does():
    # a capture of a capture is judged by the inner capture, in SQL and here alike
    assert fc.page("web.archive.org", W + W + "https://factly.in/x") == (
        "web.archive.org", "/web/20210111003633/https://factly.in/x")
    assert fc.page("indiatoday.in", "https://www.indiatoday.in/Fact-Check/x") == (
        "indiatoday.in", "/fact-check/x")
    assert fc.page("web.archive.org", W + "https://www.snopes.com/x") == ("snopes.com", "/x")


def test_suffixes_reach_every_parent_but_never_the_bare_tld():
    assert list(fc.suffixes("a.b.snopes.com")) == ["a.b.snopes.com", "b.snopes.com", "snopes.com"]


def test_a_section_is_a_path_prefix_not_a_substring():
    assert fc.under("/fact-check/story/x", "/fact-check")
    assert fc.under("/fact-check", "/fact-check")
    assert not fc.under("/fact-checkers/x", "/fact-check")
    assert not fc.under("/india/fact-check/x", "/fact-check")


def test_few_store_pages_mean_the_whole_host_is_blocked():
    got = fc.decide("snopes.com", ["/fact-check/a"] * 10, ["/ap/x"] * 30)
    assert got["scope"] == "domain" and got["store_blocked"] == 30


def test_a_publisher_with_a_desk_loses_only_the_desk():
    pages = ["/india/story/x"] * 700 + ["/fact-check/story/y"] * 5
    got = fc.decide("indiatoday.in", ["/fact-check/story/z"] * 333, pages)
    assert got["scope"] == "path" and got["paths"] == ["/fact-check"] and got["store_blocked"] == 5


def test_a_publisher_with_no_desk_section_loses_only_its_cited_fact_checks():
    # BBC's fact-checks share /news, which is all of BBC: no desk to narrow to
    pages = ["/news/world-1"] * 400
    got = fc.decide("bbc.co.uk", ["/news/world-53085347"] * 8, pages, ["d2", "d1", "d1"])
    assert got["scope"] == "docs" and got["doc_ids"] == ["d1", "d2"]
    assert got["store_blocked"] == 2 and "review" in got


def test_a_host_named_for_checking_is_blocked_whole_whatever_its_sections():
    pages = ["/fact-checks/x"] * 2 + ["/reports/y"] * 61
    got = fc.decide("africacheck.org", ["/fact-checks/reports/z"] * 607, pages)
    assert got["scope"] == "domain" and got["store_blocked"] == 63


def test_a_desk_three_segments_deep_counts():
    pages = ["/story/news/politics/x"] * 1200 + ["/story/news/factcheck/2020/y"] * 5
    got = fc.decide("usatoday.com", ["/story/news/factcheck/2020/06/02/z/"] * 102, pages)
    assert got["scope"] == "path" and got["paths"] == ["/story/news/factcheck"]
    assert got["store_blocked"] == 5


def test_a_host_whose_store_pages_are_all_fact_checks_is_blocked_whole():
    pages = ["/hoax-alert/x"] * 200 + ["/about"] * 10
    got = fc.decide("leadstories.com", ["/hoax-alert/y"] * 300, pages)
    assert got["scope"] == "domain"


def test_a_section_needs_ninety_percent_of_the_fact_check_urls():
    pages = ["/fact-check/x"] * 5 + ["/news/x"] * 600
    got = fc.decide("x.example", ["/fact-check/1"] * 8 + ["/news/1"] * 2, pages)
    assert got["scope"] == "docs" and "review" in got


def test_a_persons_decision_replaces_the_review_flag_and_nothing_else():
    bbc = {"host": "bbc.co.uk", **fc.decide("bbc.co.uk", ["/news/1"] * 8, ["/news/x"] * 400, ["d1"])}
    got = fc.decided(bbc)
    assert "review" not in got and got["decided"].startswith("the person, 2026-10-28")
    assert {k: v for k, v in got.items() if k != "decided"} == {
        k: v for k, v in bbc.items() if k != "review"}
    other = {"host": "x.example", **fc.decide("x.example", ["/news/1"] * 8, ["/news/x"] * 400)}
    assert fc.decided(other) == other and "review" in other     # undecided: still flagged
    whole = {"host": "bbc.co.uk", "scope": "domain", "store_docs": 3, "store_blocked": 3}
    assert fc.decided(whole) == whole                           # the decision is about docs scope


def test_a_host_added_by_hand_is_blocked_whole_whatever_its_pages():
    # a section and 80 news pages would make a cited host path-scoped; a hand-added one is whole
    got = fc.by_hand("hoax.sk", ["/news/x"] * 80 + ["/hoax/y"] * 4)
    assert got["scope"] == "domain" and got["store_blocked"] == got["store_docs"] == 84
    assert got["added_by_hand"].startswith("the person, 2026-10-29")


def test_a_host_nothing_cites_does_not_come_only_from_test():
    entries = [{"host": "a.example", "cited_by": {"m4fc": ["test"]}},
               {"host": "b.example", "cited_by": {"m4fc": ["test"], "mocheg": ["train", "test"]}},
               {"host": "hoax.sk", "cited_by": {}}]
    assert fc.only_from_test(entries) == ["a.example"]


class _Rules:
    sources = (("snopes.com", None), ("indiatoday.in", "/fact-check"))
    doc_ids = ("d-listed",)
    words = ("webqoof", "hoax-alert")


@pytest.mark.parametrize("doc_id, domain, url, expected", [
    ("a", "media.snopes.com", "https://media.snopes.com/x", "host"),
    ("b", "indiatoday.in", "https://www.indiatoday.in/fact-check/x", "section"),
    ("d-listed", "reuters.com", "https://www.reuters.com/article/x", "doc"),
    ("c", "anyone.example", "https://anyone.example/2020/viral-HOAX-ALERT-x", "word"),
    ("d", "web.archive.org", W + "https://anyone.example/webqoof/x", "word"),
    ("e", "indiatoday.in", "https://www.indiatoday.in/fact-check/webqoof", "section"),
    ("f", "anyone.example", "https://anyone.example/news/x?tag=webqoof", None),
    ("g", "webqoof.example", "https://webqoof.example/news/x", None),
    ("h", "indiatoday.in", "https://www.indiatoday.in/india/x", None),
])
def test_the_twin_names_the_first_rule_that_blocks_a_page(doc_id, domain, url, expected):
    assert fc.blocker(_Rules())(doc_id, *fc.page(domain, url)) == expected
