"""scripts/m4fc_audit.py: the rules its numbers rest on.

Offline and data-free: the presence rule, date shapes, slug words, the sheet's
allocation, the scorer and its majority baseline, the text classifier on toy
rows, and the entity match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import m4fc_audit as ma  # noqa: E402


def test_absent_markers_lists_and_multi_values():
    for absent in ("not enough information", "Not enough information", "unspecified",
                   "not specified", "", None, [], ["not enough information"]):
        assert not ma.present(absent), absent
    assert ma.present("1900; not enough information")
    assert ma.present(["dilma rousseff"])
    assert ma.present("Copacabana, Rio de Janeiro, Brazil")


def test_both_needs_the_claimed_and_the_actual_value():
    row = {"claimed_date": "2021-09-07", "date": "2016-03-13",
           "claimed_people": "Pro-Bolsonaro protesters", "people": "not enough information"}
    assert ma.both(row, "date")
    assert not ma.both(row, "people")


def test_a_list_is_shown_joined():
    assert ma.shown(["ahmed moussa", "mohamed el - baz"]) == "ahmed moussa; mohamed el - baz"
    assert ma.shown("2016-03-13") == "2016-03-13"


@pytest.mark.parametrize("value,shape", [
    ("2016", "normalised"), ("2016-03", "normalised"), ("2016-03-13; 2017", "normalised"),
    ("1900; not enough information", "normalised"), ("1951-1967", "range"),
    ("1943; current", "other"), ("not enough information", None)])
def test_date_shapes(value, shape):
    assert ma.date_shape(value) == shape


def test_slug_words_are_whole_tokens():
    assert ma.slug_states_finding("images/old-unrelated-images-falsely-shared-as-x.png")
    assert ma.slug_states_finding("images/this-picture-is-not-related-to-rains.png")
    assert not ma.slug_states_finding("images/golden-temple-in-bold-colours.png")


def test_allocation_gives_every_org_one_and_never_passes_the_cap():
    sizes = {"a": 510, "b": 298, "c": 4, "d": 1, **{f"o{i}": 30 for i in range(15)}}
    alloc = ma.allocate(sizes, total=60, cap=15)
    assert sum(alloc.values()) == 60
    assert set(alloc) == set(sizes) and min(alloc.values()) >= 1
    assert all(alloc[o] <= min(15, sizes[o]) for o in alloc)
    assert ma.allocate(sizes, total=60, cap=15) == alloc, "allocation must be deterministic"


def test_allocation_redistributes_what_the_cap_holds_back():
    assert ma.allocate({"big": 1000, "small": 20}, total=30, cap=15) == {"big": 15, "small": 15}


def test_allocation_refuses_what_it_cannot_fill():
    with pytest.raises(ValueError):
        ma.allocate({"a": 2, "b": 2}, total=10, cap=15)


def test_scores_the_model_and_the_majority_on_the_same_draws():
    truth = ["a"] * 30 + ["b"] * 10
    pred = ["a"] * 25 + ["b"] * 5 + ["b"] * 8 + ["a"] * 2
    one = ma.scored(truth, pred, "a", ["a", "b"])
    assert one == ma.scored(truth, pred, "a", ["a", "b"]), "the bootstrap must be seeded"
    assert one["model"]["accuracy"]["value"] == pytest.approx(33 / 40)
    assert one["majority"]["accuracy"]["value"] == pytest.approx(30 / 40)
    assert one["majority"]["per_class"]["b"]["recall"]["value"] == 0.0
    assert one["majority"]["per_class"]["b"]["precision"]["value"] == 0.0
    lo, hi = one["model"]["accuracy"]["ci"]
    assert lo <= 33 / 40 <= hi


def test_a_class_absent_from_the_scored_rows_has_no_recall():
    out = ma.scored(["a", "a"], ["a", "b"], "a", ["a", "b"])
    assert out["model"]["per_class"]["b"]["recall"]["value"] is None
    assert out["model"]["macro_f1"]["value"] == pytest.approx(out["model"]["per_class"]["a"]["f1"]["value"])


def test_the_text_classifier_finds_a_separable_signal():
    train = [("red apple fruit", "x")] * 20 + [("blue sky cloud", "y")] * 20
    val = [("apple fruit red", "x")] * 5 + [("cloud blue sky", "y")] * 5
    out = ma.text_audit([t for t, _ in train], [y for _, y in train],
                        [t for t, _ in val], [y for _, y in val])
    assert out["fit"]["converged"]
    assert out["model"]["accuracy"]["value"] == 1.0
    assert out["majority"]["accuracy"]["value"] == 0.5


def test_entities_match_whole_words_only():
    hay = ["pm narendra modi's visit", "an amedeo modigliani painting"]
    assert ma.found_in("Modi", hay[:1])
    assert not ma.found_in("Modi", hay[1:])
    assert ma.found_in("Narendra Modi", hay)


def test_entity_terms_are_distinct_and_keep_the_type():
    text = "Modi met modi in Delhi"
    spans = [{"entity_group": "PER", "start": 0, "end": 4}, {"entity_group": "PER", "start": 9, "end": 13},
             {"entity_group": "LOC", "start": 17, "end": 22}, {"entity_group": "MISC", "start": 5, "end": 6}]
    assert ma.entity_terms(spans, text) == [{"type": "PER", "text": "Modi"}, {"type": "LOC", "text": "Delhi"}]


def test_fact_check_domains_match_subdomains_not_lookalikes():
    assert ma.on_domain("www.factly.in", ["factly.in"])
    assert ma.on_domain("telugu.factly.in", ["factly.in"])
    assert not ma.on_domain("notfactly.in", ["factly.in"])
