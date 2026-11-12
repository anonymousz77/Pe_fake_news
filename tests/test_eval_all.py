"""eval_all: numbers are read, never recomputed; the reporting rules hold; a
missing source is listed, never skipped. Offline, on synthetic reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import eval_all as ea  # noqa: E402


def test_paths_resolve_keys_and_list_selections():
    doc = {"a": {"b": [{"positive": "x", "v": 1}, {"positive": "y", "v": 2}]}}
    assert ea.resolve(doc, ("a", "b", ea.sel("positive", "y"), "v")) == 2
    with pytest.raises(ea.Missing, match="no key"):
        ea.resolve(doc, ("a", "c"))
    with pytest.raises(ea.Missing, match="0 element"):
        ea.resolve(doc, ("a", "b", ea.sel("positive", "z")))


def _entry(value, tags=()):
    return {"block": "baselines", "source": "x.json", "tags": list(tags), "value": value}


GOOD_F2 = {"model": {"per_class": {"Refute": {}}, "accuracy": {"value": 0.2}},
           "applicable_majority": {"share": 0.54}}
VERITE_OK = {"evaluated": 877}


def _with_baselines(**more):
    entries = {"baselines.hostname_factify2_by_pair": _entry({"accuracy": {"value": 0.1}}),
               "baselines.hostname_factify2_by_record": _entry({"accuracy": {"value": 0.1}})}
    entries.update(more)
    return entries


def test_a_factify2_number_needs_its_baselines_and_per_class():
    assert ea.rule_problems(_with_baselines(**{"baselines.f2": _entry(GOOD_F2, ["factify2"])}), VERITE_OK) == []
    lonely = {"baselines.f2": _entry(GOOD_F2, ["factify2"])}
    assert any("hostname baseline" in p for p in ea.rule_problems(lonely, VERITE_OK))
    averaged = {"model": {"accuracy": {"value": 0.2}}, "applicable_majority": {"share": 0.5}}
    probs = ea.rule_problems(_with_baselines(**{"baselines.f2": _entry(averaged, ["factify2"])}), VERITE_OK)
    assert any("not per class" in p for p in probs)
    no_majority = {"model": {"per_class": {}}}
    probs = ea.rule_problems(_with_baselines(**{"baselines.f2": _entry(no_majority, ["factify2"])}), VERITE_OK)
    assert any("applicable majority" in p for p in probs)


def test_verite_is_evaluated_on_877():
    assert any("877" in p for p in ea.rule_problems({}, {"evaluated": 914}))


def test_isot_is_never_a_headline_and_nli_a_never_a_system_number():
    probs = ea.rule_problems({"x.isot_result": _entry({"v": 1}, ["headline"])}, VERITE_OK)
    assert any("ISOT" in p for p in probs)
    probs = ea.rule_problems({"text_side.system": _entry({"judge": "nli_a"}, ["headline"])}, VERITE_OK)
    assert any("NLI model A" in p for p in probs)
    assert ea.rule_problems({"text_side.contrast": _entry({"judge": "nli_a"}, ["contrast"])}, VERITE_OK) == []


def test_a_dgm4_number_on_verite_needs_its_transfer_gap():
    bad = _entry({"trained_on": "dgm4 train", "scored_on": "verite"})
    assert any("transfer gap" in p for p in ea.rule_problems({"x.y": bad}, VERITE_OK))
    good = _entry({"trained_on": "dgm4 train", "scored_on": "verite", "transfer_gap": 0.3})
    assert ea.rule_problems({"x.y": good}, VERITE_OK) == []


def test_a_missing_report_is_listed_not_skipped(tmp_path, monkeypatch):
    (tmp_path / "redtest.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    monkeypatch.setattr(ea, "REQUIRED", [
        ("evidence_store", "redtest_passed", "redtest.json", ("passed",), ()),
        ("evidence_store", "absent_key", "redtest.json", ("nope",), ()),
        ("text_side", "system", "textcheck.json", ("system",), ("headline",))])
    monkeypatch.setattr(ea, "image_text_rates", lambda: {"config": "c", "ledgers": {}})
    monkeypatch.setattr(ea, "verite_counts", lambda: {"evaluated": 877})
    report = ea.collect(tmp_path)
    assert report["blocks"]["evidence_store"]["redtest_passed"]["value"] is True
    reasons = {m["number"]: m["reason"] for m in report["missing"]}
    assert set(reasons) == {"evidence_store.absent_key", "text_side.system"}
    assert reasons["text_side.system"] == "the report is not there"
    assert len(report["sources"]["redtest.json"]["sha256"]) == 64


def test_every_required_entry_names_a_report_and_a_path():
    keys = [f"{b}.{n}" for b, n, *_ in ea.REQUIRED]
    assert len(keys) == len(set(keys))
    for block, name, fname, path, tags in ea.REQUIRED:
        assert fname.endswith(".json") and path, name
    tagged = {f"{b}.{n}" for b, n, _, _, t in ea.REQUIRED if "factify2" in t}
    assert tagged == {"baselines.text_only_factify2"}
