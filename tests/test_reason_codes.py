"""reason_code / modality_evidence / evaluation_only, and the loader that enforces them.

Offline: the mapping table is the contract; loader tests use a temp file.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import reason_codes as rc  # noqa: E402
from scripts import processed_loader as pl  # noqa: E402


@pytest.mark.parametrize("dataset,label,code", [
    ("verite", "out-of-context", "media_mismatch"),
    ("verite", "miscaptioned", "wrong_image"),
    ("verite", "true", "none"),
    ("mocheg", "refuted", "content_refuted"),
    ("mocheg", "NEI", "content_unverified"),
    ("mocheg", "supported", "none"),
    ("averitec", "Refuted", "content_refuted"),
    ("averitec", "Not Enough Evidence", "content_unverified"),
    ("averitec", "Supported", "none"),
    ("factify2", "Refute", "content_refuted"),
    ("factify2", "Insufficient_Text", "content_unverified"),
    ("factify2", "Insufficient_Multimodal", "content_unverified"),
    ("factify2", "Support_Text", "none"),
    ("factify2", "Support_Multimodal", "none"),
    ("m4fc", "false", "content_refuted"),
    ("m4fc", "not enough information", "content_unverified"),
    ("m4fc", "true", "none"),
])
def test_the_specified_mappings_hold_exactly(dataset, label, code):
    assert rc.reason_code(dataset, label) == code
    assert rc.MAPPING[dataset][label][1] == rc.SPECIFIED


def test_averimatec_inherits_averitecs_rule_and_says_so():
    for label in ("Refuted", "Not Enough Evidence", "Supported"):
        assert rc.reason_code("averimatec", label) == rc.reason_code("averitec", label)
        assert rc.MAPPING["averimatec"][label][1] == rc.INHERITED
    out = rc.report({"averimatec": Counter({"Refuted": 3})})
    assert {r["dataset"] for r in out["inherited_rules"]} == {"averimatec"}


def test_none_and_unmapped_are_different_values():
    assert rc.reason_code("verite", "true") == "none"
    assert rc.reason_code("liar", "true") == rc.UNMAPPED
    assert rc.is_supervised("none") and not rc.is_supervised(rc.UNMAPPED)


@pytest.mark.parametrize("dataset", ["welfake", "liar", "fakenewsnet", "fakeddit", "isot", "dgm4"])
def test_reasonless_schemes_are_unmapped_whatever_the_label(dataset):
    for label in ("fake", "true", "0", "4", "text_swap", None):
        assert rc.reason_code(dataset, label) == rc.UNMAPPED


@pytest.mark.parametrize("dataset,label", [
    ("averitec", "Conflicting Evidence/Cherrypicking"),
    ("averimatec", "Conflicting Evidence/Cherrypicking"),
    ("averitec", None),
])
def test_labels_without_a_clean_code_stay_null_and_are_listed(dataset, label):
    assert rc.reason_code(dataset, label) is None
    out = rc.report({dataset: Counter({label: 1})})
    assert [(r["dataset"], r["label"]) for r in out["not_cleanly_mapped"]] == [(dataset, label)]


def test_fakeddits_six_classes_are_decoded_and_none_is_mapped():
    names = {k: v["name"] for k, v in rc.FAKEDDIT_6WAY.items()}
    assert names == {"0": "True", "1": "Satire/Parody", "2": "False Connection",
                     "3": "Imposter Content", "4": "Manipulated Content",
                     "5": "Misleading Content"}
    assert "fakeddit" not in rc.MAPPING, "shown for decision, not mapped"


def test_report_keeps_none_and_unmapped_apart_and_counts_supervision():
    out = rc.report({"verite": Counter({"true": 2, "out-of-context": 3}),
                     "averimatec": Counter({"Refuted": 4}),
                     "liar": Counter({"false": 5})})
    v = out["datasets"]["verite"]
    assert (v["none"], v["unmapped"], v["supervised_records"]) == (2, 0, 5)
    assert out["datasets"]["liar"]["unmapped"] == 5
    assert out["total_supervised_records"] == 9
    assert out["total_supervised_trainable"] == 0, "verite and averimatec are evaluation-only"


def test_modality_evidence_comes_only_from_factify2s_own_split():
    assert rc.modality_evidence("factify2", "Support_Text") == "text"
    assert rc.modality_evidence("factify2", "Insufficient_Multimodal") == "multimodal"
    assert rc.modality_evidence("factify2", "Refute") is None
    assert rc.modality_evidence("verite", "true") is None


# --------------------------------------------------------------------------
# the loader enforces it
# --------------------------------------------------------------------------


def _write(root: Path, dataset: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / f"{dataset}.jsonl.gz", "wt", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps({"dataset": dataset, **r}) + "\n")


def test_unmapped_and_null_rows_are_never_yielded(tmp_path):
    _write(tmp_path, "averitec", [
        {"record_id": "a", "split": "train", "reason_code": "content_refuted", "evaluation_only": False},
        {"record_id": "b", "split": "train", "reason_code": None, "evaluation_only": False},
        {"record_id": "c", "split": "train", "reason_code": "unmapped", "evaluation_only": False},
    ])
    got = [r["record_id"] for r in pl.reason_code_rows("averitec", split="train",
                                                       purpose="train", root=tmp_path)]
    assert got == ["a"]


def test_a_reasonless_dataset_is_refused_outright(tmp_path):
    with pytest.raises(pl.SupervisionError, match="no reason information"):
        list(pl.reason_code_rows("welfake", split="train", purpose="evaluate", root=tmp_path))


def test_evaluation_only_corpora_can_evaluate_but_never_train(tmp_path):
    _write(tmp_path, "averimatec", [
        {"record_id": "x", "split": "test", "reason_code": "content_refuted", "evaluation_only": True}])
    with pytest.raises(pl.SupervisionError, match="evaluation-only"):
        list(pl.reason_code_rows("averimatec", split="test", purpose="train", root=tmp_path))
    assert len(list(pl.reason_code_rows("averimatec", split="test", purpose="evaluate",
                                        root=tmp_path))) == 1


def test_the_row_check_raises_on_every_violation():
    base = {"dataset": "d", "record_id": "r", "split": "train", "evaluation_only": False}
    with pytest.raises(pl.SupervisionError, match="unmapped"):
        pl.check_reason_row({**base, "reason_code": "unmapped"}, "evaluate")
    with pytest.raises(pl.SupervisionError):
        pl.check_reason_row({**base, "reason_code": None}, "evaluate")
    with pytest.raises(pl.SupervisionError, match="evaluation-only"):
        pl.check_reason_row({**base, "reason_code": "none", "evaluation_only": True}, "train")
    with pytest.raises(pl.SupervisionError, match="split"):
        pl.check_reason_row({**base, "reason_code": "none", "split": "test"}, "train")
    assert pl.check_reason_row({**base, "reason_code": "none"}, "train")


def test_fakehistoryporn_is_excluded_whatever_fakeddit_maps_to(monkeypatch):
    assert rc.reason_code("fakeddit", "2", {"subreddit": "fakehistoryporn"}) == rc.UNMAPPED
    # even if fakeddit were later taken out of the unmapped list
    monkeypatch.setattr(rc, "UNMAPPED_DATASETS", rc.UNMAPPED_DATASETS - {"fakeddit"})
    monkeypatch.setattr(rc, "MAPPING", {**rc.MAPPING, "fakeddit": {"2": ("wrong_image", rc.SPECIFIED)}})
    assert rc.reason_code("fakeddit", "2", {"subreddit": "fakehistoryporn"}) == rc.UNMAPPED
    assert rc.reason_code("fakeddit", "2", {"subreddit": "pareidolia"}) == "wrong_image"


@pytest.mark.parametrize("label,meta,code", [
    ("text_swap", {"swap_publisher": "same"}, "media_mismatch"),
    ("text_swap", {"swap_publisher": "cross"}, "unmapped"),      # style mismatch could give it away
    ("text_swap", {"swap_publisher": "ambiguous"}, "unmapped"),
    ("text_swap", None, "unmapped"),
    ("face_swap&text_swap", {"swap_publisher": "same"}, "unmapped"),  # construct ambiguous: excluded
    ("face_attribute&text_swap", {"swap_publisher": "same"}, "unmapped"),
    ("face_swap", None, "unmapped"),                              # image manipulation, not mismatch
    ("orig", None, "unmapped"),                                   # none is reserved for five labels
    ("text_attribute", None, "unmapped"),
])
def test_dgm4_row_rule_licenses_only_within_publisher_pure_text_swaps(label, meta, code):
    assert rc.dgm4_reason_code(label, meta) == code


def test_dgm4_stays_unmapped_until_its_audit_passes():
    assert rc.DGM4_AUDIT_PASSED is False
    assert "dgm4" in rc.UNMAPPED_DATASETS
    assert rc.reason_code("dgm4", "text_swap", {"swap_publisher": "same"}) == "unmapped"
    with pytest.raises(pl.SupervisionError, match="no reason information"):
        list(pl.reason_code_rows("dgm4", split="train", purpose="train"))


def test_dgm4_flag_agrees_with_the_audit_report():
    from configs.paths import REPORTS
    path = REPORTS / "dgm4_audit.json"
    if not path.is_file():
        pytest.skip("no audit report on this machine; run scripts/dgm4_audit.py")
    import json
    decision = json.loads(path.read_text(encoding="utf-8"))["decision"]
    passed = decision["whole_class"]["passes"] or decision["same_publisher_subset"]["passes"]
    assert rc.DGM4_AUDIT_PASSED == passed, decision["applied"]


def test_weak_supervision_rows_never_evaluate(tmp_path):
    assert rc.weak_supervision("dgm4", "media_mismatch")
    assert not rc.weak_supervision("dgm4", "unmapped")
    assert not rc.weak_supervision("verite", "media_mismatch")
    _write(tmp_path, "factify2", [
        {"record_id": "s", "split": "train", "reason_code": "content_refuted",
         "evaluation_only": False, "weak_supervision": True},
        {"record_id": "o", "split": "train", "reason_code": "content_refuted",
         "evaluation_only": False, "weak_supervision": False}])
    assert [r["record_id"] for r in pl.reason_code_rows("factify2", split="train", purpose="evaluate",
                                                       root=tmp_path)] == ["o"]
    with pytest.raises(pl.SupervisionError, match="weak"):
        pl.check_reason_row({"dataset": "dgm4", "record_id": "s", "split": "test",
                             "reason_code": "media_mismatch", "weak_supervision": True}, "evaluate")


def test_distribution_is_row_level_and_keeps_none_and_unmapped_apart():
    recs = [{"dataset": "verite", "label": "true"},
            {"dataset": "verite", "label": "out-of-context"},
            {"dataset": "liar", "label": "false"}]
    out = rc.distribution(recs)
    assert out["datasets"]["verite"]["per_class"]["true"] == {"none": 1}
    assert out["datasets"]["liar"]["unmapped"] == 1
    assert out["supervised_records"] == {"evaluation_only": 2}
