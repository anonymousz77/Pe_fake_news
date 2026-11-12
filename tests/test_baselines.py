"""Track B's baselines: the hostname reproduction, the text-only classifier's
inputs, MUSE's features and intervals. Offline; data-backed twins skip without
the processed layer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.paths import PROCESSED, REPORTS  # noqa: E402
from scripts import baselines as bl  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts import textcheck as tc  # noqa: E402

HAVE_PROCESSED = (PROCESSED / "records" / "factify2.jsonl.gz").is_file() and bl.INTERIM_FACTIFY2.is_file()
needs_processed = pytest.mark.skipif(not HAVE_PROCESSED, reason="needs the processed and interim layers")


# --------------------------------------------------------------------------
# intervals that resample clusters
# --------------------------------------------------------------------------


def test_cluster_intervals_hold_the_point_value_and_count_clusters():
    y = ["a", "a", "b", "b", "a", "b"]
    p = ["a", "b", "b", "b", "a", "a"]
    out = bl.scored_clusters(y, p, ["r1", "r1", "r2", "r2", "r3", "r3"], ["a", "b"], seed=1)
    assert out["rows"] == 6 and out["clusters"] == 3
    assert out["accuracy"]["value"] == round(4 / 6, 4)
    lo, hi = out["accuracy"]["ci"]
    assert lo <= out["accuracy"]["value"] <= hi
    assert out["per_class"]["a"]["support"] == 3


def test_a_cluster_moves_whole():
    # one cluster always right, one always wrong: every resample's accuracy is 0, 0.5 or 1
    y, p = ["a", "a", "b", "b"], ["a", "a", "a", "a"]
    out = bl.scored_clusters(y, p, ["r1", "r1", "r2", "r2"], ["a", "b"], seed=3)
    assert set(out["accuracy"]["ci"]) <= {0.0, 0.5, 1.0}


def test_the_applicable_majority_is_the_scored_sets_own():
    assert bl.share_of_most_common(["Refute"] * 6 + ["Support_Text"] * 4) == {
        "label": "Refute", "share": 0.6, "n": 10}


# --------------------------------------------------------------------------
# the hostname baseline
# --------------------------------------------------------------------------


def test_the_random_arm_is_sorted_ids_shuffled_and_cut_70_15_15():
    ids = [f"r{i:03d}" for i in range(100)]
    pl = bl.random_placement(list(reversed(ids)))
    assert pl == bl.random_placement(ids)
    counts = {s: sum(1 for v in pl.values() if v == s) for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_host_pairs_are_one_per_distinct_host():
    pairs = bl.host_pairs({"r1": "test", "r2": "train"}, {"r1": {"a.com", "b.com"}, "r2": {"a.com"}},
                          {"r1": "Refute", "r2": "Support_Text"}, "test")
    assert pairs == [("r1", "a.com", "Refute"), ("r1", "b.com", "Refute")]


@needs_processed
def test_the_hostname_figure_reproduces_the_final_json():
    section = bl.hostname_section()
    final = json.loads((REPORTS / "factify2_domain_only_final.json").read_text(encoding="utf-8"))
    for arm in ("source_disjoint", "random_70_15_15"):
        assert section["reproduces"][arm] == final[arm]
    assert section["by_pair"]["accuracy"]["value"] == final["source_disjoint"]["domain_only_test_accuracy"]
    assert section["by_pair"]["applicable_majority"]["share"] == final["source_disjoint"]["majority_baseline"]
    assert section["by_record"]["applicable_majority"]["share"] == round(5620 / 10325, 4)
    assert section["train_refute_records"] == 15


# --------------------------------------------------------------------------
# the text-only classifier
# --------------------------------------------------------------------------


def test_three_way_labels_agree_with_the_text_side_check():
    for ds in ("averitec", "averimatec", "m4fc"):
        assert bl.THREE_WAY[ds] == tc.GOLD[ds]
    assert set(bl.THREE_WAY["mocheg"].values()) == set(tc.VERDICT_LABELS)


def test_verite_and_averimatec_never_fit_a_text_model():
    with pytest.raises(SystemExit):
        bl.claim_rows("verite", "test")
    assert all(cfg[0] != "averimatec" for cfg in bl.TEXT_ONLY.values())
    assert all(cfg[0] != "verite" and cfg[2] != "verite" for cfg in bl.TEXT_ONLY.values())


def test_the_text_classifier_reads_the_claim_never_the_evidence(monkeypatch):
    rows_ = {("averitec", "train"): [{"text": "claim one", "evidence_text": "JUSTIFICATION", "label": "Refuted",
                                      "usable": True},
                                     {"text": "claim two", "evidence_text": "JUSTIFICATION", "label": "Supported",
                                      "usable": True}],
             ("averitec", "val"): [{"text": "claim three", "evidence_text": "JUSTIFICATION", "label": "Refuted",
                                    "usable": True}]}
    monkeypatch.setattr(bl, "claim_rows", lambda ds, split: rows_[(ds, split)])
    seen = {}

    def fake_audit(tr_x, tr_y, ev_x, ev_y):
        seen.update(tr_x=tr_x, ev_x=ev_x, tr_y=tr_y)
        return {"rows": len(ev_x), "model": {}, "majority": {}}

    import scripts.m4fc_audit as m4
    monkeypatch.setattr(m4, "text_audit", fake_audit)
    out = bl.text_only_one("averitec")
    assert seen["tr_x"] == ["claim one", "claim two"] and seen["ev_x"] == ["claim three"]
    assert "JUSTIFICATION" not in json.dumps(seen)
    assert seen["tr_y"] == ["refuted", "supported"]
    assert out["input"].startswith("the claim text only")


# --------------------------------------------------------------------------
# MUSE
# --------------------------------------------------------------------------


def test_muse_features_take_the_top_evidence_and_zero_without(tmp_path, monkeypatch):
    img = np.array([1.0, 0.0], dtype=np.float32)
    claim = np.array([0.6, 0.8], dtype=np.float32)
    t1, t2 = np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)
    keys = ["img:s1", f"txt:{bl.text_sha('c')}", f"txt:{bl.text_sha('t1')}", f"txt:{bl.text_sha('t2')}"]
    path = tmp_path / "e.npz"
    np.savez(path, keys=np.array(keys), vecs=np.stack([img, claim, t1, t2]))
    monkeypatch.setattr(bl, "MUSE_EMB", path)
    recs = [{"key": "k1", "sha256": "s1", "claim": "c"}, {"key": "k2", "sha256": "s1", "claim": "c"},
            {"key": "k3", "sha256": "s1", "claim": "c"}]
    bl.muse_features(recs, {"k1": ["t1", "t2"], "k2": []})
    assert recs[0]["img_txt"] == pytest.approx(0.6)
    assert recs[0]["img_X_txt"] == pytest.approx(1.0) and recs[0]["txt_X_txt"] == pytest.approx(0.8)
    assert recs[1]["img_X_txt"] == 0.0 and recs[1]["txt_X_txt"] == 0.0
    assert "img_X_txt" not in recs[2]


def test_a_single_auroc_interval_resamples_clusters():
    recs = [{"key": f"k{i}", "class": "true" if i % 2 else "out-of-context", "sha256": f"s{i // 2}",
             "claim_key": f"c{i}", "score": float(i % 2 == 0) + i / 100} for i in range(20)]
    out = bl.auroc_ci(recs, "out-of-context", "true", "score", n=200)
    assert out["auroc"] == 1.0 and out["clusters"] == 10 and out["ci95"][0] <= 1.0


def test_muse_trains_on_m4fc_train_only():
    with pytest.raises(cs.GuardError):
        cs.fit_threshold([{"dataset": "verite", "split": "test", "class": "true", "score": 0.1}])
    with pytest.raises(cs.GuardError):
        cs.fit_threshold([{"dataset": "m4fc", "split": "val", "class": "true", "score": 0.1}])


def test_mlp_settings_are_muses():
    s = bl.MLP_SETTINGS
    assert (s["hidden"], s["epochs"], s["early_stop"], s["batch"], s["lr_options"], s["weight_decay"]) == (
        768, 50, 10, 512, [1e-4, 5e-5], 0.01)


# --------------------------------------------------------------------------
# M4FC's official test: the baselines read it once, after the system
# --------------------------------------------------------------------------


def test_the_test_is_read_once_and_only_after_the_system(tmp_path, monkeypatch):
    monkeypatch.setattr(bl, "TEST_READ_LOG", tmp_path / "read.json")
    monkeypatch.setattr(bl, "FINAL_SCORED", tmp_path / "scored.json")
    with pytest.raises(SystemExit, match="its read comes first"):
        bl.main(["test-baselines"])
    (tmp_path / "read.json").write_text(json.dumps({"read_at": "2026-11-12T00:00:00+00:00"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="it is read once"):
        bl.main(["test-baselines"])


def test_the_test_pairs_are_the_systems():
    assert bl.TEST_PAIRS == (("false", "true"), (cs.OOC, "true"))
    assert set(bl.VP_CLASSES) == {"true", "false (out-of-context)", "false (manipulated/fake)",
                                  "false (ai-generated)"}


# --------------------------------------------------------------------------
# the source audit: lookups and within-group AUROC
# --------------------------------------------------------------------------


def test_a_lookup_scores_its_groups_false_rate_and_falls_back_for_an_unseen_one():
    train = [{"org": "a", "class": "false"}, {"org": "a", "class": "true"}, {"org": "b", "class": "false"},
             {"org": "b", "class": "false"}]
    got = bl.lookup_scores(train, [{"org": "a"}, {"org": "b"}, {"org": "c"}], "org")
    assert got == [0.5, 1.0, 0.75]


def test_within_group_auroc_ignores_pairs_across_groups():
    # across groups the score separates perfectly; within each group it is backwards
    recs = [{"org": "a", "class": "false", "s": 0.9}, {"org": "a", "class": "true", "s": 0.95},
            {"org": "b", "class": "false", "s": 0.1}, {"org": "b", "class": "true", "s": 0.2}]
    assert bl.within_group_auroc(recs, "s", "org", n=10)["auroc"] == 0.0
    assert cs.auroc([0.9, 0.1], [0.95, 0.2]) == 0.25
