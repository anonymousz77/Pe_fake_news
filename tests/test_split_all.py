"""Split atomicity: duplicate groups and cross-corpus duplicates never straddle.

Offline and synthetic: records are built by hand, no data on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import split_all  # noqa: E402
from scripts.split_all import SplitInput  # noqa: E402


def _official(ds, n, split, label="x", hashes=()):
    return {(ds, f"{ds}{i}"): SplitInput(label, split, list(hashes)) for i in range(n)}


def test_a_five_copy_group_across_official_splits_moves_to_the_held_out_side():
    records = {}
    records.update(_official("liar", 3, "train", hashes=["H"]))
    records.update({("liar", "t0"): SplitInput("x", "test", ["H"]),
                    ("liar", "t1"): SplitInput("x", "test", ["H"])})
    placement, report = split_all.assign_all(records, {"H": 5})
    assert {placement[n] for n in records} == {"test"}
    assert report["moved_by_reconciliation"] == {"liar:train->test": 3}


def test_a_group_below_five_copies_in_one_corpus_is_left_alone():
    records = {("liar", "a"): SplitInput("x", "train", ["H"]),
               ("liar", "b"): SplitInput("x", "test", ["H"])}
    placement, _ = split_all.assign_all(records, {"H": 2})
    assert placement[("liar", "a")] == "train" and placement[("liar", "b")] == "test"


def test_a_cross_corpus_duplicate_is_kept_together_whatever_its_count():
    records = {("m4fc", "a"): SplitInput("false", "train", ["H"]),
               ("mocheg", "b"): SplitInput("refuted", "val", ["H"])}
    placement, _ = split_all.assign_all(records, {"H": 2})
    assert placement[("m4fc", "a")] == placement[("mocheg", "b")] == "val"


def test_averimatec_is_evaluation_only_and_drags_its_duplicates_to_test():
    records = {("averimatec", "x"): SplitInput("Refuted", "train", ["H"]),
               ("m4fc", "y"): SplitInput("false", "train", ["H"])}
    placement, _ = split_all.assign_all(records, {"H": 2})
    assert placement[("averimatec", "x")] == placement[("m4fc", "y")] == "test"


def test_an_image_shared_with_verite_takes_its_factify2_component_to_test():
    records = {("verite", "0"): SplitInput("true", "test", ["H"])}
    for i in range(30):
        records[("factify2", f"r{i}")] = SplitInput(
            ["Refute", "Support_Text"][i % 2], None, ["H"] if i == 0 else [f"x{i}"],
            {"twitter.com" if i % 3 else "factly.in"})
    placement, report = split_all.assign_all(records, {"H": 2})
    assert placement[("factify2", "r0")] == "test"
    assert report["factify2"]["pinned_records"] == 1
    # r0's whole component (everything sharing factly.in with it, if constrained) follows
    split_all.assert_atomic(placement, split_all.constraint_groups(records, {"H": 2}))


def test_the_guard_raises_when_a_group_straddles():
    with pytest.raises(split_all.AtomicityError, match="straddle"):
        split_all.assert_atomic({("a", "1"): "train", ("b", "2"): "test"},
                                [{("a", "1"), ("b", "2")}])


def test_unsplit_corpora_get_a_seeded_split_that_keeps_groups_whole():
    records = {("welfake", f"w{i}"): SplitInput(["real", "fake"][i % 2], None,
                                                ["H"] if i < 6 else [])
               for i in range(60)}
    one, _ = split_all.assign_all(records, {"H": 6})
    two, _ = split_all.assign_all(records, {"H": 6})
    assert one == two, "the seed must make this reproducible"
    assert len({one[("welfake", f"w{i}")] for i in range(6)}) == 1
    assert set(one.values()) <= {"train", "val", "test"}


def test_verite_is_test_by_construction():
    records = {("verite", str(i)): SplitInput("true", "test", []) for i in range(5)}
    placement, _ = split_all.assign_all(records, {})
    assert set(placement.values()) == {"test"}


def test_dgm4_rows_sharing_a_caption_share_a_split():
    records = {("dgm4", "a"): SplitInput("orig", "train", [], text_key="same caption"),
               ("dgm4", "b"): SplitInput("text_swap", "val", [], text_key="same caption"),
               ("dgm4", "c"): SplitInput("orig", "train", [], text_key="other")}
    placement, _ = split_all.assign_all(records, {})
    assert placement[("dgm4", "a")] == placement[("dgm4", "b")] == "val"
    assert placement[("dgm4", "c")] == "train"


# --------------------------------------------------------------------------
# processed-layer usability: required slots vs optional evidence sets
# --------------------------------------------------------------------------


def test_a_removed_evidence_image_is_excluded_not_missing():
    from scripts.build_processed import missing_images

    ok = {"role": "evidence", "status": "usable"}
    assert missing_images([ok, {"role": "evidence", "status": "verdict"}]) == 0
    assert missing_images([ok, {"role": "evidence", "status": "furniture"}]) == 0
    assert missing_images([{"role": "evidence", "status": "wrong_image"}]) == 0
    # an evidence image that was never there IS missing
    assert missing_images([ok, {"role": "evidence", "status": "not_on_disk"}]) == 1
    assert missing_images([{"role": "evidence", "status": "undecodable"}]) == 1


def test_a_required_slot_is_missing_whatever_replaced_it():
    from scripts.build_processed import missing_images

    for status in ("placeholder", "furniture", "blank", "verdict", "wrong_image",
                   "dead:gone_404+no_snapshot", "not_on_disk"):
        for role in ("claim", "document", "pair", "post"):
            assert missing_images([{"role": role, "status": status}]) == 1, (role, status)
    assert missing_images([{"role": "claim", "status": "usable"}]) == 0
