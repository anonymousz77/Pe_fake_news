"""M4FC's loader: the context items reach meta verbatim, and nothing else changes.

Offline: a synthetic M4FC.json in a temporary directory stands in for raw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import records  # noqa: E402


def _row(**over) -> dict:
    row = {"image_path": "images/a-slug.png", "claim": "A claim.", "verdict_coarse": "false",
           "verdict": "false (out-of-context)", "split": "dev", "image_url": "https://x.example/a.png",
           "fc_org": "x.example", "fc_pub_date": "2021-12-19 19:36:09",
           "is_manipulated_fake": False, "is_ai_generated": False,
           "claimed_date": "2021", "date": "2015-06; 2016", "claimed_location": "Lagos, Nigeria",
           "location": "not enough information", "claimed_people": ["A. Person", "B. Person"],
           "people": "unspecified", "claimed_event": "a protest", "event": "a concert",
           "claimed_things": "not enough information", "things": "a stage",
           "true_caption": "should not be read", "article_url": "https://x.example/a-slug/"}
    row.update(over)
    return row


def test_m4fc_meta_carries_every_item_verbatim(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    rows = [_row(), _row(image_path="images/b.png", split="test", verdict_coarse="true",
                         verdict="true", claimed_date="not enough information")]
    (tmp_path / "data" / "M4FC.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(records, "raw_dir", lambda name: tmp_path)
    out = list(records.load_m4fc())
    assert [r.record_id for r in out] == ["images/a-slug.png", "images/b.png"]
    # the label, split and image are as before
    assert [r.label for r in out] == ["false", "true"]
    assert [r.official_split for r in out] == ["val", "test"]
    assert out[0].images[0].ref == "images/a-slug.png"
    meta = out[0].meta
    assert list(meta) == ["fc_org", "fc_pub_date", *records.M4FC_META]
    for key in records.M4FC_META:
        assert meta[key] == rows[0][key], key  # lists stay lists; absent markers stay as written
    assert meta["claimed_people"] == ["A. Person", "B. Person"]
    assert meta["location"] == "not enough information" and meta["people"] == "unspecified"
    assert out[1].meta["claimed_date"] == "not enough information"
    # fields the block does not use stay out
    assert not {"true_caption", "article_url", "image_url", "claim"} & set(meta)


def test_the_meta_keys_are_the_planned_set():
    assert records.M4FC_META == (
        "verdict", "is_manipulated_fake", "is_ai_generated",
        "claimed_date", "date", "claimed_location", "location", "claimed_people", "people",
        "claimed_event", "event", "claimed_things", "things")
