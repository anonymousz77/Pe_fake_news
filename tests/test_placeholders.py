"""The placeholder registry: images that decode but carry no evidence.

Offline and data-free, like every other suite here.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import CROSS_CORPUS_YAML, PLACEHOLDERS_YAML  # noqa: E402
from scripts import placeholders  # noqa: E402


def _png(colour=(1, 2, 3), size=(4, 4)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# the register itself
# --------------------------------------------------------------------------


def test_the_registry_file_exists_and_is_versioned():
    doc = yaml.safe_load(PLACEHOLDERS_YAML.read_text(encoding="utf-8"))
    assert doc["version"] >= 1, "the list will grow; it must carry a version"
    assert doc["entries"], "an empty registry is almost certainly a mistake"


def test_every_entry_carries_what_a_later_reader_needs():
    required = {"sha256", "status", "description", "width", "height", "bytes",
                "identified_by"}
    for entry in placeholders.registry().values():
        missing = required - set(entry)
        assert not missing, f"{entry.get('sha256')} is missing {sorted(missing)}"
        assert entry["description"].strip(), "a hash with no description is a mystery"
        assert entry["identified_by"].strip()


def test_digests_are_full_length_sha256():
    for digest in placeholders.registry():
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), digest


def test_generic_stock_is_recorded_but_not_counted_as_missing():
    """The judgement call stays visible instead of being baked into a number."""
    stock = [h for h, e in placeholders.registry().items()
             if e["status"] == "generic_stock"]
    assert stock, "the three building photos should still be on record"
    for digest in stock:
        assert not placeholders.is_missing(digest)


def test_the_imgur_placeholder_is_registered():
    known = "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9"
    assert placeholders.is_missing(known)
    assert "imgur" in placeholders.describe(known).lower()


def test_an_unknown_digest_is_not_missing():
    assert not placeholders.is_missing(hashlib.sha256(b"anything else").hexdigest())
    assert not placeholders.is_missing(None)
    assert not placeholders.is_missing("")


def test_an_unknown_status_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "placeholders.yaml"
    bad.write_text("version: 1\nentries:\n  - sha256: " + "a" * 64 +
                   "\n    status: vibes\n", encoding="utf-8")
    monkeypatch.setattr(placeholders, "PLACEHOLDERS_YAML", bad)
    placeholders.registry.cache_clear()
    with pytest.raises(ValueError, match="unknown status"):
        placeholders.registry()
    placeholders.registry.cache_clear()


# --------------------------------------------------------------------------
# the fetch-time gate
# --------------------------------------------------------------------------


def test_the_gate_rejects_registered_bytes_even_though_they_decode():
    """The whole point: these are valid images. Decodability cannot catch them."""
    from scripts import hydrate

    raw = (Path("data/raw/fakeddit/media/images/c3qexks.jpg"))
    if not raw.is_file():
        pytest.skip("fakeddit media not on disk")
    body = raw.read_bytes()
    assert hydrate.looks_like_an_image(body) is None, "it IS a well-formed image"
    assert hydrate.not_a_placeholder(body).startswith("placeholder_image")


def test_the_gate_passes_an_ordinary_image():
    from scripts import hydrate
    assert hydrate.not_a_placeholder(_png()) is None


def test_a_placeholder_response_is_retried_not_treated_as_terminal(tmp_path):
    """A host may serve junk once and the real image next time."""
    import random
    from scripts import hydrate

    known = _png()
    digest = hashlib.sha256(known).hexdigest()
    placeholders.missing_hashes.cache_clear()
    original = placeholders.missing_hashes
    placeholders.missing_hashes = lambda: frozenset({digest})
    try:
        calls = {"n": 0}

        class _S:
            def get(self, url, **kwargs):
                calls["n"] += 1

                class R:
                    status_code = 200
                    content = known
                return R()

        dest = tmp_path / "x.png"
        ok, reason, _n, status = hydrate.fetch_url(
            _S(), "https://e.example/x.png", dest, timeout=1.0, retries=2,
            rng=random.Random(0), validator=hydrate.not_a_placeholder)
        assert ok is False
        assert reason.startswith("placeholder_image")
        assert status == 200
        assert calls["n"] == 3, "retried the full budget rather than giving up at once"
        assert not dest.exists()
    finally:
        placeholders.missing_hashes = original
        placeholders.missing_hashes.cache_clear()


# --------------------------------------------------------------------------
# cross-corpus leakage
# --------------------------------------------------------------------------


def test_cross_corpus_pairs_span_two_corpora_and_are_not_placeholders():
    pairs = placeholders.cross_corpus_pairs()
    assert pairs, "the 2026-09-21 audit found shared images; an empty list is a lost file"
    for pair in pairs:
        assert len(pair["sha256"]) == 64
        assert len({m["dataset"] for m in pair["members"]}) >= 2
        for m in pair["members"]:
            assert m.get("record_ids") or m.get("key")
        # a registry match is a missing image, not a shared one
        assert not placeholders.is_missing(pair["sha256"]) or pair.get("kind") == "blank_placeholder"


SYNTHETIC = ({"sha256": "a" * 64, "kind": "shared_source_image",
              "members": [{"dataset": "factify2", "record_ids": ["train_1"]},
                          {"dataset": "verite", "record_ids": ["51", "52"]}]},
             {"sha256": "b" * 64, "kind": "shared_source_image",
              "members": [{"dataset": "mocheg", "record_ids": ["train_9"]},
                          {"dataset": "m4fc", "record_ids": ["x.jpg"]}]})


def test_excluded_ids_are_per_pair_of_corpora(monkeypatch):
    monkeypatch.setattr(placeholders, "cross_corpus_pairs", lambda: SYNTHETIC)
    assert placeholders.excluded_keys("verite", against="factify2") == {"51", "52"}
    assert placeholders.excluded_keys("factify2", against="verite") == {"train_1"}
    assert placeholders.excluded_keys("verite", against="mocheg") == frozenset()


def test_a_leaking_evaluation_set_raises_rather_than_warns(monkeypatch):
    """Enforced, not commented: a note in a README does not stop a run."""
    monkeypatch.setattr(placeholders, "cross_corpus_pairs", lambda: SYNTHETIC)
    with pytest.raises(ValueError, match="cross-corpus leakage"):
        placeholders.assert_no_cross_corpus_leak("verite", ["51", "7"], against="factify2")
    placeholders.assert_no_cross_corpus_leak("verite", ["7", "8"], against="factify2")


def test_fingerprint_rejections_name_bytes_rows_and_no_label_bearing_path():
    """Each entry names a dataset, the sha256 of the bytes rejected, the record
    rows, a label-free file alias, the evidence and a reason -- and is not ALSO
    a registry entry (content-free bytes belong in the registry, wrong-but-real
    ones here). No field may carry a VERITE image name: true_N / false_N is the
    label."""
    import re

    import yaml
    from configs.paths import FINGERPRINT_REJECTIONS_YAML
    from scripts.placeholders import registry

    if not FINGERPRINT_REJECTIONS_YAML.is_file():
        pytest.skip("no rejections recorded")
    text = FINGERPRINT_REJECTIONS_YAML.read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z])(true|false)_\d+", text)
    doc = yaml.safe_load(text)
    assert isinstance(doc["version"], int)
    seen = set()
    for e in doc["rejections"]:
        assert {"dataset", "file", "sha256", "reason", "identified_by", "cosine_to_fingerprint", "rows"} <= set(e)
        assert "path" not in e
        assert e["file"].startswith(f"data/raw/{e['dataset']}/")
        assert len(e["sha256"]) == 64 and e["sha256"] not in registry()
        assert (e["dataset"], e["sha256"]) not in seen
        seen.add((e["dataset"], e["sha256"]))