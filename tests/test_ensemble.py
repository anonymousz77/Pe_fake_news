"""The ensemble (step 8): offline contracts. No model is loaded here."""
from __future__ import annotations

import json
import re

import pytest

from scripts import consistency as cs
from scripts import ensemble as en
from tests.test_consistency import SHA, SLUG, canary_row, slug_forms

OOC = "false (out-of-context)"


def test_every_scorer_is_pinned_and_none_is_non_commercial():
    import yaml

    doc = yaml.safe_load(en.ENS_CONFIG.read_text(encoding="utf-8"))
    assert set(doc["models"]) == {"openclip_h14", "siglip_so400m", "blip_itm_large", "blip2_itm_g",
                                  "nli_deberta_large"}
    for name, spec in doc["models"].items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), name
        assert spec["licence"] in {"mit", "apache-2.0", "bsd-3-clause"}, name
        assert spec["files"] and all("*" not in f for f in spec["files"]), name  # exact files, no globs
        assert spec["kind"] in en.BATCH and spec["loader"] in {"clip", "siglip", "blip", "blip2", "nli"}
    # the person's decisions: OpenAI's CLIP is out (its card rules out deployed use); ANLI-trained NLI is out
    assert "openai/clip-vit-large-patch14" in doc["excluded"]
    assert not any("openai/" in s["repo"] or "anli" in s["repo"].casefold() for s in doc["models"].values())


def _record(row, key, sha=SHA, dataset="m4fc", split="train", cls=OOC):
    return {"key": key, "dataset": dataset, "split": split, "class": cls, "sha256": sha,
            "path": row["image_paths"][0], "claim": row["text"], "claim_key": cs.claim_key(row["text"]), "row": row}


def test_no_record_id_reaches_a_scorer_input(tmp_path, monkeypatch):
    """The canary row through the input builder: its slug is in no text a model
    reads, and its path only in the locator, which no model reads."""
    row = canary_row()
    doc = cs.Document("d" * 64, "Floods hit the coast", "2020-01-01", "news.example", "Water rose overnight.")
    pairs, nli, images, _tally = en.build_inputs(
        [_record(row, "rkey1")], {"m4fc": {SHA: ("FLOOD WARNING", "A crowded street.")}}, {SHA: (doc,)})
    texts = json.dumps({"pairs": pairs, "nli": nli}).casefold()
    assert all(form.casefold() not in texts for form in slug_forms())
    assert "data/raw" not in texts and not any("path" in p or "image" in p for p in pairs)
    assert images == {SHA: row["image_paths"][0]}  # the locator: never a model input
    assert {n["slot"] for n in nli} == {"description", "ocr", "doc1"}
    assert all(n["hypothesis"] == row["text"] for n in nli)
    # a claim may say what its own slug says (content, as in B's and C's prompts) ...
    says = canary_row(text=f"Photo {SLUG.replace('-', ' ')} again")
    assert en.build_inputs([_record(says, "rkey2")], {}, {})[0][0]["text"] == says["text"]
    # ... but an id shape in any text is refused outright, so the server's strict check never meets one
    for leaky in (canary_row(text=f"See images/{SLUG}.png for the flood"), canary_row(text="see data/raw/x")):
        with pytest.raises(cs.GuardError):
            en.build_inputs([_record(leaky, "rkey3")], {}, {})
    # the runner refuses a file that is not the one its sidecar describes
    monkeypatch.setattr(en, "PAIRS", tmp_path / "pairs_in.jsonl")
    en.write_input(en.PAIRS, pairs, "pairs")
    assert en.read_input(en.PAIRS, "pairs") == pairs
    with pytest.raises(SystemExit):
        en.read_input(en.PAIRS, "nli")
    en.PAIRS.write_text(en.PAIRS.read_text(encoding="utf-8").replace("crowd", "crown"), encoding="utf-8")
    with pytest.raises(SystemExit):
        en.read_input(en.PAIRS, "pairs")


def test_verite_and_factify2_get_no_retrieved_documents():
    row = canary_row(dataset="verite", record_id="7", image_paths=["data/raw/verite/images/x.jpg"])
    doc = cs.Document("d" * 64, "A title", "2020-01-01", "news.example", "An excerpt.")
    _pairs, nli, _images, _t = en.build_inputs([_record(row, "rv", dataset="verite", split="test", cls="true")],
                                              {"verite": {SHA: (None, "A street.")}}, {SHA: (doc,)})
    assert [n["slot"] for n in nli] == ["description"]  # no OCR, and no documents outside M4FC


def _feat(i, cls, **over):
    f = dict.fromkeys(c for cols in en.GROUPS.values() for c in cols)
    f.update({"openclip_cos": 0.3 - 0.05 * (cls != "true") + 0.001 * i, "siglip_cos": 0.1, "blip_p_match": 0.5,
              "blip_itc": 0.4, "blip2_p_match": 0.5, "blip2_itc": 0.4, "nli_desc_entail": 0.2,
              "nli_desc_contra": 0.1, "cons_score": 0.0, "cons_decided": 0.0,
              **{c: 0.0 for c in en.GROUPS["consistency"][2:]}})
    f.update(over)
    return f


def test_a_missing_value_carries_its_flag_and_never_reads_as_zero():
    feats = [_feat(0, "true"), _feat(1, OOC, nli_ocr_entail=0.4, nli_ocr_contra=0.2)]
    x, cols = en.matrix(feats, ["openclip_h14", "nli_image"], nan=False)
    assert cols[-1] == "nli_ocr_missing" and x[:, -1].tolist() == [1.0, 0.0]
    xn, coln = en.matrix(feats, ["openclip_h14", "nli_image"], nan=True)
    assert "nli_ocr_missing" not in coln and str(xn[0, coln.index("nli_ocr_entail")]) == "nan"


def test_folds_keep_a_shared_image_or_claim_together_and_targets_follow_the_tasks():
    recs = [{"key": f"k{i}", "sha256": f"img{i // 2}", "claim_key": f"c{i}"} for i in range(40)]
    folds = en.folds_of(recs)
    assert all(folds[i] == folds[i + 1] for i in range(0, 40, 2)) and len(set(folds)) > 1
    assert en.target({"dataset": "m4fc", "class": "true"}) == 0
    assert en.target({"dataset": "m4fc", "class": "false (manipulated/fake)"}) == 1
    assert en.target({"dataset": "m4fc", "class": "mixture"}) is None
    assert en.target({"dataset": "verite", "class": "miscaptioned"}) == 1
    assert en.target({"dataset": "factify2", "class": "Support_Text", "reason_code": "none"}) == 0
    assert en.target({"dataset": "factify2", "class": "Insufficient_Text", "reason_code": "content_unverified"}) == 1


def test_scikit_learn_is_used_only_at_track_b_s_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(en, "TRACKB_REQUIREMENTS", tmp_path / "requirements-trackb.txt")
    assert en.sklearn_pin()[0] is None and "does not exist" in en.sklearn_pin()[1]
    (tmp_path / "requirements-trackb.txt").write_text("numpy==2.0\nscikit-learn==0.0.1\n", encoding="utf-8")
    version, why = en.sklearn_pin()
    assert version is None and ("not installed" in why or "is not the pinned" in why)
    assert [c.kind for c in en.candidates(gb=False)] == ["lr", "lr", "lr"]


def test_the_stance_file_is_read_only_and_its_schema_is_never_guessed(tmp_path, monkeypatch):
    path = tmp_path / "verdicts.jsonl"
    monkeypatch.setattr(en, "STANCE", path)
    assert en.stance_rows()[0] == {}
    good = {"claim_key": "c1", "judge": "nli", "verdict": "refuted", "reason_code": "content_refuted",
            "p_supported": 0.1, "p_refuted": 0.8, "p_nei": 0.1, "doc_ids": ["d1"], "status": "ok"}
    qwen = dict(good, judge="qwen", verdict="supported", p_supported=0.9, p_refuted=0.05)
    path.write_text("\n".join(json.dumps(r) for r in (good, qwen, dict(good, claim_key="c2", status="no_bound"),
                                                     dict(qwen, claim_key="c3", status="uncited"))) + "\n",
                    encoding="utf-8")
    before = path.read_bytes()
    rows, found = en.stance_rows()
    assert set(rows) == {"c1"} and rows["c1"]["judge"] == "nli" and path.read_bytes() == before  # NLI-B: features
    assert set(en.qwen_rows()) == {"c1"} and en.qwen_rows()["c1"]["verdict"] == "supported"  # cited Qwen: words
    path.write_text(json.dumps({k: v for k, v in good.items() if k != "p_nei"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        en.stance_rows()


def test_the_combiner_fits_on_m4fc_train_and_reports_gate_e1(tmp_path, monkeypatch):
    """End to end on synthetic ledgers: cross-validation within M4FC train,
    the ablation, the Factify2 variant, and a report with no record id."""
    monkeypatch.setattr(en, "ENS_DIR", tmp_path)
    for name, value in (("RECORDS", "records.jsonl"), ("NLI_IN", "nli_in.jsonl"), ("SCORES", "scores"),
                        ("REPORT_E1", "e1.json"), ("REPORT_E1_MD", "e1.md"), ("STANCE", "none.jsonl"),
                        ("TRACKB_REQUIREMENTS", "none.txt"), ("TRACKB_BASELINES", "none.json")):
        monkeypatch.setattr(en, name, tmp_path / value)
    monkeypatch.setattr(cs, "C_VERDICTS", tmp_path / "verdicts.jsonl")
    monkeypatch.setattr(cs, "C_CLIP", tmp_path / "clip.jsonl")
    monkeypatch.setattr(cs, "C_FUSED", tmp_path / "fused.json")
    rng = __import__("random").Random(1)
    records, verdicts, clip, nli = [], [], [], []
    plan = ([("m4fc", "train", c) for c in ["true"] * 30 + [OOC] * 40 + ["false (manipulated/fake)"] * 10]
            + [("m4fc", "val", c) for c in ["true"] * 12 + [OOC] * 12]
            + [("verite", "test", c) for c in ["true", "miscaptioned", "out-of-context"] * 8]
            + [("factify2", "train", c) for c in ["Support_Text", "Refute"] * 10])
    for i, (ds, split, cls) in enumerate(plan):
        key = f"r{i:04d}"
        bad = cls not in ("true", "Support_Text")
        records.append({"key": key, "dataset": ds, "split": split, "class": cls, "sha256": f"img{i}",
                        "claim_key": f"c{i}", "reason_code": ("none" if not bad else "content_refuted")
                        if ds == "factify2" else None})
        s = 0.30 - 0.04 * bad + rng.uniform(-0.03, 0.03)
        for name in ("openclip_h14", "siglip_so400m"):
            en.SCORES.mkdir(exist_ok=True)
            with en.ledger_path(name).open("a", encoding="utf-8") as h:
                h.write(json.dumps({"key": key, "status": "ok", "cosine": round(s, 4)}) + "\n")
        for name in ("blip_itm_large", "blip2_itm_g"):
            with en.ledger_path(name).open("a", encoding="utf-8") as h:
                h.write(json.dumps({"key": key, "status": "ok", "p_match": round(1 - bad * 0.3, 3),
                                    "itc": round(s, 4)}) + "\n")
        nk = cs.neutral_key("n", f"{key}|description")
        nli.append({"key": nk, "record": key, "slot": "description", "premise": "a street", "hypothesis": "a claim"})
        with en.ledger_path("nli_deberta_large").open("a", encoding="utf-8") as h:
            h.write(json.dumps({"key": nk, "status": "ok", "entailment": 0.5, "neutral": 0.3,
                                "contradiction": 0.2 + 0.1 * bad}) + "\n")
        if ds != "factify2":
            sc = {"score": 0.5 * bad, "abstained": i % 3 == 0, "by": None}
            verdicts.append({"key": key, "variant": "extracted", "items": {}, "variants": {},
                             "scores": {"primary": sc}})
            clip.append({"key": key, "similarity": round(s + rng.uniform(-0.05, 0.05), 4)})
    en.write_jsonl(en.RECORDS, records)
    en.write_input(en.NLI_IN, nli, "nli")
    import numpy as np  # Track B's CLIP L/14 embeddings, in its layout: unit vectors by "img:"/"txt:" key

    keys, vecs = [], []
    for i, r in enumerate(records):
        a = np.array([1.0, 0.0])
        b = np.array([np.cos(0.5 + 0.4 * (r["class"] not in ("true", "Support_Text")) + 0.01 * (i % 7)), 0.0])
        b[1] = np.sqrt(1 - b[0] ** 2)
        keys += [f"img:{r['sha256']}", f"txt:{r['claim_key']}"]
        vecs += [a, b]
    monkeypatch.setattr(en, "TRACKB_L14", tmp_path / "clip_l14.npz")
    np.savez(en.TRACKB_L14, keys=np.array(keys), vecs=np.array(vecs, dtype=np.float32))
    en.write_jsonl(cs.C_VERDICTS, verdicts)
    en.write_jsonl(cs.C_CLIP, clip)
    assert en.main(["fit"]) == 0
    rep = json.loads(en.REPORT_E1.read_text(encoding="utf-8"))
    assert rep["train"]["records"] == 80 and rep["train"]["true"] == 30 and rep["sklearn"]["version"] is None
    assert rep["train"]["chosen"].startswith("lr(") and len(rep["train"]["cv"]) == 3
    assert rep["stance"].startswith("absent") and "stance" not in rep["groups"]
    assert set(rep["corpora"]) == {"m4fc val", "verite"}
    pairs = [(a["positive"], a["against"]) for a in rep["corpora"]["verite"]["auroc"]]
    assert ("out-of-context", "clip_b32") in pairs and ("miscaptioned", "variant:+factify2") in pairs
    assert pairs[0] == ("out-of-context", "clip_l14")  # the bar comes first, paired on the same records
    assert rep["clip_l14"]["joined"].startswith("128 of 128") and set(rep["clip_l14"]["beaten"]) == {
        "m4fc val: false (out-of-context) vs true", "m4fc val: false vs true", "verite: out-of-context vs true",
        "verite: miscaptioned vs true"}
    assert "The bar: CLIP ViT-L/14" in en.REPORT_E1_MD.read_text(encoding="utf-8")
    assert rep["factify2_variant"]["records"] == 20
    assert set(rep["ablation_train"]) == set(rep["groups"])
    assert "## Ablation" in en.REPORT_E1_MD.read_text(encoding="utf-8")


def test_the_route_reads_what_an_image_is_and_scores_on_train_percentiles():
    assert en.screenshot_layout("The image is a screenshot of a tweet. It shows a man.")
    assert not en.screenshot_layout("The image is a photograph. A screenshot of it went viral.")
    row = {"ocr": {"error": None, "input": {"width": 100, "height": 50},
                   "regions": [{"polygon": [[0, 0], [50, 0], [50, 10], [0, 10]], "text": "HEADLINE"}]}}
    assert en.ocr_share(row) == 0.1 and en.ocr_share({"ocr": {"error": "x"}}) is None
    assert en.to_m4fc_branch({"screenshot": True, "ocr_share": 0.0}, None)  # a screenshot, whatever its text
    assert en.to_m4fc_branch({"screenshot": False, "ocr_share": 0.2}, 0.1)
    assert not en.to_m4fc_branch({"screenshot": False, "ocr_share": None}, 0.1)  # a failed OCR covers nothing
    assert not en.to_m4fc_branch({"screenshot": False, "ocr_share": 0.9}, None)
    table = [0.1, 0.2, 0.2, 0.3]
    assert en.pct(table, 0.2) == 0.5 and en.pct(table, 0.0) == 0.0 and en.pct(table, 1.0) == 1.0
    tables = {c: [0.0, 1.0] for c, _ in en.PHOTO}
    f = {"openclip_cos": -0.5, "siglip_cos": -0.5, "blip_p_match": -0.5, "blip2_p_match": None}
    assert en.photo_raw(f, tables) == 0.5  # each in its mismatch direction, over the scores it has


def test_the_sign_constrained_refit_holds_every_sign():
    import random as _r

    rng = _r.Random(3)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    y = [int(x[0] + 0.5 * rng.gauss(0, 1) > 0) for x in X]  # the first feature pushes toward 1
    free = en.fit_logistic_signed(X, y, [0, 0], 1.0)
    held = en.fit_logistic_signed(X, y, [-1, 0], 1.0)  # forbid what the data want
    assert free["coef"][0] > 0.5 and held["coef"][0] <= 0 and held["at_bound"] == 1 and held["converged"]


def test_a_frozen_system_refuses_a_body_that_no_longer_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(en, "SYSTEM", tmp_path / "system.json")
    body = {"route": {"t": 0.04}, "ensemble": {"form": "lr(l2=10.0)"}}
    en.SYSTEM.write_text(json.dumps({"system_sha256": __import__("hashlib").sha256(cs.canonical(body)).hexdigest(),
                                     "system": body}), encoding="utf-8")
    assert en.read_system()[0] == body
    doc = json.loads(en.SYSTEM.read_text(encoding="utf-8"))
    doc["system"]["route"]["t"] = 0.05
    en.SYSTEM.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit):
        en.read_system()


def test_the_official_test_opens_once_and_is_scored_once(tmp_path, monkeypatch):
    """--final-test: refused without a frozen system, opened once, scored once,
    and scored only under the system it was opened with."""
    import hashlib as _h

    monkeypatch.setattr(en, "SYSTEM", tmp_path / "system.json")
    monkeypatch.setattr(en, "FINAL_OPENED", tmp_path / "opened.json")
    monkeypatch.setattr(en, "FINAL_SCORED", tmp_path / "scored.json")
    with pytest.raises((SystemExit, FileNotFoundError)):
        en.cmd_final_open(None)  # nothing frozen
    body = {"post_hoc": "post hoc", "route": {"t": 0.04}}
    sha = _h.sha256(cs.canonical(body)).hexdigest()
    en.SYSTEM.write_text(json.dumps({"system_sha256": sha, "system": body}), encoding="utf-8")
    en.FINAL_OPENED.write_text(json.dumps({"opened_at": "2026-11-11T00:00:00Z", "system_sha256": sha}),
                               encoding="utf-8")
    with pytest.raises(SystemExit, match="opens once"):
        en.cmd_final_open(None)
    en.FINAL_SCORED.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="scored once"):
        en.cmd_final_score(None)
    en.FINAL_SCORED.unlink()
    en.FINAL_OPENED.write_text(json.dumps({"opened_at": "x", "system_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the one the test was opened under"):
        en.cmd_final_score(None)
