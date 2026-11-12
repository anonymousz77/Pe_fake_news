"""Track B's text-side check: what may be read, what may reach a model, and the
contract of the ensemble's file.

Offline. A canary row -- its slug in the record id, the image path and meta --
goes through every builder of a model input and a query, and must come out in
no form. The helpers are copied from tests/test_consistency.py, not imported:
that file belongs to another block and changes under it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.paths import PROCESSED  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts import textcheck as tc  # noqa: E402

HAVE_PROCESSED = (PROCESSED / "records" / "m4fc.jsonl.gz").is_file()
needs_processed = pytest.mark.skipif(not HAVE_PROCESSED, reason="needs the processed layer")

#: A slug that states a finding, as M4FC's do, and that no content will ever hold.
SLUG = "qzx-canary-photo-falsely-shared-as-flood-vkj"
CLAIM = "A crowd gathers outside a stadium after the match."


def canary_row(**over) -> dict:
    row = {"dataset": "m4fc", "record_id": f"images/{SLUG}.png", "text": CLAIM,
           "image_paths": [f"data/raw/m4fc/media/images/images_{SLUG}_png.png"],
           "label": "false", "split": "train", "usable": True, "reason_code": "content_refuted",
           "meta": {"fc_org": "canary.example", "fc_pub_date": "2021-01-02",
                    "official_split": "train", "verdict": "false (out-of-context)",
                    "article_url": f"https://canary.example/{SLUG}/"}}
    row.update(over)
    return row


def slug_forms() -> list[str]:
    """The canary in every form it could leak in."""
    return [SLUG, SLUG.replace("-", " "), f"images/{SLUG}.png", f"images_{SLUG}_png",
            "canary.example"]


def assert_no_canary(obj) -> None:
    text = json.dumps(obj, ensure_ascii=False).casefold()
    for form in slug_forms():
        assert form.casefold() not in text, form


def ck(text: str) -> str:
    return cs.claim_key(text)


DOC = "a" * 64
DOC2 = "b" * 64


# --------------------------------------------------------------------------
# reading: which corpora, which splits, which bound
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["mocheg", "liar", "verite", "factify2", "fakeddit", "isot"])
def test_corpora_without_a_date_or_outside_the_check_are_refused(dataset):
    with pytest.raises(SystemExit, match="no evidence"):
        tc.units(dataset)


def test_the_check_reads_m4fc_train_and_val_only():
    assert tc.DATASETS["m4fc"] == ("train", "val")
    assert set(tc.DATASETS) == {"m4fc", "averitec", "averimatec"}
    assert tc.EVAL_SPLIT == {"m4fc": "val", "averitec": "val", "averimatec": "test"}


def test_m4fc_test_is_refused_through_consistency_rows():
    with pytest.raises(cs.GuardError):
        list(cs.rows("m4fc", ("train", "test")))


def _fake_rows(monkeypatch, rows_):
    monkeypatch.setattr(cs, "rows", lambda dataset, splits, usable=True: iter(rows_))


def test_a_unit_holds_the_claim_and_its_bound_and_no_identifier(monkeypatch):
    _fake_rows(monkeypatch, [canary_row()])
    (u,) = tc.units("m4fc")
    assert u["query"] == CLAIM
    assert (u["before"], u["bound"]) == ("2021-01-02", "fact_check_date")
    assert u["claim_key"] == ck(CLAIM) and u["records"] == 1 and u["splits"] == ["train"]
    assert set(u) == {"key", "claim_key", "query", "before", "bound", "records", "splits"}
    assert_no_canary(u)


def test_records_share_a_unit_only_when_claim_and_bound_agree(monkeypatch):
    late = canary_row(meta={**canary_row()["meta"], "fc_pub_date": "2021-03-04"})
    other = canary_row(text="Another claim entirely, about a bridge.")
    _fake_rows(monkeypatch, [canary_row(), canary_row(split="val"), late, other])
    us = tc.units("m4fc")
    by = {(u["claim_key"], u["before"]): u for u in us}
    assert len(us) == 3
    assert by[(ck(CLAIM), "2021-01-02")]["records"] == 2
    assert by[(ck(CLAIM), "2021-01-02")]["splits"] == ["train", "val"]
    assert by[(ck(CLAIM), "2021-03-04")]["records"] == 1
    assert tc.unit_summary("m4fc", us)["claims_with_several_bounds"] == 1


def test_a_record_without_a_date_is_a_unit_without_a_bound(monkeypatch):
    row = canary_row(dataset="averitec", meta={"claim_date": None})
    _fake_rows(monkeypatch, [row])
    (u,) = tc.units("averitec")
    assert u["before"] is None and u["bound"] is None


def test_unit_keys_are_neutral_and_stable():
    a = tc.unit_key(ck(CLAIM), "2021-01-02")
    assert a == tc.unit_key(ck(CLAIM), "2021-01-02")
    assert a != tc.unit_key(ck(CLAIM), "2021-01-03")
    assert ck(CLAIM)[:15] not in a and len(a) == 16


# --------------------------------------------------------------------------
# the view: the only thing a model sees
# --------------------------------------------------------------------------


def test_the_view_has_no_field_for_an_identifier_a_label_or_the_bound():
    names = {f.name for f in fields(tc.ClaimView)}
    assert names == {"claim", "documents"}


def test_the_view_content_is_its_claim_and_documents():
    d = cs.Document(DOC, "A title", "2020-01-01", "news.example", "An excerpt.")
    v = tc.ClaimView(CLAIM, (d,))
    for part in (CLAIM, DOC, "A title", "2020-01-01", "news.example", "An excerpt."):
        assert part in v.content()


def test_a_query_carrying_an_unexplained_slug_is_refused():
    forms = cs.id_forms(canary_row())
    v = tc.ClaimView(CLAIM)
    assert cs.refuse_ids(v.claim, forms, explained=v.content()) == CLAIM
    with pytest.raises(cs.GuardError):
        cs.refuse_ids(f"{CLAIM} {SLUG}", forms, explained=v.content())


# --------------------------------------------------------------------------
# retrieval: one recipe, the consistency block's
# --------------------------------------------------------------------------


def test_the_retrieval_config_pins_the_shared_functions(monkeypatch):
    monkeypatch.setattr(cs, "retrieval_config", lambda: {"k": cs.RETRIEVE_K})  # no index needed
    cfg = tc.retrieval_config()
    assert set(cfg["functions_sha256"]) == set(tc.SHARED_FUNCTIONS) == {
        "excerpt", "hit_row", "read_doc", "documents_of"}
    assert cfg["k"] == 10 and cfg["textcheck_version"] == tc.TEXTCHECK_VERSION
    first = tc.config_id(cfg)
    monkeypatch.setattr(cs, "excerpt", lambda text, title, terms, n=600: ("", "opening"))
    assert tc.config_id(tc.retrieval_config()) != first


def test_a_hit_keeps_the_consistency_fields_and_adds_its_source():
    class Hit:
        doc_id, title, published_at, source_domain, url, score = (
            DOC, "Stadium crowd", "2020-12-31", "news.example", "https://x", -3.2)

    doc = {"text": "Stadium crowd. A crowd gathers outside a stadium.", "title": "Stadium crowd",
           "provenance": {"source": "enrich"}}
    row = tc.hit_with_source(Hit(), ["crowd", "stadium"], doc)
    assert row["source"] == "enrich"
    assert set(row) == {"doc_id", "title", "published_at", "host", "score", "excerpt", "anchor",
                        "source"}
    assert "url" not in row
    assert tc.hit_with_source(Hit(), ["crowd"], None)["anchor"] == "doc_missing"


def test_retrieval_refuses_a_unit_without_a_bound(tmp_path):
    called = []

    def search(q, before):
        called.append(q)
        return [], [], False

    items = [{"key": "u1", "query": CLAIM, "before": None, "bound": None},
             {"key": "u2", "query": "!!!", "before": "2021-01-01", "bound": "claim_date"}]
    tc.run_retrieval(items, tmp_path / "x.jsonl", "cid", search, doc=lambda d: None)
    held = tc.read_retrieval(tmp_path / "x.jsonl", "cid")
    assert held["u1"]["error"] == "no_bound" and held["u2"]["error"] == "empty_query"
    assert not called
    with pytest.raises(SystemExit, match="refusing to mix runs"):
        tc.read_retrieval(tmp_path / "x.jsonl", "other")


# --------------------------------------------------------------------------
# the ensemble's file: the contract
# --------------------------------------------------------------------------

P = {"supported": 0.2, "refuted": 0.7, "not_enough_evidence": 0.1}


def test_a_row_has_exactly_the_contract_keys_in_order():
    row = tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", P, [DOC])
    assert tuple(row) == tc.VERDICT_KEYS == (
        "claim_key", "judge", "verdict", "reason_code", "p_supported", "p_refuted", "p_nei",
        "doc_ids", "status")
    assert row["reason_code"] == "content_refuted"
    assert (row["p_supported"], row["p_refuted"], row["p_nei"]) == (0.2, 0.7, 0.1)


@pytest.mark.parametrize("verdict,code", [("supported", "none"), ("refuted", "content_refuted"),
                                          ("not_enough_evidence", "content_unverified")])
def test_reason_codes_follow_verdicts(verdict, code):
    assert tc.verdict_row(ck(CLAIM), "nli", "ok", verdict, P, [DOC])["reason_code"] == code


@pytest.mark.parametrize("judge", ["nli_a", "nli_b", "qwen_nodocs", "qwen_noenrich", "moritzlaurer"])
def test_only_qwen_and_nli_b_may_enter_the_ensembles_file(judge):
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), judge, "ok", "refuted", P, [DOC])
    assert "nli" in tc.ENSEMBLE_JUDGES and len(tc.ENSEMBLE_JUDGES) == 2
    assert "contrast only" in tc.MODEL_PINS["nli_a"]["role"]
    assert "system" in tc.MODEL_PINS["nli_b"]["role"]


def test_abstaining_rows_are_empty():
    for status in ("no_bound", "off_format"):
        row = tc.verdict_row(ck(CLAIM), "qwen", status)
        assert (row["verdict"], row["reason_code"], row["p_supported"], row["p_refuted"],
                row["p_nei"], row["doc_ids"]) == (None, None, None, None, None, [])
        with pytest.raises(cs.GuardError):
            tc.verdict_row(ck(CLAIM), "qwen", status, "refuted", P, [DOC])


def test_no_documents_is_not_enough_evidence_with_no_judge():
    row = tc.verdict_row(ck(CLAIM), "nli", "no_documents", "not_enough_evidence")
    assert row["reason_code"] == "content_unverified" and row["p_nei"] is None
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), "nli", "no_documents", "not_enough_evidence", P)


def test_a_verdict_needs_a_distribution_and_store_doc_ids():
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", {"supported": 0.5, "refuted": 0.6,
                                                            "not_enough_evidence": 0.1}, [DOC])
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", P, ["[1]"])
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", P, [DOC, DOC])
    with pytest.raises(cs.GuardError):
        tc.verdict_row(ck(CLAIM), "qwen", "uncited", "refuted", P, [DOC])
    assert tc.verdict_row(ck(CLAIM), "qwen", "uncited", "refuted", P)["doc_ids"] == []


def test_a_row_with_another_key_breaks_the_contract():
    row = tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", P, [DOC])
    assert tc.verdict_problems({**row, "dataset": "m4fc"})
    assert tc.verdict_problems({k: v for k, v in row.items() if k != "status"})
    renamed = {("p_contradiction" if k == "p_refuted" else k): v for k, v in row.items()}
    assert tc.verdict_problems(renamed)


def test_the_file_holds_one_row_per_claim_and_judge(tmp_path):
    a = tc.verdict_row(ck(CLAIM), "qwen", "ok", "refuted", P, [DOC])
    b = tc.verdict_row(ck(CLAIM), "nli", "ok", "supported", P, [DOC2])
    with pytest.raises(cs.GuardError):
        tc.write_verdicts([a, b, a], {}, tmp_path / "verdicts.jsonl")
    meta = tc.write_verdicts([b, a], {"holds": ["m4fc"]}, tmp_path / "verdicts.jsonl")
    back = tc.read_verdicts(tmp_path / "verdicts.jsonl")
    assert back == sorted([a, b], key=lambda r: (r["claim_key"], r["judge"]))
    assert meta["rows"] == 2 and meta["keys"] == list(tc.VERDICT_KEYS)
    assert meta["join"] == "consistency.claim_key"
    side = json.loads((tmp_path / "verdicts.meta.json").read_text(encoding="utf-8"))
    body = (tmp_path / "verdicts.jsonl").read_bytes()
    assert side["sha256"] == hashlib.sha256(body).hexdigest()


def test_verite_rows_abstain_for_every_claim_and_both_judges(monkeypatch):
    rows_ = [canary_row(dataset="verite", text="Caption one."),
             canary_row(dataset="verite", text="Caption one."),
             canary_row(dataset="verite", text="Caption  two.")]
    monkeypatch.setattr(cs, "evaluation_rows", lambda: rows_)
    out = tc.verite_rows()
    assert len(out) == 4
    assert {r["status"] for r in out} == {"no_bound"}
    assert {r["judge"] for r in out} == set(tc.ENSEMBLE_JUDGES)
    assert {r["claim_key"] for r in out} == {ck("Caption one."), ck("Caption two.")}
    assert_no_canary(out)


@needs_processed
def test_verite_keys_are_the_consistency_blocks_877_rows():
    rows_ = cs.evaluation_rows()
    assert len(rows_) == 877
    assert tc.verite_claim_keys() == sorted({ck(r["text"]) for r in rows_})


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def test_every_model_is_pinned_by_revision_weights_and_licence():
    for which, pin in tc.MODEL_PINS.items():
        assert len(pin["revision"]) == 40, which
        assert len(pin["weights_sha256"]) == 64 and pin["weights_bytes"] > 0, which
        assert "model.safetensors" in pin["files"], which
        assert "card_licence" in pin, which
    assert tc.MODEL_PINS["nli_b"]["card_licence"] == "apache-2.0"
    assert tc.MODEL_PINS["nli_a"]["card_licence"] == "mit"
    assert tc.MODEL_PINS["clip_l14"]["card_licence"] is None


# --------------------------------------------------------------------------
# the judge's prompts and answers
# --------------------------------------------------------------------------


def _docs(n=3):
    return tuple(cs.Document(c * 64, f"Title {i}", f"2020-01-0{i}", "news.example",
                             f"…An excerpt about the stadium crowd, number {i}…")
                 for i, c in zip(range(1, n + 1), "abcdef"))


def test_exemplars_are_well_formed_and_balanced():
    doc = tc.fewshot()
    answers = [s["answer"] for s in doc["judge"]]
    assert sorted(answers) == sorted(["supported", "refuted", "not enough evidence"] * 2)
    assert all(s["answer"] in tc.ANSWERS for s in doc["nodocs"])


def test_a_judge_prompt_shows_no_identifier_and_no_doc_id():
    view = tc.ClaimView(CLAIM, _docs())
    forms = cs.id_forms(canary_row())
    for task in tc.TASKS:
        shots = tc.fewshot()[tc.TASKS[task]["shots"]]
        msgs = tc.TASKS[task]["messages"](view, shots)
        text = cs.messages_text(msgs)
        assert_no_canary(text)
        assert all(d.doc_id not in text for d in view.documents)
        assert CLAIM in text
        shots_text = "\n".join(
            cs.messages_text(tc.TASKS[task]["messages"](tc.ClaimView(""), shots)[1:-1]))
        assert cs.refuse_ids(text, forms, explained=f"{view.content()}\n{shots_text}") == text


def test_the_judge_sees_numbered_documents_with_dates():
    text = tc.documents_text(_docs(2))
    assert text.startswith("[1] Title 1 (2020-01-01, news.example)\n…An excerpt")
    assert "\n\n[2] Title 2 (2020-01-02, news.example)" in text


@pytest.mark.parametrize("text,n,want", [
    ("supported\nDocuments: 1, 3", 3, ("supported", [1, 3], None)),
    ("Refuted.\nDocuments: 2", 3, ("refuted", [2], None)),
    ("not enough evidence\nDocuments: none", 3, ("not enough evidence", [], None)),
    ("supported\nDocuments: 2, 9", 3,
     ("supported", [2], "a number the prompt did not give, or one twice")),
    ("supported", 3, ("supported", [], "no_documents_line")),
    ("not supported\nDocuments: 1", 3, (None, [1], None)),
    ("supported\nDocuments: 1\nBecause the first document says so.", 3,
     ("supported", [1], "extra_lines")),
])
def test_answers_are_read_strictly(text, n, want):
    got = tc.parse_answer(text, n, True)
    assert (got["answer"], got["cited"], got["cite_problem"]) == want


def test_judge_status_rules():
    base = {"label_mass": 0.9, "answer": "refuted", "cited": [1], "cite_problem": None,
            "cites": True}
    assert tc.judge_status(base) == "ok"
    assert tc.judge_status({**base, "label_mass": 0.49}) == "off_format"
    assert tc.judge_status({**base, "answer": None}) == "off_format"
    assert tc.judge_status({**base, "cited": []}) == "uncited"
    assert tc.judge_status({**base, "cite_problem": "no_documents_line"}) == "partial"
    assert tc.judge_status({**base, "answer": "not enough evidence", "cited": []}) == "ok"
    assert tc.judge_status({**base, "cites": False, "cited": []}) == "ok"


def test_the_answers_are_the_three_labels_in_tie_order():
    assert list(tc.ANSWERS) == ["supported", "refuted", "not enough evidence"]
    assert list(tc.ANSWERS.values()) == list(tc.VERDICT_LABELS)


# --------------------------------------------------------------------------
# NLI: labels by name, the decisive document, the threshold
# --------------------------------------------------------------------------


def test_nli_labels_are_read_by_name_for_both_models():
    a = tc.label_index({0: "entailment", 1: "neutral", 2: "contradiction"})
    b = tc.label_index({"0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL"})
    assert a == {"entailment": 0, "neutral": 1, "contradiction": 2}
    assert b == {"entailment": 1, "neutral": 2, "contradiction": 0}
    with pytest.raises(cs.GuardError):
        tc.label_index({0: "yes", 1: "no", 2: "maybe"})


def test_an_nli_pair_is_title_and_excerpt_against_the_claim():
    pairs = tc.nli_pairs(tc.ClaimView(CLAIM, _docs(2)))
    assert pairs[0] == {"premise": "Title 1. …An excerpt about the stadium crowd, number 1…",
                        "hypothesis": CLAIM}
    assert_no_canary(pairs)


def _pr(e, n, c):
    return {"entailment": e, "neutral": n, "contradiction": c}


def test_the_nli_verdict_rests_on_the_decisive_document():
    docs = _docs(3)
    probs = [_pr(0.1, 0.8, 0.1), _pr(0.05, 0.15, 0.8), _pr(0.7, 0.2, 0.1)]
    status, verdict, p, ids = tc.nli_verdict(probs, docs, 0.5)
    assert (status, verdict, ids) == ("ok", "refuted", [docs[1].doc_id])
    assert p == pytest.approx({"supported": 0.05, "refuted": 0.8, "not_enough_evidence": 0.15})
    status, verdict, p, ids = tc.nli_verdict(probs, docs, 0.9)
    assert verdict == "not_enough_evidence" and ids == [d.doc_id for d in docs]
    assert tc.nli_verdict(probs, (), 0.5)[:2] == ("no_documents", "not_enough_evidence")
    tc.verdict_row(ck(CLAIM), "nli", *tc.nli_verdict(probs, docs, 0.5))


def test_the_threshold_is_fitted_on_what_it_is_given():
    docs = _docs(1)
    items = [("refuted", [_pr(0.1, 0.1, 0.8)], docs), ("supported", [_pr(0.9, 0.05, 0.05)], docs),
             ("not_enough_evidence", [_pr(0.3, 0.4, 0.3)], docs),
             ("not_enough_evidence", [_pr(0.45, 0.5, 0.05)], docs)]
    fit = tc.fit_tau(items)
    assert 0.45 < fit["tau"] <= 0.8 and fit["train_macro_f1"] == 1.0 and fit["train_units"] == 4


def test_qwen_verdicts_map_citations_to_store_ids():
    docs = _docs(3)
    res = {"answer": "supported", "cited": [2], "p": {"supported": 0.7, "refuted": 0.2,
                                                     "not enough evidence": 0.1}}
    status, verdict, p, ids = tc.qwen_verdict(res, "ok", docs)
    assert (verdict, ids) == ("supported", [docs[1].doc_id])
    assert p == {"supported": 0.7, "refuted": 0.2, "not_enough_evidence": 0.1}
    nei = {**res, "answer": "not enough evidence", "cited": []}
    assert tc.qwen_verdict(nei, "ok", docs)[3] == [d.doc_id for d in docs]
    assert tc.qwen_verdict(res, "off_format", docs)[:3] == ("off_format", None, None)
    assert tc.qwen_verdict(None, None, ())[:2] == ("no_documents", "not_enough_evidence")


# --------------------------------------------------------------------------
# the server's runner, with stub models: sealed input, resumable output, recheck
# --------------------------------------------------------------------------


def _seal(tmp_path, task, rows_):
    path = tmp_path / f"x_pilot_{task}_in.jsonl"
    cs.write_prompts(path, task, rows_, {"task": task}, {r["key"]: "k" for r in rows_})
    return path


def test_the_runner_judges_a_sealed_file_and_rechecks_it(tmp_path):
    shots = tc.fewshot()["judge"]
    rows_ = [{"key": f"u{i}", "task": "tc_judge", "n_docs": 3,
              "messages": tc.judge_messages(tc.ClaimView(f"{CLAIM} {i}", _docs()), shots),
              "content": "c"} for i in range(4)]
    path = _seal(tmp_path, "tc_judge", rows_)
    assert tc.main(["run", "--input", str(path), "--stub"]) == 0
    out = tc.read_run(path)
    assert set(out) == {"u0", "u1", "u2", "u3"}
    assert all(r["status"] in tc.STATUSES for r in out.values())
    assert tc.main(["run", "--input", str(path), "--stub", "--recheck", "2"]) in (0, 1)


def test_the_runner_refuses_an_input_with_an_id_shape(tmp_path):
    shots = tc.fewshot()["judge"]
    msgs = tc.judge_messages(tc.ClaimView(f"{CLAIM} data/raw/m4fc/x.png", _docs()), shots)
    path = _seal(tmp_path, "tc_judge", [{"key": "u0", "task": "tc_judge", "n_docs": 3,
                                         "messages": msgs, "content": CLAIM}])
    with pytest.raises(cs.GuardError):
        tc.main(["run", "--input", str(path), "--stub"])


def _seal_nli(tmp_path, n_docs):
    pairs = tc.nli_pairs(tc.ClaimView(CLAIM, _docs(n_docs)))
    path = tmp_path / "x_pilot_nli_in.jsonl"
    cs.write_prompts(path, "tc_nli", [{"key": "u0", "task": "tc_nli", "pairs": pairs,
                                       "content": tc.pairs_text(pairs)}], {"task": "tc_nli"}, {})
    return path


def test_nli_runs_write_one_output_per_model(tmp_path):
    path = _seal_nli(tmp_path, 2)
    for which in ("nli_b", "nli_a"):
        assert tc.main(["run", "--input", str(path), "--stub", "--model", which]) == 0
    assert len(tc.read_run(path, "nli_b")["u0"]["result"]["probs"]) == 2
    assert tc.out_path_for(path, "nli_a").is_file() and tc.out_path_for(path, "nli_b").is_file()


def test_an_nli_run_that_is_not_model_b_at_its_pin_is_refused(tmp_path):
    path = _seal_nli(tmp_path, 1)
    tc.main(["run", "--input", str(path), "--stub", "--model", "nli_b"])
    with pytest.raises(cs.GuardError):  # a stub is no model at its pin
        tc.nli_run(path, "nli_b")


# --------------------------------------------------------------------------
# exemplars: invented, so disjoint from every corpus
# --------------------------------------------------------------------------

INVENTED = ("velmora", "dunholt", "karst bay", "olsterra", "pellow", "torvale")
CORPORA = ("averitec", "averimatec", "m4fc", "mocheg", "liar", "verite", "factify2")


def _shingles(text, n=6):
    words = " ".join(text.casefold().split()).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


@needs_processed
def test_no_exemplar_comes_from_any_corpus():
    from scripts.processed_loader import iter_rows

    doc = tc.fewshot()
    shots = doc["judge"] + doc["nodocs"]
    texts = [s["claim"] for s in shots]
    texts += [f"{d['title']} {d['excerpt']}" for s in doc["judge"] for d in s["documents"]]
    claims = {" ".join(s["claim"].casefold().split()) for s in shots}
    shingles = set().union(*(_shingles(t) for t in texts))
    for corpus in CORPORA:
        for row in iter_rows(corpus):
            text = " ".join((row.get("text") or "").casefold().split())
            assert text not in claims, corpus
            assert not (_shingles(text) & shingles), corpus
            padded = f" {text} "
            assert not any(f" {name}" in padded for name in INVENTED), corpus


# --------------------------------------------------------------------------
# the threshold sweep and the relevance filter
# --------------------------------------------------------------------------


def test_the_threshold_sweep_equals_a_brute_force_search():
    import random

    rng = random.Random(11)
    for _ in range(200):
        items = [(rng.choice(tc.VERDICT_LABELS), None if rng.random() < 0.1 else round(rng.random(), 2),
                  rng.choice(["supported", "refuted"])) for _ in range(rng.randint(1, 30))]
        best = None
        for tau in sorted({0.0} | {v for _, v, _ in items if v is not None}):
            pred = ["not_enough_evidence" if v is None or v < tau else s for _, v, s in items]
            f = tc.macro_f1([g for g, _, _ in items], pred)
            if best is None or f > best[1] + 1e-12:
                best = (tau, f)
        got = tc._sweep(items)
        assert got[0] == best[0] and got[1] == pytest.approx(best[1])


def test_a_document_counts_by_a_shared_entity_or_the_floor():
    docs = (cs.Document("a" * 64, "Floods hit Lagos", "2020-01-01", "x", "…water in the streets…"),
            cs.Document("b" * 64, "Market report", "2020-01-02", "x", "…shares rose in Lagosville…"),
            cs.Document("c" * 64, "Weather", "2020-01-03", "x", "…rain…"))
    assert tc.relevance(docs, [-5.0, -5.0, -30.0], ["lagos"], -20.0) == [True, False, True]
    assert tc.relevance(docs, [-5.0, -5.0, -30.0], [], float("-inf")) == [False, False, False]
    assert tc.relevance(docs, [-5.0, -5.0, -30.0], [], float("inf")) == [True, True, True]


def test_a_unit_whose_documents_all_fail_the_filter_has_none_usable():
    docs = _docs(2)
    probs = [_pr(0.01, 0.01, 0.98), _pr(0.02, 0.03, 0.95)]
    assert tc.nli_verdict(probs, docs, 0.5, [False, False])[:3] == ("no_documents", "not_enough_evidence", None)
    status, verdict, p, ids = tc.nli_verdict(probs, docs, 0.5, [False, True])
    assert verdict == "refuted" and ids == [docs[1].doc_id]


def test_the_filter_is_fitted_on_what_it_is_given_and_cv_never_scores_its_own_fit():
    docs = _docs(2)
    units_ = []
    for i in range(20):
        unrelated = _pr(0.01, 0.01, 0.98)  # an unrelated page: contradiction, as SNLI's model reads it
        related = _pr(0.9, 0.05, 0.05) if i % 2 else _pr(0.05, 0.9, 0.05)
        units_.append({"gold": "supported" if i % 2 else "not_enough_evidence", "probs": [related, unrelated],
                       "docs": docs, "scores": [-30.0, -2.0], "terms": []})
    fit = tc.fit_filter(units_)
    assert fit["train_macro_f1"] == 1.0 and fit["floor"] not in (None, "entities only")
    assert -30.0 <= float(fit["floor"]) < -2.0
    plain = tc.fit_variant("unfiltered", units_)
    assert plain["train_macro_f1"] < fit["train_macro_f1"]
    assert tc.cross_validated(units_, "filtered", folds=5) == [u["gold"] for u in units_]


def test_a_claim_both_corpora_carry_keeps_verites_no_bound_row(monkeypatch):
    shared, only_m4fc = ck("A caption both corpora carry."), ck(CLAIM)
    m4 = [tc.verdict_row(k, j, "ok", "refuted", P, [DOC]) for k in (shared, only_m4fc) for j in tc.ENSEMBLE_JUDGES]
    monkeypatch.setattr(tc, "unit_verdicts", lambda ds: {"taus": {"nli_b": {"tau": 0.1}}, "configs": {"qwen": "q", "nli_b": "n"}})
    monkeypatch.setattr(tc, "m4fc_rows", lambda uv: (m4, {"claims": 2}))
    monkeypatch.setattr(tc, "verite_rows", lambda: [tc.verdict_row(shared, j, "no_bound") for j in tc.ENSEMBLE_JUDGES])
    written = {}
    monkeypatch.setattr(tc, "write_verdicts", lambda rows_, about, path=None: written.update(rows=rows_, about=about) or
                        {"rows": len(rows_), "sha256": "0" * 64})
    tc.main(["verdicts"])
    by = {(r["claim_key"], r["judge"]): r["status"] for r in written["rows"]}
    assert by[(shared, "qwen")] == "no_bound" and by[(only_m4fc, "qwen")] == "ok"
    assert tc.file_problems(written["rows"]) == []
    assert written["about"]["m4fc"]["claims_shared_with_verite_left_no_bound"] == 1
