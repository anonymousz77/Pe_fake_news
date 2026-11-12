"""The consistency block: what may be read, what may reach a model, and the rebuild check.

Offline. A canary row -- its slug in the record id, the image path, meta and
the image_text row's record_ids -- goes through every builder of a model input,
and must come out in no form. Data-backed twins run the same checks over every
M4FC train+val row on disk, and skip without the processed layer.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.paths import PROCESSED  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts.image_text import CATEGORIES  # noqa: E402
from scripts.m4fc_audit import SHEET_COLUMNS  # noqa: E402
from scripts.m4fc_audit import _csv_text as _csv  # noqa: E402

HAVE_PROCESSED = (PROCESSED / "records" / "m4fc.jsonl.gz").is_file()
needs_processed = pytest.mark.skipif(not HAVE_PROCESSED, reason="needs the processed layer")
needs_gpu = pytest.mark.skipif(os.environ.get("PE_FAKE_NEWS_GPU_TESTS") != "1",
                               reason="loads Qwen2.5-VL; set PE_FAKE_NEWS_GPU_TESTS=1")

#: A slug that states a finding, as M4FC's do, and that no content will ever hold.
SLUG = "qzx-canary-photo-falsely-shared-as-flood-vkj"
#: A VERITE-style file name, in the shape its label gives it. Built here, so
#: that this file names no VERITE file by its label (scripts/prepush_check.py).
LABEL_SHAPED = "_".join(("true", "12"))
SHA = hashlib.sha256(b"canary image").hexdigest()


def canary_row(**over) -> dict:
    row = {"dataset": "m4fc", "record_id": f"images/{SLUG}.png",
           "text": "A crowd gathers outside a stadium after the match.",
           "image_paths": [f"data/raw/m4fc/media/images/images_{SLUG}_png.png"],
           "label": "false", "split": "train", "usable": True, "reason_code": "content_refuted",
           "meta": {"fc_org": "canary.example", "fc_pub_date": "2021-01-02",
                    "official_split": "train", "verdict": "false (out-of-context)",
                    "article_url": f"https://canary.example/{SLUG}/"}}
    row.update(over)
    return row


def image_row(sha: str = SHA, record_ids=(f"images/{SLUG}.png",)) -> dict:
    def cat(items, forced=False):
        return {"status": "listed" if items else "none", "unidentified": False, "forced": forced,
                "items": [{"name": n, "basis": "seen", "confidence": 0.9, "in_ocr": False,
                           "forced": forced} for n in items]}

    ents = {c: cat([]) for c in CATEGORIES}
    ents.update(PERSON=cat(["A. Person"]), GPE=cat(["Lagos"], forced=True))
    ents.update(raw="PERSON: A. Person", error=None, flags=[], finish="stop", n_tokens=20,
                confidence=0.8)
    return {"dataset": "m4fc", "sha256": sha, "record_ids": list(record_ids), "n_paths": 1,
            "status": "ok", "reason": None,
            "ocr": {"text": "FINAL SCORE 2-1", "regions": [], "mean_confidence": 0.9,
                    "input": None, "error": None},
            "description": {"text": "A crowd in front of a stadium at night.", "error": None,
                            "flags": [], "finish": "stop", "n_tokens": 12, "confidence": 0.7},
            "entities": ents}


def slug_forms() -> list[str]:
    """The canary in every form it could leak in."""
    return [SLUG, SLUG.replace("-", " "), f"images/{SLUG}.png", f"images_{SLUG}_png",
            "canary.example"]


# --------------------------------------------------------------------------
# record ids never reach a model input
# --------------------------------------------------------------------------


def test_the_view_has_whitelisted_fields_only():
    """A field added to the view is a decision, taken here, never a side effect."""
    assert [f.name for f in fields(cs.ModelView)] == [
        "image_sha256", "claim", "ocr", "description", "entities", "people", "documents"]
    assert [f.name for f in fields(cs.Document)] == [
        "doc_id", "title", "published_at", "host", "excerpt"]
    for cls in (cs.ModelView, cs.Entity, cs.Person, cs.Document):
        names = {f.name for f in fields(cls)}
        assert not names & {"record_id", "record_ids", "path", "image_path", "image_paths", "url",
                            "meta", "label", "split"}


def test_a_canary_slug_reaches_no_view():
    row = canary_row()
    view = cs.view_of(row, sha_of={row["image_paths"][0]: SHA}.get, image_rows={SHA: image_row()},
                      people=[cs.Person("A. Person", "Q1", 0.61)])
    body = cs.view_json(view).casefold()
    for form in slug_forms():
        assert form.casefold() not in body
    # what it does carry is the content, and the join key beside it
    assert view.claim == row["text"] and view.image_sha256 == SHA
    assert view.ocr == "FINAL SCORE 2-1" and view.description.startswith("A crowd")
    assert {(e.category, e.name, e.forced) for e in view.entities} == {
        ("PERSON", "A. Person", False), ("GPE", "Lagos", True)}
    assert cs.ids_in(body, cs.id_forms(row), explained=view.content()) == []


def test_a_canary_slug_reaches_no_retrieval_query():
    row = canary_row()
    [item] = cs.plan_retrieval("m4fc", [row], {row["image_paths"][0]: SHA}.get)
    assert item == {"sha256": SHA, "query": row["text"], "before": "2021-01-02",
                    "bound": "fact_check_date"}
    body = json.dumps(item).casefold()
    for form in slug_forms():
        assert form.casefold() not in body


def test_ids_in_finds_every_form_and_shape_the_content_does_not_explain():
    row = canary_row()
    forms = cs.id_forms(row)
    for form in slug_forms()[:4]:
        assert form.casefold() in forms or form.replace("_", " ").casefold() in forms
        assert cs.ids_in(f"claim: x. evidence: {form}", forms)
    # shapes are caught with no row to hand: on the server, in a report
    for leaked in (f"see {LABEL_SHAPED} here", "data/raw/verite/x.jpg", "file images/some-slug.png"):
        assert cs.ids_in(leaked)
    # a claim that says what its own slug says is content, not a leak
    spaced = SLUG.replace("-", " ")
    assert cs.ids_in(f"Claim: {spaced}", forms, explained=spaced) == []
    with pytest.raises(cs.GuardError):
        cs.refuse_ids(f"Claim: x\nSource: {SLUG}", forms, explained="x")


def test_a_failed_component_is_missing_not_empty_and_a_wrong_join_is_refused():
    bad = image_row()
    bad["ocr"] = {"text": None, "regions": None, "mean_confidence": None, "input": None,
                  "error": "ocr_failed"}
    bad["description"]["error"] = "truncated"
    ocr, desc, ents = cs.image_part(bad, SHA)
    assert ocr is None and desc is None and ents
    with pytest.raises(cs.GuardError):
        cs.image_part(image_row(), "f" * 64)


@needs_processed
def test_no_m4fc_row_on_disk_leaks_its_id_into_its_view():
    """The data-backed twin: every M4FC train+val row, with whatever image_text
    rows this machine holds."""
    from scripts import images
    from scripts.image_text import OUT_ROOT, scan_ledger

    index = images.load_index("m4fc")
    ledger = OUT_ROOT / cs.IMAGE_TEXT_CONFIG / "m4fc.jsonl"
    image_rows = {}
    if ledger.is_file():
        _, kept, _ = scan_ledger(ledger, keep=lambda r: True)
        image_rows = {r["sha256"]: r for r in kept}
    n = 0
    for row in cs.rows("m4fc", cs.M4FC_SPLITS):
        view = cs.view_of(row, sha_of=lambda p: (index.get(p) or {}).get("sha256"),
                          image_rows=image_rows)
        assert cs.ids_in(cs.view_json(view), cs.id_forms(row), explained=view.content()) == [], \
            f"row {n}"
        n += 1
    assert n == 3021


# --------------------------------------------------------------------------
# what may be read
# --------------------------------------------------------------------------


def test_m4fc_test_is_refused_however_it_is_asked_for(monkeypatch):
    with pytest.raises(cs.GuardError):
        list(cs.rows("m4fc", ("train", "test")))
    with pytest.raises(cs.GuardError):
        list(cs.rows("verite", ("test",)))
    from scripts import processed_loader

    sneaked = canary_row(meta={"official_split": "test"})
    monkeypatch.setattr(processed_loader, "iter_rows", lambda ds, **kw: iter([sneaked]))
    with pytest.raises(cs.GuardError):
        list(cs.rows("m4fc", ("train",)))
    monkeypatch.setattr(processed_loader, "iter_rows",
                        lambda ds, **kw: iter([canary_row(), canary_row(split="test"),
                                               canary_row(split="val", usable=False)]))
    assert [r["split"] for r in cs.rows("m4fc", ("train", "val"))] == ["train"]
    assert [r["split"] for r in cs.rows("m4fc", ("train", "val"), usable=False)] == ["train", "val"]


def test_verite_evaluation_drops_the_shared_records_and_asserts_it(monkeypatch):
    from scripts import placeholders, processed_loader

    vrows = [{"record_id": str(i), "usable": i != 3} for i in range(6)]
    monkeypatch.setattr(processed_loader, "iter_rows", lambda ds, **kw: iter(vrows))
    shared = {"factify2": {"1"}, "m4fc": {"2"}, "mocheg": set()}
    monkeypatch.setattr(placeholders, "excluded_keys",
                        lambda ds, *, against: frozenset(shared[against]))
    asked = []
    real = placeholders.assert_no_cross_corpus_leak
    monkeypatch.setattr(placeholders, "assert_no_cross_corpus_leak",
                        lambda ds, keys, *, against: (asked.append(against),
                                                      real(ds, keys, against=against)))
    assert [r["record_id"] for r in cs.evaluation_rows()] == ["0", "4", "5"]
    assert asked == list(cs.VERITE_AGAINST)


@needs_processed
def test_verite_evaluation_is_877_rows():
    out = cs.evaluation_rows()
    assert len(out) == 877
    assert Counter(r["label"] for r in out) == {"out-of-context": 299, "true": 289,
                                                "miscaptioned": 289}


# --------------------------------------------------------------------------
# the rebuild: nothing moves but M4FC's new meta keys
# --------------------------------------------------------------------------


def test_meta_only_passes_an_addition_and_counts_everything_else():
    raw = {"a": {"verdict": "true", "date": "2020"}, "b": {"verdict": "false", "date": "1999"}}
    before = [{"record_id": "a", "split": "train", "meta": {"fc_org": "x"}},
              {"record_id": "b", "split": "val", "meta": {"fc_org": "y"}}]
    good = [{"record_id": "a", "split": "train", "meta": {"fc_org": "x", "verdict": "true", "date": "2020"}},
            {"record_id": "b", "split": "val", "meta": {"fc_org": "y", "verdict": "false", "date": "1999"}}]
    assert cs.meta_only(before, good, raw, ("verdict", "date"))["problems"] == {}
    moved = [dict(good[0], split="test"), good[1]]
    assert "field split changed" in cs.meta_only(before, moved, raw, ("verdict", "date"))["problems"]
    edited = [good[0], dict(good[1], meta={"fc_org": "z", "verdict": "false", "date": "1999"})]
    assert cs.meta_only(before, edited, raw, ("verdict", "date"))["problems"] == {
        "an earlier meta value changed or went": 1}
    not_raw = [good[0], dict(good[1], meta={"fc_org": "y", "verdict": "false", "date": "2000"})]
    assert cs.meta_only(before, not_raw, raw, ("verdict", "date"))["problems"] == {
        "a new meta value is not M4FC.json's": 1}
    assert cs.meta_only(before, good[::-1], raw, ("verdict", "date"))["problems"][
        "the record ids, or their order, differ"] == 1


def test_gz_digest_reads_the_content_not_the_header(tmp_path):
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    for path, mtime in ((a, 1), (b, 2)):
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as h:
            h.write(b'{"x": 1}\n{"x": 2}\n')
    assert a.read_bytes() != b.read_bytes()
    assert cs.gz_digest(a) == cs.gz_digest(b) == {
        "sha256": hashlib.sha256(b'{"x": 1}\n{"x": 2}\n').hexdigest(), "rows": 2}


def test_the_processed_stage_refuses_a_partial_build_only(monkeypatch):
    from scripts import build_processed

    monkeypatch.setattr(build_processed, "build", lambda *a, **k: pytest.fail("a build ran"))
    with pytest.raises(SystemExit):
        build_processed.main(["processed", "--dataset", "m4fc"])
    with pytest.raises(SystemExit):
        build_processed.main(["processed", "--dataset", "m4fc", "--dataset", "factify2"])
    wrote = []
    monkeypatch.setattr(build_processed, "write_interim", lambda name: wrote.append(name) or 0)
    assert build_processed.main(["interim", "--dataset", "m4fc"]) == 0
    assert wrote == ["m4fc"]


# --------------------------------------------------------------------------
# statistics and rules
# --------------------------------------------------------------------------


def test_wilson_matches_known_values():
    for (k, n), (lo, hi) in {(5, 10): (0.2366, 0.7634), (8, 10): (0.4902, 0.9433),
                             (0, 10): (0.0, 0.2775)}.items():
        got = cs.wilson(k, n)
        assert abs(got[0] - lo) < 1e-4 and abs(got[1] - hi) < 1e-4
    assert cs.wilson(0, 0) == (None, None)
    assert cs.rate(3, 4) == {"k": 3, "n": 4, "rate": 0.75, "wilson95": [0.3006, 0.9544]}


def test_a_date_part_covers_its_own_granularity():
    from datetime import date as d

    assert cs.date_interval("2019") == (d(2019, 1, 1), d(2019, 12, 31))
    assert cs.date_interval("2020-02") == (d(2020, 2, 1), d(2020, 2, 29))
    assert cs.date_interval("2019-03-15") == (d(2019, 3, 15), d(2019, 3, 15))
    assert cs.date_interval("1964-1972") == (d(1964, 1, 1), d(1972, 12, 31))
    assert cs.date_interval("1990s") == (d(1990, 1, 1), d(1999, 12, 31))
    assert cs.date_interval("222 AD") == (d(222, 1, 1), d(222, 12, 31))
    for not_a_date in ("current", "2009-12/201", "2019-13", "2019-02-30", "1972-1964", "March"):
        assert cs.date_interval(not_a_date) is None, not_a_date


def test_the_date_rule_is_overlap_at_granularity_and_symmetric():
    cases = [("2019", "2019-03-15", "consistent"), ("2015-06; 2016", "2016-08", "consistent"),
             ("2015", "2016", "contradicts"), ("2019-03", "2019-04-01", "contradicts"),
             ("1964-1972", "1970", "consistent"), ("current", "2009", "unclear"),
             ("current; 2010", "2009", "unclear"), ("current; 2009", "2009-05", "consistent"),
             ("not enough information", "2019", "unclear"), ("2019", "", "unclear")]
    for claimed, actual, want in cases:
        assert cs.date_rule(claimed, actual)["label"] == want, (claimed, actual)
        assert cs.date_rule(actual, claimed)["label"] == want, (actual, claimed)
    assert cs.date_rule("2015", "2016")["score"] == 1.0
    assert cs.date_rule("2015", "2015")["score"] == 0.0
    assert cs.date_rule("current", "2015")["score"] is None


def test_the_people_qid_rule_never_contradicts_on_a_name_it_cannot_resolve():
    book = {"ann": ["Q1"], "bob": ["Q2"], "cat": ["Q3", "Q4"]}
    resolve = lambda name: book.get(name.casefold(), [])  # noqa: E731
    assert cs.people_qid_rule("Ann", "Ann; Bob", resolve)["label"] == "consistent"
    assert cs.people_qid_rule("Ann", "Bob", resolve)["label"] == "contradicts"
    assert cs.people_qid_rule("Ann", "Cat", resolve)["why"] == "ambiguous_name"
    assert cs.people_qid_rule("Ann", "Dan", resolve)["why"] == "name_not_in_gallery"
    assert cs.people_qid_rule("Ann; Dan", "Ann", resolve)["label"] == "consistent"
    assert cs.people_qid_rule("unspecified", "Ann", resolve)["label"] == "unclear"


# --------------------------------------------------------------------------
# stage A: gold, exemplars, prompts
# --------------------------------------------------------------------------


def _sheet(tmp_path, monkeypatch, rows, assistant):
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["sheet_id", "image", "claim", "item", "claimed", "actual",
                                        "mismatch", "note"], lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")
    marks = tmp_path / "assistant_rows.json"
    marks.write_text(json.dumps({"sheet_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
                                 "sheet_ids": assistant}), encoding="utf-8")
    monkeypatch.setattr(cs, "SHEET_REVIEWED", sheet)
    monkeypatch.setattr(cs, "ASSISTANT_ROWS", marks)
    return sheet, marks


def test_the_sheet_gold_drops_assistant_rows_and_things_and_maps_the_answers(tmp_path, monkeypatch):
    base = {"image": "x.jpg", "claim": "c", "claimed": "a", "actual": "b", "note": ""}
    rows = [dict(base, sheet_id="i-001-date", item="date", mismatch="yes"),
            dict(base, sheet_id="i-001-things", item="things", mismatch="yes"),
            dict(base, sheet_id="i-002-event", item="event", mismatch="no"),
            dict(base, sheet_id="i-002-people", item="people", mismatch="unclear"),
            dict(base, sheet_id="i-003-location", item="location", mismatch="Yes")]
    sheet, marks = _sheet(tmp_path, monkeypatch, rows, ["i-003-location"])
    got = {g["gold_id"]: g["label"] for g in cs.sheet_gold()}
    assert got == {"i-001-date": "contradicts", "i-002-event": "consistent",
                   "i-002-people": "unclear"}
    sheet.write_bytes(sheet.read_bytes() + b"\r\n")  # the sheet changed after the marks
    with pytest.raises(SystemExit):
        cs.sheet_gold()


def test_the_exemplars_are_balanced_and_never_a_gold_pair():
    shots = cs.fewshot()["judge"]
    assert len(shots) == 16
    assert Counter(s["item"] for s in shots) == {i: 4 for i in cs.ITEMS}
    assert Counter(s["answer"] for s in shots) == {"no": 6, "yes": 6, "unclear": 4}
    assert cs.shot_overlap(shots, [(shots[3]["actual"], shots[3]["claimed"])]) == [3]


@needs_processed
def test_no_exemplar_value_comes_from_the_sheet_m4fc_or_verite():
    """Pairs never match a gold or M4FC pair; a value of two words or more is
    never an M4FC value of its item; one of three words or more is in no
    VERITE caption. Years and single words are too common to mean a source."""
    from scripts.m4fc_audit import parts

    shots = cs.fewshot()["judge"]
    pairs, values = set(), defaultdict(set)
    for r in cs.rows("m4fc", cs.M4FC_SPLITS, usable=False):
        for item in cs.ITEMS:
            c, a = str(r["meta"].get(f"claimed_{item}")), str(r["meta"].get(item))
            pairs.add((c.casefold(), a.casefold()))
            values[item] |= {p.casefold() for p in parts(c) + parts(a)}
    if cs.SHEET_REVIEWED.is_file():
        pairs |= {(g["claimed"].casefold(), g["actual"].casefold()) for g in cs.sheet_gold()}
    captions = " ".join((r["text"] or "").casefold() for r in cs.evaluation_rows())
    assert cs.shot_overlap(shots, pairs) == []
    for s in shots:
        for v in (s["claimed"], s["actual"]):
            if len(v.split()) >= 2:
                assert v.casefold() not in values[s["item"]], v
            if len(v.split()) >= 3:
                assert v.casefold() not in captions, v
    # the predictor's worked example: no name or phrase of it from a corpus
    view, answer = cs.predict_shot()
    every = set().union(*values.values())
    texts = [e.name for e in view.entities] + [d.title for d in view.documents]
    texts += [v["value"] for v in answer.values() if v["value"]]
    for t in texts:
        if len(t.split()) >= 2:
            assert t.casefold() not in every, t
        assert t.casefold() not in captions, t
    # the extractor's exemplars: no claim, value or phrase of them from a corpus
    claims = {(r["text"] or "").casefold() for r in cs.rows("m4fc", cs.M4FC_SPLITS, usable=False)}
    for s in cs.fewshot()["extract"]:
        assert s["claim"].casefold() not in claims and s["claim"].casefold() not in captions, s["claim"]
        for v in s["answer"].values():
            if v and len(v.split()) >= 2:
                assert v.casefold() not in every and v.casefold() not in captions, v


def test_the_predictor_s_worked_example_answers_its_own_evidence_in_form():
    view, _answer = cs.predict_shot()
    msgs, evidence = cs.predict_messages(view)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    status, p = cs.parse_prediction(msgs[2]["content"][0]["text"], evidence, "eos")
    assert status == "ok" and p["flags"] == [] and p["items"]["date"]["value"] is None
    assert all(v["evidence"] for v in p["items"].values() if v["value"] is not None)
    # it is a prior turn: the image's own evidence ids come only from the image's view
    _row, canary = _canary_view()
    assert set(cs.predict_messages(canary)[1]) == {"DESC", "OCR", "E1", "E2", "P1", "D1"}


def test_a_judge_prompt_holds_the_shots_then_the_pair_and_no_canary():
    shots = cs.fewshot()["judge"]
    msgs = cs.judge_messages("location", "Lagos", "Accra", shots)
    assert msgs[0]["role"] == "system" and len(msgs) == 2 + 2 * len(shots)
    assert [m["content"][0]["text"] for m in msgs[2:2 * len(shots) + 1:2]] == [
        s["answer"] for s in shots]
    last = msgs[-1]["content"][0]["text"]
    assert "Statement A, from its caption: Lagos" in last and "the place of" in last
    assert "can't be true at once" in last
    text = cs.messages_text(msgs)
    assert cs.ids_in(text, cs.id_forms(canary_row()), explained="Lagos\nAccra") == []
    with pytest.raises(cs.GuardError):
        bad = cs.judge_messages("location", "Lagos", f"Accra ({SLUG})", shots)
        cs.refuse_ids(cs.messages_text(bad), cs.id_forms(canary_row()), explained="Lagos\nAccra")


# --------------------------------------------------------------------------
# the runner, with the stand-in model
# --------------------------------------------------------------------------


def _inputs(tmp_path, n=5, content=None):
    shots = cs.fewshot()["judge"]
    rows = [{"key": f"k{i}", "task": "judge", "content": f"v{i}\nw{i}",
             "messages": cs.judge_messages("event", f"v{i}", f"w{i}", shots)} for i in range(n)]
    if content:
        rows[0]["messages"][-1]["content"][0]["text"] += content
    path = tmp_path / "judge_in.jsonl"
    cs.write_prompts(path, "judge", rows, cs.judge_prompt_config(shots), {})
    return path


def test_the_runner_resumes_rechecks_and_refuses_what_it_should(tmp_path, capsys):
    path = _inputs(tmp_path)
    args = ["llm", "--task", "judge", "--input", str(path), "--stub"]
    assert cs.main(args) == 0
    out = tmp_path / "judge_out.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["key"] for r in rows) == [f"k{i}" for i in range(5)]
    for r in rows:
        res = r["result"]
        assert abs(sum(res["p"].values()) - 1) < 1e-5
        if r["status"] == "ok":
            assert res["label"] == cs.JUDGE_ANSWERS[res["answer"]] and res["score"] == res["p"]["no"]
        else:
            assert r["status"] == "off_format" and res["label"] == "unclear" and res["score"] is None
    assert "  [judge] 5/5" in capsys.readouterr().out  # the form the server's progress grep counts
    # a torn last line is cut, and only its row is redone -- to the same result; its timing is a
    # measurement, which a loaded machine can move by a millisecond (seen 2026-11-08)
    body = out.read_bytes()
    out.write_bytes(body[:-7])
    assert cs.main(args) == 0

    def results(raw: bytes) -> list[dict]:
        return [{k: v for k, v in json.loads(line).items() if k != "seconds"} for line in raw.splitlines()]

    assert results(out.read_bytes()) == results(body) and out.read_bytes().count(b"\n") == body.count(b"\n")
    assert cs.main(args + ["--recheck", "3"]) == 0
    # another configuration never writes into this output
    cfg = tmp_path / "judge_out.config.json"
    c = json.loads(cfg.read_text(encoding="utf-8"))
    cfg.write_text(json.dumps(dict(c, config_id="000000000000")), encoding="utf-8")
    with pytest.raises(SystemExit):
        cs.main(args)


def test_the_runner_refuses_an_edited_or_id_bearing_input(tmp_path):
    path = _inputs(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"v1", b"v9"))
    with pytest.raises(SystemExit):
        cs.main(["llm", "--task", "judge", "--input", str(path), "--stub"])
    leaky = _inputs(tmp_path, content=f" see data/raw/verite/data/VERITE/images/{LABEL_SHAPED}.jpg")
    with pytest.raises(cs.GuardError):
        cs.main(["llm", "--task", "judge", "--input", str(leaky), "--stub"])


# --------------------------------------------------------------------------
# stage A: metrics and the gate
# --------------------------------------------------------------------------


def _gold(item, labels, source):
    return [{"gold_id": f"{source}-{item}-{i}", "source": source, "item": item, "claimed": "a",
             "actual": "b", "label": lab} for i, lab in enumerate(labels)]


def _preds(gold, answers):
    return {g["gold_id"]: {"x": cs.outcome(a, "t", None, "t")} for g, a in zip(gold, answers)}


def test_a_timid_comparator_fails_the_gate_whatever_its_precision():
    sheet = _gold("event", ["contradicts"] * 30 + ["consistent"] * 5, "sheet")
    true_ = _gold("event", ["consistent"] * 50, "true_row")
    # says contradicts on 3 of 30 real contradictions, abstains on the rest
    answers = ["contradicts"] * 3 + ["unclear"] * 27 + ["consistent"] * 5 + ["consistent"] * 50
    m = cs.a_metrics(sheet + true_, _preds(sheet + true_, answers), "x")
    assert m["sheet"]["contradicts_precision"]["rate"] == 1.0
    assert m["sheet"]["contradicts_recall"]["rate"] == 0.1
    gate = cs.gate_a("event", m)
    assert gate["passed"] is False and gate["checks"]["recall"] is False
    assert gate["checks"]["precision"] is True
    # the same comparator, saying what it abstained on, passes
    bold = ["contradicts"] * 30 + ["consistent"] * 55
    assert cs.gate_a("event", cs.a_metrics(sheet + true_, _preds(sheet + true_, bold), "x"))["passed"]


@needs_processed
def test_stage_a_scoring_runs_end_to_end_on_synthetic_gold(tmp_path, monkeypatch, capsys):
    """The glue: gold, the judge's output through its neutral keys, the date
    rule, the report -- on synthetic pairs, never the person's sheet."""
    gold = (_gold("date", ["contradicts", "contradicts"], "sheet")
            + _gold("date", ["consistent"], "true_row")
            + _gold("event", ["contradicts", "consistent"], "sheet")
            + _gold("event", ["consistent", "consistent"], "true_row"))
    for g, (c, a) in zip(gold, [("2015", "2016"), ("current", "2019"), ("2019", "2019-05"),
                                ("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]):
        g.update(claimed=c, actual=a)
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    (a_dir / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold), encoding="utf-8")
    monkeypatch.setattr(cs, "A_DIR", a_dir)
    monkeypatch.setattr(cs, "GOLD", a_dir / "gold.jsonl")
    monkeypatch.setattr(cs, "REPORT_A", tmp_path / "consistency_a.json")
    monkeypatch.setattr(cs, "REPORT_A_MD", tmp_path / "consistency_a.md")
    monkeypatch.setattr(cs, "resolver", lambda: None)
    pages = []
    monkeypatch.setattr(cs, "write_contested_review", lambda contested, gold: pages.append(contested))
    monkeypatch.setattr(cs, "CONTESTED_REVIEWED", tmp_path / "reviewed.csv")
    shots = cs.fewshot()["judge"]
    rows = [{"key": cs.neutral_key("a", g["gold_id"]), "task": "judge",
             "content": f"{g['claimed']}\n{g['actual']}",
             "messages": cs.judge_messages(g["item"], g["claimed"], g["actual"], shots)} for g in gold]
    cs.write_prompts(a_dir / "judge_in.jsonl", "judge", rows, cs.judge_prompt_config(shots),
                     {r["key"]: g["gold_id"] for r, g in zip(rows, gold)})
    assert cs.main(["llm", "--task", "judge", "--input", str(a_dir / "judge_in.jsonl"), "--stub"]) == 0
    assert cs.main(["score", "--stage", "a"]) == 0
    report = json.loads((tmp_path / "consistency_a.json").read_text(encoding="utf-8"))
    assert report["judge_rows"] == len(gold)
    date = report["items"]["date"]
    assert date["rule"]["sheet"]["contradicts_recall"] == {"k": 1, "n": 2, "rate": 0.5,
                                                          "wilson95": [0.0945, 0.9055]}
    assert set(date) == {"rule", "judge", "always_contradicts", "always_consistent"}
    assert set(report["gate"]) == {"date", "location", "event"}
    assert report["gate"]["location"] == {"passed": None, "why": "not run"}  # no pairs, no verdict
    assert isinstance(report["contested_true_rows"], int)  # the public report holds a count only
    assert report["review"] == {"contested": report["contested_true_rows"], "decided": 0,
                                "relabelled": {}}
    assert pages == ([] if not report["contested_true_rows"] else [pages[0]])
    out = capsys.readouterr().out
    assert "Gate (rule)" in out and "## Gate A, as built" in out and "Not yet reviewed" in out


def test_the_person_s_review_relabels_true_rows_and_the_gate_is_reported_both_ways(tmp_path):
    true_ = _gold("event", ["consistent"] * 20, "true_row")
    sheet = _gold("event", ["contradicts"] * 10, "sheet")
    gold = sheet + true_
    # the comparator calls 4 true rows contradictions: 4/20 fails the 0.10 bar
    preds = _preds(gold, ["contradicts"] * 10 + ["contradicts"] * 4 + ["consistent"] * 16)
    for g in gold:
        preds[g["gold_id"]] = {"judge": preds[g["gold_id"]]["x"]}
    assert cs.contested_true_rows(gold, preds) == sorted(g["gold_id"] for g in true_[:4])
    path = tmp_path / "reviewed.csv"

    def export(answers):
        rows = [dict(zip(SHEET_COLUMNS, (g["gold_id"], "x.png", "c", g["item"], g["claimed"],
                                         g["actual"], a, ""))) for g, a in zip(true_, answers)]
        path.write_text(_csv(rows), encoding="utf-8-sig", newline="")

    export(["yes", "yes", "unclear", ""])  # the person: two contradict, one can't tell, one unread
    decided = cs.true_row_decisions(gold, path)
    assert decided == {true_[0]["gold_id"]: "contradicts", true_[1]["gold_id"]: "contradicts",
                       true_[2]["gold_id"]: "unclear"}
    built = cs.score_a(gold, preds)
    after = cs.score_a([dict(g, label=decided.get(g["gold_id"], g["label"])) for g in gold], preds)
    fc = lambda s: s["items"]["event"]["judge"]["true_row"]["false_contradiction"]
    assert (fc(built)["k"], fc(built)["n"]) == (4, 20) and (fc(after)["k"], fc(after)["n"]) == (1, 17)
    export(["maybe", "", "", ""])
    with pytest.raises(SystemExit):
        cs.true_row_decisions(gold, path)
    rows = [dict(zip(SHEET_COLUMNS, (true_[0]["gold_id"], "x.png", "c", "event", "other", "pair",
                                     "yes", "")))]
    path.write_text(_csv(rows), encoding="utf-8-sig", newline="")
    with pytest.raises(SystemExit):  # a row whose pair is not the gold pair of its id
        cs.true_row_decisions(gold, path)


def test_the_contested_sheet_asks_the_sheet_s_question_and_shows_no_comparator():
    gold = [{"gold_id": "true-002-event", "source": "true_row", "item": "event", "claimed": "a",
             "actual": "b", "label": "consistent"},
            {"gold_id": "true-002-date", "source": "true_row", "item": "date", "claimed": "2019",
             "actual": "2018", "label": "consistent"},
            {"gold_id": "true-001-location", "source": "true_row", "item": "location",
             "claimed": "x", "actual": "y", "label": "consistent"}]
    private = {"true-001": {"record_id": "r1", "image_path": "data/raw/m4fc/a.png"},
               "true-002": {"record_id": "r2", "image_path": "data/raw/m4fc/b.png"}}
    rows, names = cs.contested_rows([g["gold_id"] for g in gold], gold, private,
                                    {"r1": "claim one", "r2": "claim two"}, lambda p: ".jpg")
    assert [r["sheet_id"] for r in rows] == ["true-001-location", "true-002-date", "true-002-event"]
    assert names == {"true-001": "true-001.jpg", "true-002": "true-002.jpg"}
    assert tuple(rows[0]) == SHEET_COLUMNS and rows[1]["claim"] == "claim two"
    assert all(r["mismatch"] == "" and r["note"] == "" for r in rows)
    assert rows[0]["image"].endswith("consistency/a/review/copies/true-001.jpg")
    assert cs.ids_in(json.dumps(rows)) == []  # a copy's path has no id shape


def _assert_dates_never_reach_the_judge():
    assert cs.ROUTES["date"] == ("rule",) and cs.FIXED["date"] == "rule"
    got = cs.date_verdict("the 2010s", "2014-02")  # the pair the judge got wrong on the server
    assert got["verdict"] == "unsupported" and got["comparator"] == "rule"
    assert got["why"] == "comparator_unclear:date_unparsed"
    assert cs.date_verdict("2014", "2014-02")["verdict"] == "consistent"
    assert cs.date_verdict("2019", "2014-02")["verdict"] == "contradicted"


def test_a_date_is_decided_by_the_rule_alone_and_its_unclear_is_unsupported():
    _assert_dates_never_reach_the_judge()


@needs_gpu
def test_the_judge_answers_in_form_and_is_deterministic():
    """On the server: an obvious location pair (from neither the sheet nor any
    corpus), scored twice in one process. Dates never reach the judge
    (2026-10-30, the person's decision): it called "the 2010s" against
    "2014-02" a contradiction."""
    _assert_dates_never_reach_the_judge()
    llm = cs.TextLLM()
    llm.prepare()
    llm.load()
    shots = cs.fewshot()["judge"]
    msgs = cs.judge_messages("location", "Lyon, France", "Osaka, Japan", shots)
    first = llm.label_probs(msgs, tuple(cs.JUDGE_ANSWERS))
    assert llm.label_probs(msgs, tuple(cs.JUDGE_ANSWERS)) == first
    probs, mass = first
    assert mass >= cs.MIN_LABEL_MASS and max(probs, key=probs.get) == "no", (probs, mass)


def test_strict_precision_counts_a_contradiction_the_person_marked_unclear():
    sheet = _gold("location", ["contradicts"] * 4 + ["unclear"], "sheet")
    m = cs.a_metrics(sheet, _preds(sheet, ["contradicts"] * 5), "x")["sheet"]
    assert m["contradicts_precision"]["rate"] == 0.8
    assert m["contradicts_precision_lenient"]["rate"] == 1.0


def test_the_date_gate_reads_recall_and_the_true_rows():
    sheet = _gold("date", ["contradicts"] * 16, "sheet")
    true_ = _gold("date", ["consistent"] * 57, "true_row")
    answers = ["contradicts"] * 13 + ["unclear"] * 3 + ["consistent"] * 52 + ["contradicts"] * 5
    gate = cs.gate_a("date", cs.a_metrics(sheet + true_, _preds(sheet + true_, answers), "x"))
    assert gate["checks"] == {"recall": True,  # 13/16 = 0.81
                              "false_contradiction": True,  # 5/57 = 0.088
                              "false_contradiction_upper": True}  # Wilson upper 0.189
    assert "precision" not in gate["checks"]
    # few true rows: the point estimate passes, its upper bound does not
    few = _gold("date", ["consistent"] * 20, "true_row")
    answers = ["contradicts"] * 16 + ["consistent"] * 18 + ["contradicts"] * 2
    gate = cs.gate_a("date", cs.a_metrics(sheet + few, _preds(sheet + few, answers), "x"))
    assert gate["checks"]["false_contradiction"] is True  # 2/20 = 0.10
    assert gate["checks"]["false_contradiction_upper"] is False  # Wilson upper 0.30
    assert gate["passed"] is False


# --------------------------------------------------------------------------
# stage B: faces, the gallery's answers, retrieval
# --------------------------------------------------------------------------


def test_stage_b_reads_m4fc_and_verite_only():
    with pytest.raises(cs.GuardError):
        cs.b_rows("factify2")


@needs_processed
def test_stage_b_reads_3021_m4fc_and_588_verite_images():
    assert len(cs.b_images("m4fc")) == 3021
    assert len(cs.b_images("verite")) == 588


def test_a_timing_sample_is_spread_and_a_near_bar_projection_goes_to_the_terminal():
    assert cs.spread(list(range(10)), 3) == [0, 3, 6]
    assert cs.spread([1, 2], 5) == [1, 2]
    assert cs.projection(60.0, 100, 1000)["where"] == "in session"  # 10 min
    assert cs.projection(60.0, 100, 1600)["where"] == "the person's terminal"  # 16: near 20
    assert cs.projection(0.0, 0, 50)["where"] is None


def test_a_changed_image_is_missing_faces_never_no_face(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(cs, "B_FACES", tmp_path / "faces")
    img = tmp_path / "x.png"
    Image.new("RGB", (64, 48), "white").save(img)
    good, changed = hashlib.sha256(img.read_bytes()).hexdigest(), "f" * 64
    face = {"box": [1, 1, 40, 40], "landmarks": [[10, 10]] * 5, "score": 0.9, "dhash": "0",
            "embedding": ""}

    class Backend:
        def faces(self, bgr, min_side):
            return [face]

    cs.embed_faces(Backend(), "cfg", [("m4fc", good, str(img)), ("m4fc", changed, str(img))])
    rows = cs.read_faces(cs.faces_path("cfg", "m4fc"))
    assert rows[good]["faces"] == [face] and rows[good]["error"] is None
    assert rows[changed]["faces"] == [] and rows[changed]["error"] == "ImageChanged"

    class Gallery:
        def resolve(self, name, kind):
            return ["Q1"]

        def verify(self, faces, name, threshold):
            assert faces, "an image whose faces are unknown never reaches verify"
            return {"entities": [{"qid": "Q1"}], "match": True, "score": 0.7, "reason": "matched",
                    "in_gallery": True, "faces_in_image": len(faces), "matched_entity": "Q1",
                    "query_face": None, "matched_reference": None}

    named = {good: {"A. Person": {"vlm_free"}}, changed: {"A. Person": {"claim"}}}
    out = {r["sha256"]: r for r in cs.verify_names("m4fc", named, rows, Gallery(), 0.52)}
    assert out[good]["match"] is True and out[good]["reason"] == "matched"
    assert out[changed]["reason"] == "faces_error" and out[changed]["match"] is None
    assert out[changed]["qids"] == ["Q1"]
    with pytest.raises(SystemExit):  # an image with no faces row at all
        cs.verify_names("m4fc", {"e" * 64: {"A. Person": {"claim"}}}, rows, Gallery(), 0.52)


def test_the_vlm_people_keep_their_forced_mark():
    row = image_row()
    assert cs.vlm_people(row) == {"A. Person": "vlm_free"}
    row["entities"]["PERSON"]["forced"] = True
    for item in row["entities"]["PERSON"]["items"]:
        item["forced"] = True
    assert cs.vlm_people(row) == {"A. Person": "vlm_forced"}


def test_an_excerpt_starts_near_a_search_word_not_at_the_menu():
    title = "Floods hit the coast"
    menu = " ".join(["Menu Home News Sport Weather Login"] * 30)
    text = f"{title} {menu} Residents of Chennai waded through water as the rain went on."
    got, anchor = cs.excerpt(text, title, ["Chennai", "Tuesday"], n=120)
    assert anchor == "term" and "Residents of Chennai" in got and got.startswith("…")
    assert len(got) <= 122 and title not in got
    got, anchor = cs.excerpt(text, title, ["Mumbai"], n=60)
    assert anchor == "opening" and got.startswith("Menu Home") and got.endswith("…")
    assert cs.excerpt("x Chennaiite y", "", ["Chennai"])[1] == "opening"  # whole words only
    assert cs.excerpt("A B C", "", ["b"]) == ("A B C", "term")
    # the window holding the most search words wins over the first word's
    privacy = f"We respect your privacy. {menu}"
    got, _ = cs.excerpt(f"{privacy} Manmohan Singh spoke with respect.", "",
                        ["respect", "Manmohan", "Singh"], n=120)
    assert "Manmohan Singh spoke" in got and "privacy" not in got
    assert cs.excerpt("Tom&#x27;s caf&eacute; in Greece", "", ["Greece"])[0] == "Tom's café in Greece"


def test_retrieval_resumes_refuses_another_config_and_records_a_refusal(tmp_path, capsys):
    from scripts.retrieve import Hit

    path = tmp_path / "m4fc.jsonl"
    items = [{"sha256": "a" * 64, "query": "Flood in Chennai", "before": "2021-01-02",
              "bound": "fact_check_date"},
             {"sha256": "b" * 64, "query": "Fire in Lagos", "before": None, "bound": None},
             {"sha256": "c" * 64, "query": "?! --", "before": "2021-01-02",
              "bound": "fact_check_date"}]
    calls = []

    def search(query, before):
        calls.append((query, before))
        return ([Hit("d1", "Floods", "2020-12-01", "news.example", "https://news.example/x", -3.2),
                 Hit("d2", "Gone", "2020-11-01", "old.example", "https://old.example/y", -2.0)],
                ["Chennai"], False)

    docs = {"d1": {"title": "Floods", "text": "Floods Menu Home. Water rose in Chennai overnight."}}
    cs.run_retrieval(items, path, "cfg1", search, docs.get)
    got = cs.read_retrieval(path, "cfg1")
    assert calls == [("Flood in Chennai", "2021-01-02")]
    first, gone = got["a" * 64]["hits"]
    assert first["anchor"] == "term" and "Chennai" in first["excerpt"]
    assert "url" not in first and "https" not in json.dumps(got)  # a hit never keeps its URL
    assert gone["excerpt"] is None and gone["anchor"] == "doc_missing"
    assert got["b" * 64]["error"] == "no_bound" and got["b" * 64]["hits"] == []
    assert got["c" * 64]["error"] == "empty_query"
    assert "  [retrieve] 3/3" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cs.read_retrieval(path, "cfg2")


def test_verite_gets_no_retrieval():
    from scripts import retrieve

    with pytest.raises(retrieve.RetrievalError):
        retrieve.evidence_bound("verite", {"date": "2020-01-01"})
    with pytest.raises(SystemExit):
        cs.main(["retrieve", "--dataset", "verite"])


def _hit(doc_id, published_at):
    return {"doc_id": doc_id, "title": "t", "published_at": published_at, "host": "news.example",
            "score": -1.0, "excerpt": "e", "anchor": "term"}


def test_the_retrieval_check_catches_what_the_statement_must_never_return():
    from scripts import retrieve

    rules = retrieve.load_blocklist()
    host = next(h for h, p in rules.sources if p is None)
    word = rules.words[0]
    sha = "a" * 64
    items = [{"sha256": sha, "query": "q", "before": "2021-01-02", "bound": "fact_check_date"}]
    row = {"sha256": sha, "before": "2021-01-02", "bound": "fact_check_date", "error": None,
           "query_sha256": hashlib.sha256(b"q").hexdigest(), "hits": [_hit("d1", "2020-01-01")]}
    pages = {"d1": ("2020-01-01", "news.example", "https://news.example/a"),
             "d2": ("2021-01-02", "news.example", "https://news.example/b"),
             "d3": ("2020-01-01", host, f"https://{host}/c"),
             "d5": ("2020-01-01", "news.example", f"https://news.example/{word}/e")}
    assert cs.check_retrieval(items, {sha: row}, lambda ids: pages)["problems"] == {}
    row["hits"] += [_hit("d2", "2021-01-02"), _hit("d3", "2020-01-01"), _hit("d4", "2020-01-01"),
                    _hit("d5", "2020-01-01")]
    assert cs.check_retrieval(items, {sha: row}, lambda ids: pages)["problems"] == {
        "a hit at or after its bound (ledger)": 1, "a hit at or after its bound (index)": 1,
        "a hit the fact-checker rules block (host)": 1, "a hit the fact-checker rules block (word)": 1,
        "a hit the index does not hold": 1}
    assert cs.check_retrieval(items, {}, lambda ids: pages)["problems"] == {"claims without a row": 1}


# --------------------------------------------------------------------------
# stage B.3: the context predictor
# --------------------------------------------------------------------------


def _canary_view(**over):
    row = canary_row()
    kw = dict(sha_of={row["image_paths"][0]: SHA}.get, image_rows={SHA: image_row()},
              people=[cs.Person("A. Person", "Q1", 0.61)],
              documents=[cs.Document("d" * 64, "Stadium crowd", "2020-05-01", "news.example",
                                     "Fans gathered outside.")])
    kw.update(over)
    return row, cs.view_of(row, **kw)


def test_a_canary_slug_and_its_claim_reach_no_predictor_prompt():
    row, view = _canary_view()
    msgs, evidence = cs.predict_messages(view)
    text = cs.messages_text(msgs)
    assert row["text"] not in text  # the predictor never sees the claim
    for form in slug_forms():
        assert form.casefold() not in text.casefold()
    assert list(evidence) == ["DESC", "OCR", "E1", "E2", "P1", "D1"]
    assert "[E2] The image model names a place: Lagos (recognised from how it looks, " \
           "confidence 0.90; weaker)" in text
    assert "[D1] A document published 2020-05-01 on news.example: Stadium crowd" in text
    assert evidence["D1"] == {"kind": "document", "doc_id": "d" * 64, "published_at": "2020-05-01",
                              "host": "news.example"}
    cs.check_predict_prompt(msgs, view, [row])  # passes
    leaky = [msgs[0], {"role": "user", "content": [{"type": "text", "text": text + "\n" + row["text"]}]}]
    with pytest.raises(cs.GuardError):
        cs.check_predict_prompt(leaky, view, [row])
    # an image whose own text quotes the claim carries it as content, never more often
    _row, quoting = _canary_view(image_rows={SHA: dict(image_row(), ocr={
        "text": row["text"], "regions": [], "mean_confidence": 0.9, "input": None, "error": None})})
    cs.check_predict_prompt(cs.predict_messages(quoting)[0], quoting, [row])


def test_a_repeated_story_takes_one_document_slot():
    def hit(doc_id, title, excerpt):
        return {"doc_id": doc_id, "title": title, "published_at": "2020-01-01", "host": "h.example",
                "score": -1.0, "excerpt": excerpt, "anchor": "term"}

    hits = [hit("a", "Story", "Text."), hit("b", "Story ", " text."), hit("c", "Other", "More."),
            *(hit(f"d{i}", f"T{i}", "E.") for i in range(5))]
    docs = cs.documents_of({"hits": hits})
    assert [d.doc_id for d in docs] == ["a", "c", "d0", "d1", "d2"]  # b repeats a; d2 fills its place


def test_the_predictor_sees_the_gallery_s_accepts_of_the_vlm_s_names_only():
    def person(name, sides, match=True, dataset="verite"):
        return {"dataset": dataset, "sha256": SHA, "name": name, "sides": sides, "match": match,
                "matched_entity": "Q1", "score": 0.6}

    got = cs.vlm_accepts([person("Claim Only", ["claim"]), person("Both", ["claim", "vlm_free"]),
                          person("Forced", ["vlm_forced"]), person("Rejected", ["vlm_free"], False),
                          person("Elsewhere", ["vlm_free"], dataset="m4fc")], "verite")
    assert [p.name for p in got[SHA]] == ["Both", "Forced"]


def _answer(**items):
    full = {"date": {"value": None, "evidence": []}, "location": {"value": None, "evidence": []},
            "event": {"value": None, "evidence": []}}
    full.update(items)
    return json.dumps(full)


def test_a_prediction_is_read_strictly():
    allowed = {"DESC", "OCR", "E1", "D1"}
    good = _answer(date={"value": "2019-05", "evidence": ["D1"]},
                   location={"value": "Lagos, Nigeria", "evidence": ["E1", "OCR"]})
    status, p = cs.parse_prediction(good, allowed, "eos")
    assert status == "ok" and p["flags"] == []
    assert p["items"]["date"] == {"value": "2019-05", "evidence": ["D1"], "error": None}
    assert p["items"]["event"] == {"value": None, "evidence": [], "error": None}
    assert cs.parse_prediction(good, allowed, "length") == (
        "error", {"error": "truncated", "items": None, "flags": []})
    assert cs.parse_prediction(f"```json\n{good}\n```", allowed, "eos")[1]["flags"] == ["code_fence"]
    assert cs.parse_prediction(f"Here: {good} Done.", allowed, "eos")[1]["flags"] == ["extra_text"]
    assert cs.parse_prediction("no object here", allowed, "eos")[1]["error"] == "not_json"
    assert cs.parse_prediction('{"date": null}', allowed, "eos")[1]["error"] == "missing_items"

    def one(item, raw):
        status, p = cs.parse_prediction(_answer(**{item: raw}), allowed, "eos")
        return status, p["items"][item], p["flags"]

    assert one("location", {"value": "Lagos", "evidence": ["E9"]})[:2] == (
        "partial", {"value": "Lagos", "evidence": ["E9"], "error": "unknown_evidence"})
    assert one("location", {"value": "Lagos", "evidence": []})[1]["error"] == "no_evidence"
    assert one("location", {"value": 5, "evidence": ["E1"]})[1]["error"] == "bad_value"
    assert one("date", {"value": "May 2019", "evidence": ["D1"]})[1]["error"] == "bad_date"
    assert one("date", {"value": "2019-13", "evidence": ["D1"]})[1]["error"] == "bad_date"
    assert one("date", {"value": "unknown", "evidence": []}) == (
        "ok", {"value": None, "evidence": [], "error": None}, ["null_word"])
    assert one("date", {"value": 2019, "evidence": ["[D1]"]}) == (
        "ok", {"value": "2019", "evidence": ["D1"], "error": None}, ["bracketed_ids", "number_date"])


def test_ids_left_unquoted_are_read_as_ids_and_nothing_else_is_touched():
    allowed = {"E2", "E3", "D1"}
    bare = ('{"date": {"value": "2022", "evidence": [E3]}, "location": {"value": "Syria", '
            '"evidence": [E2, [D1]]}, "event": {"value": null, "evidence": []}}')  # the first pilot's form
    status, p = cs.parse_prediction(bare, allowed, "eos")
    assert status == "ok" and p["flags"] == ["bare_ids"]
    assert p["items"]["location"]["evidence"] == ["E2", "D1"]
    good = _answer(date={"value": "2022", "evidence": ["E3"]})
    assert cs.quote_bare_ids(good) == good and cs.parse_prediction(good, allowed, "eos")[1]["flags"] == []
    spaced = '{"date": {"value": "2022", "evidence": [ "E3" ,"D1"]}}'
    assert cs.quote_bare_ids(spaced) == spaced  # valid JSON is never rewritten
    assert cs.quote_bare_ids('"evidence": [null]') == '"evidence": [null]'
    assert cs.parse_prediction('{"date": {"value": "x", "evidence": [E3 is it]}}', allowed,
                               "eos")[1]["error"] == "not_json"  # not an id: left to fail


def test_the_runner_runs_the_predictor_and_rechecks_it(tmp_path):
    _row, view = _canary_view()
    msgs, evidence = cs.predict_messages(view)
    rows = [{"key": "k1", "task": "predict", "messages": msgs, "content": cs.predict_content(view),
             "evidence": evidence}]
    path = tmp_path / "predict_in.jsonl"
    cs.write_prompts(path, "predict", rows, cs.predict_prompt_config(), {})
    args = ["llm", "--task", "predict", "--input", str(path), "--stub"]
    assert cs.main(args) == 0
    [out] = [json.loads(line) for line in (tmp_path / "predict_out.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert out["status"] == "ok"
    assert out["result"]["items"]["location"] == {"value": "Somewhere", "evidence": ["DESC"],
                                                  "error": None}
    assert cs.main(args + ["--recheck", "1"]) == 0
    with pytest.raises(SystemExit):  # an input for another task
        cs.main(["llm", "--task", "judge", "--input", str(path), "--stub"])


def test_the_predictor_input_waits_for_m4fc_s_image_text_ledger(monkeypatch):
    monkeypatch.setattr(cs, "image_text_rows", lambda dataset: None)
    with pytest.raises(SystemExit, match="not here yet"):
        cs.main(["prompts", "--task", "predict", "--stage", "b", "--part", "pilot"])
    with pytest.raises(SystemExit, match="--part"):
        cs.main(["prompts", "--task", "predict", "--stage", "b"])
    with pytest.raises(SystemExit):
        cs.main(["prompts", "--task", "judge", "--stage", "b"])


@needs_processed
def test_the_pilot_slice_is_30_m4fc_train_images_and_always_the_same():
    sha_of = cs.sha_lookup()
    rows = list(cs.rows("m4fc", cs.M4FC_SPLITS))
    picked = cs.pilot_slice(rows, sha_of)
    assert len(picked) == cs.PILOT_SLICE == 30 and {r["split"] for r in picked} == {"train"}
    assert len({sha_of(r["image_paths"][0]) for r in picked}) == 30
    assert ([r["record_id"] for r in cs.pilot_slice(list(reversed(rows)), sha_of)]
            == [r["record_id"] for r in picked])


@needs_processed
def test_no_m4fc_retrieved_document_carries_an_id_into_a_predictor_prompt():
    """The data-backed twin: every M4FC train+val claim's top documents, as
    the predictor would read them, through the predictor's own check."""
    path = cs.B_RETRIEVAL / "m4fc.jsonl"
    if not path.is_file():
        pytest.skip("needs the retrieval ledger")
    held = cs.read_retrieval(path)
    sha_of = cs.sha_lookup()
    n = 0
    for row in cs.rows("m4fc", cs.M4FC_SPLITS):
        view = cs.view_of(row, sha_of=sha_of, image_rows={},
                          documents=cs.documents_of(held[sha_of(row["image_paths"][0])]))
        cs.check_predict_prompt(cs.predict_messages(view)[0], view, [row])
        n += 1
    assert n == 3021


@needs_processed
def test_every_verite_predictor_prompt_is_id_free_and_claim_free():
    """The guard, not a look: VERITE's prompts are built and checked, and
    nothing of them is read."""
    from scripts import gallery as gl

    images = cs.image_text_rows("verite")
    if images is None or not cs.B_PEOPLE.is_file():
        pytest.skip("needs VERITE's image_text ledger and verify-people's output")
    out, keys, tally = cs.predict_rows(
        "verite", cs.evaluation_rows(), "image_only", sha_of=cs.sha_lookup(), image_rows=images,
        people=cs.vlm_accepts(gl.read_jsonl(cs.B_PEOPLE), "verite"))
    assert len(out) + tally["no_evidence"] == 588 and len(keys) == len(out)
    assert all(not r["key"].startswith("verite") for r in out)  # neutral keys


# --------------------------------------------------------------------------
# stage B.4: scoring the predictor
# --------------------------------------------------------------------------


def _b4(cls="false (out-of-context)", claimed=None, actual=None, preds=None, split="train", sha=SHA):
    blank = dict.fromkeys(cs.ITEMS)
    return {"sha256": sha, "split": split, "class": cls, "row": canary_row(),
            "claimed": {**blank, **(claimed or {})}, "actual": {**blank, **(actual or {})},
            "preds": preds or {"retrieval": dict.fromkeys(cs.PREDICT_ITEMS)}, "date_grounded": {}}


def test_the_predictor_s_output_is_read_from_where_b_predict_points_now(tmp_path, monkeypatch):
    """b_results once bound its path when the module loaded, so a run pointed
    elsewhere read the real keys file (caught by the B.4 dry run)."""
    _row, view = _canary_view()
    msgs, evidence = cs.predict_messages(view)
    inp = tmp_path / "predict_in.jsonl"
    cs.write_prompts(inp, "predict", [{"key": "k1", "task": "predict", "messages": msgs,
                                       "content": cs.predict_content(view), "evidence": evidence}],
                     cs.predict_prompt_config(), {"k1": f"m4fc:{SHA}:retrieval"})
    assert cs.main(["llm", "--task", "predict", "--input", str(inp), "--stub"]) == 0
    monkeypatch.setattr(cs, "B_PREDICT", inp)
    built, results = cs.b_results()
    assert built == {f"m4fc:{SHA}:retrieval"} and set(results) == built


def test_a_refused_prediction_is_not_a_null_and_an_unbuilt_one_is():
    ok = {"status": "partial", "result": {"items": {
        "date": {"value": "2019", "evidence": ["D1"], "error": None},
        "location": {"value": "Lagos", "evidence": [], "error": "no_evidence"},
        "event": {"value": None, "evidence": [], "error": None}}}}
    assert cs.predicted(ok) == {"date": "2019", "location": cs.ERROR, "event": None}
    assert cs.predicted({"status": "error", "result": {"items": None}}) == dict.fromkeys(
        cs.PREDICT_ITEMS, cs.ERROR)
    assert cs.predicted(None, built=False) == dict.fromkeys(cs.PREDICT_ITEMS)


def test_the_floor_is_the_image_model_s_free_lines_only():
    row = image_row()
    row["entities"]["DATE"] = {"status": "listed", "unidentified": False, "forced": False, "items": [
        {"name": "2019", "basis": "text", "confidence": 0.9, "in_ocr": True, "forced": False}]}
    got = cs.floor_values(row)
    assert got == {"date": "2019", "location": None, "event": None}  # Lagos is on a forced line
    bad = image_row()
    bad["entities"]["error"] = "truncated"
    assert cs.floor_values(bad) == dict.fromkeys(cs.FLOOR_LINES, cs.ERROR)
    assert cs.floor_values(None) == dict.fromkeys(cs.FLOOR_LINES, cs.ERROR)


def test_the_judge_pairs_are_oriented_deduplicated_and_echo_only_on_ooc():
    pred = {"retrieval": {"date": None, "location": "Accra", "event": None},
            "floor": {"date": None, "location": "Accra", "event": None}}
    ooc = _b4(claimed={"location": "Lagos"}, actual={"location": "Kumasi"}, preds=pred)
    true_ = _b4(cls="true", claimed={"location": "Lagos"}, actual={"location": "Kumasi"}, preds=pred,
                sha="b" * 64)
    pairs = cs.judge_pairs([ooc, true_])
    assert set(pairs) == {("location", "Lagos", "Kumasi"),  # the claim against the actual value
                          ("location", "Accra", "Kumasi"),  # a prediction against it (once, for both)
                          ("location", "Lagos", "Accra")}  # the claim against a prediction: OOC only
    assert len(pairs[("location", "Lagos", "Accra")]) == 2  # retrieval and floor, the OOC row only
    assert all(r is ooc for r in pairs[("location", "Lagos", "Accra")])
    shots = cs.fewshot()["judge"]
    for item, a, b in pairs:
        text = cs.messages_text(cs.judge_messages(item, a, b, shots)).casefold()
        assert all(form.casefold() not in text for form in slug_forms())


def test_accuracy_and_echo_read_the_comparator_on_the_right_records():
    table = {("location", "Accra", "Kumasi"): "contradicts", ("location", "Lagos", "Kumasi"): "contradicts",
             ("location", "Lagos", "Accra"): "consistent", ("location", "Kumasi", "Kumasi"): "consistent"}
    decide = lambda item, a, b: table[(item, a, b)]
    echo = _b4(claimed={"location": "Lagos"}, actual={"location": "Kumasi"},
               preds={"retrieval": {"date": None, "location": "Accra", "event": None}})
    right = _b4(claimed={"location": "Lagos"}, actual={"location": "Kumasi"},
                preds={"retrieval": {"date": None, "location": "Kumasi", "event": None}})
    refused = _b4(actual={"location": "Kumasi"},
                  preds={"retrieval": {"date": None, "location": cs.ERROR, "event": None}})
    acc = cs.accuracy_cell([echo, right, refused], "retrieval", "location", decide)
    assert acc["rows_with_actual"] == 3 and acc["refused"] == 1
    assert (acc["predicted"]["k"], acc["right"]["k"], acc["wrong"]["k"]) == (2, 1, 1)
    e = cs.echo_cell([echo, right, refused], "retrieval", "location", decide)
    assert e["rows"] == 2 and e["echo"]["k"] == 1 and e["echo"]["n"] == 2


def test_people_are_scored_by_qid_and_an_unresolved_name_decides_nothing():
    accepts = {"a" * 64: [cs.Person("X", "Q1", 0.6)], "b" * 64: [cs.Person("Y", "Q2", 0.6)],
               "c" * 64: [cs.Person("Z", "Q3", 0.6)]}
    resolve = {"X Person": ["Q1"], "Other": ["Q9"], "Nobody": []}.get
    recs = [_b4(actual={"people": "X Person"}, sha="a" * 64), _b4(actual={"people": "Other"}, sha="b" * 64),
            _b4(actual={"people": "Nobody"}, sha="c" * 64), _b4(actual={"people": "X Person"}, sha="d" * 64)]
    got = cs.people_cell(recs, accepts, lambda n: resolve(n) or [])
    assert got["rows_with_actual"] == 4 and got["covered"]["k"] == 3
    assert (got["right"]["k"], got["wrong"]["k"], got["undecided"]["k"]) == (1, 1, 1)
    assert got["precision"]["rate"] == 0.5


def test_a_date_is_grounded_only_by_a_year_its_cited_evidence_writes():
    view = cs.ModelView(None, None, "PRESS DAY 2017", "A crowd.",
                        (cs.Entity("DATE", "September 12", "seen", 0.9, False),),
                        documents=(cs.Document("d" * 64, "Market news", "2019-04-02", "news.example",
                                               "Prices rose."),))
    lines = cs.evidence_lines(cs.predict_messages(view)[0])
    assert lines["D1"] == "Market news Prices rose."  # the publication date is not evidence of when
    assert cs.date_grounded("2017-09-12", ["OCR"], lines)
    assert not cs.date_grounded("2021-09-12", ["E1"], lines)  # a year added to "September 12"
    assert not cs.date_grounded("2019-04", ["D1"], lines)  # the document's own date
    assert cs.contains("2017-09", "press day 2017", "date") and not cs.contains("2017", "20170", "date")
    assert cs.contains("Lagos; Accra", "flood in accra", "location")


# --------------------------------------------------------------------------
# stage C: extraction, verdicts, the score, metrics, the demonstration
# --------------------------------------------------------------------------


def test_an_extraction_is_read_strictly_and_a_year_must_be_the_claim_s():
    claim = "Floods in Lagos in July 2019."

    def one(answer, finish="eos"):
        return cs.parse_extraction(json.dumps(answer) if not isinstance(answer, str) else answer, claim, finish)

    status, p = one({"date": "2019-07", "location": "Lagos", "event": "floods"})
    assert status == "ok" and p["items"]["date"] == {"value": "2019-07", "error": None}
    assert one({"date": "2021-07", "location": "Lagos", "event": None})[1]["items"]["date"]["error"] \
        == "date_not_in_claim"  # a year the claim does not write was added
    assert one({"date": "July 2019", "location": None, "event": None})[1]["items"]["date"]["error"] == "bad_date"
    assert one({"date": "unknown", "location": None, "event": None}) == (
        "ok", {"error": None, "flags": ["null_word"], "items": {
            "date": {"value": None, "error": None}, "location": {"value": None, "error": None},
            "event": {"value": None, "error": None}}})
    assert one({"date": None, "location": 5, "event": None})[1]["items"]["location"]["error"] == "bad_value"
    assert one({"date": None}, "eos")[1]["error"] == "missing_items"
    assert one("not json", "eos")[1]["error"] == "not_json"
    assert one({"date": None, "location": None, "event": None}, "length")[1]["error"] == "truncated"


def test_an_extraction_reads_a_decade_or_a_span_whose_years_the_claim_writes():
    """Parser 2 (the person's decision, 2026-11-10): read as the date rule's interval."""
    claim = "Miners on strike in the 1980s, at a pit worked from 1951-1967."

    def one(value):
        answer = json.dumps({"date": value, "location": None, "event": None})
        return cs.parse_extraction(answer, claim, "eos")[1]["items"]["date"]

    assert one("1980s") == {"value": "1980s", "error": None}
    assert one("1951-1967") == {"value": "1951-1967", "error": None}
    assert one("1940-1945")["error"] == "date_not_in_claim"  # a span the claim does not write
    assert one("1970s")["error"] == "date_not_in_claim"
    for bad in ("79", "today", "2017, 2022", "1967-1951"):
        assert one(bad)["error"] == "bad_date", bad
    assert cs.date_rule("1980s", "1985-06")["label"] == cs.CONSISTENT


def test_the_image_carries_the_claim_only_where_text_alone_holds_the_claimed_value():
    view = cs.ModelView(None, None, "BREAKING: floods in Lagos today", "A flooded street.",
                        (cs.Entity("GPE", "Lagos", "text", 0.9, False), cs.Entity("GPE", "Accra", "seen", 0.8, False),
                         cs.Entity("EVENT", "Flood", "text", None, True)),
                        documents=(cs.Document("d" * 64, "Lagos floods", "2019-04-02", "news.example",
                                               "Rain in Lagos."),))
    lines = cs.evidence_lines(cs.predict_messages(view)[0])
    assert cs.text_evidence(["OCR"], lines) == "BREAKING: floods in Lagos today"
    assert cs.text_evidence(["E1", "E3"], lines) == "Lagos\nFlood"  # read as written, a weaker line too
    assert cs.text_evidence(["OCR", "DESC"], lines) is None  # a description as well: not text alone
    assert cs.text_evidence(["E2"], lines) is None  # recognised by how it looks
    assert cs.text_evidence(["D1"], lines) is None and cs.text_evidence([], lines) is None
    assert cs.image_carries_claim("location", "Lagos", ["OCR"], lines)
    assert not cs.image_carries_claim("location", "Accra", ["OCR"], lines)  # the text does not hold the claim
    assert not cs.image_carries_claim("location", "Lagos", ["OCR", "D1"], lines)
    assert cs.image_carries_claim("event", "floods in Lagos", ["OCR"], lines)
    assert not cs.image_carries_claim("location", None, ["OCR"], lines)


def test_widened_a_place_recognised_by_how_it_looks_whose_name_the_ocr_holds_rests_on_that_text():
    """Gate C's widening: a screenshot whose place line the image model calls
    "seen", though its name is written in the OCR text."""
    view = cs.ModelView(None, None, "Floods hit LAGOS this morning", "A flooded street.",
                        (cs.Entity("GPE", "Lagos", "seen", 0.9, False), cs.Entity("GPE", "Accra", "seen", 0.8, False),
                         cs.Entity("EVENT", "Lagos floods", "seen", None, False)))
    lines = cs.evidence_lines(cs.predict_messages(view)[0])
    assert cs.text_evidence(["E1"], lines, "location") is None  # the narrow form, as fixed before C
    assert cs.text_evidence(["E1"], lines, "location", "widened") == "Floods hit LAGOS this morning"
    assert cs.text_evidence(["E1", "OCR"], lines, "location", "widened") is not None
    assert cs.text_evidence(["E2"], lines, "location", "widened") is None  # Accra is not written there
    assert cs.text_evidence(["E1", "DESC"], lines, "location", "widened") is None  # not text alone
    assert cs.text_evidence(["E3"], lines, "event", "widened") is None  # place lines, for a place only
    assert cs.text_evidence(["E1"], lines, "event", "widened") is None
    assert cs.image_carries_claim("location", "Lagos; Nigeria", ["E1"], lines)  # widened is the rule now
    assert not cs.image_carries_claim("location", "Lagos", ["E1"], lines, "narrow")
    assert not cs.image_carries_claim("location", "Ibadan", ["E1"], lines)  # the text does not hold the claim
    with pytest.raises(ValueError):
        cs.text_evidence(["E1"], lines, "location", "wide")


def test_an_item_left_out_ungrounded_or_carried_is_unsupported_and_keeps_its_outcome():
    judged = {("location", "Lagos", "Lagos"): {"result": {"label": "consistent", "answer": "yes", "score": 0.05}}}
    look = lambda item, a, b: judged.get((item, a, b))
    v = cs.item_verdict("location", "Lagos", "Lagos", True, look, carried=True)
    assert (v["verdict"], v["why"], v["image_carries_claim"]) == ("unsupported", "image_carries_claim", True)
    assert v["outcome"]["label"] == "consistent"
    assert cs.item_verdict("location", "Lagos", "Lagos", False, look, carried=True)["why"] == "gate_a_not_cleared"
    d = cs.item_verdict("date", "2019", "2019-05", True, left_out=cs.GATE_B_LEFT_OUT)
    assert (d["verdict"], d["why"], d["outcome"]["label"]) == ("unsupported", "gate_b_left_out", "consistent")
    assert cs.item_verdict("date", "2019", "2021", True, grounded=False)["why"] == "date_not_grounded"
    assert cs.item_verdict("date", "2019", "2021", True, grounded=True)["why"] == "date_disjoint"


def test_the_extractor_reads_the_claim_and_its_exemplars_only(tmp_path):
    row = canary_row()
    msgs = cs.extract_messages(row["text"])
    shots = cs.fewshot()["extract"]
    assert len(msgs) == 2 + 2 * len(shots) and msgs[-1]["role"] == "user"
    assert row["text"] in msgs[-1]["content"][0]["text"]
    text = cs.messages_text(msgs).casefold()
    assert all(form.casefold() not in text for form in slug_forms())
    twin = dict(canary_row(), record_id="images/other.png")
    built, keys = cs.extract_rows("m4fc", [row, twin], shots)
    assert len(built) == 1 and list(keys.values()) == [f"m4fc:{cs.claim_key(row['text'])}"]  # one per claim
    path = tmp_path / "extract_in.jsonl"
    cs.write_prompts(path, "extract", built, cs.extract_prompt_config(), keys)
    assert cs.main(["llm", "--task", "extract", "--input", str(path), "--stub"]) == 0
    [out] = [json.loads(line) for line in (tmp_path / "extract_out.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert out["status"] == "ok" and out["result"]["items"]["location"]["value"] == "A"


def _v(item, verdict, score, why="w"):
    return {"item": item, "verdict": verdict, "score": score, "why": why, "evidence": []}


def test_an_item_s_verdict_keeps_its_comparator_s_outcome_and_respects_the_gate():
    judged = {("location", "Lagos", "Accra"): {"result": {"label": "contradicts", "answer": "no", "score": 0.91}}}
    look = lambda item, a, b: judged.get((item, a, b))
    assert cs.item_verdict("location", None, "Accra", True, look) is None  # no claim, no verdict
    v = cs.item_verdict("location", "Lagos", "Accra", True, look, [{"kind": "ocr"}])
    assert (v["verdict"], v["why"], v["score"], v["reason_code"]) == ("contradicted", "judge_no", 0.91,
                                                                        "media_mismatch")
    v = cs.item_verdict("location", "Lagos", "Accra", False, look)
    assert v["verdict"] == "unsupported" and v["why"] == "gate_a_not_cleared"
    assert v["outcome"]["label"] == "contradicts"  # what the comparator said is kept
    assert cs.item_verdict("location", "Lagos", None, True, look)["why"] == "no_image_evidence"
    assert cs.item_verdict("location", "Lagos", cs.ERROR, True, look)["why"] == "prediction_refused"
    assert cs.item_verdict("location", "Lagos", "Kumasi", True, look)["why"] == "comparator_not_run"
    d = cs.item_verdict("date", "2023", "2019", True)
    assert (d["verdict"], d["why"], d["score"], d["comparator"]) == ("contradicted", "date_disjoint", 1.0, "rule")
    assert cs.item_verdict("date", "the 2010s", "2014", True)["why"].startswith("comparator_unclear")


def test_the_people_rule_never_contradicts_without_the_person_s_check():
    sha = SHA

    def ver(name, match, reason, qid="Q1", faces=1, sides=("claim",)):
        return {"sha256": sha, "name": name, "match": match, "reason": reason, "qids": [qid],
                "matched_entity": qid, "score": 0.6, "faces_in_image": faces, "sides": list(sides),
                "matched_reference": {"page_url": "https://commons.example/x", "licence": "CC BY"}}

    accepted = {(sha, "A. Person"): ver("A. Person", True, "matched")}
    assert cs.people_verdict(sha, ["A. Person"], accepted, [])["verdict"] == "consistent"
    rejected = {(sha, "A. Person"): ver("A. Person", False, "below_threshold")}
    assert cs.people_verdict(sha, ["A. Person"], rejected, [])["why"] == "gallery_reject"
    other = [ver("B. Other", True, "matched", qid="Q2", sides=("vlm_free",))]
    v = cs.people_verdict(sha, ["A. Person"], rejected, other)
    assert v["verdict"] == "unsupported" and v["why"] == "other_person_unchecked"
    key = cs.other_person_key(sha, "A. Person", "Q2")
    assert cs.people_verdict(sha, ["A. Person"], rejected, other, {key: "unsure"})["verdict"] == "unsupported"
    v = cs.people_verdict(sha, ["A. Person"], rejected, other, {key: "contradicts"})
    assert v["verdict"] == "contradicted" and v["why"] == "other_person_confirmed" and v["score"] == 1.0
    crowd = [ver("B. Other", True, "matched", qid="Q2", faces=3, sides=("vlm_free",))]
    assert cs.people_verdict(sha, ["A. Person"], rejected, crowd)["why"] == "gallery_reject"  # not the only face
    assert cs.people_verdict(sha, [], rejected, other) is None
    assert cs.people_verdict(sha, ["C. Unseen"], {}, [])["why"] == "not_verified"


def test_the_gallery_evidence_carries_the_licence_and_attribution_as_words():
    """The review page showed "[object Object]": the gallery keeps a
    reference's licence and attribution as Commons states them, as objects."""
    row = {"name": "B. Other", "matched_entity": "Q2", "score": 0.6,
           "matched_reference": {"page_url": "https://commons.example/x",
                                 "licence": {"short_name": "CC BY-SA 3.0", "url": "https://cc.example"},
                                 "attribution": {"artist": "A. Photographer", "credit": "Own work",
                                                 "artist_html": "<a>A. Photographer</a>"}}}
    ev = cs.gallery_evidence(row)
    assert (ev["licence"], ev["attribution"]) == ("CC BY-SA 3.0", "A. Photographer, Own work")
    plain = dict(row, matched_reference={"page_url": None, "licence": "CC0", "attribution": None})
    assert (cs.gallery_evidence(plain)["licence"], cs.gallery_evidence(plain)["attribution"]) == ("CC0", None)
    assert not hasattr(cs, "face_not_in_photo")  # the face rule was reverted: the person's review is the filter

def test_the_score_is_the_largest_decided_contradiction_and_nothing_decided_abstains():
    assert cs.record_score([_v("date", "contradicted", 1.0), _v("event", "consistent", 0.1), None]) == {
        "score": 1.0, "abstained": False, "by": "date"}
    assert cs.record_score([_v("date", "unsupported", None), None]) == {"score": 0.0, "abstained": True,
                                                                        "by": None}


def test_the_gate_c_reads_is_the_reviewed_one_once_the_person_decides():
    built = {"gate": {"date": {"passed": False}, "location": {"passed": True}, "event": {"passed": False}},
             "review": {"decided": 0}}
    assert cs.gate_cleared(built) == {"date": False, "location": True, "event": False}
    reviewed = dict(built, review={"decided": 12, "gate": {"date": {"passed": True},
                                                           "location": {"passed": True}, "event": {"passed": False}}})
    assert cs.gate_cleared(reviewed) == {"date": True, "location": True, "event": False}


def _rec(key, cls, score, clip, ds="verite", split="test", sha=None, claim=None):
    return {"key": key, "dataset": ds, "split": split, "class": cls, "score": score, "clip": clip,
            "sha256": sha or key, "claim_key": claim or key}


def test_a_threshold_is_fitted_on_m4fc_train_alone():
    train = [_rec(f"t{i}", "true", 0.1 * i, 0, "m4fc", "train") for i in range(3)]
    train += [_rec(f"o{i}", "false (out-of-context)", 0.5 + 0.1 * i, 0, "m4fc", "train") for i in range(3)]
    got = cs.fit_threshold(train)
    assert got["threshold"] == 0.5 and got["balanced_accuracy"] == 1.0
    with pytest.raises(cs.GuardError):
        cs.fit_threshold(train + [_rec("v", "true", 0.0, 0, "m4fc", "val")])
    with pytest.raises(cs.GuardError):
        cs.fit_threshold(train + [_rec("x", "true", 0.0, 0)])


def test_verite_s_triplets_are_resampled_whole_and_the_bootstrap_is_paired():
    recs = []
    for i in range(30):  # a true row and its miscaptioned row share an image; the OOC row the caption
        recs += [_rec(f"t{i}", "true", 0.1, 0.5, sha=f"img{i}", claim=f"cap{i}"),
                 _rec(f"m{i}", "miscaptioned", 0.9, 0.5, sha=f"img{i}", claim=f"mis{i}"),
                 _rec(f"o{i}", "out-of-context", 0.8, 0.5, sha=f"other{i}", claim=f"cap{i}")]
    clusters = cs.clusters_of(recs)
    assert len(set(clusters.values())) == 30 and clusters["t0"] == clusters["m0"] == clusters["o0"]
    got = cs.paired_auroc(recs, "out-of-context", "true", "score", "clip", n=200)
    assert got["auroc_score"] == 1.0 and got["auroc_clip"] == 0.5 and got["delta"] == 0.5
    assert got["excludes_zero"] and got == cs.paired_auroc(recs, "out-of-context", "true", "score", "clip", n=200)
    assert cs.flag_cell([dict(r, abstained=False) for r in recs[:3]], 0.5)["flagged"]["k"] == 2


def test_verdicts_and_report_c_run_end_to_end_on_synthetic_records():
    """The primary score reads location and people; dates and events count only
    in their labelled variants; image_carries_claim is measured on train."""
    ooc = "false (out-of-context)"

    def rec(key, ds, split, cls, sha, claim):
        meta = {"verdict": cls, "claimed_date": "2019", "claimed_location": "Lagos", "claimed_event": None}
        return {"key": key, "dataset": ds, "split": split, "class": cls, "sha256": sha, "path": None,
                "claim": claim, "claim_key": cs.claim_key(claim), "row": {"record_id": key, "meta": meta}}

    records = [rec(f"{s}{i}", "m4fc", s, "true" if i % 2 == 0 else ooc, f"img-{s}{i}", f"claim {s}{i}")
               for s in ("train", "val") for i in range(6)]
    for i in range(3):  # VERITE: a true row and its miscaptioned twin share an image; OOC reuses the caption
        records += [rec(f"vt{i}", "verite", "test", "true", f"v{i}", f"cap {i}"),
                    rec(f"vm{i}", "verite", "test", "miscaptioned", f"v{i}", f"mis {i}"),
                    rec(f"vo{i}", "verite", "test", "out-of-context", f"vx{i}", f"cap {i}")]
    ctx = lambda date, place: {"date": date, "location": place, "event": None}
    # M4FC: a true record's image agrees with its claim; VERITE: one prediction per image, so a true row
    # and its miscaptioned twin differ only in what their captions claim
    preds = {f"m4fc:{r['sha256']}:retrieval": ctx("2019" if r["class"] == "true" else "2015",
                                                  "Lagos" if r["class"] == "true" else "Accra")
             for r in records if r["dataset"] == "m4fc"}
    preds.update({f"verite:v{i}:image_only": ctx(None, "Lagos") for i in range(3)})
    preds.update({f"verite:vx{i}:image_only": ctx(None, "Accra") for i in range(3)})
    seen = "The image model names a date: {} (recognised from how it looks)"
    evidence = {name: {"ids": {"date": ["E1"], "location": ["DESC"]},
                       "lines": {"E1": seen.format(p["date"]) if p["date"] == "2019" else seen.format("May"),
                                 "DESC": "An image model's description of the photograph: a street."}}
                for name, p in preds.items()}
    # one true train record's place rests on the image's own text, which holds the claim: a screenshot
    evidence["m4fc:img-train2:retrieval"]["ids"]["location"] = ["OCR"]
    evidence["m4fc:img-train2:retrieval"]["lines"]["OCR"] = "Text read in the photograph: LAGOS TODAY"
    extracted = {f"{r['dataset']}:{r['claim_key']}": ctx("2019", "Accra" if r["claim"].startswith("mis") else "Lagos")
                 for r in records}
    judged = {("location", a, b): {"result": {"label": "consistent" if a == b else "contradicts",
                                              "answer": "yes" if a == b else "no", "score": 0.05 if a == b else 0.9}}
              for a in ("Lagos", "Accra") for b in ("Lagos", "Accra")}
    verdicts = cs.assemble_verdicts(records, preds, {}, extracted, lambda i, a, b: judged.get((i, a, b)),
                                    {"date": True, "location": True, "event": False}, {}, {}, lambda r: [], {},
                                    evidence)
    assert len(verdicts) == 24 + 9  # M4FC twice (extracted, gold), VERITE once
    v = next(x for x in verdicts if x["key"] == "train1" and x["variant"] == "gold")
    assert v["items"]["location"]["verdict"] == "contradicted"
    assert v["items"]["date"]["why"] == "gate_b_left_out" and v["items"]["date"]["outcome"]["label"] == "contradicts"
    assert v["variants"]["date"]["why"] == "date_not_grounded"  # "2015" rests on a line that writes no year
    assert v["scores"]["primary"] == {"score": 0.9, "abstained": False, "by": "location"}
    t = next(x for x in verdicts if x["key"] == "train0" and x["variant"] == "gold")
    assert t["variants"]["date"]["verdict"] == "consistent"  # grounded: its evidence writes 2019
    carried = next(x for x in verdicts if x["key"] == "train2" and x["variant"] == "extracted")
    assert carried["items"]["location"]["why"] == "image_carries_claim"
    assert carried["rule_changed"] == {"location": {"was": "consistent", "score_was": 0.05}}
    assert carried["scores"]["primary"]["abstained"] and not carried["scores_without_rule"]["primary"]["abstained"]
    assert carried["scores_narrow_rule"]["primary"]["abstained"] and carried["widening_changed"] == {}
    clip = {r["key"]: 0.3 - 0.01 * (r["class"] != "true") for r in records}
    rep = cs.c_report(verdicts, clip)
    assert rep["thresholds"]["gold:primary"]["system"]["threshold"] == 0.9
    m4fc = rep["corpora"]["m4fc:gold:primary"]
    assert m4fc["records"] == 6 and m4fc["auroc"][0]["auroc_score"] == 1.0  # val only, never train
    verite = rep["corpora"]["verite:extracted:primary"]
    assert verite["records"] == 9 and [a["positive"] for a in verite["auroc"]] == ["out-of-context", "miscaptioned"]
    assert verite["auroc"][1]["auroc_score"] == 1.0 and verite["auroc"][1]["auroc_clip"] == 1.0
    assert set(rep["corpora"]) >= {f"{ds}:extracted:{s}" for ds in ("m4fc", "verite") for s in cs.SCORE_SETS}
    effect = rep["image_carries_claim"]
    assert effect["rule"] == "widened" and effect["train"]["changed"] == {"location": {"true: was consistent": 1}}
    assert effect["train"]["widening_changed"] == {} and effect["train"]["scores"]["primary"]["narrow"]["auroc"]
    assert effect["applied_unchanged"]["verite"]["changed"] == {}
    assert set(effect["applied_unchanged"]["verite"]["primary_auroc"]) == {"out-of-context vs true",
                                                                          "miscaptioned vs true"}
    fused = rep["fused"]
    assert fused["model"]["converged"] and fused["model"]["records"] == {"true": 3, "out_of_context": 3}
    assert set(fused["corpora"]) == {"m4fc", "verite"} and fused["corpora"]["m4fc"]["records"] == 6
    assert set(fused["beats_clip"]) == {"m4fc: false (out-of-context) vs true", "verite: out-of-context vs true",
                                        "verite: miscaptioned vs true"}
    text = cs.render_c({**rep, "decisions": cs.C_DECISIONS, "people_check": {"decisions": 0},
                        "extraction": {"parser": 2, "datasets": {}}})
    assert "## image_carries_claim" in text and "## The system score: CLIP and the consistency score, fused" in text


def test_the_fused_score_is_fitted_on_m4fc_train_and_is_clip_s_where_the_block_abstains():
    ooc = "false (out-of-context)"

    def row(i, split="train", ds="m4fc"):
        mismatch = i % 2 == 1
        decided = i % 3 == 0
        return {"variant": cs.FUSED_FROM, "dataset": ds, "split": split, "class": ooc if mismatch else "true",
                "clip": -(0.30 - 0.004 * i - 0.03 * mismatch), "abstained": not decided,
                "score": (0.9 if mismatch else 0.05) if decided else 0.0}

    train = [row(i) for i in range(40)]
    model = cs.fit_fused(train)
    assert model["converged"] and model["records"] == {"true": 20, "out_of_context": 20}
    assert model["features"] == ["clip_similarity", "consistency_score", "decided"]
    # val and VERITE records never move the fit
    assert cs.fit_fused(train + [row(i, "val") for i in range(9)] + [row(i, "test", "verite") for i in range(9)]) \
        == model
    # where the block abstains, the fused score follows CLIP's similarity alone (lower similarity, higher score)
    lo, hi = (cs.fused_score(model, cs.fused_features(s, 0.0, True)) for s in (0.20, 0.30))
    assert lo > hi
    assert cs.fused_score(model, cs.fused_features(0.25, 0.7, True)) == cs.fused_score(
        model, cs.fused_features(0.25, 0.0, True))  # an abstention's score never enters
    fitted = {"model": model, "threshold": {"threshold": 0.5}}
    f = cs.fused_flag(fitted, 0.25, {"score": 0.0, "abstained": True, "by": None})
    assert f["basis"] == "No item is decided, so the flag rests on image–text similarity alone (0.250)."
    g = cs.fused_flag(fitted, 0.25, {"score": 0.9, "abstained": False, "by": "location"})
    assert g["basis"].startswith("The flag combines the image–text similarity (0.250) with the consistency verdicts")
    assert g["flag"].startswith("Flagged" if g["score"] >= 0.5 else "Not flagged")
    with pytest.raises(ValueError):
        cs.fit_logistic([[0.1], [0.2]], [1, 1])  # one class


def test_the_live_server_stages_and_record_run_on_stand_in_models(tmp_path, monkeypatch):
    """The two server stages and the laptop's verdicts, with stand-in models:
    every file a stage reads is the one the stage before wrote."""
    import io as _io

    from PIL import Image

    monkeypatch.setattr(cs, "LIVE_DIR", tmp_path)
    img = Image.new("RGB", (320, 240))
    for x in range(320):
        for y in range(0, 240, 8):
            img.putpixel((x, y), ((x * 7) % 256, (y * 3) % 256, 90))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    body = buf.getvalue()
    sha = hashlib.sha256(body).hexdigest()
    claim = "Floods in Lagos after the storm."
    key = cs.live_key(sha, claim, None)
    (tmp_path / key).mkdir()
    (tmp_path / key / "image.png").write_bytes(body)
    cs.write_json(tmp_path / key / "request.json", {"key": key, "sha256": sha, "image": "image.png", "claim": claim,
                                                    "date": None, "claim_people": [], "retrieval": None})
    assert cs.main(["live-server", "vlm", "--keys", key, "--stub"]) == 0
    row = json.loads((tmp_path / key / "image_text.json").read_text(encoding="utf-8"))
    assert row["sha256"] == sha and row["record_ids"] == [] and row["status"] in ("ok", "partial")
    ocr, desc, ents = cs.image_part(row, sha)
    msgs, evidence = cs.predict_messages(cs.ModelView(sha, claim, ocr, desc, ents))
    cs.write_json(tmp_path / key / "llm_in.json", {
        "predict": {"key": "predict", "task": "predict", "messages": msgs, "content": "", "evidence": evidence},
        "extract": {"key": "extract", "task": "extract", "messages": cs.extract_messages(claim), "content": claim}})
    assert cs.main(["live-server", "llm", "--keys", key, "--stub"]) == 0
    out = json.loads((tmp_path / key / "llm_out.json").read_text(encoding="utf-8"))
    assert out["extract"]["status"] == "ok" and out["predict"]["status"] in ("ok", "partial")
    assert list(out["judge"]) == ["location\nFloods\nSomewhere"]  # the stand-ins' place against the claim's
    cs.write_json(tmp_path / key / "people.json", [])
    rec = cs.live_record(key, expect=None)
    assert rec["items"]["location"]["claimed"] == "Floods" and rec["items"]["location"]["predicted"] == "Somewhere"
    assert set(rec["scores"]) == set(cs.SCORE_SETS)
    with pytest.raises(SystemExit):
        cs.live_record(key)  # the stand-ins' configurations are not the evaluated ones


def test_every_explanation_is_its_template_filled_from_the_record():
    base = {"item": "location", "claimed": "Lagos", "predicted": "Accra", "comparator": "judge", "score": 0.91,
            "evidence": [{"kind": "document", "host": "news.example", "published_at": "2019-05-02", "doc_id": "d"}],
            "outcome": {"label": "contradicts"}, "why": "judge_no"}
    text = cs.explain(base)
    assert text == ("The claim's place is Lagos; the image's evidence says Accra (source: news.example, "
                    "2019-05-02). The comparator finds they cannot both be true (P = 0.91).")
    assert "Kumasi" in cs.explain(dict(base, predicted="Kumasi"))  # a field changed is the sentence changed
    gate = cs.explain(dict(base, why="gate_a_not_cleared", score=None))
    assert "cannot both be true" in gate and "not cleared Gate A" in gate
    assert "not yet checked by a person" in cs.explain(dict(base, item="people", why="other_person_unchecked")).lower()
    assert "the name is not in the gallery" in cs.explain(dict(base, item="people", why="name_not_in_gallery"))
    with pytest.raises(KeyError):
        cs.explain(dict(base, why="free_text"))
    reachable = {"date_disjoint", "date_overlap", "judge_no", "judge_yes", "comparator_unclear", "comparator_not_run",
                 "gate_a_not_cleared", "gate_b_left_out", "date_not_grounded", "image_carries_claim",
                 "no_image_evidence", "prediction_refused", "claimed_person_accepted",
                 "gallery_reject", "other_person_unchecked", "other_person_not_confirmed",
                 "other_person_confirmed", "gallery_cannot"}
    assert set(cs.EXPLAIN) == reachable


def test_a_demo_page_says_when_nothing_was_retrieved_and_refuses_an_id(tmp_path):
    v = cs.item_verdict("date", "2023", "2019", True, evidence=[{"kind": "ocr"}])
    bundle = cs.demo_bundle("dkey", None, None, "A crowd gathers outside a stadium.", None, None, {"date": v})
    page = cs.demo_html(bundle, None)
    assert "no documents were retrieved" in page and "date_disjoint" not in page
    assert "These dates cannot both be true." in page and "<script" not in page
    cs.write_demo(bundle, cs.id_forms(canary_row()), tmp_path)
    leaky = dict(bundle, notes=[f"from {SLUG}"])
    with pytest.raises(cs.GuardError):
        cs.write_demo(leaky, cs.id_forms(canary_row()), tmp_path)


@needs_gpu
def test_the_predictor_answers_in_form_and_is_deterministic():
    """On the server: a synthetic photograph's evidence, from no corpus."""
    llm = cs.TextLLM()
    llm.prepare()
    llm.load()
    view = cs.ModelView(None, None, "LYON MARATHON 2014 FINISH",
                        "A photograph of runners crossing a finish line under a banner.",
                        (cs.Entity("GPE", "Lyon", "text", 0.95, False),
                         cs.Entity("EVENT", "Lyon Marathon", "text", 0.9, False)))
    msgs, evidence = cs.predict_messages(view)
    first = llm.generate(msgs, cs.PREDICT_MAX_NEW_TOKENS)
    assert llm.generate(msgs, cs.PREDICT_MAX_NEW_TOKENS) == first
    status, _parsed = cs.parse_prediction(first["text"], evidence, first["finish"])
    assert first["finish"] == "eos" and status in ("ok", "partial"), first
