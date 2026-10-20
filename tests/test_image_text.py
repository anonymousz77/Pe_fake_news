"""The image-to-text block: checkpointing, gates, schema, determinism, recognition.

Offline and data-free by default. A deterministic stub stands in for the
models, synthetic images stand in for corpora, and the kill test kills a real
subprocess. Two groups need more, and skip without it:

- real-data identity checks need the processed layer and the image index;
- the GPU group (``PE_FAKE_NEWS_GPU_TESTS=1``) loads Qwen2.5-VL and EasyOCR and
  asserts on real images: the same image processed twice is byte-identical,
  and two known Factify2 TRAIN images (never VERITE: it is the evaluation set)
  say what they actually show. Referenced by sha256 only.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.paths import PROCESSED  # noqa: E402
from scripts import image_text, images, placeholders  # noqa: E402
from scripts.image_text import (  # noqa: E402
    CATEGORIES, FatalError, Generation, Ledger, LedgerError, LockHeld, StubOCR, StubVLM, WorkItem,
)
from scripts.processed_loader import SupervisionError  # noqa: E402

SCRIPT = REPO / "scripts" / "image_text.py"


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _png(colour=(10, 20, 30), size=(64, 48)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _noise_jpeg(size=(96, 96)) -> bytes:
    from PIL import Image
    img = Image.frombytes("RGB", size, hashlib.sha256(b"seed").digest() * (size[0] * size[1] * 3 // 32 + 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _put(root: Path, rel: str, body: bytes, dataset: str = "verite", rid: str = "1") -> WorkItem:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return WorkItem(dataset, _sha(body), rel, (rid,), 1)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Images resolve under tmp_path; the registry and the rejections are empty."""
    monkeypatch.setattr(image_text, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(placeholders, "registry", lambda: {})
    monkeypatch.setattr(images, "fingerprint_rejections", lambda: {})
    return tmp_path


def _ctx(tmp_path, vlm=None, ocr=None):
    vlm, ocr = vlm or StubVLM(), ocr or StubOCR()
    return vlm, ocr, image_text.make_context(vlm, ocr, out_root=tmp_path / "out")


# --------------------------------------------------------------------------
# checkpointing: a kill loses at most the image in flight
# --------------------------------------------------------------------------


def _fake_project(root: Path, n: int) -> Path:
    lines = []
    for i in range(n):
        body = _png((i * 19 % 256, 7, 99), (40 + i, 30))
        rel = f"data/raw/verite/images/img{i:02d}.png"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
        lines.append(json.dumps({"dataset": "verite", "sha256": _sha(body), "path": rel,
                                 "record_ids": [str(i)], "n_paths": 1, "width": 40 + i, "height": 30}))
    worklist = root / "worklist.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return worklist


def _env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PE_FAKE_NEWS_ROOT"] = str(root)
    env["PE_FAKE_NEWS_MODELS"] = str(root / "models")
    return env


def _ledger_file(root: Path) -> Path | None:
    found = sorted((root / "data" / "interim" / "image_text").glob("*/verite.jsonl"))
    return found[0] if found else None


def test_a_kill_loses_at_most_the_image_in_flight(tmp_path):
    """A real process, really killed (SIGKILL / TerminateProcess) mid-run, and
    a torn half-line on top: the restart rewrites nothing already written,
    computes exactly what is missing, and ends where an uninterrupted run ends."""
    root = tmp_path / "killed"
    root.mkdir()
    worklist = _fake_project(root, 12)
    cmd = [sys.executable, str(SCRIPT), "run", "--stub", "--worklist", str(worklist)]
    err = (tmp_path / "killed.err").open("wb")
    slow = subprocess.Popen(cmd + ["--stub-delay", "0.25"], env=_env(root),
                            stdout=subprocess.DEVNULL, stderr=err)
    deadline = time.time() + 120
    while time.time() < deadline:
        ledger = _ledger_file(root)
        if ledger is not None and ledger.read_bytes().count(b"\n") >= 3:
            break
        if slow.poll() is not None:
            err.close()
            pytest.fail("the run ended before it could be killed: "
                        f"{(tmp_path / 'killed.err').read_text(errors='replace')}")
        time.sleep(0.05)
    slow.kill()
    slow.wait(timeout=30)
    err.close()
    ledger = _ledger_file(root)
    before = ledger.read_bytes()
    written = before.count(b"\n")
    assert before.endswith(b"\n") and 3 <= written < 12

    with ledger.open("ab") as fh:  # and a kill in the middle of a write
        fh.write(b'{"schema":"image_text/1","dataset":"ver')
    resumed = subprocess.run(cmd, env=_env(root), capture_output=True, text=True, timeout=180)
    assert resumed.returncode == 0, resumed.stderr

    after = ledger.read_bytes()
    assert after.startswith(before), "a row written before the kill was rewritten"
    rows = [json.loads(line) for line in after.splitlines()]
    shas = [r["sha256"] for r in rows]
    assert len(shas) == len(set(shas)) == 12, "an image is missing or was done twice"
    first, second = rows[:written], rows[written:]
    assert len(second) == 12 - written, "the restart recomputed finished images"
    assert {r["run"]["run_id"] for r in second}.isdisjoint({r["run"]["run_id"] for r in first})
    torn = json.loads(ledger.with_name("verite.torn.jsonl").read_bytes().splitlines()[-1])
    assert torn["fragment"] == '{"schema":"image_text/1","dataset":"ver', "kept, not deleted"

    reference = tmp_path / "uninterrupted"
    reference.mkdir()
    ref_list = _fake_project(reference, 12)
    ref = subprocess.run([sys.executable, str(SCRIPT), "run", "--stub", "--worklist", str(ref_list)],
                         env=_env(reference), capture_output=True, text=True, timeout=180)
    assert ref.returncode == 0, ref.stderr
    ref_rows = [json.loads(line) for line in _ledger_file(reference).read_bytes().splitlines()]
    assert ({r["sha256"]: r["result_sha256"] for r in rows}
            == {r["sha256"]: r["result_sha256"] for r in ref_rows})


def test_a_torn_tail_is_cut_kept_and_the_next_append_is_clean(fake_root):
    vlm, ocr, ctx = _ctx(fake_root)
    items = [_put(fake_root, f"data/raw/verite/a{i}.png", _png((i, 1, 2))) for i in range(3)]
    path = ctx.out_dir / "verite.jsonl"
    ledger = Ledger(path, dataset="verite", config_id=ctx.config_id)
    ledger.open()
    for item in items[:2]:
        ledger.append(image_text.process(item, vlm, ocr, ctx))
    ledger.close()
    clean = path.read_bytes()
    with path.open("ab") as fh:
        fh.write(b'{"schema": "image_text/1", "data')
    ledger = Ledger(path, dataset="verite", config_id=ctx.config_id)
    summaries, _ = ledger.open()
    assert set(summaries) == {items[0].sha256, items[1].sha256}
    assert path.read_bytes() == clean
    torn = json.loads((ctx.out_dir / "verite.torn.jsonl").read_bytes().splitlines()[-1])
    assert torn["fragment"] == '{"schema": "image_text/1", "data' and torn["offset"] == len(clean)
    ledger.append(image_text.process(items[2], vlm, ocr, ctx))
    ledger.close()
    lines = path.read_bytes().splitlines()
    assert len(lines) == 3 and all(not image_text.row_problems(json.loads(x)) for x in lines)


@pytest.mark.parametrize("damage", ["not_json", "edited"])
def test_a_damaged_line_before_the_end_stops_the_run(fake_root, damage):
    """Skipping it would hide the damage; the refusal names the line."""
    vlm, ocr, ctx = _ctx(fake_root)
    path = ctx.out_dir / "verite.jsonl"
    ledger = Ledger(path, dataset="verite", config_id=ctx.config_id)
    ledger.open()
    for i in range(3):
        ledger.append(image_text.process(_put(fake_root, f"data/raw/verite/b{i}.png",
                                              _png((i, 9, 9))), vlm, ocr, ctx))
    ledger.close()
    lines = path.read_bytes().split(b"\n")
    lines[1] = (lines[1][:40] if damage == "not_json"
                else lines[1].replace(b'"status":"ok"', b'"status":"failed"'))
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(LedgerError, match="line 2"):
        Ledger(path, dataset="verite", config_id=ctx.config_id).open()


def test_a_second_writer_is_refused_and_the_lock_dies_with_its_holder(tmp_path):
    path = tmp_path / "verite.jsonl"
    first = Ledger(path, dataset="verite", config_id="0" * 12)
    first.open()
    with pytest.raises(LockHeld):
        Ledger(path, dataset="verite", config_id="0" * 12).open(lock_wait=0)
    first.close()
    again = Ledger(path, dataset="verite", config_id="0" * 12)
    again.open(lock_wait=0)
    again.close()


def test_a_retried_image_appends_and_the_last_row_wins(fake_root):
    vlm, ocr, ctx = _ctx(fake_root)
    body = _png((5, 5, 5))
    item = WorkItem("verite", _sha(body), "data/raw/verite/late.png", ("7",), 1)
    runner = image_text.Runner(vlm, ocr, ctx)
    runner.open("verite")
    assert runner.step(item)["reason"] == "missing_file"
    assert runner.pending([item]) == [] and runner.pending([item], retry_failed=True) == [item]
    _put(fake_root, item.path, body)
    assert runner.step(item)["status"] == "ok"
    runner.close()
    summaries, _, _ = image_text.scan_ledger(ctx.out_dir / "verite.jsonl")
    assert summaries[item.sha256]["status"] == "ok" and summaries[item.sha256]["attempt"] == 2


def test_nothing_is_ever_written_under_data_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(image_text, "RAW", tmp_path / "data" / "raw")
    with pytest.raises(FatalError, match="data/raw"):
        Ledger(tmp_path / "data" / "raw" / "verite" / "x.jsonl", dataset="verite",
               config_id="0" * 12).open()


# --------------------------------------------------------------------------
# the gates: failures are rows, registry images are skipped by design
# --------------------------------------------------------------------------


def test_every_bad_image_is_a_row_with_a_reason_code(fake_root):
    vlm, ocr, ctx = _ctx(fake_root)
    good, jpeg = _png(), _noise_jpeg()
    cut = jpeg[: len(jpeg) // 2]
    (fake_root / "data/raw/verite/a_dir").mkdir(parents=True)
    zpath = fake_root / "data/raw/averimatec/images.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("images/present.png", good)
    cases = {
        "missing_file": WorkItem("verite", _sha(good), "data/raw/verite/nope.png", ("1",), 1),
        "unreadable": WorkItem("verite", _sha(good), "data/raw/verite/a_dir", ("2",), 1),
        "missing_member": WorkItem("averimatec", _sha(good),
                                   "data/raw/averimatec/images.zip::images/absent.png", ("3",), 1),
        "undecodable": _put(fake_root, "data/raw/verite/cut.jpg", cut, rid="4"),
        "sha256_mismatch": WorkItem("verite", _sha(b"other bytes"),
                                    _put(fake_root, "data/raw/verite/changed.png", good).path, ("5",), 1),
        "unsupported_geometry": _put(fake_root, "data/raw/verite/sliver.png",
                                     _png(size=(402, 2)), rid="6"),
    }
    seen = Counter()
    for reason, item in cases.items():
        row = image_text.process(item, vlm, ocr, ctx)
        assert (row["status"], row["reason"]) == ("failed", reason), reason
        assert image_text.row_problems(row) == []
        seen[row["reason"]] += 1
    assert seen == Counter(cases.keys())
    assert vlm.calls == 0 and ocr.calls == 0, "a failed image reached a model"
    ok = image_text.process(WorkItem("averimatec", _sha(good),
                                     "data/raw/averimatec/images.zip::images/present.png", ("8",), 1),
                            vlm, ocr, ctx)
    assert ok["status"] == "ok", "a zip member that exists is read"


def test_registry_images_are_skipped_recorded_and_never_shown_to_a_model(fake_root, monkeypatch):
    bodies = {name: _png(colour) for name, colour in
              (("placeholder", (1, 1, 1)), ("verdict", (2, 2, 2)), ("wrong", (3, 3, 3)),
               ("stock", (4, 4, 4)), ("plain", (5, 5, 5)))}
    registry = {_sha(bodies["placeholder"]): {"status": "placeholder", "description": "x"},
                _sha(bodies["verdict"]): {"status": "verdict", "description": "x"},
                _sha(bodies["stock"]): {"status": "generic_stock", "description": "x"}}
    monkeypatch.setattr(placeholders, "registry", lambda: registry)
    monkeypatch.setattr(images, "fingerprint_rejections",
                        lambda: {("verite", _sha(bodies["wrong"])): {"dataset": "verite"}})
    vlm, ocr, ctx = _ctx(fake_root)
    rows = {name: image_text.process(_put(fake_root, f"data/raw/verite/{name}.png", body), vlm, ocr, ctx)
            for name, body in bodies.items()}
    assert (rows["placeholder"]["status"], rows["placeholder"]["reason"]) == ("skipped", "placeholder")
    assert (rows["verdict"]["status"], rows["verdict"]["reason"]) == ("skipped", "verdict")
    assert (rows["wrong"]["status"], rows["wrong"]["reason"]) == ("skipped", "wrong_image")
    assert rows["stock"]["status"] == "ok", "generic_stock is recorded, not missing"
    assert rows["plain"]["status"] == "ok"
    assert vlm.calls == 4 and ocr.calls == 2, "only the two usable images reached the models"
    assert all(image_text.row_problems(r) == [] for r in rows.values())


def test_the_gate_uses_the_real_registry_by_content(monkeypatch):
    """imgur's "image does not exist" bitmap, identified by its sha256 in the
    tracked registry, stops at the gate whatever the index said."""
    known = "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9"
    assert placeholders.is_missing(known)
    monkeypatch.setattr(images, "examine_bytes",
                        lambda body: {"sha256": known, "loads": True, "verifies": True})
    item = WorkItem("fakeddit", known, "data/raw/fakeddit/media/images/x.jpg", ("abc",), 1)
    assert image_text.gate(item, b"...") == ("skipped", "placeholder")


# --------------------------------------------------------------------------
# the row: schema, no path / label / hostname, determinism of the pipeline
# --------------------------------------------------------------------------


def test_rows_validate_and_carry_the_provenance_asked_for(fake_root):
    vlm, ocr, ctx = _ctx(fake_root)
    row = image_text.process(_put(fake_root, "data/raw/verite/p.png", _png()), vlm, ocr, ctx)
    assert image_text.row_problems(row) == []
    p = row["provenance"]
    for key in ("model", "prompt_sha256", "seed", "quantization", "resolution_cap", "machine"):
        assert key in p
    assert p["model"]["revision"] and p["seed"] == image_text.SEED
    assert p["resolution_cap"]["max_pixels"] == image_text.MAX_PIXELS
    assert set(p["machine"]) >= {"host_id", "gpu", "os"}
    assert row["confidence"].keys() == {"description", "entities", "ocr"}
    assert row["ocr"]["regions"][0].keys() == {"text", "confidence", "polygon"}


def _mutations():
    def drop_revision(r):
        del r["provenance"]["model"]["revision"]

    def bad_status(r):
        r["status"] = "great"

    def add_path(r):
        r["entities"]["path"] = "data/raw/verite/images/x.jpg"

    def add_hostname(r):
        r["provenance"]["machine"]["hostname"] = "box"

    def add_label(r):
        r["label"] = "true"

    def sampled(r):
        r["provenance"]["decoding"]["do_sample"] = True

    def unmarked(r):
        del r["entities"]["ORG"]["forced"]

    def mark_disagrees(r):
        r["entities"]["ORG"]["items"][0]["forced"] = not r["entities"]["ORG"]["forced"]

    def cut_but_ok(r):
        r["entities"]["finish"] = "length"

    def every_line_forced_but_ok(r):
        for cat in CATEGORIES:
            r["entities"][cat]["forced"] = True
            for item in r["entities"][cat]["items"]:
                item["forced"] = True

    def an_answer_error_with_no_cause(r):
        r["status"], r["entities"]["error"] = "partial", "truncated"

    def description_cut_but_ok(r):
        r["description"]["finish"] = "length"

    def a_description_error_with_no_cause(r):
        r["status"], r["description"]["error"] = "partial", "truncated"

    return [drop_revision, bad_status, add_path, add_hostname, add_label, sampled, unmarked,
            mark_disagrees, cut_but_ok, every_line_forced_but_ok, an_answer_error_with_no_cause,
            description_cut_but_ok, a_description_error_with_no_cause]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_a_row_that_breaks_the_contract_is_refused(fake_root, mutate):
    vlm, ocr, ctx = _ctx(fake_root)
    row = image_text.process(_put(fake_root, "data/raw/verite/m.png", _png()), vlm, ocr, ctx)
    mutate(row)
    row["result_sha256"] = image_text.result_sha256(row)  # even with a fresh hash
    assert image_text.row_problems(row)
    path = ctx.out_dir / "verite.jsonl"
    ledger = Ledger(path, dataset="verite", config_id=ctx.config_id)
    ledger.open()
    with pytest.raises(FatalError):
        ledger.append(row)
    ledger.close()


def test_the_same_bytes_twice_give_identical_result_bytes(fake_root):
    """The pipeline adds no nondeterminism of its own: with a deterministic
    backend, everything but timings and the machine is byte-identical -- also
    across contexts (a new run id) and fresh backends."""
    item = _put(fake_root, "data/raw/verite/same.png", _png((9, 8, 7), (300, 200)))
    a = image_text.process(item, *_ctx(fake_root)[:3])
    b = image_text.process(item, *_ctx(fake_root)[:3])
    assert a["run"]["run_id"] != b["run"]["run_id"]
    assert image_text.canonical(image_text.result_part(a)) == image_text.canonical(image_text.result_part(b))
    assert a["result_sha256"] == b["result_sha256"]


def test_the_stub_pipeline_never_opens_a_socket(fake_root, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    vlm, ocr, ctx = _ctx(fake_root)
    runner = image_text.Runner(vlm, ocr, ctx)
    runner.open("verite")
    runner.step(_put(fake_root, "data/raw/verite/n.png", _png()))
    runner.close()


# --------------------------------------------------------------------------
# entities: the parser, OCR support, confidence
# --------------------------------------------------------------------------

FIVE = "PERSON: {}\nORG: NONE\nGPE: NONE\nEVENT: NONE\nDATE: NONE"


@pytest.mark.parametrize("raw, status, names, unidentified", [
    (FIVE.format("Narendra Modi [seen]"), "named", [("Narendra Modi", "seen")], False),
    (FIVE.format("Donald Trump [text] ; UNIDENTIFIED"), "named", [("Donald Trump", "text")], True),
    (FIVE.format("UNIDENTIFIED"), "unidentified", [], True),
    (FIVE.format("NONE"), "none", [], False),
    (FIVE.format("Joe Biden [seen], Kamala Harris [seen]"), "named",
     [("Joe Biden", "seen"), ("Kamala Harris", "seen")], False),
    (FIVE.format("Imran Khan"), "named", [("Imran Khan", "unstated")], False),
    # what the model actually wrote on Factify2 train images (2026-10-05):
    (FIVE.format("Donald Trump, Melania Trump, Narendra Modi, UNIDENTIFIED"), "named",
     [("Donald Trump", "unstated"), ("Melania Trump", "unstated"), ("Narendra Modi", "unstated")], True),
    (FIVE.format("Rahul Gandhi [seen]; others UNIDENTIFIED"), "named", [("Rahul Gandhi", "seen")], True),
    (FIVE.format("Shri. A. Pawar [Sakal], Shri. Ravi [Hindu], Prime Minister [seen]"), "named",
     [("Shri. A. Pawar", "unstated"), ("Shri. Ravi", "unstated"), ("Prime Minister", "seen")], False),
    (FIVE.format("Joe Biden, Kamala Harris [seen]"), "named",
     [("Joe Biden", "unstated"), ("Kamala Harris", "seen")], False),
    (FIVE.format("Unknown person"), "unidentified", [], True),
    (FIVE.format("No people are visible in the image"), "none", [], False),
    ("**PERSON:** Rahul Gandhi [seen]\n- ORG: NONE\nGPE: NONE\nEVENT: NONE\nDATE: NONE",
     "named", [("Rahul Gandhi", "seen")], False),
])
def test_person_lines_parse_conservatively(raw, status, names, unidentified):
    out = image_text.parse_entities(raw)
    person = out["categories"]["PERSON"]
    assert person["status"] == status
    assert [(i["name"], i["basis"]) for i in person["items"]] == names
    assert person["unidentified"] is unidentified
    assert out["error"] is None
    for item in person["items"]:
        s, e = item["span"]
        assert raw[s:e] == item["name"], "the span must point at the name in the raw text"


def test_a_missing_line_loses_that_category_only():
    out = image_text.parse_entities("PERSON: NONE\nORG: Reuters [text]\nDATE: 2020 [text]")
    cats = out["categories"]
    assert cats["ORG"]["items"][0]["name"] == "Reuters" and cats["DATE"]["status"] == "named"
    assert cats["GPE"]["status"] == "missing" and cats["EVENT"]["status"] == "missing"
    assert {"category_missing:GPE", "category_missing:EVENT"} <= set(out["flags"])
    assert out["error"] is None


def test_numbered_lines_abbreviations_and_parentheses_survive_parsing():
    raw = ("1. PERSON: NONE.\n2. ORG: Indian National Congress (INC) [text] ; Reuters (text)\n"
           "3. GPE: U.S. [text] ; India.\nEVENT: NONE\nDATE: March 4, 2020 [text]")
    cats = image_text.parse_entities(raw)["categories"]
    assert cats["PERSON"]["status"] == "none"
    assert [(i["name"], i["basis"]) for i in cats["ORG"]["items"]] == [
        ("Indian National Congress (INC)", "text"), ("Reuters", "text")]
    assert [(i["name"], i["basis"]) for i in cats["GPE"]["items"]] == [
        ("U.S.", "text"), ("India", "unstated")]
    assert [i["name"] for i in cats["DATE"]["items"]] == ["March 4, 2020"], "a date keeps its comma"


def test_a_bare_none_answer_is_none_everywhere_and_says_so():
    """What the model wrote for one image with nothing to name: one word, no lines."""
    out = image_text.parse_entities("NONE [seen]")
    assert out["error"] is None and out["flags"] == ["answered_none_for_all"]
    assert all(c["status"] == "none" and not c["items"] for c in out["categories"].values())


def test_garbage_is_unparseable_not_a_list_of_names():
    out = image_text.parse_entities("I'm sorry, I can't help with identifying people.")
    assert out["error"] == "unparseable"
    assert all(c["status"] == "missing" for c in out["categories"].values())


def test_a_name_is_checked_against_the_independent_ocr():
    ocr = "CNN PRoJEcTiON DONALD TRUMP WINS TEXAS 2020"
    assert image_text.name_in_ocr("Donald Trump", ocr)
    assert image_text.name_in_ocr("Texas", "breaking:TEXASNEWS")  # glued by OCR
    assert not image_text.name_in_ocr("Joe Biden", ocr)


def test_confidence_is_the_models_own_token_probability():
    gen = Generation("Joe Biden [seen]", [math.log(0.9), math.log(0.5), math.log(0.99)],
                     [3, 9, 16], "eos", 3)
    assert image_text.span_confidence(gen, 0, 9) == 0.5, "a name is as sure as its weakest token"
    assert image_text.text_confidence(gen) == round((0.9 * 0.5 * 0.99) ** (1 / 3), 6)
    block = image_text.entities_block(Generation(
        FIVE.format("Joe Biden [seen]"), [math.log(0.8)] * 5, [8, 12, 18, 30, 60], "eos", 5),
        ocr_text="nothing relevant")
    item = block["PERSON"]["items"][0]
    assert item["basis"] == "seen" and item["in_ocr"] is False and 0 < item["confidence"] <= 1


def test_greedy_is_verified_token_by_token_not_assumed():
    assert image_text.check_greedy([5, 7, 9], [5, 7, 9]) == []
    with pytest.raises(FatalError, match="not greedy"):
        image_text.check_greedy([5, 8, 9], [5, 7, 9])
    with pytest.raises(FatalError):
        image_text.check_greedy([5, 7], [5, 7, 9])
    # with the skeleton's picks: the steps it replaced come back, anything else still stops
    assert image_text.check_greedy([5, 4, 9], [5, 0, 9], chosen=[5, 4, 9]) == [1]
    with pytest.raises(FatalError, match="not greedy"):
        image_text.check_greedy([5, 3, 9], [5, 0, 9], chosen=[5, 4, 9])


def test_content_gates_on_a_generation():
    assert image_text.description_block(Generation("", [], [], "eos", 0))["error"] == "empty"
    assert image_text.description_block(
        Generation("I'm sorry, I can't describe this.", [0.0], [33], "eos", 1))["error"] == "refusal"
    looped = image_text.description_block(Generation("a dog " * 40, [0.0], [240], "length", 256))
    assert looped["error"] == "truncated", "half a description is not a description"
    assert {"truncated", "repetition"} <= set(looped["flags"]), "the flag stays"
    cut = image_text.description_block(Generation("A photograph of a man in a dark suit at a",
                                                  [0.0], [41], "length", 256))
    assert (cut["error"], cut["flags"]) == ("truncated", ["truncated"])


def test_the_prompts_split_reading_from_recognition():
    """The description names no one; the entity call is where recognition
    happens, and it must be allowed to name a face -- and to say UNIDENTIFIED."""
    d, e = image_text.DESCRIBE_PROMPT, image_text.ENTITIES_PROMPT
    assert "Do not name or identify anyone unless their name is written" in d
    assert "identities are recorded separately" in d
    assert "[seen]" in e and "recognise" in e and "face" in e
    assert "UNIDENTIFIED" in e and "a wrong name is worse than no name" in e


# --------------------------------------------------------------------------
# the skeleton: five lines held while they are generated, forced lines marked
# --------------------------------------------------------------------------


class ToyTokenizer:
    """Just enough of Qwen2.5-VL's tokenizer for the skeleton, with its quirks
    where they matter: "]\\n" is one token (Qwen's 921) and "GPE" is two ("G",
    "PE"). Id 0 ends the turn."""

    VOCAB = ("<|im_end|>", "PERSON", ":", " UN", "IDENT", "IFIED", " [", "seen", "]", "]\n",
             "\n", "ORG", " NONE", "G", "PE", "EVENT", "DATE", "NONE", "ORGAN", " CNN", " Note",
             " Joe", " Biden", " ;")
    END = 0
    EOS = (0,)

    def decode(self, i: int) -> str:
        return "" if i == self.END else self.VOCAB[i]

    def encode(self, text: str) -> list[int]:
        out = []
        while text:
            piece = max((v for v in self.VOCAB[1:] if text.startswith(v)), key=len)
            out.append(self.VOCAB.index(piece))
            text = text[len(piece):]
        return out

    def skeleton(self) -> image_text.Skeleton:
        return image_text.Skeleton(self.decode, self.encode, self.encode("\n")[0], self.END, self.EOS)


class QwenTokenizer(ToyTokenizer):
    """The real tokenizer, from the pinned snapshot on disk -- never the network."""

    def __init__(self):
        try:
            snapshot = image_text.QwenVLM().snapshot()
        except image_text.ModelMissing:
            pytest.skip("the Qwen2.5-VL snapshot is not on disk")
        transformers = pytest.importorskip("transformers")
        self.tok = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        self.END = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.EOS = (self.tok.convert_tokens_to_ids("<|endoftext|>"), self.END)

    def decode(self, i: int) -> str:
        return self.tok.decode([i])

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


@pytest.fixture(scope="module", params=["toy", "qwen"])
def tokenizer(request):
    return ToyTokenizer() if request.param == "toy" else QwenTokenizer()


def _drive(tok, script: list[int], limit: int | None = None) -> tuple[image_text.Skeleton, list[int]]:
    """Feed a skeleton the model's wishes one step at a time, as generate does,
    until it emits an end of turn -- or, like max_new_tokens, ``limit`` steps."""
    skeleton, emitted = tok.skeleton(), []
    for want in script[:limit]:
        emitted.append(skeleton.choose(want))
        if emitted[-1] in skeleton.eos:
            break
    return skeleton, emitted


def _held(tok, script: list[int], logprobs: list[float] | None = None,
          limit: int | None = None) -> Generation:
    """A Generation as QwenVLM.generate builds it, from a skeleton driven by ``script``."""
    skeleton, emitted = _drive(tok, script, limit)
    pieces = ["" if i in skeleton.eos else tok.decode(i) for i in emitted]  # skip_special_tokens
    ends, total = [], 0
    for piece in pieces:
        total += len(piece)
        ends.append(total)
    return Generation("".join(pieces), logprobs or [math.log(0.9)] * len(emitted), ends,
                      "eos" if emitted[-1] in skeleton.eos else "length", len(emitted),
                      tuple(skeleton.replaced), tuple(skeleton.events), skeleton.forced_lines)


def test_an_answer_that_ends_after_person_is_held_to_five_lines(tokenizer):
    """What the model did on 63 of the first pilot's 199 images: the PERSON
    line, "]" where a continuing answer writes "]\\n", then the end of its turn
    -- 9 tokens. Held, that end becomes a newline and the model writes the rest."""
    first = tokenizer.encode("PERSON: UNIDENTIFIED [seen]")
    rest = tokenizer.encode("ORG: NONE\nGPE: NONE\nEVENT: NONE\nDATE: NONE")
    assert len(first) + 1 == 9, "the stored answers were 9 tokens, the end of turn included"
    skeleton, emitted = _drive(tokenizer, first + [tokenizer.END] + rest + [tokenizer.END])
    text = "".join(tokenizer.decode(i) for i in emitted if i not in skeleton.eos)
    assert text == "PERSON: UNIDENTIFIED [seen]\nORG: NONE\nGPE: NONE\nEVENT: NONE\nDATE: NONE"
    assert emitted[:len(first)] == first, "before the first replacement, the model's own tokens"
    assert skeleton.replaced == [len(first)] and skeleton.events == ["continued_after:PERSON"]
    assert skeleton.forced_lines == ("ORG", "GPE", "EVENT", "DATE") and skeleton.complete
    parsed = image_text.parse_entities(text)
    assert parsed["error"] is None
    assert all(c["status"] != "missing" for c in parsed["categories"].values())


def test_the_skeleton_replaces_only_what_breaks_the_form():
    tok = ToyTokenizer()
    full = tok.encode(FIVE.format("NONE"))
    # a complete answer passes untouched, every line free
    skeleton, emitted = _drive(tok, full + [tok.END])
    assert emitted == full + [tok.END] and skeleton.replaced == [] and skeleton.forced_lines == ()
    # an answer that does not start with PERSON's header gets it, and every line is forced
    skeleton, emitted = _drive(tok, tok.encode("NONE: NONE\nORG: NONE\nGPE: NONE\nEVENT: NONE\n"
                                               "DATE: NONE") + [tok.END])
    assert "".join(map(tok.decode, emitted)) == FIVE.format("NONE")
    assert (skeleton.replaced, skeleton.events) == ([0], ["header_forced:PERSON"])
    assert skeleton.forced_lines == CATEGORIES
    # "ORGANIZATION" where ORG's header goes: the header is put in, PERSON stays free
    head = tok.encode("PERSON: NONE\n")
    skeleton, emitted = _drive(tok, head + [tok.VOCAB.index("ORGAN")]
                               + tok.encode(": CNN\nGPE: NONE\nEVENT: NONE\nDATE: NONE") + [tok.END])
    assert "".join(map(tok.decode, emitted)).startswith("PERSON: NONE\nORG: CNN\nGPE: NONE")
    assert (skeleton.replaced, skeleton.events) == ([len(head)], ["header_forced:ORG"])
    assert skeleton.forced_lines == ("ORG", "GPE", "EVENT", "DATE")
    # going on after DATE's line: the answer is ended there, and no line is forced
    skeleton, emitted = _drive(tok, full + tok.encode("\n") + [tok.VOCAB.index(" Note")])
    assert emitted[-1] == tok.END and skeleton.events == ["ended_after:DATE"]
    assert skeleton.forced_lines == ()
    with pytest.raises(FatalError, match="does not end generation"):
        image_text.Skeleton(tok.decode, tok.encode, tok.encode("\n")[0], 5, tok.EOS)


def test_forced_lines_are_marked_on_the_line_and_on_every_entity():
    tok = ToyTokenizer()
    first = tok.encode("PERSON: Joe Biden [seen]")
    script = first + [tok.END] + tok.encode("ORG: CNN\nGPE: NONE\nEVENT: NONE\nDATE: NONE") + [tok.END]
    logprobs = [math.log(0.9)] * len(script)
    logprobs[len(first)] = math.log(1e-6)  # the newline put in where the model would have ended
    block = image_text.entities_block(_held(tok, script, logprobs), ocr_text="")
    assert [c for c in CATEGORIES if block[c]["forced"]] == ["ORG", "GPE", "EVENT", "DATE"]
    (biden,), (cnn,) = block["PERSON"]["items"], block["ORG"]["items"]
    assert (biden["name"], biden["forced"]) == ("Joe Biden", False)
    assert (cnn["name"], cnn["forced"]) == ("CNN", True), "the mark travels with the entity"
    assert "continued_after:PERSON" in block["flags"]
    assert block["confidence"] == 0.9, "a replaced token is not the model's: it leaves confidence"


def _gen(raw: str, finish: str = "eos") -> Generation:
    ends = [m.end() for m in re.finditer(r"\S+\s*", raw)]
    return Generation(raw, [0.0] * len(ends), ends, finish, len(ends))


@pytest.mark.parametrize("raw, finish, form", [
    ("PERSON: UNIDENTIFIED [seen]", "eos", "stopped_after:PERSON"),  # the first pilot's 63
    (FIVE.format("NONE"), "eos", "complete"),
    ("PERSON: NONE\nORG: CNN [text]\nGPE: Tex", "length", "truncated"),
    (FIVE.format("NONE ; NONE ; NONE"), "length", "truncated"),  # every header, cut in DATE's line
    ("PERSON: NONE ORG: NONE GPE: NONE EVENT: NONE DATE: NONE", "eos", "collapsed"),
    ("PERSON: NONE\nORG: NONE\nDATE: NONE", "eos", "lines_missing"),
    ("NONE [seen]", "eos", "none_for_all"),
    ("I'm sorry, I can't help with that.", "eos", "unparseable"),
])
def test_an_entity_answer_is_classified_by_how_it_lost_its_form(raw, finish, form):
    """Truncated, another format, or stopped early: the diagnosis, per answer."""
    assert image_text.entity_format(image_text.entities_block(_gen(raw, finish), None)) == form


def test_a_failed_entity_call_is_failed_not_malformed():
    assert image_text.entity_format(image_text._failed_component("entities", "cuda_oom")) == "failed"


class _Unheld(StubVLM):
    """An entity answer as the model gave it before the skeleton, unconstrained."""

    answer = "PERSON: UNIDENTIFIED [seen]"

    def generate(self, image, which):
        if which != "entities":
            return super().generate(image, which)
        self.calls += 1
        ends = [m.end() for m in re.finditer(r"\S+\s*", self.answer)]
        return Generation(self.answer, [-0.1] * len(ends), ends, "eos", len(ends))


@pytest.mark.parametrize("pilot", [False, True], ids=["run", "pilot"])
@pytest.mark.parametrize("answer, form", [
    ("PERSON: UNIDENTIFIED [seen]", "stopped_after:PERSON"),  # the first pilot's 63, 9 tokens
    ("NONE [seen]", "none_for_all"),                          # its 13, 4.5 tokens on average
], ids=["stopped_after_person", "none_for_all"])
def test_an_entity_answer_that_lost_its_lines_stops_the_run_unwritten(fake_root, tmp_path, monkeypatch,
                                                                      capsys, pilot, answer, form):
    """Whatever produces it -- a prompt, a model, a decoding change -- an answer
    missing lines is never written, as ok or otherwise: the run stops. Under
    the skeleton neither can arise (the next test); this is the guard behind it."""
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(image_text, "StubVLM", _Unheld)
    monkeypatch.setattr(_Unheld, "answer", answer)
    lines = [json.dumps(_put(fake_root, f"data/raw/verite/s{i}.png", _png((i, 6, 6)), rid=str(i)).as_json())
             for i in range(2)]
    worklist = tmp_path / "wl.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    argv = ["run", "--stub", "--worklist", str(worklist)] + (["--pilot", "2"] if pilot else [])
    assert image_text.main(argv) == 2
    err = capsys.readouterr().err
    assert "STOPPED" in err and f"({form})" in err
    (ledger,) = (tmp_path / "out").glob("*/verite.jsonl")
    assert ledger.read_bytes() == b"", "a row whose entity answer lost its lines was written"


class _Held(StubVLM):
    """The stub, with its entity answer generated under the real skeleton from
    a scripted model, max_new_tokens included -- as QwenVLM.generate does."""

    def __init__(self, tok, script):
        super().__init__()
        self.tok, self.script = tok, script

    def generate(self, image, which):
        if which != "entities":
            return super().generate(image, which)
        self.calls += 1
        return _held(self.tok, self.script, limit=image_text.MAX_NEW_TOKENS["entities"])


def _script(tok, case: str) -> list[int]:
    """What the model wants at each step, whatever was emitted before."""
    end = [tok.END]
    if case == "ends_after_person":  # the first pilot's 63
        return (tok.encode("PERSON: UNIDENTIFIED [seen]") + end
                + tok.encode("ORG: NONE\nGPE: NONE\nEVENT: NONE\nDATE: NONE") + end)
    if case == "answers_nothing":  # its 13 bare "NONE"s, and then nothing at every step
        return tok.encode("NONE") + end * 40
    if case == "runs_away_in_a_line":  # its 2 runaways: on inside PERSON's line, past 192 tokens
        return tok.encode("PERSON: Joe Biden [seen]") + tok.encode(" ; Joe Biden [seen]") * 40
    if case == "goes_on_after_date":
        return tok.encode(FIVE.format("NONE") + "\n") + tok.encode(" Note") * 10
    raise ValueError(case)


@pytest.mark.parametrize("case, status, error, form, forced", [
    ("ends_after_person", "ok", None, "complete", ("ORG", "GPE", "EVENT", "DATE")),
    ("answers_nothing", "partial", "no_free_line", "complete", CATEGORIES),
    ("runs_away_in_a_line", "partial", "truncated", "truncated", ()),
    ("goes_on_after_date", "ok", None, "complete", ()),
])
def test_each_way_the_model_lost_its_form_is_written_as_what_it_is(tokenizer, fake_root, case,
                                                                    status, error, form, forced):
    """Through the pipeline, under the real skeleton. An answer the model began
    and stopped early is an answer, its forced lines marked. One it never began
    (every line forced) or never ended (cut at max_new_tokens) is written as an
    entity error -- partial, never ok, no entity out of it -- and the ledger
    refuses it as ok. Going on past DATE's line is ended by the skeleton."""
    vlm = _Held(tokenizer, _script(tokenizer, case))
    _, ocr, ctx = _ctx(fake_root, vlm=vlm)
    row = image_text.process(_put(fake_root, "data/raw/verite/h.png", _png()), vlm, ocr, ctx)
    entities = row["entities"]
    assert (row["status"], entities["error"], image_text.entity_format(entities)) == (status, error, form)
    assert tuple(c for c in CATEGORIES if entities[c]["forced"]) == forced
    assert image_text.row_problems(row) == []
    ledger = Ledger(ctx.out_dir / "verite.jsonl", dataset="verite", config_id=ctx.config_id)
    ledger.open()
    ledger.append(row)
    if case == "answers_nothing":
        assert entities["flags"][:2] == ["empty_value:PERSON", "empty_value:ORG"]
        assert "header_forced:PERSON" in entities["flags"] and entities["n_tokens"] < 30
    if case == "runs_away_in_a_line":
        assert entities["n_tokens"] == 192 and entities["finish"] == "length"
        assert entities["PERSON"]["items"][0]["name"] == "Joe Biden"
        assert {f"category_missing:{c}" for c in CATEGORIES[1:]} <= set(entities["flags"])
    if case == "goes_on_after_date":
        assert "ended_after:DATE" in entities["flags"]
    if error:
        assert list(image_text.entities_of(row, forced="mark")) == []
        as_ok = json.loads(json.dumps(row))
        as_ok["status"], as_ok["entities"]["error"] = "ok", None
        as_ok["result_sha256"] = image_text.result_sha256(as_ok)
        assert any(error in p for p in image_text.row_problems(as_ok))
        with pytest.raises(FatalError, match="invalid row"):
            ledger.append(as_ok)
    ledger.close()


class _CutDescription(StubVLM):
    """The stub, with its description cut at max_new_tokens."""

    def generate(self, image, which):
        gen = super().generate(image, which)
        if which != "description":
            return gen
        return Generation(gen.text, gen.logprobs, gen.ends, "length",
                          image_text.MAX_NEW_TOKENS["description"])


def test_a_cut_description_is_an_error_and_its_row_partial(fake_root):
    """Half a description is not a description: it would feed the consistency
    check and an explanation a person reads. The flag stays, the ledger refuses
    the row as ok, and the entities -- a separate call -- are untouched."""
    vlm = _CutDescription()
    _, ocr, ctx = _ctx(fake_root, vlm=vlm)
    row = image_text.process(_put(fake_root, "data/raw/verite/d.png", _png()), vlm, ocr, ctx)
    description = row["description"]
    assert (row["status"], description["error"], description["flags"]) == ("partial", "truncated",
                                                                          ["truncated"])
    assert image_text.row_problems(row) == []
    assert [e["name"] for e in image_text.entities_of(row, forced="mark")] == ["Stub Labs"]
    as_ok = json.loads(json.dumps(row))
    as_ok["status"], as_ok["description"]["error"] = "ok", None
    as_ok["result_sha256"] = image_text.result_sha256(as_ok)
    assert any(p.startswith("description:") and "truncated" in p for p in image_text.row_problems(as_ok))
    ledger = Ledger(ctx.out_dir / "verite.jsonl", dataset="verite", config_id=ctx.config_id)
    ledger.open()
    ledger.append(row)
    with pytest.raises(FatalError, match="invalid row"):
        ledger.append(as_ok)
    ledger.close()


def test_rows_of_earlier_schemas_still_validate_but_are_never_written(fake_root):
    """A ledger written before lines were marked (1), or before a cut answer
    was an error (2), stays readable -- status --verify checks every ledger --
    and a row of either shape is never appended. A cut answer from 2 gives no
    entities, though it never said so."""
    vlm, ocr, ctx = _ctx(fake_root)
    row = image_text.process(_put(fake_root, "data/raw/verite/old.png", _png()), vlm, ocr, ctx)
    first = json.loads(json.dumps(row))
    first["schema"], first["provenance"]["schema_version"] = "image_text/1", 1
    for cat in CATEGORIES:
        del first["entities"][cat]["forced"]
        for item in first["entities"][cat]["items"]:
            del item["forced"]
    second = json.loads(json.dumps(row))
    second["schema"], second["provenance"]["schema_version"] = "image_text/2", 2
    second["entities"]["finish"] = "length"  # written ok, as 2 wrote a cut answer
    second["description"]["finish"] = "length"  # and a cut description
    ledger = Ledger(ctx.out_dir / "verite.jsonl", dataset="verite", config_id=ctx.config_id)
    ledger.open()
    for old in (first, second):
        old["result_sha256"] = image_text.result_sha256(old)
        assert image_text.row_problems(old) == []
        with pytest.raises(FatalError, match=re.escape(f"only {image_text.SCHEMA} is written")):
            ledger.append(old)
    ledger.close()
    assert image_text.row_problems({**first, "schema": "image_text/9"})[0].startswith("schema:")
    assert (second["status"], list(image_text.entities_of(second, forced="mark"))) == ("ok", [])
    second["entities"]["error"] = "truncated"  # an error 2 did not have
    second["result_sha256"] = image_text.result_sha256(second)
    assert image_text.row_problems(second)


def test_entities_leave_a_row_only_with_their_line_marked(fake_root):
    """The sanctioned way out of a row: the caller must choose what forced
    lines mean to it, and every entity it gets says which kind it is."""
    vlm, ocr, ctx = _ctx(fake_root)
    row = image_text.process(_put(fake_root, "data/raw/verite/e.png", _png()), vlm, ocr, ctx)
    org = row["entities"]["ORG"]  # the stub names "Stub Labs" there; make it a forced line
    org["forced"] = org["items"][0]["forced"] = True
    with pytest.raises(TypeError):
        image_text.entities_of(row)  # no default
    with pytest.raises(ValueError):
        image_text.entities_of(row, forced="keep")
    assert list(image_text.entities_of(row, forced="exclude")) == []
    (entity,) = image_text.entities_of(row, forced="mark")
    assert (entity["category"], entity["name"], entity["forced"]) == ("ORG", "Stub Labs", True)
    assert set(entity) >= {"dataset", "sha256", "record_ids", "basis", "in_ocr", "confidence"}
    for cat in CATEGORIES:  # a row from before the marks cannot tell forced from free
        del row["entities"][cat]["forced"]
    with pytest.raises(SupervisionError):
        image_text.entities_of(row, forced="mark")


def test_entity_statistics_are_split_by_forced_and_free(fake_root):
    vlm, ocr, ctx = _ctx(fake_root)
    free = image_text.process(_put(fake_root, "data/raw/verite/f1.png", _png((1, 2, 3))), vlm, ocr, ctx)
    held = image_text.process(_put(fake_root, "data/raw/verite/f2.png", _png((3, 2, 1))), vlm, ocr, ctx)
    for cat in ("ORG", "GPE", "EVENT", "DATE"):  # as if the model had ended after PERSON
        held["entities"][cat]["forced"] = True
        for item in held["entities"][cat]["items"]:
            item["forced"] = True
    cut = image_text.process(_put(fake_root, "data/raw/verite/f3.png", _png((2, 2, 2))), vlm, ocr, ctx)
    cut["status"], cut["entities"]["finish"], cut["entities"]["error"] = "partial", "length", "truncated"
    q = image_text.quality([free, held, cut])
    by = q["entities"]
    assert by["free"]["lines"] == {"PERSON": 2, "ORG": 1, "GPE": 1, "EVENT": 1, "DATE": 1}, \
        "a cut answer's lines are in neither count"
    assert by["forced"]["lines"] == {"ORG": 1, "GPE": 1, "EVENT": 1, "DATE": 1}
    assert by["free"]["names_tag_vs_ocr"] == by["forced"]["names_tag_vs_ocr"] == {"text:written": 1}
    assert q["entities_complete_free"] == 1
    assert q["entity_format"] == {"complete": 2, "truncated": 1}
    assert q["component_errors"] == {"entities:truncated": 1}


# --------------------------------------------------------------------------
# the resolution cap, the order, the sample
# --------------------------------------------------------------------------


def test_the_cap_is_qwens_resize_under_constants_we_own():
    vl = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    sizes = (1, 13, 28, 55, 100, 333, 640, 1000, 1920, 4032, 9000)
    for w in sizes:
        for h in sizes:
            if max(w, h) / min(w, h) > image_text.MAX_ASPECT:
                with pytest.raises(image_text.GeometryError):
                    image_text.fit_to_cap(w, h)
                continue
            fw, fh = image_text.fit_to_cap(w, h)
            assert (fh, fw) == vl.smart_resize(h, w, factor=28, min_pixels=image_text.MIN_PIXELS,
                                                max_pixels=image_text.MAX_PIXELS)
            assert fw % 28 == 0 and fh % 28 == 0 and fw * fh <= image_text.MAX_PIXELS
    assert image_text.visual_tokens(4032, 3024) == image_text.visual_tokens(9000, 6750) <= 1024


def test_the_order_is_fixed_and_nothing_runs_past_verite_without_a_flag(tmp_path, monkeypatch):
    assert image_text.ORDER == ("verite", "averimatec", "factify2", "fakeddit")
    args = image_text.build_parser().parse_args(["run"])
    assert image_text.scope_for(args.through) == ("verite",)
    assert image_text.scope_for("factify2") == ("verite", "averimatec", "factify2")
    with pytest.raises(SystemExit):
        image_text.main(["run", "--through", "fakeddit"])  # the sample size is a decision
    with pytest.raises(SystemExit):
        image_text.main(["run", "--fakeddit-n", "10"])
    asked = []
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(image_text, "build_worklist",
                        lambda ds, **kw: (asked.append(ds), ([], {}))[1])
    assert image_text.main(["run", "--stub"]) == 0
    assert asked == ["verite"]


def test_every_fakeddit_prefix_is_proportional_and_nested():
    sizes = {"0": 590, "1": 89, "2": 285, "3": 31, "4": 448, "5": 57}
    items = [WorkItem("fakeddit", _sha(f"{c}-{i}".encode()), f"x/{c}/{i}", (f"{c}{i}",), 1,
                      stratum=c) for c, n in sizes.items() for i in range(n)]
    order = image_text.nested_stratified(items)
    assert sorted(it.sha256 for it in order) == sorted(it.sha256 for it in items)
    total, counts = len(items), Counter()
    for n, item in enumerate(order, 1):
        counts[item.stratum] += 1
        for c, size in sizes.items():
            assert abs(counts[c] - n * size / total) <= 1, (n, c)
    assert image_text.nested_stratified(list(reversed(items))) == order, "input order leaked"


def test_the_probe_takes_train_claims_naming_one_pictured_figure(tmp_path):
    def row(rid, split, text, path):
        return {"record_id": rid, "split": split, "text": text, "image_paths": [path], "usable": True}

    paths = {f"data/raw/factify2/media/images/{n}.jpg": _sha(n.encode()) for n in "abcdef"}
    cache = {"factify2": {p: {"sha256": s, "width": 100, "height": 100} for p, s in paths.items()}}
    full = [WorkItem("factify2", s, p, ("r",), 1) for p, s in paths.items()]
    p = list(paths)
    rows = [row("1", "train", "Photo shows Narendra Modi at a rally", p[0]),
            row("2", "train", "Video of Donald Trump dancing", p[1]),
            row("3", "test", "Photo of Joe Biden asleep", p[2]),           # not train
            row("4", "train", "Modi and Trump met in this photo", p[3]),   # two figures
            row("5", "train", "Modi said the economy will grow", p[4]),     # nothing pictured
            row("6", "train", "Image shows Modi with a cow", p[5])]
    probe = image_text.probe_items(10, full, rows=rows, cache=cache)
    assert {(it.sha256, it.probe) for it in probe} == {
        (paths[p[0]], "Narendra Modi"), (paths[p[1]], "Donald Trump"), (paths[p[5]], "Narendra Modi")}
    assert probe[0].probe == "Narendra Modi" and probe[1].probe == "Donald Trump", "round-robin"


def _probe_row(person_items, ocr_text, forced=False, finish="eos"):
    """A probe image's row: a whole entity answer naming ``person_items``. A
    forced PERSON line means every line was forced; ``length`` cuts it."""
    rest = {c: {"status": "none", "forced": forced, "items": []} for c in CATEGORIES[1:]}
    return {"status": "ok", "ocr": {"text": ocr_text}, "entities": {
        "raw": "(probe)", "flags": [], "finish": finish, **rest,
        "PERSON": {"status": "named" if person_items else "unidentified", "forced": forced,
                   "items": person_items}}}


def _probe_name(text, basis, in_ocr, confidence=0.9):
    return {"name": text, "basis": basis, "in_ocr": in_ocr, "confidence": confidence}


def test_the_probe_verdict_says_near_zero_out_loud():
    row, name = _probe_row, _probe_name
    probe = [WorkItem("factify2", str(i) * 64, "x", ("r",), 1, probe="Narendra Modi") for i in range(4)]
    blank = {it.sha256: row([], "") for it in probe}
    out = image_text.probe_summary(probe, blank)
    assert out["near_zero"] and out["verdict"].startswith("NEAR ZERO")

    # The model's tag is NOT the evidence: a written name tagged [seen] is reading.
    tagged_only = dict(blank)
    tagged_only[probe[0].sha256] = row([name("Narendra Modi", "seen", True)], "PM MODI SAYS")
    out = image_text.probe_summary(probe, tagged_only)
    assert out["near_zero"], "a [seen] tag on a written name is not recognition"
    assert out["person_tag_vs_ocr"] == {"seen:written": 1}

    named = dict(blank)
    named[probe[0].sha256] = row([name("Narendra Modi", "seen", False)], "")
    named[probe[1].sha256] = row([name("Narendra Modi", "text", True)], "MODI SAYS")
    named[probe[2].sha256] = row([name("Amit Shah", "seen", False, 0.3)], "")
    out = image_text.probe_summary(probe, named)
    assert not out["near_zero"] and out["verdict"].startswith("PRECISION 50.0%")
    c = out["counts"]
    assert (c["name_written"], c["name_not_written"]) == (1, 3)
    assert c["name_not_written_names_claimed_figure"] == 1
    assert c["name_not_written_names_someone"] == 2
    assert (c["recognised_names_claimed_figure"], c["recognised_names_someone_else"]) == (1, 1)
    assert out["recognised_name_confidence"]["someone_else"]["max"] == 0.3


def test_the_probe_reports_precision_and_names_both_thresholds():
    """Shaped like the first pilot's probe: 71 names produced from appearance
    alone, 23 of them the claimed figure, and a wrong name as sure as 0.989."""
    right = [0.95] * 12 + [0.6] * 11
    wrong = [0.989] + [0.5] * 47
    probe = [WorkItem("factify2", f"{i:064x}", "x", ("r",), 1, probe="Narendra Modi") for i in range(71)]
    rows = {it.sha256: _probe_row([_probe_name("Narendra Modi" if n < 23 else "Amit Shah", "seen",
                                               False, right[n] if n < 23 else wrong[n - 23])], "")
            for n, it in enumerate(probe)}
    out = image_text.probe_summary(probe, rows)
    p = out["precision"]
    assert (p["names"], p["claimed_figure"], p["someone_else"], p["value"]) == (71, 23, 48, 0.3239)
    assert p["not_fully_written"]["value"] == 0.3239
    assert [r["names"] for r in p["by_confidence"]] == [71, 13, 13, 13, 0]
    assert out["verdict"].startswith("PRECISION 32.4%: of 71 PERSON names produced from appearance")
    assert "recall-style" in out["verdict"] and "recall-style" in out["rates_are"]
    verifier, candidate = out["thresholds"]["verifier"], out["thresholds"]["candidate"]
    assert (verifier["precision_needed"], verifier["met"]) == (0.9, False)
    assert candidate["max_gallery_false_accept_ratio"] == 0.0532
    assert candidate["precision_needed_at_ratio"] == {"0.1": 0.4737, "0.05": 0.3103,
                                                      "0.01": 0.0826, "0.001": 0.0089}
    assert "replace a gallery" in out["conclusion"] and "0.053 times as often" in out["conclusion"]
    assert (out["entity_verification"], out["vlm_role"]) == ("external_gallery", "candidate")
    assert out["recognised_name_confidence"]["someone_else"]["max"] == 0.989


def test_the_candidate_bar_follows_from_the_gallerys_error_rates():
    need, ratio = image_text.candidate_precision_needed, image_text.max_false_accept_ratio
    assert (round(need(0.1), 4), round(need(0.01), 4)) == (0.4737, 0.0826)
    assert round(ratio(23 / 71), 4) == 0.0532 and ratio(1.0) is None and ratio(0.0) == 0.0
    for p in (0.05, 23 / 71, 0.6):
        assert math.isclose(need(ratio(p)), p), "one rule, read both ways"
    # at exactly that ratio, the names a gallery accepts are right VERIFIER_PRECISION of the time
    p, tar = 23 / 71, 0.9
    fmr = ratio(p) * tar
    assert math.isclose(p * tar / (p * tar + (1 - p) * fmr), image_text.VERIFIER_PRECISION)


def test_a_name_partly_in_the_ocr_was_read_and_a_forced_person_line_stands_apart():
    probe = [WorkItem("factify2", str(i) * 64, "x", ("r",), 1, probe="Narendra Modi") for i in range(3)]
    rows = {probe[0].sha256: _probe_row([_probe_name("Narendra Modi", "seen", False)],
                                        "PM MODI ADDRESSES THE NATION"),
            probe[1].sha256: _probe_row([_probe_name("Amit Shah", "seen", False)], ""),
            probe[2].sha256: _probe_row([_probe_name("Narendra Modi", "seen", False)], "", forced=True)}
    out = image_text.probe_summary(probe, rows)
    assert out["person_tag_vs_ocr"] == {"seen:not_written": 1, "seen:partly_written": 1}
    p = out["precision"]
    assert (p["names"], p["claimed_figure"]) == (1, 0), "a name partly in the image may have been read"
    assert (p["not_fully_written"]["names"], p["not_fully_written"]["claimed_figure"]) == (2, 1)
    assert out["counts"]["images"] == 2, "a forced PERSON line is not in the headline"
    assert out["forced_person_lines"] == {"claimed_figure": 1, "images": 1, "names_from_appearance": 1}


def test_a_cut_answer_is_no_evidence_for_the_probe():
    """Its PERSON line may be the line that ran away: counted, and nothing more."""
    probe = [WorkItem("factify2", str(i) * 64, "x", ("r",), 1, probe="Narendra Modi") for i in range(2)]
    name = _probe_name("Narendra Modi", "seen", False)
    rows = {probe[0].sha256: _probe_row([name], ""),
            probe[1].sha256: _probe_row([name], "", finish="length")}
    out = image_text.probe_summary(probe, rows)
    assert (out["counts"]["images"], out["counts"]["entity_answer_truncated"]) == (1, 1)
    assert (out["precision"]["names"], out["precision"]["claimed_figure"]) == (1, 1)
    assert out["forced_person_lines"] == {}


def test_a_rate_over_nothing_is_null_not_a_division():
    assert image_text._ratio(3, 0, 4) is None and image_text._ratio(0, 0) is None
    assert image_text._ratio(1, 3, 4) == 0.3333 and image_text._ratio(1, 4) == 0.25
    # the candidate bar's denominators: only a target of 1 with a ratio of 0, and a target of 0
    assert image_text.candidate_precision_needed(0.0, target=1.0) is None
    assert image_text.max_false_accept_ratio(0.5, target=0.0) is None


def test_a_stub_pilot_stops_after_n_and_reports_what_was_asked(fake_root, tmp_path, monkeypatch):
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    lines = [json.dumps({**_put(fake_root, f"data/raw/verite/q{i}.png", _png((i, 3, 3), (80 + i, 60)),
                                rid=str(i)).as_json()}) for i in range(6)]
    worklist = tmp_path / "wl.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # a delay, so every image takes measurable time on any machine (the next test: none)
    argv = ["run", "--stub", "--stub-delay", "0.002", "--worklist", str(worklist), "--pilot", "3"]
    assert image_text.main(argv) == 0
    out_dir = next((tmp_path / "out").iterdir())
    assert len((out_dir / "verite.jsonl").read_bytes().splitlines()) == 3, "the pilot went past N"
    report = json.loads((out_dir / "pilot.json").read_text(encoding="utf-8"))
    assert report["pilot"]["processed"] == 3
    assert report["determinism"] == {**report["determinism"], "checked": 3, "identical": 3}
    assert report["throughput"]["images_per_second"] > 0
    assert report["projection"]["datasets"]["verite"]["remaining"] == 3
    assert {"vram", "quality", "count_corrections", "recognition_probe"} <= set(report)
    text = json.dumps(report)
    assert "data/raw" not in text and "q0.png" not in text
    host = socket.gethostname()
    assert len(host) < 4 or host not in text


class _StoppedClock:
    """``time`` as image_text sees it, with perf_counter stopped: every duration
    is 0.0, as the stub's are on a fast machine once rounded to the millisecond."""

    def __getattr__(self, name):
        return getattr(time, name)

    @staticmethod
    def perf_counter() -> float:
        return 1000.0


def test_a_pilot_whose_images_took_no_time_reports_null_rates(fake_root, tmp_path, monkeypatch):
    """On the server the stub finished every image in 0.0 s, and the report
    divided by their sum (ZeroDivisionError). A rate or a fit over nothing is
    null; what has no denominator -- a mean, a projection -- is still 0.0."""
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(image_text, "time", _StoppedClock())
    lines = [json.dumps(_put(fake_root, f"data/raw/verite/z{i}.png", _png((i, 5, 5), (56 * (i + 1), 56)),
                             rid=str(i)).as_json()) for i in range(3)]  # 4, 8, 12 visual tokens
    worklist = tmp_path / "wl.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert image_text.main(["run", "--stub", "--worklist", str(worklist), "--pilot", "3"]) == 0
    (out_dir,) = (tmp_path / "out").iterdir()
    rows = [json.loads(line) for line in (out_dir / "verite.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {s for r in rows for s in r["run"]["seconds"].values()} == {0.0}, "the server's condition"
    report = json.loads((out_dir / "pilot.json").read_text(encoding="utf-8"))
    t = report["throughput"]
    assert t["images_per_second"] is None and t["seconds_per_image"]["mean"] == 0.0
    assert t["fit_seconds_on_visual_tokens"] == {"a": 0.0, "b": 0.0, "r2": None, "n": 3}
    assert report["projection"]["datasets"]["verite"]["hours_naive"] == 0.0
    # every image the same size in visual tokens: no slope to fit, so no fit
    same = [{**r, "model_input": {**r["model_input"], "visual_tokens": 4}} for r in rows]
    assert image_text._fit(same) is None


def test_recheck_and_status_read_what_a_run_wrote(fake_root, tmp_path, monkeypatch, capsys):
    """recheck re-runs finished rows in a new context and byte-compares them;
    status counts a ledger without a model."""
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    lines = [json.dumps(_put(fake_root, f"data/raw/verite/r{i}.png", _png((i, 4, 4)), rid=str(i)).as_json())
             for i in range(4)]
    worklist = tmp_path / "wl.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert image_text.main(["run", "--stub", "--worklist", str(worklist)]) == 0
    capsys.readouterr()
    assert image_text.main(["recheck", "--stub", "--worklist", str(worklist), "--n", "2"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["identical"] == 2
    assert image_text.main(["status", "--verify"]) == 0
    status = json.loads(capsys.readouterr().out)
    (entry,) = status.values()
    assert entry["verite"]["images"] == 4 and entry["verite"]["status"] == {"ok": 4}


# --------------------------------------------------------------------------
# real data: identity, not counts (skipped when the layer is not on disk)
# --------------------------------------------------------------------------


def _have_processed(name: str) -> bool:
    return (PROCESSED / "records" / f"{name}.jsonl.gz").is_file() and images.index_path(name).is_file()


@pytest.mark.skipif(not _have_processed("verite"), reason="the processed layer is not on disk")
def test_the_verite_work_list_is_its_usable_images_by_identity():
    """The set of sha256, computed independently from the processed rows and
    each path's OWN index (two VERITE rows point at a MOCHEG file)."""
    expected, rows_of = set(), {}
    indexes = {}
    with gzip.open(PROCESSED / "records" / "verite.jsonl.gz", "rt", encoding="utf-8") as h:
        for line in h:
            row = json.loads(line)
            if not row["usable"]:
                continue
            for path in row["image_paths"]:
                owner = path.split("/")[2]
                index = indexes.setdefault(owner, images.load_index(owner))
                expected.add(index[path]["sha256"])
                rows_of.setdefault(index[path]["sha256"], set()).add(row["record_id"])
    items, stats = image_text.build_worklist("verite")
    assert {it.sha256 for it in items} == expected
    assert all(set(it.record_ids) == rows_of[it.sha256] for it in items)
    assert stats == {"usable_rows_with_images": 914, "image_paths": 607, "distinct_images": 607}
    assert [it.sha256 for it in items] == sorted(it.sha256 for it in items)


# --------------------------------------------------------------------------
# the real model (opt-in: PE_FAKE_NEWS_GPU_TESTS=1)
# --------------------------------------------------------------------------

_GPU_ENV = "PE_FAKE_NEWS_GPU_TESTS"
gpu = pytest.mark.skipif(not os.environ.get(_GPU_ENV),
                         reason=f"set {_GPU_ENV}=1 to run Qwen2.5-VL and EasyOCR")

#: Factify2 TRAIN, a CNN 2020 election-night graphic: a portrait beside
#: "CNN PROJECTION / DONALD TRUMP WINS / TEXAS" and a CNN 2020 logo.
CNN_TEXAS = "156df5eea835dcbd94e53fa22fe56ae05e68cb754ff29c9614b420fc1eb7d03a"
#: Factify2 TRAIN, the World Health Organization's sign in front of its
#: headquarters: the emblem and "World Health Organization", no people.
WHO_SIGN = "e798c15cf72705c86ccfd9a307e94940b3ce19751ada28fe7f1b3bac06d53ca8"


def _factify2(sha: str) -> WorkItem:
    index = images.load_index("factify2")
    path = next((p for p, r in sorted(index.items()) if r.get("sha256") == sha), None)
    if path is None:
        pytest.skip("the known Factify2 image is not on disk")
    return WorkItem("factify2", sha, path, ("known",), 1)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not os.environ.get(_GPU_ENV):
        pytest.skip(f"set {_GPU_ENV}=1")
    vlm, ocr = image_text.QwenVLM(), image_text.EasyOcrBackend("cuda")
    try:
        vlm.prepare()
        ocr.prepare()
    except image_text.ModelMissing as exc:
        pytest.skip(str(exc))
    vlm.load()
    ocr.load()
    return vlm, ocr, image_text.make_context(vlm, ocr, out_root=tmp_path_factory.mktemp("out"))


def _names(row, cat):
    return [(i["name"].casefold(), i["basis"]) for i in row["entities"][cat]["items"]]


def _five_lines(row):
    """All five lines, and a line marked forced only where a replacement says
    so; an answer, not an entity error -- not cut, and not forced throughout."""
    entities = row["entities"]
    assert image_text.entity_format(entities) == "complete", entities["raw"]
    assert image_text.entity_answer_error(entities) is None and row["status"] == "ok", entities["flags"]
    replaced = [f for f in entities["flags"] if f.split(":")[0] in image_text.SKELETON_EVENTS]
    assert replaced or not any(entities[c]["forced"] for c in CATEGORIES), entities["flags"]


@gpu
def test_the_same_image_twice_is_byte_identical(real):
    """A, then B, then A again: nothing carried over from B changes A's bytes."""
    a, b = _factify2(CNN_TEXAS), _factify2(WHO_SIGN)
    first = image_text.process(a, *real)
    image_text.process(b, *real)
    again = image_text.process(a, *real)
    assert first["status"] == "ok", first
    assert image_text.canonical(image_text.result_part(first)) == \
        image_text.canonical(image_text.result_part(again))


@gpu
def test_a_cnn_projection_graphic_is_read_and_named(real):
    row = image_text.process(_factify2(CNN_TEXAS), *real)
    assert row["status"] == "ok", row
    _five_lines(row)
    ocr = " ".join(r["text"] for r in row["ocr"]["regions"]).upper()
    assert "TRUMP" in ocr and "TEXAS" in ocr
    assert all(0 <= r["confidence"] <= 1 for r in row["ocr"]["regions"])
    person = row["entities"]["PERSON"]
    assert person["status"] == "named"
    trump = [i for i in person["items"] if "trump" in i["name"].casefold()]
    # The name is printed in the image, and the independent OCR says so. The
    # model's own tag is not asserted: it says [seen] for printed names too.
    assert trump and trump[0]["in_ocr"] is True and 0 < trump[0]["confidence"] <= 1
    assert any("texas" in n for n, _ in _names(row, "GPE"))
    assert any("cnn" in n for n, _ in _names(row, "ORG"))
    assert row["model_input"]["visual_tokens"] <= 1024


@gpu
def test_the_who_sign_names_the_organisation_and_no_person(real):
    row = image_text.process(_factify2(WHO_SIGN), *real)
    assert row["status"] == "ok", row
    _five_lines(row)
    who = [i for i in row["entities"]["ORG"]["items"]
           if "world health organization" in i["name"].casefold()]
    assert who and who[0]["in_ocr"] is True
    assert row["entities"]["PERSON"]["items"] == [], "no one is in this picture"
    assert "World Health" in row["ocr"]["text"]
