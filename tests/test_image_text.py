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
    FatalError, Generation, Ledger, LedgerError, LockHeld, StubOCR, StubVLM, WorkItem,
)

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

    return [drop_revision, bad_status, add_path, add_hostname, add_label, sampled]


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
    image_text.check_greedy([5, 7, 9], [5, 7, 9])
    with pytest.raises(FatalError, match="not greedy"):
        image_text.check_greedy([5, 8, 9], [5, 7, 9])
    with pytest.raises(FatalError):
        image_text.check_greedy([5, 7], [5, 7, 9])


def test_content_gates_on_a_generation():
    assert image_text.description_block(Generation("", [], [], "eos", 0))["error"] == "empty"
    assert image_text.description_block(
        Generation("I'm sorry, I can't describe this.", [0.0], [33], "eos", 1))["error"] == "refusal"
    looped = image_text.description_block(Generation("a dog " * 40, [0.0], [240], "length", 256))
    assert looped["error"] is None and {"truncated", "repetition"} <= set(looped["flags"])


def test_the_prompts_split_reading_from_recognition():
    """The description names no one; the entity call is where recognition
    happens, and it must be allowed to name a face -- and to say UNIDENTIFIED."""
    d, e = image_text.DESCRIBE_PROMPT, image_text.ENTITIES_PROMPT
    assert "Do not name or identify anyone unless their name is written" in d
    assert "identities are recorded separately" in d
    assert "[seen]" in e and "recognise" in e and "face" in e
    assert "UNIDENTIFIED" in e and "a wrong name is worse than no name" in e


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


def test_the_probe_verdict_says_near_zero_out_loud():
    def row(person_items, ocr_text):
        return {"status": "ok", "ocr": {"text": ocr_text}, "entities": {"PERSON": {
            "status": "named" if person_items else "unidentified", "items": person_items}}}

    def name(text, basis, in_ocr, confidence=0.9):
        return {"name": text, "basis": basis, "in_ocr": in_ocr, "confidence": confidence}

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
    assert not out["near_zero"] and out["verdict"].startswith("non-trivial")
    c = out["counts"]
    assert (c["name_written"], c["name_not_written"]) == (1, 3)
    assert c["name_not_written_names_claimed_figure"] == 1
    assert c["name_not_written_names_someone"] == 2
    assert (c["recognised_names_claimed_figure"], c["recognised_names_someone_else"]) == (1, 1)
    assert out["recognised_name_confidence"]["someone_else"]["max"] == 0.3


def test_a_stub_pilot_stops_after_n_and_reports_what_was_asked(fake_root, tmp_path, monkeypatch):
    monkeypatch.setattr(image_text, "OUT_ROOT", tmp_path / "out")
    lines = [json.dumps({**_put(fake_root, f"data/raw/verite/q{i}.png", _png((i, 3, 3), (80 + i, 60)),
                                rid=str(i)).as_json()}) for i in range(6)]
    worklist = tmp_path / "wl.jsonl"
    worklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert image_text.main(["run", "--stub", "--worklist", str(worklist), "--pilot", "3"]) == 0
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
    who = [i for i in row["entities"]["ORG"]["items"]
           if "world health organization" in i["name"].casefold()]
    assert who and who[0]["in_ocr"] is True
    assert row["entities"]["PERSON"]["items"] == [], "no one is in this picture"
    assert "World Health" in row["ocr"]["text"]
