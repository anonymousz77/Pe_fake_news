"""The entity gallery: names, counts, and the rules that keep them honest.

Offline and data-free. The image_text ledgers here are written by image_text's
own pipeline with a stub model that names whom the test says, so every row is
a real current-schema row; claim NER is a fake. No test opens a socket.

One opt-in group (``PE_FAKE_NEWS_GALLERY_MODELS=1``) loads the pinned weights
from the project's model cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs import paths  # noqa: E402
from scripts import gallery, image_text, images, placeholders, prepush_check  # noqa: E402
from scripts.image_text import CATEGORIES, Generation, StubOCR, StubVLM, WorkItem  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a gallery test opened a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _png(colour, size) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,entity_type,key", [
    ("PM Modi's", "PERSON", "modi"),
    ("Prime Minister Narendra Modi", "PERSON", "narendra modi"),
    ("Union Home Minister Amit Shah", "PERSON", "amit shah"),
    ("former President Barack Obama", "PERSON", "barack obama"),
    ("Dr. Anthony Fauci", "PERSON", "anthony fauci"),
    ("Modi ji", "PERSON", "modi"),
    ("Recep Tayyip Erdoğan", "PERSON", "recep tayyip erdogan"),
    ("General Motors", "ORG", "general motors"),
    ("The New York Times", "ORG", "new york times"),
    ("“BJP”", "ORG", "bjp"),
])
def test_names_are_keyed_without_titles_possessives_or_accents(raw, entity_type, key):
    assert gallery.name_key(raw, entity_type) == key


@pytest.mark.parametrize("raw", ["UNIDENTIFIED", "##ji", "A", "1999", "people", "President",
                                 "one two three four five six seven eight nine"])
def test_what_names_no_one_has_no_key(raw):
    assert gallery.name_key(raw, "PERSON") is None


def test_claim_text_is_unescaped_and_loses_its_urls():
    text, cut = gallery.clean_claim("Modi &amp;amp; Shah https://t.co/abc  at  www.x.com/y rally")
    assert text == "Modi & Shah at rally" and not cut
    long_text, cut = gallery.clean_claim("x" * (gallery.NER_MAX_CHARS + 10))
    assert cut and len(long_text) == gallery.NER_MAX_CHARS


# --------------------------------------------------------------------------
# the VLM's names, read from a real current-schema ledger
# --------------------------------------------------------------------------


class NamingVLM(StubVLM):
    """The stub, naming whom the test says, keyed by image size."""

    def __init__(self, answers, forced=None):
        super().__init__()
        self.answers, self.forced = answers, forced or {}

    def generate(self, image, which):
        if which != "entities":
            return super().generate(image, which)
        text = self.answers[image.size]
        ends = [m.end() for m in re.finditer(r"\S+\s*", text)]
        return Generation(text, [-0.05] * len(ends), ends, "eos", len(ends),
                          forced_lines=tuple(self.forced.get(image.size, ())))


class TextOCR(StubOCR):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def read(self, image):
        self.calls += 1
        text = self.texts.get(image.size, "")
        regions = [{"text": text, "confidence": 0.9,
                    "polygon": [[0, 0], [9, 0], [9, 9], [0, 9]]}] if text else []
        return regions, image.size


def _answer(person="NONE", org="NONE"):
    return f"PERSON: {person}\nORG: {org}\nGPE: NONE\nEVENT: NONE\nDATE: NONE"


@pytest.fixture
def ledger_root(tmp_path, monkeypatch):
    monkeypatch.setattr(image_text, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(placeholders, "registry", lambda: {})
    monkeypatch.setattr(images, "fingerprint_rejections", lambda: {})
    return tmp_path


def _write_run(root: Path, specs, *, dataset="factify2"):
    """specs: [(size, entity answer, ocr text, forced lines, record_ids)] -> run dir."""
    answers = {size: ans for size, ans, _o, _f, _r in specs}
    forced = {size: f for size, _a, _o, f, _r in specs}
    vlm, ocr = NamingVLM(answers, forced), TextOCR({size: o for size, _a, o, _f, _r in specs})
    ctx = image_text.make_context(vlm, ocr, out_root=root / "image_text")
    image_text.write_config(ctx)
    ledger = image_text.Ledger(ctx.out_dir / f"{dataset}.jsonl", dataset=dataset,
                               config_id=ctx.config_id)
    ledger.open()
    try:
        for i, (size, _a, _o, _f, rids) in enumerate(specs):
            body = _png((i * 40 % 256, 90, 150), size)
            rel = f"data/raw/{dataset}/media/images/{i}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(body)
            item = WorkItem(dataset, hashlib.sha256(body).hexdigest(), rel, tuple(rids), 1)
            ledger.append(image_text.process(item, vlm, ocr, ctx))
    finally:
        ledger.close()
    return ctx.out_dir


#: Sizes are multiples of 28 inside the resolution cap, so the model sees each
#: image at its own size and the stub can tell them apart.
SPECS = [
    ((112, 84), _answer("Narendra Modi [seen] ; Amit Shah [seen]", "BJP [text]"), "BJP RALLY", (), ["1"]),
    ((140, 84), _answer("PM Modi [seen]", "Congress [seen]"), "", ("ORG", "GPE", "EVENT", "DATE"), ["2"]),
    ((168, 84), _answer("UNIDENTIFIED"), "", (), ["3"]),
]


def _fake_ner(spans_by_text):
    calls = []

    def ner(texts):
        calls.append(list(texts))
        return [spans_by_text.get(t, []) for t in texts]

    ner.calls = calls
    return ner


def _span(text, entity_type):
    return {"type": entity_type, "text": text, "start": 0, "end": len(text), "score": 0.99}


CLAIMS = [("r1", "Photo shows Narendra Modi with Rahul Gandhi", False), ("r2", "BJP poster", False)]
CLAIM_SPANS = {CLAIMS[0][1]: [_span("Narendra Modi", "PERSON"), _span("Rahul Gandhi", "PERSON")],
               CLAIMS[1][1]: [_span("BJP", "ORG")]}


def _count(run_dir, tmp_path, ner=None):
    return gallery.count(run_dir, claims=["factify2"], ner=ner or _fake_ner(CLAIM_SPANS),
                         claim_source=lambda _ds: list(CLAIMS), ner_dir=tmp_path / "ner")


def test_count_reads_the_vlm_names_with_their_line_and_their_ocr_evidence(ledger_root, tmp_path):
    table, report = _count(_write_run(ledger_root, SPECS), tmp_path)
    rows = {(r["type"], r["key"]): r for r in table.serialised()}
    assert set(rows) == {("PERSON", "narendra modi"), ("PERSON", "amit shah"), ("PERSON", "modi"),
                         ("PERSON", "rahul gandhi"), ("ORG", "bjp"), ("ORG", "congress")}
    modi = rows[("PERSON", "narendra modi")]
    assert modi["mentions"] == {"vlm": 1, "claims": 1}
    assert modi["vlm"]["factify2"]["evidence"] == {"not_written": 1}
    assert rows[("ORG", "bjp")]["vlm"]["factify2"]["evidence"] == {"written": 1}
    congress = rows[("ORG", "congress")]["vlm"]["factify2"]
    assert congress["forced"] == 1 and congress["free"] == 0, "a forced line stays marked"

    person, org = report["names"]["PERSON"], report["names"]["ORG"]
    assert person["overlap"] == {"vlm_only": 2, "claims_only": 1, "both": 1}
    assert person["vlm"]["mentions_by_line"] == {"free": 3, "forced": 0}
    assert person["vlm"]["distinct_names_not_written"] == 3
    assert org["vlm"]["mentions_by_line"] == {"free": 1, "forced": 1}
    assert org["vlm"]["distinct_names_only_on_forced_lines"] == 1
    seen = report["image_text"]["datasets"]["factify2"]
    assert seen["images"] == 3 and seen["images_naming"] == {"PERSON": 2, "ORG": 2}
    assert report["claims"]["datasets"]["factify2"]["records_naming_PERSON"] == 1


def test_the_counts_report_names_no_one_and_no_record(ledger_root, tmp_path):
    table, report = _count(_write_run(ledger_root, SPECS), tmp_path)
    blob = json.dumps(report).casefold()
    for row in table.serialised():
        for form in [row["key"], row["display"], *row["forms"]]:
            assert form.casefold() not in blob, f"the published report names {form!r}"
    assert '"r1"' not in blob and '"r2"' not in blob


def test_claim_ner_reads_each_record_once_and_resumes_past_a_torn_line(tmp_path):
    ner = _fake_ner(CLAIM_SPANS)
    first = gallery.claim_spans("factify2", ner, rows=list(CLAIMS), out_dir=tmp_path)
    assert set(first) == {"r1", "r2"} and len(ner.calls) == 1
    again = gallery.claim_spans("factify2", ner, rows=list(CLAIMS), out_dir=tmp_path)
    assert again == first and len(ner.calls) == 1, "a cached record is never read again"

    cache = tmp_path / "factify2.jsonl"
    with cache.open("ab") as fh:  # a kill mid-write
        fh.write(b'{"record_id": "r3", "sp')
    more = [*CLAIMS, ("r3", "Rahul Gandhi speaks", False)]
    out = gallery.claim_spans("factify2", ner, rows=more, out_dir=tmp_path)
    assert set(out) == {"r1", "r2", "r3"} and ner.calls[-1] == ["Rahul Gandhi speaks"]
    assert (tmp_path / "factify2.torn.jsonl").is_file(), "the torn line is kept, not dropped"
    assert all(line.endswith(b"\n") for line in cache.read_bytes().splitlines(keepends=True))


def test_a_cache_written_under_another_ner_configuration_is_refused(tmp_path):
    gallery.write_json(tmp_path / "config.json", {"model": "something else"})
    with pytest.raises(gallery.GalleryError):
        gallery.claim_spans("factify2", _fake_ner({}), rows=list(CLAIMS), out_dir=tmp_path)


def test_a_row_without_forced_marks_is_counted_and_never_read(ledger_root, tmp_path):
    run_dir = _write_run(ledger_root, SPECS[:1])
    path = run_dir / "factify2.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["schema"] = "image_text/1"
    for cat in CATEGORIES:
        row["entities"][cat].pop("forced")
        for item in row["entities"][cat]["items"]:
            item.pop("forced")
    row["result_sha256"] = image_text.result_sha256(row)
    path.write_bytes(image_text.canonical(row) + b"\n")
    read = gallery.read_ledger(path, "factify2")
    assert read.unmarked == 1 and all(not img["names"] for img in read.images.values())


def test_only_the_run_written_under_the_current_row_schema_is_read(tmp_path):
    def run(name, schema):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"schema_version": schema}), encoding="utf-8")

    with pytest.raises(gallery.GalleryError):
        gallery.current_run_dir(tmp_path)
    run("old_pilot", 1)
    run("second_pilot", 2)
    run("current", image_text.SCHEMA_VERSION)
    assert gallery.current_run_dir(tmp_path).name == "current"
    run("another", image_text.SCHEMA_VERSION)
    with pytest.raises(gallery.GalleryError):
        gallery.current_run_dir(tmp_path)


def test_ner_keeps_people_and_organisations_and_takes_spans_from_the_text():
    ner = gallery.Ner("cpu")
    ner.pipe = lambda texts, batch_size: [[
        {"entity_group": "PER", "score": 0.99, "start": 0, "end": 13, "word": "narendra modi"},
        {"entity_group": "LOC", "score": 0.98, "start": 17, "end": 22, "word": "delhi"},
        {"entity_group": "ORG", "score": 0.97, "start": 28, "end": 31, "word": "bjp"},
    ] for _ in texts]
    out = ner(["Narendra Modi in Delhi with BJP", ""])
    assert [(s["type"], s["text"]) for s in out[0]] == [("PERSON", "Narendra Modi"), ("ORG", "BJP")]
    assert out[1] == [], "an empty text is never sent to the model"


def test_the_name_growth_projection_says_it_is_one_and_needs_enough_images():
    read = gallery.LedgerRead("factify2")
    for i in range(2600):
        sha = hashlib.sha256(str(i).encode()).hexdigest()
        read.images[sha] = {"record_ids": (str(i),), "status": "ok", "entity_error": None,
                            "names": [{"type": "PERSON", "name": f"Figure {i % 1300}",
                                       "basis": "seen", "forced": False, "confidence": 0.9,
                                       "evidence": "not_written"}]}
    out = gallery.name_growth(read, 45186)
    assert "extrapolation" in out["method"]
    assert out["PERSON_all"]["now"] == 1300 and out["PERSON_all"]["projected"] >= 1300
    assert gallery.name_growth(read, 2600) is None, "nothing to project for a finished run"
    small = gallery.LedgerRead("factify2", images=dict(list(read.images.items())[:2000]))
    assert gallery.name_growth(small, 45186) is None, "too few images to fit"


def test_the_coverage_curve_stops_at_the_last_name():
    assert gallery.coverage([5, 3, 1, 1]) == {"top_100": 1.0}
    big = gallery.coverage([1] * 1000)
    assert big["top_100"] == 0.1 and big["top_500"] == 0.5 and big["top_1000"] == 1.0


# --------------------------------------------------------------------------
# what may never be published
# --------------------------------------------------------------------------


def test_the_gallery_store_is_a_data_directory():
    assert paths.GALLERY in paths._TREE
    assert prepush_check.check_path("data/gallery/media/ab/abcdef.jpg")
    assert prepush_check.check_path("data/gallery/entities.jsonl")
    assert prepush_check.check_path("scripts/face_recognition_sface_2021dec.onnx")
    for suffix in (".svg", ".tif", ".tiff", ".jfif"):
        assert prepush_check.check_path(f"docs/logo{suffix}"), suffix
    assert prepush_check.check_path("data/reports/gallery_counts.json") == []


def test_fetch_models_names_what_it_may_take():
    assert gallery._models("ner,auraface") == ["ner", "auraface"]
    with pytest.raises(Exception):
        gallery._models("buffalo_l")
    assert "auraface" not in gallery.DEFAULT_MODELS, "the bake-off contrast is asked for by name"
    for spec in gallery.FACE_MODELS.values():
        assert len(spec["sha256"]) == 64 and len(spec["licence_sha256"]) == 64
        assert len(spec["revision"]) == 40


# --------------------------------------------------------------------------
# the pinned weights (opt-in: PE_FAKE_NEWS_GALLERY_MODELS=1)
# --------------------------------------------------------------------------

models = pytest.mark.skipif(not os.environ.get("PE_FAKE_NEWS_GALLERY_MODELS"),
                            reason="set PE_FAKE_NEWS_GALLERY_MODELS=1 to load the pinned weights")


@models
def test_the_pinned_face_models_load_and_find_no_face_in_a_flat_image():
    import numpy as np

    for name in gallery.FACE_BACKENDS:
        backend = gallery.make_backend(name)
        assert backend.faces(np.full((480, 640, 3), 128, np.uint8), 32) == []
        assert backend.config()["recogniser"]["sha256"] == gallery.FACE_MODELS[name]["sha256"]


@models
def test_the_pinned_ner_finds_people_and_organisations():
    out = gallery.Ner()(["Narendra Modi met Joe Biden and officials of the World Health Organization"])
    found = {(s["type"], s["text"]) for s in out[0]}
    assert {("PERSON", "Narendra Modi"), ("PERSON", "Joe Biden")} <= found
    assert ("ORG", "World Health Organization") in found


# --------------------------------------------------------------------------
# the network, faked: one worker, five seconds per host, every request logged
# --------------------------------------------------------------------------

from urllib.parse import parse_qs, urlparse  # noqa: E402

from scripts import recover  # noqa: E402
from scripts.recover import Resp  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


def _json(doc, status=200, headers=None):
    return status, (headers or {}), json.dumps(doc).encode()


class FakeWeb:
    """Routes by host and query; records every request start per host, and the
    headers each request carried."""

    def __init__(self, clock, handler):
        self.clock, self.handler, self.starts, self.headers = clock, handler, [], []

    def __call__(self, req):
        host = urlparse(req.url).hostname
        self.starts.append((host, self.clock()))
        self.headers.append((host, dict(req.headers or {})))
        self.clock.t += 0.2
        q = {k: v[0] for k, v in parse_qs(urlparse(req.url).query).items()}
        status, headers, body = self.handler(host, q, req.url)
        return Resp(req.url, status, headers, body, None, 0.2)


@pytest.fixture
def net(tmp_path):
    clock = FakeClock()
    ledger = gallery.JsonlLedger(tmp_path / "provenance.jsonl")
    ledger.open()
    holder = {}

    def make(handler, pass_id=gallery.PASS_ID):
        web = FakeWeb(clock, handler)
        holder["web"] = web
        # the worker the live network uses, over the fake transport
        return gallery.Net(lambda: gallery.WikimediaWorker(None, delay=recover.HOST_DELAY, clock=clock,
                                                           sleeper=clock.sleep, transport=web),
                           gallery.ProvenanceLog(ledger, pass_id), clock)

    yield make, holder, ledger
    ledger.close()


def _entity(qid, label, p31=("Q5",), sitelinks=10, images=(), logos=(), category=None, subclass_of=()):
    return {"id": qid, "labels": {"en": {"value": label}}, "descriptions": {"en": {"value": "d"}},
            "aliases": {"en": []}, "sitelinks": {f"w{i}": {} for i in range(sitelinks)},
            "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": c}}}} for c in p31],
                       "P279": [{"mainsnak": {"datavalue": {"value": {"id": c}}}} for c in subclass_of],
                       "P18": [{"mainsnak": {"datavalue": {"value": f}}} for f in images],
                       "P154": [{"mainsnak": {"datavalue": {"value": f}}} for f in logos],
                       **({"P373": [{"mainsnak": {"datavalue": {"value": category}}}]} if category else {})}}


def _hit(qid, label, kind="label", text=None):
    return {"id": qid, "label": label, "match": {"type": kind, "text": text or label}}


SEARCH = {
    "Narendra Modi": [_hit("Q1", "Narendra Modi"), _hit("Q99", "Narendra Modi Stadium")],
    "Modi": [_hit("Q50", "Modi")],
    "John Smith": [_hit("Q60", "John Smith"), _hit("Q61", "John Smith")],
    "Rahul Gandhi": [_hit("Q70", "Rahul Gandhi")],
    "Zzz Nobody": [],
    "Narendra": [_hit("Q1", "Narendra Modi")],
    "BJP": [_hit("Q80", "Bharatiya Janata Party", "alias", "BJP")],
    "Apple": [_hit("Q89", "apple")],
}
ENTITIES = {"Q1": _entity("Q1", "Narendra Modi", sitelinks=200, images=["Modi 1.jpg"]),
            "Q50": _entity("Q50", "Modi", p31=("Q101352",)),
            "Q60": _entity("Q60", "John Smith", sitelinks=10), "Q61": _entity("Q61", "John Smith", sitelinks=8),
            "Q70": _entity("Q70", "Rahul Gandhi", sitelinks=90),
            "Q80": _entity("Q80", "Bharatiya Janata Party", p31=("Q7278",)),
            "Q89": _entity("Q89", "apple", p31=("Q1364",)),
            # classes: a party is a political organisation is an organisation; a fruit is a food
            "Q7278": _entity("Q7278", "political party", p31=(), subclass_of=("Q7210356",)),
            "Q7210356": _entity("Q7210356", "political organization", p31=(), subclass_of=("Q43229",)),
            "Q43229": _entity("Q43229", "organization", p31=()),
            "Q1364": _entity("Q1364", "fruit", p31=(), subclass_of=("Q2095",)),
            "Q2095": _entity("Q2095", "food", p31=())}


def _wikidata(maxlag_once=None, busy_once=None, classes_fail=False, entities_fail=False):
    state = {"maxlag": maxlag_once, "busy": busy_once}

    def handler(host, q, url):
        if q.get("action") == "wbsearchentities":
            if state["busy"] == q["search"]:
                state["busy"] = None
                return _json({"error": {"code": "cirrussearch-too-busy-error", "info": "busy"}})
            return _json({"search": SEARCH.get(q["search"], [])})
        if q.get("action") == "wbgetentities" and entities_fail and q.get("props", "").startswith("labels"):
            return 503, {}, b"<html>Service Unavailable</html>"
        if q.get("action") == "wbgetentities":
            if state["maxlag"]:
                state["maxlag"] = None
                return _json({"error": {"code": "maxlag", "info": "lagged"}},
                             headers={"retry-after": "7", "mediawiki-api-error": "maxlag"})
            if classes_fail and q.get("props") == "claims":
                return 500, {}, b"<html>server error</html>"
            return _json({"entities": {i: ENTITIES.get(i, {"id": i, "missing": ""})
                                       for i in q["ids"].split("|")}})
        return 404, {}, b"<html>not here</html>"

    return handler


def _todo(*rows):
    return [{"type": t, "key": gallery.name_key(k, t), "query": q} for t, k, q in rows]


def test_a_name_links_only_to_an_exact_label_of_the_right_type(net):
    make, _holder, _ledger = net
    rows, entities = gallery.link_names(make(_wikidata()), _todo(
        ("PERSON", "Narendra Modi", "Narendra Modi"), ("PERSON", "Modi", "Modi"),
        ("PERSON", "John Smith", "John Smith"), ("PERSON", "rahulgandhi", "Rahul Gandhi"),
        ("PERSON", "Zzz Nobody", "Zzz Nobody"), ("PERSON", "Narendra", "Narendra"),
        ("ORG", "BJP", "BJP"), ("ORG", "Apple", "Apple")))
    status = {r["key"]: (r["status"], r.get("qid")) for r in rows}
    assert status["narendra modi"] == ("linked", "Q1"), "the stadium is not an exact match"
    assert status["modi"][0] == "no_type_match", "a surname item is not a person"
    assert status["john smith"] == ("ambiguous", "Q60"), "neither candidate dominates"
    assert status["rahulgandhi"] == ("linked", "Q70"), "a hashtag links to its spaced label"
    assert status["zzz nobody"][0] == "no_result"
    assert status["narendra"][0] == "no_exact_match", "a prefix is not a name"
    assert status["bjp"] == ("linked", "Q80"), "an organisation by alias, its classes walked up to Q43229"
    assert status["apple"][0] == "no_type_match", "a fruit's classes never reach an organisation"
    assert entities["Q1"]["images"] == ["Modi 1.jpg"] and entities["Q1"]["sitelinks"] == 200


def test_requests_to_one_host_start_a_second_apart_and_every_one_is_logged(net):
    make, holder, ledger = net
    rows, _ = gallery.link_names(make(_wikidata(), gallery.BUILD_PASS_ID),
                                 _todo(("PERSON", "Narendra Modi", "Narendra Modi"),
                                       ("PERSON", "John Smith", "John Smith"), ("PERSON", "Modi", "Modi")))
    by_host = {}
    for host, t in holder["web"].starts:
        by_host.setdefault(host, []).append(t)
    for host, times in by_host.items():
        assert all(b - a >= gallery.WIKIMEDIA_DELAY for a, b in zip(times, times[1:])), host
    logged = list(gallery.read_jsonl(ledger.path))
    assert len(logged) == len(holder["web"].starts)
    assert {"url", "host", "http_status", "outcome", "attempt_reason", "stage"} <= set(logged[0])
    assert all(r["pass"] == gallery.BUILD_PASS_ID for r in logged + rows), "the build's rows say so"
    assert all(h.get("Accept-Encoding") == "gzip" for _host, h in holder["web"].headers), \
        "the robot policy asks for gzip on API answers"


def _paced(answer):
    """A WikimediaWorker over a fake transport. ``answer(url)`` gives (status,
    headers, seconds the answer takes); returns the worker and every start."""
    clock = FakeClock()
    starts: list[tuple[str, float, float]] = []

    def transport(req):
        status, headers, took = answer(req.url)
        starts.append((urlparse(req.url).hostname, clock(), took))
        clock.t += took
        return Resp(req.url, status, headers, b"{}", None, took)

    def get(url):
        yield recover.Req(url, {})

    worker = gallery.WikimediaWorker(None, delay=recover.HOST_DELAY, clock=clock, sleeper=clock.sleep,
                                     transport=transport)
    return worker, starts, get


def _gaps(starts, host):
    """(start-to-start gap, time from the previous answer's end) for one host."""
    mine = [(t, took) for h, t, took in starts if h == host]
    return [(round(b[0] - a[0], 6), round(b[0] - a[0] - a[1], 6)) for a, b in zip(mine, mine[1:])]


def test_wikimedia_hosts_go_at_one_request_a_second_and_every_other_host_at_five():
    worker, starts, get = _paced(lambda url: (200, {}, 0.2))
    for i in range(3):
        for url in (f"https://www.wikidata.org/w/api.php?n={i}", f"https://thumb.wikimedia.org/t/{i}.jpg",
                    f"https://example.org/{i}"):
            worker.start(get(url))
    worker.run()
    for host in ("www.wikidata.org", "thumb.wikimedia.org"):
        assert all(gallery.WIKIMEDIA_DELAY <= gap < 1.5 for gap, _ in _gaps(starts, host)), host
    assert all(gap >= recover.HOST_DELAY for gap, _ in _gaps(starts, "example.org")), \
        "a host not named keeps the data layer's five seconds"
    ordered = sorted(starts, key=lambda s: s[1])
    assert all(b[1] >= a[1] + a[2] for a, b in zip(ordered, ordered[1:])), "one request in flight"


def test_the_robot_policys_pauses_hold_for_the_whole_host():
    def answer(url):
        if url.endswith("slow"):
            return 200, {}, 1.6                                 # an API answer over a second
        if url.endswith("busy"):
            return 429, {"retry-after": "120"}, 0.2
        if url.endswith("broken"):
            return 503, {}, 0.2
        if url.endswith("lagged"):
            return 200, {"mediawiki-api-error": "maxlag", "retry-after": "7"}, 0.2
        if url.endswith("poolfull"):
            return 200, {"mediawiki-api-error": "cirrussearch-too-busy-error"}, 0.2
        return 200, {}, 1.6 if "thumb" in url else 0.2

    worker, starts, get = _paced(answer)
    for host, first in (("www.wikidata.org/w/api.php?q=", "slow"), ("commons.wikimedia.org/w/api.php?q=", "busy"),
                        ("upload.wikimedia.org/x/", "broken"), ("thumb.wikimedia.org/x/", "a")):
        for suffix in (first, "next"):
            worker.start(get(f"https://{host}{suffix}"))
    worker.run()
    assert _gaps(starts, "www.wikidata.org")[0][1] >= gallery.SLOW_API_PAUSE, "over a second: wait five"
    assert _gaps(starts, "commons.wikimedia.org")[0][1] >= 120, "a 429's Retry-After holds the host"
    assert _gaps(starts, "upload.wikimedia.org")[0][1] >= gallery.SERVER_ERROR_PAUSE, "a 5xx: fifteen minutes"
    assert _gaps(starts, "thumb.wikimedia.org")[0][1] < gallery.SLOW_API_PAUSE, "a slow image is not an API call"

    worker, starts, get = _paced(answer)
    for suffix in ("lagged", "next", "poolfull", "next2"):
        worker.start(get(f"https://www.wikidata.org/w/api.php?q={suffix}"))
    worker.run()
    gaps = _gaps(starts, "www.wikidata.org")
    assert gaps[0][1] >= 7.0, "a maxlag refusal holds the host for its Retry-After"
    assert gaps[2][0] < 2.0, "a busy search pool delays only the refused search"
    assert worker.pause_after("example.org", Resp("https://example.org/", 503, {}, b"", None, 9.0)) == 0.0
    assert worker.pause_after("upload.wikimedia.org", Resp("https://upload.wikimedia.org/f", 500, {}, b"", None,
                                                           0.2)) == 0.0, "one file's 500 holds only that file"


def test_maxlag_is_waited_out_and_a_200_that_is_not_json_is_a_failure(net):
    make, _holder, ledger = net
    rows, _entities = gallery.link_names(make(_wikidata(maxlag_once=True)),
                                         _todo(("PERSON", "Narendra Modi", "Narendra Modi")))
    assert rows[0]["status"] == "linked"
    logged = list(gallery.read_jsonl(ledger.path))
    assert any(r["attempt_reason"] == "maxlag" and r["retry_in_s"] == 7.0 for r in logged)

    def html_200(host, q, url):
        return 200, {}, b"<html>Wikimedia error</html>"

    rows, _ = gallery.link_names(make(html_200), _todo(("PERSON", "Narendra Modi", "Narendra Modi")))
    assert rows[0]["status"] == "error" and rows[0]["reason"] == "not_json"


def test_a_busy_search_is_waited_out_and_a_class_walk_that_cannot_finish_is_an_error(net):
    make, _holder, ledger = net
    rows, _ = gallery.link_names(make(_wikidata(busy_once="Narendra Modi")),
                                 _todo(("PERSON", "Narendra Modi", "Narendra Modi")))
    assert rows[0]["status"] == "linked"
    retried = [r for r in gallery.read_jsonl(ledger.path) if r["outcome"] == "retrying"]
    assert retried and retried[0]["attempt_reason"] == "cirrussearch-too-busy-error"
    assert retried[0]["retry_in_s"] == 30.0
    rows, _ = gallery.link_names(make(_wikidata(classes_fail=True)), _todo(("ORG", "BJP", "BJP")))
    assert rows[0]["status"] == "error" and rows[0]["reason"] == "type_check_failed", \
        "an organisation not shown to be one is an error to retry, never a 'no'"
    rows, _ = gallery.link_names(make(_wikidata(entities_fail=True)),
                                 _todo(("PERSON", "Narendra Modi", "Narendra Modi")))
    assert rows[0]["status"] == "error" and rows[0]["reason"] == "entities_failed", \
        "candidates whose data never arrived are not 'no type match'"


def test_a_name_kept_under_mul_is_still_the_items_label_and_alias():
    entity = {"id": "Q10218", "labels": {"mul": {"value": "Rahul Gandhi"}},
              "descriptions": {"en": {"value": "Indian politician"}},
              "aliases": {"en": [{"value": "RaGa"}], "mul": [{"value": "Rahul Rajiv Gandhi"}, {"value": "RaGa"}]},
              "claims": {}, "sitelinks": {}}
    summary = gallery.entity_summary(entity)
    assert summary["label"] == "Rahul Gandhi" and summary["aliases"] == ["RaGa", "Rahul Rajiv Gandhi"]
    english = gallery.entity_summary({**entity, "labels": {"en": {"value": "Rahul G."}, "mul": {"value": "x"}}})
    assert english["label"] == "Rahul G.", "English first, where it exists"


def test_the_user_agent_names_a_contact_and_the_environment_can_override_it(monkeypatch):
    monkeypatch.delenv(gallery.UA_CONTACT_ENV, raising=False)
    ua = gallery.user_agent()
    assert gallery.UA_CONTACT_DEFAULT in ua and "bot" in ua and "@" not in ua
    assert "one request at a time" in ua and f"{gallery.WIKIMEDIA_DELAY:g} s" in ua, "it states its pace"
    monkeypatch.setenv(gallery.UA_CONTACT_ENV, "https://example.org/contact")
    assert "https://example.org/contact" in gallery.user_agent()


def _scoped(key, vlm=None, claims=None, entity_type="PERSON"):
    """A names.jsonl row: mentions per corpus, from the VLM and from claims."""
    vlm, claims = vlm or {}, claims or {}
    return {"type": entity_type, "key": key, "display": key.title(), "forms": {},
            "mentions": {"vlm": sum(vlm.values()), "claims": sum(claims.values())},
            "vlm": {ds: {"mentions": m} for ds, m in vlm.items()},
            "claims": {ds: {"records": m, "mentions": m} for ds, m in claims.items()}}


def test_the_build_scope_is_per_corpus_on_both_sides():
    later = _scoped("someone", vlm={"fakeddit": 2}, claims={"dgm4": 3})
    assert not gallery.in_scope(later), "a ledger that grows to cover another corpus does not widen it"
    assert gallery.in_scope(later, ("dgm4",)) and gallery.scope_mentions(later, ("dgm4",)) == 3
    mixed = _scoped("another", vlm={"factify2": 2}, claims={"dgm4": 3})
    assert gallery.in_scope(mixed) and gallery.scope_mentions(mixed) == 2


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The gallery's files under tmp_path, the network faked, no ledger read."""
    for name in ("NAMES_PATH", "LINKS_PATH", "WIKIDATA_PATH", "PROVENANCE_PATH", "SAMPLE_PATH"):
        monkeypatch.setattr(gallery, name, tmp_path / getattr(gallery, name).name)
    monkeypatch.setattr(gallery, "_ledger_read", lambda args: None)
    monkeypatch.setattr(gallery, "_population", lambda read: ([], Counter()))
    monkeypatch.setattr(gallery, "PROBE_FIGURES", {})
    clock, webs = FakeClock(), []

    def use(handler):
        def live_net(log, budget_bytes=None):
            webs.append(FakeWeb(clock, handler))
            return gallery.Net(lambda: gallery.WikimediaWorker(None, delay=recover.HOST_DELAY, clock=clock,
                                                               sleeper=clock.sleep, transport=webs[-1]),
                               log, clock)
        monkeypatch.setattr(gallery, "live_net", live_net)

    return tmp_path, use


def _searched(path):
    return [r["key"] for r in gallery.read_jsonl(path) if r["stage"] == "search" and r["outcome"] == "ok"]


def test_the_build_is_checkpointed_and_widening_its_scope_adds_without_redoing(store, monkeypatch):
    root, use = store
    gallery.write_jsonl(gallery.NAMES_PATH, [_scoped("narendra modi", claims={"factify2": 3}),
                                             _scoped("john smith", vlm={"verite": 1}),
                                             _scoped("rahul gandhi", claims={"dgm4": 5})])
    monkeypatch.setattr(gallery, "LINK_CHUNK", 1)
    healthy = _wikidata()

    def dies_on_the_second(host, q, url):
        if q.get("search") == "John Smith":
            raise ConnectionError("the machine went to sleep")
        return healthy(host, q, url)

    use(dies_on_the_second)
    with pytest.raises(ConnectionError):
        gallery.main(["link", "--all"])
    first = list(gallery.read_jsonl(gallery.LINKS_PATH))
    assert [(r["key"], r["status"]) for r in first] == [("narendra modi", "linked")], \
        "the chunk finished before the stop is on disk"
    assert first[0]["pass"] == gallery.BUILD_PASS_ID

    use(healthy)
    gallery.main(["link", "--all"])
    assert _searched(gallery.PROVENANCE_PATH) == ["PERSON:narendra modi", "PERSON:john smith"], \
        "a rerun picks up where it stopped"
    assert [e["qid"] for e in gallery.build_entities()] == ["Q1", "Q60"], "the most-mentioned first"

    before = gallery.LINKS_PATH.read_bytes()
    gallery.main(["link", "--all", "--scope", "verite,averimatec,factify2,dgm4"])
    assert _searched(gallery.PROVENANCE_PATH)[-1:] == ["PERSON:rahul gandhi"]
    assert len(_searched(gallery.PROVENANCE_PATH)) == 3, "DGM4's one new name is the only search"
    assert gallery.LINKS_PATH.read_bytes().startswith(before), "nothing already linked is rewritten"
    widened = [e["qid"] for e in gallery.build_entities(("verite", "averimatec", "factify2", "dgm4"))]
    assert widened == ["Q70", "Q1", "Q60"]


# --------------------------------------------------------------------------
# references: every file through the gates, its licence kept
# --------------------------------------------------------------------------

GOOD = _png((200, 120, 90), (64, 48))
PLACEHOLDER = _png((255, 255, 255), (64, 48))
SMALL = _png((10, 200, 10), (32, 32))


def _info(name, url, mediatype="BITMAP", licence="CC BY 4.0", size=(64, 48)):
    meta = {"Artist": {"value": "<a href='//commons.wikimedia.org/wiki/User:J'>Jane Doe</a>"},
            "Credit": {"value": "Own work"}, "AttributionRequired": {"value": "true"},
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"},
            "Copyrighted": {"value": "True"}, "Restrictions": {"value": "personality"}}
    if licence:
        meta["LicenseShortName"] = {"value": licence}
    return {"title": f"File:{name}", "imageinfo": [{
        "url": url + "/orig", "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}",
        "width": 4000, "height": 3000, "size": 5_000_000, "sha1": "ab" * 20, "mime": "image/png",
        "mediatype": mediatype, "thumburl": url, "thumbwidth": size[0], "thumbheight": size[1],
        "extmetadata": meta}]}


DEPICTS = ("Modi 2.png", "Modi drawing.svg", "Modi 3.png", "Modi 4.png", "Modi 5.png", "Modi 6.png")


def _commons(bodies, depicts=DEPICTS, fail=()):
    """``fail``: "imageinfo" or "depicts" answer 503; a body that is an int is
    that status for the file."""
    files = {"Modi 1.jpg": _info("Modi 1.jpg", "https://upload.wikimedia.org/u1"),
             "Modi 2.png": _info("Modi 2.png", "https://upload.wikimedia.org/u2"),
             "Modi drawing.svg": _info("Modi drawing.svg", "https://upload.wikimedia.org/u3", "DRAWING"),
             "Modi 3.png": _info("Modi 3.png", "https://upload.wikimedia.org/u4"),
             "Modi 4.png": _info("Modi 4.png", "https://upload.wikimedia.org/u5"),
             "Modi 5.png": _info("Modi 5.png", "https://upload.wikimedia.org/u6", licence=None),
             "Modi 6.png": _info("Modi 6.png", "https://upload.wikimedia.org/u7")}

    def handler(host, q, url):
        if host == "upload.wikimedia.org":
            body = bodies[url.rsplit("/", 1)[-1]]
            return (body, {}, b"<html>error</html>") if isinstance(body, int) else (200, {}, body)
        if q.get("list") == "search":
            if "depicts" in fail:
                return 503, {}, b"<html>Service Unavailable</html>"
            return _json({"query": {"search": [{"title": f"File:{n}"} for n in depicts]}})
        if q.get("prop") == "imageinfo" and "imageinfo" in fail:
            return 503, {}, b"<html>Service Unavailable</html>"
        if q.get("prop") == "imageinfo":
            titles = q["titles"].split("|")
            return _json({"query": {"pages": {str(i): files[t[5:]] for i, t in enumerate(titles)}}})
        return 404, {}, b""

    return handler


MODI = {"qid": "Q1", "type": "PERSON", "label": "Narendra Modi", "description": "d", "aliases": [],
        "p31": ["Q5"], "sitelinks": 200, "images": ["Modi 1.jpg"], "logos": [],
        "commons_category": None, "names": {"narendra modi"}}


@pytest.fixture
def media_root(tmp_path, monkeypatch):
    monkeypatch.setattr(image_text, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(images, "fingerprint_rejections", lambda: {})
    return tmp_path


def test_every_reference_goes_through_the_gates_and_keeps_its_licence(net, media_root, monkeypatch):
    make, _holder, _ledger = net
    digest = hashlib.sha256(PLACEHOLDER).hexdigest()
    monkeypatch.setattr(placeholders, "registry", lambda: {digest: {"status": "placeholder"}})
    bodies = {"u1": GOOD, "u2": b"<html>429 Too Many Requests</html>", "u4": PLACEHOLDER, "u5": SMALL}
    rows = gallery.fetch_references(make(_commons(bodies)), [MODI], corpus=frozenset(),
                                    media=media_root / "media", rejected=media_root / "rejected")
    refs = {r["title"]: r for r in rows[0]["refs"]}
    assert list(refs) == ["File:Modi 1.jpg", "File:Modi 2.png", "File:Modi drawing.svg",
                          "File:Modi 3.png", "File:Modi 4.png", "File:Modi 5.png"], \
        "the item's own image first, then depicts; six candidates at most"
    assert refs["File:Modi 1.jpg"]["outcome"] == "usable"
    assert refs["File:Modi 2.png"]["outcome"] == "undecodable", "a 200 that is a page is not an image"
    assert refs["File:Modi drawing.svg"]["outcome"] == "mediatype:DRAWING"
    assert refs["File:Modi 3.png"]["outcome"] == "registry:placeholder"
    assert refs["File:Modi 4.png"]["outcome"] == "rendition_mismatch"
    assert refs["File:Modi 5.png"]["outcome"] == "no_licence_metadata"
    good = refs["File:Modi 1.jpg"]
    assert (media_root / good["path"]).read_bytes() == GOOD
    assert good["sha256"] == hashlib.sha256(GOOD).hexdigest()
    assert good["licence"]["short_name"] == "CC BY 4.0" and good["licence"]["restrictions"] == "personality"
    assert good["attribution"]["artist"] == "Jane Doe" and "<a" in good["attribution"]["artist_html"]
    assert (media_root / refs["File:Modi 3.png"]["path"]).parent.parent.name == "rejected", "quarantined"
    assert refs["File:Modi 2.png"]["path"] is None, "bytes that are not an image are not kept"
    sent = _holder["web"].headers
    assert all("Accept-Encoding" not in h for host, h in sent if host == "upload.wikimedia.org"), \
        "media is never asked for gzip"
    assert all(h.get("Accept-Encoding") == "gzip" for host, h in sent if host == "commons.wikimedia.org")


def test_an_unscaled_original_must_be_the_file_and_a_thumbnail_may_round_its_height():
    examined = images.examine_bytes(GOOD)  # 64 x 48
    small = {"mime": "image/png", "original": {"width": 64, "height": 48, "sha1": hashlib.sha1(GOOD).hexdigest()},
             "rendition": {"url": "u", "width": 960, "height": 720}}  # what imageinfo reports
    assert gallery.rendition_problem(GOOD, examined, small) is None, \
        "Wikimedia serves a small original itself, never scaled up"
    other = _png((1, 2, 3), (64, 48))
    assert gallery.rendition_problem(other, images.examine_bytes(other), small) == "not_the_original_file"
    scaled = {"mime": "image/jpeg", "original": {"width": 4096, "height": 3072, "sha1": "x"},
              "rendition": {"url": "u", "width": 64, "height": 49}}
    assert gallery.rendition_problem(GOOD, examined, scaled) is None, "a thumbnail's height rounds"
    assert gallery.rendition_problem(GOOD, examined, {**scaled, "rendition": {
        "url": "u", "width": 64, "height": 60}}) == "rendition_mismatch"


def test_a_refused_rendition_is_judged_again_from_its_quarantined_bytes(media_root, monkeypatch):
    monkeypatch.setattr(placeholders, "registry", lambda: {})
    kept = gallery.store(GOOD, images.examine_bytes(GOOD), media_root / "rejected")
    ref = {"title": "File:x.png", "source": "P18", "mime": "image/png", "outcome": "rendition_mismatch",
           "path": kept.relative_to(media_root).as_posix(),
           "original": {"width": 64, "height": 48, "sha1": hashlib.sha1(GOOD).hexdigest()},
           "rendition": {"url": "https://upload.wikimedia.org/x.png", "width": 960, "height": 720}}
    refused = {"title": "File:y.png", "outcome": "registry:placeholder", "path": "elsewhere"}
    changed, tally = gallery.regate([{"qid": "Q1", "type": "PERSON", "refs": [ref, refused]}],
                                    corpus=frozenset(), media=media_root / "media")
    new = changed[0]["refs"][0]
    assert new["outcome"] == "usable" and new["regated"]["from"] == "rendition_mismatch"
    assert (media_root / new["path"]).read_bytes() == GOOD and kept.is_file(), \
        "stored, and the quarantined copy stays"
    assert changed[0]["refs"][1] == refused, "a registry refusal is never reversed"
    assert tally == {"rendition_mismatch->usable": 1}


def test_no_answer_or_not_now_leaves_the_entity_for_the_next_run(net, media_root, monkeypatch):
    make, _holder, _ledger = net
    monkeypatch.setattr(placeholders, "registry", lambda: {})

    def fetch(bodies, depicts=(), fail=()):
        return gallery.fetch_references(make(_commons(bodies, depicts, fail)), [MODI], corpus=frozenset(),
                                        media=media_root / "media", rejected=media_root / "rejected")

    assert fetch({"u1": GOOD}, fail=("imageinfo",)) == [], "no file facts is not 'missing file'"
    assert fetch({"u1": GOOD}, depicts=DEPICTS, fail=("depicts",)) == [], "a lost search is not 'no files'"
    assert fetch({"u1": 503}) == [], "the media host said not now"
    rows = fetch({"u1": 500})
    assert rows[0]["refs"][0]["outcome"] == "http_500", "a file that cannot be rendered is that file's answer"


def test_the_build_fetches_every_linked_entity_once_in_checkpointed_chunks(store, media_root, monkeypatch):
    root, use = store
    for name in ("ENTITIES_PATH", "MEDIA_DIR", "REJECTED_DIR"):
        monkeypatch.setattr(gallery, name, root / "gallery" / getattr(gallery, name).name)
    monkeypatch.setattr(gallery, "corpus_hashes", lambda: frozenset())
    monkeypatch.setattr(gallery, "free_gb", lambda _p: 100.0)
    monkeypatch.setattr(gallery, "FETCH_CHUNK", 1)
    monkeypatch.setattr(placeholders, "registry", lambda: {})
    gallery.write_jsonl(gallery.NAMES_PATH, [_scoped("narendra modi", claims={"factify2": 3}),
                                             _scoped("john smith", vlm={"verite": 1})])
    wikidata = _wikidata()
    use(wikidata)
    gallery.main(["link", "--all"])
    bodies = {"u1": GOOD, "u2": GOOD, "u4": GOOD, "u5": GOOD, "u7": GOOD}
    commons = _commons(bodies)
    use(lambda host, q, url: wikidata(host, q, url) if host == "www.wikidata.org" else commons(host, q, url))
    gallery.main(["fetch", "--all", "--budget-gb", "1"])
    fetched = list(gallery.read_jsonl(gallery.ENTITIES_PATH))
    assert [e["qid"] for e in fetched] == ["Q1", "Q60"], "every linked entity, the most-mentioned first"
    assert all(e["pass"] == gallery.BUILD_PASS_ID for e in fetched)
    assert any(r["outcome"] == "usable" for r in fetched[0]["refs"])
    assert (root / "gallery" / "media").is_dir(), "stored where the build says, not elsewhere"
    gallery.main(["fetch", "--all", "--budget-gb", "1"])
    assert len(list(gallery.read_jsonl(gallery.ENTITIES_PATH))) == 2, "a rerun fetches nothing twice"


def test_a_reference_that_is_a_corpus_image_is_quarantined(net, media_root, monkeypatch):
    make, _holder, _ledger = net
    monkeypatch.setattr(placeholders, "registry", lambda: {})
    rows = gallery.fetch_references(make(_commons({"u1": GOOD}, depicts=())), [MODI],
                                    corpus=frozenset({hashlib.sha256(GOOD).hexdigest()}),
                                    media=media_root / "media", rejected=media_root / "rejected")
    assert rows[0]["refs"][0]["outcome"] == "corpus_duplicate"


# --------------------------------------------------------------------------
# the gallery: references that agree, and 1:1 verification
# --------------------------------------------------------------------------


#: Embeddings like SFace's: 128 dimensions. A face of the same person is its
#: identity plus noise of about half its length (cosine about 0.8 between two
#: of them); two people's are about orthogonal (cosine within +-0.3).
DIM, NOISE = 128, 0.044


def _vec(seed, base=None, noise=NOISE):
    import numpy as np

    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM) if base is None else base + noise * rng.normal(size=DIM)
    return (v / np.linalg.norm(v)).astype("float32")


def _face(vec, box=(10, 10, 100, 100), dh="0000000000000000"):
    import base64

    return {"box": list(box), "landmarks": [[0, 0]] * 5, "score": 0.99, "dhash": dh,
            "embedding": base64.b64encode(vec.tobytes()).decode("ascii")}


def _ref(sha, source="depicts", outcome="usable"):
    return {"sha256": sha, "title": f"File:{sha}.jpg", "source": source, "outcome": outcome,
            "page_url": f"https://commons.wikimedia.org/wiki/File:{sha}.jpg",
            "path": f"data/gallery/media/{sha}.jpg",
            "licence": {"short_name": "CC BY 4.0"}, "attribution": {"artist": "Jane Doe"}}


def _people(n=40):
    """``n`` people, four references each, and the faces of every reference.

    Forty, not five: the consistency threshold is a high quantile of how alike
    DIFFERENT people's faces are, and over a handful of pairs that quantile is
    simply the most alike pair -- a random stranger -- which is no threshold."""
    bases = {f"Q{i}": _vec(i) for i in range(1, n + 1)}
    entities, faces = [], {}
    for qid, base in bases.items():
        refs = []
        for j in range(4):
            sha = f"{qid}r{j}"
            faces[sha] = {"faces": [_face(_vec(100 * int(qid[1:]) + j, base))]}
            refs.append(_ref(sha, "P18" if j == 0 else "depicts"))
        entities.append({"qid": qid, "type": "PERSON", "label": f"Person {qid}", "refs": refs})
    return bases, entities, faces


def test_a_reference_that_disagrees_with_the_others_is_someone_else():
    bases, entities, faces = _people()
    faces["Q1r3"] = {"faces": [_face(_vec(999, bases["Q2"]))]}      # Q2's face filed under Q1
    faces["Q3r2"] = {"faces": [_face(_vec(31, bases["Q3"]), box=(0, 0, 90, 90)),
                               _face(_vec(32, bases["Q4"]), box=(200, 0, 90, 90))]}  # a group photo
    faces["Q4r1"] = {"faces": []}
    rows, summary = gallery.assemble(entities, faces)
    by = {r["qid"]: {ref["sha256"]: ref for ref in r["refs"]} for r in rows}
    assert by["Q1"]["Q1r3"]["decision"] == "inconsistent_reference"
    assert by["Q3"]["Q3r2"]["decision"] == "usable" and by["Q3"]["Q3r2"]["selected_by"] == "resolved_by_medoid"
    assert by["Q3"]["Q3r2"]["face"] == 0, "the face that agrees with the entity's other references"
    assert by["Q4"]["Q4r1"]["decision"] == "no_face"
    assert summary["consistency_threshold"] is not None
    assert {r["qid"]: r["usable_refs"] for r in rows}["Q1"] == 3


def test_two_references_that_disagree_keep_wikidatas_own_image():
    bases, entities, faces = _people()
    entities[0]["refs"] = entities[0]["refs"][:2]                         # Q1: P18 + one depicts
    faces["Q1r1"] = {"faces": [_face(_vec(7, bases["Q5"]))]}
    rows, _ = gallery.assemble(entities, faces)
    assert {ref["sha256"]: ref["decision"] for ref in rows[0]["refs"]} == {
        "Q1r0": "usable", "Q1r1": "inconsistent_reference"}


def test_a_pinned_consistency_threshold_leaves_every_existing_entry_as_it_is():
    bases, entities, faces = _people(40)
    faces["Q1r3"] = {"faces": [_face(_vec(999, bases["Q2"]))]}      # an outlier under Q1
    rows, summary = gallery.assemble(entities, faces)
    pin = summary["consistency_threshold"]
    _bases, grown, grown_faces = _people(120)                          # the same 40, and 80 more
    grown_faces["Q1r3"] = faces["Q1r3"]
    again, later = gallery.assemble(grown, grown_faces, pin)
    assert later["pinned"] and later["consistency_threshold"] == pin
    assert later["derived_now"] != pin, "derived again it would move"
    assert again[:40] == rows, "every existing entry is exactly what it was"


def _gallery_on_disk(tmp_path, entities, faces, links, aliases=None):
    rows, _ = gallery.assemble(entities, faces)
    cid = "testcid"
    gallery.write_jsonl(tmp_path / "faces" / cid / "gallery.jsonl", rows)
    gallery.write_jsonl(tmp_path / "faces" / cid / "refs.jsonl",
                        [{"sha256": s, **f} for s, f in faces.items()])
    gallery.write_jsonl(tmp_path / "entities.jsonl", entities)
    gallery.write_jsonl(tmp_path / "links.jsonl", links)
    gallery.write_jsonl(tmp_path / "wikidata.jsonl",
                        [{"qid": e["qid"], "label": e["label"], "aliases": (aliases or {}).get(e["qid"], [])}
                         for e in entities])
    return gallery.Gallery(cid, faces_dir=tmp_path / "faces", entities_path=tmp_path / "entities.jsonl",
                           links_path=tmp_path / "links.jsonl", wikidata_path=tmp_path / "wikidata.jsonl")


def test_people_without_a_usable_face_are_the_coverage_ceiling(tmp_path):
    bases, entities, faces = _people()
    entities[0]["refs"] = []                                           # Q1: nothing on Commons
    faces["Q2r0"] = faces["Q2r1"] = faces["Q2r2"] = faces["Q2r3"] = {"faces": []}  # Q2: no face anywhere
    g = _gallery_on_disk(tmp_path, entities, faces, [])
    coverage = gallery.verification_coverage(g)
    assert coverage["people"] == 40 and coverage["without"] == 2 and coverage["with_usable_reference_faces"] == 38
    assert coverage["without_any_commons_reference"] == 1 and coverage["references_but_no_usable_face"] == 1
    assert "can only answer 'no usable reference'" in coverage["statement"]
    _rows, summary = gallery.assemble(entities, faces)
    assert summary["people"] == 40 and summary["people_with_usable_faces"] == 38


def test_verification_returns_the_reference_it_matched_so_a_person_can_check_it(tmp_path):
    bases, entities, faces = _people()
    entities.append({"qid": "Q80", "type": "ORG", "label": "Some Party", "refs": [_ref("logo", "P154")]})
    links = [{"type": "PERSON", "key": "narendra modi", "status": "linked", "qid": "Q1"}]
    g = _gallery_on_disk(tmp_path, entities, faces, links)
    query = [_face(_vec(5000, bases["Q1"]))]
    out = g.verify(query, "PM Narendra Modi", threshold=0.5)
    assert out["in_gallery"] and out["match"] and out["matched_entity"] == "Q1"
    ref = out["matched_reference"]
    assert ref["title"].startswith("File:Q1r") and ref["licence"]["short_name"] == "CC BY 4.0"
    assert ref["page_url"].startswith("https://commons.wikimedia.org/") and ref["face_box"]
    assert len(out["per_reference"]) == 4 and out["query_face"]["box"]

    stranger = g.verify([_face(_vec(6000, bases["Q2"]))], "Narendra Modi", threshold=0.5)
    assert stranger["in_gallery"] and stranger["match"] is False and stranger["reason"] == "below_threshold"
    refused = g.verify(query, "Person Q1", None)
    assert refused["reason"].startswith("no_certified_threshold") and refused["match"] is None, \
        "without a certified threshold verify neither accepts nor rejects"
    assert refused["score"] is not None and refused["matched_reference"], "labels resolve too; a person can check"
    assert g.verify(query, "Nobody Known", 0.5)["reason"] == "name_not_in_gallery"
    assert g.verify([], "Narendra Modi", 0.5)["reason"] == "no_face_in_image"
    org = g.verify(query, "Some Party", 0.5, entity_type="ORG")
    assert org["in_gallery"] and org["match"] is None, "an organisation's match is never claimed"


# --------------------------------------------------------------------------
# calibration: Factify2 train only, scored on the VLM's own wrong names
# --------------------------------------------------------------------------


def test_the_claimed_figure_comes_from_the_probe_list_first_then_from_one_ner_person():
    spans = {"3": {"spans": [_span("Kale Khan", "PERSON")]},
             "4": {"spans": [_span("Kale Khan", "PERSON"), _span("Ganeshi Lal", "PERSON")]}}

    def row(rid, text):
        return {"record_id": rid, "text": text}

    assert gallery.claimed_figure(row("1", "Photo shows Modi at a rally"), spans) == ("probe_list", "narendra modi")
    assert gallery.claimed_figure(row("2", "Photo of Modi and Trump"), spans) is None
    assert gallery.claimed_figure(row("3", "Image shows Kale Khan"), spans) == ("claim_ner", "kale khan")
    assert gallery.claimed_figure(row("4", "Image shows two men"), spans) is None
    assert gallery.claimed_figure(row("3", "Kale Khan said so"), spans) is None, "nothing says it is pictured"


def test_everyone_an_images_claims_name_is_collected_not_only_the_figure():
    spans = {"1": {"spans": [_span("Kale Khan", "PERSON")]}, "2": {"spans": [_span("Ganeshi Lal", "PERSON")]}}
    rows = [{"record_id": "1", "text": "Putin's spokesman Kale Khan (file photo) spoke today"},
            {"record_id": "2", "text": "Ganeshi Lal met Modi"}]
    assert gallery.claimed_figure(rows[0], spans) == ("probe_list", "vladimir putin"), \
        "the figure is the listed one, though the man pictured is the spokesman"
    assert gallery.claim_people(rows, spans) == ["Ganeshi Lal", "Kale Khan", "Narendra Modi", "Vladimir Putin"]


def test_a_name_a_claims_text_carries_whole_is_flagged_for_review():
    assert gallery.mentioned("Melania Trump", "First Lady Melania Trump (@FLOTUS) poses")
    assert gallery.mentioned("Narendra Modi", "and PM @narendramodi pose for a photo"), "a handle"
    assert gallery.mentioned("Donald Trump", "US President @realDonaldTrump"), "inside a longer handle"
    assert not gallery.mentioned("Narendra Modi", "PM Modi spoke"), "a surname alone is not the name"
    assert gallery.mentioned("Modi", "PM Modi spoke") and not gallery.mentioned("Ali", "in Australia"), \
        "a one-word name must be a whole word"


def test_the_population_is_factify2_train_images_only():
    read = gallery.LedgerRead("factify2")
    names = [{"type": "PERSON", "name": "Amit Shah", "basis": "seen", "forced": False,
              "confidence": 0.9, "evidence": "not_written"}]
    for sha, rids in (("a" * 64, ("1",)), ("b" * 64, ("2", "3")), ("c" * 64, ("4",))):
        read.images[sha] = {"record_ids": rids, "status": "ok", "entity_error": None, "names": names}
    rows = [{"record_id": "1", "split": "train", "usable": True, "image_paths": ["p/a"], "text": "Photo of Modi"},
            {"record_id": "2", "split": "train", "usable": True, "image_paths": ["p/b"], "text": "Photo of Modi"},
            {"record_id": "3", "split": "test", "usable": True, "image_paths": ["p/x", "p/b"], "text": "x"},
            {"record_id": "4", "split": "train", "usable": True, "image_paths": ["p/c"], "text": "Modi said"}]
    index = {"p/a": {"sha256": "a" * 64}, "p/b": {"sha256": "b" * 64}, "p/c": {"sha256": "c" * 64}}
    population, tally = gallery.calibration_population(read, rows, index, {})
    assert [p["sha256"] for p in population] == ["a" * 64]
    assert population[0]["claim_people"] == ["Narendra Modi"] and population[0]["claim_text"] == "Photo of Modi"
    assert tally["not_train_only"] == 1 and tally["no_figure_named"] == 1
    with pytest.raises(gallery.GalleryError):
        gallery.assert_factify2_train(["1", "3"], {r["record_id"]: r["split"] for r in rows})
    with pytest.raises(gallery.GalleryError):
        gallery.calibration_population(gallery.LedgerRead("verite"), rows, index, {})


def test_a_trial_resolves_its_name_exactly_as_verification_does(tmp_path):
    bases, entities, faces = _people()
    links = [{"type": "PERSON", "key": "narendra modi", "status": "linked", "qid": "Q1"}]
    g = _gallery_on_disk(tmp_path, entities, faces, links, aliases={"Q1": ["NaMo"]})

    def said(name):
        return {"type": "PERSON", "name": name, "evidence": "not_written", "forced": False,
                "confidence": 0.9, "basis": "seen"}

    population = [{"sha256": "img1", "figure_source": "probe_list", "figure_key": "narendra modi",
                   "names": [said("NaMo"), said("Person Q2"), said("Nobody Here")]}]
    query = {"img1": {"faces": [_face(_vec(7000, bases["Q1"]))]}}
    trials, tally = gallery.build_trials(population, {("PERSON", "narendra modi"): links[0]}, query, g)
    by = {t["name"]: t for t in trials}
    assert by["NaMo"]["label"] == "right" and by["NaMo"]["scorable"] and by["NaMo"]["score"] > 0.5, \
        "an alias resolves, as it does in verify"
    assert by["NaMo"]["token_label"] == "wrong", "the probe's token rule misses the alias: it is loose"
    assert by["Person Q2"]["label"] == "wrong" and by["Person Q2"]["score"] < 0.5
    assert by["Nobody Here"]["label"] == "unlinked" and by["Nobody Here"]["unscorable"] == "name_unlinked"
    assert tally["images_with_trials"] == 1


def test_a_name_is_right_when_the_claims_name_that_person_not_only_the_figure(tmp_path):
    bases, entities, faces = _people()
    links = [{"type": "PERSON", "key": "narendra modi", "status": "linked", "qid": "Q1"}]
    g = _gallery_on_disk(tmp_path, entities, faces, links, aliases={"Q4": ["Raj Lal"], "Q5": ["Raj Lal"]})

    def said(name):
        return {"type": "PERSON", "name": name, "evidence": "not_written", "forced": False,
                "confidence": 0.9, "basis": "seen"}

    population = [{"sha256": "img1", "figure_source": "probe_list", "figure_key": "narendra modi",
                   "claim_people": ["Narendra Modi", "Person Q2", "Raj Lal"],
                   "claim_text": "Photo: PM @narendramodi with Person Q2 and Raj Lal",
                   "names": [said("Narendra Modi"), said("Person Q2"), said("Person Q4"), said("Person Q6")]}]
    query = {"img1": {"faces": [_face(_vec(7000, bases["Q2"]))]}}
    trials, _tally = gallery.build_trials(population, {("PERSON", "narendra modi"): links[0]}, query, g)
    by = {t["name"]: t for t in trials}
    assert by["Person Q2"]["label"] == "right" and by["Person Q2"]["figure_label"] == "wrong", \
        "a person the claims name beside the figure, correctly named, is right"
    assert by["Narendra Modi"]["label"] == by["Narendra Modi"]["figure_label"] == "right"
    assert by["Person Q4"]["label"] == "wrong", "a claim name that could be two people decides nothing"
    assert by["Person Q6"]["label"] == "wrong" and by["Person Q6"]["claim_qids"] == ["Q1", "Q2"]
    assert by["Narendra Modi"]["named_in_claim_text"], "a handle carries the name"
    assert not by["Person Q6"]["named_in_claim_text"]


def _trial(label, score, sha="s", faces=1):
    return {"sha256": sha, "label": label, "scorable": score is not None, "score": score,
            "forced": False, "same_photo": False, "faces": faces, "figure_source": "probe_list",
            "ambiguous_link": False, "token_label": None, "unscorable": None if score is not None else "x"}


def test_the_threshold_is_where_accepted_names_are_right_ninety_percent_of_the_time():
    trials = ([_trial("right", 0.9, f"r{i}") for i in range(40)]
              + [_trial("wrong", 0.85, f"w{i}") for i in range(3)]
              + [_trial("wrong", 0.3, f"x{i}") for i in range(40)]
              + [_trial("right", None, "u")])
    assert gallery.choose_threshold(trials, "point") == 0.85, "40 right, 3 wrong above it: 93%"
    assert gallery.choose_threshold(trials, "bound") == 0.9, "the 95% bound needs the wrong ones out"
    at = gallery.operating_point(trials, 0.9)
    assert at["accepted_right"] == 40 and at["accepted_wrong"] == 0 and at["right"] == 41
    assert at["tar"] == round(40 / 41, 4), "an unscorable right name is a rejection: a cost"
    loose = gallery.operating_point(trials, 0.85)
    p, tar, fmr = 41 / 84, 40 / 41, 3 / 43
    assert loose["precision"] == round(p * tar / (p * tar + (1 - p) * fmr), 4), \
        "accepted-name precision is the pilot's formula, measured directly"


def test_a_persons_decisions_relabel_trials_and_unsure_takes_one_out():
    trials = [{**_trial("wrong", 0.7, "a" * 64), "key": "peskov"},
              {**_trial("right", 0.6, "b" * 64), "key": "putin"},
              {**_trial("wrong", 0.5, "c" * 64), "key": "x"},
              {**_trial("unlinked", None, "d" * 64), "key": "y"},
              {**_trial("right", 0.55, "e" * 64), "key": "modi", "figure_qid": "Q1", "qids": ["Q1"]},
              {**_trial("wrong", 0.4, "f" * 64), "key": "z"},
              {**_trial("right", 0.65, "0" * 64), "key": "barr", "figure_qid": "Q1", "qids": ["Q9"]}]
    decisions = {gallery.trial_id(trials[0]): "vlm", gallery.trial_id(trials[1]): "neither",
                 gallery.trial_id(trials[2]): "unsure", gallery.trial_id(trials[4]): "claim",
                 gallery.trial_id(trials[5]): None, gallery.trial_id(trials[6]): "claim"}
    out = {x["key"]: x for x in gallery.adjudicated(trials, decisions)}
    assert out["peskov"]["label"] == "right", "the face is the person the VLM named"
    assert out["putin"]["label"] == "wrong" and "x" not in out and out["y"]["label"] == "unlinked"
    assert out["modi"]["label"] == "right", "the VLM named the claim's person, and the face is them"
    assert out["barr"]["label"] == "wrong", \
        "labelled right because the claims name the VLM's person too, but the face is the figure"
    assert out["z"]["label"] == "wrong" and "adjudicated" not in out["z"], "null is undecided: a partial sheet works"
    with pytest.raises(gallery.GalleryError):
        gallery.adjudicated(trials, {gallery.trial_id(trials[0]): "probably"})


def test_the_decisions_file_gains_slots_and_never_loses_a_decision(tmp_path):
    path = tmp_path / "sface_decisions.json"
    assert gallery.merge_decision_slots(path, ["a", "b", "a"]) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": None, "b": None}
    path.write_text(json.dumps({"a": "vlm", "b": None}), encoding="utf-8")
    assert gallery.merge_decision_slots(path, ["c", "a"]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "vlm", "b": None, "c": None}


def test_the_next_sheet_holds_every_plausible_name_whatever_its_label_and_nothing_decided():
    trials = [{**_trial("right", 0.8, "a" * 64), "key": "accepted"},    # where an unseen false accept sits
              {**_trial("wrong", 0.7, "b" * 64), "key": "lookalike"},
              {**_trial("right", 0.6, "c" * 64), "key": "decided"},
              {**_trial("right", 0.3, "d" * 64), "key": "low"},
              {**_trial("right", None, "e" * 64), "key": "unscorable"},
              {**_trial("wrong", 0.2, "f" * 64), "key": "handle", "named_in_claim_text": True},
              {**_trial("wrong", 0.1, "1" * 64), "key": "stranger", "named_in_claim_text": False},
              {**_trial("unlinked", None, "2" * 64), "key": "nobody"}]
    decided = [gallery.trial_id(trials[2])]
    queue = gallery.review_queue(trials, decided, floor=0.38, rejected=20)
    parts = [(part, x["key"]) for part, x in queue]
    assert parts[:2] == [("plausible", "accepted"), ("plausible", "lookalike")], \
        "everything at or above the floor, labelled right or wrong, by falling score"
    assert sorted(parts[2:4]) == [("rejected_right", "low"), ("rejected_right", "unscorable")]
    assert parts[4:] == [("named_in_claim_text", "handle")], "decided, unlinked and plain wrong names stay off"
    assert len(gallery.review_queue(trials, decided, floor=0.38, rejected=1)) == 4, "rejected right names: a sample"


def test_a_partial_ledger_reports_a_candidate_and_never_sets_the_threshold(tmp_path):
    bases, entities, faces = _people()
    g = _gallery_on_disk(tmp_path, entities, faces, [])
    trials = ([{**_trial("right", 0.9, f"r{i}"), "key": f"r{i}"} for i in range(40)]
              + [{**_trial("wrong", 0.85, f"w{i}"), "key": f"w{i}"} for i in range(3)]
              + [{**_trial("wrong", 0.3, f"x{i}"), "key": f"x{i}"} for i in range(40)])
    trials = [{**x, "qid": "Q1", "figure_qid": "Q1"} for x in trials]
    query = {x["sha256"]: {"faces": [_face(_vec(7000 + i, bases["Q1"]))]} for i, x in enumerate(trials)}
    partial = gallery.calibrate(trials, Counter(), g, query, "sface",
                                {"images": 10, "worklist": 100, "complete": False})
    t = partial["threshold"]
    assert t["chosen"] is None and not t["certified"] and t["candidate"] == 0.9
    assert t["candidate_by"].startswith("one-sided") and t["bound_shortfall"] is None
    assert t["chosen_by"].startswith("unset") and "10 of 100 images" in t["chosen_by"]
    assert partial["at_threshold"]["threshold"] == 0.9, "every figure is still reported, at the candidate"
    complete = {"images": 100, "worklist": 100, "complete": True}
    full = gallery.calibrate(trials, Counter(), g, query, "sface", complete)
    assert full["threshold"]["chosen"] == full["threshold"]["candidate"] == 0.9 and full["threshold"]["certified"]

    # the complete run, short of the bound: a point estimate exists, and is never adopted
    short = [x for x in trials if x["key"] not in {f"r{i}" for i in range(15, 40)}]   # 15 right names left
    out = gallery.calibrate(short, Counter(), g, query, "sface", complete)["threshold"]
    assert out["by_point_estimate"] is not None and out["by_bound"] is None
    assert out["chosen"] is None and not out["certified"] and out["bound_shortfall"]["reason"] == "too_few_trials"
    assert out["candidate"] == 0.9 and "does not decide at this confidence" in out["chosen_by"]


def _figure_trial(figure, label, score, i):
    return {**_trial(label, score, f"{figure}-{i}"), "figure_qid": figure, "key": f"k{i}"}


def test_two_figures_carrying_most_trials_are_named_and_every_figure_weighs_the_same():
    trials = ([_figure_trial("QA", "right", 0.9, i) for i in range(30)]
              + [_figure_trial("QB", "right", 0.9, i) for i in range(20)])
    for f in range(10):  # ten figures the VLM finds hard: a right name low, a wrong one high
        trials += [_figure_trial(f"QC{f}", "right", 0.5, 0), _figure_trial(f"QC{f}", "wrong", 0.8, 1)]
    table = gallery.figure_table(trials, 0.8)
    assert [r["figure_qid"] for r in table[:2]] == ["QA", "QB"] and table[0]["trials"] == 30
    concentration = gallery.concentration(table)
    assert concentration["concentrated"] and concentration["top_2_share"] == round(50 / 70, 4)
    assert "mostly about those two people" in concentration["statement"]
    pooled = gallery.operating_point(trials, 0.8)
    strat = gallery.figure_stratified(trials, 0.8)
    assert pooled["precision"] == round(50 / 60, 4)
    assert strat["figures"] == 12 and strat["vlm_precision"] == round(7 / 12, 4)
    assert strat["precision"] == round(2 / 7, 4), "each figure once: two easy ones, ten that fail"
    assert strat["tar"] == round(2 / 7, 4) and strat["fmr"] == 1.0
    low, high = strat["bootstrap_95"]["precision"]
    assert low <= strat["precision"] <= high
    assert strat["threshold_by_figure_bound"] == 0.9, "only above every wrong name does it hold per figure"


def test_a_threshold_set_on_random_strangers_accepts_the_vlms_look_alikes():
    trials = [_trial("right", 0.8, f"r{i}") for i in range(10)] + [_trial("wrong", 0.6, f"w{i}") for i in range(10)]
    strangers = [round(0.1 + 0.02 * i, 2) for i in range(10)]         # every one below 0.3
    finding = gallery.look_alike_finding(trials, strangers, 0.7, p=0.5)
    assert finding["vlm_wrong_names"]["above_every_stranger"] == 10
    assert finding["random_strangers"]["accepted_at_threshold"] == 0
    counterfactual = finding["if_calibrated_on_random_pairs"]
    assert counterfactual["threshold"] == 0.28, "one stranger in ten above it: 0.5 / (0.5 + 0.05) >= 0.9"
    on_vlm = counterfactual["on_the_vlms_own_names"]
    assert on_vlm["accepted_wrong"] == 10 and on_vlm["precision"] == 0.5
    assert "calibrated on those" in finding["finding"]


def test_the_published_calibration_names_no_one():
    report = {"figures": {"concentration": {"figures": 1},
                          "table": [{"figure_qid": "Q1058", "label": "Narendra Modi", "trials": 3}]},
              "adjudicated": {"figures": {"table": [{"figure_qid": "Q22686", "label": "Donald Trump",
                                                     "trials": 2}]}}}
    out = gallery.public_report(report)
    assert out["figures"]["table"] == [{"rank": 1, "trials": 3}]
    text = json.dumps(out)
    assert "Q1058" not in text and "Modi" not in text and "Trump" not in text
    with pytest.raises(gallery.GalleryError):
        gallery.public_report({"anywhere": ["Q42"]})


def test_no_threshold_is_claimed_where_none_reaches_the_bar():
    trials = ([_trial("right", 0.9, f"r{i}") for i in range(5)]
              + [_trial("wrong", 0.95, f"w{i}") for i in range(5)])
    assert gallery.choose_threshold(trials, "point") is None
    assert gallery.choose_threshold(trials, "bound") is None


def test_short_of_the_bound_the_report_says_whether_trials_or_overlap_are_to_blame():
    few = [_trial("right", 0.9, f"r{i}") for i in range(10)] + [_trial("wrong", 0.3, f"w{i}") for i in range(40)]
    t, why = gallery.bound_shortfall(few)
    assert t == 0.9 and why["reason"] == "too_few_trials" and why["accepted_names"] == 10
    assert why["accepted_names_needed"] == 29, "ten of ten right: 29 of 29 would clear the bound"
    mixed = ([_trial("right", 0.9, f"r{i}") for i in range(20)] + [_trial("wrong", 0.95, f"v{i}") for i in range(5)]
             + [_trial("wrong", 0.3, f"w{i}") for i in range(20)])
    t, why = gallery.bound_shortfall(mixed)
    assert t == 0.9 and why["reason"] == "overlap" and "more trials cannot fix that" in why["statement"]
    assert gallery.accepted_names_needed(1.0) == 29 and gallery.accepted_names_needed(0.9) is None
    assert gallery.bound_shortfall([_trial("right", None, "u")])[1]["reason"] == "no_scorable_trials"


def test_verify_decides_only_at_a_threshold_a_calibration_certified():
    def report(**threshold):
        return {"face_config": "cid", "threshold": threshold}

    assert gallery.certified_threshold(report(chosen=0.44, certified=True), "cid") == 0.44
    assert gallery.certified_threshold(report(chosen=0.44, certified=True), "other") is None, "another configuration"
    assert gallery.certified_threshold(report(chosen=None, certified=False, candidate=0.44), "cid") is None, \
        "a candidate is not a threshold"
    assert gallery.certified_threshold(report(chosen=0.661), "cid") is None, "an older report certifies nothing"
    assert gallery.certified_threshold(None, "cid") is None


def _name(entity_type, key, vlm=0, claims=1):
    return {"type": entity_type, "key": key, "display": key, "mentions": {"vlm": vlm, "claims": claims},
            "vlm": {}, "claims": {"factify2": {"records": claims, "mentions": claims}}}


def test_names_collapse_into_entities_through_labels_and_aliases_and_strata_weight_them():
    scope = [_name("PERSON", "narendra modi"), _name("PERSON", "modi"), _name("PERSON", "narendramodi"),
             _name("PERSON", "zz"), _name("PERSON", "amit shah", vlm=1, claims=0), _name("ORG", "bjp")]
    wiki = {"Q1": {"qid": "Q1", "label": "Narendra Modi", "aliases": ["Modi", "NaMo"]},
            "Q2": {"qid": "Q2", "label": "Amit Shah", "aliases": []}}
    assert gallery.alias_families(scope, wiki, ["Q1", "Q2"], "PERSON") == {"Q1": 3, "Q2": 1}, \
        "the full name, the surname alias and the hashtag are one entity"
    sample = {"strata": {"PERSON|claims|1": {"size": 4, "drawn": 2}, "PERSON|vlm|1": {"size": 1, "drawn": 1},
                         "ORG|claims|1": {"size": 1, "drawn": 1}},
              "names": [{"type": "PERSON", "key": "narendra modi", "why": ["random"], "stratum": "PERSON|claims|1"},
                        {"type": "PERSON", "key": "zz", "why": ["random"], "stratum": "PERSON|claims|1"},
                        {"type": "PERSON", "key": "amit shah", "why": ["random"], "stratum": "PERSON|vlm|1"},
                        {"type": "ORG", "key": "bjp", "why": ["random"], "stratum": "ORG|claims|1"}]}
    links = {("PERSON", "narendra modi"): {"status": "linked", "qid": "Q1"},
             ("PERSON", "zz"): {"status": "no_result"},
             ("PERSON", "amit shah"): {"status": "linked", "qid": "Q2"},
             ("ORG", "bjp"): {"status": "no_type_match"}}
    out = gallery.linking_estimate(sample, links, wiki, scope)
    person = out["PERSON"]
    assert person["names_in_scope"] == 5 and person["names_linked"] == 3, "2 x 1 linked + 1 x 1 linked"
    assert person["distinct_entities"] == 2, "2 x 1/3 + 1 x 1 = 1.67 entities"
    assert out["ORG"]["names_linked"] == 0 and out["status"]["no_type_match"] == 1


def test_the_alignment_maps_the_template_onto_itself():
    import numpy as np

    m = gallery.similarity_transform(gallery.ARCFACE_TEMPLATE, gallery.ARCFACE_TEMPLATE)
    assert np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-6)
    pts = np.array(gallery.ARCFACE_TEMPLATE) * 2.0 + 15.0
    m = gallery.similarity_transform(pts, gallery.ARCFACE_TEMPLATE)
    back = np.hstack([pts, np.ones((5, 1))]) @ m.T
    assert np.allclose(back, gallery.ARCFACE_TEMPLATE, atol=1e-4)
    assert gallery.hamming("ff", "0f") == 4
