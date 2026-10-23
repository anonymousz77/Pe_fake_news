#!/usr/bin/env python
"""The entity gallery: who an image shows, checked against reference images of known entities.

    python scripts/gallery.py fetch-models                # pinned NER and face weights (network)
    python scripts/gallery.py count                       # the names our corpora use; offline
    python scripts/gallery.py link [--all] [--dry-run]    # names -> Wikidata: the sample, or the build (network)
    python scripts/gallery.py fetch [--all] --budget-gb 8 # their Commons references, gated (network)
    python scripts/gallery.py embed --backend sface       # faces of references and calibration images
    python scripts/gallery.py calibrate --backend sface   # threshold and its cost: Factify2 train only
    python scripts/gallery.py review --backend sface      # a local sheet a person checks by eye
    python scripts/gallery.py sample-report               # data/reports/gallery_sample.json
    python scripts/gallery.py verify --image P --name N   # one 1:1 verification
    python scripts/gallery.py status                      # offline counts

The network commands use one worker, one request in flight, at the pace
Wikimedia's robot policy publishes (WikimediaWorker: one request start per
second per Wikimedia host, five seconds after an API answer that took over a
second, Retry-After and 5xx pauses for the whole host; 5 s for any other host).
They send a User-Agent naming the project's anonymised repository URL
(``PE_FAKE_NEWS_UA_CONTACT`` overrides it from the environment), and log
every request to ``data/gallery/provenance.jsonl``. Every reference goes
through the three gates -- status 200, a full decode, the placeholder registry
-- and two of the gallery's own: the bytes are the rendition the API described,
and they are not an image of any of our corpora.

A VLM name is a candidate, never a verification. On the image_text pilot (run
93cd9d679720) 22 of the 46 PERSON names the model produced from appearance
alone were the figure the claim names: 47.8%. That figure is APPROXIMATE. The
probe's reference is a loose token pattern over the claim text ("gandhi"
counts Rahul Gandhi as Mahatma Gandhi, "trump" counts Melania as Donald), so a
right name can be scored wrong and a wrong one right. It is also UNDERSTATED:
it scored a name against one figure per image, so a person the claim names
beside that figure, correctly named, counted as wrong. The first QID-based
figure here (0.5455, the 2026-10-21 sample pass) carried the same defect. A
name is now scored against everyone the image's claims name.

Identity is decided here instead, against a gallery of known entities:
Wikidata items with reference images on Wikimedia Commons, each image's licence
and attribution recorded as Commons states them, and a face embedding per
reference. Open-web reverse image search is not used.

**Scope: the entities our corpora name**, not Wikidata at large. ``count``
gathers PERSON and ORG names from two places and says how many there are before
a single request is made:

- the VLM's own names, from the current image_text run directory (row schema
  image_text/3). They are read through ``image_text.entities_of(forced="mark")``,
  so a name on a line the skeleton forced stays marked. Each is also checked
  against the independent OCR (``written`` / ``partly_written`` /
  ``not_written``), the evidence the recognition probe uses;
- claim text, through a pinned NER model (dslim/bert-base-NER, MIT, CoNLL-2003
  PER and ORG), over the usable rows with images of every image-bearing corpus.
  Spans are cached per record, and the cache resumes.

The counts are of distinct NORMALISED NAMES, not entities: "PM Modi" and
"Narendra Modi" are two names until linking resolves both to one Wikidata item.
Linking a sample measures how far names collapse.

**What is published.** ``data/reports/gallery_counts.json`` holds aggregates
only: no name and no record id. The name table itself
(``data/interim/gallery/names.jsonl``) is derived from corpora whose licences
do not allow redistribution, so it stays local.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import GALLERY, INTERIM, MODELS, REPORTS, free_gb  # noqa: E402
from scripts import image_text, images, recover  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402
from scripts.image_text import FileLock, canonical, config_id_of, log  # noqa: E402

GALLERY_INTERIM = INTERIM / "gallery"
NAMES_PATH = GALLERY_INTERIM / "names.jsonl"
CLAIM_NER_DIR = GALLERY_INTERIM / "claim_ner"
COUNTS_REPORT = REPORTS / "gallery_counts.json"
HF_CACHE = image_text.HF_CACHE

#: Entity types the gallery holds. PERSON is verified by face. ORG gets entries
#: and references only: logo matching is a different model and a different
#: problem, and no ORG answer claims a verification.
TYPES = ("PERSON", "ORG")

#: Claim text is read from every corpus whose records carry images...
CLAIM_DATASETS = images.IMAGE_DATASETS
#: ...and the gallery is BUILT for these unless told otherwise (--scope): the
#: corpora the image_text run covers and the November presentation path. DGM4
#: is left out (a synthetic corpus not chosen for training; its captions alone
#: name 38,165 people). It can join later without a rebuild: link, fetch and
#: embed are keyed by name, QID and sha256 and skip what is done, and a pinned
#: consistency threshold keeps every existing entry as it is.
BUILD_SCOPE = ("verite", "averimatec", "factify2")
#: Points on the coverage curve: the share of mentions the N most-mentioned names hold.
COVERAGE_TOP = (100, 500, 1000, 2000, 5000, 10000, 20000)


class GalleryError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# pinned models: read from the Hub on 2026-10-21, re-checked by fetch-models
# --------------------------------------------------------------------------

#: Claim-text NER. The licence is the model card's; fetch-models refuses if it changes.
NER_MODEL = {"id": "dslim/bert-base-NER",
             "revision": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
             "licence": "mit"}
NER_FILES = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt",
             "special_tokens_map.json", "added_tokens.json")
NER_LABELS = {"PER": "PERSON", "ORG": "ORG"}
#: Claims are short; a longer text is cut here, and the cut is counted.
NER_MAX_CHARS = 1500
NER_BATCH = 64
CLEAN_RULE = ("html.unescape twice (Factify2 keeps &amp;amp;); URLs removed; "
              f"whitespace collapsed; first {NER_MAX_CHARS} characters")

#: Face models. Each is ONE file at a pinned commit, checked by sha256, beside
#: its licence file, also checked by sha256: a licence text that changes is read
#: again before anything is fetched. The OpenCV repositories carry no licence
#: in their card metadata; their LICENSE files are the statement.
#: AuraFace is fetched ONLY for the bake-off: it measures whether the bar is
#: reachable at all. Its training data is described only as "a commercial
#: dataset", and its repository also ships InsightFace-named detector,
#: landmark and gender-age files. Those are never fetched or used.
FACE_MODELS: dict[str, dict[str, str]] = {
    "yunet": {"repo": "opencv/face_detection_yunet",
              "revision": "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
              "file": "face_detection_yunet_2023mar.onnx",
              "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
              "licence": "MIT", "licence_file": "LICENSE",
              "licence_sha256": "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"},
    "sface": {"repo": "opencv/face_recognition_sface",
              "revision": "3d7082438a6e4551e840c9b2bb60b71e8da4b524",
              "file": "face_recognition_sface_2021dec.onnx",
              "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
              "licence": "Apache-2.0", "licence_file": "LICENSE",
              "licence_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"},
    "auraface": {"repo": "fal/AuraFace-v1",
                 "revision": "af6d057c9b0ec4071d4c49c80e3539258798b609",
                 "file": "glintr100.onnx",
                 "sha256": "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
                 "licence": "Apache-2.0", "card_licence": "apache-2.0", "licence_file": "LICENSE.md",
                 "licence_sha256": "609e2cb599f84aaa41d8ef29d8fdb04d164fab22e8d9292ca34a599d0f56a338"},
}
MODEL_CHOICES = ("ner", *FACE_MODELS)
#: What fetch-models takes unless told otherwise. AuraFace is asked for by name.
DEFAULT_MODELS = ("ner", "yunet", "sface")


def snapshot_dir(repo: str, revision: str) -> Path:
    """The pinned snapshot in the project's Hugging Face cache, or ModelMissing."""
    path = HF_CACHE / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise image_text.ModelMissing(
            f"{repo}@{revision[:12]} is not in {HF_CACHE}; run: python scripts/gallery.py fetch-models")
    return path


# --------------------------------------------------------------------------
# small JSONL files: one writer, fsynced, torn tails cut and kept
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Every complete line. Strict: a damaged complete line raises with its
    number, because skipping it would hide damage. A torn last line (no
    newline: the writer died mid-write) is not a row and is left for
    JsonlLedger.open() to cut."""
    if not path.is_file():
        return
    with path.open("rb") as handle:
        for n, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                return
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise GalleryError(f"{path.name}: line {n} is not JSON ({exc}); refusing to "
                                   "resume over a damaged file") from None


def _cut_torn_tail(path: Path, torn_path: Path) -> int:
    """Move a torn last line into ``torn_path`` (kept, never deleted). Returns its size."""
    if not path.is_file():
        return 0
    with open(path, "r+b") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return 0
        fh.seek(size - 1)
        if fh.read(1) == b"\n":
            return 0
        fh.seek(0)
        body = fh.read()
        cut = body.rfind(b"\n") + 1
        fragment = body[cut:]
        with open(torn_path, "ab") as torn:
            torn.write(canonical({"cut_at": utcnow(), "offset": cut, "bytes": len(fragment),
                                  "fragment": fragment.decode("utf-8", "replace")}) + b"\n")
            torn.flush()
            os.fsync(torn.fileno())
        fh.truncate(cut)
        fh.flush()
        os.fsync(fh.fileno())
    return len(fragment)


class JsonlLedger:
    """Append-only JSONL with one writer (an OS lock that dies with its holder).

    A batch of lines is written in one write and fsynced, so a kill loses at
    most the batch in flight; on the next open a torn last line is cut and kept
    in ``<stem>.torn.jsonl``."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = FileLock(path.with_name(path.name + ".lock"))
        self.torn_path = path.with_name(path.stem + ".torn.jsonl")
        self._fh = None

    def open(self, lock_wait: float = 5.0) -> list[dict[str, Any]]:
        image_text.assert_not_raw(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire(lock_wait)
        try:
            cut = _cut_torn_tail(self.path, self.torn_path)
            if cut:
                log(f"  cut a torn last line ({cut} bytes) from {self.path.name} -> {self.torn_path.name}")
            rows = list(read_jsonl(self.path))
            self._fh = open(self.path, "ab")
        except BaseException:
            self.lock.release()
            raise
        return rows

    def append_many(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self._fh.write(b"".join(canonical(r) + b"\n" for r in rows))
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def append(self, row: dict[str, Any]) -> None:
        self.append_many([row])

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.lock.release()


def write_json(path: Path, obj: Any) -> None:
    """Whole-file JSON, written atomically (``.part`` then replace)."""
    image_text.assert_not_raw(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """A derived table, regenerated whole and replaced atomically. Not a ledger."""
    image_text.assert_not_raw(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    n = 0
    with open(tmp, "wb") as fh:
        for row in rows:
            fh.write(canonical(row) + b"\n")
            n += 1
    os.replace(tmp, path)
    return n


# --------------------------------------------------------------------------
# names: one display form and one key per name
# --------------------------------------------------------------------------

#: Leading titles and roles stripped from PERSON names, which claims and
#: captions put in front of a name ("PM Modi", "Union Home Minister Amit Shah",
#: "former President Obama"). Never applied to ORG ("General Motors").
_TITLE = re.compile(
    r"^(?:(?:former|late|ex|then|current|outgoing|incumbent)[\s-]+)?"
    r"(?:(?:prime|chief|deputy|vice|union|state|foreign|home|finance|defen[cs]e|"
    r"external\s+affairs)\s+)*"
    r"(?:minister|president|pm|cm|mr|mrs|ms|miss|dr|shri|sri|smt|hon|honou?rable|senator|sen|"
    r"rep|governor|gov|mp|mla|prof|professor|sir|dame|lt|gen|general|col|capt|justice|judge)"
    r"(?:-elect)?\b\.?\s+",
    re.I)
_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
#: The Hindi honorific written after a name ("Modi ji"); "Gandhiji" is one word and stays.
_JI = re.compile(r"\s+ji$", re.I)
_ORG_ARTICLE = re.compile(r"^the\s+", re.I)
_EDGE = " \t\"'“”‘’.,;:()[]{}"
#: Keys that name no one: what the VLM writes for an unknown face, a bare
#: title, and NER noise.
GENERIC_NAMES = frozenset({
    "unidentified", "unknown", "none", "n a", "na", "anonymous", "someone", "person",
    "people", "man", "woman", "men", "women", "boy", "girl", "child", "children", "crowd",
    "president", "prime minister", "pm", "minister", "chief minister", "cm", "governor",
    "senator", "mp", "mla", "mr", "mrs", "ms", "dr", "sir"})


def fold(text: str) -> str:
    """Casefolded, accents removed (Erdoğan -> erdogan), anything else a space.

    Not image_text._normalise, which turns an accented letter into a space;
    that is right for matching OCR text and wrong for a name key."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", " ", plain.casefold()).strip()


def display_form(name: str, entity_type: str) -> str:
    """The name as it is searched for: titles, possessives and edge punctuation
    removed, case kept."""
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name)).strip(_EDGE)
    s = _POSSESSIVE.sub("", s).strip(_EDGE)
    if entity_type == "PERSON":
        for _ in range(3):  # "former PM Dr. Manmohan Singh"
            stripped = _TITLE.sub("", s, count=1)
            if stripped == s:
                break
            s = stripped
        s = _JI.sub("", s)
    else:
        s = _ORG_ARTICLE.sub("", s)
    return s.strip(_EDGE)


def name_key(name: str, entity_type: str) -> str | None:
    """The key a name is counted under, or None when it names no one:
    a subword fragment, a lone letter, digits only, more than eight words, or
    a generic word."""
    if name.startswith("##"):
        return None
    key = fold(display_form(name, entity_type))
    if len(key) < 2 or not re.search(r"[a-z]", key) or len(key.split()) > 8 or key in GENERIC_NAMES:
        return None
    return key


# --------------------------------------------------------------------------
# the VLM's names: the current image_text run directory
# --------------------------------------------------------------------------


def run_dirs(root: Path | None = None) -> list[tuple[Path, int | None]]:
    """Every image_text run directory with its row schema."""
    root = image_text.OUT_ROOT if root is None else root
    out = []
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if (p / "config.json").is_file()):
            config = json.loads((d / "config.json").read_text(encoding="utf-8"))
            out.append((d, config.get("schema_version")))
    return out


def current_run_dir(root: Path | None = None) -> Path:
    """The one run directory written under the current row schema.

    Older directories hold earlier pilots under earlier parsers. Their images
    are re-run under the current one, and schema-1 rows cannot tell a forced
    line from a free one (entities_of refuses them)."""
    found = [d for d, schema in run_dirs(root) if schema == image_text.SCHEMA_VERSION]
    if not found:
        raise GalleryError(
            f"no image_text run directory with row schema {image_text.SCHEMA}; rsync the "
            "server's current one into data/interim/image_text/ first")
    if len(found) > 1:
        raise GalleryError(f"several current-schema run directories ({', '.join(d.name for d in found)}); "
                           "name one with --ledger-dir")
    return found[0]


@dataclass
class LedgerRead:
    """The last row of every image in one image_text ledger, reduced to its names."""

    dataset: str
    images: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: int = 0
    torn_tail_bytes: int = 0
    #: rows written before forced marks existed: their names cannot be read
    unmarked: int = 0


def read_ledger(path: Path, dataset: str) -> LedgerRead:
    """PERSON and ORG names of every image in an image_text ledger, from the
    image's LAST row (a retried image has several), with the evidence the
    recognition probe uses: whether the independent OCR holds the name."""
    from scripts.processed_loader import SupervisionError

    out = LedgerRead(dataset)

    def keep(row: dict[str, Any]) -> bool:
        out.rows += 1
        try:
            found = list(image_text.entities_of(row, forced="mark"))
        except SupervisionError:
            out.unmarked += 1
            found = []
        ocr = row.get("ocr") or {}
        ocr_text = None if ocr.get("error") else (ocr.get("text") or "")
        entities = row.get("entities") or {}
        out.images[row["sha256"]] = {
            "record_ids": tuple(row["record_ids"]), "status": row["status"],
            "entity_error": entities.get("error"),
            "names": [{"type": e["category"], "name": e["name"], "basis": e["basis"],
                       "forced": e["forced"], "confidence": e["confidence"],
                       "evidence": image_text._evidence(e, ocr_text)}
                      for e in found if e["category"] in TYPES]}
        return False

    _summaries, _kept, out.torn_tail_bytes = image_text.scan_ledger(path, dataset=dataset, keep=keep)
    return out


# --------------------------------------------------------------------------
# claim text: NER, cached per record
# --------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)


def clean_claim(text: str | None) -> tuple[str, bool]:
    """(text NER reads, whether it was cut)."""
    t = html.unescape(html.unescape(text or ""))
    t = re.sub(r"\s+", " ", _URL.sub(" ", t)).strip()
    return t[:NER_MAX_CHARS], len(t) > NER_MAX_CHARS


def claim_rows(dataset: str) -> Iterator[tuple[str, str, bool]]:
    """(record_id, cleaned text, cut) for every usable row with an image."""
    for row in image_text.usable_rows(dataset):
        text, cut = clean_claim(row.get("text"))
        yield str(row["record_id"]), text, cut


def ner_config() -> dict[str, Any]:
    return {"model": dict(NER_MODEL), "files": list(NER_FILES), "aggregation": "first",
            "labels": dict(NER_LABELS), "clean": CLEAN_RULE, "max_chars": NER_MAX_CHARS}


class Ner:
    """dslim/bert-base-NER at the pinned revision, offline: PER and ORG spans.
    Loaded on the first call, so a run with nothing new to read never loads it."""

    def __init__(self, device: str | None = None):
        self.device = device
        self.pipe = None

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        snap = snapshot_dir(NER_MODEL["id"], NER_MODEL["revision"])
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(snap), local_files_only=True)
        self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer,
                             aggregation_strategy="first", device=0 if device == "cuda" else -1)
        self.device = device
        log(f"  NER {NER_MODEL['id']}@{NER_MODEL['revision'][:12]} on {device}")

    def __call__(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if self.pipe is None:
            self.load()
        out: list[list[dict[str, Any]]] = [[] for _ in texts]
        todo = [i for i, t in enumerate(texts) if t.strip()]
        found = self.pipe([texts[i] for i in todo], batch_size=NER_BATCH) if todo else []
        for i, ents in zip(todo, found):
            text = texts[i]
            out[i] = [{"type": NER_LABELS[e["entity_group"]], "text": text[e["start"]:e["end"]],
                       "start": int(e["start"]), "end": int(e["end"]),
                       "score": round(float(e["score"]), 4)}
                      for e in ents if e["entity_group"] in NER_LABELS]
        return out


def claim_spans(dataset: str, ner: Callable[[Sequence[str]], list[list[dict[str, Any]]]], *,
                rows: Iterable[tuple[str, str, bool]] | None = None, out_dir: Path | None = None,
                chunk: int = 2048) -> dict[str, dict[str, Any]]:
    """record_id -> {spans, cut} for one corpus, from the cache where it has
    them. A record already read is never read again; a kill loses at most the
    chunk in flight."""
    directory = out_dir or (CLAIM_NER_DIR / config_id_of(ner_config()))
    config_path = directory / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != ner_config():
            raise GalleryError(f"{config_path} does not match the NER configuration; refusing to mix")
    else:
        write_json(config_path, ner_config())
    ledger = JsonlLedger(directory / f"{dataset}.jsonl")
    done = {r["record_id"]: r for r in ledger.open()}
    try:
        source = claim_rows(dataset) if rows is None else rows
        pending = [(rid, text, cut) for rid, text, cut in source if rid not in done]
        if pending:
            log(f"  [{dataset}] NER: {len(done):,} records cached, {len(pending):,} to read")
        for start in range(0, len(pending), chunk):
            part = pending[start:start + chunk]
            found = ner([text for _rid, text, _cut in part])
            batch = [{"record_id": rid, "spans": spans, "cut": cut}
                     for (rid, _text, cut), spans in zip(part, found)]
            ledger.append_many(batch)
            done.update((r["record_id"], r) for r in batch)
            if len(pending) > chunk:
                log(f"  [{dataset}] NER {min(start + chunk, len(pending)):,}/{len(pending):,}")
    finally:
        ledger.close()
    return done


# --------------------------------------------------------------------------
# the name table and the counts
# --------------------------------------------------------------------------


class NameTable:
    """Every distinct (type, key), with where it was seen."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    def _row(self, entity_type: str, key: str) -> dict[str, Any]:
        return self.rows.setdefault((entity_type, key), {
            "type": entity_type, "key": key, "forms": Counter(), "vlm": {}, "claims": {}})

    def add_vlm(self, dataset: str, sha256: str, name: dict[str, Any]) -> bool:
        key = name_key(name["name"], name["type"])
        if key is None:
            return False
        row = self._row(name["type"], key)
        row["forms"][display_form(name["name"], name["type"])] += 1
        v = row["vlm"].setdefault(dataset, {"images": set(), "mentions": 0, "free": 0, "forced": 0,
                                            "evidence": Counter(), "basis": Counter()})
        v["images"].add(sha256)
        v["mentions"] += 1
        v["forced" if name["forced"] else "free"] += 1
        v["evidence"][name["evidence"]] += 1
        v["basis"][name["basis"]] += 1
        return True

    def add_claim(self, dataset: str, record_id: str, span: dict[str, Any]) -> bool:
        key = name_key(span["text"], span["type"])
        if key is None:
            return False
        row = self._row(span["type"], key)
        row["forms"][display_form(span["text"], span["type"])] += 1
        c = row["claims"].setdefault(dataset, {"records": set(), "mentions": 0})
        c["records"].add(record_id)
        c["mentions"] += 1
        return True

    @staticmethod
    def mentions(row: dict[str, Any], source: str) -> int:
        return sum(v["mentions"] for v in row[source].values())

    def serialised(self) -> Iterator[dict[str, Any]]:
        """Rows for names.jsonl, most-mentioned first."""
        def total(r: dict[str, Any]) -> int:
            return self.mentions(r, "vlm") + self.mentions(r, "claims")

        for r in sorted(self.rows.values(), key=lambda r: (-total(r), r["type"], r["key"])):
            display = sorted(r["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            yield {"type": r["type"], "key": r["key"], "display": display,
                   "forms": dict(r["forms"].most_common()),
                   "mentions": {"vlm": self.mentions(r, "vlm"), "claims": self.mentions(r, "claims")},
                   "vlm": {ds: {"images": len(v["images"]), "mentions": v["mentions"],
                                "free": v["free"], "forced": v["forced"],
                                "evidence": dict(v["evidence"]), "basis": dict(v["basis"])}
                           for ds, v in sorted(r["vlm"].items())},
                   "claims": {ds: {"records": len(c["records"]), "mentions": c["mentions"]}
                              for ds, c in sorted(r["claims"].items())}}


def coverage(mentions: Sequence[int]) -> dict[str, float | None]:
    """The share of all mentions held by the N most-mentioned names."""
    ordered = sorted((m for m in mentions if m), reverse=True)
    total = sum(ordered)
    out: dict[str, float | None] = {}
    running, i = 0, 0
    for n in COVERAGE_TOP:
        while i < min(n, len(ordered)):
            running += ordered[i]
            i += 1
        out[f"top_{n}"] = image_text._ratio(running, total, 4)
        if n >= len(ordered):
            break
    return out


def in_scope(row: dict[str, Any], scope: Sequence[str] = BUILD_SCOPE) -> bool:
    """A name the gallery is built for: the VLM gave it on an image of, or the
    claim text of, a corpus in ``scope``. Per corpus on both sides, so a
    ledger that grows to cover another corpus does not widen the scope."""
    return any(ds in scope for ds in row["vlm"]) or any(ds in scope for ds in row["claims"])


def scope_mentions(row: dict[str, Any], scope: Sequence[str] = BUILD_SCOPE) -> int:
    return (sum(v["mentions"] for ds, v in row["vlm"].items() if ds in scope)
            + sum(c["mentions"] for ds, c in row["claims"].items() if ds in scope))


def type_counts(rows: Sequence[dict[str, Any]], ledgers: Sequence[str],
                claims: Sequence[str]) -> dict[str, Any]:
    """Aggregates for one entity type, from serialised name rows."""
    vlm = [r for r in rows if r["mentions"]["vlm"]]
    clm = [r for r in rows if r["mentions"]["claims"]]
    evidence: Counter = Counter()
    lines: Counter = Counter()
    for r in vlm:
        for v in r["vlm"].values():
            evidence.update(v["evidence"])
            lines.update({"free": v["free"], "forced": v["forced"]})
    scoped = [r for r in rows if in_scope(r)]
    return {
        "distinct_names": len(rows),
        "overlap": {"vlm_only": sum(1 for r in rows if r["mentions"]["vlm"] and not r["mentions"]["claims"]),
                    "claims_only": sum(1 for r in rows if r["mentions"]["claims"] and not r["mentions"]["vlm"]),
                    "both": sum(1 for r in rows if r["mentions"]["vlm"] and r["mentions"]["claims"])},
        "build_scope": {"datasets": list(BUILD_SCOPE), "distinct_names": len(scoped),
                        "named_once": sum(1 for r in scoped if scope_mentions(r) == 1)},
        "vlm": {
            "distinct_names": len(vlm),
            "by_dataset": {ds: sum(1 for r in vlm if ds in r["vlm"]) for ds in ledgers},
            "mentions": sum(r["mentions"]["vlm"] for r in vlm),
            "mentions_by_evidence": dict(sorted(evidence.items())),
            "mentions_by_line": {"free": lines["free"], "forced": lines["forced"]},
            # the names a gallery has to verify: produced from appearance alone
            "distinct_names_not_written": sum(1 for r in vlm if any(
                v["evidence"].get("not_written") for v in r["vlm"].values())),
            "distinct_names_only_on_forced_lines": sum(1 for r in vlm if not any(
                v["free"] for v in r["vlm"].values())),
            "coverage": coverage([r["mentions"]["vlm"] for r in vlm])},
        "claims": {
            "distinct_names": len(clm),
            "by_dataset": {ds: sum(1 for r in clm if ds in r["claims"]) for ds in claims},
            "mentions": sum(r["mentions"]["claims"] for r in clm),
            "coverage": coverage([r["mentions"]["claims"] for r in clm])},
    }


def ledger_counts(read: LedgerRead, total_images: int | None) -> dict[str, Any]:
    imgs = read.images.values()
    with_names = Counter()
    for img in imgs:
        for t in {n["type"] for n in img["names"]}:
            with_names[t] += 1
    return {"rows": read.rows, "images": len(read.images),
            "images_in_worklist": total_images,
            "share_done": image_text._ratio(len(read.images), total_images, 4) if total_images else None,
            "status": dict(sorted(Counter(i["status"] for i in imgs).items())),
            "entity_errors": dict(sorted(Counter(i["entity_error"] for i in imgs
                                                 if i["entity_error"]).items())),
            "images_naming": {t: with_names[t] for t in TYPES},
            "rows_without_forced_marks": read.unmarked,
            "torn_tail_bytes": read.torn_tail_bytes}


#: Prefix sizes the growth fit uses; the smallest are left out (too noisy).
GROWTH_POINTS = (1000, 1500, 2000, 2500)


def name_growth(read: LedgerRead, total: int | None) -> dict[str, Any] | None:
    """How many distinct VLM names a FINISHED run would hold: an extrapolation.

    The run takes images in sha256 order, which is pseudo-random, so the
    images done are a random sample of the work list. Distinct names over
    growing prefixes are fitted to V(n) = K n^beta (Heaps' law) and the fit is
    evaluated at the work list's size. This is a projection far past the
    observed range, and it says so: the full-build budget is planned from it,
    and the finished run replaces it."""
    order = sorted(read.images)
    if not total or len(order) >= total or len(order) < GROWTH_POINTS[-1]:
        return None
    out: dict[str, Any] = {"method": ("V(n) = K n^beta fitted by least squares in log-log over "
                                      f"prefixes {list(GROWTH_POINTS)} and all {len(order):,} images "
                                      "done (sha256 order: a random sample), evaluated at the "
                                      "work list's size; an extrapolation, not a measurement"),
                           "images_done": len(order), "images_in_worklist": total}
    for t in TYPES:
        for scope in ("all", "not_written"):
            seen: set[str] = set()
            points = []
            for i, sha in enumerate(order, 1):
                for n in read.images[sha]["names"]:
                    if n["type"] == t and (scope == "all" or n["evidence"] == "not_written"):
                        key = name_key(n["name"], t)
                        if key:
                            seen.add(key)
                if i in GROWTH_POINTS or i == len(order):
                    points.append((i, len(seen)))
            if len(points) < 3 or any(v == 0 for _i, v in points):
                continue
            xs = [math.log(i) for i, _v in points]
            ys = [math.log(v) for _i, v in points]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            out[f"{t}_{scope}"] = {"now": points[-1][1], "beta": round(beta, 3),
                                   "projected": round(math.exp(my + beta * (math.log(total) - mx)))}
    return out


def worklist_sizes(run_dir: Path) -> dict[str, int]:
    """Distinct images per dataset as the run itself counted them (its pilot report)."""
    pilot = run_dir / "pilot.json"
    if not pilot.is_file():
        return {}
    doc = json.loads(pilot.read_text(encoding="utf-8"))
    found = ((doc.get("count_corrections") or {}).get("datasets") or {})
    return {ds: v["distinct_images"] for ds, v in found.items() if v.get("distinct_images")}


def count(run_dir: Path, *, claims: Sequence[str], ner: Callable | None,
          claim_source: Callable[[str], Iterable[tuple[str, str, bool]]] | None = None,
          ner_dir: Path | None = None) -> tuple[NameTable, dict[str, Any]]:
    """The name table, and the report on it. ``ner`` None reads no claim text."""
    table = NameTable()
    ledgers: dict[str, LedgerRead] = {}
    for ds in image_text.ORDER:
        path = run_dir / f"{ds}.jsonl"
        if path.is_file():
            read = read_ledger(path, ds)
            ledgers[ds] = read
            for sha, img in read.images.items():
                for name in img["names"]:
                    table.add_vlm(ds, sha, name)
            log(f"  [{ds}] {len(read.images):,} images read from {run_dir.name}")
    ner_stats: dict[str, Any] = {}
    for ds in (claims if ner is not None else ()):
        found = claim_spans(ds, ner, rows=claim_source(ds) if claim_source else None, out_dir=ner_dir)
        per = Counter()
        for rid, row in found.items():
            per["records"] += 1
            per["cut"] += bool(row.get("cut"))
            kinds = set()
            for span in row["spans"]:
                per[f"spans_{span['type']}"] += 1
                if table.add_claim(ds, rid, span):
                    kinds.add(span["type"])
            for t in kinds:
                per[f"records_naming_{t}"] += 1
        ner_stats[ds] = dict(sorted(per.items()))
    rows = list(table.serialised())
    sizes = worklist_sizes(run_dir)
    report = {
        "generated_at": utcnow(),
        "what": ("distinct NORMALISED NAMES of people and organisations in our corpora, before "
                 "anything is fetched. Names, not entities: a surname and the full name count "
                 "as two until linking resolves both to one Wikidata item. Aggregates only; the "
                 "name table stays in data/interim/gallery/."),
        "image_text": {
            "run": run_dir.name, "row_schema": image_text.SCHEMA,
            "not_read": {d.name: f"row schema {s}: an earlier run under an older parser, superseded"
                         for d, s in run_dirs(run_dir.parent) if d != run_dir},
            "datasets": {ds: ledger_counts(read, sizes.get(ds)) for ds, read in ledgers.items()},
            "projection": {ds: g for ds, read in ledgers.items()
                           if (g := name_growth(read, sizes.get(ds))) is not None},
            "names_are": ("the VLM's PERSON and ORG names, every line read through "
                          "entities_of(forced='mark'); evidence is the independent OCR's "
                          "(written / partly_written / not_written / no_ocr)")},
        "claims": {
            "ner": {"model": NER_MODEL["id"], "revision": NER_MODEL["revision"],
                    "licence": NER_MODEL["licence"], "config_id": config_id_of(ner_config()),
                    "clean": CLEAN_RULE},
            "rows_read": "usable processed rows that carry at least one image",
            "datasets": ner_stats},
        "names": {t: type_counts([r for r in rows if r["type"] == t], list(ledgers), list(ner_stats))
                  for t in TYPES},
        "verite": ("VERITE's names count towards gallery membership, as an evaluation-time "
                   "lookup of the claimed name would. Nothing is ever tuned on VERITE: the "
                   "calibration reads Factify2 train only."),
    }
    return table, report


def cmd_count(args) -> int:
    run_dir = Path(args.ledger_dir) if args.ledger_dir else current_run_dir()
    if not run_dir.is_absolute() and not run_dir.is_dir():
        run_dir = image_text.OUT_ROOT / run_dir
    from scripts.processed_loader import records_path

    claims = [ds for ds in args.claims if records_path(ds).is_file()]
    missing = sorted(set(args.claims) - set(claims))
    if missing:
        log(f"  no processed records for {', '.join(missing)}: their claims are not read")
    table, report = count(run_dir, claims=claims, ner=None if args.no_claims else Ner(args.device))
    n = write_jsonl(NAMES_PATH, table.serialised())
    write_json(COUNTS_REPORT, report)
    for t in TYPES:
        c = report["names"][t]
        log(f"{t}: {c['distinct_names']:,} distinct names "
            f"(VLM {c['vlm']['distinct_names']:,}, of which {c['vlm']['distinct_names_not_written']:,} "
            f"from appearance alone; claims {c['claims']['distinct_names']:,}; both "
            f"{c['overlap']['both']:,}); build scope {c['build_scope']['distinct_names']:,}")
    log(f"wrote {n:,} names to {NAMES_PATH} and the aggregates to {COUNTS_REPORT}")
    return 0


# --------------------------------------------------------------------------
# fetch-models: the one command that downloads weights
# --------------------------------------------------------------------------


def cmd_fetch_models(args) -> int:
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    from scripts.manifest import sha256

    MODELS.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for which in args.only:
        if which == "ner":
            info = api.model_info(NER_MODEL["id"], revision=NER_MODEL["revision"], files_metadata=True)
            if info.sha != NER_MODEL["revision"]:
                raise SystemExit(f"the Hub resolved {NER_MODEL['revision']} to {info.sha}; refusing")
            licence = getattr(info.card_data, "license", None) if info.card_data else None
            if licence != NER_MODEL["licence"]:
                raise SystemExit(f"{NER_MODEL['id']} now states licence {licence!r}, not "
                                 f"{NER_MODEL['licence']!r}; read it before fetching")
            wanted = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in NER_FILES)]
            need = sum(s.size or 0 for s in wanted) / 1024 ** 3
            if free_gb(MODELS) < 1.5 * need:
                raise SystemExit(f"refusing: {free_gb(MODELS):.1f} GiB free, {1.5 * need:.2f} needed")
            snap = Path(snapshot_download(NER_MODEL["id"], revision=NER_MODEL["revision"],
                                          cache_dir=str(HF_CACHE), allow_patterns=list(NER_FILES)))
            bad = [s.rfilename for s in wanted
                   if not (snap / s.rfilename).is_file()
                   or (snap / s.rfilename).stat().st_size != s.size
                   or (s.lfs is not None and sha256(snap / s.rfilename) != s.lfs.sha256)]
            if bad:
                raise SystemExit(f"downloaded NER files do not match the Hub's metadata: {bad}")
            log(f"{NER_MODEL['id']}@{NER_MODEL['revision'][:12]} ({licence}): "
                f"{len(wanted)} files verified")
            continue
        spec = FACE_MODELS[which]
        info = api.model_info(spec["repo"], revision=spec["revision"])
        if info.sha != spec["revision"]:
            raise SystemExit(f"the Hub resolved {spec['repo']}@{spec['revision']} to {info.sha}; refusing")
        card = getattr(info.card_data, "license", None) if info.card_data else None
        if spec.get("card_licence") and card != spec["card_licence"]:
            raise SystemExit(f"{spec['repo']} now states licence {card!r}; read it before fetching")
        licence = Path(hf_hub_download(spec["repo"], spec["licence_file"], revision=spec["revision"],
                                       cache_dir=str(HF_CACHE)))
        if sha256(licence) != spec["licence_sha256"]:
            raise SystemExit(f"{spec['repo']}: {spec['licence_file']} is not the text read on "
                             "2026-10-21; read it before fetching")
        path = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                                    cache_dir=str(HF_CACHE)))
        if sha256(path) != spec["sha256"]:
            raise SystemExit(f"{spec['repo']}/{spec['file']}: sha256 mismatch; refusing")
        log(f"{spec['repo']}/{spec['file']}@{spec['revision'][:12]} ({spec['licence']}): verified")
    return 0


def face_model_path(which: str) -> Path:
    """The pinned file of one face model, re-hashed: other bytes are another experiment."""
    from scripts.manifest import sha256

    spec = FACE_MODELS[which]
    path = snapshot_dir(spec["repo"], spec["revision"]) / spec["file"]
    if not path.is_file() or sha256(path) != spec["sha256"]:
        raise image_text.ModelMissing(f"{spec['file']} is missing or not the pinned bytes; "
                                      "run: python scripts/gallery.py fetch-models")
    return path


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

#: Tags every provenance row, link and entity the sample pass writes...
PASS_ID = "gallery-2026-10"
#: ...and the full build (link/fetch --all), so either can be told apart later.
BUILD_PASS_ID = "gallery-build-2026-10"
#: Tags the link rows a person's hold or release writes (offline).
HELD_PASS_ID = "gallery-holds-2026-10"
LINKS_PATH = GALLERY / "links.jsonl"
WIKIDATA_PATH = GALLERY / "wikidata.jsonl"
ENTITIES_PATH = GALLERY / "entities.jsonl"
PROVENANCE_PATH = GALLERY / "provenance.jsonl"
MEDIA_DIR = GALLERY / "media"
REJECTED_DIR = GALLERY / "rejected"
FACES_DIR = GALLERY / "faces"
SAMPLE_PATH = GALLERY_INTERIM / "sample.json"
QUERY_FACES_DIR = GALLERY_INTERIM / "faces"
CALIBRATION_DIR = GALLERY_INTERIM / "calibration"
REVIEW_DIR = GALLERY_INTERIM / "review"
SAMPLE_REPORT = REPORTS / "gallery_sample.json"
SEED = image_text.SEED


def latest(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    """Append-only files: the last row for each key wins."""
    out: dict[Any, dict[str, Any]] = {}
    for row in rows:
        out[key(row)] = row
    return out


# --------------------------------------------------------------------------
# the network: Wikidata and Commons, one worker, at Wikimedia's published pace
# --------------------------------------------------------------------------

#: Hosts spaced at WIKIMEDIA_DELAY, named one by one: every other host keeps
#: recover.HOST_DELAY (5 s). Wikimedia's robot policy
#: (wikitech.wikimedia.org/wiki/Robot_policy, edited 2026-03-16, read
#: 2026-10-21) allows an unauthenticated Action API client "1 at a time, and
#: below 5 requests per second overall", and media "a total concurrency of at
#: most 2" within 25 Mbps. One request in flight, at most one start per second
#: per host, is inside both: two API hosts make at most 2 per second.
WIKIMEDIA_HOSTS = {"www.wikidata.org": "action_api", "commons.wikimedia.org": "action_api",
                   "upload.wikimedia.org": "media", "thumb.wikimedia.org": "media"}
WIKIMEDIA_DELAY = 1.0
#: The same policy: "if your request takes more than 1 second to serve, please
#: wait 5 seconds before making another request" (Action API).
SLOW_API_S = 1.0
SLOW_API_PAUSE = 5.0
#: "When we reply with a 429 Too Many Requests status code respect the delay
#: specified by the Retry-After header": the whole host waits, not one request.
#: A 429 without the header waits a minute.
RATE_LIMITED_PAUSE = 60.0
#: Its rule for other wikimedia.org services, kept here for the answers that
#: say the service itself is struggling (502, 503, 504): "Pause crawling for at
#: least 15 minutes if you receive a 5xx status code". A 500 on one file is that
#: file's (a thumbnail that cannot be rendered) and holds only its own retries.
#: The sample's 791 requests met no 5xx at all.
SERVER_ERROR_PAUSE = 900.0
OVERLOAD_STATUSES = (502, 503, 504)

UA_CONTACT_ENV = "PE_FAKE_NEWS_UA_CONTACT"
#: The repository's own anonymised URL. Wikimedia's User-Agent policy accepts a
#: URL as contact, and this one keeps the review blind. A real contact can be
#: supplied through the environment and never enters tracked code.
UA_CONTACT_DEFAULT = "https://github.com/anonymousz77/Pe_fake_news"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
#: Non-interactive API use asks the servers to refuse it when they are lagged
#: (mediawiki.org, API:Etiquette); a refusal is waited out, then retried.
MAXLAG = 5
MAXLAG_RETRIES = 3
#: API errors that say "not now", sent with status 200, and the first wait for
#: each (doubled on every retry). The sample pass met the search one 10 times in
#: 252 searches; the server's Retry-After wins when it is longer.
TRANSIENT_API_ERRORS = {"maxlag": float(MAXLAG), "cirrussearch-too-busy-error": 30.0,
                        "ratelimited": 60.0}
#: The two that are about the host's load or our rate (API:Etiquette): the whole
#: host waits, as for a 429. A busy search pool only delays the refused search.
HOST_WIDE_API_ERRORS = ("maxlag", "ratelimited")
#: A Wikimedia standard thumbnail width. Bots must use a standard size or the
#: original (mediawiki.org, "Common thumbnail sizes", read 2026-10-20), and a
#: fixed width keeps every reference comparable.
THUMB_WIDTH = 960
SEARCH_LIMIT = 10
COMMONS_SEARCH_LIMIT = 10
#: Reference candidates tried per entity: Wikidata's own image first, then
#: files whose structured data says they depict the entity, then its category.
CANDIDATES = {"PERSON": 6, "ORG": 2}
BATCH = 50  # wbgetentities ids, imageinfo titles
Q_HUMAN = "Q5"
Q_ORGANISATION = "Q43229"
#: Subclass-of steps walked upward from an item's classes before giving up.
CLASS_DEPTH = 12
#: A name is ambiguous when its second exact candidate has at least a third of
#: the first's sitelinks: neither dominates.
AMBIGUITY_RATIO = 3.0
EXTMETADATA = ("LicenseShortName", "LicenseUrl", "License", "UsageTerms", "AttributionRequired",
               "Copyrighted", "Restrictions", "Artist", "Credit", "DateTimeOriginal")


def user_agent() -> str:
    """Wikimedia's policy: client/version (contact) library/version, with "bot"."""
    import requests

    contact = os.environ.get(UA_CONTACT_ENV) or UA_CONTACT_DEFAULT
    return (f"Pe_Fake_News_Dec-gallery-bot/1.1 ({contact}; research; one request at a time, at "
            f"most one per {WIKIMEDIA_DELAY:g} s per host) python-requests/{requests.__version__}")


def retry_after(resp) -> float | None:
    try:
        return float(resp.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


class WikimediaWorker(recover.Worker):
    """recover.Worker at the pace Wikimedia publishes, on the hosts named in
    WIKIMEDIA_HOSTS only: one second between request starts, five after an API
    answer that took over a second, and for the whole host the Retry-After of
    a 429 (or of a maxlag refusal) and fifteen minutes after a 502, 503 or 504.
    Every other host keeps recover.HOST_DELAY."""

    def spacing(self, host: str) -> float:
        return WIKIMEDIA_DELAY if host in WIKIMEDIA_HOSTS else self.delay

    def pause_after(self, host: str, resp) -> float:
        kind = WIKIMEDIA_HOSTS.get(host)
        if kind is None:
            return 0.0
        if resp.status == 429:
            return max(retry_after(resp) or 0.0, RATE_LIMITED_PAUSE)
        if resp.status in OVERLOAD_STATUSES:
            return SERVER_ERROR_PAUSE
        code = resp.headers.get("mediawiki-api-error")
        if code in HOST_WIDE_API_ERRORS:
            return max(retry_after(resp) or 0.0, TRANSIENT_API_ERRORS[code])
        if kind == "action_api" and resp.elapsed > SLOW_API_S:
            return SLOW_API_PAUSE
        return 0.0


class ProvenanceLog:
    """Every request this block makes, one row each (URL, host, status,
    outcome, reason), in the gallery's own ledger. recover.get() logs its
    redirects and retries through ``attempt``; callers log the final outcome."""

    def __init__(self, ledger: JsonlLedger, pass_id: str = PASS_ID):
        self.ledger = ledger
        #: tags every row written through this log's Net: the sample's or the build's
        self.pass_id = pass_id

    def attempt(self, unit: str, route: str, resp, outcome: str, reason: str | None,
                **extra: Any) -> None:
        from scripts.recover import host_of

        self.ledger.append({"event": "attempt", "pass": self.pass_id, "at": utcnow(), "stage": route,
                            "key": unit, "url": resp.url, "host": host_of(resp.url),
                            "http_status": resp.status, "outcome": outcome,
                            "attempt_reason": reason, "bytes": len(resp.body),
                            "elapsed_s": round(resp.elapsed, 3), **extra})


@dataclass
class Net:
    """The single worker, its clock and the log: real, or a test's fakes.

    ONE worker per Net, kept across runs: its per-host spacing must hold from
    one phase to the next (search, entities, type check), not restart with
    each."""

    make_worker: Callable[[], Any]
    log: ProvenanceLog
    clock: Callable[[], float]
    requests: Counter = field(default_factory=Counter)
    #: Why the network may not be used again in this process (a spent budget).
    stopped: str | None = None
    _worker: Any = field(default=None, repr=False)

    def run(self, gens: Iterable[Any]) -> None:
        if self._worker is None:
            self._worker = self.make_worker()
        worker = self._worker
        before = (worker.requests, worker.bytes_in)
        for gen in gens:
            worker.start(gen)
        try:
            worker.run()
        finally:
            self.requests["total"] += worker.requests - before[0]
            self.requests["bytes"] += worker.bytes_in - before[1]


def live_net(log: ProvenanceLog, budget_bytes: int | None = None) -> Net:
    import time

    import requests

    from scripts import recover

    session = requests.Session()
    session.headers.clear()  # every request carries its own headers
    return Net(lambda: WikimediaWorker(session, delay=recover.HOST_DELAY, budget_bytes=budget_bytes),
               log, time.monotonic)


def api_json(net: Net, base: str, params: dict[str, Any], key: str, stage: str):
    """GET one API call through the worker; (document, None) or (None, reason).

    A 200 is not the answer: the body must be JSON, and a JSON error -- maxlag
    above all, which Wikimedia sends with status 200 -- is a failure (maxlag is
    waited out and retried)."""
    from urllib.parse import urlencode

    from scripts import recover

    url = f"{base}?{urlencode(params)}"
    # the robot policy asks for gzip on everything but media
    headers = {"User-Agent": user_agent(), "Accept": "application/json", "Accept-Encoding": "gzip"}
    for attempt in range(MAXLAG_RETRIES + 1):
        resp, _chain = yield from recover.get(url, headers, net.log, key, stage, net.clock)
        if resp.error or resp.status != 200:
            reason = resp.error or f"http_{resp.status}"
            net.log.attempt(key, stage, resp, "failed", reason)
            return None, reason
        try:
            doc = json.loads(resp.body)
        except ValueError:
            net.log.attempt(key, stage, resp, "failed", "not_json")
            return None, "not_json"
        error = (doc.get("error") or {}).get("code") if isinstance(doc, dict) else None
        if error in TRANSIENT_API_ERRORS and attempt < MAXLAG_RETRIES:
            wait = TRANSIENT_API_ERRORS[error] * 2 ** attempt
            try:
                wait = max(wait, float(resp.headers.get("retry-after") or 0))
            except ValueError:
                pass
            net.log.attempt(key, stage, resp, "retrying", error, retry_in_s=wait)
            yield recover.Req(None, None, net.clock() + wait)  # a pure wait
            continue
        if error:
            net.log.attempt(key, stage, resp, "failed", f"api_error:{error}")
            return None, f"api_error:{error}"
        net.log.attempt(key, stage, resp, "ok", None)
        return doc, None
    return None, f"api_error:{error}"


# --------------------------------------------------------------------------
# linking names to Wikidata items
# --------------------------------------------------------------------------


def claim_values(entity: dict[str, Any], prop: str) -> list[str]:
    out = []
    for claim in (entity.get("claims") or {}).get(prop, []):
        value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(value, dict) and "id" in value:
            out.append(value["id"])
        elif isinstance(value, str):
            out.append(value)
    return out


#: Languages asked for. "mul" holds a label every language shares: since 2024
#: Wikidata keeps many people's names there and not under "en" (on the sample,
#: 17 of 556 items had no English label and none had an English alias).
LANGUAGES = ("en", "mul")


def entity_summary(entity: dict[str, Any]) -> dict[str, Any]:
    """What the gallery keeps of a Wikidata item (CC0)."""
    def first(field_: str) -> str | None:
        for lang in LANGUAGES:
            value = ((entity.get(field_) or {}).get(lang) or {}).get("value")
            if value:
                return value
        return None

    aliases = [a["value"] for lang in LANGUAGES for a in (entity.get("aliases") or {}).get(lang, [])]
    return {"qid": entity["id"], "label": first("labels"), "description": first("descriptions"),
            "aliases": list(dict.fromkeys(aliases)),
            "p31": claim_values(entity, "P31"), "sitelinks": len(entity.get("sitelinks") or {}),
            "images": claim_values(entity, "P18"), "logos": claim_values(entity, "P154"),
            "commons_category": next(iter(claim_values(entity, "P373")), None)}


def same_name(text: str, key: str) -> bool:
    """A label or alias that IS the name: equal once folded, or once spaces are
    dropped too ("ArvindKejriwal", a hashtag, is "Arvind Kejriwal")."""
    folded = fold(text)
    return bool(folded) and (folded == key or folded.replace(" ", "") == key.replace(" ", ""))


def is_type(entity: dict[str, Any], entity_type: str, organisations: set[str]) -> bool:
    if entity_type == "PERSON":
        return Q_HUMAN in entity["p31"]
    return entity["qid"] in organisations


def exact_candidates(results: Sequence[dict[str, Any]], key: str) -> list[str]:
    out: list[str] = []
    for r in results:
        texts = [r.get("label") or "", (r.get("match") or {}).get("text") or "",
                 *(r.get("aliases") or [])]
        if any(same_name(t, key) for t in texts) and r["id"] not in out:
            out.append(r["id"])
    return out


def decide(entity_type: str, key: str, results: Sequence[dict[str, Any]] | None,
           error: str | None, entities: dict[str, dict[str, Any]], organisations: set[str],
           unchecked: set[str], failed: set[str] = frozenset()) -> dict[str, Any]:
    """Which item a name is, or why it is none.

    Only an EXACT label or alias counts: search ranks prefixes too, and a name
    linked to the wrong person verifies the wrong face. Among exact candidates
    of the right type, the most-linked wins; one that does not dominate the
    next makes the name ``ambiguous``, kept, chosen and marked. A candidate
    whose data never arrived (``failed``) makes the name an error to retry,
    never a "no"."""
    if results is None:
        return {"status": "error", "reason": error}
    if not results:
        return {"status": "no_result"}
    exact = exact_candidates(results, key)
    if not exact:
        return {"status": "no_exact_match", "top": [r["id"] for r in results[:3]]}
    if any(q in failed for q in exact):
        return {"status": "error", "reason": "entities_failed", "exact": exact}
    if entity_type == "ORG" and any(q in unchecked for q in exact):
        return {"status": "error", "reason": "type_check_failed", "exact": exact}
    typed = [q for q in exact if q in entities and is_type(entities[q], entity_type, organisations)]
    if not typed:
        return {"status": "no_type_match", "exact": exact}
    ranked = sorted(typed, key=lambda q: (-entities[q]["sitelinks"], q))
    top = ranked[0]
    ambiguous = (len(ranked) > 1
                 and entities[ranked[1]]["sitelinks"] * AMBIGUITY_RATIO >= entities[top]["sitelinks"])
    return {"status": "ambiguous" if ambiguous else "linked", "qid": top,
            "candidates": [{"qid": q, "label": entities[q]["label"],
                            "description": entities[q]["description"],
                            "sitelinks": entities[q]["sitelinks"]} for q in ranked]}


def fetch_entities(net: Net, qids: Sequence[str], failed: set[str] | None = None
                   ) -> dict[str, dict[str, Any]]:
    """Item summaries; the ids of a batch that got no answer go to ``failed``,
    so that what the network lost is never read as what Wikidata said."""
    out: dict[str, dict[str, Any]] = {}

    def batch(ids: Sequence[str]):
        doc, _err = yield from api_json(net, WIKIDATA_API, {
            "action": "wbgetentities", "ids": "|".join(ids), "languages": "|".join(LANGUAGES),
            "format": "json", "props": "labels|aliases|descriptions|claims|sitelinks", "maxlag": MAXLAG},
            f"entities:{ids[0]}+{len(ids) - 1}", "entities")
        if doc is None and failed is not None:
            failed.update(ids)
        for qid, entity in ((doc or {}).get("entities") or {}).items():
            if "missing" not in entity:
                out[qid] = entity_summary(entity)

    net.run(batch(qids[i:i + BATCH]) for i in range(0, len(qids), BATCH))
    return out


def organisations(net: Net, entities: dict[str, dict[str, Any]], qids: Sequence[str],
                  parents: dict[str, list[str]] | None = None) -> tuple[set[str], set[str]]:
    """(items that are an organisation -- an instance of a class that is, by
    subclass of (P279) steps, Q43229 --, items whose check could not finish).

    The class hierarchy is walked through wbgetentities, on the host every
    other call uses and at its pace: the Query Service timed out on five of the
    sample's six batched subclass queries (2026-10-21), and a check that
    depends on it fails whenever it is loaded. ``parents`` (class -> its P279
    classes) may be shared between calls, so a class is fetched once a run."""
    parents = {} if parents is None else parents
    failed: set[str] = set()

    def batch(ids: Sequence[str]):
        doc, _err = yield from api_json(net, WIKIDATA_API, {
            "action": "wbgetentities", "ids": "|".join(ids), "props": "claims", "format": "json",
            "maxlag": MAXLAG}, f"classes:{ids[0]}+{len(ids) - 1}", "type_check")
        if doc is None:
            failed.update(ids)
            return
        for qid, entity in (doc.get("entities") or {}).items():
            parents[qid] = [] if "missing" in entity else claim_values(entity, "P279")

    frontier = sorted({c for q in qids for c in entities[q]["p31"]})
    for _depth in range(CLASS_DEPTH):
        todo = [c for c in frontier if c not in parents and c not in failed and c != Q_ORGANISATION]
        if not todo:
            break
        net.run(batch(todo[i:i + BATCH]) for i in range(0, len(todo), BATCH))
        frontier = sorted({p for c in todo for p in parents.get(c, [])})

    def reaches(cls: str, seen: set[str]) -> bool | None:
        """True if Q43229 is reached; None if an unknown class blocks the answer."""
        if cls == Q_ORGANISATION:
            return True
        if cls in seen:
            return False
        seen.add(cls)
        if cls not in parents:
            return None
        answers = [reaches(p, seen) for p in parents[cls]]
        return True if any(a is True for a in answers) else None if None in answers else False

    found, unchecked = set(), set()
    for q in qids:
        answers = [reaches(c, set()) for c in entities[q]["p31"]]
        if any(a is True for a in answers):
            found.add(q)
        elif None in answers:
            unchecked.add(q)
    return found, unchecked


def link_names(net: Net, todo: Sequence[dict[str, Any]], *,
               known: dict[str, dict[str, Any]] | None = None,
               classes: dict[str, list[str]] | None = None
               ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Link rows for ``todo`` ({type, key, query}), and the items they reach.
    Three passes, each through the one worker: search; the exact candidates'
    data (an item in ``known`` is not fetched again); for organisations, the
    type check (``classes`` carries the class hierarchy from call to call).

    A person's name with initials is searched twice, as written and with them
    dotted and spaced (dotted_initials), and the answers are pooled; exact
    still means same_name. A search that got no answer makes the name an
    error to retry, never a "no"."""
    known = known or {}
    answers: dict[tuple[str, str], list[tuple[list[dict[str, Any]] | None, str | None]]] = {}

    def search(row: dict[str, Any], query: str):
        doc, err = yield from api_json(net, WIKIDATA_API, {
            "action": "wbsearchentities", "search": query, "language": "en", "uselang": "en",
            "type": "item", "limit": SEARCH_LIMIT, "format": "json", "maxlag": MAXLAG},
            f"{row['type']}:{row['key']}", "search")
        answers.setdefault((row["type"], row["key"]), []).append(
            ((doc.get("search") or []) if doc is not None else None, err))

    queries = {(r["type"], r["key"]): [r["query"], *filter(None, [dotted_initials(r["query"], r["type"])])]
               for r in todo}
    net.run(search(r, q) for r in todo for q in queries[(r["type"], r["key"])])
    found: dict[tuple[str, str], tuple[list[dict[str, Any]] | None, str | None]] = {}
    for k, got in answers.items():
        failed_search = next(((None, err) for results, err in got if results is None), None)
        pooled = list({hit["id"]: hit for results, _err in got for hit in results or []}.values())
        found[k] = failed_search or (pooled, None)
    wanted: dict[str, set[str]] = {}
    for r in todo:
        for qid in exact_candidates(found[(r["type"], r["key"])][0] or [], r["key"]):
            wanted.setdefault(qid, set()).add(r["type"])
    entities = {q: known[q] for q in wanted if q in known}
    failed: set[str] = set()
    entities.update(fetch_entities(net, sorted(q for q in wanted if q not in known), failed))
    orgs, unchecked = organisations(net, entities, sorted(q for q, t in wanted.items()
                                                          if "ORG" in t and q in entities), classes)
    rows = []
    for r in todo:
        results, err = found[(r["type"], r["key"])]
        rows.append({"schema": "gallery_link/1", "pass": net.log.pass_id, "type": r["type"], "key": r["key"],
                     "query": r["query"], "queries": queries[(r["type"], r["key"])], "at": utcnow(),
                     **decide(r["type"], r["key"], results, err, entities, orgs, unchecked, failed)})
    return rows, entities


# --------------------------------------------------------------------------
# the sample: who is linked and fetched in this pass
# --------------------------------------------------------------------------

SAMPLE_RANDOM = 200
#: Linked random-sample entities whose references are fetched, per type: what
#: the reference yield (and so the full-build budget) is measured on.
SAMPLE_FETCH_RANDOM = 25
#: The full build's checkpoints: names linked, and entities fetched, between
#: two writes. A stop loses at most one chunk (a few minutes of requests).
LINK_CHUNK = 200
FETCH_CHUNK = 40


def frequency_band(mentions: int) -> str:
    return "1" if mentions <= 1 else "2-4" if mentions <= 4 else "5+"


def source_of(row: dict[str, Any]) -> str:
    v, c = row["mentions"]["vlm"], row["mentions"]["claims"]
    return "both" if v and c else "vlm" if v else "claims"


def stratum_of(row: dict[str, Any]) -> str:
    total = row["mentions"]["vlm"] + row["mentions"]["claims"]
    return f"{row['type']}|{source_of(row)}|{frequency_band(total)}"


def stratified_sample(rows: Sequence[dict[str, Any]], n: int, seed: int = SEED
                      ) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """A proportional stratified sample (type x source x frequency band), every
    stratum represented, each pick by a seeded hash priority. Returns the picks
    and each stratum's size and draw, so estimates can be weighted back."""
    strata: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        strata.setdefault(stratum_of(r), []).append(r)
    total = len(rows)
    quota = {s: max(1, round(n * len(v) / total)) for s, v in strata.items()} if total else {}
    picks: list[dict[str, Any]] = []
    for s, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda r: image_text._priority(seed, f"{r['type']}:{r['key']}"))
        quota[s] = min(quota[s], len(ordered))
        picks += [{**r, "stratum": s} for r in ordered[:quota[s]]]
    return picks, {s: {"size": len(strata[s]), "drawn": quota[s]} for s in sorted(strata)}


def glue_map(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Spaced names by (type, key without spaces): a hashtag ("rahulgandhi") is
    searched for under its spaced twin, when the table has one."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if " " in r["key"]:
            k = (r["type"], r["key"].replace(" ", ""))
            if k not in out or (r["mentions"]["vlm"] + r["mentions"]["claims"]
                                > out[k]["mentions"]["vlm"] + out[k]["mentions"]["claims"]):
                out[k] = r
    return out


def query_for(row: dict[str, Any], glued: dict[tuple[str, str], dict[str, Any]]) -> str:
    if " " not in row["key"]:
        twin = glued.get((row["type"], row["key"]))
        if twin is not None:
            return twin["display"]
    return row["display"]


_DOTTED = re.compile(r"(?:[a-z]\.)+[a-z]?", re.I)


def dotted_initials(query: str, entity_type: str = "PERSON") -> str | None:
    """A person's name with its initials dotted and spaced, the form Wikidata
    labels use and its prefix search needs ("P Chidambaram" is found only as
    "P. Chidambaram"), or None when there is none to dot or it already reads
    so. An initial is a lone letter, a run of two or three consonants
    ("SS Ahluwalia": "S. S. Ahluwalia"; _INITIALS), or letters already dotted
    ("K.S."), before the last word. The letters stay the letters: this is a
    form, never a spelling."""
    if entity_type != "PERSON":
        return None
    words = query.split()
    out = []
    for i, word in enumerate(words):
        letters = word.replace(".", "")
        initials = (i < len(words) - 1 and letters.isalpha()
                    and (len(letters) == 1 or _INITIALS.fullmatch(letters.lower()) or _DOTTED.fullmatch(word)))
        out.append(" ".join(f"{c.upper()}." for c in letters) if initials else word)
    dotted = " ".join(out)
    return dotted if dotted != " ".join(words) else None


#: The figures the recognition probe lists, by name key, with the name searched.
PROBE_FIGURES = {name_key(name, "PERSON"): name for name, _claim, _token in image_text.PUBLIC_FIGURES}


def claimed_figure(row: dict[str, Any], spans: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """(source, name key) of the one figure a claim names and says an image
    shows, or None. The probe's own list decides first, exactly as the probe
    selects; otherwise a claim whose NER finds exactly one person names that
    person. Two listed figures, or none and several NER names, name no one."""
    text = row.get("text") or ""
    if not image_text.PICTURED.search(text):
        return None
    hits = [name for name, claim, _token in image_text.PUBLIC_FIGURES if re.search(claim, text, re.I)]
    if len(hits) == 1:
        return ("probe_list", name_key(hits[0], "PERSON"))
    if hits:
        return None
    keys = {name_key(s["text"], "PERSON")
            for s in (spans.get(str(row["record_id"])) or {}).get("spans", [])
            if s["type"] == "PERSON"} - {None}
    return ("claim_ner", keys.pop()) if len(keys) == 1 else None


def claim_people(rows: Iterable[dict[str, Any]], spans: dict[str, dict[str, Any]]) -> list[str]:
    """Every person an image's claims name: the probe's listed figures their
    text matches, and claim NER's PERSON spans. A VLM name is right when it is
    any of them -- not only the one figure, which a claim may name beside the
    person pictured ("the minister's spokesman, pictured, said ...")."""
    names: set[str] = set()
    for row in rows:
        text = row.get("text") or ""
        names |= {name for name, claim, _token in image_text.PUBLIC_FIGURES if re.search(claim, text, re.I)}
        names |= {s["text"] for s in (spans.get(str(row["record_id"])) or {}).get("spans", [])
                  if s["type"] == "PERSON"}
    return sorted(names)


def assert_factify2_train(record_ids: Iterable[str], split: dict[str, str]) -> None:
    """Calibration reads Factify2 TRAIN and nothing else: never VERITE, never
    val or test. Raises on any record that is not a Factify2 train record."""
    bad = [r for r in record_ids if split.get(r) != "train"]
    if bad:
        raise GalleryError(f"calibration may read Factify2 train only; {len(bad)} record(s) are not "
                           f"(first: {bad[0]!r}, split {split.get(bad[0])!r})")


def calibration_population(read: LedgerRead, rows: Iterable[dict[str, Any]],
                           index: dict[str, dict[str, Any]], spans: dict[str, dict[str, Any]]
                           ) -> tuple[list[dict[str, Any]], Counter]:
    """Factify2 TRAIN claim images whose claims say an image shows someone and
    name one figure: exactly one of the probe's listed figures (other people
    may be named beside it), or else exactly one person by claim NER. The
    probe's population.

    An image any val or test record also uses is left out, so neither split is
    ever read. The figure is the image's stratum for per-figure reporting. What
    a VLM name is scored against is everyone the image's claims name
    (claim_people): scoring against the figure alone counted a correctly
    named bystander the claim also names as wrong."""
    if read.dataset != "factify2":
        raise GalleryError(f"calibration reads the Factify2 ledger, not {read.dataset!r}")
    split: dict[str, str] = {}
    claims_of: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split[str(row["record_id"])] = row["split"]
        if row.get("usable") and row.get("image_paths"):
            entry = index.get(row["image_paths"][0])  # the claim image
            if entry and entry.get("sha256"):
                claims_of.setdefault(entry["sha256"], []).append(row)
    out, tally = [], Counter()
    for sha, img in sorted(read.images.items()):
        if not all(split.get(r) == "train" for r in img["record_ids"]):
            tally["not_train_only"] += 1
            continue
        figures = {f for f in (claimed_figure(r, spans) for r in claims_of.get(sha, [])) if f}
        if not figures:
            tally["no_figure_named"] += 1
            continue
        if len({key for _s, key in figures}) > 1:
            tally["several_figures"] += 1
            continue
        # one figure; named by the probe's list on any record, that is its source
        source, key = sorted(figures, key=lambda f: f[0] != "probe_list")[0]
        assert_factify2_train(img["record_ids"], split)
        tally[f"images_{source}"] += 1
        claims = claims_of.get(sha, [])
        out.append({"sha256": sha, "record_ids": list(img["record_ids"]),
                    "figure_source": source, "figure_key": key, "names": img["names"],
                    "claim_people": claim_people(claims, spans),
                    "claim_text": "\n".join(r.get("text") or "" for r in claims)})
    return out, tally


def calibration_names(population: Sequence[dict[str, Any]]) -> set[str]:
    """PERSON keys of the names produced from appearance alone on the
    population's images: the VLM's own names, the ones calibration scores."""
    return {k for img in population for n in img["names"]
            if n["type"] == "PERSON" and n["evidence"] == "not_written"
            and (k := name_key(n["name"], "PERSON"))}


# --------------------------------------------------------------------------
# references: Commons files, through the three gates
# --------------------------------------------------------------------------


def _plain(text: str | None) -> str | None:
    """Commons's HTML attribution as plain text (the HTML is kept beside it)."""
    if text is None:
        return None
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def licence_of(meta: dict[str, Any]) -> dict[str, Any]:
    """The licence and attribution exactly as Commons states them."""
    value = {k: (meta.get(k) or {}).get("value") for k in EXTMETADATA}
    return {"licence": {"short_name": value["LicenseShortName"], "url": value["LicenseUrl"],
                        "id": value["License"], "usage_terms": value["UsageTerms"],
                        "attribution_required": value["AttributionRequired"],
                        "copyrighted": value["Copyrighted"], "restrictions": value["Restrictions"]},
            "attribution": {"artist": _plain(value["Artist"]), "credit": _plain(value["Credit"]),
                            "artist_html": value["Artist"], "credit_html": value["Credit"]},
            "date_original": _plain(value["DateTimeOriginal"])}


@lru_cache(maxsize=1)
def corpus_hashes() -> frozenset[str]:
    """sha256 of every image in every corpus index: a reference that IS a corpus
    image would verify a record with its own pixels."""
    out: set[str] = set()
    for ds in images.IMAGE_DATASETS:
        out.update(r["sha256"] for r in images.load_index(ds).values() if r.get("sha256"))
    return frozenset(out)


EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "TIFF": "tif", "BMP": "bmp"}


def rendition_problem(body: bytes, examined: dict[str, Any], info: dict[str, Any]) -> str | None:
    """None when the bytes are the file the API described, else why not.

    Wikimedia never scales a thumbnail UP: for an original no wider than
    THUMB_WIDTH it serves the original itself (``thumbnail_unscaled``), though
    imageinfo reports the size it would have scaled to. Those bytes must BE the
    file, by Commons's own SHA-1 -- the strongest check there is. A scaled
    thumbnail must have the requested width exactly, and the height the API
    gave within the thumbnailer's rounding (the sample saw 1 to 3 pixels)."""
    import hashlib

    original, rendition = info["original"], info["rendition"]
    if (original.get("width") and original["width"] <= THUMB_WIDTH
            and info.get("mime") != "image/svg+xml"):
        return None if hashlib.sha1(body).hexdigest() == original.get("sha1") else "not_the_original_file"
    if examined["width"] != rendition["width"]:
        return "rendition_mismatch"
    tolerance = max(2, math.ceil(0.01 * rendition["height"]))
    return None if abs(examined["height"] - rendition["height"]) <= tolerance else "rendition_mismatch"


def gate_reference(resp, info: dict[str, Any], corpus: frozenset[str]) -> dict[str, Any]:
    """The three gates, and two of the gallery's own.

    status 200, the bytes decode in full, they are not in the placeholder
    registry; then they are the file the API described (rendition_problem),
    and they are not an image of any of our corpora."""
    if resp.error or resp.status != 200:
        return {"outcome": resp.error or f"http_{resp.status}"}
    examined = images.examine_bytes(resp.body)
    if not examined["loads"]:
        return {"outcome": "undecodable", "detail": examined["error"]}
    status = images.status({**examined, "path": None})
    common = {"examined": examined, "quarantine": True}
    if status != images.USABLE:
        return {"outcome": f"registry:{status}", **common}
    problem = rendition_problem(resp.body, examined, info)
    if problem:
        return {"outcome": problem, **common}
    if examined["sha256"] in corpus:
        return {"outcome": "corpus_duplicate", **common}
    return {"outcome": "usable", "examined": examined}


def store(body: bytes, examined: dict[str, Any], directory: Path) -> Path:
    """Content-addressed and written once: ``<dir>/<sha[:2]>/<sha>.<ext>``."""
    sha = examined["sha256"]
    path = directory / sha[:2] / f"{sha}.{EXT.get(examined['format'], 'img')}"
    if not path.is_file():
        image_text.assert_not_raw(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(body)
        os.replace(tmp, path)
    return path


def commons_candidates(net: Net, entities: Sequence[dict[str, Any]], *, limit: int = COMMONS_SEARCH_LIMIT,
                       cap: int | None = None) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """(qid -> [(File: title, source)], qids whose search got no answer). In
    order: the item's own image (P18; for an organisation its logo, P154,
    first), files whose structured data says they depict the item, then files
    in its Commons category. ``limit`` files per listing; at most ``cap`` per
    item (CANDIDATES, the references a build takes, by default), and the
    category is listed only while fewer are found."""
    out: dict[str, list[tuple[str, str]]] = {}
    failed: set[str] = set()
    for e in entities:
        own = [("P154", f) for f in e["logos"]] if e["type"] == "ORG" else []
        own += [("P18", f) for f in e["images"]]
        out[e["qid"]] = [(f"File:{name}", source) for source, name in own]

    def depicts(e: dict[str, Any]):
        doc, _err = yield from api_json(net, COMMONS_API, {
            "action": "query", "list": "search", "srnamespace": 6, "srlimit": limit,
            "srsearch": f"haswbstatement:P180={e['qid']} filetype:bitmap", "format": "json",
            "maxlag": MAXLAG}, f"depicts:{e['qid']}", "depicts")
        if doc is None:
            failed.add(e["qid"])
        for hit in ((doc or {}).get("query") or {}).get("search") or []:
            out[e["qid"]].append((hit["title"], "depicts"))

    def category(e: dict[str, Any]):
        doc, _err = yield from api_json(net, COMMONS_API, {
            "action": "query", "list": "categorymembers", "cmtype": "file",
            "cmtitle": f"Category:{e['commons_category']}", "cmlimit": limit,
            "format": "json", "maxlag": MAXLAG}, f"category:{e['qid']}", "category")
        if doc is None:
            failed.add(e["qid"])
        for hit in ((doc or {}).get("query") or {}).get("categorymembers") or []:
            out[e["qid"]].append((hit["title"], "category"))

    people = [e for e in entities if e["type"] == "PERSON"]
    net.run(depicts(e) for e in people)
    net.run(category(e) for e in people
            if e["commons_category"] and len({t for t, _s in out[e["qid"]]}) < (cap or CANDIDATES["PERSON"]))
    kinds = {e["qid"]: e["type"] for e in entities}
    for qid, found in out.items():
        seen: set[str] = set()
        unique = []
        for title, source in found:
            if title.replace("_", " ") not in seen:
                seen.add(title.replace("_", " "))
                unique.append((title, source))
        out[qid] = unique[:cap or CANDIDATES[kinds[qid]]]
    return out, failed


def image_info(net: Net, titles: Sequence[str]) -> dict[str, dict[str, Any]]:
    """title -> the file's facts, its licence and attribution, and its
    THUMB_WIDTH rendition (what is downloaded)."""
    out: dict[str, dict[str, Any]] = {}

    def batch(part: Sequence[str]):
        doc, _err = yield from api_json(net, COMMONS_API, {
            "action": "query", "prop": "imageinfo", "titles": "|".join(part),
            "iiprop": "url|size|sha1|mime|mediatype|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": "|".join(EXTMETADATA), "iiextmetadatalanguage": "en",
            "format": "json", "maxlag": MAXLAG}, f"imageinfo:{len(part)}", "imageinfo")
        if doc is None:  # no answer is not "no such file"
            out.update({t: {"lookup_failed": _err} for t in part})
            return
        query = doc.get("query") or {}
        renamed = {n["to"]: n["from"] for n in query.get("normalized") or []}
        for page in (query.get("pages") or {}).values():
            info = (page.get("imageinfo") or [None])[0]
            asked = renamed.get(page.get("title"), page.get("title"))
            if info is None:
                out[asked] = {"missing": True}
                continue
            out[asked] = {"page_url": info.get("descriptionurl"), "mime": info.get("mime"),
                          "mediatype": info.get("mediatype"),
                          "original": {"url": info.get("url"), "width": info.get("width"),
                                       "height": info.get("height"), "bytes": info.get("size"),
                                       "sha1": info.get("sha1")},
                          "rendition": {"url": info.get("thumburl") or info.get("url"),
                                        "width": info.get("thumbwidth") or info.get("width"),
                                        "height": info.get("thumbheight") or info.get("height")},
                          **licence_of(info.get("extmetadata") or {})}

    unique = list(dict.fromkeys(titles))
    net.run(batch(unique[i:i + BATCH]) for i in range(0, len(unique), BATCH))
    return out


def eligible(info: dict[str, Any], entity_type: str) -> str | None:
    """Why a file is not tried, or None. A person's reference is a photograph:
    a drawing (SVG) is not. Every file must carry its licence."""
    if info.get("missing"):
        return "missing_file"
    kinds = ("BITMAP",) if entity_type == "PERSON" else ("BITMAP", "DRAWING")
    if info.get("mediatype") not in kinds:
        return f"mediatype:{info.get('mediatype')}"
    if not info["licence"]["short_name"]:
        return "no_licence_metadata"
    return None


#: Download outcomes that mean "not now", not "not this file" (recover.get has
#: already retried them twice): an entity holding one is not written, and the
#: next run fetches it again. A 500 or 404 on a file is that file's answer.
TRANSIENT_OUTCOMES = frozenset({"timeout", "connection_error", "dns_failure", "ssl_error",
                                "http_429", "http_502", "http_503", "http_504"})


def fetch_references(net: Net, entities: Sequence[dict[str, Any]], *,
                     corpus: frozenset[str] | None = None, media: Path = MEDIA_DIR,
                     rejected: Path = REJECTED_DIR,
                     candidates: dict[str, list[tuple[str, str]]] | None = None) -> list[dict[str, Any]]:
    """Entity rows for ``entities`` (Wikidata summaries with a ``type``): each
    candidate file tried once, gated, stored or quarantined, with its licence.
    ``candidates`` (qid -> [(File: title, source)]) replaces the Commons
    search: the side views a census found."""
    from scripts import recover

    corpus = corpus_hashes() if corpus is None else corpus
    if candidates is None:
        candidates, lookup_failed = commons_candidates(net, entities)
    else:
        lookup_failed = set()
    infos = image_info(net, [t for found in candidates.values() for t, _s in found])
    results: dict[str, dict[str, Any]] = {}

    def download(url: str, info: dict[str, Any]):
        resp, chain = yield from recover.get(url, {"User-Agent": user_agent()}, net.log,
                                             f"file:{url.rsplit('/', 1)[-1][:80]}", "download",
                                             net.clock)
        gated = gate_reference(resp, info, corpus)
        examined = gated.get("examined")
        if gated["outcome"] == "usable":
            path = store(resp.body, examined, media)
        elif gated.get("quarantine"):
            path = store(resp.body, examined, rejected)
        else:
            path = None
        net.log.attempt(f"file:{url.rsplit('/', 1)[-1][:80]}", "download", resp,
                        "ok" if gated["outcome"] == "usable" else "rejected",
                        None if gated["outcome"] == "usable" else gated["outcome"],
                        sha256=(examined or {}).get("sha256"))
        results[url] = {"outcome": gated["outcome"], "redirects": chain,
                        "path": None if path is None else path.relative_to(image_text.PROJECT_ROOT).as_posix()
                        if path.is_relative_to(image_text.PROJECT_ROOT) else str(path),
                        **({k: examined[k] for k in ("sha256", "bytes", "format", "width", "height")}
                           if examined else {})}

    # an entity whose lookups got no answer is left whole for the next run
    unanswered = set(lookup_failed) | {e["qid"] for e in entities for t, _s in candidates[e["qid"]]
                                       if (infos.get(t) or {}).get("lookup_failed")}
    wanted: dict[str, dict[str, Any]] = {}
    for e in entities:
        if e["qid"] in unanswered:
            continue
        for title, _source in candidates[e["qid"]]:
            info = infos.get(title) or {"missing": True}
            if eligible(info, e["type"]) is None:
                wanted.setdefault(info["rendition"]["url"], info)
    try:
        net.run(download(url, info) for url, info in wanted.items())
    except recover.BudgetExceeded as exc:
        net.stopped = f"budget: {exc}"
        log(f"  stopped: the transfer budget is spent ({exc}); finished entities are kept")
    rows = []
    for e in entities:
        if e["qid"] in unanswered:
            continue
        refs, complete = [], True
        for title, source in candidates[e["qid"]]:
            info = infos.get(title) or {"missing": True}
            why = eligible(info, e["type"])
            ref = {"title": title, "source": source,
                   **{k: info.get(k) for k in ("page_url", "mime", "mediatype", "original",
                                                "rendition", "licence", "attribution",
                                                "date_original")}}
            if why is not None:
                refs.append({**ref, "outcome": why})
                continue
            got = results.get(info["rendition"]["url"])
            if got is None or got["outcome"] in TRANSIENT_OUTCOMES:
                complete = False  # not downloaded, or "not now": the next run tries again
                break
            refs.append({**ref, **got})
        if complete:
            rows.append({"schema": "gallery_entity/1", "pass": net.log.pass_id, "fetched_at": utcnow(),
                         **{k: e[k] for k in ("qid", "type", "label", "description", "aliases", "p31",
                                              "sitelinks", "commons_category")},
                         "names": sorted(e.get("names") or []), "refs": refs})
    return rows


# --------------------------------------------------------------------------
# faces: YuNet finds them, SFace (or, for the bake-off, AuraFace) embeds them
# --------------------------------------------------------------------------

FACE_BACKENDS = ("sface", "auraface")
#: Fixed before any data was seen; never tuned.
DET_SCORE = 0.8
DET_NMS = 0.3
#: YuNet finds faces of about 10 to 300 pixels (its model card), so an image is
#: searched at a long side of 640 and at its own size up to 1920, and the two
#: lists merged.
DET_LONG_SIDES = (640, 1920)
MERGE_IOU = 0.4
#: Smallest face side (pixels, in the original image) that is embedded.
MIN_FACE = {"query": 32, "reference": 48}
#: A reference with several faces is one person's only if its largest face is
#: this many times the area of the next.
DOMINANCE = 2.0
#: References of one entity must agree: one further from the entity's medoid
#: than this quantile of similarities between DIFFERENT entities' medoids is
#: someone else. Measured on the gallery's own Commons references, never on
#: corpus images.
CONSISTENCY_QUANTILE = 0.999
CONSISTENCY_MAX_PAIRS = 2_000_000
#: Aligned face crops this close (dHash bits) are the same photograph, not
#: another photograph of the same person.
SAME_PHOTO_BITS = 10
#: The five-point template both recognisers are aligned to (ArcFace's; OpenCV's
#: FaceRecognizerSF.alignCrop uses the same points, in YuNet's landmark order).
ARCFACE_TEMPLATE = ((38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
                    (41.5493, 92.3655), (70.7299, 92.2041))


def load_bgr(body: bytes):
    """Pixels as the recogniser sees them: EXIF-rotated, first frame, BGR."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(body)) as img:
        img.seek(0)
        rgb = ImageOps.exif_transpose(img).convert("RGB")
    return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def similarity_transform(src, dst):
    """Least-squares similarity (Umeyama) mapping five landmarks to the template."""
    import numpy as np

    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rot = u @ sign @ vt
    scale = (sig * np.diag(sign)).sum() / s.var(0).sum()
    return np.hstack([scale * rot, (md - scale * rot @ ms)[:, None]])


def dhash(aligned) -> str:
    import cv2

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceBackend:
    """YuNet finds faces and five landmarks, at two scales; a recogniser embeds
    each aligned face. Subclasses name the recogniser."""

    name = ""

    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(face_model_path("yunet")), "", (320, 320),
                                                  DET_SCORE, DET_NMS, 5000)

    def recogniser_config(self) -> dict[str, Any]:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        spec = FACE_MODELS["yunet"]
        return {"schema": "gallery_faces/1", "backend": self.name, "opencv": self.cv2.__version__,
                "detector": {"model": f"{spec['repo']}/{spec['file']}@{spec['revision']}",
                             "sha256": spec["sha256"], "licence": spec["licence"],
                             "score": DET_SCORE, "nms": DET_NMS, "long_sides": list(DET_LONG_SIDES),
                             "merge_iou": MERGE_IOU},
                "recogniser": self.recogniser_config(), "min_face": dict(MIN_FACE),
                "dhash": "9x8 grey, on the aligned 112x112 crop",
                "preprocessing": "EXIF transpose, first frame, RGB -> BGR"}

    def detect(self, bgr) -> list[Any]:
        """YuNet rows (box, five landmarks, score) in the image's own pixels."""
        h, w = bgr.shape[:2]
        scales = sorted({min(1.0, side / max(h, w)) for side in DET_LONG_SIDES})
        found: list[Any] = []
        for s in scales:
            img = bgr if s == 1.0 else self.cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                                                         interpolation=self.cv2.INTER_AREA)
            self.detector.setInputSize((img.shape[1], img.shape[0]))
            _ok, rows = self.detector.detect(img)
            for row in ([] if rows is None else rows):
                row = row.copy()
                row[:14] /= s
                same = [i for i, f in enumerate(found) if _iou(f, row) >= MERGE_IOU]
                if not same:
                    found.append(row)
                elif row[14] > found[same[0]][14]:
                    found[same[0]] = row
        return sorted(found, key=lambda r: -float(r[2] * r[3]))

    def align(self, bgr, row):
        raise NotImplementedError

    def embed(self, aligned):
        raise NotImplementedError

    def faces(self, bgr, min_side: int) -> list[dict[str, Any]]:
        import base64

        import numpy as np

        out = []
        for row in self.detect(bgr):
            if min(row[2], row[3]) < min_side:
                continue
            aligned = self.align(bgr, row)
            vec = np.asarray(self.embed(aligned), dtype=np.float32).flatten()
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            out.append({"box": [round(float(v), 1) for v in row[:4]],
                        "landmarks": [[round(float(row[4 + 2 * i]), 1), round(float(row[5 + 2 * i]), 1)]
                                      for i in range(5)],
                        "score": round(float(row[14]), 4), "dhash": dhash(aligned),
                        "embedding": base64.b64encode(vec.tobytes()).decode("ascii")})
        return out


class SFaceBackend(FaceBackend):
    name = "sface"

    def __init__(self) -> None:
        super().__init__()
        self.recogniser = self.cv2.FaceRecognizerSF.create(str(face_model_path("sface")), "")

    def recogniser_config(self) -> dict[str, Any]:
        spec = FACE_MODELS["sface"]
        return {"model": f"{spec['repo']}/{spec['file']}@{spec['revision']}", "sha256": spec["sha256"],
                "licence": spec["licence"], "alignment": "FaceRecognizerSF.alignCrop (five points)",
                "dimensions": 128}

    def align(self, bgr, row):
        return self.recogniser.alignCrop(bgr, row)

    def embed(self, aligned):
        return self.recogniser.feature(aligned)


class AuraFaceBackend(FaceBackend):
    """The bake-off contrast: ONLY its recogniser (glintr100.onnx), behind
    YuNet. Its graph opens with a convolution (a PyTorch export), so input
    normalisation is outside it: (x - 127.5) / 127.5 on RGB, InsightFace's rule
    for such graphs."""

    name = "auraface"

    def __init__(self) -> None:
        super().__init__()
        self.net = self.cv2.dnn.readNetFromONNX(str(face_model_path("auraface")))
        first = [n for n in self.net.getLayerNames() if not n.startswith("trlayout")][:1]
        if not first or not first[0].endswith("Conv"):
            raise image_text.ModelMissing(f"AuraFace's graph opens with {first}, not a "
                                          "convolution; its input normalisation is unknown")

    def recogniser_config(self) -> dict[str, Any]:
        spec = FACE_MODELS["auraface"]
        return {"model": f"{spec['repo']}/{spec['file']}@{spec['revision']}", "sha256": spec["sha256"],
                "licence": spec["licence"], "alignment": "similarity transform to the ArcFace template",
                "normalisation": "(x - 127.5) / 127.5, RGB", "dimensions": 512,
                "caveat": "training data described only as 'a commercial dataset'"}

    def align(self, bgr, row):
        landmarks = [(row[4 + 2 * i], row[5 + 2 * i]) for i in range(5)]
        matrix = similarity_transform(landmarks, ARCFACE_TEMPLATE)
        return self.cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0)

    def embed(self, aligned):
        blob = self.cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5),
                                          swapRB=True)
        self.net.setInput(blob)
        return self.net.forward()


def make_backend(name: str) -> FaceBackend:
    return {"sface": SFaceBackend, "auraface": AuraFaceBackend}[name]()


def face_config_id(backend: FaceBackend) -> str:
    return config_id_of(backend.config())


def decode_embedding(text: str):
    """An embedding as stored (float32), in float64. A product of two float32
    values is exact in float64, so a similarity, a medoid and a threshold
    comparison come out the same on any machine; in float32 a near-tie at
    1e-6 could break one way here and the other way there (decided
    2026-10-23)."""
    import base64

    import numpy as np

    return np.frombuffer(base64.b64decode(text), dtype=np.float32).astype(np.float64)


def embed_images(backend: FaceBackend, items: Sequence[tuple[str, Callable[[], bytes]]],
                 path: Path, min_side: int) -> dict[str, dict[str, Any]]:
    """sha256 -> its faces, one row per image in ``path``; an image already
    there is not read again."""
    ledger = JsonlLedger(path)
    done = latest(ledger.open(), lambda r: r["sha256"])
    try:
        todo = [(sha, read) for sha, read in items if sha not in done]
        batch = []
        for i, (sha, read) in enumerate(todo, 1):
            try:
                bgr = load_bgr(read())
                row = {"sha256": sha, "width": int(bgr.shape[1]), "height": int(bgr.shape[0]),
                       "faces": backend.faces(bgr, min_side), "error": None}
            except Exception as exc:  # noqa: BLE001 -- an unreadable image is a row
                row = {"sha256": sha, "width": None, "height": None, "faces": [],
                       "error": f"{type(exc).__name__}"}
            batch.append(row)
            done[sha] = row
            if len(batch) >= 64 or i == len(todo):
                ledger.append_many(batch)
                batch = []
            if i % 500 == 0:
                log(f"  [{path.stem}] faces {i:,}/{len(todo):,}")
    finally:
        ledger.close()
    return done


# --------------------------------------------------------------------------
# the gallery: references that agree with each other
# --------------------------------------------------------------------------


def _area(face: dict[str, Any]) -> float:
    return face["box"][2] * face["box"][3]


def choose_reference_face(faces: Sequence[dict[str, Any]]) -> tuple[int | None, str]:
    """(face index, decision) for one reference image of a person."""
    if not faces:
        return None, "no_face"
    if len(faces) == 1:
        return 0, "single_face"
    order = sorted(range(len(faces)), key=lambda i: -_area(faces[i]))
    if _area(faces[order[0]]) >= DOMINANCE * _area(faces[order[1]]):
        return order[0], "dominant_face"
    return None, "ambiguous_faces"


def medoid_of(vectors: dict[int, Any]) -> int | None:
    """The key of the vector most similar, on average, to the others."""
    import numpy as np

    if not vectors:
        return None
    keys = list(vectors)
    m = np.stack([vectors[k] for k in keys])
    return keys[int((m @ m.T).mean(1).argmax())]


def consistency_threshold(medoids: dict[str, Any], seed: int = SEED) -> float | None:
    """The CONSISTENCY_QUANTILE of similarities between the medoid faces of
    DIFFERENT entities, from the gallery's own references.

    Medoids, not every face: a reference filed under the wrong entity is not a
    medoid, so it cannot raise the very threshold that is meant to catch it."""
    import numpy as np

    qids = sorted(medoids)
    n = len(qids)
    if n < 2:
        return None
    m = np.stack([medoids[q] for q in qids])
    if n * (n - 1) // 2 <= CONSISTENCY_MAX_PAIRS:
        values = (m @ m.T)[np.triu_indices(n, 1)]
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, CONSISTENCY_MAX_PAIRS)
        j = rng.integers(0, n, CONSISTENCY_MAX_PAIRS)
        keep = i != j
        values = np.einsum("ij,ij->i", m[i[keep]], m[j[keep]])
    return float(np.quantile(values, CONSISTENCY_QUANTILE))


#: A reference fetched because it shows the person side-on (fetch-side-views).
SIDE_VIEW = "side_view"
#: What the consistency check answers when it refuses a reference's face.
REFUSED_BY_CHECK = ("inconsistent_reference", "unresolved_pair", "side_view_unchecked")


def reference_id(qid: str, ref: dict[str, Any]) -> str:
    """A reference's key in the reference decisions file: person and file."""
    return f"{qid}:{(ref.get('sha256') or '')[:12]}"


def reference_picks(entity: dict[str, Any], ref_faces: dict[str, dict[str, Any]]
                    ) -> tuple[list[tuple[int, int | None, str]], dict[int, Any], dict[int, Any]]:
    """A person's references: (index, chosen face, decision) for each, and the
    embeddings of the faces chosen, frontal and side views (SIDE_VIEW) apart."""
    rows: list[tuple[int, int | None, str]] = []
    frontal: dict[int, Any] = {}
    side: dict[int, Any] = {}
    for i, ref in enumerate(entity.get("refs") or []):
        if ref.get("outcome") != "usable":
            rows.append((i, None, ref.get("outcome") or "not_fetched"))
            continue
        faces = (ref_faces.get(ref["sha256"]) or {}).get("faces") or []
        idx, decision = choose_reference_face(faces)
        rows.append((i, idx, decision))
        if idx is not None:
            (side if ref.get("source") == SIDE_VIEW else frontal)[i] = decode_embedding(faces[idx]["embedding"])
    return rows, frontal, side


def person_medoid(refs: Sequence[dict[str, Any]], vectors: dict[int, Any]) -> int | None:
    """The medoid of a person's single-face frontal references (reference
    index -> embedding). Of two it is Wikidata's own image (P18), else the
    first: their medoid is a tie, which float rounding breaks, and another
    machine's rounding can break it the other way (decided 2026-10-23: a
    gallery that differs by machine is not reproducible)."""
    if len(vectors) == 2:
        p18 = [i for i in vectors if refs[i]["source"] == "P18"]
        return p18[0] if len(p18) == 1 else next(iter(vectors))
    return medoid_of(vectors)


def consistency_check(refs: Sequence[dict[str, Any]], vectors: dict[int, Any], threshold: float | None
                      ) -> tuple[dict[int, tuple[str | None, float | None]], int | None]:
    """The consistency check over a person's single-face frontal references
    (reference index -> embedding): for each, its refusal (None when kept) and
    its similarity, and the reference the person's other faces are checked
    against -- group photos' and side views'. That is their medoid
    (person_medoid) when the check keeps it, and never a reference the check
    refused.

    Two references are checked against each other. When they disagree,
    Wikidata's own image (P18) is kept and the other refused; with no P18 both
    are refused (unresolved_pair), and there is none. Three or more are
    checked against their medoid, which is never refused. One is its own
    medoid."""
    checked: dict[int, tuple[str | None, float | None]] = {i: (None, None) for i in vectors}
    medoid = person_medoid(refs, vectors)
    if medoid is None or threshold is None:
        return checked, None
    if len(vectors) == 2:
        a, b = list(vectors)
        sim = float(vectors[a] @ vectors[b])
        if sim < threshold:
            anchors = [i for i in (a, b) if refs[i]["source"] == "P18"]
            for i in (a, b):
                if anchors and i not in anchors:
                    checked[i] = ("inconsistent_reference", round(sim, 4))
                elif not anchors:
                    checked[i] = ("unresolved_pair", round(sim, 4))
    elif len(vectors) > 2:
        for i, v in vectors.items():
            sim = float(v @ vectors[medoid])
            checked[i] = ("inconsistent_reference" if i != medoid and sim < threshold else None, round(sim, 4))
    return checked, None if checked[medoid][0] else medoid


def assemble(entities: Sequence[dict[str, Any]], ref_faces: dict[str, dict[str, Any]],
             pinned: float | None = None, decisions: dict[str, str | None] | None = None,
             flags: dict[str, list[str]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per entity: every reference with what the gallery decided about
    it. A person's usable references are faces that agree with each other.

    ``pinned`` replaces the derived consistency threshold: an entity's
    decisions then depend on its own references alone, so entities added later
    (another corpus, the rest of a run) leave every existing row unchanged.

    A group photo's face and a side view (source SIDE_VIEW) are checked
    against the medoid the consistency check keeps (consistency_check), never
    one it refused, and a side view never enters it: added side views leave
    every other reference's decision, the medoid and the threshold as they
    were. ``decisions`` (reference_id -> a person's decision) admit a
    reference the check refused when a person says it is the same person,
    and refuse one it kept when a person says it is someone else
    (person_refused): a single-face reference then takes no part in the
    check, which runs again without it, so it is never the medoid. The
    threshold itself is never moved. ``flags`` (reference_id -> a person's
    flags, REFERENCE_FLAG_VALUES) ride on the reference's row."""
    import numpy as np

    picks: dict[str, list[tuple[int, int | None, str]]] = {}
    vectors: dict[str, dict[int, Any]] = {}
    side: dict[str, dict[int, Any]] = {}
    medoids: dict[str, Any] = {}
    for e in entities:
        if e["type"] != "PERSON":
            continue
        picks[e["qid"]], vec, side[e["qid"]] = reference_picks(e, ref_faces)
        if vec:
            vectors[e["qid"]] = vec
            medoids[e["qid"]] = vec[person_medoid(e["refs"], vec)]
    derived = consistency_threshold(medoids)
    threshold = derived if pinned is None else pinned
    out = []
    for e in entities:
        refs = []
        if e["type"] != "PERSON":
            for ref in e["refs"]:
                refs.append({"sha256": ref.get("sha256"), "title": ref["title"], "source": ref["source"],
                             "decision": "org_reference" if ref.get("outcome") == "usable"
                             else ref.get("outcome"), "face": None})
            out.append({"qid": e["qid"], "type": e["type"], "refs": refs, "usable_refs": 0})
            continue
        rows = picks[e["qid"]]
        vec = vectors.get(e["qid"], {})
        verdicts = {i: (decisions or {}).get(reference_id(e["qid"], ref)) for i, ref in enumerate(e["refs"])}
        decided: dict[int, tuple[int | None, str, float | None]] = {i: (idx, d, None) for i, idx, d in rows}
        # a person's "someone_else" refuses a reference the check keeps, and the check runs again without
        # it: no face is checked against a reference a person refused. One the check refuses stays as it was
        overruled: dict[int, float | None] = {}
        while True:
            checked, medoid = consistency_check(e["refs"], {i: v for i, v in vec.items() if i not in overruled},
                                                threshold)
            more = {i: sim for i, (refusal, sim) in checked.items()
                    if not refusal and verdicts.get(i) == "someone_else"}
            if not more:
                break
            overruled.update(more)
        for i, (refusal, sim) in checked.items():
            decided[i] = (None, refusal, sim) if refusal else (decided[i][0], decided[i][1], sim)
        for i, sim in overruled.items():
            decided[i] = (None, "person_refused", sim)
        # group photos: the face that agrees with the medoid the check kept -- of two references or more
        if medoid is not None and len(vec) >= 2:
            for i, idx, decision in rows:
                if idx is None and decision == "ambiguous_faces":
                    faces = ref_faces[e["refs"][i]["sha256"]]["faces"]
                    sims = [float(decode_embedding(f["embedding"]) @ vec[medoid]) for f in faces]
                    best = int(np.argmax(sims))
                    if sims[best] >= threshold:
                        decided[i] = (best, "resolved_by_medoid", round(sims[best], 4))
        # side views: against that medoid (their one reference, if one), never in it
        for i, v in side[e["qid"]].items():
            if medoid is None:
                decided[i] = (None, "side_view_unchecked", None)
                continue
            sim = round(float(v @ vec[medoid]), 4)
            agrees = sim >= threshold
            decided[i] = (decided[i][0], decided[i][1], sim) if agrees else (None, "inconsistent_reference", sim)
        # a person's decision admits a reference the check refused, or refuses a group photo's face or a
        # side view it kept; it never relaxes the threshold
        chosen = {i: idx for i, idx, _d in rows}
        for i in range(len(e["refs"])):
            idx, decision, sim = decided.get(i, (None, None, None))
            if (idx is None and chosen.get(i) is not None and decision in REFUSED_BY_CHECK
                    and verdicts.get(i) == "same_person"):
                decided[i] = (chosen[i], "person_confirmed", sim)
            elif idx is not None and verdicts.get(i) == "someone_else":
                decided[i] = (None, "person_refused", sim)
        usable = 0
        for i, ref in enumerate(e["refs"]):
            idx, decision, sim = decided.get(i, (None, ref.get("outcome"), None))
            ok = idx is not None
            usable += ok
            flagged = (flags or {}).get(reference_id(e["qid"], ref))
            refs.append({"sha256": ref.get("sha256"), "title": ref["title"], "source": ref["source"],
                         "decision": "usable" if ok else decision, "selected_by": decision if ok else None,
                         "face": idx if ok else None, "similarity_to_medoid": sim,
                         **({"flags": flagged} if flagged else {})})
        out.append({"qid": e["qid"], "type": e["type"], "refs": refs, "usable_refs": usable,
                    "single_reference": usable == 1})
    people = [r for r in out if r["type"] == "PERSON"]
    summary = {"consistency_threshold": None if threshold is None else round(threshold, 4),
               "pinned": pinned is not None,
               "derived_now": None if derived is None else round(derived, 4),
               "people": len(people),
               "people_with_usable_faces": sum(1 for r in people if r["usable_refs"]),
               "consistency_quantile": CONSISTENCY_QUANTILE, "entities_with_faces": len(medoids),
               "threshold_from": ("similarities between the medoid faces of different entities; "
                                  "gallery references only, never corpus images")}
    return out, summary


# --------------------------------------------------------------------------
# 1:1 verification: a claimed name, an image, and the reference it matched
# --------------------------------------------------------------------------


#: A face is side-on when its nose sits this many eye spans or more from the
#: eyes' midpoint, along the eye line: turned so far that the nose has passed
#: the far eye.
SIDE_ON = 0.5
#: The smallest face verify scores, eyes to mouth (face_geometry): the smallest
#: face the gallery recognised among the names a person confirmed on the
#: sample (2026-10-22). Below it a score is noise presented as a measurement,
#: and verify answers no_usable_face.
MIN_EYE_MOUTH = 20.0


def face_geometry(face: dict[str, Any]) -> dict[str, float | None]:
    """A face's size and pose, from its box and five landmarks (eyes, nose,
    mouth corners):

    - face_side, the box's short side, and eye_span, how far apart the eyes are;
    - eye_mouth, from the eyes' midpoint to the mouth's: a face turned left or
      right keeps it, so it measures size apart from pose;
    - nose_offset, the nose from the eyes' midpoint along the eye line, in eye
      spans: about 0 face-on, SIDE_ON or more side-on. None when the eyes
      coincide."""
    (rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy) = face["landmarks"][:5]
    ex, ey = lx - rx, ly - ry
    span = math.hypot(ex, ey)
    mx, my = (rx + lx) / 2, (ry + ly) / 2
    return {"face_side": round(min(face["box"][2:]), 1), "eye_span": round(span, 1),
            "eye_mouth": round(math.hypot((rmx + lmx) / 2 - mx, (rmy + lmy) / 2 - my), 1),
            "nose_offset": round(((nx - mx) * ex + (ny - my) * ey) / span ** 2, 2) if span else None}


def side_on(geometry: dict[str, Any]) -> bool:
    return geometry.get("eye_span") == 0 or abs(geometry.get("nose_offset") or 0.0) >= SIDE_ON


def too_small(face: dict[str, Any]) -> bool:
    """Under MIN_EYE_MOUTH: a face verify never scores."""
    return face_geometry(face)["eye_mouth"] < MIN_EYE_MOUTH


#: How a name resolves (Gallery.resolve_key), beyond an exact key.
RESOLVE_RULES = ("spacing", "one_word")


class Gallery:
    """The assembled gallery for one face configuration, with the name index.

    Verification is 1:1 by design: it answers whether an image shows the
    entity a CLAIMED name resolves to. It does not search the gallery for an
    unnamed face."""

    def __init__(self, cid: str, *, faces_dir: Path = FACES_DIR, entities_path: Path = ENTITIES_PATH,
                 links_path: Path = LINKS_PATH, wikidata_path: Path = WIKIDATA_PATH) -> None:
        self.cid = cid
        self.rows = latest(read_jsonl(faces_dir / cid / "gallery.jsonl"), lambda r: r["qid"])
        self.entities = latest(read_jsonl(entities_path), lambda r: r["qid"])
        ref_faces = latest(read_jsonl(faces_dir / cid / "refs.jsonl"), lambda r: r["sha256"])
        self.refs: dict[str, list[tuple[dict[str, Any], dict[str, Any], Any]]] = {}
        for qid, row in self.rows.items():
            entity = self.entities.get(qid) or {"refs": []}
            by_sha = {r.get("sha256"): r for r in entity["refs"] if r.get("sha256")}
            for ref in row["refs"]:
                if ref["decision"] == "usable":
                    face = ref_faces[ref["sha256"]]["faces"][ref["face"]]
                    stored = by_sha.get(ref["sha256"], {})
                    self.refs.setdefault(qid, []).append(
                        ({**stored, "flags": ref["flags"]} if ref.get("flags") else stored, face,
                         decode_embedding(face["embedding"])))
        self.index: dict[tuple[str, str], set[str]] = {}
        held: dict[tuple[str, str], str] = {}
        for link in latest(read_jsonl(links_path), lambda r: (r["type"], r["key"])).values():
            if link.get("qid") and link["status"] in ("linked", "ambiguous"):
                self.index.setdefault((link["type"], link["key"]), set()).add(link["qid"])
            elif link["status"] == HELD:
                held[(link["type"], link["key"])] = link["held"]["qid"]
        mononyms: dict[str, set[str]] = {}
        for summary in latest(read_jsonl(wikidata_path), lambda r: r["qid"]).values():
            entity = self.entities.get(summary["qid"])
            if entity is None:
                continue
            for text in [summary.get("label") or "", *summary.get("aliases", [])]:
                key = name_key(text, entity["type"])
                if key:
                    self.index.setdefault((entity["type"], key), set()).add(summary["qid"])
            label = name_key(summary.get("label") or "", entity["type"])
            if entity["type"] == "PERSON" and label and " " not in label:
                mononyms.setdefault(label, set()).add(summary["qid"])
        # a held link resolves its name to no one, not even through that person's own label
        for key, qid in held.items():
            self.index.get(key, set()).discard(qid)
        # spacing: a name is looked up by its letters, as linking's same_name matches it
        self.letters: dict[tuple[str, str], set[str]] = {}
        for (entity_type, key), qids in self.index.items():
            self.letters.setdefault((entity_type, key.replace(" ", "")), set()).update(qids)
        # one word: who holds it -- a one-word label, a word of a longer name, a longer name run
        # together. A one-word alias holds nothing: a nickname, a pen name, initials
        self.holders: dict[str, set[str]] = {k: set(qids) for k, qids in mononyms.items()}
        for (entity_type, key), qids in self.index.items():
            if entity_type == "PERSON" and " " in key:
                for word in {*key.split(), key.replace(" ", "")}:
                    self.holders.setdefault(word, set()).update(qids)

    def resolve(self, name: str, entity_type: str = "PERSON") -> list[str]:
        key = name_key(name, entity_type)
        return self.resolve_key(key, entity_type) if key else []

    def resolve_key(self, key: str, entity_type: str = "PERSON",
                    rules: Sequence[str] = RESOLVE_RULES) -> list[str]:
        """The QIDs a name key resolves to: one person; several, which verify
        refuses; or none. Both rules apply wherever a name is resolved;
        ``rules`` narrows them only to report what each changes
        (resolution_rules).

        - spacing: the name is looked up by its letters, spaces dropped, as
          linking's same_name matches it: "MK Stalin" is "M. K. Stalin".
        - one_word: a one-word person name resolves only to a gallery person
          who holds it -- as a one-word label (a mononym), as a word of one of
          their longer names (a surname), or as a longer name run together (a
          hashtag) -- when that word is also their label, alias or linked
          name, and when no one else is a reading of it: no other holder, and
          no one whose alias it is. Two readings or more make it ambiguous:
          all of them are returned, never the likeliest. A one-word alias that
          is no word of its person's names holds nothing, so a word never
          resolves through a nickname, a pen name or initials ("Modi" is not
          Amedeo Modigliani's "Modì"); it is still a reading ("Bapu", Mahatma
          Gandhi's, is ambiguous with Asaram Bapu)."""
        found = (self.letters.get((entity_type, key.replace(" ", "")), set()) if "spacing" in rules
                 else self.index.get((entity_type, key), set()))
        if "one_word" in rules and entity_type == "PERSON" and " " not in key:
            holders = self.holders.get(key, set())
            readings = holders | found
            if len(readings) > 1:
                return sorted(readings)
            return sorted(readings) if readings and readings <= holders and readings <= found else []
        return sorted(found)

    def best(self, faces: Sequence[dict[str, Any]], qid: str) -> dict[str, Any] | None:
        """The best (query face, reference) pair for one entity, or None."""
        best = None
        for qi, face in enumerate(faces):
            q = decode_embedding(face["embedding"])
            for ref, ref_face, vec in self.refs.get(qid, []):
                score = float(q @ vec)
                if best is None or score > best["score"]:
                    best = {"score": score, "query_face": qi, "reference": ref, "reference_face": ref_face,
                            "same_photo": hamming(face["dhash"], ref_face["dhash"]) <= SAME_PHOTO_BITS}
        return best

    def verify(self, faces: Sequence[dict[str, Any]], name: str, threshold: float | None,
               entity_type: str = "PERSON") -> dict[str, Any]:
        """Whether the image shows the entity ``name`` resolves to, at a
        CERTIFIED ``threshold`` (certified_threshold). With none, verify
        accepts nothing and rejects nothing -- ``match`` stays None -- and
        returns the score and the reference it matched, for a person to check;
        it never falls back to another threshold. A name that resolves to
        several people is refused, unscored: verification is 1:1, and the best
        of several candidates is a search among them. So is a face under the
        size floor: when the face that best matches is smaller than
        MIN_EYE_MOUTH, the answer is no_usable_face, with no score."""
        qids = self.resolve(name, entity_type)
        out: dict[str, Any] = {"claimed_name": name, "entity_type": entity_type,
                               "entities": [{"qid": q, "label": (self.entities.get(q) or {}).get("label")}
                                            for q in qids],
                               "ambiguous_name": len(qids) > 1, "in_gallery": False, "match": None,
                               "score": None, "threshold": threshold, "faces_in_image": len(faces),
                               "matched_entity": None, "matched_reference": None, "query_face": None,
                               "per_reference": []}
        if entity_type != "PERSON":
            out["in_gallery"] = bool(qids)
            out["reason"] = "org_entries_only: an organisation's visual match is not verified"
            return out
        usable = [q for q in qids if self.refs.get(q)]
        out["in_gallery"] = bool(usable)
        if not qids:
            out["reason"] = "name_not_in_gallery"
            return out
        if len(qids) > 1:
            out["reason"] = ("ambiguous_name: the name is more than one person's; verification is 1:1, "
                             "so it answers for none of them")
            return out
        if not usable:
            out["reason"] = "no_usable_reference"
            return out
        if not faces:
            out["reason"] = "no_face_in_image"
            return out
        best_qid = usable[0]
        best = self.best(faces, best_qid)
        face = faces[best["query_face"]]
        out["query_face"] = {**{k: face[k] for k in ("box", "landmarks", "score")}, **face_geometry(face)}
        if too_small(face):
            out["reason"] = (f"no_usable_face: the face that best matches is {out['query_face']['eye_mouth']} px "
                             f"eyes to mouth, under the {MIN_EYE_MOUTH:g} px floor; a score from it would be "
                             "noise presented as a measurement")
            return out
        scored = [f for f in faces if not too_small(f)]
        for ref, _rf, vec in self.refs[best_qid]:
            out["per_reference"].append({"qid": best_qid, "title": ref.get("title"), "score": round(max(
                float(decode_embedding(f["embedding"]) @ vec) for f in scored), 4)})
        ref = best["reference"]
        out.update(score=round(best["score"], 4), matched_entity=best_qid,
                   matched_reference={"title": ref.get("title"), "page_url": ref.get("page_url"),
                                      "path": ref.get("path"), "sha256": ref.get("sha256"),
                                      "licence": ref.get("licence"), "attribution": ref.get("attribution"),
                                      "face_box": best["reference_face"]["box"],
                                      **({"flags": ref["flags"]} if ref.get("flags") else {})},
                   same_photo=best["same_photo"])
        if threshold is None:
            out["reason"] = ("no_certified_threshold: the gallery neither accepts nor rejects at this "
                             "confidence; the score and the matched reference are for a person to check")
        else:
            out["match"] = best["score"] >= threshold
            out["reason"] = "matched" if out["match"] else "below_threshold"
        return out


# --------------------------------------------------------------------------
# calibration: the gallery's accept rate on the VLM's own wrong names
# --------------------------------------------------------------------------

#: The accepted-name precision the gallery must reach (image_text's bar).
TARGET_PRECISION = image_text.VERIFIER_PRECISION
CV_FOLDS = 5
BOOTSTRAP = 2000


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float | None:
    """One-sided lower confidence bound on a proportion k/n."""
    if n == 0:
        return None
    if k == 0:
        return 0.0
    from scipy.stats import beta

    return float(beta.ppf(alpha, k, n - k + 1))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    if n == 0:
        return None
    if k == n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1 - alpha, k + 1, n - k))


def mentioned(name: str, text: str) -> bool:
    """Whether a claim's text carries a name whole: as consecutive words, or,
    for a name of two words or more, run together inside one word as in a
    handle or hashtag (@narendramodi, @realDonaldTrump). A review flag, never a
    label: it is how a person the label rule could not resolve gets looked at."""
    words, tokens = fold(name).split(), fold(text).split()
    if not words:
        return False
    if any(tokens[i:i + len(words)] == words for i in range(len(tokens) - len(words) + 1)):
        return True
    joined = "".join(words)
    return len(words) > 1 and any(joined in t for t in tokens)


def build_trials(population: Sequence[dict[str, Any]], links: dict[tuple[str, str], dict[str, Any]],
                 query_faces: dict[str, dict[str, Any]], gallery: Gallery | None
                 ) -> tuple[list[dict[str, Any]], Counter]:
    """One trial per name the VLM produced from appearance alone on a
    population image: right when it links to anyone the image's claims name,
    wrong when it links to someone else, unlinked when it links to no one. A
    name that resolves to several people is never scored (ambiguous_name),
    because verify refuses it.

    ``figure_label`` keeps the earlier rule beside it -- right only for the
    figure -- because the figures first reported (the pilot's 47.8%, the
    sample's 0.5455) were measured that way and understated by it."""
    token = {name_key(name, "PERSON"): re.compile(tok) for name, _c, tok in image_text.PUBLIC_FIGURES}
    out, tally = [], Counter()
    for img in population:
        figure = links.get(("PERSON", img["figure_key"])) or {}
        figure_qid = figure.get("qid") if figure.get("status") in ("linked", "ambiguous") else None
        if figure_qid is None:
            tally["images_figure_unlinked"] += 1
            continue
        claim_qids = {figure_qid}
        for person in img.get("claim_people", ()):
            if gallery is not None:
                resolved = gallery.resolve(person)
            else:
                plink = links.get(("PERSON", name_key(person, "PERSON") or ""))
                resolved = [plink["qid"]] if plink and plink.get("status") in ("linked", "ambiguous") else []
            if len(resolved) == 1:  # a claim name that could be several people decides nothing
                claim_qids.add(resolved[0])
        faces = (query_faces.get(img["sha256"]) or {}).get("faces") or []
        names = [n for n in img["names"] if n["type"] == "PERSON" and n["evidence"] == "not_written"]
        tally["images"] += 1
        tally["images_with_trials"] += bool(names)
        for n in names:
            key = name_key(n["name"], "PERSON")
            link = links.get(("PERSON", key)) if key else None
            # resolved exactly as Gallery.verify resolves a claimed name, so the
            # calibration measures the system verify runs
            if gallery is not None:
                qids = gallery.resolve(n["name"])
            else:
                qids = [link["qid"]] if link and link.get("status") in ("linked", "ambiguous") else []
            label = "unlinked" if not qids else "right" if claim_qids.intersection(qids) else "wrong"
            trial = {"sha256": img["sha256"], "figure_source": img["figure_source"],
                     "figure_key": img["figure_key"], "figure_qid": figure_qid,
                     "figure_ambiguous": figure.get("status") == "ambiguous", "name": n["name"],
                     "key": key, "qids": qids, "qid": qids[0] if qids else None, "label": label,
                     "figure_label": "unlinked" if not qids else "right" if figure_qid in qids else "wrong",
                     "claim_qids": sorted(claim_qids),
                     "named_in_claim_text": mentioned(n["name"], img.get("claim_text") or ""),
                     "forced": n["forced"], "vlm_confidence": n["confidence"], "faces": len(faces),
                     "ambiguous_link": len(qids) > 1 or bool(link and link.get("status") == "ambiguous"),
                     "token_label": (None if img["figure_source"] != "probe_list" else
                                     "right" if token[img["figure_key"]].search(image_text._normalise(n["name"]))
                                     else "wrong"),
                     "score": None, "scorable": False, "unscorable": None, "same_photo": None}
            if gallery is not None:
                if not qids:
                    trial["unscorable"] = "name_unlinked"
                elif len(qids) > 1:
                    trial["unscorable"] = "ambiguous_name"
                elif not gallery.refs.get(qids[0]):
                    trial["unscorable"] = "no_usable_reference"
                elif not faces:
                    trial["unscorable"] = "no_face_in_image"
                else:
                    q = qids[0]
                    best = gallery.best(faces, q)
                    geometry = face_geometry(faces[best["query_face"]])
                    if geometry["eye_mouth"] < MIN_EYE_MOUTH:  # as verify: under the floor, no score
                        trial.update(unscorable="no_usable_face", qid=q, refs=len(gallery.refs[q]), **geometry)
                    else:
                        trial.update(score=round(best["score"], 6), scorable=True, qid=q,
                                     same_photo=best["same_photo"],
                                     matched_reference=best["reference"].get("sha256"),
                                     refs=len(gallery.refs[q]), **geometry)
                        if best["reference"].get("flags"):
                            trial["matched_flags"] = best["reference"]["flags"]
            out.append(trial)
    return out, tally


def flagged_references(gallery: Gallery, trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The usable references a person flagged (REFERENCE_FLAGS), and the
    scorable trials whose best match is one of them: an age-gap reference may
    match unlike the person's others, so what it decides is counted apart."""
    out = {}
    for flag in REFERENCE_FLAG_VALUES:
        hit = [x for x in trials if x["scorable"] and flag in (x.get("matched_flags") or [])]
        out[flag] = {"usable_references": sum(1 for refs in gallery.refs.values() for ref, _f, _v in refs
                                              if flag in (ref.get("flags") or [])),
                     "trials_matched": len(hit), "by_label": dict(Counter(x["label"] for x in hit))}
    return out


def verification_coverage(gallery: Gallery) -> dict[str, Any]:
    """The ceiling on what verification can answer, whatever the threshold:
    a person none of whose references holds a usable face can only ever get
    "no usable reference"."""
    people = [r for r in gallery.rows.values() if r["type"] == "PERSON"]
    without = [r for r in people if not r["usable_refs"]]
    no_refs = sum(1 for r in without if not r["refs"])
    return {
        "people": len(people), "with_usable_reference_faces": len(people) - len(without),
        "without": len(without), "share_without": image_text._ratio(len(without), len(people), 4),
        "without_any_commons_reference": no_refs, "references_but_no_usable_face": len(without) - no_refs,
        "statement": (f"{len(without):,} of the {len(people):,} people in the gallery have no usable "
                      f"reference face ({no_refs:,} have no Commons reference at all): for them verify can "
                      "only answer 'no usable reference'. That is a ceiling on coverage, apart from the "
                      "threshold. In calibration, a name that resolves only to such a person is unscorable, "
                      "as is one that resolves to several people (verify refuses it): it counts as a "
                      "rejection against TAR (cost_in_true_accepts; tar_scorable leaves it out), as "
                      "coverage when a person confirms it (review_findings), and never as a true or a "
                      "false accept.")}


# --------------------------------------------------------------------------
# alias gaps: names our corpora use for a gallery person that do not resolve to them
# --------------------------------------------------------------------------

#: Two or three consonants are initials ("MK", "RKS"); a vowel makes a name ("Om", "Raj").
_INITIALS = re.compile(r"[b-df-hj-np-tv-z]{2,3}")
#: name_variant's rules, the most certain first.
VARIANT_RULES = ("spacing", "initials_or_middle", "spelling")
#: A one-word name is counted as a gallery person's when that person holds this
#: share of the mentions of our corpora's full names with that surname (or
#: first name)...
WORD_SHARE = 0.9
#: ...out of at least this many mentions.
WORD_SUPPORT = 10
#: A one-word name counts as a spaced name run together ("mkstalin") from this
#: many letters: a shorter one meets names that an apostrophe split ("du'a").
GLUED_LETTERS = 6
#: How many gaps the local report lists, the most mentioned first.
GAP_EXAMPLES = 30


def _initials(words: Sequence[str]) -> list[str]:
    out: list[str] = []
    for w in words:
        out += list(w) if _INITIALS.fullmatch(w) else [w]
    return out


def _word_match(a: str, b: str) -> bool:
    """The same word, or an initial and a word it begins."""
    return a == b or (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b))


def _one_edit(a: str, b: str) -> bool:
    """One letter apart: substituted, inserted or dropped."""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    return len(long) - len(short) == 1 and any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def name_variant(key: str, other: str) -> str | None:
    """How two name keys could be one person's, or None. A rule for COUNTING
    where resolution may miss someone, never for linking:

    - "spacing": the same letters, spaced differently ("mk stalin", "m k stalin");
    - "initials_or_middle", for two words or more: the same surname, and the
      same first name with names left out ("sushil modi", "sushil kumar modi"),
      or initials, one per name ("bs dhanoa", "birender singh dhanoa");
    - "spelling": either of those, one letter apart in a surname of six
      letters or more ("lav aggarwal", "lav agarwal").

    An initial stands for a first name only where one side's names before the
    surname are all initials, two or more: "kanhaiya kumar" is not "k vijay
    kumar", and "selva kumar" is not "s kumar"."""
    a, b = key.split(), other.split()
    if a == b:
        return None
    if "".join(a) == "".join(b):
        return "spacing"
    a, b = _initials(a), _initials(b)
    if len(a) < 2 or len(b) < 2:
        return None
    kind = "initials_or_middle"
    if a[-1] != b[-1]:
        if min(len(a[-1]), len(b[-1])) < 6 or not _one_edit(a[-1], b[-1]):
            return None
        kind = "spelling"
    pa, pb = a[:-1], b[:-1]
    if pa[0] == pb[0]:
        short, long = sorted((pa, pb), key=len)
        rest = iter(long)
        return kind if all(any(_word_match(s, t) for t in rest) for s in short) else None
    initials_only = len(pa) > 1 and (all(len(w) == 1 for w in pa) or all(len(w) == 1 for w in pb))
    return kind if initials_only and len(pa) == len(pb) and all(map(_word_match, pa, pb)) else None


def alias_gaps(names: Sequence[dict[str, Any]], gallery: Gallery, wiki: dict[str, dict[str, Any]]
               ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """How the build scope's PERSON names (name-table rows) resolve, how far
    they collapse into one person's QID, and the PLAUSIBLE ALIASES that do not:
    a name our corpora use for a gallery person that resolves to no one, to
    several people (a claim name that does decides nothing), or to someone
    else. Plausible by rule, never linked: for a name of two words or more that
    resolves to no one, name_variant against every gallery person's label and
    aliases; for one word, the same letters as one person's spaced name, or else
    the person who holds WORD_SHARE of the mentions of our corpora's full names
    with that surname -- or, for a word our corpora never use as a surname,
    that first name. The one-word rule can be wrong where the person is known
    by that one word: a mononym the corpora rarely write in full is outvoted by
    a namesake they do.

    Returns the report and the gaps by name key, each with its ``target``."""
    people = {q for q, e in gallery.entities.items() if e["type"] == "PERSON"}
    resolved = {r["key"]: gallery.resolve_key(r["key"]) for r in names}
    mentions = {r["key"]: scope_mentions(r) for r in names}
    forms: dict[str, list[tuple[str, str]]] = {}  # by first letter, which every rule keeps
    spaced: dict[str, set[str]] = {}
    for q in sorted(people):
        summary = wiki.get(q) or {}
        for text in [summary.get("label") or "", *summary.get("aliases", [])]:
            k = name_key(text, "PERSON")
            if k and " " in k:
                forms.setdefault(k[0], []).append((q, k))
                spaced.setdefault(k.replace(" ", ""), set()).add(q)
    surname: dict[str, Counter] = {}
    first: dict[str, Counter] = {}
    for k, qs in resolved.items():
        words = k.split()
        if len(words) > 1:
            who = qs[0] if len(qs) == 1 and qs[0] in people else ""
            surname.setdefault(words[-1], Counter())[who] += mentions[k]
            first.setdefault(words[0], Counter())[who] += mentions[k]

    def dominant(counts: Counter | None) -> str | None:
        total = sum((counts or {}).values())
        q, n = next(((q, n) for q, n in (counts or Counter()).most_common() if q), ("", 0))
        return q if q and total >= WORD_SUPPORT and n >= WORD_SHARE * total else None

    gaps: dict[str, dict[str, Any]] = {}
    several_people = 0
    for r in names:
        key, qs = r["key"], resolved[r["key"]]
        if " " in key:
            if qs:
                continue  # an exact label or alias: it resolves, to one person or to several
            found: dict[str, str] = {}
            for q, form in forms.get(key[0], ()):
                kind = name_variant(key, form)
                if kind and (q not in found or VARIANT_RULES.index(kind) < VARIANT_RULES.index(found[q])):
                    found[q] = kind
            several_people += len(found) > 1
            if len(found) != 1:
                continue
            (target, rule), = found.items()
        else:
            if len(key) < 3:
                continue
            glued = spaced.get(key, set()) if len(key) >= GLUED_LETTERS else set()
            if len(glued) == 1:
                target, rule = next(iter(glued)), "spacing"
            elif key in surname:
                target, rule = dominant(surname[key]), "surname"
            else:
                target, rule = dominant(first.get(key)), "first_name"
            if target is None or qs == [target]:
                continue
        gaps[key] = {"key": key, "rule": rule, "target": target, "resolves": qs,
                     "outcome": "unresolved" if not qs else "several" if target in qs else "someone_else",
                     "mentions": mentions[key], "vlm_mentions": r["mentions"]["vlm"]}
    one = [qs[0] for qs in resolved.values() if len(qs) == 1]
    families = Counter(one)
    shared = [n for n in families.values() if n > 1]
    listed = sorted(gaps.values(), key=lambda g: (-g["mentions"], g["key"]))
    by_rule: dict[str, dict[str, int]] = {}
    for g in listed:
        b = by_rule.setdefault(g["rule"], {"names": 0, "mentions": 0, "unresolved": 0, "several": 0,
                                           "someone_else": 0})
        b["names"] += 1
        b["mentions"] += g["mentions"]
        b[g["outcome"]] += 1
    total, gap_mentions = sum(mentions.values()), sum(g["mentions"] for g in listed)
    outcomes = Counter(g["outcome"] for g in listed)
    label = {q: (gallery.entities.get(q) or {}).get("label") for q in people}
    return {
        "what": ("the build scope's PERSON names as resolution sees them (the name table, mentions in "
                 "scope), and plausible aliases that do not collapse: rules for counting, never links. "
                 f"Names of two words or more that resolve to no one: {', '.join(VARIANT_RULES)} "
                 "(name_variant) against gallery people's labels and aliases. One word: the same letters "
                 "as one person's spaced name, or the person holding "
                 f"{WORD_SHARE:.0%} of {WORD_SUPPORT}+ mentions of our full names with that surname (first "
                 "name, for a word never used as a surname)"),
        "names": len(names), "mentions": total,
        "resolve": {"to_one_person": len(one), "to_several": sum(1 for qs in resolved.values() if len(qs) > 1),
                    "to_no_one": sum(1 for qs in resolved.values() if not qs)},
        "collapse": {"people": len(families), "people_with_two_names_or_more": len(shared),
                     "names_in_them": sum(shared), "most_names_for_one_person": max(families.values(), default=0)},
        "plausible_aliases_not_collapsing": {
            "names": len(listed), "mentions": gap_mentions,
            "share_of_mentions": image_text._ratio(gap_mentions, total, 4),
            "with_vlm_mentions": sum(1 for g in listed if g["vlm_mentions"]),
            "by_outcome": dict(sorted(outcomes.items())), "by_rule": by_rule,
            "names_that_could_be_several_people": several_people},
        "statement": (f"Of {len(names):,} person names, {len(one):,} resolve to one person; "
                      f"{sum(shared):,} of them collapse into the {len(shared):,} people who have two names or "
                      f"more. {len(listed):,} names ({gap_mentions:,} of {total:,} mentions) plausibly name a "
                      f"gallery person they do not resolve to: {outcomes['unresolved']:,} resolve to no one, "
                      f"{outcomes['several']:,} to several people and {outcomes['someone_else']:,} to someone "
                      "else"),
        "local": {"gaps": [{**g, "target_label": label.get(g["target"]),
                            "resolves_to": [label.get(q) or q for q in g["resolves"]]}
                           for g in listed[:GAP_EXAMPLES]]},
    }, gaps


def alias_gap_labels(trials: Sequence[dict[str, Any]], claim_keys: dict[str, set[str]],
                     gaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The trials the label rule does not call right that an alias gap may
    decide: a claim name on the image plausibly names the VLM's person but
    resolves elsewhere (claim side), or the VLM's own name plausibly names a
    person the claims name and resolves elsewhere (VLM side). Beside them,
    those whose name a claim's text carries although no claim name resolves to
    it (named_in_claim_text): a handle, or a name NER did not find."""
    flagged: dict[str, str] = {}
    not_right = [x for x in trials if x["label"] != "right"]
    for x in not_right:
        targets = {gaps[k]["target"] for k in claim_keys.get(x["sha256"], ()) if k in gaps}
        own = gaps.get(x["key"] or "")
        if targets.intersection(x.get("qids") or ()):
            flagged[trial_id(x)] = "claim_side"
        elif own and own["target"] in targets.union(x["claim_qids"]):
            flagged[trial_id(x)] = "vlm_side"
    named = sum(1 for x in not_right if x.get("named_in_claim_text"))
    return {"trials_not_labelled_right": len(not_right), "an_alias_gap_may_decide": dict(Counter(flagged.values())),
            "named_in_claim_text": named,
            "statement": (f"Of the {len(not_right)} trials the label rule does not call right, an alias gap may "
                          f"decide {len(flagged)}; a claim's text carries the name of {named} (a handle, or a "
                          "name NER did not find), which the review sheet shows a person"),
            "local": {"trials": dict(sorted(flagged.items()))}}


def _resolution_kind(qids: Sequence[str]) -> str:
    return "no_one" if not qids else "one" if len(qids) == 1 else "several"


def resolution_rules(names: Sequence[dict[str, Any]], gallery: Gallery) -> dict[str, Any]:
    """What each of RESOLVE_RULES changes over the build scope's person names,
    against an exact key: how many names resolve differently, from what to
    what, and how many that resolved to one person no longer do -- names
    verify now refuses, wrongly resolved ("Modi", Amedeo Modigliani) or not
    ("Trump", one of ten Trumps)."""
    mentions = {r["key"]: scope_mentions(r) for r in names}
    total = sum(mentions.values())
    label = {q: e.get("label") for q, e in gallery.entities.items()}
    exact = {r["key"]: gallery.resolve_key(r["key"], rules=()) for r in names}
    out: dict[str, Any] = {}
    local: dict[str, Any] = {}
    for rule, rules in (*((r, (r,)) for r in RESOLVE_RULES), ("both", RESOLVE_RULES)):
        changes = []
        for r in names:
            now = gallery.resolve_key(r["key"], rules=rules)
            if now != exact[r["key"]]:
                changes.append((r["key"], f"{_resolution_kind(exact[r['key']])}_to_{_resolution_kind(now)}",
                                len(now)))
        by: dict[str, dict[str, int]] = {}
        for key, change, _n in changes:
            b = by.setdefault(change, {"names": 0, "mentions": 0})
            b["names"] += 1
            b["mentions"] += mentions[key]
        lost = [c for c in changes if c[1] in ("one_to_no_one", "one_to_several")]
        lost_mentions = sum(mentions[k] for k, _c, _n in lost)
        out[rule] = {"names_changed": len(changes), "mentions": sum(mentions[k] for k, _c, _n in changes),
                     "by_change": dict(sorted(by.items())),
                     "resolved_to_one_person_no_longer": {
                         "names": len(lost), "mentions": lost_mentions,
                         "share_of_mentions": image_text._ratio(lost_mentions, total, 4)}}
        local[rule] = [{"key": k, "change": c, "was": [label.get(q) or q for q in exact[k]], "now_people": n,
                        "mentions": mentions[k]}
                       for k, c, n in sorted(changes, key=lambda c: (-mentions[c[0]], c[0]))[:GAP_EXAMPLES]]
    both = out["both"]
    lost = both["resolved_to_one_person_no_longer"]
    cost = (f"{lost['names']:,} names that resolved to one person ({lost['mentions']:,} mentions, "
            f"{(lost['share_of_mentions'] or 0):.1%} of all) now refuse rather than guess: the word is several "
            "people's, or only a nickname. A design choice, not a regression: a guess checks one person's face, and "
            "nothing downstream can tell a wrong guess from a right one")
    return {
        "what": ("over the build scope's person names, what each resolver rule changes against an exact key: "
                 "spacing looks a name up by its letters, as linking does; one_word lets a one-word name "
                 "resolve only to the one gallery person who holds it, never through a nickname alias, and "
                 "makes a word with two readings or more ambiguous"),
        "cost_by_design": {**lost, "statement": cost},
        **out,
        "statement": (f"{cost}. Against an exact key, the rules change how {both['names_changed']:,} names resolve "
                      f"({both['mentions']:,} mentions): spacing {out['spacing']['names_changed']:,}, one_word "
                      f"{out['one_word']['names_changed']:,}"),
        "local": local}


#: Wikidata items for a name itself, which any number of people bear: family
#: and given names, and a page listing several items by one name.
NAME_ITEMS = frozenset({"Q101352", "Q202444", "Q12308941", "Q11879590", "Q3409032", "Q4167410"})
#: What the link step's search answered, by link status, for a name that resolves to no one.
NO_ONE_LINK = {"no_result": "no_item", "no_exact_match": "no_exact_item"}


def no_one_names(names: Sequence[dict[str, Any]], gallery: Gallery,
                 links: dict[tuple[str, str], dict[str, Any]], wiki: dict[str, dict[str, Any]],
                 gaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The build scope's person names that resolve to no one, by what is known
    of why, kept apart because each has its own fix. The first that applies:

    - held: an exact match a person must check first (HELD);
    - a_gallery_person: an alias gap (alias_gaps) -- a variant of a gallery
      person's name. A linking failure with the entry in hand;
    - refused_one_word: a one-word name that matches only a nickname alias
      (a pen name, initials), which the one_word rule never resolves. The
      resolver's choice, not a gap;
    - then what Wikidata's search answered when the name was linked:
      - no_item: no item's label or alias begins with it. No entry under that
        form: a person Wikidata does not have, or a form its search cannot
        match ("P Chidambaram" for "P. Chidambaram", a handle);
      - no_exact_item: items begin with it, none is it;
      - name_only: what is exactly it is a name item (NAME_ITEMS) -- a
        surname or a first name alone, which anyone may bear;
      - not_a_person: what is exactly it is something else (a party, a place,
        a deity, a character): what NER or the VLM took for a person;
      - name_or_thing: both;
      - not_linked: the search never finished.

    Each by names and mentions, and how many are one word."""
    mentions = {r["key"]: scope_mentions(r) for r in names}
    total = sum(mentions.values())

    def cause(key: str) -> str:
        if (links.get(("PERSON", key)) or {}).get("status") == HELD:
            return "held"
        gap = gaps.get(key)
        if gap and gap["outcome"] == "unresolved":
            return "a_gallery_person"
        if gallery.resolve_key(key, rules=()):
            return "refused_one_word"
        link = links.get(("PERSON", key)) or {}
        if link.get("status") in NO_ONE_LINK:
            return NO_ONE_LINK[link["status"]]
        if link.get("status") == "no_type_match":
            kinds = {"name" if NAME_ITEMS.intersection((wiki.get(q) or {}).get("p31") or ()) else "thing"
                     for q in link.get("exact", ())}
            return "name_only" if kinds == {"name"} else "name_or_thing" if "name" in kinds else "not_a_person"
        return "not_linked"

    unresolved = [r for r in names if not gallery.resolve_key(r["key"])]
    by: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[tuple[int, str]]] = {}
    for r in unresolved:
        c = cause(r["key"])
        b = by.setdefault(c, {"names": 0, "mentions": 0, "one_word_names": 0, "single_mention_names": 0})
        b["names"] += 1
        b["mentions"] += mentions[r["key"]]
        b["one_word_names"] += " " not in r["key"]
        b["single_mention_names"] += mentions[r["key"]] == 1
        examples.setdefault(c, []).append((mentions[r["key"]], r["key"]))
    n_mentions = sum(b["mentions"] for b in by.values())
    order = ("held", "a_gallery_person", "refused_one_word", "no_item", "no_exact_item", "name_only",
             "name_or_thing", "not_a_person", "not_linked")
    by = {c: {**by[c], "share_of_mentions": image_text._ratio(by[c]["mentions"], total, 4)}
          for c in order if c in by}

    def part(c: str) -> str:
        b = by.get(c) or {"names": 0, "mentions": 0}
        return f"{b['names']:,} ({b['mentions']:,} mentions)"

    return {
        "what": ("the build scope's person names that resolve to no one, by the first cause that applies: an "
                 "alias gap; a one-word name the resolver refuses; what Wikidata's search answered at linking"),
        "names": len(unresolved), "mentions": n_mentions,
        "share_of_mentions": image_text._ratio(n_mentions, total, 4), "by_cause": by,
        "statement": (f"{len(unresolved):,} person names ({n_mentions:,} of {total:,} mentions) resolve to no one. "
                      f"Exact matches held for a person's check: {part('held')}. "
                      f"Linking failures with the person in the gallery, by rule: {part('a_gallery_person')}. "
                      f"One-word names the resolver refuses: {part('refused_one_word')}. No Wikidata item by "
                      f"that name: {part('no_item')}, and items that begin with it but are not it: "
                      f"{part('no_exact_item')} -- people Wikidata does not have, or forms its search cannot "
                      f"match. Exactly a name item, a surname or first name alone: {part('name_only')}; exactly "
                      f"something that is no person: {part('not_a_person')}; both: {part('name_or_thing')}"),
        "local": {c: [k for _m, k in sorted(examples[c], key=lambda e: (-e[0], e[1]))[:GAP_EXAMPLES]] for c in by}}


def operating_point(trials: Sequence[dict[str, Any]], t: float | None) -> dict[str, Any]:
    """Accepts at threshold ``t``: an unscorable trial is never accepted."""
    right = [x for x in trials if x["label"] == "right"]
    wrong = [x for x in trials if x["label"] != "right"]
    acc_r = sum(1 for x in right if t is not None and x["scorable"] and x["score"] >= t)
    acc_w = sum(1 for x in wrong if t is not None and x["scorable"] and x["score"] >= t)
    accepted = acc_r + acc_w
    return {"threshold": None if t is None else round(t, 4), "right": len(right), "wrong": len(wrong),
            "accepted_right": acc_r, "accepted_wrong": acc_w,
            "tar": image_text._ratio(acc_r, len(right), 4), "fmr": image_text._ratio(acc_w, len(wrong), 4),
            "fmr_upper_95": None if not wrong else round(clopper_pearson_upper(acc_w, len(wrong)), 4),
            "precision": image_text._ratio(acc_r, accepted, 4),
            "precision_lower_95": None if not accepted else round(clopper_pearson_lower(acc_r, accepted), 4)}


def choose_threshold(trials: Sequence[dict[str, Any]], rule: str) -> float | None:
    """The lowest threshold -- the most right names kept -- at which the names
    the gallery accepts are right TARGET_PRECISION of the time: by the one-sided
    95% lower bound (``bound``), or by the point estimate (``point``)."""
    scores = sorted({x["score"] for x in trials if x["scorable"]})
    for t in scores:
        p = operating_point(trials, t)
        value = p["precision_lower_95"] if rule == "bound" else p["precision"]
        if value is not None and value >= TARGET_PRECISION:
            return t
    return None


def accepted_names_needed(precision: float, cap: int = 1_000_000) -> int | None:
    """About how many accepted names a precision, held, needs before its
    one-sided 95% lower bound reaches TARGET_PRECISION. None at or below the
    bar, or past ``cap``: a precision that close to 0.90 is not certifiable."""
    if precision <= TARGET_PRECISION:
        return None

    def reaches(n: int) -> bool:
        lower = clopper_pearson_lower(math.floor(precision * n), n)
        return lower is not None and lower >= TARGET_PRECISION

    hi = 1
    while not reaches(hi):
        if hi > cap:
            return None
        hi *= 2
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if reaches(mid) else (mid, hi)
    return hi


def bound_shortfall(trials: Sequence[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    """When no threshold reaches the one-sided 95% bound: the threshold that
    comes closest (the highest lower bound; the lowest threshold on a tie) and
    why it falls short. Falling back to a point estimate would hide which:

    - "too_few_trials": its precision clears 0.90 as a point estimate, and the
      same precision over more trials would certify it (accepted_names_needed);
    - "overlap": not above 0.90 even as a point estimate. Wrong names score
      among the right ones, and more trials cannot fix that."""
    best_t, best = None, None
    for t in sorted({x["score"] for x in trials if x["scorable"]}):
        p = operating_point(trials, t)
        if p["precision_lower_95"] is not None and (best is None
                                                    or p["precision_lower_95"] > best["precision_lower_95"]):
            best_t, best = t, p
    if best is None:
        return None, {"threshold": None, "reason": "no_scorable_trials", "statement": "no name can be scored"}
    n = best["accepted_right"] + best["accepted_wrong"]
    head = (f"the highest lower bound of accepted-name precision is {best['precision_lower_95']}, at "
            f"{best['threshold']}: {best['accepted_right']} of {n} accepted names right ({best['precision']})")
    out = {"threshold": best["threshold"], "at": best, "accepted_names": n}
    if best["precision"] > TARGET_PRECISION:
        needed = accepted_names_needed(best["precision"])
        return best_t, {**out, "reason": "too_few_trials", "accepted_names_needed": needed,
                        "statement": head + (". That clears 0.90 as a point estimate, on too few trials to "
                                             "certify it: at that precision about "
                                             f"{needed:,} accepted names would" if needed else
                                             ". That clears 0.90 as a point estimate, but too narrowly for "
                                             "any practical number of trials to certify it")}
    return best_t, {**out, "reason": "overlap",
                    "statement": head + (", not above 0.90 even as a point estimate: wrong names score among "
                                         "the right ones, and more trials cannot fix that")}


def certified_threshold(report: dict[str, Any] | None, cid: str) -> float | None:
    """The threshold verify may decide at: one a calibration of THIS face
    configuration certified, by the one-sided 95% bound on the complete run.
    Anything else -- no report, another configuration, a candidate, an older
    report without the flag -- is no threshold, and verify then refuses."""
    if not report or report.get("face_config") != cid:
        return None
    t = report.get("threshold") or {}
    return t.get("chosen") if t.get("certified") is True and t.get("chosen") is not None else None


def cross_validated(trials: Sequence[dict[str, Any]], rule: str, seed: int = SEED) -> dict[str, Any]:
    """The threshold chosen on four folds of IMAGES and applied to the fifth:
    an estimate of what the chosen threshold does on images it never saw."""
    images_ = sorted({x["sha256"] for x in trials}, key=lambda s: image_text._priority(seed, s))
    fold_of = {s: i % CV_FOLDS for i, s in enumerate(images_)}
    held: list[dict[str, Any]] = []
    thresholds = []
    for k in range(CV_FOLDS):
        train = [x for x in trials if fold_of[x["sha256"]] != k]
        test = [x for x in trials if fold_of[x["sha256"]] == k]
        t = choose_threshold(train, rule)
        thresholds.append(None if t is None else round(t, 4))
        held += [{**x, "accepted": t is not None and x["scorable"] and x["score"] >= t} for x in test]
    right = [x for x in held if x["label"] == "right"]
    wrong = [x for x in held if x["label"] != "right"]
    acc_r = sum(x["accepted"] for x in right)
    acc_w = sum(x["accepted"] for x in wrong)
    return {"rule": rule, "fold_thresholds": thresholds, "accepted_right": acc_r, "accepted_wrong": acc_w,
            "tar": image_text._ratio(acc_r, len(right), 4), "fmr": image_text._ratio(acc_w, len(wrong), 4),
            "precision": image_text._ratio(acc_r, acc_r + acc_w, 4)}


def bootstrap(trials: Sequence[dict[str, Any]], t: float, seed: int = SEED) -> dict[str, Any]:
    """95% intervals at ``t`` from resampling IMAGES (a name's trials share one)."""
    import numpy as np

    by_image: dict[str, list[dict[str, Any]]] = {}
    for x in trials:
        by_image.setdefault(x["sha256"], []).append(x)
    keys = sorted(by_image)
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {"tar": [], "fmr": [], "precision": []}
    for _ in range(BOOTSTRAP):
        sample = [x for i in rng.integers(0, len(keys), len(keys)) for x in by_image[keys[i]]]
        p = operating_point(sample, t)
        for k in stats:
            if p[k] is not None:
                stats[k].append(p[k])
    return {k: ([round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
                if v else None) for k, v in stats.items()}


def random_pair_scores(trials: Sequence[dict[str, Any]], gallery: Gallery, query_faces: dict[str, Any],
                       seed: int = SEED) -> list[float]:
    """CONTRAST ONLY, never used to choose anything: the faces of the right
    names, each scored against a random OTHER person in the gallery -- what a
    calibration on random pairs would measure."""
    import numpy as np

    people = sorted(q for q, refs in gallery.refs.items() if refs)
    if len(people) < 2:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for x in trials:
        if x["label"] != "right" or not x["scorable"]:
            continue
        others = [q for q in people if q != x["qid"]]
        faces = query_faces[x["sha256"]]["faces"]
        best = gallery.best(faces, others[int(rng.integers(0, len(others)))])
        if best is not None and not too_small(faces[best["query_face"]]):  # what verify would score
            out.append(best["score"])
    return out


def look_alike_finding(trials: Sequence[dict[str, Any]], random_scores: Sequence[float],
                       t: float | None, p: float | None) -> dict[str, Any] | None:
    """Why the threshold is calibrated on the VLM's own wrong names: the names a
    VLM gets wrong are people who look like the face, and they score far above
    random strangers. Measured two ways: how the two score, and what a
    threshold set on random pairs would accept of the VLM's own names."""
    import numpy as np

    right = [x for x in trials if x["label"] == "right"]
    wrong = [x for x in trials if x["label"] != "right"]
    wrong_scores = [x["score"] for x in wrong if x["scorable"]]
    if not random_scores or not right or p is None:
        return None
    top = max(random_scores)

    def precision_on_strangers(th: float) -> float | None:
        tar = sum(1 for x in right if x["scorable"] and x["score"] >= th) / len(right)
        fmr = sum(s >= th for s in random_scores) / len(random_scores)
        return None if not tar else p * tar / (p * tar + (1 - p) * fmr)

    candidates = sorted({x["score"] for x in trials if x["scorable"]} | set(random_scores))
    t_random = next((th for th in candidates if (v := precision_on_strangers(th)) is not None
                     and v >= TARGET_PRECISION), None)
    on_vlm = operating_point(trials, t_random)
    above = sum(s > top for s in wrong_scores)
    holds = on_vlm["precision"] is not None and on_vlm["precision"] >= TARGET_PRECISION
    finding = (
        f"Scored against a random other person, the right names' faces reach at most {top:.3f} "
        f"(median {float(np.median(random_scores)):.3f}); {above} of the {len(wrong_scores)} scorable "
        f"wrong names the VLM gave score above every stranger. "
        + ("No threshold reaches the bar even on random strangers. " if t_random is None else
           f"A threshold calibrated on random pairs would sit at {t_random:.3f}; on the VLM's own names it "
           f"accepts {on_vlm['accepted_wrong']} of {on_vlm['wrong']} wrong names, an accepted-name precision "
           f"of {on_vlm['precision']} against the {TARGET_PRECISION} required. ")
        + ("A random-pair false-match rate measures strangers; the names a gallery must refuse are "
           "the look-alikes the VLM chose, so the threshold is calibrated on those."
           if not holds else
           "On this population a random-pair threshold would also hold the bar; the VLM's own wrong "
           "names remain the measure."))
    return {
        "what": ("the VLM's wrong names against random strangers: the right names' faces, each scored "
                 "against a random other person in the gallery (seeded). A contrast, never an input"),
        "random_strangers": {"pairs": len(random_scores), "max_score": round(top, 4),
                             "median_score": round(float(np.median(random_scores)), 4),
                             "accepted_at_threshold": None if t is None else sum(s >= t for s in random_scores)},
        "vlm_wrong_names": {"wrong": len(wrong), "scorable": len(wrong_scores),
                            "median_score": (round(float(np.median(wrong_scores)), 4)
                                             if wrong_scores else None),
                            "above_every_stranger": above,
                            "accepted_at_threshold": None if t is None else sum(s >= t for s in wrong_scores)},
        "right_names_median_score": round(float(np.median([x["score"] for x in right if x["scorable"]]
                                                          or [np.nan])), 4),
        "if_calibrated_on_random_pairs": {
            "threshold": None if t_random is None else round(t_random, 4),
            "rule": ("the lowest threshold at which p*TAR / (p*TAR + (1-p)*FMR) >= 0.90 with FMR measured "
                     "on random strangers (point estimate), p the VLM's QID precision"),
            "on_the_vlms_own_names": on_vlm},
        "finding": finding}


#: When the two most-tried figures carry more than this share of the trials,
#: the pooled numbers describe those two people more than the method.
CONCENTRATION_SHARE = 0.5


def figure_table(trials: Sequence[dict[str, Any]], t: float | None) -> list[dict[str, Any]]:
    """Per figure the claims name, most trials first: its trials and what the
    threshold accepts of them."""
    by: dict[str, list[dict[str, Any]]] = {}
    for x in trials:
        by.setdefault(x["figure_qid"], []).append(x)
    rows = []
    for qid, xs in by.items():
        at = operating_point(xs, t)
        rows.append({"figure_qid": qid, "figure_source": sorted({x["figure_source"] for x in xs}),
                     "trials": len(xs), "right": at["right"], "wrong": at["wrong"],
                     "unlinked": sum(1 for x in xs if x["label"] == "unlinked"),
                     "scorable": sum(1 for x in xs if x["scorable"]),
                     "accepted_right": at["accepted_right"], "accepted_wrong": at["accepted_wrong"]})
    return sorted(rows, key=lambda r: (-r["trials"], r["figure_qid"]))


def concentration(table: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How far the trials are a few people's. The effective number of figures
    is 1 / sum(share^2): equal to the figures when all carry the same."""
    n = sum(r["trials"] for r in table)
    if not n:
        return {"figures": 0, "trials": 0, "concentrated": None, "statement": "no trials"}
    shares = [r["trials"] / n for r in table]
    top2 = sum(shares[:2])
    concentrated = top2 > CONCENTRATION_SHARE
    return {"figures": len(table), "trials": n, "top_1_share": round(shares[0], 4),
            "top_2_share": round(top2, 4), "effective_figures": round(1 / sum(s * s for s in shares), 2),
            "figures_with_5_trials_or_more": sum(1 for r in table if r["trials"] >= 5),
            "figures_with_1_trial": sum(1 for r in table if r["trials"] == 1),
            "concentrated": concentrated,
            "statement": (f"The two most-tried figures carry {top2:.0%} of the trials: the pooled numbers "
                          "are mostly about those two people, not about the method. Read the "
                          "figure-stratified estimate, which weighs every figure the same."
                          if concentrated else
                          f"The two most-tried figures carry {top2:.0%} of the trials; the "
                          "figure-stratified estimate weighs every figure the same, beside the pooled one.")}


def figure_stratified(trials: Sequence[dict[str, Any]], t: float | None,
                      seed: int = SEED) -> dict[str, Any] | None:
    """Every figure weighs the same: each of its trials counts 1 / its trials.
    At the pooled threshold, with 95% intervals from resampling FIGURES, so the
    interval reflects how many people were tried, not how many names; and the
    lowest threshold whose one-sided 95% figure-bootstrap bound reaches the bar."""
    import numpy as np

    if not trials:
        return None
    figs = sorted({x["figure_qid"] for x in trials})
    position = {q: i for i, q in enumerate(figs)}
    idx = np.array([position[x["figure_qid"]] for x in trials])
    per_figure = np.bincount(idx, minlength=len(figs)).astype(float)
    right = np.array([x["label"] == "right" for x in trials])
    score = np.array([x["score"] if x["scorable"] else -np.inf for x in trials])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(figs), size=(BOOTSTRAP, len(figs)))
    copies = np.zeros((BOOTSTRAP, len(figs)))
    np.add.at(copies, (np.repeat(np.arange(BOOTSTRAP), len(figs)), draws.ravel()), 1)
    # row 0 is the estimate itself, every figure once; the rest are bootstrap draws
    weights = np.vstack([np.ones(len(figs)), copies])[:, idx] / per_figure[idx]

    def rates(th: float | None) -> dict[str, Any]:
        accepted = score >= (np.inf if th is None else th)
        w_right, w_wrong = (weights * right).sum(1), (weights * ~right).sum(1)
        a_right, a_wrong = (weights * (right & accepted)).sum(1), (weights * (~right & accepted)).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"vlm_precision": w_right / (w_right + w_wrong), "tar": a_right / w_right,
                    "fmr": a_wrong / w_wrong, "precision": a_right / (a_right + a_wrong)}

    def value(v: float) -> float | None:
        return None if np.isnan(v) else round(float(v), 4)

    def interval(v) -> list[float] | None:
        v = v[~np.isnan(v)]
        return None if not len(v) else [round(float(np.quantile(v, 0.025)), 4),
                                        round(float(np.quantile(v, 0.975)), 4)]

    def lower(v) -> float | None:
        v = v[~np.isnan(v)]
        return None if not len(v) else float(np.quantile(v, 0.05))

    at = rates(t)
    # every candidate threshold at once: trials by falling score, weights summed cumulatively
    order = np.argsort(-score, kind="stable")
    falling = score[order]
    cum_right = np.cumsum(weights[:, order] * right[order], axis=1)
    cum_wrong = np.cumsum(weights[:, order] * ~right[order], axis=1)
    t_bound = None
    for th in sorted({float(s) for s in falling if np.isfinite(s)}):
        j = int(np.searchsorted(-falling, -th, side="right")) - 1  # the last trial scoring >= th
        with np.errstate(invalid="ignore", divide="ignore"):
            lb = lower((cum_right[1:, j] / (cum_right[1:, j] + cum_wrong[1:, j])))
        if lb is not None and lb >= TARGET_PRECISION:
            t_bound = th
            break
    return {"what": ("every figure weighs the same: each trial counts 1 / its figure's trials; 95% "
                     "intervals by resampling figures"),
            "figures": len(figs), "threshold": None if t is None else round(t, 4),
            **{k: value(v[0]) for k, v in at.items()},
            "precision_lower_95": None if (lb := lower(at["precision"][1:])) is None else round(lb, 4),
            "bootstrap_95": {k: interval(v[1:]) for k, v in at.items()},
            "threshold_by_figure_bound": None if t_bound is None else round(t_bound, 4),
            "note": ("the threshold at which the one-sided 95% lower bound of figure-weighted precision "
                     "reaches 0.90; when it is above the pooled one, the pooled threshold leans on the "
                     "most-tried figures")}


def calibrate(trials: Sequence[dict[str, Any]], tally: Counter, gallery: Gallery,
              query_faces: dict[str, Any], backend: str, ledger: dict[str, Any]) -> dict[str, Any]:
    """The report: the candidate threshold, what it costs in true accepts, and
    the rest. The threshold is set only by the one-sided 95% bound, and only
    by a calibration over the complete Factify2 run (``ledger["complete"]``):
    a partial ledger gives a partial population. Short of the bound it stays
    unset even on the complete run -- a point estimate is never adopted -- and
    the report gives the curve, the threshold closest to the bound and why it
    falls short (bound_shortfall). Unset, verify accepts and rejects nothing."""
    headline = [x for x in trials if not x.get("same_photo") and not x["forced"]]
    named = [x for x in headline if x["label"] != "unlinked"]
    right = sum(1 for x in headline if x["label"] == "right")
    p = image_text._ratio(right, len(headline), 4)
    r_max = None if p is None else image_text.max_false_accept_ratio(p)
    t_bound = choose_threshold(headline, "bound")
    t_point = choose_threshold(headline, "point")
    candidate, shortfall = (t_bound, None) if t_bound is not None else bound_shortfall(headline)
    candidate_by = ("one-sided 95% lower bound of accepted-name precision >= 0.90" if t_bound is not None else
                    f"closest to the bound and short of it: {shortfall['statement']}")
    chosen = t_bound if ledger.get("complete") else None
    of = f"{ledger['worklist']:,}" if ledger.get("worklist") else "an unknown number of"
    at = operating_point(headline, candidate)
    rejected_right = [x for x in headline if x["label"] == "right"
                      and not (candidate is not None and x["scorable"] and x["score"] >= candidate)]
    scorable_right = sum(1 for x in headline if x["label"] == "right" and x["scorable"])
    grid = sorted({round(x["score"], 2) for x in headline if x["scorable"]})
    strata = {}
    for name, test in (("one_face", lambda x: x["faces"] == 1), ("several_faces", lambda x: x["faces"] > 1),
                       ("figure_from_probe_list", lambda x: x["figure_source"] == "probe_list"),
                       ("figure_from_claim_ner", lambda x: x["figure_source"] == "claim_ner"),
                       ("ambiguous_link", lambda x: x["ambiguous_link"]),
                       ("figure_link_ambiguous", lambda x: x.get("figure_ambiguous"))):
        subset = [x for x in headline if test(x)]
        if subset:
            strata[name] = operating_point(subset, candidate)
    probe_list = [x for x in headline if x["token_label"] is not None]
    table = figure_table(headline, candidate)
    top_two = {r["figure_qid"] for r in table[:2]}
    for r in table:
        r["label"] = (gallery.entities.get(r["figure_qid"]) or {}).get("label")
    return {
        "backend": backend, "face_config": gallery.cid,
        "coverage": verification_coverage(gallery),
        "reference_flags": flagged_references(gallery, headline),
        "population": {"what": ("Factify2 TRAIN claim images (every record using the image is "
                                "train) whose claims say an image shows someone and name one figure: "
                                "exactly one of the probe's listed figures, or else exactly one person "
                                "by claim NER. The figure is the stratum for per-figure reporting; a "
                                "name is scored against everyone the image's claims name"),
                       **dict(sorted(tally.items()))},
        "trials": {"total": len(trials), "headline": len(headline),
                   "excluded_same_photo": sum(1 for x in trials if x.get("same_photo")),
                   "forced_lines": sum(1 for x in trials if x["forced"]),
                   "by_label": dict(Counter(x["label"] for x in headline)),
                   "unscorable": dict(Counter(x["unscorable"] for x in headline if x["unscorable"]))},
        "vlm_precision": {
            "qid": p, "names": len(headline), "right": right,
            "excluding_unlinked": image_text._ratio(right, len(named), 4),
            "rule": ("right when the name resolves, by QID, to anyone the image's claims name: the "
                     "probe's listed figures and claim NER's people. A claim name that resolves to "
                     "several people decides nothing"),
            "single_figure_rule": image_text._ratio(
                sum(1 for x in headline if x.get("figure_label") == "right"), len(headline), 4),
            "token_rule_on_probe_list_images": image_text._ratio(
                sum(1 for x in probe_list if x["token_label"] == "right"), len(probe_list), 4),
            "note": ("The earlier figures carried a defect in this measurement: the pilot's 47.8% and "
                     "the first QID-based figure (0.5455, the 2026-10-21 sample pass) scored a name "
                     "against one figure per image, so a person the claims name beside that figure, "
                     "correctly named, counted as wrong. Both were understated. single_figure_rule is "
                     "that defective figure recomputed on these trials. The token rule is the pilot's "
                     "own method on the same images: its loose token pattern, with the same defect. "
                     "The pilot's 47.8% is also APPROXIMATE for that token pattern")},
        "bar": {"accepted_name_precision": TARGET_PRECISION,
                "max_fmr_over_tar": None if r_max is None else round(r_max, 4),
                "rule": "p*TAR / (p*TAR + (1-p)*FMR) >= 0.90, i.e. FMR/TAR <= p / (9 (1 - p))"},
        "threshold": {"chosen": None if chosen is None else round(chosen, 4),
                      "certified": chosen is not None,
                      "chosen_by": (candidate_by if chosen is not None else
                                    "unset: only a calibration over the complete Factify2 run sets the "
                                    f"threshold, and this ledger holds {ledger.get('images', 0):,} of {of} "
                                    "images. The candidate, and every figure at it, are reported and "
                                    "never adopted" if not ledger.get("complete") else
                                    "unset: no threshold reaches the one-sided 95% lower bound of 0.90, "
                                    "and a point estimate is never adopted, so the gallery does not decide "
                                    f"at this confidence. Short of the bound: {shortfall['statement']}"),
                      "candidate": None if candidate is None else round(candidate, 4),
                      "candidate_by": candidate_by,
                      "bound_shortfall": shortfall,
                      "by_bound": None if t_bound is None else round(t_bound, 4),
                      "by_point_estimate": None if t_point is None else round(t_point, 4)},
        "at_threshold": at,
        "fmr_over_tar": (None if not at["tar"] or at["fmr"] is None else round(at["fmr"] / at["tar"], 4)),
        "cost_in_true_accepts": {
            "right_names": at["right"], "rejected": len(rejected_right),
            "rejected_unscorable": dict(Counter(x["unscorable"] for x in rejected_right if x["unscorable"])),
            "rejected_below_threshold": sum(1 for x in rejected_right if x["scorable"]),
            "right_names_scorable": scorable_right,
            "tar_scorable": image_text._ratio(at["accepted_right"], scorable_right, 4),
            "note": ("TAR is (right names the gallery can score / right names) x tar_scorable: a name it "
                     "cannot score is coverage, and tar_scorable is the recognizer's own rate without it")},
        "bootstrap_95": None if candidate is None else bootstrap(headline, candidate),
        "cross_validated": {"bound": cross_validated(headline, "bound"),
                            "point": cross_validated(headline, "point")},
        "curve": [operating_point(headline, t) for t in grid],
        "strata": strata,
        "figures": {"what": ("per figure, the image's stratum, most trials first: trials and what the "
                             "candidate threshold accepts of them. Local report only: the published one "
                             "gives ranks, never a name or QID"),
                    "concentration": concentration(table), "table": table},
        "figure_stratified": {**(figure_stratified(headline, candidate) or {}),
                              "pooled_without_top_two_figures": operating_point(
                                  [x for x in headline if x["figure_qid"] not in top_two], candidate)},
        "look_alike_finding": look_alike_finding(
            headline, random_pair_scores(headline, gallery, query_faces), candidate, p),
        "verite": "not read: calibration reads Factify2 train only",
    }


# --------------------------------------------------------------------------
# commands for the network, the faces and the calibration
# --------------------------------------------------------------------------


def _run_dir(args) -> Path:
    run_dir = Path(args.ledger_dir) if getattr(args, "ledger_dir", None) else current_run_dir()
    if not run_dir.is_absolute() and not run_dir.is_dir():
        run_dir = image_text.OUT_ROOT / run_dir
    return run_dir


def _ledger_read(args) -> LedgerRead:
    return read_ledger(_run_dir(args) / "factify2.jsonl", "factify2")


def _population(read: LedgerRead) -> tuple[list[dict[str, Any]], Counter]:
    from scripts.processed_loader import iter_rows

    spans = {r["record_id"]: r for r in read_jsonl(
        CLAIM_NER_DIR / config_id_of(ner_config()) / "factify2.jsonl")}
    return calibration_population(read, iter_rows("factify2"), images.load_index("factify2"), spans)


def refresh_entities() -> int:
    """Fetch every stored item's summary again (the last row wins): for a
    summary schema that changed, as when "mul" labels were added."""
    qids = sorted(latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"]))
    wiki, prov = JsonlLedger(WIKIDATA_PATH), JsonlLedger(PROVENANCE_PATH)
    wiki.open()
    prov.open()
    try:
        entities = fetch_entities(live_net(ProvenanceLog(prov)), qids)
        wiki.append_many([{"pass": PASS_ID, "at": utcnow(), **e} for e in entities.values()])
    finally:
        wiki.close()
        prov.close()
    log(f"refreshed {len(entities):,} of {len(qids):,} item summaries")
    return 0


#: A link a person must check before it counts: the exact match is kept in
#: the row (``held``), with why and by whom it was held, and nothing resolves
#: the name to it. Never searched again; ``link --release`` restores it.
HELD = "held"


def held_row(row: dict[str, Any], because: str, by: str) -> dict[str, Any]:
    """The link row that holds ``row`` (a linked or ambiguous link): its match
    and candidates kept under ``held`` with the reason, so a later pass
    revisits the hold instead of rediscovering it."""
    match = {k: row.get(k) for k in ("status", "qid", "candidates")}
    return {**{k: row[k] for k in ("schema", "type", "key", "query") if k in row}, "queries": row.get("queries"),
            "pass": HELD_PASS_ID, "at": utcnow(), "status": HELD, "held": {**match, "because": because, "by": by}}


def reheld_row(row: dict[str, Any], because: str, by: str) -> dict[str, Any]:
    """A held row held again by whoever reviewed it: the same match, their
    reason and name, and the hold it replaces, so the ledger says who decided
    (an agent's hold a person kept is the person's hold now)."""
    held = row["held"]
    return {**{k: row[k] for k in ("schema", "type", "key", "query", "queries") if k in row},
            "pass": HELD_PASS_ID, "at": utcnow(), "status": HELD,
            "held": {**{k: held.get(k) for k in ("status", "qid", "candidates")}, "because": because, "by": by,
                     "replaces": {"because": held.get("because"), "by": held.get("by"), "at": row.get("at")}}}


def released_row(row: dict[str, Any], by: str) -> dict[str, Any]:
    """The link a held row kept, restored, with who released it."""
    match = row["held"]
    return {**{k: row[k] for k in ("schema", "type", "key", "query", "queries") if k in row},
            "pass": HELD_PASS_ID, "at": utcnow(), "status": match["status"], "qid": match["qid"],
            "candidates": match.get("candidates"), "released": {"from_hold": match["because"], "by": by}}


#: The namesake check's keywords (2026-10-23): a description that reads as one
#: of these domains, other than the news, marks a link a person should check
#: before its references are fetched. A life that ended before 1950 reads
#: historical. Keywords, not a judgment: they flag right people too.
NAMESAKE_DOMAINS = {
    "academic": re.compile(
        r"\b(research(er)?|scientists?|physicist|chemist|biologist|botanist|mycologist|entomologist|zoologist|"
        r"geologist|mathematician|astronomer|historian|archaeologist|academic|professor|scholar|indologist|"
        r"linguist|economist|orcid|ichthyologist|herpetologist|arachnologist|pathologist)\b", re.I),
    "athlete": re.compile(
        r"\b(cricket(er| player)|footballer|football player|athlete|wrestler|boxer|hockey|tennis|sprinter|"
        r"olympian|basketball|baseball|racing driver|umpire|cyclist|swimmer|badminton|chess|golfer|"
        r"sportsperson|jockey|shooter)\b", re.I),
}
_LIFESPAN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*[-–—]\s*(1[0-9]{3}|20[0-2][0-9])\b")
HISTORICAL_BEFORE = 1950


def description_domain(description: str | None) -> str:
    """The namesake check's reading of a linked item's description: the
    domains it reads as ("academic+historical"), "no description", or "none
    of these"."""
    if not description:
        return "no description"
    out = [domain for domain, rx in NAMESAKE_DOMAINS.items() if rx.search(description)]
    lived = _LIFESPAN.search(description)
    if lived and int(lived.group(2)) < HISTORICAL_BEFORE:
        out.append("historical")
    return "+".join(out) or "none of these"


def name_shape(query: str, entity_type: str = "PERSON") -> str:
    """How a name was searched, for the namesake check: "one word",
    "initials" (initials linking dots, dotted_initials), or "full name". By
    the query, never the key: a hashtag's key ("madhukishwar") has no space,
    but its search ("Madhu Kishwar") was a full name's."""
    if len(query.split()) == 1:
        return "one word"
    return "initials" if dotted_initials(query, entity_type) else "full name"


def link_due(row: dict[str, Any] | None, query: str, entity_type: str, known: dict[str, Any]) -> bool:
    """Whether a name is linked (again) this run, given its last link row:
    never linked; an error; a "no type match" whose candidates' data never
    arrived (a batch lost before failed batches were told apart); or a name
    with initials that linked nothing and was never searched with them
    dotted and spaced (link_names searches both forms). A name that linked
    keeps its link."""
    if row is None or row["status"] == "error":
        return True
    if row["status"] == "no_type_match" and any(q not in known for q in row.get("exact", [])):
        return True
    dotted = dotted_initials(query, entity_type)
    return (row["status"] in ("no_result", "no_exact_match", "no_type_match") and dotted is not None
            and dotted not in (row.get("queries") or [row.get("query")]))


def cmd_hold(args) -> int:
    """Hold or release links, offline: a hold keeps the match in the row with
    its reason; a release restores the match a hold kept. Holding a held link
    again records who holds it now and why, beside the hold it replaces."""
    if not args.by:
        raise GalleryError("say who holds or releases them (--by)")
    done = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    reasons = json.loads(Path(args.reasons).read_text(encoding="utf-8")) if args.reasons else {}
    rows, refused = [], []
    for key in args.hold or args.release:
        row = done.get(("PERSON", key))
        because = reasons.get(key) or args.because
        if args.hold and not because:
            refused.append(f"{key}: no reason given (--because or --reasons)")
        elif args.hold and row and row["status"] in ("linked", "ambiguous"):
            rows.append(held_row(row, because, args.by))
        elif args.hold and row and row["status"] == HELD:
            rows.append(reheld_row(row, because, args.by))
        elif args.release and row and row["status"] == HELD:
            rows.append(released_row(row, args.by))
        else:
            refused.append(f"{key}: {row['status'] if row else 'never linked'}")
    if refused:
        raise GalleryError(f"nothing written; cannot {'hold' if args.hold else 'release'}: {'; '.join(refused)}")
    ledger = JsonlLedger(LINKS_PATH)
    ledger.open()
    try:
        ledger.append_many(rows)
    finally:
        ledger.close()
    log(f"{'held' if args.hold else 'released'} {len(rows)} link(s) by {args.by}")
    return 0


def cmd_link(args) -> int:
    if args.hold or args.release:
        return cmd_hold(args)
    if args.refresh_entities:
        return refresh_entities()
    names = list(read_jsonl(NAMES_PATH))
    if not names:
        raise GalleryError(f"{NAMES_PATH} is empty; run: python scripts/gallery.py count")
    by_key = {(r["type"], r["key"]): r for r in names}
    glued = glue_map(names)
    population, tally = _population(_ledger_read(args))
    wanted: dict[tuple[str, str], dict[str, Any]] = {}

    def want(entity_type: str, key: str, why: str, query: str | None = None, **extra: Any) -> None:
        row = by_key.get((entity_type, key)) or {"type": entity_type, "key": key, "display": query or key,
                                                 "mentions": {"vlm": 0, "claims": 0}}
        entry = wanted.setdefault((entity_type, key), {
            "type": entity_type, "key": key, "query": query or query_for(row, glued), "why": []})
        entry["why"].append(why)
        entry.update(extra)

    for key, name in PROBE_FIGURES.items():
        want("PERSON", key, "probe_figure", name)
    for img in population:
        want("PERSON", img["figure_key"], f"figure_{img['figure_source']}")
        for person in img["claim_people"]:  # what a VLM name is scored against
            if key := name_key(person, "PERSON"):
                want("PERSON", key, "claim_person_on_calibration_image")
    for key in calibration_names(population):
        want("PERSON", key, "vlm_name_on_calibration_image")
    scoped = [r for r in names if in_scope(r, args.scope)]
    strata = None
    if args.all:
        for r in sorted(scoped, key=lambda r: (-scope_mentions(r, args.scope), r["type"], r["key"])):
            want(r["type"], r["key"], "build_scope")
    else:
        picks, strata = stratified_sample(scoped, args.random)
        for r in picks:
            want(r["type"], r["key"], "random", stratum=r["stratum"])
    done = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    known = latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"])

    todo = [w for k, w in wanted.items() if link_due(done.get(k), w["query"], w["type"], known)]
    whys = Counter(why for w in wanted.values() for why in set(w["why"]))
    searches = sum(1 + (dotted_initials(w["query"], w["type"]) is not None) for w in todo)
    log(f"{'build' if args.all else 'sample'} (scope {','.join(args.scope)}): {len(wanted):,} names "
        f"({len(population):,} calibration images {dict(tally)}; by reason {dict(whys)}); "
        f"{len(todo):,} to link: about {searches:,} searches and "
        f"{math.ceil(len(todo) * 2 / BATCH):,}+ entity calls on www.wikidata.org, at most one "
        f"per {WIKIMEDIA_DELAY:g} s")
    if args.dry_run:
        return 0
    if not args.all:  # the sample's definition, which sample-report reads back
        write_json(SAMPLE_PATH, {"generated_at": utcnow(), "ledger_images": tally,
                                 "population_images": len(population), "strata": strata,
                                 "names": sorted(wanted.values(), key=lambda w: (w["type"], w["key"]))})
    if not todo:
        return 0
    links, wiki = JsonlLedger(LINKS_PATH), JsonlLedger(WIKIDATA_PATH)
    prov = JsonlLedger(PROVENANCE_PATH)
    links.open()
    wiki.open()
    prov.open()
    classes: dict[str, list[str]] = {}
    net = live_net(ProvenanceLog(prov, BUILD_PASS_ID if args.all else PASS_ID))
    status: Counter = Counter()
    try:
        # checkpointed: each chunk's items and links are written before the next
        # starts, so a stop loses at most one chunk, and a rerun skips the rest
        for start in range(0, len(todo), LINK_CHUNK):
            rows, entities = link_names(net, todo[start:start + LINK_CHUNK], known=known, classes=classes)
            wiki.append_many([{"pass": net.log.pass_id, "at": utcnow(), **e}
                              for q, e in entities.items() if q not in known])
            known.update(entities)
            links.append_many(rows)
            status.update(r["status"] for r in rows)
            log(f"  linked {min(start + LINK_CHUNK, len(todo)):,}/{len(todo):,} "
                f"({net.requests['total']:,} requests): {dict(status)}")
    finally:
        for ledger in (links, wiki, prov):
            ledger.close()
    log(json.dumps(dict(status), indent=1))
    return 0


#: Outcomes a corrected rendition rule may reverse. Only these: a reference
#: refused by the registry, or as a corpus image, stays refused.
REGATE_OUTCOMES = ("rendition_mismatch",)
REGATE_RULE = ("rendition_problem: an unscaled original must be the file by Commons's SHA-1; a "
               "scaled thumbnail must have the requested width and the height within rounding")


def regate(entities: Sequence[dict[str, Any]], *, corpus: frozenset[str] | None = None,
           media: Path = MEDIA_DIR) -> tuple[list[dict[str, Any]], Counter]:
    """Entities whose rendition-refused references are judged again, by the
    current rule, from their quarantined bytes. No request is made and nothing
    is deleted: the quarantined copy stays, a reference that now passes every
    gate is stored, and each judged reference records what it was and why."""
    from scripts.recover import Resp

    corpus = corpus_hashes() if corpus is None else corpus
    changed, tally = [], Counter()
    for e in entities:
        refs, touched = [], False
        for ref in e["refs"]:
            if ref.get("outcome") not in REGATE_OUTCOMES or not ref.get("path"):
                refs.append(ref)
                continue
            body = (image_text.PROJECT_ROOT / ref["path"]).read_bytes()
            gated = gate_reference(Resp(ref["rendition"]["url"], 200, {}, body, None, 0.0), ref, corpus)
            tally[f"{ref['outcome']}->{gated['outcome']}"] += 1
            new = {**ref, "outcome": gated["outcome"],
                   "regated": {"from": ref["outcome"], "at": utcnow(), "rule": REGATE_RULE}}
            if gated["outcome"] == "usable":
                new["path"] = store(body, gated["examined"], media).relative_to(
                    image_text.PROJECT_ROOT).as_posix()
            refs.append(new)
            touched = True
        if touched:
            changed.append({**e, "refs": refs})
    return changed, tally


def sample_entities(seed: int = SEED) -> list[dict[str, Any]]:
    """The entities this pass fetches: every one the calibration needs, and
    SAMPLE_FETCH_RANDOM linked random-sample entities per type (reference yield)."""
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    links = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    wiki = latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"])
    chosen: dict[str, dict[str, Any]] = {}
    random_pool: dict[str, list[tuple[str, dict[str, Any]]]] = {t: [] for t in TYPES}
    for w in sample["names"]:
        link = links.get((w["type"], w["key"]))
        if not link or link["status"] not in ("linked", "ambiguous") or link["qid"] not in wiki:
            continue
        entry = chosen.get(link["qid"]) or {**wiki[link["qid"]], "type": w["type"], "names": set()}
        entry["names"].add(w["key"])
        if any(why != "random" for why in w["why"]):
            chosen[link["qid"]] = entry
        else:
            random_pool[w["type"]].append((link["qid"], entry))
    for entity_type, pool in random_pool.items():
        ordered = sorted({q: e for q, e in pool}.items(), key=lambda qe: image_text._priority(seed, qe[0]))
        for qid, entry in [qe for qe in ordered if qe[0] not in chosen][:SAMPLE_FETCH_RANDOM]:
            chosen[qid] = entry
    return list(chosen.values())


def build_entities(scope: Sequence[str] = BUILD_SCOPE) -> list[dict[str, Any]]:
    """Every item a name in ``scope`` (or a probe figure) is linked to, the
    most-mentioned first: the full build's work list. A name that links to an
    item already fetched adds nothing to fetch."""
    links = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    wiki = latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"])
    chosen: dict[str, dict[str, Any]] = {}
    weight: Counter = Counter()

    def add(entity_type: str, key: str, mentions: int) -> None:
        link = links.get((entity_type, key))
        if not link or link["status"] not in ("linked", "ambiguous") or link["qid"] not in wiki:
            return
        entry = chosen.setdefault(link["qid"], {**wiki[link["qid"]], "type": entity_type, "names": set()})
        entry["names"].add(key)
        weight[link["qid"]] += mentions

    for r in read_jsonl(NAMES_PATH):
        if in_scope(r, scope):
            add(r["type"], r["key"], scope_mentions(r, scope))
    for key in PROBE_FIGURES:
        add("PERSON", key, 0)
    return sorted(chosen.values(), key=lambda e: (-weight[e["qid"]], e["qid"]))


def cmd_fetch(args) -> int:
    if args.regate:
        stored = list(latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"]).values())
        changed, tally = regate(stored)
        ledger = JsonlLedger(ENTITIES_PATH)
        ledger.open()
        try:
            ledger.append_many(changed)
        finally:
            ledger.close()
        log(f"regated {sum(tally.values())} references of {len(changed)} entities, offline: {dict(tally)}")
        return 0
    if free_gb(GALLERY) < 5:
        raise GalleryError(f"refusing: {free_gb(GALLERY):.1f} GiB free under {GALLERY}")
    entities = build_entities(args.scope) if args.all else sample_entities()
    done = latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"])
    todo = [e for e in entities if e["qid"] not in done]
    log(f"fetch: {len(entities):,} {'build' if args.all else 'sample'} entities "
        f"(scope {','.join(args.scope)}), {len(todo):,} to fetch; budget {args.budget_gb} GiB; "
        f"User-Agent: {user_agent()}")
    if not todo:
        return 0
    out, prov = JsonlLedger(ENTITIES_PATH), JsonlLedger(PROVENANCE_PATH)
    out.open()
    prov.open()
    net = live_net(ProvenanceLog(prov, BUILD_PASS_ID if args.all else PASS_ID), int(args.budget_gb * 1024 ** 3))
    outcomes: Counter = Counter()
    written = left = 0
    try:
        # checkpointed like link: each chunk's entities are written before the next
        for start in range(0, len(todo), FETCH_CHUNK):
            chunk = todo[start:start + FETCH_CHUNK]
            try:
                rows = fetch_references(net, chunk, media=MEDIA_DIR, rejected=REJECTED_DIR)
            except recover.BudgetExceeded as exc:
                net.stopped = f"budget: {exc}"
                rows = []
            out.append_many(rows)
            written += len(rows)
            left += 0 if net.stopped else len(chunk) - len(rows)
            outcomes.update(r.get("outcome") for row in rows for r in row["refs"])
            log(f"  fetched {written:,}/{len(todo):,} entities ({net.requests['total']:,} requests, "
                f"{net.requests['bytes'] / 1024 ** 3:.2f} GiB): usable references "
                f"{outcomes['usable']:,}; {left:,} left for the next run (no answer, or not now)")
            if net.stopped:
                log(f"  stopped ({net.stopped}); rerun to continue from here")
                break
    finally:
        out.close()
        prov.close()
    log(f"{written:,} entities written, {left:,} left for the next run; reference outcomes {dict(outcomes)}")
    return 0


def cmd_embed(args) -> int:
    backend = make_backend(args.backend)
    cid = face_config_id(backend)
    config_path = FACES_DIR / cid / "config.json"
    if not config_path.is_file():
        write_json(config_path, {"config_id": cid, **backend.config()})
    entities = list(latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"]).values())
    refs = {r["sha256"]: image_text.PROJECT_ROOT / r["path"] for e in entities if e["type"] == "PERSON"
            for r in e["refs"] if r.get("outcome") == "usable"}
    ref_faces = embed_images(backend, [(sha, (lambda p=p: p.read_bytes())) for sha, p in sorted(refs.items())],
                             FACES_DIR / cid / "refs.jsonl", MIN_FACE["reference"])
    pin_path = FACES_DIR / cid / "consistency_pin.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.is_file() else None
    if pin is not None and args.pin_consistency:
        raise GalleryError(f"{pin_path} already pins {pin['threshold']}; delete it to derive again")
    decisions = all_reference_decisions()
    flags = reference_flags(known={reference_id(e["qid"], r) for e in entities for r in e["refs"] if r.get("sha256")})
    rows, summary = assemble(entities, ref_faces, None if pin is None else pin["threshold"], decisions, flags)
    summary["side_views"] = side_view_outcomes(rows)
    summary["reference_review"] = reference_review(rows, reference_decisions())
    summary["p18_review"] = p18_review(rows, reference_decisions(P18_DECISIONS),
                                       disagreeing_p18_pairs(entities, ref_faces, summary["consistency_threshold"]))
    summary["reference_flags"] = dict(Counter(f for r in rows for ref in r["refs"] for f in ref.get("flags", [])))
    if args.pin_consistency:
        pin = {"threshold": summary["consistency_threshold"], "pinned_at": utcnow(),
               "entities_with_faces": summary["entities_with_faces"],
               "quantile": CONSISTENCY_QUANTILE}
        write_json(pin_path, pin)
        summary["pinned"] = True
    summary["pin"] = pin
    write_jsonl(FACES_DIR / cid / "gallery.jsonl", rows)
    write_json(FACES_DIR / cid / "gallery_summary.json", summary)
    population, _tally = _population(_ledger_read(args))
    index = images.load_index("factify2")
    by_sha: dict[str, str] = {}
    for path, row in index.items():
        if row.get("sha256"):
            by_sha.setdefault(row["sha256"], path)
    embed_images(backend, [(img["sha256"], (lambda p=by_sha[img["sha256"]]: image_text.read_bytes(p)))
                           for img in population if img["sha256"] in by_sha],
                 QUERY_FACES_DIR / cid / "factify2.jsonl", MIN_FACE["query"])
    usable = Counter(r["usable_refs"] for r in rows if r["type"] == "PERSON")
    log(f"[{args.backend}] config {cid}: {len(ref_faces):,} reference images, "
        f"{len(population):,} calibration images; usable references per person {dict(sorted(usable.items()))}; "
        f"consistency threshold {summary['consistency_threshold']} "
        f"({'pinned' if summary['pinned'] else 'derived'}; derived now {summary['derived_now']}); "
        f"{summary['people_with_usable_faces']:,} of {summary['people']:,} people have a usable "
        "reference face, and verify can only answer 'no usable reference' for the rest")
    if summary["reference_review"]["judged"]:
        log(f"  reviewed: {summary['reference_review']['statement']}")
    log(f"  two P18 images: {summary['p18_review']['statement']}")
    if summary["reference_flags"]:
        log(f"  flagged references: {summary['reference_flags']}")
    if summary["side_views"]["fetched"]:
        log(f"  side views: {summary['side_views']['statement']}")
        if SIDE_VIEWS_REPORT.is_file():  # the census's published aggregate gains what the check decided
            published = json.loads(SIDE_VIEWS_REPORT.read_text(encoding="utf-8"))
            write_json(SIDE_VIEWS_REPORT, {**published, "fetched": summary["side_views"],
                                           "reviewed": summary["reference_review"]})
    return 0


#: The published calibration: one entry per backend, aggregates only.
CALIBRATION_REPORT = REPORTS / "gallery_calibration.json"
_QID = re.compile(r"Q\d+")


def _without_local(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_local(v) for k, v in value.items() if k != "local"}
    if isinstance(value, list):
        return [_without_local(v) for v in value]
    return value


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    """The calibration report with no name and no QID: figures by rank only,
    and no ``local`` part (the trials and names a person reads, kept in the
    local report). Refuses if an identifier is left anywhere in it."""
    out = _without_local(json.loads(json.dumps(report)))
    for part in (out, out.get("adjudicated") or {}):
        figures = part.get("figures")
        if figures:
            figures["table"] = [{"rank": i, **{k: v for k, v in r.items() if k not in ("figure_qid", "label")}}
                                for i, r in enumerate(figures["table"], 1)]

    def check(value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
        elif isinstance(value, str) and _QID.fullmatch(value):
            raise GalleryError(f"the published calibration would carry an identifier ({value})")

    check(out)
    return out


def consistency_floor(cid: str) -> float:
    """The gallery's consistency threshold (pinned): a score below it sits
    where different people's faces score. The review sheet's floor, and where
    a face a person recognises becomes a recognition failure."""
    summary = json.loads((FACES_DIR / cid / "gallery_summary.json").read_text(encoding="utf-8"))
    return summary["consistency_threshold"]


def cmd_calibrate(args) -> int:
    backend = make_backend(args.backend)
    cid = face_config_id(backend)
    gallery = Gallery(cid)
    run_dir = _run_dir(args)
    read = read_ledger(run_dir / "factify2.jsonl", "factify2")
    population, tally = _population(read)
    links = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    query_faces = latest(read_jsonl(QUERY_FACES_DIR / cid / "factify2.jsonl"), lambda r: r["sha256"])
    trials, more = build_trials(population, links, query_faces, gallery)
    worklist = worklist_sizes(run_dir).get("factify2")
    ledger = {
        "run": run_dir.name, "images": len(read.images), "worklist": worklist,
        "share_done": image_text._ratio(len(read.images), worklist, 4) if worklist else None,
        "complete": bool(worklist) and len(read.images) >= worklist,
        "note": ("the full calibration needs the finished run; a partial ledger gives a partial "
                 "population and never sets the threshold")}
    report = calibrate(trials, tally + more, gallery, query_faces, args.backend, ledger)
    report["factify2_ledger"] = ledger
    if NAMES_PATH.is_file():
        names = [r for r in read_jsonl(NAMES_PATH) if r["type"] == "PERSON" and in_scope(r)]
        wiki = latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"])
        report["alias_gaps"], gaps = alias_gaps(names, gallery, wiki)
        claim_keys = {img["sha256"]: {name_key(p, "PERSON") for p in img["claim_people"]} - {None}
                      for img in population}
        report["alias_gaps"]["labels"] = alias_gap_labels(trials, claim_keys, gaps)
        report["alias_gaps"]["resolution_rules"] = resolution_rules(names, gallery)
        report["alias_gaps"]["no_one"] = no_one_names(names, gallery, links, wiki, gaps)
    else:
        report["alias_gaps"] = {"status": f"not measured: no name table at {NAMES_PATH} (run count)"}
    report["adjudicated"] = None
    path = decisions_path(args.backend)
    decisions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    decided = {k: v for k, v in decisions.items() if v is not None}
    if decided:
        ids = {trial_id(x) for x in trials}
        relabelled = adjudicated(trials, decisions)
        before = {trial_id(x): x["label"] for x in trials}
        after = {trial_id(x): x["label"] for x in relabelled}
        changes = Counter("taken_out" if k not in after else "unchanged" if before[k] == after[k]
                          else f"{before[k]}_to_{after[k]}" for k in decided if k in before)
        report["adjudicated"] = {
            "decisions": dict(Counter(decided.values())), "by": "a person, by eye, on the review sheet",
            "applied": sum(1 for k in decided if k in ids),
            "not_among_these_trials": sum(1 for k in decided if k not in ids),
            "left_undecided_in_file": len(decisions) - len(decided),
            "relabelled": dict(sorted(changes.items())),
            "note": ("a trial without a decision keeps its label; relabelled says how many decisions "
                     "changed a label, and in which direction, and how many confirmed it"),
            "review_findings": review_findings(trials, relabelled, consistency_floor(cid)),
            **{k: v for k, v in calibrate(relabelled, tally + more, gallery, query_faces,
                                          args.backend, ledger).items()
               if k in ("trials", "vlm_precision", "bar", "threshold", "at_threshold", "fmr_over_tar",
                        "cost_in_true_accepts", "bootstrap_95", "cross_validated", "figure_stratified",
                        "look_alike_finding")}}
    write_json(CALIBRATION_DIR / f"{args.backend}.json", report)
    write_jsonl(CALIBRATION_DIR / f"{args.backend}_trials.jsonl", trials)
    published = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8")) if CALIBRATION_REPORT.is_file() else {}
    published[args.backend] = {"generated_at": utcnow(), **public_report(report)}
    write_json(CALIBRATION_REPORT, published)
    t = report["threshold"]
    at = report["at_threshold"]
    strat = report["figure_stratified"]
    log(f"[{args.backend}] {report['trials']['headline']} trials; VLM precision (QID) "
        f"{report['vlm_precision']['qid']} (scored against the figure alone, as first measured: "
        f"{report['vlm_precision']['single_figure_rule']}); bar FMR/TAR <= {report['bar']['max_fmr_over_tar']}")
    log(f"  threshold {t['chosen']} ({t['chosen_by']})")
    log(f"  candidate {t['candidate']} ({t['candidate_by']}): TAR {at['tar']} FMR {at['fmr']} "
        f"precision {at['precision']} (lower 95% {at['precision_lower_95']})")
    adj = report["adjudicated"]
    if adj:
        ta, aa = adj["threshold"], adj["at_threshold"]
        log(f"  adjudicated ({adj['applied']} decisions; {adj['relabelled']}): VLM precision "
            f"{adj['vlm_precision']['qid']}; candidate {ta['candidate']} ({ta['candidate_by']}): TAR "
            f"{aa['tar']} FMR {aa['fmr']} precision {aa['precision']} (lower 95% {aa['precision_lower_95']}); "
            f"TAR on names the gallery can score {adj['cost_in_true_accepts']['tar_scorable']}")
        log(f"  review: {adj['review_findings']['statement']}")
    log(f"  coverage: {report['coverage']['statement']}")
    aliases = report["alias_gaps"]
    log(f"  aliases: {aliases.get('statement') or aliases['status']}")
    if "labels" in aliases:
        log(f"    {aliases['labels']['statement']}")
        log(f"  COST BY DESIGN: {aliases['resolution_rules']['cost_by_design']['statement']}")
        log(f"    {aliases['resolution_rules']['statement']}")
        log(f"    {aliases['no_one']['statement']}")
    log(f"  figures: {report['figures']['concentration']['statement']}")
    for r in report["figures"]["table"][:5]:
        log(f"    {r['label'] or r['figure_qid']}: {r['trials']} trials, {r['right']} right, "
            f"{r['wrong']} wrong; accepted {r['accepted_right']} right, {r['accepted_wrong']} wrong")
    log(f"  figure-stratified at {strat.get('threshold')}: precision {strat.get('precision')} "
        f"(lower 95% {strat.get('precision_lower_95')}), TAR {strat.get('tar')}, FMR {strat.get('fmr')}; "
        f"threshold by the figure bound {strat.get('threshold_by_figure_bound')}")
    if report["look_alike_finding"]:
        log(f"  look-alikes: {report['look_alike_finding']['finding']}")
    log(f"wrote {CALIBRATION_DIR / (args.backend + '.json')} (local) and {CALIBRATION_REPORT} (aggregates)")
    return 0


def cmd_verify(args) -> int:
    backend = make_backend(args.backend)
    gallery = Gallery(face_config_id(backend))
    body = Path(args.image).read_bytes()
    faces = backend.faces(load_bgr(body), MIN_FACE["query"])
    path = CALIBRATION_DIR / f"{args.backend}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    out = gallery.verify(faces, args.name, certified_threshold(report, gallery.cid), args.type)
    if args.threshold is not None:  # a comparison a person asked for, never the gallery's decision
        out["given_threshold"] = {
            "threshold": args.threshold,
            "score_at_or_above": None if out["score"] is None else out["score"] >= args.threshold,
            "note": "given on the command line, not certified by calibration: a comparison, not a decision"}
    log(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------
# the sample's report, and the review sheet a person checks
# --------------------------------------------------------------------------


def alias_families(scope: Sequence[dict[str, Any]], wiki: dict[str, dict[str, Any]],
                   qids: Iterable[str], entity_type: str) -> dict[str, int]:
    """qid -> how many of the build scope's name keys of that type ARE one of
    the item's label or aliases: the names that collapse into it."""
    by_nospace: dict[str, set[str]] = {}
    for r in scope:
        if r["type"] == entity_type:
            by_nospace.setdefault(r["key"].replace(" ", ""), set()).add(r["key"])
    out = {}
    for qid in qids:
        summary = wiki.get(qid) or {}
        texts = [summary.get("label") or "", *summary.get("aliases", [])]
        keys = set()
        for t in texts:
            folded = fold(t)
            if folded:
                keys |= by_nospace.get(folded.replace(" ", ""), set())
        out[qid] = max(1, len(keys))
    return out


def linking_estimate(sample: dict[str, Any], links: dict[tuple[str, str], dict[str, Any]],
                     wiki: dict[str, dict[str, Any]], scope: Sequence[dict[str, Any]],
                     seed: int = SEED) -> dict[str, Any]:
    """From the stratified random names: the share that link, and how many
    distinct entities the build scope's names come to -- each linked name
    weighted by 1 / (names that collapse into its entity), and by its stratum."""
    import numpy as np

    strata = sample["strata"]
    drawn = [w for w in sample["names"] if "random" in w["why"]]
    rows = []
    for w in drawn:
        link = links.get((w["type"], w["key"])) or {}
        linked = link.get("status") in ("linked", "ambiguous")
        rows.append({"stratum": w["stratum"], "type": w["type"], "linked": linked,
                     "status": link.get("status"), "qid": link.get("qid") if linked else None})
    families = {}
    for t in TYPES:
        families.update(alias_families(scope, wiki, {r["qid"] for r in rows if r["type"] == t and r["qid"]}, t))

    def estimate(sampled: Sequence[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in TYPES:
            linked = entities = 0.0
            for s, info in strata.items():
                if not s.startswith(f"{t}|"):
                    continue
                members = [r for r in sampled if r["stratum"] == s]
                if not members:
                    continue
                weight = info["size"] / len(members)
                linked += weight * sum(r["linked"] for r in members)
                entities += weight * sum(1 / families[r["qid"]] for r in members if r["qid"])
            out[f"{t}_names_linked"] = linked
            out[f"{t}_entities"] = entities
        return out

    point = estimate(rows)
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_stratum.setdefault(r["stratum"], []).append(r)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(BOOTSTRAP):
        resample = [m[i] for m in by_stratum.values() for i in rng.integers(0, len(m), len(m))]
        for k, v in estimate(resample).items():
            draws[k].append(v)
    names_in_scope = Counter(r["type"] for r in scope)
    return {
        "method": ("stratified random names (type x source x frequency band), each weighted by its "
                   "stratum; an entity's share of a linked name is 1 / the build-scope names that "
                   "are one of its label or aliases; 95% intervals by stratified bootstrap"),
        "drawn": len(rows), "status": dict(Counter(r["status"] for r in rows)),
        "by_stratum": {s: {"size": strata[s]["size"], "drawn": len(m),
                           "linked": sum(r["linked"] for r in m)} for s, m in sorted(by_stratum.items())},
        **{t: {"names_in_scope": names_in_scope[t],
               "names_linked": round(point[f"{t}_names_linked"]),
               "link_rate": image_text._ratio(point[f"{t}_names_linked"], names_in_scope[t], 4),
               "distinct_entities": round(point[f"{t}_entities"]),
               "distinct_entities_95": [round(float(np.quantile(draws[f"{t}_entities"], q)))
                                        for q in (0.025, 0.975)]}
           for t in TYPES}}


def reference_yield(entities: Sequence[dict[str, Any]], assembled: dict[str, list[dict[str, Any]]]
                    ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in TYPES:
        mine = [e for e in entities if e["type"] == t]
        if not mine:
            continue
        refs = [r for e in mine for r in e["refs"]]
        usable = [r for r in refs if r.get("outcome") == "usable"]
        out[t] = {"entities": len(mine),
                  "candidates_per_entity": round(len(refs) / len(mine), 2),
                  "outcomes": dict(Counter(r.get("outcome") for r in refs).most_common()),
                  "fetched_usable_per_entity": dict(sorted(Counter(
                      sum(r.get("outcome") == "usable" for r in e["refs"]) for e in mine).items())),
                  "bytes_per_usable_image": (round(statistics.fmean(r["bytes"] for r in usable))
                                             if usable else None),
                  "licences": dict(Counter((r.get("licence") or {}).get("short_name")
                                           for r in usable).most_common()),
                  "restrictions": dict(Counter((r.get("licence") or {}).get("restrictions") or "none"
                                               for r in usable).most_common()),
                  "attribution_required": dict(Counter((r.get("licence") or {}).get("attribution_required")
                                                       for r in usable).most_common())}
    for backend, rows in assembled.items():
        people = [r for r in rows if r["type"] == "PERSON"]
        out.setdefault("faces", {})[backend] = {
            "decisions": dict(Counter(ref["decision"] for r in people for ref in r["refs"]).most_common()),
            "usable_references_per_person": dict(sorted(Counter(r["usable_refs"] for r in people).items())),
            "people_with_none": sum(1 for r in people if not r["usable_refs"])}
    return out


def request_costs(provenance: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_host = Counter(r["host"] for r in provenance if r.get("outcome") != "redirect")
    by_stage = Counter(r["stage"] for r in provenance)
    return {"requests_by_host": dict(by_host.most_common()), "requests_by_stage": dict(by_stage.most_common()),
            "outcomes": dict(Counter(f"{r['stage']}:{r['outcome']}" for r in provenance).most_common()),
            "bytes": sum(r.get("bytes") or 0 for r in provenance)}


#: Which requests each part of the build makes, by provenance stage.
STAGES = {"link": ("search", "entities", "type_check"), "commons": ("depicts", "category", "imageinfo"),
          "media": ("download",)}


def seconds_per_request(rows: Sequence[dict[str, Any]]) -> float | None:
    """At WikimediaWorker's pace, measured on the sample's own answers: a
    request start every WIKIMEDIA_DELAY, or the answer's time plus the policy's
    pause when an API answer took over SLOW_API_S."""
    times = []
    for r in rows:
        elapsed = r.get("elapsed_s") or 0.0
        api = WIKIMEDIA_HOSTS.get(r.get("host")) == "action_api"
        times.append(max(WIKIMEDIA_DELAY, elapsed + (SLOW_API_PAUSE if api and elapsed > SLOW_API_S else 0.0)))
    return statistics.fmean(times) if times else None


def projection(estimate: dict[str, Any], costs: dict[str, Any], fetched: dict[str, Any],
               linked_now: int, provenance: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The full build, from what the sample cost: requests per linked name and
    per fetched entity, each at the pace WikimediaWorker keeps (its time per
    request measured on the sample's answers), and disk."""
    names = sum(estimate[t]["names_in_scope"] for t in TYPES)
    entities = {t: estimate[t]["distinct_entities"] for t in TYPES}
    stage = costs["requests_by_stage"]
    fetched_entities = sum(fetched[t]["entities"] for t in TYPES if t in fetched)
    total_entities = sum(entities.values())
    per_unit = {"link": names / max(1, linked_now), "commons": total_entities / max(1, fetched_entities),
                "media": total_entities / max(1, fetched_entities)}
    requests = {part: sum(stage.get(s, 0) for s in stages) * per_unit[part] for part, stages in STAGES.items()}
    pace = {part: seconds_per_request([r for r in provenance if r.get("stage") in stages
                                       and r.get("host") in WIKIMEDIA_HOSTS and r.get("outcome") != "redirect"])
            for part, stages in STAGES.items()}
    hours = {part: round(requests[part] * (pace[part] or WIKIMEDIA_DELAY) / 3600, 1) for part in STAGES}
    gb = sum(entities[t] * sum(k * v for k, v in fetched[t]["fetched_usable_per_entity"].items())
             / max(1, fetched[t]["entities"]) * (fetched[t]["bytes_per_usable_image"] or 0)
             for t in TYPES if t in fetched) / 1024 ** 3
    return {"names_to_link": names, "entities": entities,
            "requests": {part: round(v) for part, v in requests.items()},
            "seconds_per_request": {part: None if v is None else round(v, 2) for part, v in pace.items()},
            "hours": hours, "hours_total": round(sum(hours.values()), 1),
            "wikimedia_delay_s": WIKIMEDIA_DELAY, "gb": round(gb, 2),
            "note": ("the build scope's names as counted now; Factify2's VLM names grow as its run "
                     "finishes (gallery_counts.json, projection). The parts run one after another "
                     "through the one worker, so their hours add: an upper bound")}


def sample_report() -> dict[str, Any]:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    links = latest(read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"]))
    wiki = latest(read_jsonl(WIKIDATA_PATH), lambda r: r["qid"])
    # the sample pass's own rows: a full build that followed is not the sample
    entities = [e for e in latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"]).values()
                if e.get("pass") == PASS_ID]
    scope = [r for r in read_jsonl(NAMES_PATH) if in_scope(r)]
    sampled = [links[(w["type"], w["key"])] for w in sample["names"] if (w["type"], w["key"]) in links]
    estimate = linking_estimate(sample, links, wiki, scope)
    assembled, calibration = {}, {}
    for backend in FACE_BACKENDS:
        if (CALIBRATION_DIR / f"{backend}.json").is_file():
            calibration[backend] = json.loads((CALIBRATION_DIR / f"{backend}.json").read_text(encoding="utf-8"))
            cid = calibration[backend]["face_config"]
            assembled[backend] = list(read_jsonl(FACES_DIR / cid / "gallery.jsonl"))
    fetched = reference_yield(entities, assembled)
    provenance = [r for r in read_jsonl(PROVENANCE_PATH) if r.get("pass") == PASS_ID]
    costs = request_costs(provenance)

    def headline(c: dict[str, Any]) -> dict[str, Any]:
        keep = ("population", "trials", "vlm_precision", "bar", "threshold", "at_threshold", "fmr_over_tar",
                "cost_in_true_accepts", "bootstrap_95", "cross_validated", "strata", "look_alike_finding",
                "figures", "figure_stratified", "curve", "adjudicated")
        published = public_report(c)  # figures by rank: no name, no QID
        return {k: published[k] for k in keep if k in published}

    def at_tar(c: dict[str, Any], tar: float) -> dict[str, Any] | None:
        reach = [p for p in c["curve"] if p["tar"] is not None and p["tar"] >= tar]
        return None if not reach else min(reach, key=lambda p: p["fmr"] if p["fmr"] is not None else 1)

    bake_off = {b: {"face_config": c["face_config"], "threshold": c["threshold"],
                    "at_threshold": c["at_threshold"],
                    "lowest_fmr_at_tar": {str(t): at_tar(c, t) for t in (0.5, 0.7, 0.8, 0.9)}}
                for b, c in calibration.items()}
    return {
        "generated_at": utcnow(), "pass": PASS_ID,
        "what": ("the sample pass: the design validated on a sample before any full build. Aggregates "
                 "only: no name, no record id, no file name"),
        "sample": {"calibration_images": sample["population_images"],
                   "ledger_images": sample["ledger_images"],
                   "names": len(sample["names"]),
                   "by_reason": dict(Counter(why for w in sample["names"] for why in set(w["why"]))),
                   "link_status": dict(Counter(r["status"] for r in sampled))},
        "linking": estimate,
        "references": fetched,
        "requests": costs,
        "projection_full_build": projection(estimate, costs, fetched, len(sampled), provenance),
        "calibration": {b: headline(c) for b, c in calibration.items()},
        "why_calibrate_on_the_vlms_own_errors": {
            b: c["look_alike_finding"] for b, c in calibration.items() if c.get("look_alike_finding")},
        "bake_off": bake_off,
        "notes": [
            "Random strangers are not the VLM's errors: why_calibrate_on_the_vlms_own_errors compares the "
            "two and shows what a threshold set on random pairs would accept of the VLM's own names. That "
            "is why FMR is measured on the VLM's wrong names, never on random pairs.",
            "The pilot's 47.8% VLM precision is APPROXIMATE: its reference is a loose token pattern over "
            "the claim. It is also UNDERSTATED, like the first QID-based figure (0.5455): both scored a name "
            "against one figure per image, so a person the claims name beside it, correctly named, counted "
            "as wrong. calibration.<backend>.vlm_precision.qid scores a name against everyone the image's "
            "claims name; single_figure_rule beside it is the defective figure on the same trials.",
            "Calibration reads Factify2 TRAIN only, images every one of whose records is train; VERITE is "
            "never read for any threshold.",
            "This is a sample on the part of the Factify2 run done so far: the full calibration runs on "
            "the server once the run finishes.",
            "Organisations have entries and references only: no visual match is claimed for them.",
            "Every reference's licence and attribution are recorded as Commons states them, in the local "
            "entity ledger; nothing here is redistributable media."],
    }


def cmd_sample_report(args) -> int:
    report = sample_report()
    write_json(SAMPLE_REPORT, report)
    log(json.dumps({k: report[k] for k in ("sample", "linking", "projection_full_build")}, indent=1))
    for b, c in report["calibration"].items():
        t, at = c["threshold"], c["at_threshold"]
        log(f"[{b}] threshold {t['chosen']} ({t['chosen_by']}): TAR {at['tar']} FMR {at['fmr']} "
            f"precision {at['precision']} (lower 95% {at['precision_lower_95']}); "
            f"bar FMR/TAR <= {c['bar']['max_fmr_over_tar']}, here {c['fmr_over_tar']}")
    log(f"wrote {SAMPLE_REPORT}")
    return 0


#: The colours a sheet draws a face's five landmarks in: eyes, nose, mouth corners.
#: What every sheet says about them.
SCORED_FACE_NOTE = ("Every face crop has the face the gallery scored boxed in red, with its eyes (green), nose "
                    "(yellow) and mouth corners (blue): a box or points that span two people, or sit on a "
                    "hand, mean the detection is wrong, whoever the face is.")
LANDMARK_COLOURS = ((0, 200, 0), (0, 200, 0), (255, 215, 0), (0, 120, 255), (0, 120, 255))


def _crop_image(body: bytes, face: dict[str, Any], side: int = 160):
    """A face crop, a little wider than its box, with the box (red) and the
    five landmarks drawn. The margin shows neighbours, and one detection can
    span two people with its landmarks on a hand (2026-10-23): only the drawn
    box and points show a reviewer which face was scored."""
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(io.BytesIO(body)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    x, y, w, h = face["box"]
    m = 0.25 * max(w, h)
    left, top = max(0, int(x - m)), max(0, int(y - m))
    right, bottom = min(img.width, int(x + w + m) + 1), min(img.height, int(y + h + m) + 1)
    crop = img.crop((left, top, right, bottom))
    crop.thumbnail((side, side))
    sx, sy = crop.width / (right - left), crop.height / (bottom - top)
    draw = ImageDraw.Draw(crop)
    draw.rectangle(((x - left) * sx, (y - top) * sy, (x + w - left) * sx, (y + h - top) * sy),
                   outline=(255, 0, 0), width=2)
    for (px, py), colour in zip(face.get("landmarks") or [], LANDMARK_COLOURS):
        cx, cy = (px - left) * sx, (py - top) * sy
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=colour)
    return crop


def _crop(body: bytes, face: dict[str, Any], side: int = 160) -> str:
    """_crop_image as an inline JPEG."""
    import base64

    buf = io.BytesIO()
    _crop_image(body, face, side).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


#: Side of the review sheet's face crops, in pixels.
REVIEW_CROP = 224


def _whole(body: bytes, box: Sequence[float] | None, side: int = 480) -> str:
    """The whole image, small, the scored face boxed (if a face was scored):
    context for a reviewer."""
    import base64

    from PIL import Image, ImageDraw, ImageOps

    with Image.open(io.BytesIO(body)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = min(1.0, side / max(img.width, img.height))
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
        if box is not None:
            x, y, w, h = (v * scale for v in box)
            ImageDraw.Draw(small).rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


#: What a person may decide a reviewed face is.
DECISIONS = ("vlm", "claim", "neither", "unsure")


def merge_decision_slots(path: Path, ids: Sequence[str]) -> int:
    """Adds a null slot, in sheet order, for every trial not yet in the
    decisions file. A decision already written is never changed or dropped."""
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    new = [i for i in dict.fromkeys(ids) if i not in current]
    if new or not path.is_file():
        write_json(path, {**current, **{i: None for i in new}})
    return len(new)


def trial_id(trial: dict[str, Any]) -> str:
    return f"{trial['sha256'][:12]}:{trial['key']}"


def decisions_path(backend: str) -> Path:
    return REVIEW_DIR / f"{backend}_decisions.json"


def adjudicated(trials: Sequence[dict[str, Any]], decisions: dict[str, str | None]) -> list[dict[str, Any]]:
    """Trials relabelled by a PERSON's decisions: whom the face is. "vlm" makes
    the VLM's name right; "claim" makes it right only when the VLM named the
    claim's figure (then the two columns are one person), wrong otherwise --
    whoever else the claims name, the face is the figure; "neither" makes it
    wrong; "unsure" takes the trial out. A trial with no decision (absent, or
    null in the file) keeps its label, so a partial adjudication is usable."""
    unknown = sorted(set(decisions.values()) - set(DECISIONS) - {None})
    if unknown:
        raise GalleryError(f"decisions may only be {DECISIONS} or null, not {unknown}")
    out = []
    for x in trials:
        d = decisions.get(trial_id(x))
        if d == "unsure":
            continue
        if d is None or x["label"] == "unlinked":
            out.append(x)
        else:
            right = d == "vlm" or (d == "claim" and x["figure_qid"] in (x.get("qids") or []))
            out.append({**x, "label": "right" if right else "wrong", "adjudicated": d})
    return out


def review_findings(trials: Sequence[dict[str, Any]], relabelled: Sequence[dict[str, Any]],
                    floor: float) -> dict[str, Any]:
    """What a person's decisions found, kept apart because each means something
    different (``trials`` as the label rule labelled them, ``relabelled`` as
    adjudicated). A name a person confirmed right is one of:

    - recognised: the gallery scores it at or above ``floor``, its own
      consistency threshold;
    - a RECOGNITION FAILURE: it scores below the floor, where different
      people's faces score. The recognizer failing on a face a person
      recognises, not a label error: it lowers TAR, and it is the honest limit
      of the recognizer on these images;
    - COVERAGE: the gallery cannot score it (no usable reference, no face). It
      could not have checked the name: never a true accept, and not the
      recognizer's failure either.

    Label noise -- a decision that changed the rule's label -- is counted by
    where the name falls. Wrong to right at or above the floor is a match the
    labels had called false; below it, a recognition failure they had hidden.

    A recognition failure's causes are kept apart, because their remedies
    differ (face_geometry):

    - pose: the face is side-on (SIDE_ON). References in more poses can help:
      the gallery recognises some side-on faces;
    - size: eyes to mouth, the face is smaller than every face the gallery
      recognised. A floor no reference lifts: under MIN_EYE_MOUTH a face is
      never scored at all, and a confirmed name there is coverage
      (no_usable_face).

    A face can have both, or neither."""
    before = {trial_id(x): x["label"] for x in trials}
    decided = [x for x in relabelled if x.get("adjudicated")]

    def where(x: dict[str, Any]) -> str:
        if not x["scorable"]:
            return "unscorable"
        return "at_or_above_floor" if x["score"] >= floor else "below_floor"

    def faces(xs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        def median(field_: str) -> float | None:
            values = [abs(x[field_]) for x in xs if x.get(field_) is not None]
            return round(statistics.median(values), 2 if field_ == "nose_offset" else 1) if values else None

        return None if not xs else {
            "median_face_side": median("face_side"), "median_eye_span": median("eye_span"),
            "median_eye_mouth": median("eye_mouth"), "median_nose_offset": median("nose_offset"),
            "side_on": sum(1 for x in xs if side_on(x))}

    right = [x for x in decided if x["label"] == "right"]
    wrong = [x for x in decided if x["label"] != "right"]
    recognised = [x for x in right if where(x) == "at_or_above_floor"]
    failures = sorted((x for x in right if where(x) == "below_floor"), key=lambda x: -x["score"])
    uncovered = [x for x in right if where(x) == "unscorable"]
    smallest = min((x["eye_mouth"] for x in recognised if x.get("eye_mouth") is not None), default=None)

    def causes(x: dict[str, Any]) -> list[str]:
        small = smallest is not None and x.get("eye_mouth") is not None and x["eye_mouth"] < smallest
        return [c for c, on in (("pose", side_on(x)), ("size", small)) if on]

    cause_of = {trial_id(x): causes(x) for x in failures}
    by_cause = {"pose": sum("pose" in c for c in cause_of.values()),
                "size": sum("size" in c for c in cause_of.values()),
                "both": sum(len(c) == 2 for c in cause_of.values()),
                "neither": sum(not c for c in cause_of.values())}
    noise: dict[str, Counter] = {}
    for x in decided:
        was = before[trial_id(x)]
        if was != x["label"]:
            noise.setdefault(f"{was}_to_{x['label']}", Counter())[where(x)] += 1
    changed = "; ".join(f"{k.replace('_', ' ')}: {sum(c.values())} ({c['at_or_above_floor']} at or above the "
                        f"floor, {c['below_floor']} below it, {c['unscorable']} unscorable)"
                        for k, c in sorted(noise.items()))
    return {
        "what": ("every name a person decided, by where the gallery puts it against its consistency "
                 "threshold: a confirmed name below it is a recognition failure, one it cannot score is "
                 "coverage; label noise (decisions that changed a label) is counted apart"),
        "floor": round(floor, 4),
        "confirmed_right": {"names": len(right), "recognised": len(recognised),
                            "recognition_failures": len(failures),
                            "coverage": dict(Counter(x["unscorable"] for x in uncovered))},
        "confirmed_wrong": {"names": len(wrong), **dict(sorted(Counter(where(x) for x in wrong).items()))},
        "label_noise": {k: dict(sorted(c.items())) for k, c in sorted(noise.items())},
        "faces": {"recognised": faces(recognised), "recognition_failures": faces(failures)},
        "recognition_failures_by_cause": {
            **by_cause,
            "rules": {"pose": (f"side-on: the nose {SIDE_ON} eye spans or more from the eyes' midpoint, along "
                               "the eye line -- past the far eye"),
                      "size": ("eyes to mouth, smaller than every face the gallery recognised among the confirmed "
                               f"names (the smallest: {smallest} px)")},
            "remedies": {"pose": "references in more poses, a side view for a side view",
                         "size": (f"none in the gallery: under the {MIN_EYE_MOUTH:g} px floor verify answers "
                                  "no_usable_face, and only a larger image helps")}},
        "statement": (f"A person confirmed {len(right)} names right. The gallery scores {len(recognised)} at or "
                      f"above its floor of {floor:.3f}; {len(failures)} below it, among different people's "
                      "faces -- recognition failures, not label errors: they lower TAR, and they are the "
                      f"recognizer's limit on these images (pose: {by_cause['pose']}, size: {by_cause['size']}, "
                      f"both: {by_cause['both']}, neither: {by_cause['neither']}); {len(uncovered)} it cannot "
                      f"score -- coverage, never a true accept. {sum(sum(c.values()) for c in noise.values())} "
                      f"decisions changed a label (label noise){': ' + changed if changed else ''}"),
        "local": {"recognition_failures": [{"trial": trial_id(x), "score": round(x["score"], 4),
                                            "causes": cause_of[trial_id(x)], "faces_in_image": x["faces"],
                                            **{k: x.get(k) for k in ("face_side", "eye_span", "eye_mouth",
                                                                     "nose_offset")}}
                                           for x in failures],
                  "coverage": [{"trial": trial_id(x), "unscorable": x["unscorable"]} for x in uncovered]}}


def review_queue(trials: Sequence[dict[str, Any]], decided: Iterable[str], floor: float,
                 rejected: int) -> list[tuple[str, dict[str, Any]]]:
    """The trials a person looks at next, none already decided, in three parts.

    1. Every trial the gallery scores as plausibly the person (at or above
       ``floor``), WHATEVER its label, by falling score. An accepted name
       labelled right is exactly where an unseen false accept would sit: a
       bound that holds only because no one looked is not a bound.
    2. Names labelled right that the gallery rejects at any threshold above the
       floor (scoring below it, or unscorable): what they cost in true
       accepts. A seeded sample of ``rejected``.
    3. Names labelled wrong that a claim's text on the image carries although
       no claim names them by QID (a handle, a spelling the linker missed):
       the likely mislabels the label rule cannot see."""
    done = set(decided)
    todo = [x for x in trials if x["label"] != "unlinked" and trial_id(x) not in done]
    high = [x for x in todo if x["scorable"] and x["score"] >= floor]
    low = [x for x in todo if not (x["scorable"] and x["score"] >= floor)]
    rejected_right = sorted((x for x in low if x["label"] == "right"),
                            key=lambda x: image_text._priority(SEED, x["sha256"] + (x["key"] or "")))
    named = sorted((x for x in low if x["label"] == "wrong" and x.get("named_in_claim_text")),
                   key=lambda x: -(x["score"] if x["scorable"] else -1.0))
    return ([("plausible", x) for x in sorted(high, key=lambda x: -x["score"])]
            + [("rejected_right", x) for x in rejected_right[:rejected]]
            + [("named_in_claim_text", x) for x in named])


def cmd_review(args) -> int:
    """A sheet a PERSON reads: each trial still to decide (review_queue), its
    face beside the reference it matched, with the reference's Commons page,
    licence and attribution. Local only (it shows corpus images); the
    adjudication is a person's."""
    backend = make_backend(args.backend)
    gallery = Gallery(face_config_id(backend))
    report = json.loads((CALIBRATION_DIR / f"{args.backend}.json").read_text(encoding="utf-8"))
    t = report["threshold"]["chosen"]
    trials = list(read_jsonl(CALIBRATION_DIR / f"{args.backend}_trials.jsonl"))
    query_faces = latest(read_jsonl(QUERY_FACES_DIR / gallery.cid / "factify2.jsonl"), lambda r: r["sha256"])
    index = images.load_index("factify2")
    by_sha: dict[str, str] = {}
    for path, row in index.items():
        if row.get("sha256"):
            by_sha.setdefault(row["sha256"], path)
    # The floor is the gallery's own consistency threshold: a score above it is
    # plausibly the person, so it is looked at whether a threshold is set or not.
    floor = consistency_floor(gallery.cid) if t is None else min(t, consistency_floor(gallery.cid))
    decisions_file = decisions_path(args.backend)
    decisions = json.loads(decisions_file.read_text(encoding="utf-8")) if decisions_file.is_file() else {}
    decided = [k for k, v in decisions.items() if v is not None]
    queue = review_queue(trials, decided, floor, args.rejected)

    def reference_cell(qid: str, faces: Sequence[dict[str, Any]]) -> str:
        best = gallery.best(faces, qid) if gallery.refs.get(qid) else None
        if best is None:
            return "<td>(no usable reference)</td>"
        ref = best["reference"]
        lic, att = ref.get("licence") or {}, ref.get("attribution") or {}
        crop = _crop((image_text.PROJECT_ROOT / ref["path"]).read_bytes(), best["reference_face"])
        score = (f"face under the {MIN_EYE_MOUTH:g} px floor: no score" if too_small(faces[best["query_face"]])
                 else f"score {best['score']:.3f}")
        return (f"<td><img src='{crop}'><br>{html.escape((gallery.entities.get(qid) or {}).get('label') or qid)}"
                f"<br>{score}<br><a href='{html.escape(ref.get('page_url') or '')}'>"
                f"{html.escape(ref.get('title') or '')}</a><br>{html.escape(lic.get('short_name') or '')}; "
                f"{html.escape(att.get('artist') or '')}</td>")

    cells = []
    for n, (part, x) in enumerate(queue, 1):
        faces = (query_faces.get(x["sha256"]) or {}).get("faces") or []
        best = gallery.best(faces, x["qid"]) if x["scorable"] else None
        body = image_text.read_bytes(by_sha[x["sha256"]])
        trial = f"trial <code>{html.escape(trial_id(x))}</code><br>{x['faces']} face(s) in the image"
        if best is not None:
            scored = faces[best["query_face"]]
            face = (f"<img src='{_crop(body, scored, REVIEW_CROP)}'><br>{trial}<br><details><summary>whole "
                    f"image</summary><img class='whole' src='{_whole(body, scored['box'])}'></details>")
        else:
            face = (f"<img class='whole' src='{_whole(body, None)}'><br>{trial}<br>no face is scored for "
                    f"this name ({html.escape(str(x['unscorable']))}): decide whether the image shows the "
                    "person the VLM named")
        kind = {"plausible": (("ACCEPTED" if t is not None and x["score"] >= t else "PLAUSIBLE")
                              + f", labelled {x['label']}"),
                "rejected_right": (f"BELOW {floor:.3f}" if x["scorable"] else "UNSCORABLE") + ", labelled right",
                "named_in_claim_text": "LABELLED WRONG, though a claim's text on the image carries the name",
                }[part]
        # a name several people hold has no one reference: the sheet names them all, shows none
        named = (f"<td>ambiguous, {len(x['qids'])} people:<br>"
                 f"{html.escape('; '.join((gallery.entities.get(q) or {}).get('label') or q for q in x['qids']))}"
                 "</td>" if x.get("unscorable") == "ambiguous_name" else reference_cell(x["qid"], faces))
        cells.append(
            f"<tr><td><b>{n}</b></td><td>{face}</td>{named}"
            f"{reference_cell(x['figure_qid'], faces)}<td><b>{kind}</b><br>VLM said: "
            f"{html.escape(x['name'])}<br>{'score ' + format(x['score'], '.3f') if x['scorable'] else ''}</td></tr>")
    written = merge_decision_slots(decisions_file, [trial_id(x) for _part, x in queue])
    parts = Counter(part for part, _x in queue)
    shown = (f"threshold {t}" if t is not None else
             f"threshold unset (candidate {report['threshold'].get('candidate')}, not adopted)")
    page = ("<!doctype html><meta charset='utf-8'><title>Gallery review</title>"
            "<style>body{font:14px sans-serif;margin:16px}td{vertical-align:top;padding:6px;"
            f"border-bottom:1px solid #ccc}}img{{max-width:{REVIEW_CROP}px}}img.whole{{max-width:480px}}"
            "</style>"
            f"<h1>Gallery review: {args.backend}, {shown}, {len(queue)} faces</h1>"
            "<p>" + SCORED_FACE_NOTE + " Columns: the face in the Factify2 TRAIN image (open <i>whole "
            "image</i> for context: the face scored is boxed); the reference of the person the VLM named; "
            "the reference of the "
            "image's figure, one of the people its claims name. A name is labelled right when the image's "
            "claims name that person, which does not prove the image shows them, and wrong otherwise. "
            "For each row decide, by looking, whom the face is, and write it against the row's trial in "
            f"<code>{html.escape(str(decisions_file))}</code> (every trial on this sheet is already there, "
            "as <code>null</code>):</p><ul>"
            "<li><code>\"vlm\"</code>: the person the VLM named (second column);</li>"
            "<li><code>\"claim\"</code>: the image's figure (third column);</li>"
            "<li><code>\"neither\"</code>: someone else;</li>"
            "<li><code>\"unsure\"</code>: you cannot tell (the trial leaves the adjudicated figures).</li>"
            "</ul><p>When the VLM named the figure both columns show them: <code>\"vlm\"</code> if it is "
            "them, <code>\"neither\"</code> if not. A <code>null</code> keeps the label, so part of the "
            f"sheet is usable. No trial already decided is here ({len(decided)} are). First comes every "
            f"name the gallery scores at or above {floor:.3f}, its consistency threshold, whatever its "
            "label: an accepted name is where an unseen false accept would sit. Then names labelled right "
            "that score below it or cannot be scored, then names labelled wrong that a claim's text on "
            "the image carries. <code>calibrate</code> reports adjudicated figures beside the raw ones.</p>"
            f"<table>{''.join(cells)}</table>")
    out = REVIEW_DIR / f"{args.backend}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    log(f"{parts['plausible']} undecided names scoring at least {floor:.3f}, {parts['rejected_right']} labelled "
        f"right below it or unscorable, {parts['named_in_claim_text']} labelled wrong that a claim's text "
        f"carries ({len(decided)} trials decided already) -> {out}; {written} new trial slots in "
        f"{decisions_file}")
    return 0


# --------------------------------------------------------------------------
# status and the command line
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# side views: does Commons hold the people's other poses? (a measurement)
# --------------------------------------------------------------------------

#: The people a side-view census looks at: the most mentioned.
SIDE_VIEW_TOP = 50
#: Commons files sampled per person, beyond the gallery's own references.
SIDE_VIEW_PER_PERSON = 20
#: Files a depicts search, or a category listing, returns per person.
SIDE_VIEW_LISTED = 50
#: Coverage is worth fetching for when this share of the people have a side
#: view the gallery's consistency check would keep; below it, pose stays a
#: stated limitation.
SIDE_VIEW_ENOUGH = 0.5
SIDE_VIEWS_LOCAL = GALLERY_INTERIM / "side_views.json"
SIDE_VIEWS_REPORT = REPORTS / "gallery_side_views.json"
SIDE_VIEWS_PASS = "gallery-side-views-2026-10"


def people_by_mentions(names: Sequence[dict[str, Any]], gallery: Gallery) -> list[tuple[str, int]]:
    """(qid, mentions) of gallery people, the most mentioned first: the build
    scope's mentions of every name that resolves to that person alone."""
    weight: Counter = Counter()
    for r in names:
        qids = gallery.resolve_key(r["key"])
        if len(qids) == 1 and (gallery.entities.get(qids[0]) or {}).get("type") == "PERSON":
            weight[qids[0]] += scope_mentions(r)
    return sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))


def side_view_census(net: Net, backend: FaceBackend, gallery: Gallery, people: Sequence[tuple[str, int]],
                     ref_faces: dict[str, dict[str, Any]], floor: float, *,
                     per_person: int = SIDE_VIEW_PER_PERSON, corpus: frozenset[str] | None = None,
                     seed: int = SEED) -> tuple[dict[str, Any], dict[str, list[tuple[str, bool, Any]]]]:
    """For each of ``people`` (qid, mentions), what the gallery holds and what
    Commons offers, by pose. A MEASUREMENT: nothing is stored, nothing is
    added to the gallery.

    - In the gallery: each fetched reference's chosen face, side-on or not
      (SIDE_ON), usable or refused by the consistency check.
    - On Commons: up to ``per_person`` more files, drawn (seeded) from what
      Commons lists as depicting the person or in their category, through the
      same gates as a reference. A file counts when it shows one face, or one
      face the person's medoid resolves, of reference size. A side-on face
      AGREES when it scores at or above ``floor`` (the consistency threshold)
      against the person's medoid: the check would keep it. One that does not
      may be the person turned away, or someone else; only a person can say.

    The medoid is the one the consistency check measures against
    (consistency_check), as assemble would check the file -- not the medoid of
    the references the check kept. For some people those are different faces,
    and the first census, measured against the second, predicted 67 kept side
    views where the check kept 49 (2026-10-23).

    Returns the report (people's rows under ``local``) and, per qid, the
    side-on faces found: (title, agrees, embedding)."""
    import numpy as np

    from scripts import recover

    corpus = corpus_hashes() if corpus is None else corpus
    entities = {q: (gallery.entities.get(q) or {}) for q, _m in people}
    # (embedding, whether a group photo resolves against it: a medoid of two references or more)
    medoids: dict[str, tuple[Any, bool]] = {}
    for q, e in entities.items():
        _rows, vecs, _side = reference_picks(e, ref_faces)
        _checked, medoid = consistency_check(e.get("refs") or [], vecs, floor)
        if medoid is not None:
            medoids[q] = (vecs[medoid], len(vecs) >= 2)
    own_rows: dict[str, dict[str, Any]] = {}
    for q in entities:
        row = {"usable": 0, "usable_side_on": 0, "refused": 0, "refused_side_on": 0}
        for ref in (gallery.rows.get(q) or {}).get("refs", []):
            faces = (ref_faces.get(ref.get("sha256") or "") or {}).get("faces") or []
            idx = ref["face"] if ref["decision"] == "usable" else choose_reference_face(faces)[0]
            if idx is None or idx >= len(faces):
                continue
            kind = "usable" if ref["decision"] == "usable" else "refused"
            row[kind] += 1
            row[f"{kind}_side_on"] += side_on(face_geometry(faces[idx]))
        own_rows[q] = row
    listed, failed = commons_candidates(
        net, [{**{k: e.get(k) for k in ("qid", "type", "commons_category")}, "images": [], "logos": []}
              for e in entities.values() if e], limit=SIDE_VIEW_LISTED, cap=2 * SIDE_VIEW_LISTED)
    drawn: dict[str, list[str]] = {}
    for q, e in entities.items():
        taken = {r["title"].replace("_", " ") for r in e.get("refs") or []}
        fresh = [t for t, _s in listed.get(q, []) if t.replace("_", " ") not in taken]
        drawn[q] = sorted(fresh, key=lambda t: hashlib.sha256(f"{seed}:{q}:{t}".encode()).hexdigest())[:per_person]
    infos = image_info(net, [t for ts in drawn.values() for t in ts])
    found: dict[str, dict[str, Any]] = {}

    def download(q: str, title: str, info: dict[str, Any]):
        url = info["rendition"]["url"]
        resp, _chain = yield from recover.get(url, {"User-Agent": user_agent()}, net.log,
                                              f"file:{url.rsplit('/', 1)[-1][:80]}", "side_view", net.clock)
        gated = gate_reference(resp, info, corpus)
        net.log.attempt(f"file:{url.rsplit('/', 1)[-1][:80]}", "side_view", resp,
                        "ok" if gated["outcome"] == "usable" else "rejected",
                        None if gated["outcome"] == "usable" else gated["outcome"])
        got = {"outcome": gated["outcome"], "face": None}
        if gated["outcome"] == "usable":
            try:
                faces = backend.faces(load_bgr(resp.body), MIN_FACE["reference"])
            except Exception as exc:  # noqa: BLE001 -- a file the detector cannot read is a row
                got["outcome"] = f"unreadable:{type(exc).__name__}"
                found[f"{q}\n{title}"] = got
                return
            idx, decision = choose_reference_face(faces)
            medoid, resolves = medoids.get(q, (None, False))
            if idx is None and decision == "ambiguous_faces" and resolves:
                sims = [float(decode_embedding(f["embedding"]) @ medoid) for f in faces]
                if max(sims) >= floor:
                    idx, decision = int(np.argmax(sims)), "resolved_by_medoid"
            got["decision"] = decision
            if idx is not None:
                vec = decode_embedding(faces[idx]["embedding"])
                sim = None if medoid is None else round(float(vec @ medoid), 4)
                got["face"] = {"side_on": side_on(face_geometry(faces[idx])), "similarity_to_medoid": sim,
                               "agrees": sim is not None and sim >= floor, "vector": vec}
        found[f"{q}\n{title}"] = got

    try:
        net.run(download(q, t, infos[t]) for q, ts in drawn.items() for t in ts
                if eligible(infos.get(t) or {"missing": True}, "PERSON") is None)
    except recover.BudgetExceeded as exc:
        net.stopped = f"budget: {exc}"
    rows, side_faces = [], {}
    for rank, (q, mentions) in enumerate(people, 1):
        got = [(t, found.get(f"{q}\n{t}")) for t in drawn.get(q, [])]
        faces = [(t, g["face"]) for t, g in got if g and g["face"]]
        side_faces[q] = [(t, f["agrees"], f["vector"]) for t, f in faces if f["side_on"]]
        rows.append({"rank": rank, "qid": q, "label": entities[q].get("label"), "mentions": mentions,
                     "listing_failed": q in failed, "has_medoid": q in medoids, "gallery": own_rows[q],
                     "commons": {"listed": len(listed.get(q, [])), "drawn": len(drawn.get(q, [])),
                                 "passed_gates": sum(1 for _t, g in got if g and g["outcome"] == "usable"),
                                 "one_face": len(faces),
                                 "side_on": sum(1 for _t, f in faces if f["side_on"]),
                                 "side_on_agreeing": sum(1 for _t, f in faces if f["side_on"] and f["agrees"]),
                                 "face_on_agreeing": sum(1 for _t, f in faces if not f["side_on"] and f["agrees"])},
                     "side_on_files": [{"title": t, "similarity_to_medoid": f["similarity_to_medoid"]}
                                       for t, f in faces if f["side_on"]]})

    def count(test: Callable[[dict[str, Any]], bool]) -> int:
        return sum(1 for r in rows if test(r))

    n = len(rows)
    keep = count(lambda r: r["gallery"]["usable_side_on"] or r["commons"]["side_on_agreeing"])
    totals = {k: sum(r["commons"][k] for r in rows) for k in ("listed", "drawn", "passed_gates", "one_face",
                                                               "side_on", "side_on_agreeing", "face_on_agreeing")}
    enough = n > 0 and keep >= SIDE_VIEW_ENOUGH * n
    return {
        "what": (f"the {n} most-mentioned gallery people: their references by pose, and up to {per_person} more "
                 "Commons files each, drawn from what Commons lists as depicting them or in their category, "
                 "through the reference gates; a side-on face (SIDE_ON) agrees when it scores at or above the "
                 f"consistency threshold ({floor:.4f}) against the person's medoid. A measurement: nothing is "
                 "stored or added"),
        "people": n, "per_person": per_person,
        "gallery": {"people_with_a_usable_side_on_reference": count(lambda r: r["gallery"]["usable_side_on"]),
                    **{k: sum(r["gallery"][k] for r in rows) for k in ("usable", "usable_side_on", "refused",
                                                                       "refused_side_on")}},
        "commons": {**totals,
                    "people_with_a_side_on_face": count(lambda r: r["commons"]["side_on"]),
                    "people_with_an_agreeing_side_on_face": count(lambda r: r["commons"]["side_on_agreeing"]),
                    "people_without_a_medoid": count(lambda r: not r["has_medoid"]),
                    "listings_failed": count(lambda r: r["listing_failed"])},
        "people_with_a_side_view_the_gallery_would_keep": keep,
        "enough": {"rule": f"at least {SIDE_VIEW_ENOUGH:.0%} of the people", "met": enough},
        "statement": (f"Of the {n} most-mentioned people, {keep} have a side view the gallery would keep: one "
                      "among their references, or a side-on Commons face that agrees with their medoid "
                      f"({totals['side_on_agreeing']} agreeing of {totals['side_on']} side-on faces in "
                      f"{totals['one_face']} single-face files drawn). "
                      + ("Enough to fetch for." if enough else
                         "Too thin to fetch for: pose stays a stated limitation.")),
        "local": {"people": rows}}, side_faces


def cmd_side_views(args) -> int:
    backend = make_backend(args.backend)
    cid = face_config_id(backend)
    gallery = Gallery(cid)
    names = [r for r in read_jsonl(NAMES_PATH) if r["type"] == "PERSON" and in_scope(r)]
    people = people_by_mentions(names, gallery)[:args.top]
    wanted = {sha for q, _m in people for r in (gallery.rows.get(q) or {}).get("refs", []) if (sha := r["sha256"])}
    ref_faces = {r["sha256"]: r for r in read_jsonl(FACES_DIR / cid / "refs.jsonl") if r["sha256"] in wanted}
    floor = consistency_floor(cid)
    log(f"side views: the {len(people)} most-mentioned people, up to {args.per_person} Commons files each "
        f"(about {len(people) * (args.per_person + 2) + len(people) // 2:,} requests, one a second per host); "
        f"User-Agent: {user_agent()}")
    prov = JsonlLedger(PROVENANCE_PATH)
    prov.open()
    try:
        net = live_net(ProvenanceLog(prov, SIDE_VIEWS_PASS), int(args.budget_gb * 1024 ** 3))
        report, side_faces = side_view_census(net, backend, gallery, people, ref_faces, floor,
                                              per_person=args.per_person)
    finally:
        prov.close()
    report["requests"] = dict(net.requests)
    report["stopped"] = net.stopped
    # the sample's pose failures, scored against the side-on faces found for their person
    trials_path = CALIBRATION_DIR / f"{args.backend}_trials.jsonl"
    decisions = decisions_path(args.backend)
    if trials_path.is_file() and decisions.is_file():
        query_faces = latest(read_jsonl(QUERY_FACES_DIR / cid / "factify2.jsonl"), lambda r: r["sha256"])
        relabelled = adjudicated(list(read_jsonl(trials_path)),
                                 json.loads(decisions.read_text(encoding="utf-8")))
        rows = []
        for x in relabelled:
            if not (x.get("adjudicated") and x["label"] == "right" and x["scorable"] and x["score"] < floor
                    and side_on(x) and x["qid"] in side_faces):
                continue
            faces = query_faces[x["sha256"]]["faces"]
            best = gallery.best(faces, x["qid"])
            vec = decode_embedding(faces[best["query_face"]]["embedding"])
            scores = [(float(vec @ v), agrees) for _t, agrees, v in side_faces[x["qid"]]]
            rows.append({"trial": trial_id(x), "score_now": round(x["score"], 4),
                         "best_side_on": round(max((s for s, _a in scores), default=float("nan")), 4)
                         if scores else None,
                         "best_agreeing_side_on": round(max(s for s, a in scores if a), 4)
                         if any(a for _s, a in scores) else None})
        report["pose_failures"] = {
            "what": ("the sample's confirmed side-on recognition failures, scored against the side-on Commons "
                     "faces the census found for their person"),
            "failures": len(rows),
            "at_or_above_floor_against_an_agreeing_side_view": sum(
                1 for r in rows if r["best_agreeing_side_on"] is not None and r["best_agreeing_side_on"] >= floor),
            "local": {"trials": rows}}
    write_json(SIDE_VIEWS_LOCAL, {"generated_at": utcnow(), "face_config": cid, **report})
    write_json(SIDE_VIEWS_REPORT, {"generated_at": utcnow(), "face_config": cid, **public_report(report)})
    log(f"  {report['statement']}")
    if "pose_failures" in report:
        p = report["pose_failures"]
        log(f"  pose failures: {p['at_or_above_floor_against_an_agreeing_side_view']} of {p['failures']} reach "
            f"the floor against an agreeing side view")
    log(f"wrote {SIDE_VIEWS_LOCAL} (local) and {SIDE_VIEWS_REPORT} (aggregates)")
    return 0


SIDE_VIEWS_FETCH_PASS = "gallery-side-views-fetch-2026-10"
#: A person's decisions on references the consistency check refused, by
#: reference_id: "same_person" admits one, "someone_else" and "unsure" leave it
#: refused, null is undecided. assemble reads them at every embed.
REFERENCE_DECISIONS = REVIEW_DIR / "reference_decisions.json"
REFERENCE_DECISION_VALUES = ("same_person", "someone_else", "unsure")


def reference_decisions(path: Path = REFERENCE_DECISIONS) -> dict[str, str | None]:
    """A person's reference decisions, refused whole when one of them is not a
    decision: a mistyped "same person" would otherwise leave its reference
    refused, and nobody would see it."""
    decisions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    wrong = {k: v for k, v in decisions.items() if v is not None and v not in REFERENCE_DECISION_VALUES}
    if wrong:
        raise GalleryError(f"{path}: not a decision ({', '.join(REFERENCE_DECISION_VALUES)}, or null): "
                           + "; ".join(f"{k}: {v!r}" for k, v in sorted(wrong.items())))
    return decisions


#: A person's flags on references, by reference_id: how a reference may behave
#: at verify time. "age_gap": the person at a very different age from their
#: other references (Gandhi young, beside Gandhi old). They ride on the
#: gallery's rows; verify returns them with the reference it matched, and
#: calibrate counts the trials a flagged reference decides.
REFERENCE_FLAGS = REVIEW_DIR / "reference_flags.json"
REFERENCE_FLAG_VALUES = ("age_gap",)


def reference_flags(path: Path = REFERENCE_FLAGS, known: Iterable[str] | None = None) -> dict[str, list[str]]:
    """A person's reference flags, refused whole when one is not a flag, or
    (given the ``known`` reference ids) names no reference: a flag that lands
    nowhere would leave its reference unflagged, and nobody would see it."""
    flags = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    wrong = {k: v for k, v in flags.items()
             if not isinstance(v, list) or not v or any(f not in REFERENCE_FLAG_VALUES for f in v)}
    if wrong:
        raise GalleryError(f"{path}: not a list of flags ({', '.join(REFERENCE_FLAG_VALUES)}): "
                           + "; ".join(f"{k}: {v!r}" for k, v in sorted(wrong.items())))
    unknown = sorted(set(flags) - set(known)) if known is not None else []
    if unknown:
        raise GalleryError(f"{path}: no such reference: {', '.join(unknown)}")
    return flags


#: The decisions on a person's two disagreeing P18 images (review-p18-pairs).
#: Each sheet keeps its own file, so each sheet's figures come from its own
#: decisions; assemble applies them all.
P18_DECISIONS = REVIEW_DIR / "p18_decisions.json"
DECISION_FILES = (REFERENCE_DECISIONS, P18_DECISIONS)


def all_reference_decisions(paths: Sequence[Path] = DECISION_FILES) -> dict[str, str | None]:
    """Every sheet's reference decisions, each file checked as
    reference_decisions checks one; refused when two files decide one
    reference differently."""
    merged: dict[str, str | None] = {}
    for path in paths:
        for key, value in reference_decisions(path).items():
            if value is not None and merged.get(key) not in (None, value):
                raise GalleryError(f"{path}: {key} is {value!r} here and {merged[key]!r} in another decisions file")
            if value is not None or key not in merged:
                merged[key] = value
    return merged


def disagreeing_p18_pairs(entities: Sequence[dict[str, Any]], ref_faces: dict[str, dict[str, Any]],
                          threshold: float | None) -> list[dict[str, Any]]:
    """People whose single-face references are exactly two images from
    Wikidata (P18) that disagree. The P18 rule keeps both, so their medoid,
    the first, decides every match for that person, though one of the two is
    likely wrong: P18 is not guaranteed to be a usable reference photo -- a
    statue bust and a 1991 postage stamp serve as P18 here (2026-10-23). A
    person picks (review-p18-pairs)."""
    out: list[dict[str, Any]] = []
    for e in entities:
        if e["type"] != "PERSON" or threshold is None:
            continue
        _rows, vec, _side = reference_picks(e, ref_faces)
        if len(vec) != 2 or any(e["refs"][i]["source"] != "P18" for i in vec):
            continue
        a, b = list(vec)
        sim = float(vec[a] @ vec[b])
        if sim < threshold:
            out.append({"qid": e["qid"], "refs": [a, b], "ids": [reference_id(e["qid"], e["refs"][i]) for i in (a, b)],
                        "similarity": round(sim, 4), "medoid": person_medoid(e["refs"], vec)})
    return out


def p18_review(rows: Sequence[dict[str, Any]], decisions: dict[str, str | None],
               pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What a person decided about people's two disagreeing P18 images. Two a
    person confirmed are both the person measure how far apart the recognizer
    can put one face across pose, lighting and headwear: a genuine pair under
    the threshold, which bears on the TAR calibration can expect."""
    tally = decision_tally(rows, decisions)
    undecided = sum(1 for p in pairs if not any(decisions.get(i) for i in p["ids"]))
    same = sorted(p["similarity"] for p in pairs if all(decisions.get(i) == "same_person" for i in p["ids"]))
    return {**tally, "people": len(pairs), "people_undecided": undecided, "same_person_pairs": same,
            "statement": (f"{len(pairs)} people have two P18 images that disagree, which the P18 rule keeps "
                          f"until a person picks; {undecided} of them undecided (review-p18-pairs)"
                          + (f"; a person refused {tally['someone_else']} of their images and confirmed "
                             f"{tally['same_person']}" if tally["judged"] else "")
                          + (f"; {len(same)} pair(s) a person confirmed as the same person score "
                             f"{', '.join(f'{s:.4f}' for s in same)} against each other" if same else ""))}


def check_state(ref: dict[str, Any]) -> str:
    """What the consistency check itself decided about an assembled reference,
    whatever a person decided after it: refused, kept, or never checked."""
    if ref["decision"] in REFUSED_BY_CHECK or ref.get("selected_by") == "person_confirmed":
        return "refused"
    return "kept" if ref["decision"] in ("usable", "person_refused") else "unchecked"


def decision_tally(rows: Sequence[dict[str, Any]], decisions: dict[str, str | None]) -> dict[str, Any]:
    """A person's decisions, counted, and beside each what the check decides
    about that reference now."""
    decided = {k: v for k, v in decisions.items() if v is not None}
    judged = Counter(decided.values())
    now: Counter = Counter()
    for row in rows:
        for ref in row["refs"]:
            verdict = decided.get(reference_id(row["qid"], ref))
            if verdict is not None:
                now[(check_state(ref), verdict)] += 1
    return {"judged": len(decided), **{v: judged[v] for v in REFERENCE_DECISION_VALUES},
            "now": {state: {v: now[(state, v)] for v in REFERENCE_DECISION_VALUES if now[(state, v)]}
                    for state in ("refused", "kept", "unchecked")
                    if any(now[(state, v)] for v in REFERENCE_DECISION_VALUES)}}


def reference_review(rows: Sequence[dict[str, Any]], decisions: dict[str, str | None]) -> dict[str, Any]:
    """The consistency check's measured accuracy where a person has looked:
    the side-view sheet gives a reference a decision slot only when the check
    refused it (review-side-views), so every decision from that sheet is a
    refusal a person judged -- right when it shows someone else. And what the
    check decides about them now, which can differ after a change to it."""
    tally = decision_tally(rows, decisions)
    right = tally["someone_else"]
    share = image_text._ratio(right, tally["judged"], 4)
    kept = sum((tally["now"].get("kept") or {}).values())
    return {"what": ("references the consistency check refused that a person has judged (review-side-views: "
                     "side-on references of the most-mentioned people): a refusal is right when the reference "
                     "shows someone else"),
            **{k: tally[k] for k in ("judged", *REFERENCE_DECISION_VALUES)}, "refusals_right": share,
            "now": tally["now"],
            "statement": (f"A person judged {tally['judged']} refusals of the consistency check: {right} right "
                          f"(someone else), {tally['same_person']} wrong (the same person), {tally['unsure']} "
                          f"unsure; the check's refusals were right {(share or 0):.0%} of the time"
                          + (f". The check keeps {kept} of them now" if kept else ""))}


def side_view_outcomes(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the gallery decided about its side views (assembled rows): kept by
    the consistency check, refused by it, admitted or refused by a person
    against it, or out before the check (no single face, or the gates)."""
    got = Counter()
    for row in rows:
        for ref in row["refs"]:
            if ref["source"] != SIDE_VIEW:
                continue
            got["fetched"] += 1
            if ref["decision"] == "usable":
                got["admitted_by_a_person" if ref.get("selected_by") == "person_confirmed" else "kept"] += 1
            elif ref["decision"] == "person_refused":
                got["refused_by_a_person"] += 1
            elif ref["decision"] in REFUSED_BY_CHECK:
                got["refused_by_the_check"] += 1
            else:
                got["no_single_face_or_gated"] += 1
    kept = got["kept"] + got["refused_by_a_person"]
    checked = kept + got["refused_by_the_check"] + got["admitted_by_a_person"]
    rate = image_text._ratio(got["refused_by_the_check"] + got["admitted_by_a_person"], checked, 4)
    return {"fetched": got["fetched"], "kept": got["kept"], "refused_by_the_check": got["refused_by_the_check"],
            "admitted_by_a_person": got["admitted_by_a_person"], "refused_by_a_person": got["refused_by_a_person"],
            "no_single_face_or_gated": got["no_single_face_or_gated"], "refusal_rate": rate,
            "statement": (f"{got['fetched']} side views fetched: the consistency check kept {kept} and "
                          f"refused {got['refused_by_the_check'] + got['admitted_by_a_person']} of the "
                          f"{checked} it checked ({(rate or 0):.0%}), {got['admitted_by_a_person']} of them "
                          "since admitted by a person"
                          + (f", and a person refused {got['refused_by_a_person']} it kept"
                             if got["refused_by_a_person"] else "")
                          + f"; {got['no_single_face_or_gated']} never reached the check")}


def cmd_fetch_side_views(args) -> int:
    """The side-on files a side-views census found for its people, fetched as
    references (SIDE_VIEW) through the reference gates. The next embed's
    consistency check decides which the gallery keeps, at the pinned
    threshold, never relaxed."""
    if not SIDE_VIEWS_LOCAL.is_file():
        raise GalleryError(f"no census at {SIDE_VIEWS_LOCAL}: run side-views first; it finds the side views")
    census = json.loads(SIDE_VIEWS_LOCAL.read_text(encoding="utf-8"))
    stored = latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"])
    candidates: dict[str, list[tuple[str, str]]] = {}
    for p in census["local"]["people"][:args.top]:
        e = stored.get(p["qid"])
        if e is None:
            continue
        taken = {r["title"].replace("_", " ") for r in e["refs"]}
        files = [f["title"] for f in p["side_on_files"] if f["title"].replace("_", " ") not in taken]
        if files:
            candidates[p["qid"]] = [(t, SIDE_VIEW) for t in dict.fromkeys(files)]
    n = sum(len(v) for v in candidates.values())
    log(f"fetch side views: {n} files for {len(candidates)} of the census's {min(args.top, census['people'])} "
        f"people; User-Agent: {user_agent()}")
    if not candidates:
        return 0
    out, prov = JsonlLedger(ENTITIES_PATH), JsonlLedger(PROVENANCE_PATH)
    out.open()
    prov.open()
    try:
        net = live_net(ProvenanceLog(prov, SIDE_VIEWS_FETCH_PASS), int(args.budget_gb * 1024 ** 3))
        rows = fetch_references(net, [stored[q] for q in candidates], candidates=candidates)
        out.append_many([{**stored[row["qid"]], "pass": SIDE_VIEWS_FETCH_PASS, "side_views_fetched_at": utcnow(),
                          "refs": stored[row["qid"]]["refs"] + row["refs"]} for row in rows])
    finally:
        out.close()
        prov.close()
    outcomes = Counter(r.get("outcome") for row in rows for r in row["refs"])
    log(f"  {len(rows)} people written ({len(candidates) - len(rows)} left for the next run); side-view "
        f"outcomes {dict(outcomes)}; run embed next, and the consistency check decides")
    return 0


def cmd_review_references(args) -> int:
    """A sheet a PERSON reads: every side-on reference the consistency check
    refused for the census's people, beside the references the gallery kept
    for that person, with its Commons page and licence. Local only. The
    decisions are a person's, in REFERENCE_DECISIONS; "same_person" admits a
    reference at the next embed."""
    backend = make_backend(args.backend)
    gallery = Gallery(face_config_id(backend))
    census = json.loads(SIDE_VIEWS_LOCAL.read_text(encoding="utf-8"))
    stored = latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"])
    people = [p["qid"] for p in census["local"]["people"][:args.top]]
    wanted = {r["sha256"] for q in people for r in (stored.get(q) or {}).get("refs", []) if r.get("sha256")}
    ref_faces = {r["sha256"]: r for r in read_jsonl(FACES_DIR / gallery.cid / "refs.jsonl") if r["sha256"] in wanted}
    queue = []
    for q in people:
        decided = {r["sha256"]: r for r in (gallery.rows.get(q) or {}).get("refs", []) if r.get("sha256")}
        for ref in (stored.get(q) or {}).get("refs", []):
            d = decided.get(ref.get("sha256"))
            if not d or d["decision"] not in REFUSED_BY_CHECK:
                continue
            faces = (ref_faces.get(ref["sha256"]) or {}).get("faces") or []
            idx, _how = choose_reference_face(faces)
            if idx is not None and side_on(face_geometry(faces[idx])):
                queue.append((q, ref, faces[idx], d.get("similarity_to_medoid")))
    written = merge_decision_slots(REFERENCE_DECISIONS, [reference_id(q, ref) for q, ref, _f, _s in queue])
    cells = []
    for n, (q, ref, face, sim) in enumerate(queue, 1):
        body = (image_text.PROJECT_ROOT / ref["path"]).read_bytes()
        kept = "".join(f"<img src='{_crop((image_text.PROJECT_ROOT / r['path']).read_bytes(), f)}'>"
                       for r, f, _v in gallery.refs.get(q, [])[:3])
        lic = (ref.get("licence") or {}).get("short_name") or ""
        cells.append(
            f"<tr><td><b>{n}</b></td><td><img src='{_crop(body, face, REVIEW_CROP)}'><br>"
            f"<code>{html.escape(reference_id(q, ref))}</code><br>{html.escape(ref['source'])}; similarity to "
            f"the medoid {sim}<br><a href='{html.escape(ref.get('page_url') or '')}'>"
            f"{html.escape(ref.get('title') or '')}</a><br>{html.escape(lic)}</td><td>{kept or '(none kept)'}"
            f"<br>{html.escape((gallery.entities.get(q) or {}).get('label') or q)}</td></tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>Side-view review</title>"
            "<style>body{font:14px sans-serif;margin:16px}td{vertical-align:top;padding:6px;"
            f"border-bottom:1px solid #ccc}}img{{max-width:{REVIEW_CROP}px;margin-right:4px}}</style>"
            f"<h1>Side-view references the consistency check refused: {len(queue)}</h1>"
            "<p>Each row: a side-on face from a Commons file (left), and the references the gallery kept for "
            "that person (right). " + SCORED_FACE_NOTE + " The check refused it at the pinned threshold, which "
            "is never relaxed: a face turned away scores low against a face-on medoid, so only a person can say "
            "whether it is the same person. Write a decision against the row's id in "
            f"<code>{html.escape(str(REFERENCE_DECISIONS))}</code> (every row is already there, as "
            "<code>null</code>): <code>\"same_person\"</code> admits it at the next embed; "
            "<code>\"someone_else\"</code> or <code>\"unsure\"</code> leave it refused.</p>"
            f"<table>{''.join(cells)}</table>")
    path = REVIEW_DIR / f"{args.backend}_side_views.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    log(f"{len(queue)} refused side-on references for {len({q for q, *_r in queue})} people; {written} new "
        f"decision slot(s) in {REFERENCE_DECISIONS}; sheet {path} (local: it shows people's faces)")
    return 0


def cmd_review_p18_pairs(args) -> int:
    """A sheet a PERSON reads: each person whose two single-face references
    are both Wikidata's own image (P18) and disagree, the two side by side,
    with the faces their medoid decides. Local only. The decisions are a
    person's, in P18_DECISIONS: "someone_else" against an image refuses it as
    the person's reference, and the other becomes the one their faces are
    checked against; "same_person" confirms one; "unsure" leaves both."""
    backend = make_backend(args.backend)
    gallery = Gallery(face_config_id(backend))
    floor = consistency_floor(gallery.cid)
    ref_faces = latest(read_jsonl(FACES_DIR / gallery.cid / "refs.jsonl"), lambda r: r["sha256"])
    pairs = disagreeing_p18_pairs(list(gallery.entities.values()), ref_faces, floor)
    written = merge_decision_slots(P18_DECISIONS, [i for p in pairs for i in p["ids"]])
    decisions = reference_decisions(P18_DECISIONS)

    def crop_of(ref: dict[str, Any], face: dict[str, Any], side: int = REVIEW_CROP) -> str:
        return _crop((image_text.PROJECT_ROOT / ref["path"]).read_bytes(), face, side)

    cells = []
    for n, p in enumerate(pairs, 1):
        e = gallery.entities[p["qid"]]
        row = gallery.rows.get(p["qid"]) or {"refs": []}
        # the medoid now: once a person refuses one image, the other (if a person kept either)
        refused = {i for i in p["refs"] if i < len(row["refs"]) and row["refs"][i]["decision"] == "person_refused"}
        medoid = p["medoid"] if not refused else next((i for i in p["refs"] if i not in refused), None)
        sides = []
        for i, rid in zip(p["refs"], p["ids"]):
            ref = e["refs"][i]
            faces = ref_faces[ref["sha256"]]["faces"]
            face = faces[choose_reference_face(faces)[0]]
            body = (image_text.PROJECT_ROOT / ref["path"]).read_bytes()
            lic, att = ref.get("licence") or {}, ref.get("attribution") or {}
            sides.append(
                f"<td><img src='{_crop(body, face, REVIEW_CROP)}'><br><img class='whole' "
                f"src='{_whole(body, face['box'])}'><br><code>{html.escape(rid)}</code>"
                f"{' <b>(the medoid now)</b>' if i == medoid else ''}<br><a href='"
                f"{html.escape(ref.get('page_url') or '')}'>{html.escape(ref.get('title') or '')}</a><br>"
                f"{html.escape(lic.get('short_name') or '')}; {html.escape(att.get('artist') or '')}<br>"
                f"decided: <b>{html.escape(str(decisions.get(rid)))}</b></td>")
        # what the medoid decides: the group photos' faces checked against it, and the side views
        hangs = [(e["refs"][j], ref_faces[r["sha256"]]["faces"][r["face"]]) for j, r in enumerate(row["refs"])
                 if r["decision"] == "usable" and (r.get("selected_by") == "resolved_by_medoid"
                                                   or r["source"] == SIDE_VIEW)]
        small = "".join(f"<img class='small' src='{crop_of(r, f, 96)}'>" for r, f in hangs)
        cells.append(
            f"<tr><td><b>{n}</b></td>{''.join(sides)}<td><b>{html.escape(e.get('label') or p['qid'])}</b>"
            f"<br><small>{html.escape(e.get('description') or '')}</small><br>the two score "
            f"{p['similarity']} against each other, under the threshold {floor}<br>{len(hangs)} other "
            f"face(s) checked against the medoid now:<br>{small}</td></tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>Two P18 images</title>"
            "<style>body{font:14px sans-serif;margin:16px}td{vertical-align:top;padding:6px;"
            f"border-bottom:1px solid #ccc}}img{{max-width:{REVIEW_CROP}px}}img.whole{{max-width:320px}}"
            "img.small{max-width:96px;margin-right:4px}</style>"
            f"<h1>People with two P18 images that disagree: {len(pairs)}</h1>"
            "<p>Each row: the two images Wikidata gives as the person's own (P18), side by side, each with the "
            "face the gallery took boxed in red and its landmarks drawn (eyes green, nose yellow, mouth corners "
            "blue), and the whole image below. The two disagree, so one is likely wrong, and the P18 rule keeps "
            "both: the medoid, the first, decides every match for the person, including the faces at the "
            "right. P18 is no guarantee of a usable photo: a statue bust and a postage stamp serve as P18 here. "
            "Write a decision against an image's id in "
            f"<code>{html.escape(str(P18_DECISIONS))}</code> (every image is already there, as "
            "<code>null</code>): <code>\"someone_else\"</code> refuses it as the person's reference, and the "
            "other becomes the one their faces are checked against; <code>\"same_person\"</code> confirms it; "
            "<code>\"unsure\"</code> leaves it. Refusing both leaves the person no usable face from these two. "
            "Run <code>embed</code> after.</p>"
            f"<table>{''.join(cells)}</table>")
    path = REVIEW_DIR / f"{args.backend}_p18_pairs.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    log(f"{len(pairs)} people with two P18 images that disagree; {written} new decision slot(s) in "
        f"{P18_DECISIONS}; sheet {path} (local: it shows people's faces)")
    return 0


def cmd_status(args) -> int:
    out: dict[str, Any] = {"names": None, "claim_ner": {}}
    if NAMES_PATH.is_file():
        types = Counter(r["type"] for r in read_jsonl(NAMES_PATH))
        out["names"] = dict(types)
    directory = CLAIM_NER_DIR / config_id_of(ner_config())
    if directory.is_dir():
        for path in sorted(directory.glob("*.jsonl")):
            if not path.name.endswith(".torn.jsonl"):
                out["claim_ner"][path.stem] = sum(1 for _ in read_jsonl(path))
    out["links"] = dict(Counter(r["status"] for r in latest(
        read_jsonl(LINKS_PATH), lambda r: (r["type"], r["key"])).values()))
    entities = latest(read_jsonl(ENTITIES_PATH), lambda r: r["qid"])
    out["entities"] = dict(Counter(e["type"] for e in entities.values()))
    out["reference_outcomes"] = dict(Counter(r.get("outcome") for e in entities.values() for r in e["refs"]))
    log(json.dumps(out, indent=2))
    return 0


def _datasets(text: str) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    unknown = sorted(set(names) - set(CLAIM_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"not an image-bearing corpus: {', '.join(unknown)}")
    return names


def _models(text: str) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    unknown = sorted(set(names) - set(MODEL_CHOICES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model {', '.join(unknown)}; choose from "
                                         f"{', '.join(MODEL_CHOICES)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-models", help="download the pinned weights (network)")
    f.add_argument("--only", type=_models, default=list(DEFAULT_MODELS),
                   help=f"comma-separated, from {', '.join(MODEL_CHOICES)} "
                        f"(default {','.join(DEFAULT_MODELS)}; auraface is the bake-off contrast)")
    c = sub.add_parser("count", help="names in our corpora, before anything is fetched; offline")
    c.add_argument("--ledger-dir", default=None,
                   help="image_text run directory (default: the one under the current row schema)")
    c.add_argument("--claims", type=_datasets, default=list(CLAIM_DATASETS),
                   help="comma-separated corpora whose claim text NER reads (default: every "
                        "image-bearing corpus)")
    c.add_argument("--no-claims", action="store_true", help="the VLM's names only")
    c.add_argument("--device", choices=("cuda", "cpu"), default=None)
    k = sub.add_parser("link", help="names to Wikidata items: the sample, or --all (network)")
    k.add_argument("--random", type=int, default=SAMPLE_RANDOM,
                   help=f"stratified random names, for the linking rate (default {SAMPLE_RANDOM})")
    k.add_argument("--dry-run", action="store_true", help="size the work; no request, no write")
    k.add_argument("--refresh-entities", action="store_true",
                   help="fetch every stored item's summary again (network)")
    held = k.add_mutually_exclusive_group()
    held.add_argument("--hold", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                      help="comma-separated PERSON name keys whose links a person must check first; offline")
    held.add_argument("--release", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                      help="comma-separated name keys whose held links a person checked; offline")
    k.add_argument("--because", default=None, help="why the links are held (or --reasons)")
    k.add_argument("--reasons", default=None, help="a JSON file: name key -> why it is held")
    k.add_argument("--by", default=None, help="who holds or releases them")
    f2 = sub.add_parser("fetch", help="Commons references through the gates: the sample, or --all (network)")
    f2.add_argument("--budget-gb", type=float, default=1.0, help="transfer budget (default 1 GiB)")
    f2.add_argument("--regate", action="store_true",
                    help="judge rendition-refused references again from their quarantined bytes; offline")
    for p in (k, f2):
        p.add_argument("--all", action="store_true",
                       help="the full build: every name in --scope (linked names, for fetch), not the sample")
        p.add_argument("--scope", type=_datasets, default=list(BUILD_SCOPE),
                       help=f"comma-separated corpora the gallery is built for (default "
                            f"{','.join(BUILD_SCOPE)}); widening it later adds, never redoes")
    e = sub.add_parser("embed", help="faces of references and calibration images; offline, CPU")
    e.add_argument("--pin-consistency", action="store_true",
                   help="pin the consistency threshold derived now, so later additions leave "
                        "every existing entry as it is")
    a = sub.add_parser("calibrate", help="threshold and its cost, Factify2 train only; offline")
    v = sub.add_parser("verify", help="one 1:1 verification, with the reference it matched")
    v.add_argument("--image", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--type", choices=TYPES, default="PERSON")
    v.add_argument("--threshold", type=float, default=None,
                   help="also compare the score with this threshold: a comparison, never a decision; only "
                        "a threshold calibration certified decides, and without one verify refuses")
    r = sub.add_parser("review", help="a local HTML sheet of decisive trials, for a person to check")
    r.add_argument("--rejected", type=int, default=20, help="rejected right names to include")
    sub.add_parser("sample-report", help="data/reports/gallery_sample.json from the sample's stages")
    s = sub.add_parser("side-views", help="does Commons hold side views of the most-mentioned people? "
                                          "A measurement; nothing is stored (network)")
    s.add_argument("--top", type=int, default=SIDE_VIEW_TOP, help=f"people (default {SIDE_VIEW_TOP})")
    s.add_argument("--per-person", type=int, default=SIDE_VIEW_PER_PERSON,
                   help=f"Commons files drawn per person (default {SIDE_VIEW_PER_PERSON})")
    s.add_argument("--budget-gb", type=float, default=1.0, help="transfer budget (default 1 GiB)")
    fs = sub.add_parser("fetch-side-views", help="the census's side-on files, fetched as references (network)")
    fs.add_argument("--top", type=int, default=SIDE_VIEW_TOP, help=f"people (default {SIDE_VIEW_TOP})")
    fs.add_argument("--budget-gb", type=float, default=1.0, help="transfer budget (default 1 GiB)")
    rr = sub.add_parser("review-side-views", help="a local sheet of side views the consistency check refused")
    rr.add_argument("--top", type=int, default=SIDE_VIEW_TOP, help=f"people (default {SIDE_VIEW_TOP})")
    rp = sub.add_parser("review-p18-pairs", help="a local sheet of people whose two P18 images disagree")
    for p in (k, e, a):
        p.add_argument("--ledger-dir", default=None,
                       help="image_text run directory (default: the one under the current row schema)")
    for p in (e, a, v, r, s, rr, rp):
        p.add_argument("--backend", choices=FACE_BACKENDS, default="sface")
    sub.add_parser("status", help="offline counts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"fetch-models": cmd_fetch_models, "count": cmd_count, "link": cmd_link,
                "fetch": cmd_fetch, "embed": cmd_embed, "calibrate": cmd_calibrate,
                "verify": cmd_verify, "review": cmd_review, "sample-report": cmd_sample_report,
                "side-views": cmd_side_views, "fetch-side-views": cmd_fetch_side_views,
                "review-side-views": cmd_review_references,
                "review-p18-pairs": cmd_review_p18_pairs, "status": cmd_status}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
