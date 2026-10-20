#!/usr/bin/env python
"""Images to text: what each usable image shows, whom and what it names, and the text in it.

    python scripts/image_text.py fetch-models                 # once; the only command that opens a socket
    python scripts/image_text.py run --pilot 200              # measure, report, STOP: nothing at scale
    python scripts/image_text.py run                          # VERITE: 607 images, resumable
    python scripts/image_text.py run --through averimatec     # ... then AVerImaTeC: 1,345
    python scripts/image_text.py run --through factify2       # ... then Factify2: 45,186
    python scripts/image_text.py run --through fakeddit --fakeddit-n 20000
    python scripts/image_text.py recheck --n 3                # a fresh process re-runs finished rows; bytes must match
    python scripts/image_text.py status                       # offline counts, no model

On a shared server with no tmux, start it detached and simply start it again
whenever it dies; a re-run resumes::

    nohup python scripts/image_text.py run --through factify2 >> image_text.log 2>&1 &

**One row per distinct image** -- per sha256, per dataset -- appended to
``data/interim/image_text/<config_id>/<dataset>.jsonl``. Identical bytes give
identical output, so Factify2's 75,613 image paths are 45,186 model runs. A
row carries the sha256 and the (label-free) record_ids that use it, and never
a path or a label: VERITE's file names are its labels.

Three readings of every image, each its own call so that one failing never
costs the others:

- **OCR**: EasyOCR, independent of the VLM. Every region keeps its text,
  confidence and polygon (original-image coordinates).
- **description**: Qwen2.5-VL-7B-Instruct, one appearance-only paragraph. It
  names no one unless the name is written in the image -- an invented name in
  a description a human reads is worse than none.
- **entities**: a second, separate generate call, five lines (PERSON / ORG /
  GPE / EVENT / DATE). Here recognition is the point -- the downstream check is
  "the caption claims this is X; the image is not X" -- so a person may be
  named from their appearance, tagged ``[seen]``, or from text in the image,
  ``[text]``. When the model cannot name someone it must say UNIDENTIFIED.
  Line-oriented, not JSON: a malformed line loses one category, not the
  description too. The tag is kept as the model gave it but is NOT evidence --
  on Factify2 train images it tagged printed names ``[seen]`` -- so every name
  is also checked against the independent OCR text (``in_ocr``): a name the
  OCR cannot find was not read. Nothing is dropped by a threshold; tag, OCR
  support and confidence stay visible for the consumer to judge.

  The answer is held to its five lines while it is generated (``Skeleton``):
  each line opens with its header, in order, and the answer may end only
  after DATE's line. Where the model would break that form -- the first pilot
  ended 63 of its 199 answers right after the PERSON line -- the skeleton puts
  in the newline or the header, records the replacement, and marks every line
  from there on ``forced``, on the category and on each of its items. A forced
  line is weaker evidence than a free one: statistics are split by it, and
  entities leave a row only through ``entities_of``, which makes the caller
  say what to do with forced lines. Lines before the first replacement are
  exactly what the model wrote unprompted. An answer that lost its lines
  anyway stops the run instead of being written.

  A VLM name is a candidate, never a verification. On the first pilot's
  recognition probe most names produced from appearance alone were not the
  figure the claim names, and confidence did not separate them; identity is
  decided by an external gallery. The pilot report names both bars: the
  precision a name needs to replace a gallery, and what it needs as a
  candidate that a gallery then verifies.

**Confidence** is the model's own and is not calibrated: the description's is
the geometric-mean probability of its tokens, an entity's the minimum over the
tokens of its name, OCR's EasyOCR's per region. Every emitted token is checked
against the argmax of its step's scores -- except where the entity skeleton
replaced it, and those steps are recorded and left out of every confidence --
so "greedy" is verified, not assumed from a config.

**Determinism.** NF4 via bitsandbytes, bf16 compute, greedy decoding, batch
size 1, fixed seed, deterministic cuDNN/cuBLAS, and a resolution cap that is a
constant (``MAX_PIXELS``), recorded per row with the size actually fed. The
same image twice gives byte-identical output: every row carries
``result_sha256`` over everything except timings and the machine.

**Checkpointing.** One JSONL line per image, written in a single write and
fsynced before the next image starts. A kill at any moment loses at most the
image in flight: on restart the ledger is read, a torn last line is cut (and
kept, in ``<dataset>.torn.jsonl``), and finished images are skipped. A damaged
line anywhere else stops the run; it is never skipped. One writer per ledger,
enforced by an OS lock that dies with its process.

**Failures are rows.** An image that is missing, unreadable, changed since the
index (``sha256_mismatch``: raw data is immutable), undecodable or of an
unsupported shape gets a row with a reason code. An image in the placeholder
registry -- placeholder, furniture, blank, verdict -- or a fingerprint
rejection is ``skipped`` by design, recorded as such, and never shown to the
model. The gates run on the bytes read now, not on what the index said.

**Order.** VERITE, then AVerImaTeC, then Factify2, then a proportional,
nested, stratified Fakeddit sample; nothing past VERITE runs without
``--through``. ``--pilot N`` processes N images, runs a recognition probe on
Factify2 train images of public figures, re-checks determinism, prints
throughput, peak VRAM and projected wall-clock, and stops.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import sys
import time
import unicodedata
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

# cuBLAS reads this when its first handle is created; without it some GEMM
# reductions may change between runs. It must be set before torch touches CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, MODELS, PROJECT_ROOT, RAW, REPORTS, free_gb  # noqa: E402
from scripts import images  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402
from scripts.placeholders import MISSING_STATUSES  # noqa: E402

#: 2: every entity category, and every item in it, carries ``forced``. Rows of
#: an earlier version stay readable under their own schema (ROW_SCHEMAS); only
#: the current one is written.
SCHEMA_VERSION = 2
SCHEMA = f"image_text/{SCHEMA_VERSION}"
#: Bump when parse_entities() or the content gates change: rows record it, and
#: it is part of the config, so a new parser never writes into an old ledger.
#: 2: a whole answer of "NONE" means none in every category (was: unparseable).
#: 3: lines are split on "\n" only, the line the skeleton holds; skeleton
#:    replacements are flags, forced lines are marked, and replaced tokens are
#:    left out of every confidence.
PARSER_VERSION = 3
SEED = 20260901

#: The order datasets are processed in. ``--through X`` runs ORDER up to X.
ORDER = ("verite", "averimatec", "factify2", "fakeddit")

OUT_ROOT = INTERIM / "image_text"
PILOT_REPORT = REPORTS / "image_text_pilot.json"

# --------------------------------------------------------------------------
# the model, pinned
# --------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
#: The Hub's ``main`` on 2026-10-05, pinned so every machine loads the same bytes.
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
#: Read from the Hub's model-card metadata on 2026-10-05, not from memory;
#: fetch-models re-reads it and refuses if it has changed.
MODEL_LICENCE = "apache-2.0"
HF_CACHE = MODELS / "hf"
#: Files fetch-models takes from the repo (not README.md / .gitattributes).
MODEL_FILES = ("*.json", "*.safetensors", "*.txt")

#: What NF4 means here, verified on the loaded modules (QwenVLM._check_quantization).
#: lm_head stays bf16 (the transformers default); every other Linear, the vision
#: tower's included, is 4-bit -- a bf16 vision tower does not fit beside the LLM
#: in 8 GB. Embeddings, norms and the vision patch convolution are not Linear
#: layers and stay bf16.
QUANTIZATION = {"method": "bitsandbytes", "load_in_4bit": True, "quant_type": "nf4",
                "compute_dtype": "bfloat16", "double_quant": True, "not_quantized": ["lm_head"]}

EASYOCR_DIR = MODELS / "easyocr"
#: sha256 of the EasyOCR 1.7.2 weights for ['en'] (hashed 2026-10-05). A run
#: refuses other bytes: different OCR weights are a different experiment.
EASYOCR_WEIGHTS = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    "english_g2.pth": "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8",
}
OCR_LANGS = ("en",)
#: OCR input cap: images are downscaled (never up) to this longest side, and
#: EasyOCR's own canvas is set to the same value so it resizes nothing further.
OCR_MAX_SIDE = 1600

# --------------------------------------------------------------------------
# the resolution cap: a constant, never a function of the image
# --------------------------------------------------------------------------

#: Qwen2.5-VL sees 14-px patches merged 2x2, so a side must be a multiple of 28.
FACTOR = 28
#: Qwen's own minimum (4 merged tokens): small images are not inflated.
MIN_PIXELS = 4 * FACTOR * FACTOR
#: At most 1,024 visual tokens per image, whatever its size.
MAX_PIXELS = 1024 * FACTOR * FACTOR
#: Qwen's processor refuses anything more elongated than this.
MAX_ASPECT = 200

# --------------------------------------------------------------------------
# prompts: fixed here, and NOT tuned on VERITE (it is the evaluation set);
# any iteration happens on Factify2 train images
# --------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

DESCRIBE_PROMPT = (
    "Describe this image in one paragraph of at most 120 words.\n"
    "Start with the kind of image: photograph, screenshot, document, chart, map, meme, "
    "illustration, or other.\n"
    "Then describe what is visible: the setting, the people and what they are doing, notable "
    "objects, and any text, briefly summarised.\n"
    "Describe people only by their appearance and role, for example \"a grey-haired man in a "
    "dark suit speaking at a podium\". Do not name or identify anyone unless their name is "
    "written in the image; identities are recorded separately.\n"
    "Do not guess where or when the image was taken, or whether it is genuine. Report only "
    "what is visible."
)

ENTITIES_PROMPT = (
    "List the named entities in this image: people, organisations, places, events and dates.\n"
    "\n"
    "Name an entity when its name is written in the image (a caption, sign, logo, banner, label "
    "or on-screen text), or when you recognise it from how it looks, such as a well-known "
    "politician, celebrity or public figure, a famous landmark, or a familiar logo. If you are "
    "not confident who or what it is, do not guess: a wrong name is worse than no name.\n"
    "\n"
    "Answer with exactly these five lines and nothing else:\n"
    "PERSON: ...\n"
    "ORG: ...\n"
    "GPE: ...\n"
    "EVENT: ...\n"
    "DATE: ...\n"
    "\n"
    "Rules:\n"
    "- Separate several entries with \" ; \".\n"
    "- After each entry write [text] if its name is written in the image, or [seen] if you "
    "recognised it from its appearance.\n"
    "- PERSON: name every person you can identify, whether from written text or from their face "
    "and appearance. If people are visible but you cannot identify them, write UNIDENTIFIED. If "
    "some can be identified and others cannot, list the names and then UNIDENTIFIED. If no "
    "people are visible, write NONE.\n"
    "- On any other line with nothing to report, write NONE.\n"
    "- GPE means countries, states and cities. DATE means a date or year written in the image."
)

PROMPTS = {"description": DESCRIBE_PROMPT, "entities": ENTITIES_PROMPT}
MAX_NEW_TOKENS = {"description": 256, "entities": 192}
CATEGORIES = ("PERSON", "ORG", "GPE", "EVENT", "DATE")
#: The entity answer's form: each line opens with its header, in this order.
HEADERS = tuple(f"{c}:" for c in CATEGORIES)
#: What a skeleton replacement did, as the prefix of its flag ("continued_after:PERSON").
SKELETON_EVENTS = ("continued_after", "header_forced", "ended_after")

# --------------------------------------------------------------------------
# statuses and reason codes
# --------------------------------------------------------------------------

STATUSES = ("ok", "partial", "skipped", "failed")
#: Skipped by design: the registry's missing statuses and fingerprint rejections.
SKIP_REASONS = tuple(sorted(MISSING_STATUSES | {images.WRONG_IMAGE}))
FAIL_REASONS = ("missing_file", "missing_member", "unreadable", "sha256_mismatch",
                "undecodable", "unsupported_geometry")
COMPONENTS = ("ocr", "description", "entities")
#: A row names no file and no label; a machine is named by a hash, never by its
#: hostname (the repository is double-blind).
FORBIDDEN_KEYS = frozenset({"path", "paths", "label", "labels", "url", "urls", "hostname", "host"})

# --------------------------------------------------------------------------
# the recognition probe (pilot only)
# --------------------------------------------------------------------------

#: Public figures Factify2 train claims name: (name, claim pattern, name token).
#: The token is what counts as the model naming that figure, or the figure's
#: name being written in the image. Used only to SELECT and to SCORE the probe;
#: nothing here reaches a prompt.
PUBLIC_FIGURES: tuple[tuple[str, str, str], ...] = (
    ("Narendra Modi", r"\bmodi\b", r"\bmodi\b"),
    ("Donald Trump", r"\btrump\b", r"\btrump\b"),
    ("Amit Shah", r"\bamit\s+shah\b", r"\bshah\b"),
    ("Joe Biden", r"\bbiden\b", r"\bbiden\b"),
    ("Mamata Banerjee", r"\bmamata\b", r"\bmamata\b|\bbanerjee\b"),
    ("Arvind Kejriwal", r"\bkejriwal\b", r"\bkejriwal\b"),
    ("Rahul Gandhi", r"\brahul\s+gandhi\b", r"\brahul\b"),
    ("Sonia Gandhi", r"\bsonia\s+gandhi\b", r"\bsonia\b"),
    ("Kamala Harris", r"\bkamala\b", r"\bkamala\b|\bharris\b"),
    ("Vladimir Putin", r"\bputin\b", r"\bputin\b"),
    ("Priyanka Gandhi", r"\bpriyanka\s+gandhi\b", r"\bpriyanka\b"),
    ("Manmohan Singh", r"\bmanmohan\b", r"\bmanmohan\b"),
    ("Yogi Adityanath", r"\badityanath\b", r"\badityanath\b|\byogi\b"),
    ("Jawaharlal Nehru", r"\bnehru\b", r"\bnehru\b"),
    ("Barack Obama", r"\bobama\b", r"\bobama\b"),
    ("Bernie Sanders", r"\bbernie\b", r"\bbernie\b|\bsanders\b"),
    ("Amitabh Bachchan", r"\bbachchan\b", r"\bbachchan\b"),
    ("Imran Khan", r"\bimran\s+khan\b", r"\bimran\b"),
    ("Nancy Pelosi", r"\bpelosi\b", r"\bpelosi\b"),
    ("Mahatma Gandhi", r"\bmahatma\b", r"\bmahatma\b|\bgandhi\b"),
    ("Xi Jinping", r"\bxi\s+jinping\b", r"\bjinping\b"),
    ("Anthony Fauci", r"\bfauci\b", r"\bfauci\b"),
    ("Mike Pence", r"\bpence\b", r"\bpence\b"),
    ("Boris Johnson", r"\bboris\s+johnson\b", r"\bboris\b|\bjohnson\b"),
    ("Hillary Clinton", r"\bhillary\b", r"\bhillary\b|\bclinton\b"),
    ("Elon Musk", r"\bmusk\b", r"\bmusk\b"),
    ("Bill Gates", r"\bbill\s+gates\b", r"\bgates\b"),
)
#: Claims that say an image or video shows the figure, so the figure is likely pictured.
PICTURED = re.compile(r"\b(photos?|photographs?|images?|pictures?|pics?|videos?|visuals?|clips?)\b", re.I)
PROBE_N = 60
#: Below this, PERSON-from-[seen] counts as near zero.
NEAR_ZERO = 0.05
#: Precision a name source needs to REPLACE a gallery: a name taken as the
#: identity with nothing behind it. A candidate that a gallery then verifies
#: needs far less (candidate_precision_needed); what must reach this bar there
#: is the precision of the names the gallery accepts.
VERIFIER_PRECISION = 0.90
#: Confidence floors at which the probe's precision is reported: fixed, not tuned.
CONFIDENCE_GRID = (0.5, 0.7, 0.9, 0.95, 0.99)
#: Gallery operating points -- false-accept over true-accept rate -- at which the
#: candidate bar is reported. Hypothetical: the gallery is not built yet.
GALLERY_RATIOS = (0.1, 0.05, 0.01, 0.001)

#: Image counts quoted in the brief for this block, reported beside the measured
#: ones so the wrong set gets corrected rather than carried.
BRIEF_FIGURES = {"verite": 914, "averimatec": 1392, "factify2": 77505}


class FatalError(RuntimeError):
    """An invariant of the whole run broke (not one image): stop, do not write."""


class GeometryError(ValueError):
    """An image the model cannot be fed at all."""


class LedgerError(RuntimeError):
    """The ledger on disk cannot be trusted; resuming over it would hide the damage."""


class LockHeld(RuntimeError):
    """Another live process is writing this ledger."""


class ModelMissing(RuntimeError):
    """Weights are not on disk, or are not the pinned bytes."""


class WorklistError(RuntimeError):
    """The processed layer and the image index disagree."""


def log(*parts: Any) -> None:
    print(*parts, flush=True)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def fit_to_cap(width: int, height: int) -> tuple[int, int]:
    """(width, height) the model is fed: Qwen2-VL's smart_resize under our fixed cap.

    Each side a multiple of 28, aspect ratio kept as closely as possible, pixel
    count within [MIN_PIXELS, MAX_PIXELS]. The arithmetic is transformers' own,
    copied so the cap is ours and cannot drift with a library default
    (tests/test_image_text.py holds the two equal).
    """
    if width < 1 or height < 1:
        raise GeometryError("empty image")
    if max(width, height) / min(width, height) > MAX_ASPECT:
        raise GeometryError(f"aspect ratio {max(width, height) / min(width, height):.0f}:1")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return w_bar, h_bar


def visual_tokens(width: int | None, height: int | None) -> int | None:
    """Merged visual tokens the model will see for an image of this size."""
    if not width or not height:
        return None
    try:
        w, h = fit_to_cap(width, height)
    except GeometryError:
        return None
    return (w // FACTOR) * (h // FACTOR)


# --------------------------------------------------------------------------
# canonical bytes, and the hash that makes "identical" checkable
# --------------------------------------------------------------------------


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def result_part(row: dict[str, Any]) -> dict[str, Any]:
    """Everything the computation produced: the row minus timings and the machine.

    This is what must be byte-identical when the same image is processed twice.
    """
    out = {k: v for k, v in row.items() if k not in ("run", "result_sha256")}
    out["provenance"] = {k: v for k, v in (row.get("provenance") or {}).items() if k != "machine"}
    return out


def result_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(result_part(row))).hexdigest()


def host_id() -> str:
    """A stable machine identifier that does not publish the hostname."""
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def _r(x: float | None, places: int = 6) -> float | None:
    return None if x is None else round(float(x), places)


# --------------------------------------------------------------------------
# work lists
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One distinct image of one dataset. ``path`` is the file read -- the
    first of its copies -- and is never written to a row; ``stratum`` and
    ``probe`` exist in memory only."""

    dataset: str
    sha256: str
    path: str
    record_ids: tuple[str, ...]
    n_paths: int
    width: int | None = None
    height: int | None = None
    stratum: str | None = None
    probe: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "sha256": self.sha256, "path": self.path,
                "record_ids": list(self.record_ids), "n_paths": self.n_paths,
                "width": self.width, "height": self.height}


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _index_row(path: str, cache: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    """The index row for ``path``, from the index of the dataset that owns the
    FILE: two VERITE rows point at a MOCHEG file."""
    owner = images.dataset_of(path)
    if owner is None:
        raise WorklistError(f"{path} is not under data/raw/<dataset>/")
    if owner not in cache:
        cache[owner] = images.load_index(owner)
    return cache[owner].get(path)


def usable_rows(dataset: str) -> Iterable[dict[str, Any]]:
    """Processed rows that are usable and carry at least one image."""
    from scripts.processed_loader import iter_rows

    for row in iter_rows(dataset):
        if row.get("usable") and row.get("image_paths"):
            yield row


def build_worklist(dataset: str, *, rows: Iterable[dict[str, Any]] | None = None,
                   cache: dict | None = None) -> tuple[list[WorkItem], dict[str, int]]:
    """Every distinct image of the dataset's usable rows, and the counts behind it.

    Usable rows only: the processed layer already excludes forbidden splits,
    and its ``image_paths`` hold only images that passed the gates at build
    time (they are gated again when read). Order is by sha256 -- label-free and
    pseudo-random, so a pilot's first N are not all one class -- except for
    Fakeddit, whose order is the nested stratified sample.
    """
    cache = {} if cache is None else cache
    groups: dict[str, dict[str, Any]] = {}
    n_rows = 0
    for row in (usable_rows(dataset) if rows is None else rows):
        n_rows += 1
        for path in row["image_paths"]:
            entry = _index_row(path, cache)
            if not entry or not entry.get("sha256"):
                raise WorklistError(
                    f"{dataset}: a usable record's image is missing from the image index "
                    f"({path}); rebuild it with scripts/images.py index before running")
            g = groups.setdefault(entry["sha256"], {"paths": set(), "records": set(),
                                                    "labels": Counter(),
                                                    "w": entry.get("width"),
                                                    "h": entry.get("height")})
            g["paths"].add(path)
            g["records"].add(str(row["record_id"]))
            g["labels"][str(row.get("label"))] += 1
    items = []
    for digest, g in groups.items():
        stratum = sorted(g["labels"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        items.append(WorkItem(dataset, digest, min(g["paths"]), tuple(sorted(g["records"])),
                              len(g["paths"]), g["w"], g["h"], stratum=stratum))
    stats = {"usable_rows_with_images": n_rows,
             "image_paths": sum(len(g["paths"]) for g in groups.values()),
             "distinct_images": len(items)}
    if dataset == "fakeddit":
        return nested_stratified(items), stats
    return sorted(items, key=lambda it: it.sha256), stats


def _priority(seed: int, digest: str) -> str:
    return hashlib.sha256(f"{seed}:{digest}".encode("ascii")).hexdigest()


def nested_stratified(items: Sequence[WorkItem], seed: int = SEED) -> list[WorkItem]:
    """An order in which every prefix is a proportional stratified sample.

    Classes are interleaved by largest deficit -- the next item comes from the
    class furthest below its share -- so each class's count in any prefix of
    length N stays within one of N times its share, and within a class items
    follow a seeded hash priority. A longer prefix always contains a shorter
    one, so ``--fakeddit-n`` can grow without wasting finished work.
    (fetchlib.stratified_indices is not reused: its random draws are not
    nested across sample sizes.)
    """
    groups: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        groups[str(item.stratum)].append(item)
    for group in groups.values():
        group.sort(key=lambda it: _priority(seed, it.sha256))
    names = sorted(groups)
    total = len(items)
    share = {c: len(groups[c]) / total for c in names}
    taken = dict.fromkeys(names, 0)
    out: list[WorkItem] = []
    for n in range(1, total + 1):
        best = max((c for c in names if taken[c] < len(groups[c])),
                   key=lambda c: (n * share[c] - taken[c], -names.index(c)))
        out.append(groups[best][taken[best]])
        taken[best] += 1
    return out


def load_worklist_file(path: Path) -> dict[str, list[WorkItem]]:
    """Items from a JSONL file of ``WorkItem.as_json()`` rows (tests, targeted reruns)."""
    out: dict[str, list[WorkItem]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["dataset"]].append(WorkItem(d["dataset"], d["sha256"], d["path"],
                                              tuple(d["record_ids"]), int(d["n_paths"]),
                                              d.get("width"), d.get("height")))
    return dict(out)


def probe_items(n: int, factify2: Sequence[WorkItem], *,
                rows: Iterable[dict[str, Any]] | None = None,
                cache: dict | None = None) -> list[WorkItem]:
    """Factify2 TRAIN images whose claim names exactly one public figure and
    says an image shows them: the recognition probe.

    The items are the full work list's own (same record_ids), so a probe row is
    exactly the row the Factify2 run would write, and is reused by it. Figures
    are taken round-robin, most-claimed first, so the probe is not all one face.
    """
    if n <= 0:
        return []
    by_sha = {it.sha256: it for it in factify2}
    cache = {} if cache is None else cache
    patterns =[(name, re.compile(claim, re.I)) for name, claim, _token in PUBLIC_FIGURES]
    found: dict[str, dict[str, WorkItem]] = defaultdict(dict)
    for row in (usable_rows("factify2") if rows is None else rows):
        if row.get("split") != "train":
            continue
        text = row.get("text") or ""
        hits = [name for name, pattern in patterns if pattern.search(text)]
        if len(hits) != 1 or not PICTURED.search(text):
            continue
        entry = _index_row(row["image_paths"][0], cache)  # the claim image
        item = by_sha.get((entry or {}).get("sha256"))
        if item is not None:
            found[hits[0]].setdefault(item.sha256, replace(item, probe=hits[0]))
    queues = {name: sorted(group.values(), key=lambda it: _priority(SEED, it.sha256))
              for name, group in found.items()}
    order = sorted(queues, key=lambda name: (-len(queues[name]), name))
    out: list[WorkItem] = []
    seen: set[str] = set()
    while len(out) < n and any(queues.values()):
        for name in order:
            if queues[name] and len(out) < n:
                item = queues[name].pop(0)
                if item.sha256 not in seen:
                    seen.add(item.sha256)
                    out.append(item)
    return out


# --------------------------------------------------------------------------
# reading an image, and the gates applied to the bytes read now
# --------------------------------------------------------------------------


def read_bytes(path: str) -> bytes:
    """A file, or ``archive.zip::member`` as the image index names zip members."""
    if "::" in path:
        archive, member = path.split("::", 1)
        with zipfile.ZipFile(_abs(archive)) as zf:
            return zf.read(member)
    return _abs(path).read_bytes()


def gate(item: WorkItem, body: bytes) -> tuple[str | None, str | None]:
    """(status, reason) for an image that must not reach the model, else (None, None).

    The same three gates every count in the project goes through
    (images.status), applied to these bytes: a registry entry added after the
    processed layer was built still stops the image here.
    """
    examined = images.examine_bytes(body)
    if examined["sha256"] != item.sha256:
        return "failed", "sha256_mismatch"
    status = images.status({**examined, "path": item.path})
    if status == images.UNDECODABLE:
        return "failed", "undecodable"
    if status != images.USABLE:
        return "skipped", status
    return None, None


def prepare(body: bytes):
    """Decoded RGB image (EXIF-oriented, alpha flattened onto white, first frame)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(body)) as img:
        info = {"format": img.format, "width": img.width, "height": img.height,
                "mode": img.mode, "bytes": len(body),
                "frames": int(getattr(img, "n_frames", 1) or 1), "exif_rotated": None}
        try:
            orientation = img.getexif().get(0x0112)
            oriented = ImageOps.exif_transpose(img)
            info["exif_rotated"] = orientation not in (None, 1)
        except Exception:  # noqa: BLE001 -- a broken EXIF block is not a broken image
            oriented = img.copy()
        if oriented.mode in ("RGBA", "LA", "PA") or (oriented.mode == "P"
                                                     and "transparency" in oriented.info):
            rgba = oriented.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = oriented.convert("RGB")
    return rgb, info


# --------------------------------------------------------------------------
# generations: what a backend returns, and the gates on what it said
# --------------------------------------------------------------------------


@dataclass
class Generation:
    text: str
    logprobs: list[float]   # chosen token's log-probability, per generated token
    ends: list[int]         # character offset in ``text`` where each token ends
    finish: str             # "eos" | "length"
    n_tokens: int
    replaced: tuple[int, ...] = ()      # steps whose token the skeleton put in place of the model's
    events: tuple[str, ...] = ()        # what those replacements did (Skeleton.events)
    forced_lines: tuple[str, ...] = ()  # categories written from the first replacement on


def _spans(gen: Generation) -> Iterable[tuple[float, int, int]]:
    """(log-probability, start, end) of every token the model chose -- never of
    one the skeleton put in its place."""
    prev, replaced = 0, set(gen.replaced)
    for i, (lp, end) in enumerate(zip(gen.logprobs, gen.ends)):
        if i not in replaced:
            yield lp, prev, end
        prev = end


def text_confidence(gen: Generation) -> float | None:
    """Geometric-mean probability of the tokens that produced text."""
    lps = [lp for lp, s, e in _spans(gen) if e > s]
    return _r(math.exp(sum(lps) / len(lps))) if lps else None


def span_confidence(gen: Generation, start: int, end: int) -> float | None:
    """Probability of the least certain token overlapping ``text[start:end]``."""
    lps = [lp for lp, s, e in _spans(gen) if e > start and s < end and e > s]
    return _r(math.exp(min(lps))) if lps else None


def check_greedy(emitted: Sequence[int], argmax: Sequence[int],
                 chosen: Sequence[int] | None = None) -> list[int]:
    """Raise unless every emitted token is the one chosen at its step -- by
    default the step's argmax: greedy, verified. With ``chosen`` (the entity
    skeleton's picks), return the steps whose choice was not the argmax: the
    replacements, which the caller holds against the skeleton's own record."""
    chosen = argmax if chosen is None else chosen
    if list(emitted) != list(chosen) or len(chosen) != len(argmax):
        raise FatalError(
            f"decoding was not greedy: {len(emitted)} tokens emitted, {len(chosen)} steps "
            f"recorded, first difference at {next((i for i, (a, b) in enumerate(zip(emitted, chosen)) if a != b), min(len(emitted), len(chosen)))}")
    return [i for i, (a, c) in enumerate(zip(argmax, chosen)) if a != c]


REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i can not|"
                     r"i am unable|i'm unable|as an ai)", re.I)


def repetitive(text: str, times: int = 4) -> bool:
    """True if some run of 1-6 words repeats ``times`` times back to back."""
    words = text.split()
    for n in range(1, 7):
        for start in range(0, max(0, len(words) - n * times + 1)):
            gram = words[start:start + n]
            if all(words[start + k * n:start + (k + 1) * n] == gram for k in range(1, times)):
                return True
    return False


def description_block(gen: Generation) -> dict[str, Any]:
    text = gen.text.strip()
    error = "empty" if not text else "refusal" if REFUSAL.match(text) else None
    flags = (["truncated"] if gen.finish == "length" else []) + (
        ["repetition"] if text and repetitive(text) else [])
    return {"text": text, "error": error, "flags": flags, "finish": gen.finish,
            "n_tokens": gen.n_tokens, "confidence": text_confidence(gen)}


# --------------------------------------------------------------------------
# the entity skeleton: the five lines, held while they are generated
# --------------------------------------------------------------------------


class Skeleton:
    """Holds an entity answer to its five lines; the model writes every value.

    ``choose`` is given each step's greedy token and returns the token to emit:
    the model's own wherever the answer keeps its form, otherwise the next
    piece of the form --

    - a newline where the model would end before DATE's line
      (``continued_after:<CAT>``; on the first pilot it ended 63 of 199
      answers after PERSON, closing "[seen]" with "]" instead of "]\\n");
    - the line's header where it wrote something else (``header_forced:<CAT>``);
    - the end where it would go on after DATE's line (``ended_after:DATE``).

    Values are never touched. Every replacement is recorded, and every line
    from the first one that put something in is forced: the model would not
    have written it unprompted. Lines before that are the unconstrained greedy
    answer, token for token (every token there is its step's argmax, and
    check_greedy verifies it). Pure Python, so it is tested without a model;
    ``decode`` and ``encode`` are the tokenizer's.
    """

    def __init__(self, decode: Callable[[int], str], encode: Callable[[str], list[int]],
                 newline: int, end: int, eos: Iterable[int]):
        self.decode, self.encode = decode, encode
        self.newline, self.end, self.eos = newline, end, frozenset(eos)
        if end not in self.eos:
            raise FatalError(f"the skeleton's end token {end} does not end generation")
        self.line = 0        # index into HEADERS
        self.pos = 0         # characters of the line's header written so far
        self.value = False   # the header is written: the model is writing the value
        self.done = False    # DATE's line has ended with a newline: only the end may follow
        self.steps = 0
        self.replaced: list[int] = []
        self.events: list[str] = []
        self._first_forced: int | None = None

    @property
    def complete(self) -> bool:
        """Every header is written and DATE's line is begun."""
        return self.done or (self.value and self.line == len(HEADERS) - 1)

    @property
    def forced_lines(self) -> tuple[str, ...]:
        return () if self._first_forced is None else CATEGORIES[self._first_forced:]

    def _after(self, text: str) -> tuple[int, int, bool, bool] | None:
        """The state once ``text`` is written, or None if it breaks the form.
        Whitespace may come before a header; in a value, only "\\n" matters."""
        line, pos, value, done = self.line, self.pos, self.value, self.done
        for ch in text:
            if done:
                return None
            if value:
                if ch == "\n":
                    if line == len(HEADERS) - 1:
                        done = True
                    else:
                        line, pos, value = line + 1, 0, False
            elif pos == 0 and ch.isspace():
                continue
            elif ch == HEADERS[line][pos]:
                pos += 1
                value = pos == len(HEADERS[line])
            else:
                return None
        return line, pos, value, done

    def choose(self, want: int) -> int:
        step = self.steps
        self.steps += 1
        if want in self.eos:
            if self.complete:
                return want
            if self.value:  # the model would end before DATE's line
                pick, kind, line = self.newline, "continued_after", self.line
                first: int | None = self.line + 1
            else:
                pick, kind, line = self.encode(HEADERS[self.line][self.pos:])[0], "header_forced", self.line
                first = self.line
        else:
            state = self._after(self.decode(want))
            if state is not None:
                self.line, self.pos, self.value, self.done = state
                return want
            if self.complete:
                pick, kind, line, first = self.end, "ended_after", len(HEADERS) - 1, None
            elif self.value:  # a newline, then something other than the next header
                pick, kind, line, first = self.newline, "header_forced", self.line + 1, self.line + 1
            else:
                pick, kind, line = self.encode(HEADERS[self.line][self.pos:])[0], "header_forced", self.line
                first = self.line
        if pick not in self.eos:
            state = self._after(self.decode(pick))
            if state is None:
                raise FatalError(f"the skeleton's own token {pick} breaks the form it holds")
            self.line, self.pos, self.value, self.done = state
        self.replaced.append(step)
        event = f"{kind}:{CATEGORIES[line]}"
        if event not in self.events:
            self.events.append(event)
        if first is not None and (self._first_forced is None or first < self._first_forced):
            self._first_forced = first
        return pick


# --------------------------------------------------------------------------
# entity parsing: pure, re-runnable offline from the stored raw text
# --------------------------------------------------------------------------

#: A category line; tolerates bullets, numbering and markdown bold ("1. **PERSON:** ...").
_LINE = re.compile(r"^[\s*#>\-•\d.)]*(PERSON|ORG|GPE|EVENT|DATE)\b\s*\**\s*[:：]\s*(.*)$", re.I)
_TAG = re.compile(r"[\[(]\s*(text|seen)\b[^\])]*[\])]", re.I)
#: A trailing bracketed note that is not a basis tag: "Shri. A. Pawar [Sakal]".
_NOTE = re.compile(r"\s*\[[^\]]*\]?\s*$")
_NONE_WORDS = {"none", "n/a", "na", "nil", "nothing", "no", "-", "--", "none visible",
               "not applicable", "not visible", "unknown date", ""}
_UNIDENTIFIED = re.compile(r"\b(unidentified|unidentifiable|unknown)\b|cannot (be )?identif|"
                           r"can'?t identif|not identifiable", re.I)
_SENTENCE = re.compile(r"^(no|not|there (is|are)|none)\b", re.I)
_TRIM = " \t\"'`*_,;:{}"
#: The model separates names with commas as often as with " ; ". Dates and
#: events keep theirs ("March 4, 2020" is one date).
COMMA_LISTS = frozenset({"PERSON", "ORG", "GPE"})


def _trimmed(text: str, start: int, end: int) -> tuple[str, int, int, bool]:
    """(name, start, end, had_note) with punctuation and a trailing bracketed note removed."""
    had_note = False
    for _ in range(2):
        while start < end and text[start] in _TRIM:
            start += 1
        while end > start and text[end - 1] in _TRIM:
            end -= 1
        note = _NOTE.search(text, start, end)
        if note and note.start() > start:
            had_note, end = True, note.start()
    # a sentence's full stop goes; an abbreviation's ("U.S.", "D.C.") stays
    if end > start and text[end - 1] == "." and "." not in text[start:end - 1]:
        end -= 1
    return text[start:end], start, end, had_note


def _entries(value: str, base: int, commas: bool) -> tuple[list[tuple[str, str, int, int]], bool]:
    """(name, basis, start, end) per entry -- offsets into the raw text -- and
    whether any non-tag note was removed. A tag belongs to the name just before
    it: in "A, B [seen]" only B is tagged, and A's basis is ``unstated``."""
    out, noted = [], False
    for part in re.finditer(r"[^;]+", value):
        seg, cursor, pieces = part.group(0), 0, []
        for tag in _TAG.finditer(seg):
            pieces.append((cursor, tag.start(), tag.group(1).lower()))
            cursor = tag.end()
        pieces.append((cursor, len(seg), "unstated"))
        for a, b, basis in pieces:
            spans = ([(m.start() + a, m.end() + a) for m in re.finditer(r"[^,]+", seg[a:b])]
                     if commas else [(a, b)])
            found = []
            for ca, cb in spans:
                name, s, e, note = _trimmed(seg, ca, cb)
                noted |= note
                if name:
                    found.append((name, s, e))
            for i, (name, s, e) in enumerate(found):
                out.append((name, basis if i == len(found) - 1 else "unstated",
                            base + part.start() + s, base + part.start() + e))
    return out, noted


def parse_entities(raw: str) -> dict[str, Any]:
    """The five category lines -> {CATEGORY: {status, unidentified, items}}, flags, error.

    status: ``named`` (at least one name), ``unidentified`` (UNIDENTIFIED and no
    name), ``none``, or ``missing`` (no such line: the model broke the format
    for that category only). Items keep ``span`` -- offsets into ``raw`` -- for
    the confidence of their tokens. Lines are split on "\\n" only: the line the
    skeleton holds.
    """
    cats: dict[str, dict[str, Any]] = {c: {"status": "missing", "unidentified": False, "items": []}
                                       for c in CATEGORIES}
    flags: list[str] = []
    seen_lines: set[str] = set()
    offset = 0
    for line in raw.split("\n"):
        start, offset = offset, offset + len(line) + 1
        m = _LINE.match(line.rstrip("\r"))
        if not m:
            continue
        cat = m.group(1).upper()
        if cat in seen_lines:
            flags.append(f"duplicate_line:{cat}")
            continue
        seen_lines.add(cat)
        items, unidentified, empty = [], False, not m.group(2).strip(_TRIM)
        entries, noted = _entries(m.group(2), start + m.start(2), cat in COMMA_LISTS)
        if noted:
            flags.append(f"note_removed:{cat}")
        for name, basis, s, e in entries:
            word = name.casefold()
            if word in _NONE_WORDS or (_SENTENCE.match(name) and len(name.split()) > 1):
                continue
            if _UNIDENTIFIED.search(name):
                unidentified = True
                continue
            if len(name) > 60 or len(name.split()) > 8:
                flags.append(f"implausible_entry:{cat}")
                continue
            if word not in {i["name"].casefold() for i in items}:
                items.append({"name": name, "basis": basis, "span": [s, e]})
        if empty:
            flags.append(f"empty_value:{cat}")
        cats[cat] = {"status": "named" if items else "unidentified" if unidentified else "none",
                     "unidentified": unidentified, "items": items}
    parsed = sum(1 for c in CATEGORIES if cats[c]["status"] != "missing")
    if not parsed and _TAG.sub("", raw).strip(_TRIM + ".").casefold() in _NONE_WORDS - {""}:
        # The whole answer is "NONE" (seen on an image with nothing to name):
        # nothing in any category -- said once instead of five times. Flagged,
        # because whether people were visible but unnamed is then unknown.
        for cat in CATEGORIES:
            cats[cat] = {"status": "none", "unidentified": False, "items": []}
        return {"categories": cats, "flags": flags + ["answered_none_for_all"], "error": None}
    for cat in CATEGORIES:
        if cats[cat]["status"] == "missing":
            flags.append(f"category_missing:{cat}")
    return {"categories": cats, "flags": flags, "error": None if parsed else "unparseable"}


#: A category header anywhere in a line: two on one line is a collapsed answer.
_HEADER_IN_LINE = re.compile(r"\b(?:PERSON|ORG|GPE|EVENT|DATE)\b\s*\**\s*[:：]", re.I)
#: Forms an entity row may be written with. Anything else means the skeleton
#: did not hold, and the run stops (process).
ENTITY_FORMS_WRITTEN = ("complete", "truncated", "failed")


def entity_format(block: dict[str, Any] | None) -> str:
    """How an entity answer kept, or lost, its five lines.

    ``complete``; ``truncated`` (max_new_tokens cut it); ``stopped_after:<CAT>``
    (it ended after CAT's line: stopping early); ``collapsed`` (several headers
    on one line), ``lines_missing``, ``none_for_all`` or ``unparseable``
    (another format); ``failed`` (no answer: the call itself errored).
    """
    if block is None or block.get("raw") is None:
        return "failed"
    if "answered_none_for_all" in block["flags"]:
        return "none_for_all"
    present = [c for c in CATEGORIES if block[c]["status"] != "missing"]
    if len(present) == len(CATEGORIES):
        return "complete"
    if block["finish"] == "length":
        return "truncated"
    if any(len(_HEADER_IN_LINE.findall(line)) > 1 for line in block["raw"].split("\n")):
        return "collapsed"
    if not present:
        return "unparseable"
    if present == list(CATEGORIES[:len(present)]):
        return f"stopped_after:{present[-1]}"
    return "lines_missing"


def _normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _ocr_words(name: str, ocr_text: str) -> tuple[int, int]:
    """(words of ``name`` found in the OCR text, words of two characters or
    more) -- found as a word, or for longer words inside a run OCR glued together."""
    words = [w for w in _normalise(name).split() if len(w) >= 2]
    hay = f" {_normalise(ocr_text)} "
    glued = hay.replace(" ", "")
    return sum(1 for w in words if f" {w} " in hay or (len(w) >= 4 and w in glued)), len(words)


def name_in_ocr(name: str, ocr_text: str) -> bool:
    """Every word of ``name`` (two characters or more) is in the OCR text --
    as a word, or for longer words inside a run OCR glued together."""
    found, total = _ocr_words(name, ocr_text)
    return total > 0 and found == total


def entities_block(gen: Generation, ocr_text: str | None) -> dict[str, Any]:
    """The parsed answer. A line the skeleton forced is marked ``forced`` on the
    category and on every item in it, so the mark travels with the entity."""
    parsed = parse_entities(gen.text)
    forced = set(gen.forced_lines)
    out: dict[str, Any] = {"raw": gen.text}
    for cat in CATEGORIES:
        c, line_forced = parsed["categories"][cat], cat in forced
        out[cat] = {"status": c["status"], "unidentified": c["unidentified"], "forced": line_forced,
                    "items": [
            {"name": it["name"], "basis": it["basis"],
             "confidence": span_confidence(gen, *it["span"]),
             "in_ocr": None if ocr_text is None else name_in_ocr(it["name"], ocr_text),
             "forced": line_forced}
            for it in c["items"]]}
    out.update(error=parsed["error"],
               flags=parsed["flags"] + list(gen.events)
               + (["truncated"] if gen.finish == "length" else []),
               finish=gen.finish, n_tokens=gen.n_tokens, confidence=text_confidence(gen))
    return out


#: What a caller of entities_of must choose for entities on forced lines.
FORCED_CHOICES = ("exclude", "mark")


def entities_of(row: dict[str, Any], *, forced: str) -> Iterator[dict[str, Any]]:
    """The named entities of one row: the only sanctioned way to read them out.

    A line the skeleton forced is weaker evidence than one the model wrote
    unprompted, so the caller must say what to do with it -- there is no
    default. ``forced="exclude"`` drops every entity on a forced line;
    ``forced="mark"`` keeps them. Either way each entity carries ``forced``, so
    the distinction survives the trip into a training set. A row whose lines
    are not marked (written under image_text/1) cannot tell the two apart and
    is refused. Splits and evaluation-only corpora (VERITE, AVerImaTeC) are the
    training loader's to enforce, as for every other supervision
    (processed_loader).
    """
    from scripts.processed_loader import SupervisionError

    if forced not in FORCED_CHOICES:
        raise ValueError(f"forced must be one of {FORCED_CHOICES}, not {forced!r}")
    entities = row.get("entities")
    if row.get("status") not in ("ok", "partial") or not entities or entities.get("error"):
        return iter(())
    if any("forced" not in entities[c] for c in CATEGORIES):
        raise SupervisionError(
            f"{row['dataset']}/{row['sha256'][:12]}: its entity lines are not marked forced or "
            "free (written under image_text/1), so a forced line cannot be told from a free one")
    return ({"dataset": row["dataset"], "sha256": row["sha256"],
             "record_ids": list(row["record_ids"]), "category": cat, **item}
            for cat in CATEGORIES if not (forced == "exclude" and entities[cat]["forced"])
            for item in entities[cat]["items"])


def ocr_block(regions: list[dict[str, Any]], size: tuple[int, int]) -> dict[str, Any]:
    chars = sum(len(r["text"]) for r in regions)
    mean = sum(r["confidence"] * len(r["text"]) for r in regions) / chars if chars else None
    return {"text": " ".join(r["text"] for r in regions), "regions": regions,
            "mean_confidence": _r(mean), "input": {"width": size[0], "height": size[1]},
            "error": None}


def _failed_component(name: str, error: str) -> dict[str, Any]:
    if name == "ocr":
        return {"text": None, "regions": None, "mean_confidence": None, "input": None,
                "error": error}
    if name == "description":
        return {"text": None, "error": error, "flags": [], "finish": None, "n_tokens": 0,
                "confidence": None}
    block: dict[str, Any] = {"raw": None}
    for cat in CATEGORIES:
        block[cat] = {"status": "missing", "unidentified": False, "forced": False, "items": []}
    block.update(error=error, flags=[], finish=None, n_tokens=0, confidence=None)
    return block


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


def _messages(which: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[which]}]}]


def prompt_config(rendered: dict[str, str]) -> dict[str, Any]:
    """Hashes of the prompts as written and as the chat template renders them."""
    return {"prompt_sha256": hashlib.sha256(canonical(rendered)).hexdigest(),
            "by_call": {which: sha256_text(text) for which, text in sorted(rendered.items())},
            "system_sha256": sha256_text(SYSTEM_PROMPT),
            "texts": {"system": SYSTEM_PROMPT, **PROMPTS}}


def decoding_config(eos: list[int]) -> dict[str, Any]:
    return {"strategy": "greedy", "do_sample": False, "num_beams": 1, "temperature": 0.0,
            "repetition_penalty": 1.0, "max_new_tokens": dict(MAX_NEW_TOKENS),
            "batch_size": 1, "eos_token_id": list(eos), "argmax_verified": True,
            "entities_skeleton": {
                "headers": list(HEADERS), "order": "fixed", "end": "only after the last line",
                "replaced_tokens": "recorded in the entity flags; left out of every confidence",
                "forced_lines": "every line from the first replacement that puts something in"}}


def deterministic_torch(torch) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


SOFTWARE = ("torch", "transformers", "bitsandbytes", "easyocr", "pillow",
            "opencv-python-headless", "numpy")


def software_versions(names: Sequence[str] = SOFTWARE) -> dict[str, str | None]:
    """Versions of what computes the outputs. Part of the config: a different
    version is a different directory, never rows mixed into the old one."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


class QwenVLM:
    """Qwen2.5-VL-7B-Instruct, NF4 via bitsandbytes, greedy, batch size 1."""

    kind = "qwen"

    def __init__(self, cache_dir: Path = HF_CACHE, revision: str = MODEL_REVISION):
        self.cache_dir, self.revision = cache_dir, revision
        self.processor = None
        self.model = None
        self.rendered: dict[str, str] = {}
        self.eos: list[int] = []
        self.pad: int | None = None
        self.newline: int | None = None
        self.end: int | None = None

    def snapshot(self) -> Path:
        snap = (self.cache_dir / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
                / self.revision)
        if not (snap / "config.json").is_file():
            raise ModelMissing(f"{MODEL_ID}@{self.revision[:12]} is not in {self.cache_dir}; "
                               "run: python scripts/image_text.py fetch-models")
        return snap

    def prepare(self) -> None:
        """Processor and rendered prompts only -- what the config hashes. The
        weights load later, and only if there is work to do."""
        from transformers import AutoProcessor

        snap = self.snapshot()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=self.revision, cache_dir=str(self.cache_dir), local_files_only=True)
        self.rendered = {which: self.processor.apply_chat_template(
            _messages(which), tokenize=False, add_generation_prompt=True) for which in PROMPTS}
        hub = json.loads((snap / "generation_config.json").read_text(encoding="utf-8"))
        eos = hub.get("eos_token_id")
        self.eos = sorted(eos if isinstance(eos, list) else [eos])
        self.pad = hub.get("pad_token_id")
        # What the skeleton puts in, verified on this tokenizer, not assumed.
        tok = self.processor.tokenizer
        newline = tok.encode("\n", add_special_tokens=False)
        end = tok.convert_tokens_to_ids("<|im_end|>")
        if len(newline) != 1 or tok.decode(newline) != "\n" or end not in self.eos:
            raise FatalError("the tokenizer gives the entity skeleton no single newline token, "
                             "or its end-of-turn token does not end generation")
        self.newline, self.end = newline[0], end

    def config(self) -> dict[str, Any]:
        return {"id": MODEL_ID, "revision": self.revision, "licence": MODEL_LICENCE,
                "dtype": "bfloat16", "attn_implementation": "sdpa",
                "quantization": dict(QUANTIZATION), "decoding": decoding_config(self.eos),
                "prompts": prompt_config(self.rendered)}

    def load(self) -> None:
        import torch
        from transformers import (BitsAndBytesConfig, GenerationConfig, LogitsProcessor,
                                  Qwen2_5_VLForConditionalGeneration)

        if not torch.cuda.is_available():
            raise ModelMissing("the Qwen backend needs CUDA; none is visible")
        deterministic_torch(torch)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=self.revision, cache_dir=str(self.cache_dir), local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval()
        self._check_quantization()
        # transformers fills every unset generation field from the Hub's
        # generation_config (which samples); replace it so nothing is inherited.
        base = dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
                    eos_token_id=self.eos, pad_token_id=self.pad)
        self.model.generation_config = GenerationConfig(**base)
        self.gen = {which: GenerationConfig(**base, max_new_tokens=MAX_NEW_TOKENS[which])
                    for which in PROMPTS}

        class Recorder(LogitsProcessor):
            """Records each step's argmax and its log-probability, changing
            nothing -- unless it holds a skeleton (the entity call). Then the
            skeleton may replace the step's token: every other token is pushed
            to -inf, so generate's own argmax emits the replacement, and the
            argmax, the pick and the pick's log-probability are all recorded."""

            def __init__(self, skeleton: Skeleton | None = None):
                self.skeleton = skeleton
                self.ids, self.lps, self.chosen = [], [], []

            def __call__(self, input_ids, scores):
                if self.skeleton is None:
                    top = torch.log_softmax(scores.float(), dim=-1).max(dim=-1)
                    self.lps.append(top.values)
                    self.ids.append(top.indices)
                    return scores
                want = int(scores[0].argmax())  # what generate would emit
                pick = self.skeleton.choose(want)
                self.ids.append(want)
                self.chosen.append(pick)
                self.lps.append(float(torch.log_softmax(scores[0].float(), dim=-1)[pick]))
                if pick == want:
                    return scores
                steered = torch.full_like(scores, float("-inf"))
                steered[0, pick] = 0.0
                return steered

        self._recorder = Recorder
        torch.cuda.reset_peak_memory_stats()

    def _check_quantization(self) -> None:
        """The loaded weights must be what the config records -- NF4, double
        quantized, every Linear 4-bit except the declared ones -- checked on the
        modules themselves, not on the arguments that asked for them."""
        import torch

        declared = QUANTIZATION["not_quantized"]

        def is_declared(name: str) -> bool:
            return any(name == d or name.endswith("." + d) for d in declared)

        plain = [name for name, mod in self.model.named_modules() if type(mod) is torch.nn.Linear]
        four_bit = [mod for _, mod in self.model.named_modules()
                    if type(mod).__name__ == "Linear4bit"]
        problems = []
        undeclared = sorted(n for n in plain if not is_declared(n))
        if undeclared:
            problems.append(f"{len(undeclared)} Linear layers are not 4-bit (e.g. {undeclared[:3]})")
        absent = [d for d in declared if not any(n == d or n.endswith("." + d) for n in plain)]
        if absent:
            problems.append(f"declared bf16 modules are quantized or missing: {absent}")
        if not four_bit:
            problems.append("no 4-bit layers at all")
        else:
            weight = four_bit[0].weight
            if getattr(weight, "quant_type", None) != "nf4":
                problems.append(f"quant_type is {getattr(weight, 'quant_type', None)!r}, not 'nf4'")
            if not getattr(getattr(weight, "quant_state", None), "nested", False):
                problems.append("the 4-bit weights are not double-quantized")
        if problems:
            raise FatalError("the loaded model is not the recorded configuration: "
                             + "; ".join(problems))

    def generate(self, image, which: str) -> Generation:
        import torch
        from transformers import LogitsProcessorList

        inputs = self.processor(text=[self.rendered[which]], images=[image],
                                return_tensors="pt", do_resize=False)
        grid = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
        want = [1, image.height // 14, image.width // 14]
        if grid != want:
            raise FatalError(f"the processor fed grid {grid}, not {want}: the recorded "
                             "resolution would be false")
        inputs = inputs.to(self.model.device)
        tok = self.processor.tokenizer
        skeleton = (Skeleton(lambda i: tok.decode([i]),
                             lambda s: tok.encode(s, add_special_tokens=False),
                             self.newline, self.end, self.eos)
                    if which == "entities" else None)
        recorder = self._recorder(skeleton)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=self.gen[which],
                                      logits_processor=LogitsProcessorList([recorder]))
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        if skeleton is None:
            chosen = torch.stack(recorder.ids).view(-1).tolist() if recorder.ids else []
            lps = torch.stack(recorder.lps).view(-1).tolist() if recorder.lps else []
            check_greedy(new, chosen)
            replaced: list[int] = []
        else:
            lps = list(recorder.lps)
            replaced = check_greedy(new, recorder.ids, recorder.chosen)
            if replaced != skeleton.replaced:
                raise FatalError(f"generate emitted replacements at {replaced}, the skeleton made "
                                 f"them at {skeleton.replaced}")
        text = tok.decode(new, skip_special_tokens=True)
        ends, last = [], 0
        for i in range(1, len(new) + 1):
            last = max(last, len(tok.decode(new[:i], skip_special_tokens=True)))
            ends.append(min(last, len(text)))
        finish = "eos" if new and new[-1] in self.eos else "length"
        if skeleton is None:
            return Generation(text, lps, ends, finish, len(new))
        if finish == "eos" and not skeleton.complete:
            raise FatalError("the entity answer ended before its five lines: the skeleton did not hold")
        return Generation(text, lps, ends, finish, len(new), tuple(replaced),
                          tuple(skeleton.events), skeleton.forced_lines)

    def recover_oom(self) -> None:
        import torch

        torch.cuda.empty_cache()

    def memory(self) -> dict[str, int] | None:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {"allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved(),
                "device_free": free, "device_total": total}

    def device_info(self) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"gpu": None, "gpu_capability": None, "gpu_memory_gb": None,
                    "cuda": torch.version.cuda, "cudnn": None}
        props = torch.cuda.get_device_properties(0)
        return {"gpu": props.name, "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def verify_easyocr_weights(directory: Path = EASYOCR_DIR) -> dict[str, str]:
    from scripts.manifest import sha256

    out = {}
    for name, digest in EASYOCR_WEIGHTS.items():
        path = directory / name
        if not path.is_file():
            raise ModelMissing(f"EasyOCR weight {name} is not in {directory}; "
                               "run: python scripts/image_text.py fetch-models")
        got = sha256(path)
        if got != digest:
            raise ModelMissing(f"{path.name} is not the pinned EasyOCR weight "
                               f"({got[:12]} != {digest[:12]})")
        out[name] = got
    return out


class EasyOcrBackend:
    """EasyOCR, separate from the VLM: per-region text, confidence and polygon."""

    kind = "easyocr"

    def __init__(self, device: str = "cuda", directory: Path = EASYOCR_DIR):
        self.device, self.directory = device, directory
        self.reader = None
        self.weights: dict[str, str] = {}

    def prepare(self) -> None:
        self.weights = verify_easyocr_weights(self.directory)

    def config(self) -> dict[str, Any]:
        return {"engine": "easyocr", "version": software_versions(["easyocr"])["easyocr"],
                "langs": list(OCR_LANGS), "max_side": OCR_MAX_SIDE, "canvas_size": OCR_MAX_SIDE,
                "mag_ratio": 1.0, "paragraph": False, "decoder": "greedy",
                "detector": "craft", "recognizer": "standard", "resample": "lanczos",
                "device": self.device, "quantize": False, "weights_sha256": dict(self.weights)}

    def load(self) -> None:
        import easyocr
        import torch

        deterministic_torch(torch)
        self.reader = easyocr.Reader(
            list(OCR_LANGS), gpu=self.device == "cuda",
            model_storage_directory=str(self.directory),
            user_network_directory=str(self.directory / "user_network"),
            download_enabled=False, verbose=False, quantize=False, cudnn_benchmark=False)
        deterministic_torch(torch)  # EasyOCR sets cudnn.benchmark itself

    def read(self, image) -> tuple[list[dict[str, Any]], tuple[int, int]]:
        import numpy as np
        from PIL import Image

        w, h = image.size
        scale = min(1.0, OCR_MAX_SIDE / max(w, h))
        img = image if scale == 1.0 else image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        found = self.reader.readtext(np.asarray(img), detail=1, paragraph=False,
                                     canvas_size=OCR_MAX_SIDE, mag_ratio=1.0, batch_size=1,
                                     workers=0, decoder="greedy")
        regions = [{"text": str(text), "confidence": _r(conf),
                    "polygon": [[int(round(float(x) / scale)), int(round(float(y) / scale))]
                                for x, y in box]}
                   for box, text, conf in found]
        return regions, img.size


class StubVLM:
    """A deterministic stand-in: the whole pipeline without a model (tests, and
    ``run --stub`` to check paths, gates, checkpointing and the lock on a new
    machine). Its config differs, so its rows never mix with real ones."""

    kind = "stub"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0
        self.rendered = {which: f"<system>{SYSTEM_PROMPT}</system><image/>{text}"
                         for which, text in PROMPTS.items()}

    def prepare(self) -> None:
        pass

    def load(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"id": "stub", "revision": "stub-1", "licence": None, "dtype": None,
                "attn_implementation": None, "quantization": None,
                "decoding": decoding_config([0]), "prompts": prompt_config(self.rendered)}

    def generate(self, image, which: str) -> Generation:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        text = (f"A stub photograph, {image.width} by {image.height} pixels, digest {digest[:12]}."
                if which == "description" else
                "PERSON: UNIDENTIFIED\nORG: Stub Labs [text]\nGPE: NONE\nEVENT: NONE\nDATE: NONE")
        ends = [m.end() for m in re.finditer(r"\S+\s*", text)]
        lps = [-((int(digest[i % 60:i % 60 + 2], 16) % 50) / 100.0) for i in range(len(ends))]
        return Generation(text, lps, ends, "eos", len(ends))

    def recover_oom(self) -> None:
        pass

    def memory(self) -> None:
        return None

    def device_info(self) -> dict[str, Any]:
        return {"gpu": None, "gpu_capability": None, "gpu_memory_gb": None, "cuda": None,
                "cudnn": None}


class StubOCR:
    kind = "stub"

    def __init__(self):
        self.calls = 0

    def prepare(self) -> None:
        pass

    def load(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"engine": "stub", "version": None, "langs": list(OCR_LANGS),
                "max_side": OCR_MAX_SIDE, "device": "cpu", "weights_sha256": {}}

    def read(self, image) -> tuple[list[dict[str, Any]], tuple[int, int]]:
        self.calls += 1
        return ([{"text": "STUB LABS", "confidence": 0.9,
                  "polygon": [[0, 0], [image.width, 0], [image.width, 9], [0, 9]]}], image.size)


# --------------------------------------------------------------------------
# configuration and the run directory
# --------------------------------------------------------------------------


def build_config(vlm, ocr) -> dict[str, Any]:
    """Everything that defines the computation. Its hash names the directory."""
    return {"schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION, "seed": SEED,
            "model": vlm.config(), "ocr": ocr.config(),
            "resolution_cap": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                               "factor": FACTOR, "max_aspect": MAX_ASPECT,
                               "resample": "bicubic", "resized_by": "image_text.fit_to_cap",
                               "processor_do_resize": False},
            "preprocessing": {"exif_transpose": True, "alpha": "composite_on_white",
                              "frame": "first", "mode": "RGB"},
            "software": software_versions(),
            "determinism": {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                            "deterministic_algorithms": "warn_only",
                            "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False}}


def config_id_of(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(config)).hexdigest()[:12]


@dataclass
class Context:
    config: dict[str, Any]
    config_id: str
    provenance: dict[str, Any]
    machine: dict[str, Any]
    run_id: str
    out_dir: Path


def make_context(vlm, ocr, out_root: Path | None = None) -> Context:
    config = build_config(vlm, ocr)
    cid = config_id_of(config)
    model = config["model"]
    provenance = {
        "config_id": cid,
        "model": {"id": model["id"], "revision": model["revision"], "licence": model["licence"],
                  "dtype": model["dtype"], "attn_implementation": model["attn_implementation"]},
        "quantization": model["quantization"],
        "prompt_sha256": model["prompts"]["prompt_sha256"],
        "prompt_sha256_by_call": model["prompts"]["by_call"],
        "seed": SEED,
        "decoding": model["decoding"],
        "resolution_cap": config["resolution_cap"],
        "ocr_engine": {k: v for k, v in config["ocr"].items()},
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "software": config["software"],
    }
    machine = {"host_id": host_id(), "os": platform.platform(), "arch": platform.machine(),
               "python": platform.python_version(), "cpus": os.cpu_count(), **vlm.device_info()}
    out_dir = (OUT_ROOT if out_root is None else out_root) / cid
    return Context(config, cid, provenance, machine, uuid.uuid4().hex[:12], out_dir)


def assert_not_raw(path: Path) -> None:
    resolved, raw = path.resolve(), RAW.resolve()
    if resolved == raw or raw in resolved.parents:
        raise FatalError(f"refusing to write under data/raw: {path}")


def write_config(ctx: Context) -> None:
    """config.json, written once; on resume it must say exactly the same thing."""
    assert_not_raw(ctx.out_dir)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.out_dir / "config.json"
    body = json.dumps({"config_id": ctx.config_id, **ctx.config}, indent=1, sort_keys=True,
                      ensure_ascii=False) + "\n"
    if path.is_file():
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        if canonical(on_disk) != canonical({"config_id": ctx.config_id, **ctx.config}):
            raise FatalError(f"{path} does not match the configuration that hashes to "
                             f"{ctx.config_id}; refusing to mix runs")
        return
    tmp = path.with_suffix(".json.part")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the row schema
# --------------------------------------------------------------------------

_P01 = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
_FLAGS = {"type": "array", "items": {"type": "string"}}
_ERROR = {"anyOf": [{"type": "null"}, {"type": "string",
                                       "pattern": r"^(empty|refusal|unparseable|cuda_oom|"
                                                  r"exception:[A-Za-z_][A-Za-z0-9_.]*)$"}]}
_FINISH = {"enum": ["eos", "length", None]}


def _category(forced: bool) -> dict[str, Any]:
    """One entity category. From image_text/2 the category and each of its
    items carry ``forced``: the line was written after a skeleton replacement."""
    mark = {"forced": {"type": "boolean"}} if forced else {}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status", "unidentified", *mark, "items"],
        "properties": {
            "status": {"enum": ["named", "unidentified", "none", "missing"]},
            "unidentified": {"type": "boolean"}, **mark,
            "items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "basis", "confidence", "in_ocr", *mark],
                "properties": {"name": {"type": "string", "minLength": 1},
                               "basis": {"enum": ["text", "seen", "unstated"]},
                               "confidence": _P01, "in_ocr": {"type": ["boolean", "null"]},
                               **mark}}}}}


_NULLABLE_STR = {"type": ["string", "null"]}

ROW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False,
    "required": ["schema", "dataset", "sha256", "record_ids", "n_paths", "status", "reason",
                 "image", "model_input", "ocr", "description", "entities", "confidence",
                 "provenance", "run", "result_sha256"],
    "properties": {
        "schema": {"const": SCHEMA},
        "dataset": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "record_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "n_paths": {"type": "integer", "minimum": 1},
        "status": {"enum": list(STATUSES)},
        "reason": {"enum": [None, *SKIP_REASONS, *FAIL_REASONS]},
        "image": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["format", "width", "height", "mode", "bytes", "frames", "exif_rotated"],
            "properties": {"format": _NULLABLE_STR, "width": {"type": "integer", "minimum": 1},
                           "height": {"type": "integer", "minimum": 1}, "mode": {"type": "string"},
                           "bytes": {"type": "integer", "minimum": 1},
                           "frames": {"type": "integer", "minimum": 1},
                           "exif_rotated": {"type": ["boolean", "null"]}}}]},
        "model_input": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["width", "height", "grid_thw", "visual_tokens"],
            "properties": {"width": {"type": "integer", "multipleOf": FACTOR, "minimum": FACTOR},
                           "height": {"type": "integer", "multipleOf": FACTOR, "minimum": FACTOR},
                           "grid_thw": {"type": "array", "items": {"type": "integer"},
                                        "minItems": 3, "maxItems": 3},
                           "visual_tokens": {"type": "integer", "minimum": 1,
                                             "maximum": MAX_PIXELS // (FACTOR * FACTOR)}}}]},
        "ocr": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["text", "regions", "mean_confidence", "input", "error"],
            "properties": {
                "text": _NULLABLE_STR, "mean_confidence": _P01, "error": _ERROR,
                "input": {"anyOf": [{"type": "null"}, {
                    "type": "object", "additionalProperties": False, "required": ["width", "height"],
                    "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}}}]},
                "regions": {"anyOf": [{"type": "null"}, {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["text", "confidence", "polygon"],
                    "properties": {"text": {"type": "string"}, "confidence": _P01,
                                   "polygon": {"type": "array", "items": {
                                       "type": "array", "items": {"type": "integer"},
                                       "minItems": 2, "maxItems": 2}}}}}]}}}]},
        "description": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["text", "error", "flags", "finish", "n_tokens", "confidence"],
            "properties": {"text": _NULLABLE_STR, "error": _ERROR, "flags": _FLAGS,
                           "finish": _FINISH, "n_tokens": {"type": "integer", "minimum": 0},
                           "confidence": _P01}}]},
        "entities": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["raw", *CATEGORIES, "error", "flags", "finish", "n_tokens", "confidence"],
            "properties": {"raw": _NULLABLE_STR, **{c: _category(forced=True) for c in CATEGORIES},
                           "error": _ERROR, "flags": _FLAGS, "finish": _FINISH,
                           "n_tokens": {"type": "integer", "minimum": 0}, "confidence": _P01}}]},
        "confidence": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["description", "entities", "ocr"],
            "properties": {"description": _P01, "entities": _P01, "ocr": _P01}}]},
        "provenance": {
            "type": "object", "additionalProperties": False,
            "required": ["config_id", "model", "quantization", "prompt_sha256",
                         "prompt_sha256_by_call", "seed", "decoding", "resolution_cap",
                         "ocr_engine", "parser_version", "schema_version", "software", "machine"],
            "properties": {
                "config_id": {"type": "string", "pattern": r"^[0-9a-f]{12}$"},
                "model": {"type": "object", "required": ["id", "revision"],
                          "properties": {"id": {"type": "string"}, "revision": {"type": "string"}}},
                "quantization": {"type": ["object", "null"]},
                "prompt_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "prompt_sha256_by_call": {"type": "object", "required": list(PROMPTS)},
                "seed": {"const": SEED},
                "decoding": {"type": "object", "required": ["strategy", "do_sample", "num_beams"],
                             "properties": {"strategy": {"const": "greedy"},
                                            "do_sample": {"const": False},
                                            "num_beams": {"const": 1}}},
                "resolution_cap": {"type": "object", "required": ["min_pixels", "max_pixels"],
                                   "properties": {"max_pixels": {"const": MAX_PIXELS},
                                                  "min_pixels": {"const": MIN_PIXELS}}},
                "ocr_engine": {"type": "object", "required": ["engine", "version", "device"]},
                "parser_version": {"type": "integer"},
                "schema_version": {"const": SCHEMA_VERSION},
                "software": {"type": "object"},
                "machine": {
                    "type": "object", "additionalProperties": False,
                    "required": ["host_id", "os", "arch", "python", "cpus", "gpu",
                                 "gpu_capability", "gpu_memory_gb", "cuda", "cudnn"],
                    "properties": {"host_id": {"type": "string", "pattern": r"^[0-9a-f]{12}$"},
                                   "os": {"type": "string"}, "arch": {"type": "string"},
                                   "python": {"type": "string"},
                                   "cpus": {"type": ["integer", "null"]},
                                   "gpu": _NULLABLE_STR, "gpu_capability": _NULLABLE_STR,
                                   "gpu_memory_gb": {"type": ["number", "null"]},
                                   "cuda": _NULLABLE_STR,
                                   "cudnn": {"type": ["integer", "null"]}}}}},
        "run": {"type": "object", "additionalProperties": False,
                "required": ["run_id", "at", "seconds", "attempt"],
                "properties": {"run_id": {"type": "string"}, "at": {"type": "string"},
                               "seconds": {"type": "object",
                                           "additionalProperties": {"type": "number"}},
                               "attempt": {"type": "integer", "minimum": 1}}},
        "result_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    },
}


def _first_schema() -> dict[str, Any]:
    """image_text/1: the current schema without the forced marks."""
    old = copy.deepcopy(ROW_SCHEMA)
    old["properties"]["schema"] = {"const": "image_text/1"}
    old["properties"]["provenance"]["properties"]["schema_version"] = {"const": 1}
    entities = old["properties"]["entities"]["anyOf"][1]["properties"]
    for cat in CATEGORIES:
        entities[cat] = _category(forced=False)
    return old


#: Every row is validated against the schema it declares, so a ledger written
#: before a schema change stays readable; only SCHEMA is ever written.
ROW_SCHEMAS: dict[str, dict[str, Any]] = {"image_text/1": _first_schema(), SCHEMA: ROW_SCHEMA}

_VALIDATORS: dict[str, Any] = {}


def _validator(schema: str = SCHEMA):
    if schema not in _VALIDATORS:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(ROW_SCHEMAS[schema])
        _VALIDATORS[schema] = Draft202012Validator(ROW_SCHEMAS[schema])
    return _VALIDATORS[schema]


def _keys(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def row_problems(row: Any, *, schema: bool = True) -> list[str]:
    """Why ``row`` may not be written or trusted; empty when it may.

    ``schema=False`` is the resume check: every row was validated against the
    schema before it was written, so on reading it suffices that the hash
    still matches what was validated.
    """
    if not isinstance(row, dict):
        return ["not an object"]
    problems = []
    if schema:
        declared = row.get("schema")
        if declared not in ROW_SCHEMAS:
            return [f"schema: {declared!r} is not one of {sorted(ROW_SCHEMAS)}"]
        problems += [f"{'/'.join(map(str, e.path)) or '<row>'}: {e.message}"
                     for e in sorted(_validator(declared).iter_errors(row),
                                     key=lambda e: list(e.path))]
        if problems:
            return problems
    bad = sorted(set(_keys(row)) & FORBIDDEN_KEYS)
    if bad:
        problems.append(f"carries forbidden key(s) {bad}: rows name no file, label or host")
    for cat in CATEGORIES:  # the mark travels with the entity, and says what its line says
        block = (row.get("entities") or {}).get(cat)
        if isinstance(block, dict) and "forced" in block and any(
                item.get("forced") != block["forced"] for item in block.get("items") or []):
            problems.append(f"entities/{cat}: an item's forced mark is not its line's")
    status, reason = row.get("status"), row.get("reason")
    present = [c for c in COMPONENTS if row.get(c) is not None]
    if status == "skipped" and reason not in SKIP_REASONS:
        problems.append(f"skipped with reason {reason!r}")
    if status == "failed" and reason not in FAIL_REASONS:
        problems.append(f"failed with reason {reason!r}")
    if status in ("skipped", "failed") and present:
        problems.append(f"{status} row carries outputs {present}")
    if status in ("ok", "partial"):
        if reason is not None:
            problems.append(f"{status} row carries reason {reason!r}")
        if len(present) != len(COMPONENTS) or row.get("image") is None:
            problems.append(f"{status} row lacks outputs")
        else:
            errors = [row[c].get("error") for c in COMPONENTS]
            if status == "ok" and any(errors):
                problems.append("ok row carries a component error")
            if status == "partial" and not any(errors):
                problems.append("partial row has no component error")
    if row.get("result_sha256") != result_sha256(row):
        problems.append("result_sha256 does not match the row's content")
    return problems


# --------------------------------------------------------------------------
# processing one image
# --------------------------------------------------------------------------


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _attempt(fn: Callable[[], Any], vlm) -> tuple[Any, str | None]:
    """(value, None), or (None, error code). An OOM is retried once in place."""
    for tries in (1, 2):
        try:
            return fn(), None
        except FatalError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the code IS the measurement
            if _is_oom(exc):
                vlm.recover_oom()
                if tries == 1:
                    continue
                return None, "cuda_oom"
            log(f"    component error: {type(exc).__name__}: {str(exc)[:300]}")
            return None, f"exception:{type(exc).__name__}"
    return None, "cuda_oom"


def seal(row: dict[str, Any], ctx: Context, seconds: dict[str, float], attempt: int) -> dict[str, Any]:
    row["provenance"] = {**ctx.provenance, "machine": ctx.machine}
    row["run"] = {"run_id": ctx.run_id, "at": utcnow(),
                  "seconds": {k: round(v, 3) for k, v in seconds.items()}, "attempt": attempt}
    row["result_sha256"] = result_sha256(row)
    return row


def process(item: WorkItem, vlm, ocr, ctx: Context, *, attempt: int = 1) -> dict[str, Any]:
    """One image -> one row. Never raises for a bad image: a bad image is a row.

    It raises FatalError only when the run's own guarantees break -- among them
    an entity answer that lost its lines, which means the skeleton did not hold:
    such a row is never written, as ``ok`` or otherwise."""
    from PIL import Image

    t0 = time.perf_counter()
    seconds: dict[str, float] = {}
    row: dict[str, Any] = {"schema": SCHEMA, "dataset": item.dataset, "sha256": item.sha256,
                           "record_ids": list(item.record_ids), "n_paths": item.n_paths,
                           "status": None, "reason": None, "image": None, "model_input": None,
                           "ocr": None, "description": None, "entities": None,
                           "confidence": None}

    def finish(status: str, reason: str | None = None) -> dict[str, Any]:
        row["status"], row["reason"] = status, reason
        seconds["total"] = time.perf_counter() - t0
        return seal(row, ctx, seconds, attempt)

    try:
        body = read_bytes(item.path)
    except FileNotFoundError:
        return finish("failed", "missing_file")
    except KeyError:
        return finish("failed", "missing_member")
    except (OSError, zipfile.BadZipFile):
        return finish("failed", "unreadable")
    seconds["read"] = time.perf_counter() - t0
    status, reason = gate(item, body)
    if status:
        return finish(status, reason)
    try:
        rgb, info = prepare(body)
    except Exception:  # noqa: BLE001
        return finish("failed", "undecodable")
    try:
        fed_w, fed_h = fit_to_cap(*rgb.size)
    except GeometryError:
        return finish("failed", "unsupported_geometry")
    row["image"] = info
    fed = rgb if rgb.size == (fed_w, fed_h) else rgb.resize((fed_w, fed_h), Image.Resampling.BICUBIC)
    row["model_input"] = {"width": fed_w, "height": fed_h,
                          "grid_thw": [1, fed_h // 14, fed_w // 14],
                          "visual_tokens": (fed_w // FACTOR) * (fed_h // FACTOR)}
    seconds["prepare"] = time.perf_counter() - t0 - sum(seconds.values())

    t = time.perf_counter()
    found, error = _attempt(lambda: ocr.read(rgb), vlm)
    row["ocr"] = ocr_block(*found) if found is not None else _failed_component("ocr", error)
    seconds["ocr"] = time.perf_counter() - t

    t = time.perf_counter()
    gen, error = _attempt(lambda: vlm.generate(fed, "description"), vlm)
    row["description"] = (description_block(gen) if gen is not None
                          else _failed_component("description", error))
    seconds["description"] = time.perf_counter() - t

    t = time.perf_counter()
    gen, error = _attempt(lambda: vlm.generate(fed, "entities"), vlm)
    row["entities"] = (entities_block(gen, row["ocr"]["text"]) if gen is not None
                       else _failed_component("entities", error))
    seconds["entities"] = time.perf_counter() - t
    form = entity_format(row["entities"])
    if form not in ENTITY_FORMS_WRITTEN:
        raise FatalError(f"an entity answer lost its five lines ({form}): "
                         f"{(row['entities']['raw'] or '')[:200]!r}; the skeleton did not hold")

    row["confidence"] = {"description": row["description"]["confidence"],
                         "entities": row["entities"]["confidence"],
                         "ocr": row["ocr"]["mean_confidence"]}
    broken = any(row[c]["error"] for c in COMPONENTS)
    return finish("partial" if broken else "ok")


# --------------------------------------------------------------------------
# the ledger: append-only, fsynced, one writer, torn tails cut and kept
# --------------------------------------------------------------------------


class FileLock:
    """An exclusive OS lock, released by the OS when its holder dies -- even on
    SIGKILL -- so a lock left by a killed run is never stale."""

    #: Windows locks are mandatory: lock a byte past the holder note, so a
    #: refused process can still read who holds it.
    _OFFSET = 1 << 20
    _NOTE = 256

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self, wait: float = 5.0) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        fh = os.fdopen(fd, "r+b")
        deadline = time.monotonic() + wait
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(self._OFFSET)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    fh.close()
                    raise LockHeld(f"{self.path.name} is held by another process "
                                   f"({self.holder() or 'unknown'}); one writer per ledger") from None
                time.sleep(0.2)
        self._fh = fh
        note = json.dumps({"pid": os.getpid(), "host_id": host_id(), "since": utcnow()})
        fh.seek(0)
        fh.write(note.encode("utf-8").ljust(self._NOTE)[:self._NOTE])
        fh.flush()

    def holder(self) -> str | None:
        try:
            return self.path.read_bytes()[:self._NOTE].decode("utf-8", "replace").strip() or None
        except OSError:
            return None

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(self._OFFSET)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def summary_of(row: dict[str, Any]) -> dict[str, Any]:
    return {"status": row["status"], "reason": row["reason"], "attempt": row["run"]["attempt"],
            "result_sha256": row["result_sha256"],
            "errors": sorted({row[c]["error"] for c in COMPONENTS
                              if row.get(c) and row[c].get("error")})}


def scan_ledger(path: Path, *, dataset: str | None = None, config_id: str | None = None,
                keep: Callable[[dict[str, Any]], bool] | None = None,
                schema: bool = False) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    """(sha256 -> summary of its last row, rows ``keep`` asked for, torn tail bytes).

    Strict: a complete line that is not a valid row raises -- with its line
    number -- because skipping it would hide damage. A torn LAST line (no
    newline: the process died mid-write) is reported, not read; Ledger.open()
    cuts it. Streams, so a 140k-row ledger is never held in memory.
    """
    summaries: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    torn = 0
    if not path.is_file():
        return summaries, kept, torn
    with path.open("rb") as handle:
        for n, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                torn = len(line)
                break
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise LedgerError(f"{path.name}: line {n} is not JSON ({exc}); refusing to "
                                  "resume over a damaged ledger") from None
            problems = row_problems(row, schema=schema)
            if problems:
                raise LedgerError(f"{path.name}: line {n}: {problems[0]}")
            if dataset is not None and row["dataset"] != dataset:
                raise LedgerError(f"{path.name}: line {n} belongs to {row['dataset']!r}")
            if config_id is not None and row["provenance"]["config_id"] != config_id:
                raise LedgerError(f"{path.name}: line {n} was written under config "
                                  f"{row['provenance']['config_id']}, not {config_id}")
            summaries[row["sha256"]] = summary_of(row)
            if keep is not None and keep(row):
                kept.append(row)
    return summaries, kept, torn


class Ledger:
    def __init__(self, path: Path, *, dataset: str, config_id: str):
        self.path, self.dataset, self.config_id = path, dataset, config_id
        self.lock = FileLock(path.with_name(path.name + ".lock"))
        self.torn_path = path.with_name(path.stem + ".torn.jsonl")
        self._fh = None

    def open(self, *, keep=None, lock_wait: float = 5.0):
        """Lock, cut a torn tail, read what exists. Returns (summaries, kept rows)."""
        assert_not_raw(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire(lock_wait)
        try:
            self._cut_torn_tail()
            summaries, kept, _ = scan_ledger(self.path, dataset=self.dataset,
                                             config_id=self.config_id, keep=keep)
            self._fh = open(self.path, "ab")
        except BaseException:
            self.lock.release()
            raise
        return summaries, kept

    def _cut_torn_tail(self) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "r+b") as fh:
            size = fh.seek(0, os.SEEK_END)
            if size == 0:
                return
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return
            cut, pos = 0, size
            while pos > 0:
                start = max(0, pos - (1 << 16))
                fh.seek(start)
                chunk = fh.read(pos - start)
                i = chunk.rfind(b"\n")
                if i >= 0:
                    cut = start + i + 1
                    break
                pos = start
            fh.seek(cut)
            fragment = fh.read()
            with open(self.torn_path, "ab") as torn:  # kept, never deleted
                torn.write(canonical({"cut_at": utcnow(), "offset": cut, "bytes": len(fragment),
                                      "fragment": fragment.decode("utf-8", "replace")}) + b"\n")
                torn.flush()
                os.fsync(torn.fileno())
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        log(f"  [{self.dataset}] cut a torn last line ({len(fragment)} bytes) -> {self.torn_path.name}")

    def append(self, row: dict[str, Any]) -> None:
        problems = row_problems(row)
        if row.get("schema") != SCHEMA:
            problems.append(f"schema {row.get('schema')!r}: only {SCHEMA} is written")
        if problems:
            raise FatalError(f"refusing to write an invalid row: {problems[:3]}")
        self._fh.write(canonical(row) + b"\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.lock.release()


def needs_work(summary: dict[str, Any] | None, retry_failed: bool = False) -> bool:
    if summary is None:
        return True
    return retry_failed and summary["status"] in ("failed", "partial")


class Runner:
    """Ledgers for the run, the images it computes, and what it measured."""

    def __init__(self, vlm, ocr, ctx: Context, *, lock_wait: float = 5.0):
        self.vlm, self.ocr, self.ctx, self.lock_wait = vlm, ocr, ctx, lock_wait
        self.ledgers: dict[str, Ledger] = {}
        self.summaries: dict[str, dict[str, dict[str, Any]]] = {}
        self.rows: list[dict[str, Any]] = []
        self.memory: list[dict[str, int]] = []
        self.consecutive_oom = 0

    def open(self, dataset: str) -> None:
        if dataset not in self.ledgers:
            ledger = Ledger(self.ctx.out_dir / f"{dataset}.jsonl", dataset=dataset,
                            config_id=self.ctx.config_id)
            self.summaries[dataset], _ = ledger.open(lock_wait=self.lock_wait)
            self.ledgers[dataset] = ledger

    def pending(self, items: Sequence[WorkItem], retry_failed: bool = False) -> list[WorkItem]:
        return [it for it in items
                if needs_work(self.summaries[it.dataset].get(it.sha256), retry_failed)]

    def step(self, item: WorkItem) -> dict[str, Any]:
        prev = self.summaries[item.dataset].get(item.sha256)
        row = process(item, self.vlm, self.ocr, self.ctx,
                      attempt=(prev["attempt"] + 1) if prev else 1)
        self.ledgers[item.dataset].append(row)
        self.summaries[item.dataset][item.sha256] = summary_of(row)
        self.rows.append(row)
        sample = self.vlm.memory()
        if sample:
            self.memory.append(sample)
        oom = any(row[c] and row[c].get("error") == "cuda_oom" for c in COMPONENTS)
        self.consecutive_oom = self.consecutive_oom + 1 if oom else 0
        if self.consecutive_oom >= 3:
            raise FatalError("three images in a row ran out of GPU memory")
        return row

    def close(self) -> None:
        for ledger in self.ledgers.values():
            ledger.close()


def _progress(i: int, n: int, row: dict[str, Any], started: float) -> str:
    secs = row["run"]["seconds"]
    rate = (time.perf_counter() - started) / i
    eta = rate * (n - i)
    stages = " ".join(f"{k[:4]} {secs[k]:.1f}" for k in ("ocr", "description", "entities")
                      if k in secs)
    reason = f" {row['reason']}" if row["reason"] else ""
    return (f"  [{row['dataset']}] {i:,}/{n:,} {row['status']}{reason} {secs['total']:.1f}s "
            f"({stages}) eta {eta / 3600:.1f}h")


# --------------------------------------------------------------------------
# the pilot
# --------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))]

    return {"mean": round(statistics.fmean(ordered), 3), "p50": round(q(0.5), 3),
            "p90": round(q(0.9), 3), "max": round(ordered[-1], 3)}


def _fit(rows: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    """Least squares: seconds = a + b * visual_tokens, over images the model saw."""
    pts = [(r["model_input"]["visual_tokens"], r["run"]["seconds"]["total"]) for r in rows
           if r["status"] in ("ok", "partial")]
    if len(pts) < 3:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"a": round(my, 4), "b": 0.0, "r2": 0.0, "n": len(pts)}
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in pts)
    return {"a": round(a, 4), "b": round(b, 6), "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else 0.0,
            "n": len(pts)}


def sizes_of(items: Sequence[WorkItem]) -> list[tuple[str, int | None]]:
    """(sha256, visual tokens) per item: all a projection needs, in a fraction
    of the memory of the items themselves (Fakeddit's pool is 140,839)."""
    return [(it.sha256, visual_tokens(it.width, it.height)) for it in items]


def _project(sizes: Sequence[tuple[str, int | None]], done: dict[str, Any],
             per_image: float | None, fit: dict[str, float] | None) -> dict[str, Any]:
    remaining = [t for sha, t in sizes if sha not in done]
    tokens = remaining
    known = [t for t in tokens if t is not None]
    mean_tokens = statistics.fmean(known) if known else None
    out: dict[str, Any] = {"distinct_images": len(sizes), "done": len(sizes) - len(remaining),
                           "remaining": len(remaining),
                           "mean_visual_tokens": round(mean_tokens, 1) if mean_tokens else None,
                           "hours_naive": None, "hours_token_adjusted": None}
    if per_image is not None:
        out["hours_naive"] = round(len(remaining) * per_image / 3600, 2)
    if fit is not None and mean_tokens is not None:
        total = sum(fit["a"] + fit["b"] * (t if t is not None else mean_tokens) for t in tokens)
        out["hours_token_adjusted"] = round(max(0.0, total) / 3600, 2)
    return out


def _labels(dataset: str) -> dict[str, str]:
    """record_id -> label, for per-class AGGREGATES in reports. Never written to a row."""
    from scripts.processed_loader import iter_rows

    try:
        return {str(r["record_id"]): str(r.get("label")) for r in iter_rows(dataset)}
    except FileNotFoundError:
        return {}


def per_class(rows: Iterable[dict[str, Any]], labels: dict[str, str]) -> dict[str, dict[str, int]]:
    """status (and reason) counts per class; an image counts once for each class using it."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for label in sorted({labels.get(r, "unknown") for r in row["record_ids"]}):
            out[label][row["status"]] += 1
            if row["reason"]:
                out[label][f"reason:{row['reason']}"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _evidence(item: dict[str, Any], ocr_text: str | None) -> str:
    """Whether the independent OCR found the name: the evidence for reading vs
    recognising. ``partly_written``: some of its words are in the image
    ("Narendra Modi" where it reads "MODI"), so it may have been read;
    ``not_written``: none are, so it was produced from appearance alone."""
    if item["in_ocr"] is None:
        return "no_ocr"
    if item["in_ocr"]:
        return "written"
    return "partly_written" if _ocr_words(item["name"], ocr_text or "")[0] else "not_written"


#: The evidence a name can have, in the order reports list it.
EVIDENCE = ("written", "partly_written", "not_written")


def entity_stats(seen: Sequence[dict[str, Any]], forced: bool) -> dict[str, Any]:
    """Entity statistics over the lines that were -- or were not -- forced.

    A forced line is weaker evidence than a free one, so the two are never
    pooled: every figure here is over one kind of line only."""
    lines: Counter = Counter()
    status: dict[str, Counter] = defaultdict(Counter)
    names: list[tuple[dict[str, Any], str]] = []
    people: list[tuple[dict[str, Any], str]] = []
    naming_unwritten = 0
    for r in seen:
        ocr = r["ocr"]["text"]
        for cat in CATEGORIES:
            block = r["entities"][cat]
            if block["forced"] != forced:
                continue
            lines[cat] += 1
            status[cat][block["status"]] += 1
            found = [(i, _evidence(i, ocr)) for i in block["items"]]
            names += found
            if cat == "PERSON":
                people += found
                naming_unwritten += any(ev == "not_written" for _i, ev in found)
    return {"lines": {c: lines[c] for c in CATEGORIES if lines[c]},
            "status": {c: dict(sorted(status[c].items())) for c in CATEGORIES if status[c]},
            # the model's own tag, against the OCR's evidence: [seen] on a
            # written name means the tag cannot be trusted on its own
            "names_tag_vs_ocr": dict(sorted(Counter(f"{i['basis']}:{ev}" for i, ev in names).items())),
            "person_names_tag_vs_ocr": dict(sorted(Counter(f"{i['basis']}:{ev}"
                                                           for i, ev in people).items())),
            "images_naming_a_person_not_written": naming_unwritten,
            "person_name_confidence": {ev: _stats([i["confidence"] for i, e in people
                                                   if e == ev and i["confidence"] is not None])
                                       for ev in EVIDENCE},
            "person_names_below_0_5": sum(1 for i, _ev in people
                                          if i["confidence"] is not None and i["confidence"] < 0.5)}


def quality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seen = [r for r in rows if r["status"] in ("ok", "partial")]
    desc_flags = Counter(f for r in seen for f in r["description"]["flags"])
    ent_flags = Counter(f.split(":")[0] for r in seen for f in r["entities"]["flags"])
    missing = Counter(f.split(":")[1] for r in seen for f in r["entities"]["flags"]
                      if f.startswith("category_missing:"))
    events = Counter(f for r in seen for f in r["entities"]["flags"]
                     if f.split(":")[0] in SKELETON_EVENTS)
    errors = Counter(f"{c}:{r[c]['error']}" for r in seen for c in COMPONENTS if r[c]["error"])
    return {"images_seen_by_model": len(seen),
            "component_errors": dict(sorted(errors.items())),
            "description_flags": dict(sorted(desc_flags.items())),
            "entity_format": dict(sorted(Counter(entity_format(r["entities"]) for r in seen).items())),
            "entity_skeleton_events": dict(sorted(events.items())),
            "entity_flags": dict(sorted(ent_flags.items())),
            "entity_lines_missing": dict(sorted(missing.items())),
            "entities_fully_parsed": sum(1 for r in seen if not any(
                f.startswith("category_missing:") for f in r["entities"]["flags"])),
            # what the model finished unprompted: all five lines, none forced
            "entities_complete_free": sum(
                1 for r in seen if entity_format(r["entities"]) == "complete"
                and not any(r["entities"][c]["forced"] for c in CATEGORIES)),
            "entities": {"free": entity_stats(seen, False), "forced": entity_stats(seen, True)},
            "images_with_ocr_text": sum(1 for r in seen if (r["ocr"]["text"] or "").strip()),
            "generated_tokens": {"description": _stats([r["description"]["n_tokens"] for r in seen]),
                                 "entities": _stats([r["entities"]["n_tokens"] for r in seen])}}


def candidate_precision_needed(ratio: float, target: float = VERIFIER_PRECISION) -> float:
    """Precision a candidate name needs for the names a gallery ACCEPTS to be
    right ``target`` of the time, when the gallery accepts a wrong name
    ``ratio`` times as often as a right one (FMR / TAR). Accepted names are
    right p*TAR / (p*TAR + (1-p)*FMR) of the time; this is that, solved for p."""
    return target * ratio / (target * ratio + (1 - target))


def max_false_accept_ratio(precision: float, target: float = VERIFIER_PRECISION) -> float | None:
    """The largest FMR / TAR a gallery may have for candidates of this
    precision to give accepted names right ``target`` of the time -- the
    inverse of candidate_precision_needed. None when every candidate is right."""
    if precision >= 1:
        return None
    return precision / (1 - precision) * (1 - target) / target


def probe_summary(probe: Sequence[WorkItem], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Does the entity call recognise public figures from their appearance --
    and when it names someone, how often is it right?

    Recognition is measured on EVIDENCE, not on the model's tag (which, tested
    on Factify2 train images, says [seen] for printed names too): a PERSON name
    none of whose words the independent OCR finds in the image cannot have
    been read; it was produced from appearance alone. A name the OCR holds in
    part ("Narendra Modi" where the image reads "MODI") may have been read, and
    is not counted as recognition.

    Two rates. The recall-style rate -- the claimed figure named on images
    whose figure's name is not written -- counts hits and is silent on wrong
    names. PRECISION counts both: of the names produced from appearance alone,
    the share that are the figure the claim names. Its reference is the claim,
    not a hand label: a correctly named bystander counts against it, and the
    figure named on an image that does not show them counts for it.

    Two thresholds. VERIFIER_PRECISION is what a name needs to replace a
    gallery. As a candidate that a gallery then verifies -- the role a VLM name
    plays -- it needs less, by how rarely the gallery accepts a wrong name
    (candidate_precision_needed); the report gives the largest false-accept
    rate the measured precision allows. Headline figures are over free PERSON
    lines; a PERSON line the skeleton forced is weaker evidence and counted
    apart.
    """
    token = {name: re.compile(tok) for name, _claim, tok in PUBLIC_FIGURES}
    tally: Counter = Counter()
    forced_tally: Counter = Counter()
    per_figure: dict[str, Counter] = defaultdict(Counter)
    conf: dict[str, list[float]] = defaultdict(list)
    loose_conf: dict[str, list[float]] = defaultdict(list)
    scored: list[tuple[float | None, bool]] = []
    tags: Counter = Counter()
    for item in probe:
        row = rows.get(item.sha256)
        if row is None or row["status"] not in ("ok", "partial"):
            tally["not_seen_by_model"] += 1
            continue
        person = row["entities"]["PERSON"]
        pattern = token[item.probe]
        ocr = row["ocr"]["text"] or ""
        found = [(i, _evidence(i, ocr), bool(pattern.search(_normalise(i["name"]))))
                 for i in person["items"]]
        if person["forced"]:
            forced_tally["images"] += 1
            for _i, ev, claimed in found:
                if ev == "not_written":
                    forced_tally["names_from_appearance"] += 1
                    forced_tally["claimed_figure" if claimed else "someone_else"] += 1
            continue
        tally["images"] += 1
        tally[f"person_{person['status']}"] += 1
        written = bool(pattern.search(_normalise(ocr)))
        named_claimed = any(claimed for _i, _ev, claimed in found)
        recognised = [(i, claimed) for i, ev, claimed in found if ev == "not_written"]
        tags.update(f"{i['basis']}:{ev}" for i, ev, _claimed in found)
        tally["name_written" if written else "name_not_written"] += 1
        tally["names_claimed_figure"] += named_claimed
        tally["any_person_tagged_seen"] += any(i["basis"] == "seen" for i in person["items"])
        if not written:
            tally["name_not_written_names_claimed_figure"] += named_claimed
            tally["name_not_written_names_someone"] += bool(recognised)
            tally["name_not_written_unidentified_or_none"] += not person["items"]
        for i, claimed in recognised:
            key = "claimed_figure" if claimed else "someone_else"
            tally[f"recognised_names_{key}"] += 1
            scored.append((i["confidence"], claimed))
            if i["confidence"] is not None:
                conf[key].append(i["confidence"])
        for i, _ev, claimed in found:  # as first counted: every name the OCR lacks in full
            if i["in_ocr"] is False:
                key = "claimed_figure" if claimed else "someone_else"
                tally[f"not_fully_written_names_{key}"] += 1
                if i["confidence"] is not None:
                    loose_conf[key].append(i["confidence"])
        f = per_figure[item.probe]
        f["images"] += 1
        f["name_not_written"] += not written
        f["recognised_claimed_figure"] += bool(not written and named_claimed)
    n, k = tally["images"], tally["name_not_written"]
    key = tally["name_not_written_names_claimed_figure"] / k if k else None
    anyone = tally["name_not_written_names_someone"] / k if k else None
    near_zero = key is not None and key < NEAR_ZERO
    c, w = tally["recognised_names_claimed_figure"], tally["recognised_names_someone_else"]
    precision = c / (c + w) if c + w else None
    ratio = None if precision is None else max_false_accept_ratio(precision)
    lc, lw = tally["not_fully_written_names_claimed_figure"], tally["not_fully_written_names_someone_else"]
    by_confidence = []
    for floor in CONFIDENCE_GRID:
        kept = [hit for cf, hit in scored if cf is not None and cf >= floor]
        by_confidence.append({"min_confidence": floor, "names": len(kept), "claimed_figure": sum(kept),
                              "precision": _r(sum(kept) / len(kept), 4) if kept else None})
    thresholds = {
        "verifier": {
            "role": "a name taken as the identity, with no gallery behind it: replacing the gallery",
            "precision_needed": VERIFIER_PRECISION,
            "met": None if precision is None else precision >= VERIFIER_PRECISION},
        "candidate": {
            "role": "a name proposed to a gallery that then verifies it: the role a VLM name plays",
            "rule": (f"what must reach {VERIFIER_PRECISION} is the precision of the names the "
                     "gallery accepts. With candidate precision p, a gallery accepting a right name "
                     "at rate TAR and a wrong one at rate FMR accepts names that are right "
                     "p*TAR / (p*TAR + (1-p)*FMR) of the time, so p must be at least "
                     f"{VERIFIER_PRECISION}*r / ({VERIFIER_PRECISION}*r + {1 - VERIFIER_PRECISION:.2f}), "
                     "r = FMR/TAR"),
            "precision_needed_at_ratio": {str(r): _r(candidate_precision_needed(r), 4)
                                          for r in GALLERY_RATIOS},
            "ratios_are": "hypothetical gallery operating points (r = FMR/TAR): the gallery is not built yet",
            "max_gallery_false_accept_ratio": _r(ratio, 4),
            "caveat": ("FMR here is the gallery's false-accept rate on the VLM's own wrong names, "
                       "which are look-alikes more often than random impostors: measure it on them, "
                       "not on random pairs")}}
    if key is None:
        verdict = None
    elif near_zero:
        verdict = (f"NEAR ZERO: on {k} public-figure images whose figure's name is not written in "
                   f"them, the entity call named that figure from appearance on "
                   f"{tally['name_not_written_names_claimed_figure']} ({key:.1%}). The VLM is not "
                   "doing recognition; the entity gallery in the next block has to carry the "
                   "whole recognition load.")
    else:
        verdict = (f"PRECISION {precision:.1%}: of {c + w} PERSON names produced from appearance "
                   f"alone (no word of them in the OCR text), {c} are the figure the claim names and "
                   f"{w} someone else. " if precision is not None else
                   "PRECISION undefined: no PERSON name was produced from appearance alone. ")
        verdict += (f"The recall-style rate -- the figure named on "
                    f"{tally['name_not_written_names_claimed_figure']} of {k} images whose figure's "
                    f"name is not written ({key:.1%}) -- counts hits and is silent on wrong names. ")
        if conf.get("claimed_figure") and conf.get("someone_else"):
            top = max(conf["someone_else"])
            verdict += (f"Confidence: mean {statistics.fmean(conf['claimed_figure']):.3f} for the "
                        f"claimed figure, {statistics.fmean(conf['someone_else']):.3f} for someone else; "
                        f"the most confident name of someone else is at {top:.3f}, above "
                        f"{sum(1 for x in conf['claimed_figure'] if x < top)} of the "
                        f"{len(conf['claimed_figure'])} names of the claimed figure. ")
        verdict += "The model's [seen]/[text] tag is not evidence (see person_tag_vs_ocr); the OCR is."
    if precision is not None and precision < VERIFIER_PRECISION:
        conclusion = (f"Entity verification cannot come from the VLM: at {precision:.1%} precision "
                      f"its names fall short of the {VERIFIER_PRECISION:.0%} a name needs to replace a "
                      "gallery. ")
        conclusion += (f"As candidates that a gallery then verifies -- the role they play -- they are "
                       f"usable if the gallery accepts a wrong VLM name at most {ratio:.3f} times as "
                       "often as a right one (FMR/TAR), measured on the VLM's own wrong names. "
                       if c else "None of them was the claimed figure, so they are not worth "
                       "proposing to a gallery either. ")
        conclusion += "Identity is decided by an external gallery."
        decision, role = "external_gallery", ("candidate" if c else None)
    elif precision is None and near_zero:
        conclusion = ("Entity verification cannot come from the VLM: it names almost no one from "
                      "appearance, so an external gallery carries the whole recognition load and "
                      "the VLM proposes no candidates.")
        decision, role = "external_gallery", None
    elif precision is not None:
        conclusion = (f"At {precision:.1%} precision the VLM's names meet the "
                      f"{VERIFIER_PRECISION:.0%} a name needs to replace a gallery on this probe, "
                      "whose reference is the claim, not a hand label: check by hand before "
                      "relying on it.")
        decision, role = None, None
    else:
        conclusion, decision, role = None, None, None
    return {"source": "factify2 TRAIN claims naming exactly one public figure and saying an "
                      "image or video shows them; the claim image",
            "requested": len(probe), "counts": dict(sorted(tally.items())),
            "precision": {
                "value": _r(precision, 4), "names": c + w, "claimed_figure": c, "someone_else": w,
                "population": ("PERSON names produced from appearance alone -- no word of the name "
                               "is in the independent OCR text -- on free PERSON lines"),
                "reference": ("the figure the claim names, not a hand label: a correctly named "
                              "bystander counts against it, and the figure named on an image that "
                              "does not show them counts for it"),
                "by_confidence": by_confidence,
                "not_fully_written": {
                    "population": ("as first counted: names the OCR does not contain in full "
                                   "(in_ocr false), including names it contains in part"),
                    "names": lc + lw, "claimed_figure": lc, "someone_else": lw,
                    "value": _r(lc / (lc + lw), 4) if lc + lw else None,
                    "confidence": {k_: _stats(v) for k_, v in sorted(loose_conf.items())}}},
            "rate_claimed_figure_named_where_name_not_written": _r(key, 4),
            "rate_someone_named_from_appearance_where_name_not_written": _r(anyone, 4),
            "rate_any_person_tagged_seen": _r(tally["any_person_tagged_seen"] / n if n else None, 4),
            "rates_are": ("recall-style: the share of images on which a name was given; they count "
                          "hits and are silent on wrong names -- precision is in `precision`"),
            "person_tag_vs_ocr": dict(sorted(tags.items())),
            "recognised_name_confidence": {k_: _stats(v) for k_, v in sorted(conf.items())},
            "forced_person_lines": dict(sorted(forced_tally.items())),
            "per_figure": {k_: dict(sorted(v.items())) for k_, v in sorted(per_figure.items())},
            "thresholds": thresholds,
            "near_zero": near_zero, "near_zero_threshold": NEAR_ZERO,
            "verdict": verdict, "conclusion": conclusion,
            "entity_verification": decision, "vlm_role": role}


def count_corrections(stats: dict[str, dict[str, int]],
                      index_files: dict[str, int]) -> dict[str, Any]:
    out = {}
    for ds in ORDER:
        if ds not in stats:
            continue
        out[ds] = {**stats[ds], "files_in_image_index": index_files.get(ds),
                   "brief_figure": BRIEF_FIGURES.get(ds)}
    return {"note": ("The brief for this block quoted VERITE 914, AVerImaTeC 1,392 and Factify2 "
                     "77,505 images. Those figures were wrong. 914 is VERITE's usable ROWS: its "
                     "607 distinct images are fewer, because a true and a miscaptioned row share "
                     "one image. Nor is it the 613 VERITE files in the image index: those are 606 "
                     "usable + 7 the registry and the fingerprint check remove, and the 607th "
                     "image is a MOCHEG file two VERITE rows reach. 1,392 and 77,505 match no "
                     "measured quantity. The unit of work is a distinct image (sha256) of a usable "
                     "row; the measured figures below are the only ones to use, and "
                     "docs/counts_ledger.md carries the same numbers."),
            "datasets": out}


def build_pilot_report(*, ctx: Context, vlm, target: str | None, requested: int,
                       pilot_rows: list[dict[str, Any]], probe: list[WorkItem],
                       probe_rows: dict[str, dict[str, Any]], probe_new: list[dict[str, Any]],
                       load_seconds: float, determinism: dict[str, Any],
                       memory: list[dict[str, int]],
                       sizes: dict[str, list[tuple[str, int | None]]],
                       stats: dict[str, dict[str, int]], index_files: dict[str, int],
                       done: dict[str, dict[str, Any]],
                       wall_seconds: float, labels: dict[str, str]) -> dict[str, Any]:
    totals = [r["run"]["seconds"]["total"] for r in pilot_rows]
    per_image = statistics.fmean(totals) if totals else None
    fit = _fit(pilot_rows + probe_new)
    stage = {k: _stats([r["run"]["seconds"][k] for r in pilot_rows if k in r["run"]["seconds"]])
             for k in ("read", "prepare", "ocr", "description", "entities", "total")}
    vram = None
    if memory:
        gb = 1024 ** 3
        total = memory[-1]["device_total"]
        others = max(m["device_total"] - m["device_free"] - m["reserved"] for m in memory)
        peak_reserved = max(m["max_reserved"] for m in memory)
        vram = {"device_total_gb": round(total / gb, 2),
                "torch_peak_allocated_gb": round(max(m["max_allocated"] for m in memory) / gb, 2),
                "torch_peak_reserved_gb": round(peak_reserved / gb, 2),
                "outside_torch_gb": round(others / gb, 2),
                "device_peak_estimate_gb": round((peak_reserved + others) / gb, 2),
                "note": "device_peak_estimate = torch's peak reserved + what the device held "
                        "outside torch's allocator (CUDA context, other processes), sampled "
                        "after each image"}
    projection = {ds: _project(s, done.get(ds, {}), per_image, fit) for ds, s in sizes.items()}
    if "fakeddit" in sizes and per_image is not None:
        tokens = [t for _sha, t in sizes["fakeddit"] if t is not None]
        mean_t = statistics.fmean(tokens) if tokens else None
        projection["fakeddit"]["note"] = ("the pool; --fakeddit-n N takes a prefix of the nested "
                                          "stratified order, proportional to the 6-way class")
        projection["fakeddit"]["hours_per_10k_naive"] = round(10_000 * per_image / 3600, 2)
        if fit and mean_t:
            projection["fakeddit"]["hours_per_10k_token_adjusted"] = round(
                10_000 * (fit["a"] + fit["b"] * mean_t) / 3600, 2)
    report = {
        "generated_at": utcnow(), "config_id": ctx.config_id,
        "model": {"id": ctx.provenance["model"]["id"], "revision": ctx.provenance["model"]["revision"],
                  "quantization": ctx.provenance["quantization"]},
        "machine": ctx.machine, "software": ctx.provenance["software"],
        "resolution_cap": ctx.provenance["resolution_cap"],
        "pilot": {"dataset": target, "requested": requested, "processed": len(pilot_rows),
                  "status": dict(Counter(r["status"] for r in pilot_rows)),
                  "reasons": dict(Counter(r["reason"] for r in pilot_rows if r["reason"])),
                  "per_class": per_class(pilot_rows, labels)},
        "throughput": {"model_load_seconds": round(load_seconds, 1),
                       "wall_seconds": round(wall_seconds, 1),
                       "images_per_second": round(len(pilot_rows) / sum(totals), 4) if totals else None,
                       "seconds_per_image": stage["total"], "stage_seconds": stage,
                       "mean_visual_tokens": round(statistics.fmean(
                           [r["model_input"]["visual_tokens"] for r in pilot_rows
                            if r["model_input"]]), 1) if any(r["model_input"] for r in pilot_rows) else None,
                       "fit_seconds_on_visual_tokens": fit,
                       "probe_seconds_per_image": _stats([r["run"]["seconds"]["total"] for r in probe_new])},
        "vram": vram,
        "quality": quality(pilot_rows),
        "determinism": determinism,
        "recognition_probe": probe_summary(probe, probe_rows) if probe else None,
        "projection": {"basis": ("this machine; naive = remaining x mean seconds per pilot image; "
                                 "token_adjusted = sum over remaining images of a + b x visual "
                                 "tokens (fit above), tokens from each image's index dimensions"),
                       "datasets": projection},
        "count_corrections": count_corrections(stats, index_files),
    }
    if report["recognition_probe"] is not None:
        report["recognition_probe"]["rows_written_to_factify2_ledger"] = len(probe_new)
    return report


def print_pilot(report: dict[str, Any]) -> None:
    t, v = report["throughput"], report["vram"]
    p = report["pilot"]
    log(f"\npilot: {p['dataset']}, {p['processed']} images {p['status']}")
    if t["seconds_per_image"]:
        s = t["stage_seconds"]

        def mean(stage: str):
            return (s.get(stage) or {}).get("mean")

        log(f"  throughput {t['images_per_second']} images/s; {mean('total')} s/image "
            f"(p50 {s['total']['p50']}, p90 {s['total']['p90']}): ocr {mean('ocr')}, "
            f"description {mean('description')}, entities {mean('entities')}; "
            f"model load {t['model_load_seconds']} s")
    if v:
        log(f"  peak VRAM: torch reserved {v['torch_peak_reserved_gb']} GB (allocated "
            f"{v['torch_peak_allocated_gb']}); device peak about {v['device_peak_estimate_gb']} "
            f"of {v['device_total_gb']} GB")
    d = report["determinism"]
    log(f"  determinism: {d['identical']}/{d['checked']} byte-identical ({d['kind']})")
    q = report["quality"]
    free, forced = q["entities"]["free"], q["entities"]["forced"]
    log(f"  entity answers {q['entity_format']}; complete and free {q['entities_complete_free']} "
        f"of {q['images_seen_by_model']}; skeleton replacements {q['entity_skeleton_events']}")
    log(f"  entity lines: free {free['lines']}, forced {forced['lines']}")
    log(f"  quality, free lines: PERSON {free['status'].get('PERSON', {})}; person names, model "
        f"tag vs OCR {free['person_names_tag_vs_ocr']}; images naming a person not written in "
        f"them {free['images_naming_a_person_not_written']}; description flags "
        f"{q['description_flags']}; errors {q['component_errors']}")
    log(f"  quality, forced lines: status {forced['status']}; names, model tag vs OCR "
        f"{forced['names_tag_vs_ocr']}")
    probe = report["recognition_probe"]
    if probe:
        c, p, th = probe["counts"], probe["precision"], probe["thresholds"]
        log(f"  recognition probe: {c.get('images', 0)} Factify2 train images of public figures; "
            f"figure's name not written in {c.get('name_not_written', 0)}, figure named from "
            f"appearance in {c.get('name_not_written_names_claimed_figure', 0)} (recall-style "
            f"{probe['rate_claimed_figure_named_where_name_not_written']}); model tag vs OCR "
            f"{probe['person_tag_vs_ocr']}")
        log(f"    precision {p['value']}: {p['claimed_figure']} of {p['names']} names from appearance "
            f"alone (as first counted: {p['not_fully_written']['claimed_figure']} of "
            f"{p['not_fully_written']['names']}); forced PERSON lines {probe['forced_person_lines']}")
        log(f"    to replace a gallery: precision {th['verifier']['precision_needed']} (met: "
            f"{th['verifier']['met']}); as a candidate a gallery verifies: gallery FMR/TAR at most "
            f"{th['candidate']['max_gallery_false_accept_ratio']}")
        log(f"    {probe['verdict']}")
        if probe["conclusion"]:
            log(f"    {probe['conclusion']}")
    log("  projected wall-clock on this machine (naive / token-adjusted hours):")
    for ds, info in report["projection"]["datasets"].items():
        extra = (f"; per 10k {info.get('hours_per_10k_naive')} / "
                 f"{info.get('hours_per_10k_token_adjusted')}" if ds == "fakeddit" else "")
        log(f"    {ds:10s} {info['remaining']:>7,} of {info['distinct_images']:>7,} images: "
            f"{info['hours_naive']} / {info['hours_token_adjusted']}{extra}")
    log(f"  counts: {report['count_corrections']['note']}")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def warm_up() -> None:
    """Load the registries the gates read (the placeholder YAML takes seconds
    to parse) before anything is timed, so the first image is not charged."""
    from scripts.placeholders import registry

    registry()
    images.fingerprint_rejections()


def scope_for(through: str) -> tuple[str, ...]:
    return ORDER[:ORDER.index(through) + 1]


def backends(args):
    if args.stub:
        return StubVLM(getattr(args, "stub_delay", 0.0)), StubOCR()
    # Nothing past fetch-models may reach the network.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return QwenVLM(), EasyOcrBackend(args.ocr_device)


def worklists(args, scope: Sequence[str]) -> tuple[dict[str, list[WorkItem]], dict[str, dict[str, int]]]:
    if args.worklist:
        found = load_worklist_file(args.worklist)
        return {ds: found[ds] for ds in scope if ds in found}, {}
    lists, stats, cache = {}, {}, {}
    for ds in scope:
        items, stats[ds] = build_worklist(ds, cache=cache)
        lists[ds] = items[:args.fakeddit_n] if ds == "fakeddit" else items
    return lists, stats


def cmd_run(args) -> int:
    scope = scope_for(args.through)
    vlm, ocr = backends(args)
    vlm.prepare()
    ocr.prepare()
    ctx = make_context(vlm, ocr)
    write_config(ctx)
    log(f"config {ctx.config_id} -> {ctx.out_dir}")
    lists, stats = worklists(args, scope)
    runner = Runner(vlm, ocr, ctx)
    try:
        for ds in lists:
            runner.open(ds)
        if args.pilot is not None:
            return run_pilot(args, runner, lists, stats)
        todo = [it for ds in scope for it in runner.pending(lists.get(ds, []), args.retry_failed)]
        for ds in lists:
            log(f"  [{ds}] {len(lists[ds]):,} images, {len(runner.pending(lists[ds], args.retry_failed)):,} to do")
        if not todo:
            log("nothing to do")
            return 0
        t = time.perf_counter()
        vlm.load()
        ocr.load()
        warm_up()
        log(f"models loaded in {time.perf_counter() - t:.0f}s")
        started = time.perf_counter()
        for i, item in enumerate(todo, 1):
            log(_progress(i, len(todo), runner.step(item), started))
        for ds in lists:
            log(f"  [{ds}] {dict(Counter(s['status'] for s in runner.summaries[ds].values()))}")
        return 0
    except LockHeld as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    except FatalError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    finally:
        runner.close()


def recheck(pairs: Sequence[tuple[WorkItem, str]], vlm, ocr, ctx: Context, kind: str) -> dict[str, Any]:
    """Re-run images and compare result_sha256 with what was written."""
    out: dict[str, Any] = {"kind": kind, "checked": 0, "identical": 0, "mismatched": []}
    for item, expected in pairs:
        row = process(item, vlm, ocr, ctx)
        out["checked"] += 1
        if row["result_sha256"] == expected:
            out["identical"] += 1
        else:
            out["mismatched"].append(item.sha256[:12])
    return out


def run_pilot(args, runner: Runner, lists: dict[str, list[WorkItem]],
              stats: dict[str, dict[str, int]]) -> int:
    vlm, ocr, ctx = runner.vlm, runner.ocr, runner.ctx
    target = next((ds for ds in ORDER if ds in lists and runner.pending(lists[ds])), None)
    batch = runner.pending(lists[target])[:args.pilot] if target else []
    # Every dataset's work list, for the projections and the count corrections
    # -- reduced to (sha256, tokens) and the indexes dropped BEFORE the models
    # load: holding five image indexes and 140k Fakeddit items through the run
    # is what got the first attempt reaped for memory on a 16 GB machine.
    all_stats = dict(stats)
    sizes = {ds: sizes_of(items) for ds, items in lists.items()}
    index_files: dict[str, int] = {}
    probe: list[WorkItem] = []
    if not args.worklist:
        cache: dict = {}
        for ds in ORDER:
            items = lists.get(ds)
            if items is None:
                items, all_stats[ds] = build_worklist(ds, cache=cache)
                sizes[ds] = sizes_of(items)
            if ds == "factify2" and args.probe_n:
                probe = probe_items(args.probe_n, items, cache=cache)
            del items
        index_files = {ds: len(cache[ds]) if ds in cache else len(images.load_index(ds))
                       for ds in ORDER}
        del cache
        gc.collect()
    if probe:
        runner.open("factify2")
    probe_todo = runner.pending(probe)
    log(f"pilot: {len(batch)} {target} images, recognition probe {len(probe)} "
        f"({len(probe_todo)} to compute)")
    if not batch and not probe_todo:
        log("nothing left to measure in scope")
        return 0
    t = time.perf_counter()
    vlm.load()
    ocr.load()
    warm_up()
    load_seconds = time.perf_counter() - t
    log(f"models loaded in {load_seconds:.0f}s")
    started = time.perf_counter()
    pilot_rows = []
    for i, item in enumerate(batch, 1):
        pilot_rows.append(runner.step(item))
        log(_progress(i, len(batch), pilot_rows[-1], started))
    wall = time.perf_counter() - started
    probe_new = []
    for i, item in enumerate(probe_todo, 1):
        probe_new.append(runner.step(item))
        log(_progress(i, len(probe_todo), probe_new[-1], started))
    ok = [(it, r["result_sha256"]) for it, r in zip(batch, pilot_rows) if r["status"] == "ok"]
    log(f"determinism: re-running {min(args.determinism_check, len(ok))} images")
    determinism = recheck(ok[:args.determinism_check], vlm, ocr, ctx,
                          "same process, after the whole pilot ran in between")
    probe_rows: dict[str, dict[str, Any]] = {}
    if probe:
        wanted = {it.sha256 for it in probe}
        _, kept, _ = scan_ledger(ctx.out_dir / "factify2.jsonl", keep=lambda r: r["sha256"] in wanted)
        probe_rows = {r["sha256"]: r for r in kept}
    done = {ds: runner.summaries.get(ds) or scan_ledger(ctx.out_dir / f"{ds}.jsonl")[0]
            for ds in sizes}
    report = build_pilot_report(
        ctx=ctx, vlm=vlm, target=target, requested=args.pilot, pilot_rows=pilot_rows,
        probe=probe, probe_rows=probe_rows, probe_new=probe_new, load_seconds=load_seconds,
        determinism=determinism, memory=runner.memory, sizes=sizes, stats=all_stats,
        index_files=index_files, done=done, wall_seconds=wall,
        labels=_labels(target) if target and not args.worklist else {})
    body = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (ctx.out_dir / "pilot.json").write_text(body, encoding="utf-8", newline="\n")
    if not args.stub and not args.worklist:
        REPORTS.mkdir(parents=True, exist_ok=True)
        PILOT_REPORT.write_text(body, encoding="utf-8", newline="\n")
    print_pilot(report)
    log(f"-> {ctx.out_dir / 'pilot.json'}" + ("" if args.stub or args.worklist else f" and {PILOT_REPORT}"))
    if determinism["identical"] != determinism["checked"]:
        print("STOPPED: the same image did not give byte-identical output; do not scale.",
              file=sys.stderr)
        return 2
    return 0


def cmd_recheck(args) -> int:
    """A FRESH process re-runs finished rows: determinism across processes."""
    vlm, ocr = backends(args)
    vlm.prepare()
    ocr.prepare()
    ctx = make_context(vlm, ocr)
    summaries, _, _ = scan_ledger(ctx.out_dir / f"{args.dataset}.jsonl", dataset=args.dataset,
                                  config_id=ctx.config_id)
    ok = sorted(sha for sha, s in summaries.items() if s["status"] == "ok")[:args.n]
    if not ok:
        log(f"no finished {args.dataset} rows under config {ctx.config_id}")
        return 1
    if args.worklist:
        by_sha = {it.sha256: it for it in load_worklist_file(args.worklist).get(args.dataset, [])}
    else:
        by_sha = {it.sha256: it for it in build_worklist(args.dataset)[0]}
    vlm.load()
    ocr.load()
    result = recheck([(by_sha[s], summaries[s]["result_sha256"]) for s in ok], vlm, ocr, ctx,
                     "fresh process against rows an earlier process wrote")
    log(json.dumps(result))
    return 0 if result["identical"] == result["checked"] else 2


def cmd_status(args) -> int:
    out: dict[str, Any] = {}
    if not OUT_ROOT.is_dir():
        log(json.dumps(out))
        return 0
    for cfg_dir in sorted(p for p in OUT_ROOT.iterdir() if (p / "config.json").is_file()):
        entry: dict[str, Any] = {}
        for ledger in sorted(cfg_dir.glob("*.jsonl")):
            if ledger.name.endswith(".torn.jsonl"):
                continue
            machines: Counter = Counter()
            # per image, its last row's entity answer: (form, finish, tokens, forced lines or
            # None when the row's lines are not marked)
            answers: dict[str, tuple[str, str, int, tuple[str, ...] | None]] = {}

            def tally(r: dict[str, Any]) -> bool:
                machines.update([f"{r['provenance']['machine']['host_id']} "
                                 f"{r['provenance']['machine']['gpu']}"])
                block = r.get("entities")
                if block is None:
                    answers.pop(r["sha256"], None)
                else:
                    marked = all("forced" in block[c] for c in CATEGORIES)
                    answers[r["sha256"]] = (
                        entity_format(block), str(block["finish"]), block["n_tokens"],
                        tuple(c for c in CATEGORIES if block[c]["forced"]) if marked else None)
                return False

            summaries, _, torn = scan_ledger(ledger, schema=args.verify, keep=tally)
            forms: dict[str, dict[str, Any]] = {}
            for form, finish, tokens, _forced in answers.values():
                f = forms.setdefault(form, {"images": 0, "finish": Counter(), "tokens": []})
                f["images"] += 1
                f["finish"][finish] += 1
                f["tokens"].append(tokens)
            marked = [lines for *_rest, lines in answers.values() if lines is not None]
            entry[ledger.stem] = {
                "images": len(summaries),
                "status": dict(Counter(s["status"] for s in summaries.values())),
                "reasons": dict(Counter(s["reason"] for s in summaries.values() if s["reason"])),
                "component_errors": dict(Counter(e for s in summaries.values() for e in s["errors"])),
                "retried": sum(1 for s in summaries.values() if s["attempt"] > 1),
                # how each entity answer kept or lost its five lines: truncated
                # (finish "length"), another format, or stopped early
                "entity_format": {form: {"images": f["images"], "finish": dict(f["finish"]),
                                         "tokens": _stats(f["tokens"])}
                                  for form, f in sorted(forms.items())},
                # lines the skeleton forced, per category; null when no row is marked
                "forced_lines": (dict(Counter(c for lines in marked for c in lines))
                                 if marked else None),
                "machines": dict(machines), "torn_tail_bytes": torn}
        out[cfg_dir.name] = entry
    log(json.dumps(out, indent=2))
    return 0


def cmd_fetch_models(args) -> int:
    """The one command that opens a socket: Qwen at the pinned commit, EasyOCR's weights."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    from scripts.manifest import sha256

    MODELS.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise SystemExit(f"the Hub resolved {MODEL_REVISION} to {info.sha}; refusing")
    licence = getattr(info.card_data, "license", None) if info.card_data else None
    if licence != MODEL_LICENCE:
        raise SystemExit(f"{MODEL_ID} now states licence {licence!r}, not {MODEL_LICENCE!r}; "
                         "read it before fetching")
    wanted = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in MODEL_FILES)]
    need_gb = sum(s.size or 0 for s in wanted) / 1024 ** 3
    have_gb = free_gb(MODELS)
    log(f"{MODEL_ID}@{MODEL_REVISION[:12]} ({licence}): {len(wanted)} files, {need_gb:.2f} GiB; "
        f"{have_gb:.1f} GiB free at {MODELS}")
    if have_gb < 1.5 * need_gb:
        raise SystemExit(f"refusing: {have_gb:.1f} GiB free, {1.5 * need_gb:.1f} GiB needed")
    snap = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_CACHE),
                                  allow_patterns=list(MODEL_FILES)))
    bad = []
    for s in wanted:
        path = snap / s.rfilename
        if not path.is_file() or path.stat().st_size != s.size:
            bad.append(f"{s.rfilename}: size")
        elif s.lfs is not None and sha256(path) != s.lfs.sha256:
            bad.append(f"{s.rfilename}: sha256")
    if bad:
        raise SystemExit(f"downloaded files do not match the Hub's metadata: {bad}")
    log(f"  verified {len(wanted)} files against the Hub (sizes; sha256 of every LFS file)")
    EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
    if args.easyocr_from:
        for name in EASYOCR_WEIGHTS:
            if not (EASYOCR_DIR / name).is_file():
                shutil.copy2(args.easyocr_from / name, EASYOCR_DIR / name)
    elif any(not (EASYOCR_DIR / name).is_file() for name in EASYOCR_WEIGHTS):
        import easyocr

        easyocr.Reader(list(OCR_LANGS), gpu=False, model_storage_directory=str(EASYOCR_DIR),
                       user_network_directory=str(EASYOCR_DIR / "user_network"),
                       download_enabled=True, verbose=False)
    verify_easyocr_weights(EASYOCR_DIR)
    log(f"  EasyOCR weights verified in {EASYOCR_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-models", help="download the pinned weights (network)")
    f.add_argument("--easyocr-from", type=Path, default=None,
                   help="copy EasyOCR's weights from this directory instead of downloading")
    r = sub.add_parser("run", help="process images; resumable")
    r.add_argument("--through", choices=ORDER, default="verite",
                   help="process ORDER up to and including this dataset (default: verite only)")
    r.add_argument("--fakeddit-n", type=int, default=None,
                   help="size of the Fakeddit sample (required with --through fakeddit)")
    r.add_argument("--pilot", type=int, default=None, metavar="N",
                   help="process N images, report throughput / VRAM / projections, and stop")
    r.add_argument("--probe-n", type=int, default=PROBE_N,
                   help="pilot only: Factify2 train public-figure images for the recognition probe")
    r.add_argument("--determinism-check", type=int, default=3,
                   help="pilot only: images re-run at the end and byte-compared")
    r.add_argument("--retry-failed", action="store_true",
                   help="re-attempt rows whose status is failed or partial")
    r.add_argument("--worklist", type=Path, default=None,
                   help="JSONL of work items instead of the processed layer")
    for p in (r, c := sub.add_parser("recheck", help="re-run finished rows in a fresh process")):
        p.add_argument("--ocr-device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--stub", action="store_true",
                       help="deterministic stand-in models: checks everything but the models")
        p.add_argument("--stub-delay", type=float, default=0.0, help=argparse.SUPPRESS)
    c.add_argument("--dataset", choices=ORDER, default="verite")
    c.add_argument("--n", type=int, default=3)
    c.add_argument("--worklist", type=Path, default=None)
    s = sub.add_parser("status", help="offline counts per ledger")
    s.add_argument("--verify", action="store_true", help="validate every row against the schema")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.through == "fakeddit" and not args.fakeddit_n:
            parser.error("--through fakeddit needs --fakeddit-n N: the sample size is a decision")
        if args.fakeddit_n is not None and args.through != "fakeddit":
            parser.error("--fakeddit-n only applies with --through fakeddit")
        if args.pilot is not None and args.pilot < 1:
            parser.error("--pilot N needs N >= 1")
        return cmd_run(args)
    if args.command == "recheck":
        return cmd_recheck(args)
    if args.command == "status":
        return cmd_status(args)
    return cmd_fetch_models(args)


if __name__ == "__main__":
    raise SystemExit(main())
