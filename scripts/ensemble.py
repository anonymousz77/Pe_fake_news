#!/usr/bin/env python3
"""The ensemble (step 8): pretrained scorers, stacked by one small model.

The person's plan and decisions (2026-11-10):
- Scorers: OpenCLIP ViT-H/14 and SigLIP so400m (image-text similarity), BLIP
  and BLIP-2 (image-text matching heads), a DeBERTa-v3 NLI cross-encoder
  (the claim against the image's description and OCR, and against the
  retrieved documents), our own consistency verdicts, and Track B's
  text-side stance. Each pinned in ``configs/ensemble.yaml``; nothing
  non-commercial. OpenAI's CLIP is out of the ensemble (its card rules out
  deployed use) and stays C's baseline only.
- The combiner fits on M4FC train alone, true against false, its form chosen
  by grouped cross-validation within train. Val reports; VERITE is
  evaluation only and fits nothing. Factify2's source-disjoint train enters
  only a labelled variant: its labels say whether a document supports the
  claim, never whether the image fits it.
- Explanations still come from the consistency verdicts; the ensemble
  decides the flag.

Machines:
  [server] python scripts/ensemble.py fetch-models          # the one network command
  [laptop] python scripts/ensemble.py inputs                # id-free scorer inputs
  [server] python scripts/ensemble.py score --scorer NAME   # GPU, resumable
  [server] python scripts/ensemble.py recheck --scorer NAME --n 3
  [laptop] python scripts/ensemble.py fit                   # combiner, ablation -> gate E1's report

The scorers' inputs carry neutral keys and text only; images are found by
sha256 through a separate locator file, and every image's bytes are checked
against its sha256 before a model sees them. Labels never leave the laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import CONFIGS, INTERIM, MODELS, PROJECT_ROOT, REPORTS, free_gb  # noqa: E402
from scripts import consistency as cs  # noqa: E402

ENS_CONFIG = CONFIGS / "ensemble.yaml"
ENS_DIR = INTERIM / "ensemble"
PAIRS = ENS_DIR / "pairs_in.jsonl"
NLI_IN = ENS_DIR / "nli_in.jsonl"
IMAGES = ENS_DIR / "images.jsonl"
RECORDS = ENS_DIR / "records.jsonl"  # laptop only: keys to classes and splits; never shipped
SCORES = ENS_DIR / "scores"
MODEL_MANIFEST = ENS_DIR / "models.json"
HF_CACHE = MODELS / "hf"
REPORT_E1 = REPORTS / "ensemble_e1.json"
REPORT_E1_MD = REPORTS / "ensemble_e1.md"
TRACKB_REQUIREMENTS = PROJECT_ROOT / "requirements-trackb.txt"
#: Track B's text-side check, read and never written (the person's interface,
#: 2026-11-10): joined on cs.claim_key.
STANCE = INTERIM / "textcheck" / "verdicts.jsonl"
STANCE_KEYS = ("claim_key", "judge", "verdict", "reason_code", "p_supported", "p_refuted", "p_nei",
               "doc_ids", "status")
SEED = cs.SEED
RUNNER_VERSION = 1
BATCH = {"dual": 32, "itm": 16, "nli": 16}
#: How far a recheck in a fresh process may sit from the ledger (GPU float order).
RECHECK_TOLERANCE = 1e-3
NLI_SLOTS = ("description", "ocr", *(f"doc{i}" for i in range(1, cs.PREDICT_DOCS + 1)))
FOLDS = 5


#: The ensemble's own files ("eval": M4FC train+val, VERITE, Factify2's train)
#: and the final test's, kept apart: the official test is opened once.
FINAL_DIR = ENS_DIR / "final_test"
LIVE_ENS = ENS_DIR / "live"  # the scorers' inputs and ledgers for live requests
def _set(name: str) -> dict[str, Path]:
    """A set's paths, read at call time (tests move ENS_DIR's files)."""
    if name == "eval":
        return {"pairs": PAIRS, "nli": NLI_IN, "images": IMAGES, "scores": SCORES}
    base = {"final": FINAL_DIR, "live": LIVE_ENS}[name]
    return {"pairs": base / "pairs_in.jsonl", "nli": base / "nli_in.jsonl", "images": base / "images.jsonl",
            "scores": base / "scores"}


def _config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(ENS_CONFIG.read_text(encoding="utf-8"))


def models() -> dict[str, dict[str, Any]]:
    """The ensemble's scorers."""
    return _config()["models"]


def baselines() -> dict[str, dict[str, Any]]:
    """Scorers run as baselines only, never as ensemble inputs (CLIP L/14)."""
    return _config().get("baselines") or {}


def all_scorers() -> dict[str, dict[str, Any]]:
    return {**models(), **baselines()}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sidecar(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


# --------------------------------------------------------------------------
# fetch-models: the pinned weights (server; the one network command)
# --------------------------------------------------------------------------


def cmd_fetch_models(args) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    from scripts.manifest import sha256

    api, manifest = HfApi(), {}
    specs = {k: v for k, v in all_scorers().items() if not args.scorer or k in args.scorer}
    plan = []
    for name, spec in specs.items():
        info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise SystemExit(f"{spec['repo']}: the Hub resolved {spec['revision']} to {info.sha}; refusing")
        licence = getattr(info.card_data, "license", None) if info.card_data else None
        if licence != spec["licence"]:
            raise SystemExit(f"{spec['repo']} now states licence {licence!r}, not {spec['licence']!r}: "
                             "read it before fetching")
        by_name = {s.rfilename: s for s in info.siblings}
        if missing := [f for f in spec["files"] if f not in by_name]:
            raise SystemExit(f"{spec['repo']}@{spec['revision'][:12]} has no {missing}")
        plan.append((name, spec, [by_name[f] for f in spec["files"]]))
    need = sum(s.size or 0 for _n, _s, files in plan for s in files) / 1024 ** 3
    have = free_gb(MODELS) if MODELS.exists() else free_gb(PROJECT_ROOT)
    print(f"{len(plan)} models, {need:.2f} GiB; {have:.1f} GiB free at {MODELS}", flush=True)
    if have < 1.5 * need:
        raise SystemExit(f"refusing: {have:.1f} GiB free, {1.5 * need:.1f} GiB needed")
    for name, spec, files in plan:
        got = {}
        for s in files:
            path = Path(hf_hub_download(spec["repo"], s.rfilename, revision=spec["revision"],
                                        cache_dir=str(HF_CACHE)))
            if path.stat().st_size != s.size:
                raise SystemExit(f"{spec['repo']}/{s.rfilename}: size {path.stat().st_size} != {s.size}")
            digest = sha256(path)
            if s.lfs is not None and digest != s.lfs.sha256:
                raise SystemExit(f"{spec['repo']}/{s.rfilename}: sha256 differs from the Hub's")
            got[s.rfilename] = {"bytes": s.size, "sha256": digest}
        manifest[name] = {"repo": spec["repo"], "revision": spec["revision"], "licence": spec["licence"],
                          "files": got}
        print(f"  [fetch] {name}: {spec['repo']}@{spec['revision'][:12]} ({spec['licence']}), "
              f"{len(got)} files verified", flush=True)
    old = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8")) if MODEL_MANIFEST.is_file() else {}
    cs.write_json(MODEL_MANIFEST, {**old, **manifest})
    return 0


# --------------------------------------------------------------------------
# inputs (laptop): every record the ensemble scores, id-free
# --------------------------------------------------------------------------


def factify2_records(sha_of: Callable[[str], str | None]) -> tuple[list[dict[str, Any]], Counter]:
    """Factify2's source-disjoint train rows, through reason_code_rows: each
    with its claim and its claim image (never the document's), for the
    labelled variant only."""
    from scripts.processed_loader import reason_code_rows

    out, tally = [], Counter()
    for row in reason_code_rows("factify2", split="train", purpose="train"):
        claim = " ".join((row.get("text") or "").split())
        image = next((p for p in row.get("image_paths") or () if Path(p).stem.endswith("_claim_image")), None)
        sha = sha_of(image) if image else None
        if not claim or not sha:
            tally["no claim text" if not claim else "no usable claim image"] += 1
            continue
        out.append({"key": cs.record_key("factify2", row), "dataset": "factify2", "split": "train",
                    "class": row["label"], "reason_code": row["reason_code"], "sha256": sha, "path": image,
                    "claim": row.get("text") or "", "claim_key": cs.claim_key(row.get("text") or ""), "row": row})
        tally["kept"] += 1
    return out, tally


def image_texts(dataset: str, shas: set[str]) -> dict[str, tuple[str | None, str | None]]:
    """sha256 -> (OCR text, description) from the image_text ledger,
    streamed: only the images asked for are held, and a failed component is
    None (cs.image_part's rule). The last row of an image wins."""
    from scripts.image_text import OUT_ROOT

    path = OUT_ROOT / cs.IMAGE_TEXT_CONFIG / f"{dataset}.jsonl"
    out: dict[str, tuple[str | None, str | None]] = {}
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["sha256"] not in shas:
                continue
            if row["provenance"]["config_id"] != cs.IMAGE_TEXT_CONFIG:
                raise SystemExit(f"{path.name}: a row of another image_text configuration")
            ocr, desc = row.get("ocr") or {}, row.get("description") or {}
            out[row["sha256"]] = (None if ocr.get("error") else ocr.get("text"),
                                  None if desc.get("error") else desc.get("text"))
    return out


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def nli_premises(ocr: str | None, desc: str | None, docs: Sequence[cs.Document]) -> dict[str, str]:
    """slot -> premise: the image's description and OCR as the predictor saw
    them (cut at its TEXT_CHARS), and each retrieved document's title and
    excerpt. Empty ones are left out."""
    out = {"description": cs._clip(desc or "", cs.TEXT_CHARS), "ocr": cs._clip(ocr or "", cs.TEXT_CHARS)}
    for i, d in enumerate(docs[:cs.PREDICT_DOCS], 1):
        out[f"doc{i}"] = clean(f"{cs._clip(d.title, cs.TITLE_CHARS)}. {d.excerpt}")
    return {k: v for k, v in out.items() if v.strip()}


def build_inputs(records: Sequence[dict[str, Any]], texts: dict[str, dict[str, tuple[str | None, str | None]]],
                 docs: dict[str, tuple[cs.Document, ...]]
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], Counter]:
    """(image-text pairs, NLI pairs, sha256 -> image path, counts). Every
    text a model reads is a whitelisted field -- the claim, the image's OCR
    and description, a retrieved title and excerpt. Each is refused if it
    holds its record's id in a form its own content does not explain (as B's
    and C's prompts were: a claim may say what its own slug says), or an id
    shape at all, so the runner's strict shape check never meets one."""
    pairs, nli, images, tally = [], [], {}, Counter()

    def check(text: str, source: str, forms, what: str) -> None:
        cs.refuse_ids(text, forms, explained=source, what=what)
        if shapes := [s.pattern for s in cs.ID_SHAPES if s.search(text.casefold())]:
            raise cs.GuardError(f"{what} holds an id shape ({len(shapes)}): refusing")

    for r in records:
        forms = cs.id_forms(r["row"])
        claim = clean(r["claim"])
        if not claim or not r["sha256"]:
            tally[f"{r['dataset']}: skipped (no claim or image)"] += 1
            continue
        check(claim, r["claim"], forms, "a claim the ensemble reads")
        pairs.append({"key": r["key"], "sha256": r["sha256"], "text": claim})
        images.setdefault(r["sha256"], r["path"])
        ocr, desc = texts.get(r["dataset"], {}).get(r["sha256"], (None, None))
        record_docs = docs.get(r["sha256"], ()) if r["dataset"] == "m4fc" else ()
        source = "\n".join([ocr or "", desc or "", *(f"{d.title}\n{d.excerpt}" for d in record_docs)])
        for slot, premise in nli_premises(ocr, desc, record_docs).items():
            check(premise, source, forms, f"an NLI premise ({slot})")
            nli.append({"key": cs.neutral_key("n", f"{r['key']}|{slot}"), "record": r["key"], "slot": slot,
                        "premise": premise, "hypothesis": claim})
            tally[f"nli {slot.rstrip('0123456789')}"] += 1
        tally[f"{r['dataset']} {r['split']}"] += 1
    return pairs, nli, images, tally


def write_input(path: Path, rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    n = write_jsonl(path, rows)
    meta = {"kind": kind, "rows": n, "sha256": cs.sha256_file(path), "id_check": {"passed": True},
            "written_at": cs.utcnow()}
    cs.write_json(sidecar(path), meta)
    return meta


def cmd_inputs(args) -> int:
    sha_of = cs.sha_lookup()
    records = cs.c_records(sha_of)
    f2, f2_tally = factify2_records(sha_of)
    records += f2
    texts = {ds: image_texts(ds, {r["sha256"] for r in records if r["dataset"] == ds and r["sha256"]})
             for ds in ("m4fc", "verite", "factify2")}
    _cid, _items, held = cs.retrieval_ledger(cs.M4FC_SPLITS, sha_of)
    docs = {s: cs.documents_of(r) for s, r in held.items()}
    pairs, nli, images, tally = build_inputs(records, texts, docs)
    write_input(PAIRS, pairs, "pairs")
    write_input(NLI_IN, nli, "nli")
    write_jsonl(IMAGES, [{"sha256": s, "path": p} for s, p in sorted(images.items())])
    write_jsonl(RECORDS, [{k: r[k] for k in ("key", "dataset", "split", "class", "sha256", "claim_key")}
                          | {"reason_code": r.get("reason_code")} for r in records])
    print(f"{len(pairs):,} image-text pairs, {len(images):,} distinct images, {len(nli):,} NLI pairs -> "
          f"{cs.rel(ENS_DIR)}")
    print("  " + json.dumps(cs.ordered(tally)))
    print("  factify2: " + json.dumps(cs.ordered(f2_tally)))
    return 0


def read_input(path: Path, kind: str) -> list[dict[str, Any]]:
    """An input file, refused unless it is the one its sidecar describes and
    no text in it holds an id shape."""
    meta = json.loads(sidecar(path).read_text(encoding="utf-8"))
    if cs.sha256_file(path) != meta["sha256"] or meta["kind"] != kind or not meta["id_check"]["passed"]:
        raise SystemExit(f"{path.name} is not the {kind} input its sidecar describes: refusing")
    rows = read_jsonl(path)
    for r in rows:
        for field in ("text", "premise", "hypothesis"):
            if field in r:
                cs.refuse_ids(r[field], what=f"input row {r['key']}")
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit(f"{path.name}: a key appears twice")
    return rows


# --------------------------------------------------------------------------
# score (server, GPU): one scorer at a time, resumable
# --------------------------------------------------------------------------


class Images:
    """sha256 -> the image as a model reads it (EXIF-oriented RGB, alpha on
    white: image_text's prepare), refused if the bytes are not that sha256's."""

    def __init__(self, locator: Path | None = None) -> None:
        self.path_of = {r["sha256"]: r["path"] for r in read_jsonl(locator or IMAGES)}

    def __call__(self, sha: str):
        from scripts.image_text import prepare, read_bytes

        body = read_bytes(self.path_of[sha])
        if hashlib.sha256(body).hexdigest() != sha:
            raise cs.GuardError("an image's bytes are not the ones its sha256 names")
        return prepare(body)[0]


def load_scorer(name: str, spec: dict[str, Any], device: str):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, Blip2ForImageTextRetrieval,
                              BlipForImageTextRetrieval, CLIPModel, SiglipModel)

    kw = {"revision": spec["revision"], "cache_dir": str(HF_CACHE), "local_files_only": True}
    cls = {"clip": CLIPModel, "siglip": SiglipModel, "blip": BlipForImageTextRetrieval,
           "blip2": Blip2ForImageTextRetrieval, "nli": AutoModelForSequenceClassification}[spec["loader"]]
    torch.manual_seed(SEED)
    # full float32: TF32 (on by default for cuDNN's convolutions) let a recheck in a fresh
    # process differ by 1.15e-3 in cosine; deterministic kernels, chosen without benchmarking
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = cls.from_pretrained(spec["repo"], torch_dtype=torch.float32, **kw).eval().to(device)
    proc = (AutoTokenizer if spec["loader"] == "nli" else AutoProcessor).from_pretrained(spec["repo"], **kw)
    return model, proc


def scorer_config(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    from scripts.image_text import software_versions

    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))[name]
    return {"runner": RUNNER_VERSION, "scorer": name, "repo": spec["repo"], "revision": spec["revision"],
            "files": {f: v["sha256"] for f, v in manifest["files"].items()}, "dtype": "float32", "precision": "TF32 off (matmul and cuDNN); cuDNN deterministic, no benchmark",
            "text_tokens": spec["text_tokens"], "batch": BATCH[spec["kind"]],
            "image": "image_text.prepare (EXIF-oriented RGB, alpha on white), then the model's processor",
            "software": software_versions(("torch", "transformers", "pillow", "tokenizers", "sentencepiece"))}


def _norm(t):
    return t / t.norm(dim=-1, keepdim=True)


def _tensor(out):
    from scripts.clip_pairs import _features

    return _features(out)


def dual_scores(model, proc, spec, rows: Sequence[dict[str, Any]], images: Images, device: str,
                progress: Callable[[int], None]) -> dict[str, dict[str, Any]]:
    """Cosine of each pair's image and text embeddings, each distinct image
    and text embedded once."""
    img, txt, errors = dual_embed(model, proc, spec, rows, images, device, progress)
    return {r["key"]: ({"status": "error", "error": errors[r["sha256"]]} if r["sha256"] in errors else
                       {"status": "ok", "cosine": round(float(img[r["sha256"]] @ txt[r["text"]]), 6)})
            for r in rows}


def dual_embed(model, proc, spec, rows: Sequence[dict[str, Any]], images: Images, device: str,
               progress: Callable[[int], None]):
    """(sha256 -> unit image embedding, text -> unit text embedding, sha256 ->
    error) for the pairs' distinct images and texts."""
    import torch

    shas = list(dict.fromkeys(r["sha256"] for r in rows))
    texts = list(dict.fromkeys(r["text"] for r in rows))
    img, txt, errors = {}, {}, {}
    b = BATCH["dual"]
    with torch.inference_mode():
        for i in range(0, len(shas), b):
            part, pil = [], []
            for s in shas[i:i + b]:
                try:
                    pil.append(images(s))
                    part.append(s)
                except Exception as exc:  # noqa: BLE001 -- an image that cannot be read is an error row
                    errors[s] = type(exc).__name__
            if part:
                inputs = proc(images=pil, return_tensors="pt").to(device)
                img.update(zip(part, _norm(_tensor(model.get_image_features(**inputs))).float().cpu()))
            progress(min(i + b, len(shas)))
        pad = {"padding": "max_length"} if spec["loader"] == "siglip" else {"padding": True}
        for i in range(0, len(texts), 64):
            part = texts[i:i + 64]
            inputs = proc(text=part, truncation=True, max_length=spec["text_tokens"], return_tensors="pt",
                          **pad).to(device)
            txt.update(zip(part, _norm(_tensor(model.get_text_features(**inputs))).float().cpu()))
    return img, txt, errors


def itm_scores(model, proc, spec, rows: Sequence[dict[str, Any]], images: Images, device: str,
               progress: Callable[[int], None]) -> dict[str, dict[str, Any]]:
    """Per pair: the matching head's P(match), and the contrastive similarity
    of the same image and text (the diagonal of its similarity matrix)."""
    import torch

    out: dict[str, dict[str, Any]] = {}
    b = BATCH["itm"]
    # A BLIP-2 processor that sets num_query_tokens puts that many image tokens
    # before the text, inside max_length. One that does not (blip2-itm-vit-g's)
    # adds none and cannot take a max_length: its text and images are prepared
    # apart, the text cut at text_tokens, as the model then expects.
    queries = getattr(proc, "num_query_tokens", None) if spec["loader"] == "blip2" else None
    apart = spec["loader"] == "blip2" and not queries
    with torch.inference_mode():
        for i in range(0, len(rows), b):
            part, pil = [], []
            for r in rows[i:i + b]:
                try:
                    pil.append(images(r["sha256"]))
                    part.append(r)
                except Exception as exc:  # noqa: BLE001
                    out[r["key"]] = {"status": "error", "error": type(exc).__name__}
            if part:
                texts = [r["text"] for r in part]
                if apart:
                    enc = proc.tokenizer(texts, padding=True, truncation=True, max_length=spec["text_tokens"],
                                         return_token_type_ids=False, return_tensors="pt")
                    pix = proc.image_processor(pil, return_tensors="pt")
                    inputs = {"input_ids": enc["input_ids"].to(device),
                              "attention_mask": enc["attention_mask"].to(device),
                              "pixel_values": pix["pixel_values"].to(device)}
                else:
                    inputs = proc(images=pil, text=texts, padding=True, truncation=True,
                                  max_length=spec["text_tokens"] + (queries or 0), return_tensors="pt").to(device)
                if spec["loader"] == "blip":
                    itm = model(**inputs, use_itm_head=True).itm_score
                    itc = model(**inputs, use_itm_head=False).itm_score
                else:
                    itm = model(**inputs, use_image_text_matching_head=True).logits_per_image
                    itc = model(**inputs, use_image_text_matching_head=False).logits_per_image
                p = torch.softmax(itm.float(), dim=-1)[:, 1].cpu()
                diag = torch.diagonal(itc.float()).cpu()
                for r, pm, sim in zip(part, p, diag):
                    out[r["key"]] = {"status": "ok", "p_match": round(float(pm), 6), "itc": round(float(sim), 6)}
            progress(min(i + b, len(rows)))
    return out


def nli_scores(model, tok, spec, rows: Sequence[dict[str, Any]], device: str,
               progress: Callable[[int], None]) -> dict[str, dict[str, Any]]:
    """Per pair: P(entailment), P(neutral), P(contradiction) of the claim
    (hypothesis) given the premise, labels read from the model's own config;
    the premise is cut first when the pair is too long."""
    import torch

    labels = {i: name.casefold() for i, name in model.config.id2label.items()}
    if set(labels.values()) != {"entailment", "neutral", "contradiction"}:
        raise SystemExit(f"unexpected NLI labels {labels}")
    out = {}
    b = BATCH["nli"]
    with torch.inference_mode():
        for i in range(0, len(rows), b):
            part = rows[i:i + b]
            enc = tok([r["premise"] for r in part], [r["hypothesis"] for r in part], truncation="only_first",
                      max_length=spec["text_tokens"], padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
            for r, p in zip(part, probs):
                out[r["key"]] = {"status": "ok", **{labels[j]: round(float(p[j]), 6) for j in range(3)}}
            progress(min(i + b, len(rows)))
    return out


def run_scorer(name: str, rows: Sequence[dict[str, Any]], device: str = "cuda",
               loaded=None, set_name: str = "eval") -> dict[str, dict[str, Any]]:
    spec = all_scorers()[name]
    model, proc = loaded or load_scorer(name, spec, device)
    started = time.perf_counter()
    total = len(rows) if spec["kind"] != "dual" else len({r["sha256"] for r in rows})

    def progress(n: int) -> None:
        if n and (n % (BATCH[spec["kind"]] * 20) < BATCH[spec["kind"]] or n == total):
            per = (time.perf_counter() - started) / n
            print(f"  [{name}] {n:,}/{total:,} ok {per:.3f}s eta {per * (total - n) / 3600:.2f}h", flush=True)

    images = Images(_set(set_name)["images"])
    if spec["kind"] == "dual":
        return dual_scores(model, proc, spec, rows, images, device, progress)
    if spec["kind"] == "itm":
        return itm_scores(model, proc, spec, rows, images, device, progress)
    return nli_scores(model, proc, spec, rows, device, progress)


def ledger_path(name: str, set_name: str = "eval") -> Path:
    return _set(set_name)["scores"] / f"{name}.jsonl"


def scorer_rows(name: str, set_name: str, keys: Path | None = None) -> list[dict[str, Any]]:
    """A scorer's input rows in a set, limited to the keys a file lists."""
    paths = _set(set_name)
    nli = all_scorers()[name]["kind"] == "nli"
    rows = read_input(paths["nli"], "nli") if nli else read_input(paths["pairs"], "pairs")
    if keys:
        wanted = set(json.loads(keys.read_text(encoding="utf-8")))
        rows = [r for r in rows if r["key"] in wanted]
    return rows


def cmd_score(args) -> int:
    name, set_name = args.scorer, args.set
    spec = all_scorers()[name]
    rows = scorer_rows(name, set_name, args.keys)
    config = scorer_config(name, spec)
    cid = hashlib.sha256(cs.canonical(config)).hexdigest()[:12]
    cfg = ledger_path(name, set_name).with_name(f"{name}.config.json")
    if cfg.is_file() and json.loads(cfg.read_text(encoding="utf-8"))["config_id"] != cid:
        raise SystemExit(f"{cfg.name} is another configuration: refusing to mix runs")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cs.write_json(cfg, {"config_id": cid, **config})
    done = {r["key"] for r in read_jsonl(ledger_path(name, set_name))}
    todo = [r for r in rows if r["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{name} ({set_name}), config {cid}: {len(rows):,} rows, {len(done):,} done, {len(todo):,} to do",
          flush=True)
    if not todo:
        return 0
    got = run_scorer(name, todo, set_name=set_name)
    with ledger_path(name, set_name).open("a", encoding="utf-8", newline="\n") as handle:
        for r in todo:
            handle.write(json.dumps({"key": r["key"], "config_id": cid, **got[r["key"]]}, sort_keys=True) + "\n")
    errors = Counter(v.get("error") for v in got.values() if v["status"] != "ok")
    print(f"  {len(got):,} written; errors {json.dumps(cs.ordered(errors)) if errors else 'none'}", flush=True)
    return 0


def cmd_recheck(args) -> int:
    """n rows recomputed in this fresh process, spread over the ledger, against
    what the ledger holds: within RECHECK_TOLERANCE, or exit 2."""
    name, set_name = args.scorer, args.set
    held = {r["key"]: r for r in read_jsonl(ledger_path(name, set_name)) if r["status"] == "ok"}
    rows = [r for r in scorer_rows(name, set_name) if r["key"] in held]
    picked = [rows[int(i * len(rows) / args.n)] for i in range(args.n)] if rows else []
    got = run_scorer(name, picked, set_name=set_name)
    worst = 0.0
    for r in picked:
        for k, v in got[r["key"]].items():
            if isinstance(v, float):
                worst = max(worst, abs(v - held[r["key"]][k]))
    ok = worst <= RECHECK_TOLERANCE
    print(f"{name} recheck ({set_name}): {len(picked)} rows, largest difference {worst:.2e} "
          f"({'within' if ok else 'BEYOND'} {RECHECK_TOLERANCE:g})")
    return 0 if ok else 2


def embeddings_path(name: str, set_name: str = "eval") -> Path:
    return _set(set_name)["scores"] / f"{name}.embeddings.npz"


def text_key(text: str) -> str:
    return "txt:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def cmd_embed(args) -> int:
    """[server, GPU] A dual scorer's unit image and text embeddings for the
    pairs a key file lists: the linear probe's input. Keys "img:<sha256>" and
    "txt:<sha256 of the text>"; nothing else is stored."""
    import numpy as np

    name, set_name = args.scorer, args.set
    spec = all_scorers()[name]
    if spec["kind"] != "dual":
        raise SystemExit(f"{name} is not a dual encoder: no embeddings")
    rows = scorer_rows(name, set_name, args.keys)
    print(f"{name} embeddings ({set_name}): {len(rows):,} pairs", flush=True)
    model, proc = load_scorer(name, spec, "cuda")
    started = time.perf_counter()
    img, txt, errors = dual_embed(model, proc, spec, rows, Images(_set(set_name)["images"]), "cuda",
                                  lambda n: None)
    keys = [f"img:{s}" for s in img] + [text_key(t) for t in txt]
    vecs = np.stack([v.numpy() for v in img.values()] + [v.numpy() for v in txt.values()]).astype("float32")
    out = embeddings_path(name, set_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, keys=np.array(keys), vecs=vecs)
    cfg = {"scorer": name, "config_id": hashlib.sha256(cs.canonical(scorer_config(name, spec))).hexdigest()[:12],
           "images": len(img), "texts": len(txt), "errors": len(errors), "sha256": cs.sha256_file(out)}
    cs.write_json(out.with_suffix(".json"), cfg)
    print(f"  {len(img):,} images, {len(txt):,} texts, {len(errors)} errors, "
          f"{time.perf_counter() - started:.0f}s -> {cs.rel(out)}", flush=True)
    return 0


# --------------------------------------------------------------------------
# features (laptop): one row per record
# --------------------------------------------------------------------------

#: The combiner's features by group; a group is what the ablation leaves out.
GROUPS = {
    "openclip_h14": ["openclip_cos"],
    "siglip_so400m": ["siglip_cos"],
    "blip_itm_large": ["blip_p_match", "blip_itc"],
    "blip2_itm_g": ["blip2_p_match", "blip2_itc"],
    "nli_image": ["nli_desc_entail", "nli_desc_contra", "nli_ocr_entail", "nli_ocr_contra"],
    "nli_documents": ["nli_docs_entail", "nli_docs_contra"],
    "consistency": ["cons_score", "cons_decided", "cons_location_consistent", "cons_location_contradicted",
                    "cons_people_consistent", "cons_people_contradicted", "cons_event_consistent",
                    "cons_event_contradicted", "cons_date_consistent", "cons_date_contradicted"],
    "stance": ["stance_supported", "stance_refuted", "stance_nei"],
}
#: A feature group that some records lack carries a missing flag (the flag's
#: name, and the column whose absence it marks), so that a missing value the
#: logistic regression reads as 0 is never mistaken for a 0.
MISSING_FLAG = {"nli_image": ("nli_ocr_missing", "nli_ocr_entail"),
                "nli_documents": ("nli_docs_missing", "nli_docs_entail"),
                "consistency": ("cons_missing", "cons_score"),
                "stance": ("stance_missing", "stance_supported")}
#: Each single score's direction, fixed by what it measures and never fitted:
#: +1 where a higher value means more of a mismatch.
SINGLE = {"openclip_cos": -1, "siglip_cos": -1, "blip_p_match": -1, "blip_itc": -1, "blip2_p_match": -1,
          "blip2_itc": -1, "nli_desc_entail": -1, "nli_desc_contra": 1, "nli_docs_entail": -1,
          "nli_docs_contra": 1, "cons_score": 1, "stance_refuted": 1}


def stance_rows() -> tuple[dict[str, dict[str, Any]], str]:
    """claim_key -> Track B's stance, read only; and what was found."""
    if not STANCE.is_file():
        return {}, f"absent ({cs.rel(STANCE)} does not exist)"
    out = {}
    for r in read_jsonl(STANCE):
        if missing := [k for k in STANCE_KEYS if k not in r]:
            raise SystemExit(f"{STANCE.name}: a row lacks {missing}: refusing to guess its schema")
        if r["judge"] == "nli" and r["status"] == "ok":  # NLI-B: a feature only, never an explanation
            out[r["claim_key"]] = r
    return out, f"{len(out):,} claims with an NLI-B verdict (status ok); Qwen's cited verdicts serve explanations only"


def features(records: Sequence[dict[str, Any]], scores: dict[str, dict[str, dict[str, Any]]],
             verdicts: dict[str, dict[str, Any]], stance: dict[str, dict[str, Any]],
             nli_by_record: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Each record's features: None where a value is missing."""
    out = []

    def one(name, key, field):
        r = scores.get(name, {}).get(key)
        return r[field] if r and r["status"] == "ok" else None

    def verdict_flags(v, item, prefix):
        x = (v or {}).get(item)
        return {f"{prefix}_consistent": float(bool(x) and x["verdict"] == "consistent"),
                f"{prefix}_contradicted": float(bool(x) and x["verdict"] == "contradicted")}

    for r in records:
        k = r["key"]
        f: dict[str, Any] = {"openclip_cos": one("openclip_h14", k, "cosine"),
                             "siglip_cos": one("siglip_so400m", k, "cosine"),
                             "blip_p_match": one("blip_itm_large", k, "p_match"), "blip_itc": one("blip_itm_large", k, "itc"),
                             "blip2_p_match": one("blip2_itm_g", k, "p_match"), "blip2_itc": one("blip2_itm_g", k, "itc")}
        slots = nli_by_record.get(k, {})
        for slot, name in (("description", "desc"), ("ocr", "ocr")):
            x = slots.get(slot)
            f[f"nli_{name}_entail"] = x["entailment"] if x else None
            f[f"nli_{name}_contra"] = x["contradiction"] if x else None
        docs = [slots[s] for s in NLI_SLOTS if s.startswith("doc") and s in slots]
        f["nli_docs_entail"] = max(d["entailment"] for d in docs) if docs else None
        f["nli_docs_contra"] = max(d["contradiction"] for d in docs) if docs else None
        v = verdicts.get(k)
        if v:
            sc = v["scores"]["primary"]
            f.update({"cons_score": sc["score"], "cons_decided": float(not sc["abstained"]),
                      **verdict_flags(v["items"], "location", "cons_location"),
                      **verdict_flags(v["items"], "people", "cons_people"),
                      **verdict_flags(v["variants"], "event", "cons_event"),
                      **verdict_flags(v["variants"], "date", "cons_date")})
        else:
            f.update(dict.fromkeys(GROUPS["consistency"]))
        st = stance.get(r["claim_key"])
        f.update({"stance_supported": st["p_supported"] if st else None,
                  "stance_refuted": st["p_refuted"] if st else None, "stance_nei": st["p_nei"] if st else None})
        out.append(f)
    return out


def matrix(feats: Sequence[dict[str, Any]], groups: Sequence[str], *, nan: bool):
    """The design matrix of ``groups``: missing values as NaN (``nan``) or as 0
    beside the group's missing flag."""
    import numpy as np

    cols = [c for g in groups for c in GROUPS[g]]
    flags = [MISSING_FLAG[g] for g in groups if g in MISSING_FLAG]
    x = np.array([[np.nan if f[c] is None else float(f[c]) for c in cols] for f in feats], dtype=float)
    if nan:
        return x, cols
    miss = np.array([[float(f[col] is None) for _flag, col in flags] for f in feats], dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    return (np.hstack([x, miss]) if flags else x), cols + [flag for flag, _col in flags]


# --------------------------------------------------------------------------
# the combiner: grouped cross-validation within M4FC train
# --------------------------------------------------------------------------

M4FC_FALSE = ("false (out-of-context)", "false (manipulated/fake)", "false (ai-generated)")
VERITE_FALSE = ("out-of-context", "miscaptioned")
LR_L2 = (0.1, 1.0, 10.0)
GB_GRID = ({"max_depth": 2, "max_iter": 100}, {"max_depth": 2, "max_iter": 300},
           {"max_depth": 3, "max_iter": 100}, {"max_depth": 3, "max_iter": 300})


def target(r: dict[str, Any]) -> int | None:
    """True (0) against false (1): M4FC's verdict prediction (out-of-context,
    manipulated, AI-generated; mixture and unverified are not scored), VERITE's
    classes, and for the Factify2 variant its reason code (none against the rest)."""
    if r["dataset"] == "m4fc":
        return 0 if r["class"] == "true" else 1 if r["class"] in M4FC_FALSE else None
    if r["dataset"] == "verite":
        return 0 if r["class"] == "true" else 1 if r["class"] in VERITE_FALSE else None
    return 0 if r.get("reason_code") == "none" else 1


def sklearn_pin() -> tuple[str | None, str]:
    """The scikit-learn release Track B pins, if it is the one installed here:
    (version, why) -- one pin in the repo, never a second."""
    if not TRACKB_REQUIREMENTS.is_file():
        return None, f"{TRACKB_REQUIREMENTS.name} does not exist yet"
    pins = [line.split("==", 1)[1].strip() for line in TRACKB_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip().casefold().startswith("scikit-learn==")]
    if len(pins) != 1:
        return None, f"{TRACKB_REQUIREMENTS.name} pins scikit-learn {len(pins)} times"
    try:
        import sklearn
    except ImportError:
        return None, f"scikit-learn=={pins[0]} is pinned but not installed here"
    if sklearn.__version__ != pins[0]:
        return None, f"installed scikit-learn {sklearn.__version__} is not the pinned {pins[0]}"
    return pins[0], "pinned and installed"


class Candidate:
    """One combiner form: an L2 logistic regression (numpy, consistency's
    fit_logistic) or scikit-learn's histogram gradient boosting."""

    def __init__(self, kind: str, params: dict[str, Any]) -> None:
        self.kind, self.params = kind, params

    @property
    def name(self) -> str:
        return f"{self.kind}(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"

    def fit(self, feats, y, groups):
        import numpy as np

        x, self.cols = matrix(feats, groups, nan=self.kind == "gb")
        if self.kind == "lr":
            self.model = cs.fit_logistic(x.tolist(), list(y), l2=self.params["l2"])
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model = HistGradientBoostingClassifier(
                learning_rate=0.05, l2_regularization=1.0, class_weight="balanced", early_stopping=False,
                random_state=SEED, **self.params).fit(x, np.asarray(y))
        self.groups = list(groups)
        return self

    def predict(self, feats):
        x, _ = matrix(feats, self.groups, nan=self.kind == "gb")
        if self.kind == "lr":
            return [cs.fused_score(self.model, row) for row in x.tolist()]
        return [round(float(p), 6) for p in self.model.predict_proba(x)[:, 1]]


def candidates(gb: bool) -> list[Candidate]:
    out = [Candidate("lr", {"l2": l2}) for l2 in LR_L2]
    return out + ([Candidate("gb", dict(p)) for p in GB_GRID] if gb else [])


def folds_of(records: Sequence[dict[str, Any]]) -> list[int]:
    """Each record's fold: whole clusters (records joined by an image or a
    claim) together, by a seeded hash of the cluster."""
    clusters = cs.clusters_of(records)
    return [int(hashlib.sha256(f"{SEED}:{clusters[r['key']]}".encode()).hexdigest(), 16) % FOLDS
            for r in records]


def out_of_fold(cand: Candidate, train, feats, y, groups) -> list[float]:
    folds = folds_of(train)
    oof = [0.0] * len(train)
    for k in range(FOLDS):
        fit_i = [i for i, f in enumerate(folds) if f != k]
        held = [i for i, f in enumerate(folds) if f == k]
        m = type(cand)(cand.kind, cand.params).fit([feats[i] for i in fit_i], [y[i] for i in fit_i], groups)
        for i, p in zip(held, m.predict([feats[i] for i in held])):
            oof[i] = p
    return oof


def ba_threshold(scores: Sequence[float], y: Sequence[int]) -> dict[str, Any]:
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    best = None
    for t in sorted(set(scores)):
        ba = (sum(s >= t for s in pos) / len(pos) + sum(s < t for s in neg) / len(neg)) / 2
        if best is None or ba > best[1]:
            best = (t, ba)
    return {"threshold": best[0], "balanced_accuracy": round(best[1], 4)}


def auc(scores, y) -> float | None:
    return cs._round(cs.auroc([s for s, t in zip(scores, y) if t == 1], [s for s, t in zip(scores, y) if t == 0]))


# --------------------------------------------------------------------------
# the report: gate E1
# --------------------------------------------------------------------------

PAIRS_OF = {"m4fc val": [("false (out-of-context)", "true"), ("false", "true")],
            "verite": [("out-of-context", "true"), ("miscaptioned", "true")]}


def eval_rows(records, preds: dict[str, dict[str, float]], corpus: str) -> list[dict[str, Any]]:
    """The records of one evaluation corpus with every score a comparison reads."""
    sel = (lambda r: r["dataset"] == "m4fc" and r["split"] == "val") if corpus == "m4fc val" else \
        (lambda r: r["dataset"] == "verite")
    return [dict(r, **{k: v[r["key"]] for k, v in preds.items() if r["key"] in v}) for r in records if sel(r)]


def compare(rows, score: str, against: str, pos: str, neg: str) -> dict[str, Any]:
    if neg == "true" and pos == "false":
        rows = [dict(r, **{"class": "false" if r["class"] in M4FC_FALSE else r["class"]}) for r in rows]
    rows = [r for r in rows if r.get(score) is not None and r.get(against) is not None]
    return cs.paired_auroc(rows, pos, neg, score, against)


def accuracy(rows, score: str, threshold: float, pos: str) -> dict[str, Any]:
    part = [r for r in rows if r["class"] in (pos, "true") and r.get(score) is not None]
    right = sum(1 for r in part if (r[score] >= threshold) == (r["class"] == pos))
    return cs.rate(right, len(part))


def macro_f1(rows, score: str, threshold: float) -> float | None:
    part = [r for r in rows if r["class"] == "true" or r["class"] in M4FC_FALSE]
    if not part:
        return None
    f1s = []
    for positive in (True, False):
        tp = sum(1 for r in part if (r[score] >= threshold) == positive and (r["class"] != "true") == positive)
        fp = sum(1 for r in part if (r[score] >= threshold) == positive and (r["class"] != "true") != positive)
        fn = sum(1 for r in part if (r[score] >= threshold) != positive and (r["class"] != "true") == positive)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / 2, 4)


TRACKB_BASELINES = REPORTS / "baselines.json"
#: Track B's CLIP ViT-L/14 embeddings (read, never written): keys "img:<sha256>"
#: and "txt:<claim_key>", unit vectors. OpenAI's model is a baseline only (the
#: person's decision, 2026-11-10); its similarity alone scores 0.852 AUROC on
#: VERITE out-of-context vs true, the number the ensemble must beat.
TRACKB_L14 = INTERIM / "textcheck" / "muse" / "clip_l14.npz"
L14_BAR = {"verite: out-of-context vs true": 0.852}


def l14_scores(records: Sequence[dict[str, Any]]) -> tuple[dict[str, float], str]:
    """record key -> minus Track B's CLIP L/14 cosine of its image and claim
    (higher, more of a mismatch); and what was joined."""
    if not TRACKB_L14.is_file():
        return {}, f"absent ({cs.rel(TRACKB_L14)})"
    import numpy as np

    z = np.load(TRACKB_L14, allow_pickle=False)
    idx = {str(k): i for i, k in enumerate(z["keys"])}
    vecs = z["vecs"]
    out = {}
    for r in records:
        i, t = idx.get(f"img:{r['sha256']}"), idx.get(f"txt:{r['claim_key']}")
        if i is not None and t is not None:
            out[r["key"]] = round(-float(vecs[i] @ vecs[t]), 6)
    scored = [r for r in records if r["dataset"] != "factify2"]
    return out, f"{sum(1 for r in scored if r['key'] in out):,} of {len(scored):,} records joined"


def track_b_baselines() -> dict[str, Any]:
    """Track B's baselines on M4FC val and VERITE, as its report states them
    (read, never written): per variant and pair, its AUROC with its own
    interval. Unpaired with ours: its records and resamples are its own."""
    if not TRACKB_BASELINES.is_file():
        return {}
    doc = json.loads(TRACKB_BASELINES.read_text(encoding="utf-8"))
    out = {}
    for name, v in ((doc.get("muse") or {}).get("variants") or {}).items():
        for corpus, field in (("m4fc val", "m4fc_val"), ("verite", "verite")):
            for a in (v.get(field) or {}).get("auroc") or []:
                out.setdefault(f"muse:{name}", {})[f"{corpus}: {a['positive']} vs {a['negative']}"] = {
                    "auroc": a.get("auroc"), "ci95": a.get("ci95"), "records": a.get("records")}
    return out


def cmd_fit(args) -> int:
    from scripts import gallery as gl

    records = read_jsonl(RECORDS)
    if not records:
        raise SystemExit("run `inputs` first")
    names = list(models())
    scores = {n: {r["key"]: r for r in read_jsonl(ledger_path(n))} for n in names if models()[n]["kind"] != "nli"}
    nli_keys = {r["key"]: r for r in read_input(NLI_IN, "nli")}
    nli_by_record: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(ledger_path("nli_deberta_large")):
        if r["status"] == "ok" and r["key"] in nli_keys:
            nli_by_record[nli_keys[r["key"]]["record"]][nli_keys[r["key"]]["slot"]] = r
    missing = {n: len(records) - len(scores.get(n, {})) for n in scores}
    verdicts = {v["key"]: v for v in gl.read_jsonl(cs.C_VERDICTS) if v["variant"] == "extracted"}
    stance, stance_found = stance_rows()
    feats = features(records, scores, verdicts, stance, nli_by_record)
    by_key = dict(zip((r["key"] for r in records), feats))
    groups = [g for g in GROUPS if g != "stance" or stance]
    pin, pin_why = sklearn_pin()
    cands = candidates(gb=pin is not None)

    train = [r for r in records if r["dataset"] == "m4fc" and r["split"] == "train" and target(r) is not None]
    y = [target(r) for r in train]
    tf = [by_key[r["key"]] for r in train]
    cv = []
    for c in cands:
        oof = out_of_fold(c, train, tf, y, groups)
        cv.append({"candidate": c.name, "oof_auroc": auc(oof, y), "oof": oof})
        print(f"  [cv] {c.name}: out-of-fold AUROC {cv[-1]['oof_auroc']}", flush=True)
    best_i = max(range(len(cands)), key=lambda i: (cv[i]["oof_auroc"] or 0, -i))
    chosen = Candidate(cands[best_i].kind, cands[best_i].params).fit(tf, y, groups)
    thr = ba_threshold(cv[best_i]["oof"], y)

    clip = {r["key"]: -r["similarity"] for r in gl.read_jsonl(cs.C_CLIP) if r["similarity"] is not None}
    fitted = cs.read_fused()
    l14, l14_found = l14_scores(records)
    preds: dict[str, dict[str, float]] = {"ensemble": dict(zip((r["key"] for r in records), chosen.predict(feats))),
                                          "clip_b32": clip}
    if l14:
        preds["clip_l14"] = l14
        l14_t = cs.fit_threshold([dict(r, score=l14[r["key"]]) for r in records if r["dataset"] == "m4fc"
                                  and r["split"] == "train" and r["class"] in (cs.OOC, "true")
                                  and r["key"] in l14])["threshold"]
    if fitted:
        preds["fused"] = {k: cs.fused_score(fitted["model"], cs.fused_features(
            -clip[k], verdicts[k]["scores"]["primary"]["score"], verdicts[k]["scores"]["primary"]["abstained"]))
            for k in verdicts if k in clip}
    for col, sign in SINGLE.items():
        preds[f"single:{col}"] = {r["key"]: sign * f[col] for r, f in zip(records, feats) if f[col] is not None}
    single_train = {col: auc([preds[f"single:{col}"].get(r["key"]) for r in train], y)
                    if all(r["key"] in preds[f"single:{col}"] for r in train) else None for col in SINGLE}
    # chosen on train among the scores every evaluation record can have: documents are M4FC's alone
    everywhere = [c for c in single_train if single_train[c] is not None and c not in GROUPS["nli_documents"]]
    best_single = max(everywhere, key=lambda c: single_train[c])

    ablation = {}
    for g in groups:
        rest = [x for x in groups if x != g]
        m = Candidate(chosen.kind, chosen.params).fit(tf, y, rest)
        preds[f"without:{g}"] = dict(zip((r["key"] for r in records), m.predict(feats)))
        ablation[g] = {"oof_auroc": auc(out_of_fold(chosen, train, tf, y, rest), y)}
    f2 = [r for r in records if r["dataset"] == "factify2" and target(r) is not None]
    variant = Candidate(chosen.kind, chosen.params).fit(tf + [by_key[r["key"]] for r in f2],
                                                        y + [target(r) for r in f2], groups)
    preds["variant:+factify2"] = dict(zip((r["key"] for r in records), variant.predict(feats)))

    corpora = {}
    for corpus, pairs in PAIRS_OF.items():
        rows = eval_rows(records, preds, corpus)
        e = {"records": len(rows), "auroc": [], "accuracy_at_threshold": {}, "by_class": {}, "ablation": {},
             "singles": {}}
        for pos, neg in pairs:
            tag = f"{pos} vs {neg}"
            if "clip_l14" in preds:  # the bar first: Track B's CLIP L/14, on the same records, paired
                e["auroc"].append({"against": "clip_l14", **compare(rows, "ensemble", "clip_l14", pos, neg)})
            e["auroc"].append({"against": "clip_b32", **compare(rows, "ensemble", "clip_b32", pos, neg)})
            for other in ("fused", f"single:{best_single}", "variant:+factify2"):
                if other in preds:
                    e["auroc"].append({"against": other, **compare(rows, "ensemble", other, pos, neg)})
            base = "clip_l14" if "clip_l14" in preds else "clip_b32"  # each single against the bar, paired
            e["singles"][tag] = {}
            for c in SINGLE:
                if f"single:{c}" in preds:
                    cmp_ = compare(rows, f"single:{c}", base, pos, neg)
                    e["singles"][tag][c] = {"auroc": cmp_["auroc_" + f"single:{c}"], "against": base,
                                            "delta": cmp_["delta"], "delta_ci95": cmp_["delta_ci95"],
                                            "excludes_zero": cmp_["excludes_zero"]}
            e["ablation"][tag] = {g: compare(rows, f"without:{g}", "ensemble", pos, neg)[f"auroc_without:{g}"]
                                  for g in groups}
            if corpus == "verite":
                e["accuracy_at_threshold"][tag] = accuracy(rows, "ensemble", thr["threshold"], pos)
        if corpus == "m4fc val":
            e["macro_f1_true_vs_false"] = macro_f1(rows, "ensemble", thr["threshold"])
        clip_t = cs.read_fused()["clip_threshold"] if fitted else None
        for c in sorted({r["class"] for r in rows}):
            sub = [r for r in rows if r["class"] == c]
            e["by_class"][cs.CLASS_NAMES.get(c, c)] = {
                "ensemble": cs.flag_cell(sub, thr["threshold"], "ensemble", None),
                "clip_b32": cs.flag_cell(sub, clip_t, "clip_b32", None) if clip_t is not None else None,
                "clip_l14": cs.flag_cell([r for r in sub if r.get("clip_l14") is not None], l14_t, "clip_l14", None)
                if l14 else None}
        corpora[corpus] = e
    m4, ve = corpora["m4fc val"]["auroc"][0], corpora["verite"]["auroc"][0]
    report = {
        "written_at": cs.utcnow(), "gate": "E1",
        "scorers": {n: {"repo": models()[n]["repo"], "revision": models()[n]["revision"],
                        "licence": models()[n]["licence"], "records_missing": missing.get(n)} for n in names},
        "stance": stance_found, "sklearn": {"version": pin, "why": pin_why},
        "groups": groups, "target": "true (0) against false (1); M4FC: out-of-context, manipulated, AI-generated",
        "train": {"records": len(train), "true": y.count(0), "false": y.count(1), "folds": FOLDS,
                  "cv": [{k: v for k, v in c.items() if k != "oof"} for c in cv], "chosen": chosen.name,
                  "threshold": thr, "single_auroc": single_train, "best_single": best_single,
                  "model": chosen.model if chosen.kind == "lr" else None, "columns": chosen.cols},
        "ablation_train": ablation,
        "factify2_variant": {"records": len(f2), "label": "reason code none (0) against content_unverified "
                             "or content_refuted (1): whether a document supports the claim, not whether the "
                             "image fits it", "by_label": dict(Counter(r["class"] for r in f2))},
        "corpora": corpora,
        "track_b_baselines": track_b_baselines(),
        "clip_l14": {"joined": l14_found, "bar": L14_BAR,
                     "threshold_m4fc_train": l14_t if l14 else None,
                     "beaten": {f"{corpus}: {a['positive']} vs {a['negative']}":
                                bool(a["excludes_zero"] and a["delta"] > 0)
                                for corpus, e in corpora.items() for a in e["auroc"] if a["against"] == "clip_l14"}},
        "transfer_gap": {"m4fc_val_ooc_vs_true": m4["auroc_ensemble"], "verite_ooc_vs_true": ve["auroc_ensemble"],
                         "gap": cs._round(m4["auroc_ensemble"] - ve["auroc_ensemble"])
                         if m4["auroc_ensemble"] is not None and ve["auroc_ensemble"] is not None else None},
    }
    cs.public_report(report, REPORT_E1)
    REPORT_E1_MD.write_text(render_e1(report), encoding="utf-8", newline="\n")
    print(f"chosen {chosen.name}; threshold {thr}; -> {cs.rel(REPORT_E1)} and {cs.rel(REPORT_E1_MD)} (local)")
    for corpus, e in corpora.items():
        for a in e["auroc"]:
            print(f"  {corpus} {a['positive']} vs {a['negative']}: ensemble {a['auroc_ensemble']} against "
                  f"{a['against']} {a['auroc_' + a['against']]}, Δ {a['delta']} {a['delta_ci95']}")
    return 0


def _delta(a: dict[str, Any]) -> str:
    if a["delta"] is None:
        return "–"
    ci = a["delta_ci95"]
    return f"{a['delta']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def _l14(e: dict[str, Any], tag: str) -> float | None:
    """CLIP L/14's AUROC on one corpus's pair, from its paired comparison."""
    return next((a["auroc_clip_l14"] for a in e["auroc"]
                 if a["against"] == "clip_l14" and f"{a['positive']} vs {a['negative']}" == tag), None)


def render_e1(rep: dict[str, Any]) -> str:
    """Gate E1's report for a person to read; the JSON beside it is the record.
    Track B's CLIP L/14 stands beside every result: it is the number to beat."""
    r3 = cs._r3
    l14 = rep.get("clip_l14") or {}
    beaten = l14.get("beaten") or {}
    bar = ("**The bar: CLIP ViT-L/14 similarity alone scores 0.852 AUROC on VERITE out-of-context vs true "
           "(Track B).** Here it is recomputed from Track B's embeddings on the same records "
           f"({l14.get('joined', 'not joined')}) and compared paired. "
           + ("The ensemble beats it on: " + ", ".join(k for k, ok in beaten.items() if ok) + "."
              if any(beaten.values()) else "The ensemble does not beat it on any pair: no interval excludes 0 "
              "in its favour."))
    lines = ["# Gate E1: the ensemble on M4FC val and VERITE", "", bar, "",
             "Local. `data/reports/ensemble_e1.json` is the record. The combiner was chosen by grouped "
             f"{rep['train']['folds']}-fold cross-validation within M4FC train ({rep['train']['true']} true, "
             f"{rep['train']['false']} false); val and VERITE fit nothing.", "",
             f"- Chosen: **{rep['train']['chosen']}**; threshold {r3(rep['train']['threshold']['threshold'])} "
             f"(out-of-fold balanced accuracy {r3(rep['train']['threshold']['balanced_accuracy'])}).",
             f"- scikit-learn: {rep['sklearn']['version'] or 'not used'} ({rep['sklearn']['why']}).",
             f"- Track B's stance: {rep['stance']}.", "",
             "## Cross-validation within M4FC train", "", "| candidate | out-of-fold AUROC |", "| --- | ---: |"]
    lines += [f"| {c['candidate']} | {r3(c['oof_auroc'])} |" for c in rep["train"]["cv"]]
    lines += ["", "## AUROC, paired cluster bootstrap (CLIP L/14 first: the bar)", "",
              "| corpus | pair | records | ensemble | against | its AUROC | Δ [95%] | excludes 0 |",
              "| --- | --- | ---: | ---: | --- | ---: | --- | --- |"]
    for corpus, e in rep["corpora"].items():
        for a in e["auroc"]:
            lines.append(f"| {corpus} | {a['positive']} vs {a['negative']} | {a['records']} | "
                         f"{r3(a['auroc_ensemble'])} | {a['against']} | {r3(a['auroc_' + a['against']])} | "
                         f"{_delta(a)} | {'yes' if a['excludes_zero'] else 'no'} |")
    lines += ["", "## Single scorers (direction fixed, nothing fitted), each paired against CLIP L/14 "
              "(* where the interval excludes 0)", ""]
    for corpus, e in rep["corpora"].items():
        for tag, singles in e["singles"].items():
            lines.append(f"- {corpus}, {tag}: CLIP L/14 {r3(_l14(e, tag))}; " + ", ".join(
                f"{c} {r3(v['auroc'])}" + (f" (Δ {_delta(v)}{'*' if v['excludes_zero'] else ''})"
                                           if v["delta"] is not None else "") for c, v in singles.items()))
    lines += ["", "## Ablation: the chosen form refitted without each group, CLIP L/14 beside", ""]
    lines.append("- M4FC train, out of fold: " + ", ".join(
        f"without {g} {r3(v['oof_auroc'])}" for g, v in rep["ablation_train"].items()))
    for corpus, e in rep["corpora"].items():
        for tag, abl in e["ablation"].items():
            lines.append(f"- {corpus}, {tag}: CLIP L/14 {r3(_l14(e, tag))}; "
                         + ", ".join(f"without {g} {r3(v)}" for g, v in abl.items()))
    lines += ["", "## Per class at each score's own M4FC-train threshold", "",
              "| corpus | class | records | ensemble flags | CLIP L/14 flags | CLIP B/32 flags |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for corpus, e in rep["corpora"].items():
        for c, cell in e["by_class"].items():
            lines.append(f"| {corpus} | {c} | {cell['ensemble']['records']} | {cs._pct(cell['ensemble']['flagged'])} | "
                         f"{cs._pct((cell.get('clip_l14') or {}).get('flagged'))} | "
                         f"{cs._pct((cell['clip_b32'] or {}).get('flagged'))} |")
    if rep.get("track_b_baselines"):
        lines += ["", "## Track B's baselines, as its report states them (unpaired with ours)", "",
                  "| baseline | pair | AUROC | its 95% interval |", "| --- | --- | ---: | --- |"]
        for name, pairs in rep["track_b_baselines"].items():
            for pair, a in pairs.items():
                ci = a.get("ci95")
                lines.append(f"| {name} | {pair} | {r3(a.get('auroc'))} | "
                             f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '–'} |")
    v = rep["corpora"]["verite"]["accuracy_at_threshold"]
    lines += ["", "VERITE accuracy at the train threshold: " + "; ".join(f"{t} {cs._pct(c)}" for t, c in v.items()),
              f"M4FC val macro F1, true vs false: {r3(rep['corpora']['m4fc val'].get('macro_f1_true_vs_false'))}",
              f"Transfer gap (M4FC val minus VERITE, out-of-context vs true): {r3(rep['transfer_gap']['gap'])}", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the routed system (the person's E1 decision, 2026-11-10). POST HOC: decided
# after VERITE was seen at gate E1, and labelled so wherever it is reported.
# --------------------------------------------------------------------------

REPORT_ROUTED = REPORTS / "ensemble_routed.json"
REPORT_ROUTED_MD = REPORTS / "ensemble_routed.md"
SYSTEM_CANDIDATE = ENS_DIR / "system_candidate.json"
SYSTEM = CONFIGS / "ensemble_system.json"
POST_HOC = ("post hoc: the person's E1 decision (2026-11-10), taken after VERITE was seen at gate E1; "
            "the route rule and every fitted number come from M4FC train alone")
#: The photo branch: an unfitted, equal-weight average of the image-text
#: scorers, each in its mismatch direction and rank-normalised on M4FC train.
#: BLIP's and BLIP-2's scores are their matching heads' P(match).
PHOTO = (("openclip_cos", -1), ("siglip_cos", -1), ("blip_p_match", -1), ("blip2_p_match", -1))
#: The route threshold's candidates: M4FC train's OCR text shares of the
#: images not in a screenshot layout, at these quantiles; also 0 (every image
#: to the M4FC branch) and none (screenshot layouts only).
ROUTE_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: The sign-constrained refit's signs, fixed by what each feature means (an
#: ablation, post hoc): +1 may only push toward false, -1 only toward true,
#: 0 free. Similarity and entailment can only argue for the claim;
#: contradiction and a contradicted verdict only against it.
SIGNS = {"openclip_cos": -1, "siglip_cos": -1, "blip_p_match": -1, "blip_itc": -1, "blip2_p_match": -1,
         "blip2_itc": -1, "nli_desc_entail": -1, "nli_desc_contra": 1, "nli_ocr_entail": -1, "nli_ocr_contra": 1,
         "nli_docs_entail": -1, "nli_docs_contra": 1, "cons_score": 1, "cons_decided": 0,
         "cons_location_consistent": -1, "cons_location_contradicted": 1, "cons_people_consistent": -1,
         "cons_people_contradicted": 1, "cons_event_consistent": -1, "cons_event_contradicted": 1,
         "cons_date_consistent": -1, "cons_date_contradicted": 1, "stance_supported": -1, "stance_refuted": 1,
         "stance_nei": 0}
PROBE_SCORER = "openclip_h14"
PROBE_L2 = (10.0, 100.0, 1000.0)


def screenshot_layout(description: str | None) -> bool:
    """Whether the image model's description (claim-free) calls the image a
    screenshot in its first sentence."""
    first = re.split(r"(?<=[.!?])\s+", (description or "").strip(), maxsplit=1)[0]
    return bool(re.search(r"\bscreenshots?\b", first, re.IGNORECASE))


def _area(poly: Sequence[Sequence[float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]]))) / 2


def ocr_share(image_row: dict[str, Any] | None) -> float | None:
    """The share of the image the OCR's text regions cover, in the OCR's own
    input frame; None where the OCR failed."""
    ocr = (image_row or {}).get("ocr") or {}
    if not image_row or ocr.get("error"):
        return None
    size = (ocr.get("input") or {})
    w, h = size.get("width"), size.get("height")
    if not w or not h:
        return None
    return round(min(1.0, sum(_area(r["polygon"]) for r in ocr.get("regions") or []) / (w * h)), 6)


def route_features(records: Sequence[dict[str, Any]], image_rows: dict[str, dict[str, dict[str, Any]]]
                   ) -> dict[str, dict[str, Any]]:
    """record key -> {screenshot, ocr_share}, from its image's image_text row
    (dataset -> sha256 -> row)."""
    out = {}
    for r in records:
        row = image_rows.get(r["dataset"], {}).get(r["sha256"])
        out[r["key"]] = {"screenshot": screenshot_layout(((row or {}).get("description") or {}).get("text")),
                         "ocr_share": ocr_share(row)}
    return out


def to_m4fc_branch(rf: dict[str, Any], t: float | None) -> bool:
    """The route rule: a screenshot layout, or OCR text covering at least t
    of the image (t None: screenshots only). An OCR that failed covers 0."""
    return rf["screenshot"] or (t is not None and (rf["ocr_share"] or 0.0) >= t)


def pct(table: Sequence[float], v: float) -> float:
    """v's mid-rank percentile in a sorted table (M4FC train's values)."""
    import bisect

    lo, hi = bisect.bisect_left(table, v), bisect.bisect_right(table, v)
    return round((lo + 0.5 * (hi - lo)) / len(table), 6)


def photo_raw(f: dict[str, Any], tables: dict[str, Sequence[float]]) -> float | None:
    """The photo branch before its own normalisation: the mean of the
    image-text scorers' train percentiles, each in its mismatch direction,
    over those the record has."""
    got = [pct(tables[c], sign * f[c]) for c, sign in PHOTO if f.get(c) is not None]
    return round(sum(got) / len(got), 6) if got else None


def fit_logistic_signed(X, y, signs: Sequence[int], l2: float) -> dict[str, Any]:
    """fit_logistic's objective (classes balanced, L2 on standardised
    features), each coefficient held to its sign: L-BFGS-B with bounds."""
    import numpy as np
    from scipy.optimize import minimize

    x, t = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    z = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
    w = np.where(t == 1, len(t) / (2 * t.sum()), len(t) / (2 * (len(t) - t.sum())))
    pen = np.full(z.shape[1], float(l2))
    pen[0] = 0.0

    def f(beta):
        m = z @ beta
        loss = np.sum(w * (np.logaddexp(0, m) - t * m)) + 0.5 * np.sum(pen * beta ** 2)
        p = 1.0 / (1.0 + np.exp(-m))
        return loss, z.T @ (w * (p - t)) + pen * beta

    bounds = [(None, None)] + [(0.0, None) if s > 0 else (None, 0.0) if s < 0 else (None, None) for s in signs]
    res = minimize(f, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    beta = res.x
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std], "intercept": float(beta[0]),
            "coef": [float(v) for v in beta[1:]], "iterations": int(res.nit), "converged": bool(res.success),
            "coef_raw_units": [float(b / s) for b, s in zip(beta[1:], std)],
            "signs": list(signs), "at_bound": int(sum(1 for b, s in zip(beta[1:], signs) if s and abs(b) < 1e-9))}


class SignedCandidate(Candidate):
    """The sign-constrained refit (an ablation, post hoc)."""

    def fit(self, feats, y, groups):
        x, self.cols = matrix(feats, groups, nan=False)
        self.model = fit_logistic_signed(x.tolist(), list(y), [SIGNS.get(c, 0) for c in self.cols],
                                         self.params["l2"])
        self.groups = list(groups)
        return self

    def predict(self, feats):
        x, _ = matrix(feats, self.groups, nan=False)
        return [cs.fused_score(self.model, row) for row in x.tolist()]


def load_eval() -> dict[str, Any]:
    """Everything the combiner and the routed system read for the evaluation
    set: records, features, verdicts and the baselines' scores."""
    from scripts import gallery as gl

    records = read_jsonl(RECORDS)
    if not records:
        raise SystemExit("run `inputs` first")
    scores = {n: {r["key"]: r for r in read_jsonl(ledger_path(n))} for n in models() if models()[n]["kind"] != "nli"}
    nli_keys = {r["key"]: r for r in read_input(NLI_IN, "nli")}
    nli_by_record: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(ledger_path("nli_deberta_large")):
        if r["status"] == "ok" and r["key"] in nli_keys:
            nli_by_record[nli_keys[r["key"]]["record"]][nli_keys[r["key"]]["slot"]] = r
    verdicts = {v["key"]: v for v in gl.read_jsonl(cs.C_VERDICTS) if v["variant"] == "extracted"}
    stance, stance_found = stance_rows()
    feats = features(records, scores, verdicts, stance, nli_by_record)
    train = [r for r in records if r["dataset"] == "m4fc" and r["split"] == "train" and target(r) is not None]
    by_key = dict(zip((r["key"] for r in records), feats))
    clip = {r["key"]: -r["similarity"] for r in gl.read_jsonl(cs.C_CLIP) if r["similarity"] is not None}
    l14, l14_found = l14_scores(records)
    own_l14 = {r["key"]: -r["cosine"] for r in read_jsonl(ledger_path("clip_l14")) if r["status"] == "ok"}
    return {"records": records, "feats": feats, "by_key": by_key, "verdicts": verdicts, "train": train,
            "y": [target(r) for r in train], "tf": [by_key[r["key"]] for r in train],
            "groups": [g for g in GROUPS if g != "stance" or stance], "stance": stance_found, "clip_b32": clip,
            "clip_l14": l14, "clip_l14_found": l14_found, "clip_l14_ours": own_l14}


def choose_ensemble(ctx: dict[str, Any]) -> tuple[Candidate, list[float], list[dict[str, Any]]]:
    """The M4FC branch as E1 chose it: the candidate with the best grouped
    out-of-fold AUROC on M4FC train; its out-of-fold scores."""
    pin, _why = sklearn_pin()
    cands = candidates(gb=pin is not None)
    cv = []
    for c in cands:
        oof = out_of_fold(c, ctx["train"], ctx["tf"], ctx["y"], ctx["groups"])
        cv.append({"candidate": c.name, "oof_auroc": auc(oof, ctx["y"]), "oof": oof})
    best = max(range(len(cands)), key=lambda i: (cv[i]["oof_auroc"] or 0, -i))
    chosen = Candidate(cands[best].kind, cands[best].params).fit(ctx["tf"], ctx["y"], ctx["groups"])
    return chosen, cv[best]["oof"], [{k: v for k, v in c.items() if k != "oof"} for c in cv]


def embedding_features(records: Sequence[dict[str, Any]], texts: dict[str, str], path: Path
                       ) -> dict[str, list[float]]:
    """record key -> the probe's input: the elementwise product of its image's
    and its claim's unit embeddings (OpenCLIP H/14)."""
    import numpy as np

    if not path.is_file():
        return {}
    z = np.load(path, allow_pickle=False)
    idx = {str(k): i for i, k in enumerate(z["keys"])}
    vecs = z["vecs"]
    out = {}
    for r in records:
        i, t = idx.get(f"img:{r['sha256']}"), idx.get(text_key(texts.get(r["key"], "")))
        if i is not None and t is not None:
            out[r["key"]] = (vecs[i] * vecs[t]).astype(float).tolist()
    return out


def flags_accuracy(rows, score: str, flag: str, pos: str) -> dict[str, Any]:
    part = [r for r in rows if r["class"] in (pos, "true") and r.get(flag) is not None]
    return cs.rate(sum(1 for r in part if r[flag] == (r["class"] == pos)), len(part))


def macro_f1_flags(rows, flag: str) -> float | None:
    part = [r for r in rows if (r["class"] == "true" or r["class"] in M4FC_FALSE) and r.get(flag) is not None]
    if not part:
        return None
    f1s = []
    for positive in (True, False):
        tp = sum(1 for r in part if r[flag] == positive and (r["class"] != "true") == positive)
        fp = sum(1 for r in part if r[flag] == positive and (r["class"] != "true") != positive)
        fn = sum(1 for r in part if r[flag] != positive and (r["class"] != "true") == positive)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / 2, 4)


def branch_threshold(scores: Sequence[float], y: Sequence[int], fallback: tuple[Sequence[float], Sequence[int]]
                     ) -> dict[str, Any]:
    """Balanced accuracy's threshold on the train records a branch takes,
    or on every train record where those hold one class only."""
    if 0 < sum(y) < len(y):
        return {**ba_threshold(scores, y), "on": "the branch's own train records"}
    return {**ba_threshold(*fallback), "on": "every train record (the branch's own hold one class)"}


def system_scores(system: dict[str, Any], records: Sequence[dict[str, Any]], feats: Sequence[dict[str, Any]],
                  routes: dict[str, dict[str, Any]], probe_x: dict[str, list[float]] | None = None,
                  oof: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """A frozen system's output per record: its branch, its score (a
    percentile on the branch's M4FC-train scale) and its flag. ``oof``: the
    M4FC branch's out-of-fold scores for train records, so no train record is
    scored by a model that saw it."""
    ens, photo, probe = system["ensemble"], system["photo"], system.get("probe") or {}
    x, cols = matrix(feats, ens["groups"], nan=False)
    if cols != ens["columns"]:
        raise SystemExit("the features' columns are not the frozen system's: refusing")
    out = {}
    for r, f, row in zip(records, feats, x.tolist()):
        m4fc = to_m4fc_branch(routes[r["key"]], system["route"]["t"])
        raw = (oof or {}).get(r["key"])
        e = pct(ens["oof_table"], raw if raw is not None else cs.fused_score(ens["model"], row))
        praw = photo_raw(f, photo["tables"])
        avg = None if praw is None else pct(photo["raw_table"], praw)
        prb = None
        if probe.get("adopted") and (probe_x or {}).get(r["key"]) is not None:
            prb = pct(probe["oof_table"], cs.fused_score(probe["model"], probe_x[r["key"]]))
        p = prb if prb is not None else avg
        branch = "m4fc" if m4fc or p is None else "photo"
        score = e if branch == "m4fc" else p
        t = (ens if branch == "m4fc" else (probe if prb is not None else photo))["threshold"]["threshold"]
        out[r["key"]] = {"branch": branch, "score": score, "flag": score >= t, "ensemble": e, "average": avg,
                         "probe": prb}
    return out


def cmd_route(args) -> int:
    """The routed system: fit on M4FC train, report on M4FC val and VERITE
    (post hoc), and write the system for `freeze`."""
    import numpy as np

    ctx = load_eval()
    records, feats, train, y, tf = ctx["records"], ctx["feats"], ctx["train"], ctx["y"], ctx["tf"]
    evals = [r for r in records if r["dataset"] != "factify2"]
    it_rows = {ds: cs.image_text_rows(ds) or {} for ds in ("m4fc", "verite")}
    routes = route_features(evals, it_rows)
    print(f"routes read for {len(routes):,} records", flush=True)

    # the M4FC branch: E1's choice; percentiles on its out-of-fold train scale
    chosen, oof, cv = choose_ensemble(ctx)
    oof_of = dict(zip((r["key"] for r in train), oof))
    ens_table = sorted(oof)
    # the photo branch: the scorers' train tables, the average's train table
    tables = {c: sorted(sign * ctx["by_key"][r["key"]][c] for r in train if ctx["by_key"][r["key"]][c] is not None)
              for c, sign in PHOTO}
    raw_train = [photo_raw(f, tables) for f in tf]
    raw_table = sorted(v for v in raw_train if v is not None)
    ens_pct_train = [pct(ens_table, v) for v in oof]
    photo_pct_train = [pct(raw_table, v) if v is not None else None for v in raw_train]

    # the route threshold, on M4FC train only. The route says what an image IS: the OCR share
    # that best tells a screenshot layout from the rest (balanced accuracy) marks a text-heavy
    # image. The first objective tried, the routed train AUROC, chose t = 0 -- every image to
    # the M4FC branch, since on M4FC train its ensemble beats the average on photographs too --
    # which leaves no photograph to route; it is reported beside, never used.
    shares = [(routes[r["key"]]["ocr_share"] or 0.0) for r in train]
    shots = [routes[r["key"]]["screenshot"] for r in train]
    grid = [*dict.fromkeys(round(float(np.quantile(shares, q)), 6) for q in ROUTE_QUANTILES), None]
    route_cv = []
    for t in dict.fromkeys([0.0, *grid]):
        hit = [t is not None and v >= t for v in shares]
        tp, pos = sum(1 for h, sh in zip(hit, shots) if h and sh), sum(shots)
        tn, neg = sum(1 for h, sh in zip(hit, shots) if not h and not sh), len(shots) - sum(shots)
        routed = [e if (to_m4fc_branch(routes[r["key"]], t) or p is None) else p
                  for r, e, p in zip(train, ens_pct_train, photo_pct_train)]
        route_cv.append({"t": t, "to_m4fc_branch": sum(1 for r in train if to_m4fc_branch(routes[r["key"]], t)),
                         "screenshot_agreement_ba": round((tp / pos + tn / neg) / 2, 4) if pos and neg else None,
                         "routed_train_auroc": auc(routed, y), "candidate": t != 0.0})
    cands_ = [g for g in route_cv if g["candidate"]]
    t = max(cands_, key=lambda g: (g["screenshot_agreement_ba"] or 0,
                                   float("inf") if g["t"] is None else g["t"]))["t"]
    first_objective = max(route_cv, key=lambda g: (g["routed_train_auroc"] or 0,
                                                   float("inf") if g["t"] is None else g["t"]))["t"]
    in_m4fc = [to_m4fc_branch(routes[r["key"]], t) or p is None for r, p in zip(train, photo_pct_train)]
    sub = lambda vals, mask: ([v for v, m in zip(vals, mask) if m], [yy for yy, m in zip(y, mask) if m])  # noqa: E731
    ens_t = branch_threshold(*sub(ens_pct_train, in_m4fc), (ens_pct_train, y))
    photo_ok = [not m for m in in_m4fc]
    photo_t = branch_threshold(*sub([p or 0.0 for p in photo_pct_train], photo_ok),
                               ([p if p is not None else 0.0 for p in photo_pct_train], y))

    # misroutes on M4FC train: the branch taken is wrong where the other is right
    mis = Counter()
    for r, yy, e, p, m in zip(train, y, ens_pct_train, photo_pct_train, in_m4fc):
        if p is None:
            continue
        right_e, right_p = (e >= ens_t["threshold"]) == bool(yy), (p >= photo_t["threshold"]) == bool(yy)
        taken, other = (right_e, right_p) if m else (right_p, right_e)
        mis[("m4fc" if m else "photo", "misrouted" if (not taken and other) else "kept")] += 1
    misroute = {b: cs.rate(mis[(b, "misrouted")], mis[(b, "misrouted")] + mis[(b, "kept")]) for b in ("m4fc", "photo")}
    misroute["all"] = cs.rate(sum(v for (b, k), v in mis.items() if k == "misrouted"), sum(mis.values()))

    # the sign-constrained refit (ablation, post hoc)
    signed = SignedCandidate("lr_signed", dict(chosen.params)).fit(tf, y, ctx["groups"])
    signed_oof = out_of_fold(SignedCandidate("lr_signed", dict(chosen.params)), train, tf, y, ctx["groups"])
    signed_table = sorted(signed_oof)

    # the linear probe on the photo branch, kept only if it gains in M4FC train cross-validation
    texts = {r["key"]: r["text"] for r in read_input(PAIRS, "pairs")}
    probe_x = embedding_features(evals, texts, embeddings_path(PROBE_SCORER))
    probe = {"adopted": False, "why": f"no embeddings ({cs.rel(embeddings_path(PROBE_SCORER))})"}
    photo_train = [(r, yy) for r, yy, m in zip(train, y, in_m4fc) if not m and r["key"] in probe_x]
    if photo_train and probe_x:
        pr = [r for r, _ in photo_train]
        py = [yy for _, yy in photo_train]
        folds = folds_of(pr)
        best_p = None
        for l2 in PROBE_L2:
            o = [0.0] * len(pr)
            for k in range(FOLDS):
                fit_i = [i for i, f in enumerate(folds) if f != k]
                held = [i for i, f in enumerate(folds) if f == k]
                m = cs.fit_logistic([probe_x[pr[i]["key"]] for i in fit_i], [py[i] for i in fit_i], l2=l2)
                for i in held:
                    o[i] = cs.fused_score(m, probe_x[pr[i]["key"]])
            a = auc(o, py)
            if best_p is None or (a or 0) > (best_p["oof_auroc"] or 0):
                best_p = {"l2": l2, "oof_auroc": a, "oof": o}
        base = [photo_pct_train[train.index(r)] for r in pr]
        rows_ = [{"key": r["key"], "sha256": r["sha256"], "claim_key": r["claim_key"],
                  "class": "false" if yy else "true", "probe": o, "photo": b}
                 for r, yy, o, b in zip(pr, py, best_p["oof"], base)]
        cmp_ = cs.paired_auroc(rows_, "false", "true", "probe", "photo")
        gain = bool(cmp_["excludes_zero"] and cmp_["delta"] > 0)
        probe = {"adopted": gain, "records": len(pr), "true": py.count(0), "l2": best_p["l2"],
                 "oof_auroc": best_p["oof_auroc"], "photo_auroc_same_records": cmp_["auroc_photo"],
                 "delta": cmp_["delta"], "delta_ci95": cmp_["delta_ci95"],
                 "why": "adopted: it gains in M4FC train cross-validation (paired interval excludes 0)" if gain
                 else "not adopted: no gain in M4FC train cross-validation (paired interval does not exclude 0 "
                      "in its favour)"}
        if gain:
            model = cs.fit_logistic([probe_x[r["key"]] for r in pr], py, l2=best_p["l2"])
            table = sorted(best_p["oof"])
            probe.update(model=model, oof_table=table,  # its threshold on the same percentile scale as its score
                         threshold=ba_threshold([pct(table, v) for v in best_p["oof"]], py))
    print(f"probe: {probe['why']}", flush=True)

    system = {"post_hoc": POST_HOC, "target": "true (0) against false (1)",
              "route": {"rule": "a screenshot layout (the image model's description, first sentence), or OCR text "
                                "covering at least t of the image", "t": t, "grid": route_cv},
              "ensemble": {"form": chosen.name, "groups": list(ctx["groups"]), "columns": chosen.cols,
                           "model": chosen.model, "oof_table": [round(v, 6) for v in ens_table], "threshold": ens_t},
              "photo": {"scorers": [list(p) for p in PHOTO], "tables": tables, "raw_table": raw_table,
                        "threshold": photo_t},
              "probe": probe if probe.get("adopted") else {"adopted": False, **{k: v for k, v in probe.items()
                                                                                  if k not in ("model", "oof_table")}},
              "stance": ctx["stance"],
              "declared_ablations": ["routed without the probe", "the M4FC branch alone",
                                     "the photo average alone", "the probe alone"]}
    out = system_scores(system, evals, [ctx["by_key"][r["key"]] for r in evals], routes, probe_x, oof_of)
    signed_sys = {**system, "ensemble": {**system["ensemble"], "model": signed.model,
                                         "oof_table": [round(v, 6) for v in signed_table]}}
    signed_out = system_scores(signed_sys, evals, [ctx["by_key"][r["key"]] for r in evals], routes, probe_x,
                               dict(zip((r["key"] for r in train), signed_oof)))
    bare = system_scores({**system, "probe": {"adopted": False}}, evals, [ctx["by_key"][r["key"]] for r in evals],
                         routes, probe_x, oof_of)
    preds = {"routed": {k: v["score"] for k, v in out.items()},
             "routed_without_probe": {k: v["score"] for k, v in bare.items()},
             "m4fc_branch_alone": {k: v["ensemble"] for k, v in out.items()},
             "photo_average_alone": {k: v["average"] for k, v in out.items() if v["average"] is not None},
             "probe_alone": {k: v["probe"] for k, v in out.items() if v["probe"] is not None},
             "signed_alone": {k: v["ensemble"] for k, v in signed_out.items()},
             "routed_signed": {k: v["score"] for k, v in signed_out.items()},
             "clip_b32": ctx["clip_b32"]}
    if ctx["clip_l14"]:
        preds["clip_l14"] = ctx["clip_l14"]
    if ctx["clip_l14_ours"]:
        preds["clip_l14_ours"] = ctx["clip_l14_ours"]
    systems = [s_ for s_ in ("routed", "routed_without_probe", "m4fc_branch_alone", "photo_average_alone",
                             "probe_alone", "signed_alone", "routed_signed") if preds.get(s_)]
    corpora = {}
    for corpus, pairs in PAIRS_OF.items():
        rows = eval_rows(evals, preds, corpus)
        for r in rows:
            r["flag"], r["branch"] = out[r["key"]]["flag"], out[r["key"]]["branch"]
        e = {"records": len(rows), "auroc": [], "routed_vs_branches": [], "branch_records": {},
             "accuracy_routed": {}}
        for pos, neg in pairs:
            for s in systems:
                for base in ("clip_l14", "clip_l14_ours"):
                    if base in preds:
                        e["auroc"].append({"system": s, "against": base, **compare(rows, s, base, pos, neg)})
            for s in ("m4fc_branch_alone", "photo_average_alone", "routed_without_probe"):
                if preds.get(s):
                    e["routed_vs_branches"].append({"against": s, **compare(rows, "routed", s, pos, neg)})
            if corpus == "verite":
                e["accuracy_routed"][f"{pos} vs {neg}"] = flags_accuracy(rows, "routed", "flag", pos)
        for c in sorted({r["class"] for r in rows}):
            e["branch_records"][cs.CLASS_NAMES.get(c, c)] = dict(Counter(r["branch"] for r in rows if r["class"] == c))
        if corpus == "m4fc val":
            e["macro_f1_routed"] = macro_f1_flags(rows, "flag")
        corpora[corpus] = e
    train_branch = Counter("m4fc" if m else "photo" for m in in_m4fc)
    report = {"written_at": cs.utcnow(), "post_hoc": POST_HOC, "stance": ctx["stance"],
              "clip_l14_joined": ctx["clip_l14_found"],
              "route": {"t": t, "grid": route_cv, "train_branch_records": dict(train_branch),
                        "fitted_by": "the OCR share that best agrees with the screenshot layout on M4FC train "
                                     "(balanced accuracy); 0 is not a candidate",
                        "first_objective": {"objective": "the routed system's AUROC on M4FC train",
                                            "chose_t": first_objective,
                                            "why_not_used": "it chose t = 0: every image to the M4FC branch, "
                                                            "leaving no photograph to route"},
                        "misroute_m4fc_train": misroute,
                        "misroute_definition": "a train record whose branch flags it wrongly at that branch's "
                                               "threshold while the other branch flags it rightly"},
              "ensemble": {"form": chosen.name, "cv": cv, "threshold": ens_t},
              "photo": {"scorers": [p[0] for p in PHOTO], "threshold": photo_t},
              "signed": {"converged": signed.model["converged"], "at_bound": signed.model["at_bound"],
                         "oof_auroc_train": auc(signed_oof, y)},
              "probe": {k: v for k, v in probe.items() if k not in ("model", "oof_table")},
              "corpora": corpora}
    cs.public_report(report, REPORT_ROUTED)
    REPORT_ROUTED_MD.write_text(render_routed(report), encoding="utf-8", newline="\n")
    cs.write_json(SYSTEM_CANDIDATE, system)
    print(f"route t={t}; train branches {dict(train_branch)}; misroute {misroute['all']}")
    for corpus, e in corpora.items():
        for a in e["auroc"]:
            if a["against"] == "clip_l14":
                print(f"  {corpus} {a['positive']} vs {a['negative']}: {a['system']} {a['auroc_' + a['system']]} "
                      f"vs L/14 {a['auroc_clip_l14']}, Δ {a['delta']} {a['delta_ci95']}")
        print(f"  {corpus} branches: {e['branch_records']}")
    print(f"-> {cs.rel(REPORT_ROUTED)}, {cs.rel(REPORT_ROUTED_MD)} (local), {cs.rel(SYSTEM_CANDIDATE)}")
    return 0


def render_routed(rep: dict[str, Any]) -> str:
    r3 = cs._r3
    lines = ["# The routed system (post hoc), against CLIP L/14", "", f"**{rep['post_hoc']}.**", "",
             "CLIP ViT-L/14 similarity alone scores 0.852 AUROC on VERITE out-of-context vs true (Track B); it "
             "stands beside every number, paired on the same records: `clip_l14` from Track B's embeddings, "
             "`clip_l14_ours` from this pipeline's own run of the same weights.", "",
             f"- Route: screenshot layout or OCR share >= {rep['route']['t']} -> the M4FC branch; train records "
             f"per branch {json.dumps(rep['route']['train_branch_records'])}; misroute on M4FC train "
             f"{cs._pct(rep['route']['misroute_m4fc_train']['all'])} ({rep['route']['misroute_definition']}).",
             f"- M4FC branch: {rep['ensemble']['form']}; photo branch: equal-weight average of "
             f"{', '.join(rep['photo']['scorers'])}, rank-normalised on M4FC train.",
             f"- Probe: {rep['probe'].get('why')}.", f"- Stance: {rep['stance']}.", "",
             "| corpus | pair | system | AUROC | against | its AUROC | Δ [95%] | excludes 0 |",
             "| --- | --- | --- | ---: | --- | ---: | --- | --- |"]
    for corpus, e in rep["corpora"].items():
        for a in e["auroc"]:
            lines.append(f"| {corpus} | {a['positive']} vs {a['negative']} | {a['system']} | "
                         f"{r3(a['auroc_' + a['system']])} | {a['against']} | {r3(a['auroc_' + a['against']])} | "
                         f"{_delta(a)} | {'yes' if a['excludes_zero'] else 'no'} |")
    lines += ["", "Routed against each branch alone:", ""]
    for corpus, e in rep["corpora"].items():
        for a in e["routed_vs_branches"]:
            lines.append(f"- {corpus}, {a['positive']} vs {a['negative']}: routed {r3(a['auroc_routed'])} vs "
                         f"{a['against']} {r3(a['auroc_' + a['against']])}, Δ {_delta(a)}")
    lines += ["", "Records per branch:", ""]
    for corpus, e in rep["corpora"].items():
        lines.append(f"- {corpus}: {json.dumps(e['branch_records'])}")
    lines += ["", f"Route threshold on M4FC train, fitted by {rep['route']['fitted_by']}. The first objective "
              f"({rep['route']['first_objective']['objective']}) chose t = {rep['route']['first_objective']['chose_t']}: "
              f"{rep['route']['first_objective']['why_not_used']}.", "",
              "| t | candidate | to the M4FC branch | agreement with screenshot layout (BA) | routed train AUROC |",
              "| ---: | --- | ---: | ---: | ---: |"]
    lines += [f"| {g['t']} | {'yes' if g['candidate'] else 'no'} | {g['to_m4fc_branch']} | "
              f"{r3(g['screenshot_agreement_ba'])} | {r3(g['routed_train_auroc'])} |" for g in rep["route"]["grid"]]
    v = rep["corpora"]["verite"]["accuracy_routed"]
    lines += ["", "VERITE accuracy (routed, at its branch thresholds): "
              + "; ".join(f"{t} {cs._pct(c)}" for t, c in v.items()),
              f"M4FC val macro F1 (routed): {r3(rep['corpora']['m4fc val'].get('macro_f1_routed'))}", ""]
    return "\n".join(lines) + "\n"


def cmd_freeze(args) -> int:
    """The chosen system, frozen: configs/ensemble_system.json with its
    sha256, the scorer configurations it reads, and the code commit."""
    if not SYSTEM_CANDIDATE.is_file():
        raise SystemExit("run `route` first")
    system = json.loads(SYSTEM_CANDIDATE.read_text(encoding="utf-8"))
    inputs = {n: json.loads(ledger_path(n).with_name(f"{n}.config.json").read_text(encoding="utf-8"))["config_id"]
              for n in models()}
    frozen = {"frozen_at": cs.utcnow(), "frozen_from_commit": cs._code_commit() or _git_head(),
              "scorer_configs": inputs, "image_text_config": cs.IMAGE_TEXT_CONFIG,
              "text_model_configs": dict(cs.LIVE_CONFIGS), **system}
    body = cs.canonical(frozen)
    sha = hashlib.sha256(body).hexdigest()
    cs.write_json(SYSTEM, {"system_sha256": sha, "system": frozen})
    print(f"frozen -> {cs.rel(SYSTEM)}; system sha256 {sha}")
    return 0


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def read_system(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """A frozen system (the tested one by default), refused if its body no
    longer hashes to its sha256."""
    path = path or SYSTEM
    doc = json.loads(path.read_text(encoding="utf-8"))
    if hashlib.sha256(cs.canonical(doc["system"])).hexdigest() != doc["system_sha256"]:
        raise SystemExit(f"{path.name} does not hash to its system_sha256: refusing")
    return doc["system"], doc["system_sha256"]


# --------------------------------------------------------------------------
# --final-test: M4FC's official test, opened once (approved by the person,
# 2026-11-10). `open` records the opening and refuses a second; `score` reads
# the labels once and refuses a second run. Between them, the live path
# (consistency.py live ...) computes every stage under the evaluated
# configurations, and the scorers run on the "final" set.
# --------------------------------------------------------------------------

FINAL_OPENED = FINAL_DIR / "opened.json"
FINAL_SCORED = FINAL_DIR / "scored.json"
FINAL_REQUESTS = FINAL_DIR / "requests.jsonl"  # local: image paths, claims, dates; no label
FINAL_KEYS = FINAL_DIR / "keys.json"  # test record key -> live key, both neutral
REPORT_FINAL = REPORTS / "ensemble_final_test.json"
REPORT_FINAL_MD = REPORTS / "ensemble_final_test.md"
#: M4FC's verdict prediction (the authors' task): true against false; mixture
#: and unverified are not scored.
VP_CLASSES = ("true", *M4FC_FALSE)


def test_rows() -> Iterator[dict[str, Any]]:
    """M4FC's official test rows, from the interim layer: the one reader of
    them, called by `final-test open` (claims and images) and by
    `final-test score` (labels, once)."""
    import gzip

    from scripts.build_processed import INTERIM_RECORDS

    with gzip.open(INTERIM_RECORDS / "m4fc.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["official_split"] == "test":
                yield row


def cmd_final_open(args) -> int:
    """Open the official test once: label-free live requests (each image
    through the three gates, the claim, its evidence bound), and the record
    of the opening. Refused without a frozen system, or a second time."""
    from scripts import images
    from scripts.build_processed import Resolver
    from scripts.image_text import read_bytes

    system, sha = read_system()
    if FINAL_OPENED.exists():
        raise SystemExit(f"the official test was opened {json.loads(FINAL_OPENED.read_text(encoding='utf-8'))['opened_at']}:"
                         " it opens once")
    resolver = Resolver(["m4fc"])
    reqs, keys, tally = {}, {}, Counter()
    for row in test_rows():
        claim = row.get("text") or ""
        ref = next((i for i in row.get("images") or () if i.get("role") == "claim"), None)
        status, path, _ = resolver.resolve("m4fc", ref, None) if ref else ("no_image", None, None)
        if not claim.strip() or not path:
            tally["no claim" if not claim.strip() else f"image {status}"] += 1
            continue
        body = read_bytes(path)
        examined = images.examine_bytes(body)
        if images.status({**examined, "path": path}) != images.USABLE:
            tally["image fails the gates"] += 1
            continue
        bound = cs.retrieval_items_bound(row["meta"])
        key = cs.live_key(examined["sha256"], claim, bound)
        reqs.setdefault(key, {"image": path, "claim": claim, "date": bound})
        keys[cs.record_key("m4fc", row)] = key
        tally["scored"] += 1
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(FINAL_REQUESTS, reqs.values())
    cs.write_json(FINAL_KEYS, keys)
    cs.write_json(FINAL_OPENED, {"opened_at": cs.utcnow(), "system_sha256": sha, "code_commit": _git_head(),
                                 "records": dict(tally), "requests": len(reqs)})
    print(f"official test opened: {json.dumps(dict(tally))}; {len(reqs):,} live requests -> {cs.rel(FINAL_REQUESTS)}")
    print(f"system {sha[:16]}; next: consistency.py live prepare --requests {cs.rel(FINAL_REQUESTS)}")
    return 0


def final_live_keys() -> list[str]:
    return sorted(set(json.loads(FINAL_KEYS.read_text(encoding="utf-8")).values()))


def cmd_final_inputs(args) -> int:
    """The scorers' inputs for the test, from the live run: each request's
    claim, image (its live copy, by sha256), OCR and description, and its
    retrieved documents."""
    pairs, nli, images_, tally = [], [], {}, Counter()
    for k in final_live_keys():
        req, row = cs._read_live(k, "request.json"), cs._read_live(k, "image_text.json")
        if row["sha256"] != req["sha256"]:
            raise cs.GuardError("an image_text row was joined to another image")
        claim = clean(req["claim"])
        pairs.append({"key": k, "sha256": req["sha256"], "text": claim})
        images_.setdefault(req["sha256"], cs.rel(cs.LIVE_DIR / k / req["image"]))
        ocr, desc, _ents = cs.image_part(row, req["sha256"])
        retrieval = req.get("retrieval")
        docs = cs.documents_of(retrieval) if retrieval and not retrieval.get("error") else ()
        if any(s.search(claim.casefold()) for s in cs.ID_SHAPES):
            raise cs.GuardError("a test claim holds an id shape: refusing")
        for slot, premise in nli_premises(ocr, desc, docs).items():
            # the runner refuses any id shape; a premise holding one (a scraped page's asset
            # path) is left out and counted, where the evaluation set's builder would refuse
            if any(s.search(premise.casefold()) for s in cs.ID_SHAPES):
                tally["premise left out: holds an id shape"] += 1
                continue
            nli.append({"key": cs.neutral_key("n", f"{k}|{slot}"), "record": k, "slot": slot, "premise": premise,
                        "hypothesis": claim})
            tally[f"nli {slot.rstrip('0123456789')}"] += 1
    paths = _set("final")
    write_input(paths["pairs"], pairs, "pairs")
    write_input(paths["nli"], nli, "nli")
    write_jsonl(paths["images"], [{"sha256": s, "path": p} for s, p in sorted(images_.items())])
    cs.write_json(FINAL_DIR / "pair_keys.json", [p["key"] for p in pairs])
    print(f"{len(pairs):,} test pairs, {len(nli):,} NLI pairs -> {cs.rel(FINAL_DIR)}; {json.dumps(cs.ordered(tally))}")
    return 0


def cmd_final_score(args) -> int:
    """The frozen system on the official test, its labels read once; refused
    a second time. Reports the routed system, each branch alone, CLIP L/14
    (this pipeline's run of Track B's weights) and CLIP B/32, paired."""
    system, sha = read_system()
    opened = json.loads(FINAL_OPENED.read_text(encoding="utf-8"))
    if opened["system_sha256"] != sha:
        raise SystemExit("the frozen system is not the one the test was opened under: refusing")
    if FINAL_SCORED.exists():
        raise SystemExit("the official test was scored already: it is scored once")
    live = final_live_keys()
    names = list(models())
    scores = {n: {r["key"]: r for r in read_jsonl(ledger_path(n, "final"))} for n in names
              if models()[n]["kind"] != "nli"}
    if missing := {n: len(live) - len(v) for n, v in scores.items() if len(v) < len(live)}:
        raise SystemExit(f"scorers have not covered the test: {missing}")
    nli_in = {r["key"]: r for r in read_input(_set("final")["nli"], "nli")}
    nli_by_record: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(ledger_path("nli_deberta_large", "final")):
        if r["status"] == "ok" and r["key"] in nli_in:
            nli_by_record[nli_in[r["key"]]["record"]][nli_in[r["key"]]["slot"]] = r
    reqs = {k: cs._read_live(k, "request.json") for k in live}
    recs = [{"key": k, "dataset": "m4fc", "sha256": reqs[k]["sha256"], "claim_key": cs.claim_key(reqs[k]["claim"])}
            for k in live]
    verdicts = {k: json.loads((cs.LIVE_DIR / k / "verdicts.json").read_text(encoding="utf-8")) for k in live}
    stance, stance_found = stance_rows() if "stance" in system["ensemble"]["groups"] else ({}, "not in the system")
    feats = features(recs, scores, verdicts, stance, nli_by_record)
    routes = route_features(recs, {"m4fc": {reqs[k]["sha256"]: cs._read_live(k, "image_text.json") for k in live}})
    probe_x = None
    if (system.get("probe") or {}).get("adopted"):
        probe_x = embedding_features(recs, {k: clean(reqs[k]["claim"]) for k in live},
                                     embeddings_path(PROBE_SCORER, "final"))
    out = system_scores(system, recs, feats, routes, probe_x)
    bare = system_scores({**system, "probe": {"adopted": False}}, recs, feats, routes, probe_x)
    l14 = {r["key"]: -r["cosine"] for r in read_jsonl(ledger_path("clip_l14", "final")) if r["status"] == "ok"}
    b32 = {}
    for k in live:
        p = cs.LIVE_DIR / k / "clip.json"
        if p.is_file() and (sim := json.loads(p.read_text(encoding="utf-8")).get("similarity")) is not None:
            b32[k] = -sim

    # the labels, read once
    keys = json.loads(FINAL_KEYS.read_text(encoding="utf-8"))
    rows = []
    for row in test_rows():
        rk = cs.record_key("m4fc", row)
        k = keys.get(rk)
        if k is None or row["meta"]["verdict"] not in VP_CLASSES:
            continue
        o = out[k]
        rows.append({"key": rk, "sha256": reqs[k]["sha256"], "claim_key": cs.claim_key(reqs[k]["claim"]),
                     "class": row["meta"]["verdict"], "routed": o["score"], "flag": o["flag"], "branch": o["branch"],
                     "routed_without_probe": bare[k]["score"], "flag_without_probe": bare[k]["flag"],
                     "m4fc_branch_alone": o["ensemble"], "photo_average_alone": o["average"],
                     "probe_alone": o["probe"], "clip_l14": l14.get(k), "clip_b32": b32.get(k)})
    pairs = [("false", "true"), (cs.OOC, "true")]
    systems = ["routed", "routed_without_probe", "m4fc_branch_alone", "photo_average_alone", "probe_alone"]
    auroc = [{"system": s, "against": base, **compare(rows, s, base, pos, neg)}
             for pos, neg in pairs for s in systems for base in ("clip_l14", "clip_b32")
             if any(r.get(s) is not None for r in rows)]
    report = {"written_at": cs.utcnow(), "system_sha256": sha, "post_hoc": system["post_hoc"],
              "opened": opened, "stance": stance_found, "records": len(rows),
              "classes": dict(Counter(r["class"] for r in rows)),
              "branch_records": {cs.CLASS_NAMES.get(c, c): dict(Counter(r["branch"] for r in rows if r["class"] == c))
                                 for c in sorted({r["class"] for r in rows})},
              "declared_ablations": system.get("declared_ablations"),
              "auroc": auroc, "macro_f1_routed": macro_f1_flags(rows, "flag"),
              "macro_f1_routed_without_probe": macro_f1_flags(rows, "flag_without_probe"),
              "accuracy_routed": cs.rate(sum(1 for r in rows if r["flag"] == (r["class"] != "true")), len(rows)),
              "by_class": {cs.CLASS_NAMES.get(c, c): cs.rate(sum(1 for r in rows if r["class"] == c and r["flag"]),
                                                             sum(1 for r in rows if r["class"] == c))
                           for c in sorted({r["class"] for r in rows})},
              "clip_l14_scored": sum(1 for r in rows if r["clip_l14"] is not None),
              "clip_b32_scored": sum(1 for r in rows if r["clip_b32"] is not None)}
    cs.public_report(report, REPORT_FINAL)
    REPORT_FINAL_MD.write_text(render_final(report), encoding="utf-8", newline="\n")
    cs.write_json(FINAL_SCORED, {"scored_at": report["written_at"], "system_sha256": sha,
                                 "report_sha256": cs.sha256_file(REPORT_FINAL)})
    for a in auroc:
        if a["against"] == "clip_l14":
            print(f"  test {a['positive']} vs {a['negative']}: {a['system']} {a['auroc_' + a['system']]} vs L/14 "
                  f"{a['auroc_clip_l14']}, Δ {a['delta']} {a['delta_ci95']}")
    print(f"macro F1 (routed) {report['macro_f1_routed']}; -> {cs.rel(REPORT_FINAL)}")
    return 0


def render_final(rep: dict[str, Any]) -> str:
    r3 = cs._r3
    lines = ["# The frozen system on M4FC's official test (scored once)", "", f"**{rep['post_hoc']}.**", "",
             f"System {rep['system_sha256'][:16]}; opened {rep['opened']['opened_at']}; {rep['records']} records "
             f"{json.dumps(rep['classes'])}; stance: {rep['stance']}.", "",
             "| pair | system | AUROC | against | its AUROC | Δ [95%] | excludes 0 |",
             "| --- | --- | ---: | --- | ---: | --- | --- |"]
    for a in rep["auroc"]:
        lines.append(f"| {a['positive']} vs {a['negative']} | {a['system']} | {r3(a['auroc_' + a['system']])} | "
                     f"{a['against']} | {r3(a['auroc_' + a['against']])} | {_delta(a)} | "
                     f"{'yes' if a['excludes_zero'] else 'no'} |")
    lines += ["", f"Routed, at its branch thresholds: macro F1 (true vs false) {r3(rep['macro_f1_routed'])}; "
              f"accuracy {cs._pct(rep['accuracy_routed'])}.", "",
              "Flag rate by class: " + "; ".join(f"{c} {cs._pct(v)}" for c, v in rep["by_class"].items()), "",
              f"Records per branch: {json.dumps(rep['branch_records'])}", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# v1.1, the demo system (the person's decision, 2026-11-11). POST HOC: the
# frozen system without its photo-branch probe, so the photo branch is the
# unfitted average; decided after VERITE was seen. It has no test number.
# --------------------------------------------------------------------------

SYSTEM_V11 = CONFIGS / "ensemble_system_v1_1.json"
REPORT_V11 = REPORTS / "ensemble_v1_1.json"
REPORT_V11_MD = REPORTS / "ensemble_v1_1.md"
REPORT_STANCE = REPORTS / "ensemble_stance_val.json"
REPORT_STANCE_MD = REPORTS / "ensemble_stance_val.md"
V11_NOTE = ("post hoc: v1.1 is the frozen system without its photo-branch probe, so the photo branch is the "
            "unfitted average; decided after VERITE was seen at gate E1, and after the frozen system's test was "
            "scored. v1.1 has no test number: the test report's 'routed without the probe' row is an ablation of "
            "the frozen system, declared before the test opened, and v1.1 was chosen after it was seen")
STANCE_NOTE = ("a val-only variant (the person's rule): Track B's NLI-B probabilities as features of the M4FC "
               "branch, refitted in the frozen form on M4FC train and reported on M4FC val only, never on VERITE "
               "or the test")


def write_system(path: Path, system: dict[str, Any]) -> str:
    sha = hashlib.sha256(cs.canonical(system)).hexdigest()
    cs.write_json(path, {"system_sha256": sha, "system": system})
    return sha


def cmd_v11(args) -> int:
    """v1.1 from the frozen system, and its report on M4FC val and VERITE
    beside the frozen system's, CLIP L/14 beside both."""
    frozen, fsha = read_system()
    v11 = {**frozen, "version": "1.1", "derived_from": fsha, "post_hoc": V11_NOTE,
           "probe": {"adopted": False, "removed": "v1.1 drops the frozen system's probe"}}
    v11.pop("declared_ablations", None)
    vsha = write_system(SYSTEM_V11, v11)
    ctx = load_eval()
    evals = [r for r in ctx["records"] if r["dataset"] != "factify2" and not (r["dataset"] == "m4fc"
                                                                                and r["split"] == "train")]
    routes = route_features(evals, {ds: cs.image_text_rows(ds) or {} for ds in ("m4fc", "verite")})
    texts = {r["key"]: r["text"] for r in read_input(PAIRS, "pairs")}
    probe_x = embedding_features(evals, texts, embeddings_path(PROBE_SCORER))
    feats = [ctx["by_key"][r["key"]] for r in evals]
    outs = {"v1.1": system_scores(v11, evals, feats, routes), "frozen": system_scores(frozen, evals, feats, routes,
                                                                                      probe_x)}
    preds = {n: {k: v["score"] for k, v in o.items()} for n, o in outs.items()}
    preds.update({b: ctx[b] for b in ("clip_l14", "clip_l14_ours") if ctx[b]})
    corpora = {}
    for corpus, pairs in PAIRS_OF.items():
        rows = eval_rows(evals, preds, corpus)
        for r in rows:
            for n, o in outs.items():
                r[f"flag:{n}"], r[f"branch:{n}"] = o[r["key"]]["flag"], o[r["key"]]["branch"]
        e = {"records": len(rows), "auroc": [], "by_class": {}, "accuracy": {}}
        for pos, neg in pairs:
            for n in outs:
                for base in ("clip_l14", "clip_l14_ours"):
                    if base in preds:
                        e["auroc"].append({"system": n, "against": base, **compare(rows, n, base, pos, neg)})
            e["auroc"].append({"system": "v1.1", "against": "frozen", **compare(rows, "v1.1", "frozen", pos, neg)})
            if corpus == "verite":
                e["accuracy"][f"{pos} vs {neg}"] = {n: flags_accuracy(rows, n, f"flag:{n}", pos) for n in outs}
        if corpus == "m4fc val":
            e["macro_f1"] = {n: macro_f1_flags(rows, f"flag:{n}") for n in outs}
        for c in sorted({r["class"] for r in rows}):
            sub = [r for r in rows if r["class"] == c]
            e["by_class"][cs.CLASS_NAMES.get(c, c)] = {
                n: {"flagged": cs.rate(sum(1 for r in sub if r[f"flag:{n}"]), len(sub)),
                    "branches": dict(Counter(r[f"branch:{n}"] for r in sub))} for n in outs}
        corpora[corpus] = e
    report = {"written_at": cs.utcnow(), "post_hoc": V11_NOTE, "system_sha256": vsha, "derived_from": fsha,
              "test": "none: v1.1 has no test number", "corpora": corpora}
    cs.public_report(report, REPORT_V11)
    REPORT_V11_MD.write_text(render_v11(report), encoding="utf-8", newline="\n")
    print(f"v1.1 {vsha[:16]} (from {fsha[:16]}) -> {cs.rel(SYSTEM_V11)}; report {cs.rel(REPORT_V11)}")
    for corpus, e in corpora.items():
        for a in e["auroc"]:
            if a["against"] in ("clip_l14", "frozen"):
                print(f"  {corpus} {a['positive']} vs {a['negative']}: {a['system']} {a['auroc_' + a['system']]} vs "
                      f"{a['against']} {a['auroc_' + a['against']]}, Δ {a['delta']} {a['delta_ci95']}")
    return 0


def render_v11(rep: dict[str, Any]) -> str:
    r3 = cs._r3
    lines = ["# v1.1, the demo system (post hoc), beside the frozen system and CLIP L/14", "",
             f"**{rep['post_hoc']}.**", "", f"v1.1 {rep['system_sha256'][:16]}, from {rep['derived_from'][:16]}. "
             f"Test: {rep['test']}.", "",
             "| corpus | pair | system | AUROC | against | its AUROC | Δ [95%] | excludes 0 |",
             "| --- | --- | --- | ---: | --- | ---: | --- | --- |"]
    for corpus, e in rep["corpora"].items():
        for a in e["auroc"]:
            lines.append(f"| {corpus} | {a['positive']} vs {a['negative']} | {a['system']} | "
                         f"{r3(a['auroc_' + a['system']])} | {a['against']} | {r3(a['auroc_' + a['against']])} | "
                         f"{_delta(a)} | {'yes' if a['excludes_zero'] else 'no'} |")
    for corpus, e in rep["corpora"].items():
        lines += ["", f"{corpus}, flags by class: "
                  + "; ".join(f"{c}: " + ", ".join(f"{n} {cs._pct(v['flagged'])}" for n, v in cell.items())
                              for c, cell in e["by_class"].items())]
        if e.get("accuracy"):
            lines.append("accuracy: " + "; ".join(f"{t}: " + ", ".join(f"{n} {cs._pct(v)}" for n, v in a.items())
                                                  for t, a in e["accuracy"].items()))
        if e.get("macro_f1"):
            lines.append("macro F1: " + ", ".join(f"{n} {r3(v)}" for n, v in e["macro_f1"].items()))
    return "\n".join(lines) + "\n"


def frozen_form(system: dict[str, Any]) -> Candidate:
    """The frozen M4FC branch's form, as a candidate to refit (never reselected)."""
    m = re.fullmatch(r"lr\(l2=([0-9.]+)\)", system["ensemble"]["form"])
    if not m:
        raise SystemExit(f"the frozen form {system['ensemble']['form']!r} is not a logistic regression")
    return Candidate("lr", {"l2": float(m.group(1))})


def cmd_stance(args) -> int:
    """The val-only stance variant: the M4FC branch refitted in the frozen form
    with Track B's NLI-B probabilities added, on M4FC train; reported on M4FC
    val only, beside v1.1 without it."""
    v11, vsha = read_system(SYSTEM_V11)
    stance, found = stance_rows()
    if not stance:
        raise SystemExit(f"no stance: {found}")
    ctx = load_eval()
    train, y, tf = ctx["train"], ctx["y"], ctx["tf"]
    groups = [*v11["ensemble"]["groups"], "stance"]
    form = frozen_form(v11)
    m = Candidate(form.kind, form.params).fit(tf, y, groups)
    oof = out_of_fold(form, train, tf, y, groups)
    table = sorted(oof)
    routes_train = route_features(train, {"m4fc": cs.image_text_rows("m4fc") or {}})
    in_m4fc = [to_m4fc_branch(routes_train[r["key"]], v11["route"]["t"]) for r in train]
    ens_pct = [pct(table, v) for v in oof]
    thr = branch_threshold([v for v, i in zip(ens_pct, in_m4fc) if i], [t for t, i in zip(y, in_m4fc) if i],
                           (ens_pct, y))
    variant = {**v11, "ensemble": {**v11["ensemble"], "groups": groups, "columns": m.cols, "model": m.model,
                                   "oof_table": [round(v, 6) for v in table], "threshold": thr}}
    val = [r for r in ctx["records"] if r["dataset"] == "m4fc" and r["split"] == "val"]
    routes = route_features(val, {"m4fc": cs.image_text_rows("m4fc") or {}})
    feats = [ctx["by_key"][r["key"]] for r in val]
    outs = {"v1.1": system_scores(v11, val, feats, routes), "v1.1 + stance": system_scores(variant, val, feats, routes)}
    preds = {n: {k: v["score"] for k, v in o.items()} for n, o in outs.items()}
    preds.update({f"{n} (M4FC branch alone)": {k: v["ensemble"] for k, v in o.items()} for n, o in outs.items()})
    preds.update({b: ctx[b] for b in ("clip_l14", "clip_l14_ours") if ctx[b]})
    rows = eval_rows(val, preds, "m4fc val")
    auroc = []
    for pos, neg in PAIRS_OF["m4fc val"]:
        auroc.append({"system": "v1.1 + stance", "against": "v1.1", **compare(rows, "v1.1 + stance", "v1.1", pos, neg)})
        auroc.append({"system": "v1.1 + stance (M4FC branch alone)", "against": "v1.1 (M4FC branch alone)",
                      **compare(rows, "v1.1 + stance (M4FC branch alone)", "v1.1 (M4FC branch alone)", pos, neg)})
        for n in ("v1.1 + stance", "v1.1"):
            if "clip_l14" in preds:
                auroc.append({"system": n, "against": "clip_l14", **compare(rows, n, "clip_l14", pos, neg)})
    coverage = cs.rate(sum(1 for r in val if r["claim_key"] in stance), len(val))
    report = {"written_at": cs.utcnow(), "note": STANCE_NOTE, "derived_from": vsha, "stance": found,
              "stance_coverage_m4fc_val": coverage, "train_oof_auroc": {"with_stance": auc(oof, y)},
              "columns": m.cols, "coef_stance": {c: round(w, 4) for c, w in zip(m.cols, m.model["coef"])
                                                 if c.startswith("stance")},
              "auroc_m4fc_val": auroc}
    cs.public_report(report, REPORT_STANCE)
    REPORT_STANCE_MD.write_text(render_stance(report), encoding="utf-8", newline="\n")
    for a in auroc:
        print(f"  m4fc val {a['positive']} vs {a['negative']}: {a['system']} {a['auroc_' + a['system']]} vs "
              f"{a['against']} {a['auroc_' + a['against']]}, Δ {a['delta']} {a['delta_ci95']}")
    print(f"stance coverage on M4FC val {coverage}; -> {cs.rel(REPORT_STANCE)}")
    return 0


def render_stance(rep: dict[str, Any]) -> str:
    r3 = cs._r3
    lines = ["# Text-side stance: a val-only variant of the M4FC branch", "", f"**{rep['note']}.**", "",
             f"Stance: {rep['stance']}; coverage on M4FC val {cs._pct(rep['stance_coverage_m4fc_val'])}. "
             f"Stance coefficients (standardised): {json.dumps(rep['coef_stance'])}.", "",
             "| pair | system | AUROC | against | its AUROC | Δ [95%] | excludes 0 |",
             "| --- | --- | ---: | --- | ---: | --- | --- |"]
    for a in rep["auroc_m4fc_val"]:
        lines.append(f"| {a['positive']} vs {a['negative']} | {a['system']} | {r3(a['auroc_' + a['system']])} | "
                     f"{a['against']} | {r3(a['auroc_' + a['against']])} | {_delta(a)} | "
                     f"{'yes' if a['excludes_zero'] else 'no'} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the demo with v1.1: its flag, the verdicts' explanations, and a cited Qwen
# verdict from Track B's text-side check where one exists
# --------------------------------------------------------------------------

V11_BASIS = {
    "m4fc": "The image carries text (a screenshot or a text-heavy image), so the flag comes from the M4FC branch: "
            "one logistic regression over the image-text models, the NLI checks and the consistency verdicts, "
            "fitted on M4FC train.",
    "photo": "The image is a photograph, so the flag rests on image-text similarity alone: the average of four "
             "image-text models (OpenCLIP, SigLIP, BLIP, BLIP-2), unfitted.",
}
V11_FLAG = {True: "Flagged: its score ({score}, a percentile on its branch's M4FC-train scale) is at or above the "
                  "branch's threshold ({threshold}).",
            False: "Not flagged: its score ({score}, a percentile on its branch's M4FC-train scale) is below the "
                   "branch's threshold ({threshold})."}
V11_PAGE_NOTE = ("Demo system v1.1 (post hoc: chosen after VERITE was seen; no test number). It decides the flag; "
                 "the reasons below come from the consistency verdicts alone.")
QWEN_WORDS = {"supported": "supports the claim", "refuted": "refutes the claim",
              "not_enough_evidence": "finds them not enough to settle the claim"}


def v11_flag(system: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    branch = out["branch"]
    t = (system["ensemble"] if branch == "m4fc" else system["photo"])["threshold"]["threshold"]
    words = {"score": f"{out['score']:.2f}", "threshold": f"{t:.2f}"}
    return {"flagged": out["flag"], "branch": branch, "score": out["score"], "threshold": t,
            "flag": V11_FLAG[out["flag"]].format(**words), "basis": V11_BASIS[branch], "note": V11_PAGE_NOTE}


def qwen_rows() -> dict[str, dict[str, Any]]:
    """claim_key -> Track B's Qwen verdict, where it cites documents (status
    ok): the only text-side verdicts an explanation may carry."""
    if not STANCE.is_file():
        return {}
    return {r["claim_key"]: r for r in read_jsonl(STANCE)
            if r["judge"] == "qwen" and r["status"] == "ok" and r.get("doc_ids") and r.get("verdict")}


def text_side(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """A cited Qwen verdict as a template sentence: its verdict and each cited
    document's site and date from the evidence store. NLI-B never speaks here."""
    if not row:
        return None
    docs = [cs.read_doc(d) for d in row["doc_ids"]]
    sources = "; ".join(dict.fromkeys(f"{d.get('source_domain')}, {d.get('published_at')}" for d in docs if d))
    n = len(row["doc_ids"])
    return {"verdict": row["verdict"], "doc_ids": list(row["doc_ids"]),
            "sentence": f"A separate text-side check (Qwen, reading {n} retrieved document{'s' if n != 1 else ''}, "
                        f"dated before the claim) {QWEN_WORDS[row['verdict']]} (sources: {sources}).",
            "note": "Track B's text-side check; shown as evidence, not counted in the flag."}


def cmd_demo_v11(args) -> int:
    """Demo pages under v1.1 for records C scored: the flag from v1.1, the
    verdicts' explanations, and a cited Qwen verdict where one exists."""
    v11, vsha = read_system(SYSTEM_V11)
    ctx = load_eval()
    keys = [k for k in re.split(r"[,\s]+", args.keys) if k]
    recs = {r["key"]: r for r in cs.c_records()}
    by_rec = {r["key"]: r for r in ctx["records"]}
    wanted = [by_rec[k] for k in keys]
    routes = route_features(wanted, {ds: cs.image_text_rows(ds) or {} for ds in ("m4fc", "verite")})
    out = system_scores(v11, wanted, [ctx["by_key"][r["key"]] for r in wanted], routes)
    qwen = qwen_rows()
    for r in wanted:
        v, rec = ctx["verdicts"][r["key"]], recs[r["key"]]
        bound = cs.retrieval_items_bound(rec["row"]["meta"]) if rec["dataset"] == "m4fc" else None
        bundle = cs.demo_bundle(r["key"], rec["path"], rec["sha256"], rec["claim"], None, bound, v["items"],
                                variants=v.get("variants"))
        bundle["system"] = {**v11_flag(v11, out[r["key"]]), "system_sha256": vsha}
        if ts := text_side(qwen.get(r["claim_key"])):
            bundle["text_side"] = ts
        path = cs.write_demo(bundle)
        print(f"  {r['key']}: {bundle['system']['flag']} [{bundle['system']['branch']}]"
              + (f" | text side: {ts['verdict']}" if ts else "") + f" -> {cs.rel(path)}")
    return 0


# --------------------------------------------------------------------------
# the live path under v1.1: a new image and claim through every stage, then
# the five scorers on the server, then v1.1's flag on the laptop
# --------------------------------------------------------------------------

def build_live_inputs(keys: Sequence[str], set_name: str) -> Counter:
    """The scorers' inputs for live requests: each request's claim, its image
    (the live copy, by sha256), its OCR and description, and its retrieved
    documents. A premise holding an id shape is left out and counted."""
    pairs, nli, images_, tally = [], [], {}, Counter()
    for k in keys:
        req, row = cs._read_live(k, "request.json"), cs._read_live(k, "image_text.json")
        if row["sha256"] != req["sha256"]:
            raise cs.GuardError("an image_text row was joined to another image")
        claim = clean(req["claim"])
        if any(s.search(claim.casefold()) for s in cs.ID_SHAPES):
            raise cs.GuardError("a claim holds an id shape: refusing")
        pairs.append({"key": k, "sha256": req["sha256"], "text": claim})
        images_.setdefault(req["sha256"], cs.rel(cs.LIVE_DIR / k / req["image"]))
        ocr, desc, _ents = cs.image_part(row, req["sha256"])
        retrieval = req.get("retrieval")
        docs = cs.documents_of(retrieval) if retrieval and not retrieval.get("error") else ()
        for slot, premise in nli_premises(ocr, desc, docs).items():
            if any(s.search(premise.casefold()) for s in cs.ID_SHAPES):
                tally["premise left out: holds an id shape"] += 1
                continue
            nli.append({"key": cs.neutral_key("n", f"{k}|{slot}"), "record": k, "slot": slot, "premise": premise,
                        "hypothesis": claim})
            tally[f"nli {slot.rstrip('0123456789')}"] += 1
    paths = _set(set_name)
    paths["scores"].parent.mkdir(parents=True, exist_ok=True)
    write_input(paths["pairs"], pairs, "pairs")
    write_input(paths["nli"], nli, "nli")
    write_jsonl(paths["images"], [{"sha256": s, "path": p} for s, p in sorted(images_.items())])
    tally["pairs"] = len(pairs)
    return tally


def live_features(keys: Sequence[str], set_name: str = "live"
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """(records, features, routes) of live requests, from the set's ledgers,
    each request's verdicts and its image_text row."""
    reqs = {k: cs._read_live(k, "request.json") for k in keys}
    scores = {n: {r["key"]: r for r in read_jsonl(ledger_path(n, set_name))} for n in models()
              if models()[n]["kind"] != "nli"}
    if missing := {n: [k for k in keys if k not in v] for n, v in scores.items() if any(k not in v for k in keys)}:
        raise SystemExit(f"the scorers have not scored every request: {missing}")
    nli_in = {r["key"]: r for r in read_input(_set(set_name)["nli"], "nli")}
    nli_by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(ledger_path("nli_deberta_large", set_name)):
        if r["status"] == "ok" and r["key"] in nli_in:
            nli_by[nli_in[r["key"]]["record"]][nli_in[r["key"]]["slot"]] = r
    recs = [{"key": k, "dataset": "live", "sha256": reqs[k]["sha256"], "claim_key": cs.claim_key(reqs[k]["claim"])}
            for k in keys]
    verdicts = {k: json.loads((cs.LIVE_DIR / k / "verdicts.json").read_text(encoding="utf-8")) for k in keys}
    feats = features(recs, scores, verdicts, {}, nli_by)
    routes = route_features(recs, {"live": {reqs[k]["sha256"]: cs._read_live(k, "image_text.json") for k in keys}})
    return recs, feats, routes


def cmd_live(args) -> int:
    """[laptop] The ensemble's part of the live path: `inputs` for the
    server's scorers, `score` for v1.1's flag and the page, `check` against
    the batch for requests whose image and claim are C's records."""
    keys = cs.live_keys(args.keys)
    if args.step == "inputs":
        print(f"live scorer inputs: {json.dumps(cs.ordered(build_live_inputs(keys, 'live')))} -> {cs.rel(LIVE_ENS)}")
        return 0
    v11, vsha = read_system(SYSTEM_V11)
    recs, feats, routes = live_features(keys)
    out = system_scores(v11, recs, feats, routes)
    if args.step == "score":
        qwen = qwen_rows()
        # a request whose image and claim are an M4FC train record is in-sample: the M4FC branch was fitted on it
        train = {(t["sha256"], t["claim_key"]) for t in read_jsonl(RECORDS)
                 if t["dataset"] == "m4fc" and t["split"] == "train"}
        for r in recs:
            k, d = r["key"], cs.LIVE_DIR / r["key"]
            req, rec = cs._read_live(k, "request.json"), cs._read_live(k, "verdicts.json")
            flag = {**v11_flag(v11, out[k]), "system_sha256": vsha}
            in_sample = (r["sha256"], r["claim_key"]) in train
            if in_sample:
                flag["note"] += (" In-sample: this image and claim are an M4FC train record, and the M4FC branch "
                                 "was fitted on M4FC train, so this flag is not a held-out result.")
            cs.write_json(d / "v11.json", {"system_sha256": vsha, "in_sample": in_sample, **out[k]})
            bundle = cs.demo_bundle(k, str(d / req["image"]), req["sha256"], req["claim"], req["date"], req["date"],
                                    rec["items"], [f"A live run under demo system v1.1 ({vsha[:12]}): every stage "
                                                   "ran under the evaluated configurations."], rec["variants"])
            bundle["system"] = flag
            if ts := text_side(qwen.get(r["claim_key"])):
                bundle["text_side"] = ts
            path = cs.write_demo(bundle, (), d)
            print(f"  {k}: {flag['flag']} [{flag['branch']}] -> {cs.rel(path)}")
        return 0
    # check: each request whose image and claim are a C record, against v1.1 on the batch's own inputs
    ctx = load_eval()
    by = {(r["sha256"], r["claim_key"]): r for r in ctx["records"] if r["dataset"] != "factify2"}
    batch_recs = [by.get((r["sha256"], r["claim_key"])) for r in recs]
    pairs_ = [(r, b) for r, b in zip(recs, batch_recs) if b]
    if not pairs_:
        print("no request is a C record: nothing to compare")
        return 0
    bmine = [b for _, b in pairs_]
    broutes = route_features(bmine, {ds: cs.image_text_rows(ds) or {} for ds in ("m4fc", "verite")})
    bout = system_scores(v11, bmine, [ctx["by_key"][b["key"]] for b in bmine], broutes)
    failed = 0
    for (r, b), f in zip(pairs_, [feats[recs.index(r)] for r, _ in pairs_]):
        lf, bf = f, ctx["by_key"][b["key"]]
        worst = max((abs(lf[c] - bf[c]) for c in lf if isinstance(lf[c], float) and isinstance(bf.get(c), float)),
                    default=0.0)
        same = (out[r["key"]]["branch"] == bout[b["key"]]["branch"] and out[r["key"]]["flag"] == bout[b["key"]]["flag"]
                and abs(out[r["key"]]["score"] - bout[b["key"]]["score"]) <= 1e-6)
        failed += 0 if (same and worst <= 1e-4) else 1
        print(f"  {r['key']}: features differ by at most {worst:.1e}; v1.1 live {out[r['key']]['score']} "
              f"[{out[r['key']]['branch']}] vs batch {bout[b['key']]['score']} [{bout[b['key']]['branch']}]: "
              f"{'identical' if same else 'DIFFERS'}")
    return 0 if not failed else 2


# --------------------------------------------------------------------------
# the test, a second time, for comparison only (the person's approval,
# 2026-11-12): the frozen system's per-record outputs replayed from the stored
# inputs (they must reproduce the sealed report to four decimals), then one
# logged second read of the labels, for paired differences against Track B's
# baselines. Nothing is tuned or changed.
# --------------------------------------------------------------------------

FINAL_REPLAY = FINAL_DIR / "system_scores.jsonl"
FINAL_SECOND_READ = FINAL_DIR / "second_read.json"
TRACKB_TEST = INTERIM / "textcheck" / "test_baselines.jsonl"
REPORT_COMPARE = REPORTS / "ensemble_test_comparison.json"
REPORT_COMPARE_MD = REPORTS / "ensemble_test_comparison.md"
#: Track B's test baselines the frozen system is paired against: its text-only
#: claim classifier, its best MUSE variant (image, text and evidence, MLP) and
#: CLIP L/14's similarity.
TRACKB_AGAINST = ("text_only_p_false", "muse_img_text_evidence_mlp", "clip_l14_cosine")


def final_outputs(system: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The frozen system's per-record test outputs, from the stored inputs by
    final-test score's own path: no label is read."""
    live = final_live_keys()
    scores = {n: {r["key"]: r for r in read_jsonl(ledger_path(n, "final"))} for n in models()
              if models()[n]["kind"] != "nli"}
    nli_in = {r["key"]: r for r in read_input(_set("final")["nli"], "nli")}
    nli_by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(ledger_path("nli_deberta_large", "final")):
        if r["status"] == "ok" and r["key"] in nli_in:
            nli_by[nli_in[r["key"]]["record"]][nli_in[r["key"]]["slot"]] = r
    reqs = {k: cs._read_live(k, "request.json") for k in live}
    recs = [{"key": k, "dataset": "m4fc", "sha256": reqs[k]["sha256"], "claim_key": cs.claim_key(reqs[k]["claim"])}
            for k in live]
    verdicts = {k: json.loads((cs.LIVE_DIR / k / "verdicts.json").read_text(encoding="utf-8")) for k in live}
    feats = features(recs, scores, verdicts, {}, nli_by)
    routes = route_features(recs, {"m4fc": {reqs[k]["sha256"]: cs._read_live(k, "image_text.json") for k in live}})
    probe_x = None
    if (system.get("probe") or {}).get("adopted"):
        probe_x = embedding_features(recs, {k: clean(reqs[k]["claim"]) for k in live},
                                     embeddings_path(PROBE_SCORER, "final"))
    out = system_scores(system, recs, feats, routes, probe_x)
    bare = system_scores({**system, "probe": {"adopted": False}}, recs, feats, routes, probe_x)
    l14 = {r["key"]: -r["cosine"] for r in read_jsonl(ledger_path("clip_l14", "final")) if r["status"] == "ok"}
    res = {}
    for k in live:
        p = cs.LIVE_DIR / k / "clip.json"
        sim = json.loads(p.read_text(encoding="utf-8")).get("similarity") if p.is_file() else None
        o = out[k]
        res[k] = {"live": k, "sha256": reqs[k]["sha256"], "claim_key": cs.claim_key(reqs[k]["claim"]),
                  "routed": o["score"], "flag": o["flag"], "branch": o["branch"],
                  "routed_without_probe": bare[k]["score"], "flag_without_probe": bare[k]["flag"],
                  "m4fc_branch_alone": o["ensemble"], "photo_average_alone": o["average"], "probe_alone": o["probe"],
                  "clip_l14": l14.get(k), "clip_b32": None if sim is None else -sim}
    return res


def cmd_final_replay(args) -> int:
    """The frozen system's per-record test outputs, replayed and stored; no
    label is read."""
    system, sha = read_system()
    if not FINAL_SCORED.exists():
        raise SystemExit("the test has not been scored: nothing to replay")
    res = final_outputs(system)
    write_jsonl(FINAL_REPLAY, res.values())
    cs.write_json(FINAL_REPLAY.with_suffix(".meta.json"), {"system_sha256": sha, "rows": len(res),
                                                           "sha256": cs.sha256_file(FINAL_REPLAY),
                                                           "written_at": cs.utcnow()})
    print(f"{len(res):,} test outputs replayed (no label read) -> {cs.rel(FINAL_REPLAY)}")
    return 0


def cmd_final_compare(args) -> int:
    """The one second read of the test labels, logged before it happens, for
    paired comparison only: it first requires the replay to reproduce the
    sealed report to four decimals, then pairs the frozen system against Track
    B's baselines. Refused a second time."""
    system, sha = read_system()
    if FINAL_SECOND_READ.exists():
        raise SystemExit("the second, comparison-only read happened already: it happens once")
    for p in (FINAL_SCORED, FINAL_REPLAY, TRACKB_TEST, REPORT_FINAL):
        if not p.exists():
            raise SystemExit(f"{cs.rel(p)} is missing")
    meta = json.loads(FINAL_REPLAY.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if meta["system_sha256"] != sha or cs.sha256_file(FINAL_REPLAY) != meta["sha256"]:
        raise SystemExit("the replay is not the frozen system's, or not the file its sidecar describes")
    replay = {r["live"]: r for r in read_jsonl(FINAL_REPLAY)}
    trackb = {r["key"]: r for r in read_jsonl(TRACKB_TEST)}
    log = {"read_at": cs.utcnow(), "what": "M4FC's official test verdicts, a second time, for paired comparison "
                                           "only: nothing is tuned or changed", "system_sha256": sha,
           "replay_sha256": meta["sha256"], "trackb_test_sha256": cs.sha256_file(TRACKB_TEST),
           "code_commit": _git_head()}
    cs.write_json(FINAL_SECOND_READ, log)  # logged before the read
    keys = json.loads(FINAL_KEYS.read_text(encoding="utf-8"))
    rows = []
    for row in test_rows():
        rk = cs.record_key("m4fc", row)
        k = keys.get(rk)
        if k is None or row["meta"]["verdict"] not in VP_CLASSES:
            continue
        rows.append({**replay[k], "key": rk, "class": row["meta"]["verdict"],
                     **{f"trackb:{s}": (trackb.get(rk) or {}).get(s) for s in TRACKB_AGAINST}})
    # the replay must reproduce the sealed report, to four decimals
    sealed = json.loads(REPORT_FINAL.read_text(encoding="utf-8"))
    mismatches = []
    for a in sealed["auroc"]:
        got = compare(rows, a["system"], a["against"], a["positive"], a["negative"])
        for f in (f"auroc_{a['system']}", f"auroc_{a['against']}", "delta"):
            if round(got[f], 4) != round(a[f], 4):
                mismatches.append(f"{a['system']} vs {a['against']} {a['positive']}: {f} {got[f]} != {a[f]}")
    if sealed["records"] != len(rows) or macro_f1_flags(rows, "flag") != sealed["macro_f1_routed"]:
        mismatches.append("records or macro F1")
    if mismatches:
        cs.write_json(FINAL_SECOND_READ, {**log, "reproduced": False, "mismatches": mismatches})
        raise SystemExit(f"the replay does not reproduce the sealed report: {mismatches[:3]}")
    joined = sum(1 for r in rows if all(r[f"trackb:{s}"] is not None for s in TRACKB_AGAINST))
    pairs = []
    for pos, neg in (("false", "true"), (cs.OOC, "true")):
        for s in TRACKB_AGAINST:
            pairs.append({"system": "routed", "against": f"trackb:{s}", **compare(rows, "routed", f"trackb:{s}", pos, neg)})
    report = {"written_at": cs.utcnow(), "second_read": log, "reproduced_sealed_report": True,
              "records": len(rows), "trackb_joined": joined, "system_sha256": sha,
              "note": "a second, comparison-only read of the test labels; the frozen system's outputs are its "
                      "replayed scores, which reproduce the sealed report to four decimals; nothing is tuned",
              "auroc": pairs}
    cs.public_report(report, REPORT_COMPARE)
    REPORT_COMPARE_MD.write_text(render_compare(report), encoding="utf-8", newline="\n")
    cs.write_json(FINAL_SECOND_READ, {**log, "reproduced": True, "report_sha256": cs.sha256_file(REPORT_COMPARE)})
    for a in pairs:
        print(f"  test {a['positive']} vs {a['negative']}: frozen {a['auroc_routed']} vs {a['against']} "
              f"{a['auroc_' + a['against']]}, Δ {a['delta']} {a['delta_ci95']}")
    return 0


def render_compare(rep: dict[str, Any]) -> str:
    r3 = cs._r3
    lines = ["# The frozen system against Track B's baselines on M4FC's official test", "", f"**{rep['note']}.**", "",
             f"{rep['records']} records; Track B's scores joined for {rep['trackb_joined']}.", "",
             "| pair | frozen system | against | its AUROC | Δ [95%] | excludes 0 |", "| --- | ---: | --- | ---: | --- | --- |"]
    for a in rep["auroc"]:
        lines.append(f"| {a['positive']} vs {a['negative']} | {r3(a['auroc_routed'])} | {a['against']} | "
                     f"{r3(a['auroc_' + a['against']])} | {_delta(a)} | {'yes' if a['excludes_zero'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch-models", help="[server] the pinned weights; the one network command")
    s.add_argument("--scorer", action="append")
    sub.add_parser("inputs", help="[laptop] id-free scorer inputs")
    s = sub.add_parser("score", help="[server, GPU] one scorer over its input; resumable")
    s.add_argument("--scorer", required=True, choices=list(all_scorers()))
    s.add_argument("--set", default="eval", choices=("eval", "final", "live"))
    s.add_argument("--keys", type=Path, help="a JSON list of the pair keys to score")
    s.add_argument("--limit", type=int, default=0, help="score only the first N rows still to do (a pilot)")
    s = sub.add_parser("recheck", help="[server, GPU] n rows again in a fresh process")
    s.add_argument("--scorer", required=True, choices=list(all_scorers()))
    s.add_argument("--set", default="eval", choices=("eval", "final", "live"))
    s.add_argument("--n", type=int, default=3)
    s = sub.add_parser("embed", help="[server, GPU] a dual scorer's embeddings, for the linear probe")
    s.add_argument("--scorer", required=True, choices=list(all_scorers()))
    s.add_argument("--set", default="eval", choices=("eval", "final", "live"))
    s.add_argument("--keys", type=Path)
    sub.add_parser("fit", help="[laptop] the combiner, its ablation and gate E1's report")
    sub.add_parser("route", help="[laptop] the routed system (post hoc): fit on M4FC train, report")
    sub.add_parser("freeze", help="[laptop] freeze the routed system: configs/ensemble_system.json + sha256")
    sub.add_parser("v11", help="[laptop] demo system v1.1 (post hoc): the frozen system without its probe")
    sub.add_parser("stance", help="[laptop] the val-only stance variant of the M4FC branch")
    s = sub.add_parser("demo-v11", help="[laptop] demo pages under v1.1 for C's records")
    s.add_argument("--keys", required=True)
    s = sub.add_parser("live", help="[laptop] the live path under v1.1: inputs, score, check")
    s.add_argument("step", choices=("inputs", "score", "check"))
    s.add_argument("--keys", required=True)
    s = sub.add_parser("final-test", help="[laptop] M4FC's official test, once: open, inputs, score")
    s.add_argument("step", choices=("open", "inputs", "score", "replay", "compare"))
    args = parser.parse_args(argv)
    return {"fetch-models": cmd_fetch_models, "inputs": cmd_inputs, "score": cmd_score,
            "recheck": cmd_recheck, "embed": cmd_embed, "fit": cmd_fit, "route": cmd_route,
            "freeze": cmd_freeze, "v11": cmd_v11, "stance": cmd_stance, "demo-v11": cmd_demo_v11, "live": cmd_live,
            "final-test": lambda a: {"open": cmd_final_open, "inputs": cmd_final_inputs,
                                     "score": cmd_final_score, "replay": cmd_final_replay,
                                     "compare": cmd_final_compare}[a.step](a)}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
