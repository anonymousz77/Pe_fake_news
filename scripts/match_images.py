#!/usr/bin/env python
"""Route 4: find a missing image among the files already on disk.

    python scripts/match_images.py url --all
    python scripts/match_images.py clip-embed            # GPU; every usable image
    python scripts/match_images.py clip-calibrate        # on VERITE images we hold
    python scripts/match_images.py clip-verify           # every counted VERITE image vs its fingerprint
    python scripts/match_images.py clip-match --threshold 0.97

A content-hash match needs the missing bytes, which by definition we do not
have, so a literal hash match is impossible for a missing image. What a
dataset can carry instead is either the same ADDRESS for another of its items
or a FINGERPRINT of the content:

* url       -- the exact URL (scheme and host case normalised) already resolved
               to a usable file for another item, in any dataset. Same address,
               same bytes; the match points at that file rather than copying it.
* clip      -- VERITE ships a CLIP ViT-B/32 image embedding for every row,
               computed by its authors from the images they had, including
               the 173 we cannot fetch. Every usable image on disk is embedded
               with the same model and a missing row is matched when cosine
               similarity clears a threshold calibrated on VERITE rows whose
               image we DO hold. Candidates are reviewed before acceptance.

Matches are written to data/interim/recovery/<dataset>_matches.jsonl; nothing
is copied into data/raw. scripts/recover.resolve_image() consults them last.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, PROJECT_ROOT, raw_dir  # noqa: E402
from scripts import images, recover  # noqa: E402
from scripts.hydrate import RESOLVERS  # noqa: E402

EMB_DIR = INTERIM / "clip"


def norm_url(url: str) -> str:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{p.path}" + (f"?{p.query}" if p.query else "")


def _items(name: str):
    resolver = RESOLVERS[name]
    items, _l, _e = resolver(None) if name == "fakeddit" else resolver()
    return items


def url_matches(names: list[str]) -> dict[str, int]:
    """Missing items whose URL already resolved to a usable file elsewhere."""
    index = {n: images.load_index(n) for n in recover.DATASETS}
    have: dict[str, tuple[str, str, Path]] = {}
    for n in recover.DATASETS:
        recorded = recover.ledger_paths(n)
        for item in _items(n):
            path = recover.resolve_image(n, item, index[n], recorded, None)
            if path is not None:
                have.setdefault(norm_url(item.url), (n, item.key, path))
    out = {}
    for name in names:
        _all, missing = recover.missing_items(name, index[name])
        rows = []
        for item in missing:
            hit = have.get(norm_url(item.url))
            if hit is None:
                for alt, _kind in recover.declared_alternates(name, item):
                    hit = have.get(norm_url(alt))
                    if hit:
                        break
            if hit is None:
                continue
            src_ds, src_key, path = hit
            row = index[src_ds].get(images.rel(path))
            rows.append({"key": item.key, "path": images.rel(path), "sha256": row["sha256"],
                         "method": "url_identity", "source_dataset": src_ds,
                         "source_key": src_key, "row": row})
        _write_matches(name, rows, method="url_identity")
        out[name] = len(rows)
    return out


def _write_matches(name: str, rows: list[dict[str, Any]], method: str) -> None:
    path = recover.RECOVERY / f"{name}_matches.jsonl"
    keep = []
    if path.is_file():
        keep = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        keep = [r for r in keep if r.get("method") != method]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in keep + rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# CLIP fingerprints (VERITE)
# --------------------------------------------------------------------------

CLIP_MODEL = "openai/clip-vit-base-patch32"


def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval().to("cuda")
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return torch, model, proc


#: Images are decoded no smaller than this on their SHORT side -- twice what
#: CLIP's processor keeps (224) -- so JPEG draft decoding can skip the work of
#: full-resolution pixels without changing what the model sees.
CLIP_MIN_SIDE = 448
SHARD = 20_000

_local = threading.local()


def clip_load(relpath: str):
    """RGB image for CLIP: draft-decoded, short side >= CLIP_MIN_SIDE where it was."""
    from PIL import Image

    if "::" in relpath:
        archive, member = relpath.split("::", 1)
        zips = _local.__dict__.setdefault("zips", {})
        if archive not in zips:
            zips[archive] = zipfile.ZipFile(PROJECT_ROOT / archive)
        img = Image.open(io.BytesIO(zips[archive].read(member)))
    else:
        img = Image.open(PROJECT_ROOT / relpath)
    w, h = img.size
    scale = CLIP_MIN_SIDE / max(1, min(w, h))
    if scale < 1:
        img.draft("RGB", (math.ceil(w * scale), math.ceil(h * scale)))
    img = img.convert("RGB")
    w, h = img.size
    scale = CLIP_MIN_SIDE / max(1, min(w, h))
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return img


def _load_or_blank(relpath: str):
    try:
        return clip_load(relpath), True
    except Exception:  # noqa: BLE001
        from PIL import Image
        return Image.new("RGB", (224, 224)), False


def embed(paths: list[str], batch: int = 128, threads: int = 6, model=None):
    """L2-normalised image embeddings, in the order given (unloadable -> zero row)."""
    import numpy as np
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    torch, clip_model, proc = model or _clip()

    out = []
    with ThreadPoolExecutor(threads) as pool:
        chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]
        # a bounded window of decoded chunks: pool.map would decode the whole
        # shard ahead of the GPU and hold every image in memory
        window: deque = deque()
        feed = iter(chunks)
        for c in feed:
            window.append(pool.submit(lambda c=c: [_load_or_blank(p) for p in c]))
            if len(window) >= threads + 2:
                break
        k = -1
        while window:
            loaded = window.popleft().result()
            nxt = next(feed, None)
            if nxt is not None:
                window.append(pool.submit(lambda c=nxt: [_load_or_blank(p) for p in c]))
            k += 1
            imgs = [im for im, _ok in loaded]
            with torch.no_grad():
                inputs = proc(images=imgs, return_tensors="pt").to("cuda")
                feats = clip_model.get_image_features(**inputs)
                feats = feats if hasattr(feats, "norm") else feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.float().cpu().numpy()
            arr[[not ok for _im, ok in loaded]] = 0.0
            out.append(arr)
            if k % 20 == 0:
                print(f"  {k * batch + len(imgs):,}/{len(paths):,}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 512), dtype="float32")


def clip_embed(datasets: list[str]) -> dict[str, int]:
    """Embed every usable image, caching by content hash.

    The cache (``<name>.cache/*.npz``, one file per shard of SHARD new hashes)
    is keyed by sha256, so a re-run after recovery adds files embeds only the
    new bytes, and an interrupted run resumes at the last finished shard.
    ``<name>.npy`` and ``<name>.paths.json`` are then written in path order.
    """
    import numpy as np

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    counts = {}
    for name in datasets:
        rows = [r for r in images.load_index(name).values() if images.is_usable(r)]
        paths = sorted(r["path"] for r in rows)
        sha = {r["path"]: r["sha256"] for r in rows}
        listing = [{"path": p, "sha256": sha[p]} for p in paths]
        cache_dir = EMB_DIR / f"{name}.cache"
        cache_dir.mkdir(exist_ok=True)
        cache: dict[str, Any] = {}
        for f in sorted(cache_dir.glob("*.npz")):
            with np.load(f) as z:
                cache.update(zip(z["sha256"].tolist(), z["vecs"]))
        todo: dict[str, str] = {}
        for p in paths:
            if sha[p] not in cache:
                todo.setdefault(sha[p], p)
        items = list(todo.items())
        for s, i in enumerate(range(0, len(items), SHARD)):
            chunk = items[i:i + SHARD]
            print(f"[{name}] {len(cache):,} cached, shard {s + 1}/{-(-len(items) // SHARD)} "
                  f"of new bytes", flush=True)
            model = model or _clip()
            vecs = embed([p for _h, p in chunk], model=model).astype("float16")
            stamp = hashlib.sha256("".join(h for h, _p in chunk).encode()).hexdigest()[:16]
            np.savez(cache_dir / f"{stamp}.npz", sha256=np.array([h for h, _p in chunk]), vecs=vecs)
            cache.update(zip([h for h, _p in chunk], vecs))
        vecs = (np.stack([cache[sha[p]] for p in paths]) if paths
                else np.zeros((0, 512), dtype="float16"))
        np.save(EMB_DIR / f"{name}.npy", vecs.astype("float16"))
        (EMB_DIR / f"{name}.paths.json").write_text(json.dumps(listing), encoding="utf-8")
        counts[name] = len(paths)
    return counts


def verite_fingerprints():
    """Row index -> shipped CLIP embedding (normalised), and row index -> image key."""
    import csv

    import numpy as np

    base = next(raw_dir("verite").rglob("VERITE.csv")).parent
    vecs = np.load(base / "VERITE_clip_image_embeddings_ViTB32.npy").astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    with (base / "VERITE.csv").open(encoding="utf-8") as h:
        keys = [row["image_path"] for row in csv.DictReader(h)]
    return vecs, keys


def clip_calibrate() -> dict[str, Any]:
    """Similarity of the shipped fingerprint to OUR embedding of the same image."""
    import numpy as np

    vecs, keys = verite_fingerprints()
    paths = json.loads((EMB_DIR / "verite.paths.json").read_text(encoding="utf-8"))
    mine = np.load(EMB_DIR / "verite.npy").astype("float32")
    pos = {p["path"]: i for i, p in enumerate(paths)}
    same, other = [], []
    for row, key in enumerate(keys):
        rel = images.rel(raw_dir("verite") / "data" / "VERITE" / key)
        if rel in pos:
            sims = mine @ vecs[row]
            same.append(float(sims[pos[rel]]))
            other.append(float(np.max(np.delete(sims, pos[rel]))))
    out = {"pairs": len(same),
           "same_image": {"min": min(same), "p01": float(np.percentile(same, 1)),
                          "median": float(np.median(same))},
           "best_other_image": {"max": max(other), "p99": float(np.percentile(other, 99)),
                                "median": float(np.median(other))}}
    (EMB_DIR / "verite_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _public(path: str) -> str:
    """A VERITE image path under its label-free alias (records.verite_alias)."""
    from scripts.records import verite_alias, verite_rows_by_image

    if not path.startswith("data/raw/verite/"):
        return path
    name = path.rsplit("/", 1)[-1]
    alias = verite_alias(name, verite_rows_by_image())
    return path[: -len(name)] + alias if alias else path


def clip_verify(review_below: float = 0.90) -> dict[str, Any]:
    """Is every VERITE image we count the image VERITE's authors used?

    For every row that resolves to a usable file, cosine between OUR embedding
    of that file and the authors' shipped fingerprint for the row. Rows below
    ``review_below`` are listed for visual review; files found to be wrong go
    into data/fingerprint_rejections.yaml (or, if content-free anywhere, the
    placeholder registry) and stop resolving.
    """
    import numpy as np
    from scripts.hydrate import Item
    from scripts.records import load_verite

    vecs, _keys = verite_fingerprints()
    index = images.load_index("verite")
    recorded, matches = recover.ledger_paths("verite"), recover.load_matches("verite")
    resolved = []
    for row, rec in enumerate(load_verite()):
        ref = rec.images[0]
        p = recover.resolve_image("verite", Item(ref.ref, ref.url or "", rec.label), index, recorded, matches)
        if p is not None:
            resolved.append((row, rec.label, images.rel(p)))
    paths = sorted({p for _r, _l, p in resolved})
    mine = dict(zip(paths, embed(paths)))
    rows, by_source = [], {}
    # no label in the written rows: VERITE is not redistributable, and the tracked
    # report carries row indices and paths only
    for row, _label, p in resolved:
        sim = float(mine[p] @ vecs[row])
        if not p.startswith("data/raw/verite/"):
            source = f"match:{p.split('/')[2]}"
        else:
            source = next((d for d in ("images_wayback", "images_retry", "images_alternate")
                           if f"/{d}/" in p), "images")
        by_source.setdefault(source, []).append(sim)
        rows.append({"row": row, "file": _public(p), "source": source, "cosine": round(sim, 4)})
    out = {"rows_resolved": len(rows), "review_below": review_below,
           "by_source": {s: {"rows": len(v), "median": round(float(np.median(v)), 4),
                             "min": round(min(v), 4), "below_review": sum(x < review_below for x in v)}
                         for s, v in sorted(by_source.items())},
           "for_review": sorted((r for r in rows if r["cosine"] < review_below), key=lambda r: r["cosine"]),
           "rejected": list(images.fingerprint_rejections().values())}
    from configs.paths import REPORTS

    (REPORTS / "verite_fingerprint_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def clip_match(threshold: float, datasets: list[str]) -> list[dict[str, Any]]:
    """Missing VERITE rows whose fingerprint matches an on-disk image."""
    import numpy as np

    vecs, keys = verite_fingerprints()
    _all, missing = recover.missing_items("verite")
    missing_keys = {it.key for it in missing}
    pool_paths, pool = [], []
    for name in datasets:
        f = EMB_DIR / f"{name}.npy"
        if f.is_file():
            pool.append(np.load(f).astype("float32"))
            pool_paths += json.loads((EMB_DIR / f"{name}.paths.json").read_text(encoding="utf-8"))
    mat = np.concatenate(pool)
    # a manipulated image (DGM4's face edits) can never stand in for the
    # authors' photograph, however close its embedding
    for i, p in enumerate(pool_paths):
        if "/dgm4/manipulation/" in p["path"]:
            mat[i] = 0.0
    found: dict[str, dict[str, Any]] = {}
    for row, key in enumerate(keys):
        if key not in missing_keys or key in found:
            continue
        sims = mat @ vecs[row]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            found[key] = {"key": key, "path": pool_paths[best]["path"],
                          "sha256": pool_paths[best]["sha256"], "cosine": float(sims[best]),
                          "method": "clip_fingerprint", "row_index": row}
    return list(found.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    u = sub.add_parser("url")
    u.add_argument("--dataset", action="append", dest="datasets", choices=recover.DATASETS)
    u.add_argument("--all", action="store_true")
    e = sub.add_parser("clip-embed")
    e.add_argument("--dataset", action="append", dest="datasets", choices=images.IMAGE_DATASETS)
    sub.add_parser("clip-calibrate")
    sub.add_parser("clip-verify")
    m = sub.add_parser("clip-match")
    m.add_argument("--threshold", type=float, required=True)
    m.add_argument("--accept", action="store_true",
                   help="write the matches (after reviewing the printed candidates)")
    m.add_argument("--keys", nargs="*", default=None,
                   help="with --accept: write only these reviewed keys")
    args = parser.parse_args(argv)
    if args.command == "url":
        names = list(recover.DATASETS) if args.all or not args.datasets else args.datasets
        print(json.dumps(url_matches(names), indent=2))
    elif args.command == "clip-embed":
        print(json.dumps(clip_embed(args.datasets or list(images.IMAGE_DATASETS)), indent=2))
    elif args.command == "clip-calibrate":
        print(json.dumps(clip_calibrate(), indent=2))
    elif args.command == "clip-verify":
        out = clip_verify()
        print(json.dumps({k: v for k, v in out.items() if k != "rejected"}, indent=2))
    else:
        others = [n for n in images.IMAGE_DATASETS if n != "verite"]
        found = clip_match(args.threshold, others)
        for f in found:
            print(f"  {f['key']}  cos {f['cosine']:.4f}  -> {f['path']}")
        if args.accept and args.keys is not None:
            found = [f for f in found if f["key"] in set(args.keys)]
        if args.accept:
            idx = {}
            for n in images.IMAGE_DATASETS:
                idx.update(images.load_index(n))
            for f in found:
                f["row"] = idx.get(f["path"])
            _write_matches("verite", found, method="clip_fingerprint")
        print(f"{len(found)} candidate(s){' written' if args.accept else ' (not written)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
