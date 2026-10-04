#!/usr/bin/env python
"""Is the mismatch a property of the image-text PAIR? Ask CLIP.

    python scripts/clip_pairs.py --orig-sample 8000      # GPU; -> data/reports/clip_pairs.json

The text-side audits (scripts/fakeddit_classes.py, scripts/dgm4_audit.py) show
what a model can learn WITHOUT the pair. This is the converse: if the defect
lives in the relationship between image and caption, the image-text
similarity of a pretrained joint embedding should be lower for defective pairs
than for true ones, with the text and the image each held fixed in kind.

For each comparison: cosine(image, caption) under CLIP ViT-B/32 (the same
model whose embeddings VERITE ships), and the AUC of that score for telling
the true pairs from the defective ones, with a seeded bootstrap 95% interval.
AUC 0.5 means similarity carries no information about the defect.

    dgm4    orig  vs  text_swap (all)          vs  text_swap (same-publisher only)
    verite  true  vs  out-of-context           vs  miscaptioned

Only images that pass all three gates are used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import REPORTS  # noqa: E402
from scripts import images, recover  # noqa: E402

SEED = 20260901
MODEL = "openai/clip-vit-base-patch32"


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random true pair > score of a random defective pair)."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(pos, neg, n=1000) -> tuple[float, float]:
    rng = random.Random(SEED)
    vals = sorted(auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg])
                  for _ in range(n))
    return round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)


def _features(out):
    """transformers 4 returns the projected tensor; 5 wraps it as pooler_output."""
    return out if hasattr(out, "norm") else out.pooler_output


def similarities(pairs: list[tuple[str, str]], batch: int = 64) -> list[float]:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from scripts.ocr_audit import load_image

    model = CLIPModel.from_pretrained(MODEL).eval().to("cuda")
    proc = CLIPProcessor.from_pretrained(MODEL)
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        imgs = [load_image(p) for p, _ in chunk]
        with torch.no_grad():
            inputs = proc(text=[t for _, t in chunk], images=imgs, return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to("cuda")
            img = _features(model.get_image_features(pixel_values=inputs["pixel_values"]))
            txt = _features(model.get_text_features(input_ids=inputs["input_ids"],
                                                    attention_mask=inputs["attention_mask"]))
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            out += (img * txt).sum(-1).float().cpu().tolist()
        if (i // batch) % 50 == 0:
            print(f"  {i + len(chunk):,}/{len(pairs):,}", flush=True)
    return out


def dgm4_pairs(orig_sample: int) -> dict[str, list[tuple[str, str]]]:
    from scripts.dgm4_audit import load_rows, TEXT_SWAP

    index = images.load_index("dgm4")
    rows = load_rows()
    origin_text = defaultdict(set)
    for x in rows:
        if x["fake_cls"] == "orig":
            origin_text[x["text"]].add(x["publisher"])

    def path(x):
        parts = x["image"].split("/")
        rel = f"data/raw/dgm4/{parts[1]}/{parts[2]}.zip::{'/'.join(parts[2:])}"
        return rel if images.is_usable(index.get(rel)) else None

    orig = [(path(x), x["text"]) for x in rows if x["fake_cls"] == "orig" and path(x)]
    random.Random(SEED).shuffle(orig)
    swap_all = [(path(x), x["text"]) for x in rows if x["fake_cls"] == TEXT_SWAP and path(x)]
    swap_same = [(path(x), x["text"]) for x in rows if x["fake_cls"] == TEXT_SWAP and path(x)
                 and origin_text.get(x["text"]) == {x["publisher"]}]
    # Matched: the SAME image with its true caption and with a same-publisher
    # swapped caption. Image identity is held fixed, so only the caption's
    # relation to the image can move the score.
    orig_by_image = {}
    for x in rows:
        if x["fake_cls"] == "orig" and path(x):
            orig_by_image.setdefault(x["image"], (path(x), x["text"]))
    matched_true, matched_swap = [], []
    for x in rows:
        if (x["fake_cls"] == TEXT_SWAP and path(x) and x["image"] in orig_by_image
                and origin_text.get(x["text"]) == {x["publisher"]}):
            matched_true.append(orig_by_image[x["image"]])
            matched_swap.append((path(x), x["text"]))
    return {"orig": orig[:orig_sample], "text_swap": swap_all, "text_swap_same_publisher": swap_same,
            "orig_matched": matched_true, "text_swap_matched": matched_swap}


def verite_pairs() -> dict[str, list[tuple[str, str]]]:
    from scripts.hydrate import Item
    from scripts.records import load_verite

    index = images.load_index("verite")
    recorded = recover.ledger_paths("verite")
    matches = recover.load_matches("verite")
    out = defaultdict(list)
    for rec in load_verite():
        ref = rec.images[0]
        p = recover.resolve_image("verite", Item(ref.ref, ref.url or "", rec.label),
                                  index, recorded, matches)
        if p is not None:
            out[rec.label].append((images.rel(p), rec.text))
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orig-sample", type=int, default=8000)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"model": MODEL, "seed": SEED, "comparisons": {}}
    groups = {**{f"dgm4:{k}": v for k, v in dgm4_pairs(args.orig_sample).items()},
              **{f"verite:{k}": v for k, v in verite_pairs().items()}}
    sims = {k: similarities(v) for k, v in groups.items()}
    for true_key, bad_key in (("dgm4:orig", "dgm4:text_swap"),
                              ("dgm4:orig", "dgm4:text_swap_same_publisher"),
                              ("dgm4:orig_matched", "dgm4:text_swap_matched"),
                              ("verite:true", "verite:out-of-context"),
                              ("verite:true", "verite:miscaptioned")):
        pos, neg = sims[true_key], sims[bad_key]
        if true_key.endswith("_matched"):
            # paired by construction: also report how often the true caption wins
            wins = sum(p > n for p, n in zip(pos, neg)) / len(pos)
            report.setdefault("paired", {})[f"{true_key} vs {bad_key}"] = {
                "pairs": len(pos), "true_caption_scores_higher": round(wins, 4)}
        report["comparisons"][f"{true_key} vs {bad_key}"] = {
            "n_true": len(pos), "n_defective": len(neg),
            "mean_cosine_true": round(sum(pos) / len(pos), 4),
            "mean_cosine_defective": round(sum(neg) / len(neg), 4),
            "auc": round(auc(pos, neg), 4), "auc_95ci": bootstrap(pos, neg)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "clip_pairs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
