#!/usr/bin/env python
"""Shortcut audit of DGM4's text-swap class before it may supervise media_mismatch.

    python scripts/dgm4_audit.py          # -> data/reports/dgm4_audit.json

The same questions asked of Fakeddit's False Connection class, so the two
answers are comparable:

1. publisher distribution per class -- is any publisher label-pure?
2. can a text-only classifier recover the label (vs the majority baseline)?
3. can a text-only classifier recover the PUBLISHER, and does that track the
   label?
4. is a swapped caption drawn from the image's own publisher or another one?
   (Found by exact match of the swapped caption against DGM4's own pristine
   captions.) If swaps cross publishers, "the caption's style does not match
   the image's publisher" detects the swap without looking at the image.
5. are the face-edit-plus-text-swap rows separable from pure text-swap rows?
6. integrity: train.json's byte-identical repeats are dropped before any
   count, and nothing (row, id, image, swapped caption) straddles train/val.

Then the decision rule of 2026-10-02, applied by this script, not by hand:
text_swap maps to media_mismatch iff (a) no publisher is label-pure, (b) the
text-only classifier is near the majority baseline, and (c) swaps are
within-publisher. (c) fails for the class as shipped -- half the swaps cross
publishers -- so the rule is evaluated on the within-publisher subset, the only
rows that could be mapped, with (b) measured on folds that keep every caption
on one side (random folds let a reused caption be memorised, which is a
property of the CV, not of the text).

Over the 179,295 distinct train+val records (the test split is forbidden).
Text-only Naive Bayes, unigrams+bigrams, 5-fold, seed 20260901 -- a LOWER
bound on what the text gives away. Pixels are not examined here; the CLIP
pair test (scripts/clip_pairs.py) is the image-side check.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import REPORTS, raw_dir  # noqa: E402
from scripts.fakeddit_classes import NaiveBayes, SEED, cross_val, scores, tokens  # noqa: E402

TEXT_SWAP = "text_swap"
COMBINED_SWAP = ("face_swap&text_swap", "face_attribute&text_swap")
PURITY_LIMIT = 0.90          # the Factify2 label-predictive threshold
NEAR_BASELINE_PP = 1.0       # text-only accuracy within 1 point of majority


def grouped_cross_val(docs, labels, groups) -> list[str]:
    """5-fold CV where every row of a group (here: a caption) shares a fold."""
    from scripts.fakeddit_classes import FOLDS

    keys = sorted(set(groups))
    random.Random(SEED).shuffle(keys)
    fold_of = {g: k % FOLDS for k, g in enumerate(keys)}
    f = [fold_of[g] for g in groups]
    pred = [None] * len(docs)
    for k in range(FOLDS):
        model = NaiveBayes().fit([docs[i] for i in range(len(docs)) if f[i] != k],
                                 [labels[i] for i in range(len(docs)) if f[i] != k])
        for i in range(len(docs)):
            if f[i] == k:
                pred[i] = model.predict(docs[i])
    return pred


def integrity() -> dict[str, Any]:
    """Repeats inside each shipped file, and anything shared across train/val."""
    raw = {s: json.loads((raw_dir("dgm4") / "metadata" / f"{s}.json").read_text(encoding="utf-8"))
           for s in ("train", "val")}

    def key(x):
        return json.dumps(x, sort_keys=True)

    out: dict[str, Any] = {}
    for s, rows in raw.items():
        c = Counter(key(x) for x in rows)
        out[s] = {"rows_in_file": len(rows), "distinct": len(c),
                  "byte_identical_repeats": sum(v - 1 for v in c.values()),
                  "rows_in_repeated_groups_by_class": dict(Counter(
                      x["fake_cls"] for x in rows if c[key(x)] > 1))}
    t, v = raw["train"], raw["val"]
    shared_text = {x["text"] for x in t} & {x["text"] for x in v}
    out["straddle_train_val"] = {
        "byte_identical_rows": len({key(x) for x in t} & {key(x) for x in v}),
        "source_ids": len({x["id"] for x in t} & {x["id"] for x in v}),
        "image_paths": len({x["image"] for x in t} & {x["image"] for x in v}),
        "caption_texts": len(shared_text),
        "rows_carrying_shared_captions_by_class": dict(Counter(
            x["fake_cls"] for x in t + v if x["text"] in shared_text))}
    out["deduplicated_before_every_count"] = True
    return out


def load_rows() -> list[dict[str, Any]]:
    rows, seen = [], set()
    for split in ("train", "val"):
        for x in json.loads((raw_dir("dgm4") / "metadata" / f"{split}.json").read_text(encoding="utf-8")):
            key = json.dumps(x, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            parts = x["image"].split("/")
            x["folder"] = f"{parts[1]}/{parts[2]}"
            x["split"] = split
            rows.append(x)
    # Publisher: the origin folder, or -- for face-edited images, whose folder
    # names the generator -- the origin folder of the same source id.
    pub_of_id: dict[Any, str] = {}
    for x in rows:
        if x["folder"].startswith("origin/"):
            pub_of_id.setdefault(x["id"], x["folder"].split("/")[1])
    for x in rows:
        x["publisher"] = (x["folder"].split("/")[1] if x["folder"].startswith("origin/")
                          else pub_of_id.get(x["id"]))
    return rows


def nmi(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    joint = Counter(pairs)
    a, b = Counter(p for p, _ in pairs), Counter(q for _, q in pairs)
    mi = sum(c / n * math.log((c / n) / ((a[p] / n) * (b[q] / n))) for (p, q), c in joint.items())
    ha = -sum(c / n * math.log(c / n) for c in a.values())
    hb = -sum(c / n * math.log(c / n) for c in b.values())
    return round(mi / math.sqrt(ha * hb), 4) if ha and hb else 0.0


def main() -> int:
    rows = load_rows()
    rng = random.Random(SEED)
    out: dict[str, Any] = {"records": len(rows), "seed": SEED}

    # 1. publisher distribution per class, and purity per publisher
    per_class = defaultdict(Counter)
    per_pub = defaultdict(Counter)
    for x in rows:
        per_class[x["fake_cls"]][str(x["publisher"])] += 1
        per_pub[str(x["publisher"])][x["fake_cls"]] += 1
    out["publisher_by_class"] = {c: dict(v.most_common()) for c, v in sorted(per_class.items())}
    out["class_by_publisher"] = {
        p: {"records": sum(v.values()),
            "purity": round(max(v.values()) / sum(v.values()), 4),
            "dominant_class": v.most_common(1)[0][0],
            "text_swap_share": round(v[TEXT_SWAP] / sum(v.values()), 4)}
        for p, v in sorted(per_pub.items())}
    out["image_folder_by_class"] = {c: dict(Counter(x["folder"] for x in rows if x["fake_cls"] == c))
                                    for c in sorted(per_class)}

    # 2. text-only label recovery: all nine classes, and text_swap vs orig
    docs = [tokens(x.get("text")) for x in rows]
    labels = [x["fake_cls"] for x in rows]
    out["text_only_9way"] = scores(labels, cross_val(docs, labels))
    pair_idx = [i for i, x in enumerate(rows) if x["fake_cls"] in (TEXT_SWAP, "orig")]
    pl = [labels[i] for i in pair_idx]
    out["text_only_text_swap_vs_orig"] = scores(pl, cross_val([docs[i] for i in pair_idx], pl))

    # 3. text-only publisher recovery, and whether it tracks the label
    pub_idx = [i for i, x in enumerate(rows) if x["publisher"]]
    pubs = [rows[i]["publisher"] for i in pub_idx]
    pub_pred = cross_val([docs[i] for i in pub_idx], pubs)
    out["text_only_publisher"] = {k: v for k, v in scores(pubs, pub_pred).items() if k != "per_class"}
    out["text_only_publisher"]["per_class"] = scores(pubs, pub_pred)["per_class"]
    out["nmi_publisher_label"] = nmi([(rows[i]["publisher"], labels[i]) for i in pub_idx])
    # the style-mismatch detector: "caption reads like another publisher"
    mismatch = defaultdict(Counter)
    for k, i in enumerate(pub_idx):
        if labels[i] in (TEXT_SWAP, "orig"):
            mismatch[labels[i]]["style_mismatch" if pub_pred[k] != pubs[k] else "style_match"] += 1
    out["style_mismatch_rule"] = {
        c: {**dict(v), "mismatch_rate": round(v["style_mismatch"] / max(1, sum(v.values())), 4)}
        for c, v in mismatch.items()}

    # 4. where did each swapped caption come from?
    origin_text: dict[str, set[str]] = defaultdict(set)
    for x in rows:
        if x["fake_cls"] == "orig":
            origin_text[x["text"]].add(x["publisher"])
    same = cross = unknown = ambiguous = 0
    for x in rows:
        if x["fake_cls"] != TEXT_SWAP:
            continue
        src = origin_text.get(x["text"])
        if not src:
            unknown += 1
        elif len(src) > 1:
            ambiguous += 1
        elif x["publisher"] in src:
            same += 1
        else:
            cross += 1
    found = same + cross
    out["swap_source"] = {
        "text_swap_rows": same + cross + unknown + ambiguous,
        "caption_found_among_dgm4_pristine_captions": found + ambiguous,
        "same_publisher": same, "cross_publisher": cross, "ambiguous_publisher": ambiguous,
        "source_not_in_dgm4": unknown,
        "cross_publisher_share_of_found": round(cross / found, 4) if found else None}

    # 5. face-edit + text-swap vs pure text-swap
    comb = [i for i, x in enumerate(rows) if x["fake_cls"] in (TEXT_SWAP,) + COMBINED_SWAP]
    cl = ["combined" if labels[i] != TEXT_SWAP else TEXT_SWAP for i in comb]
    out["combined_vs_text_swap"] = {
        "by_metadata": {
            "text_swap_with_fake_image_box": sum(1 for x in rows if x["fake_cls"] == TEXT_SWAP
                                                 and x["fake_image_box"]),
            "combined_without_fake_image_box": sum(1 for x in rows if x["fake_cls"] in COMBINED_SWAP
                                                   and not x["fake_image_box"]),
            "text_swap_image_folders": dict(Counter(rows[i]["folder"].split("/")[0] for i in comb
                                                    if labels[i] == TEXT_SWAP)),
            "combined_image_folders": dict(Counter(rows[i]["folder"].split("/")[0] for i in comb
                                                   if labels[i] != TEXT_SWAP))},
        "text_only": scores(cl, cross_val([docs[i] for i in comb], cl))}
    out["sample_text_swap_captions"] = [x["text"] for x in rng.sample(
        [x for x in rows if x["fake_cls"] == TEXT_SWAP], 8)]

    # caption reuse: one swapped caption can be pasted onto many images
    swap_rows = [x for x in rows if x["fake_cls"] == TEXT_SWAP]
    reuse = Counter(x["text"] for x in swap_rows)
    out["swap_caption_reuse"] = {
        "distinct_captions": len(reuse), "max_reuse": max(reuse.values()),
        "rows_from_captions_used_5_plus_times": sum(n for n in reuse.values() if n >= 5)}
    by_split = defaultdict(set)
    for x in swap_rows:
        by_split[x["split"]].add(x["text"])
    out["caption_split_straddle"] = {
        "swapped_captions_in_both_train_and_val": len(by_split["train"] & by_split["val"]),
        "of": len(reuse)}

    # 6. integrity
    out["integrity"] = integrity()

    # the within-publisher subset: the only rows the rule could map
    def same_pub(x):
        return x["fake_cls"] == TEXT_SWAP and origin_text.get(x["text"]) == {x["publisher"]}

    sub = [i for i, x in enumerate(rows) if x["fake_cls"] == "orig" or same_pub(x)]
    sl = [labels[i] for i in sub]
    sd = [docs[i] for i in sub]
    out["caption_grouped_cv"] = {
        "all swaps": scores(pl, grouped_cross_val([docs[i] for i in pair_idx], pl,
                                                  [rows[i]["text"] or "" for i in pair_idx])),
        "same-publisher swaps": scores(sl, grouped_cross_val(sd, sl, [rows[i]["text"] or "" for i in sub]))}
    sub_pub = defaultdict(Counter)
    for i in sub:
        sub_pub[rows[i]["publisher"]][labels[i]] += 1
    style = defaultdict(Counter)
    pos = {i: k for k, i in enumerate(pub_idx)}
    for i in sub:
        if i in pos:
            k = pos[i]
            style[labels[i]]["style_mismatch" if pub_pred[k] != pubs[k] else "style_match"] += 1
    swaps_in_sub = [i for i in sub if labels[i] == TEXT_SWAP]
    # publisher-only classifier (the analogue of Factify2's hostname-only one):
    # predict each publisher's majority class; purity alone is base-rate bound
    pub_major = {p: c.most_common(1)[0][0] for p, c in sub_pub.items()}
    pub_only_pred = [pub_major.get(rows[i]["publisher"], "orig") for i in sub]
    swap_rate = {p: round(c[TEXT_SWAP] / sum(c.values()), 4) for p, c in sorted(sub_pub.items())}
    out["same_publisher_subset"] = {
        "publisher_only_classifier": {k: v for k, v in scores(sl, pub_only_pred).items()},
        "text_swap_rate_by_publisher": swap_rate,
        "text_swap_rate_overall": round(len(swaps_in_sub) / len(sub), 4),
        "text_swap_rows": len(swaps_in_sub),
        "by_split": dict(Counter(rows[i]["split"] for i in swaps_in_sub)),
        "by_publisher": dict(Counter(rows[i]["publisher"] for i in swaps_in_sub)),
        "distinct_captions": len({rows[i]["text"] for i in swaps_in_sub}),
        "text_only_random_folds": scores(sl, cross_val(sd, sl)),
        "purity_by_publisher": {p: round(max(c.values()) / sum(c.values()), 4)
                                for p, c in sorted(sub_pub.items())},
        "nmi_publisher_label": nmi([(rows[i]["publisher"], labels[i]) for i in sub
                                    if rows[i]["publisher"]]),
        "style_mismatch_rule": {c: {**dict(v), "mismatch_rate": round(
            v["style_mismatch"] / max(1, sum(v.values())), 4)} for c, v in style.items()}}

    # the decision, mechanically
    def gap(sc):
        return round(100 * (sc["accuracy"] - sc["majority_baseline"]), 2)

    sp = out["same_publisher_subset"]
    max_purity = max(v["purity"] for v in out["class_by_publisher"].values())
    whole = {"a_no_label_pure_publisher": max_purity < PURITY_LIMIT,
             "b_text_only_near_baseline":
                 abs(gap(out["text_only_text_swap_vs_orig"])) <= NEAR_BASELINE_PP,
             "c_swaps_within_publisher": out["swap_source"]["cross_publisher"] == 0}
    subset = {"a_no_label_pure_publisher": max(sp["purity_by_publisher"].values()) < PURITY_LIMIT,
              "b_text_only_near_baseline":
                  abs(gap(out["caption_grouped_cv"]["same-publisher swaps"])) <= NEAR_BASELINE_PP,
              "c_swaps_within_publisher": True}
    out["decision"] = {
        "rule": "map text_swap -> media_mismatch iff (a) no publisher label-pure (purity < 0.90), "
                "(b) text-only within 1 point of majority, (c) swaps within-publisher",
        "whole_class": {**whole, "max_publisher_purity": max_purity,
                        "text_only_gap_pp_random_folds": gap(out["text_only_text_swap_vs_orig"]),
                        "text_only_gap_pp_caption_grouped": gap(out["caption_grouped_cv"]["all swaps"]),
                        "passes": all(whole.values())},
        "same_publisher_subset": {**subset,
                                  "max_publisher_purity": max(sp["purity_by_publisher"].values()),
                                  "text_only_gap_pp_caption_grouped":
                                      gap(out["caption_grouped_cv"]["same-publisher swaps"]),
                                  "text_only_gap_pp_random_folds": gap(sp["text_only_random_folds"]),
                                  "passes": all(subset.values())},
        "applied": ("map the within-publisher pure text swaps only, as weak supervision, training only"
                    if all(subset.values()) else "leave DGM4 unmapped"),
        "note": "(a) on the subset is an absolute purity >= 0.90 over orig + same-publisher swaps, "
                "where orig is 88.6% of rows, so a publisher with no information already sits near "
                "0.886; the publisher-only classifier in same_publisher_subset is the base-rate-"
                "corrected view. The rule was fixed before this run and is applied as written."}

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dgm4_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
