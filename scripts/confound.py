#!/usr/bin/env python
"""Quantify how much a dataset's LABEL is predictable from image PROVENANCE.

    python scripts/confound.py --dataset factify2

Factify2's Refute class draws ~34% of its images from snopes.com. If the
domain an image came from predicts its label, then a model can score well by
recognising the source rather than by looking at the content -- and it will
collapse on any evaluation where that correlation does not hold.

This is the same defect as ISOT's Reuters shortcut, in the image modality.
The point of this script is to put a number on it *before* training, so the
number can go in the paper rather than into a reviewer's objection.

Three measures are reported:

* **Normalised mutual information** between domain and label. 0 means domain
  carries no information about the label; 1 means it fully determines it.
* **Domain-only classifier accuracy, resubstitution.** Predict each domain's
  majority label, scored on the same rows. An upper bound, badly inflated by
  domains seen once, which are trivially "predicted" perfectly.
* **Domain-only classifier accuracy, held out.** The same rule learned on a
  training fold and scored on unseen rows, with unseen domains falling back to
  the global majority. This is the honest number, and the one to quote.

Compare both against the majority-class baseline, printed alongside.

Only records whose media actually landed are counted, since those are the rows
a model would train on.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import PROJECT_ROOT, REPORTS  # noqa: E402
from scripts.fetchlib import ORIGIN, WAYBACK, FetchError  # noqa: E402
from scripts.hydrate import (  # noqa: E402
    RESOLVERS,
    destination,
    existing_destination,
)

DEFAULT_FOLDS = 5
DEFAULT_SEED = 20260901


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "unknown").lower()
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------
# information theory, implemented directly
# --------------------------------------------------------------------------
# sklearn and scipy are not installed in this environment, and adding a
# dependency for two textbook formulas is not worth the supply-chain surface.


def entropy(counts: Sequence[int]) -> float:
    """Shannon entropy in nats."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            acc -= p * math.log(p)
    return acc


def mutual_information(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """I(X;Y) plus the entropies and both common normalisations."""
    n = len(pairs)
    if n == 0:
        return {"mutual_information": 0.0, "h_domain": 0.0, "h_label": 0.0,
                "nmi_sqrt": 0.0, "nmi_arithmetic": 0.0}

    joint = Counter(pairs)
    xs = Counter(x for x, _ in pairs)
    ys = Counter(y for _, y in pairs)

    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        px = xs[x] / n
        py = ys[y] / n
        mi += pxy * math.log(pxy / (px * py))
    mi = max(0.0, mi)  # guard against -0.0 from floating point

    hx = entropy(list(xs.values()))
    hy = entropy(list(ys.values()))
    denom_sqrt = math.sqrt(hx * hy)
    denom_mean = (hx + hy) / 2
    return {
        "mutual_information": mi,
        "h_domain": hx,
        "h_label": hy,
        "nmi_sqrt": (mi / denom_sqrt) if denom_sqrt > 0 else 0.0,
        "nmi_arithmetic": (mi / denom_mean) if denom_mean > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# the domain-only classifier
# --------------------------------------------------------------------------


def majority_map(pairs: Sequence[tuple[str, str]]) -> tuple[dict[str, str], str]:
    """Per-domain majority label, plus the global majority fallback."""
    by_domain: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for domain, label in pairs:
        by_domain[domain][label] += 1
        overall[label] += 1
    # ties broken by label name so the result is deterministic
    table = {d: min(c.most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
             for d, c in by_domain.items()}
    fallback = min(overall.most_common(), key=lambda kv: (-kv[1], kv[0]))[0]
    return table, fallback


def resubstitution_accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    """Fit and score on the same rows: an inflated upper bound."""
    table, fallback = majority_map(pairs)
    correct = sum(1 for d, y in pairs if table.get(d, fallback) == y)
    return correct / len(pairs) if pairs else 0.0


def heldout_accuracy(pairs: Sequence[tuple[str, str]], folds: int = DEFAULT_FOLDS,
                     seed: int = DEFAULT_SEED) -> float:
    """K-fold: learn the mapping on train, score on unseen rows."""
    if len(pairs) < folds:
        return 0.0
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    correct = 0
    for k in range(folds):
        test_idx = {i for j, i in enumerate(indices) if j % folds == k}
        train = [pairs[i] for i in indices if i not in test_idx]
        test = [pairs[i] for i in sorted(test_idx)]
        table, fallback = majority_map(train)
        correct += sum(1 for d, y in test if table.get(d, fallback) == y)
    return correct / len(pairs)


def baseline_accuracy(pairs: Sequence[tuple[str, str]]) -> float:
    """Always predict the most common label."""
    if not pairs:
        return 0.0
    counts = Counter(y for _, y in pairs)
    return counts.most_common(1)[0][1] / len(pairs)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _present(name: str, item, source: str) -> bool:
    """Is this item's media on disk, restricted to the requested source?"""
    if source == "all":
        return existing_destination(name, item) is not None
    want = ORIGIN if source == "origin" else WAYBACK
    path = destination(name, item, want)
    return path.exists() and path.stat().st_size > 0


def complete_cases(name: str, items, source: str,
                   exclude: set[str]) -> dict[str, Any]:
    """Per class: how many ROWS survive a complete-cases-only design.

    A Factify2 row carries two images. Dropping any row with a missing image
    is the alternative to recovery, and its cost is not the missing-image count
    -- one absent image discards the row's text, its other image and its label
    too.
    """
    by_record: dict[str, list] = defaultdict(list)
    for item in items:
        by_record[item.record_id].append(item)

    total = Counter()
    complete = Counter()
    for parts in by_record.values():
        label = parts[0].label
        total[label] += 1
        if all(_present(name, it, source) and domain_of(it.url) not in exclude
               for it in parts):
            complete[label] += 1

    rows = {}
    for label in sorted(total):
        kept, all_rows = complete[label], total[label]
        rows[label] = {
            "records_total": all_rows,
            "records_complete": kept,
            "records_lost": all_rows - kept,
            "retained": round(kept / all_rows, 4) if all_rows else 0.0,
        }
    return {
        "images_per_record": round(len(items) / max(1, len(by_record)), 2),
        "records_total": sum(total.values()),
        "records_complete": sum(complete.values()),
        "records_lost": sum(total.values()) - sum(complete.values()),
        "retained_overall": round(sum(complete.values()) / max(1, sum(total.values())), 4),
        "per_class": rows,
    }


def analyse(name: str, top_n: int = 20, folds: int = DEFAULT_FOLDS,
            seed: int = DEFAULT_SEED, source: str = "all",
            exclude_domains: Sequence[str] = ()) -> dict[str, Any]:
    resolver = RESOLVERS[name]
    items, label_col, _extra = resolver() if name != "fakeddit" else resolver(None)
    exclude = {d.lower().removeprefix("www.") for d in exclude_domains}

    recovered = [
        it for it in items
        if _present(name, it, source) and domain_of(it.url) not in exclude
    ]
    if not recovered:
        raise FetchError(
            f"{name}: no media on disk for source={source!r}, so there is "
            "nothing to analyse."
        )

    cases = complete_cases(name, items, source, exclude)
    pairs = [(domain_of(it.url), it.label) for it in recovered]
    info = mutual_information(pairs)

    by_domain: dict[str, Counter] = defaultdict(Counter)
    for domain, label in pairs:
        by_domain[domain][label] += 1

    freq = Counter(d for d, _ in pairs)
    top = []
    for domain, count in freq.most_common(top_n):
        dist = by_domain[domain]
        dominant, dominant_n = min(dist.most_common(), key=lambda kv: (-kv[1], kv[0]))
        top.append({
            "domain": domain,
            "images": count,
            "share_of_corpus": round(count / len(pairs), 4),
            "class_distribution": dict(sorted(dist.items())),
            "dominant_class": dominant,
            "dominant_share": round(dominant_n / count, 4),
        })

    singletons = sum(1 for d, c in freq.items() if c == 1)
    label_totals = Counter(y for _, y in pairs)

    # Per class: how concentrated is its provenance?
    per_class_top_domain = {}
    by_label: dict[str, Counter] = defaultdict(Counter)
    for domain, label in pairs:
        by_label[label][domain] += 1
    for label, counter in by_label.items():
        domain, count = counter.most_common(1)[0]
        per_class_top_domain[label] = {
            "domain": domain,
            "images": count,
            "share_of_class": round(count / label_totals[label], 4),
        }

    resub = resubstitution_accuracy(pairs)
    held = heldout_accuracy(pairs, folds=folds, seed=seed)
    base = baseline_accuracy(pairs)

    n_classes = len(label_totals)
    return {
        "dataset": name,
        "label_field": label_col,
        "source_filter": source,
        "excluded_domains": sorted(exclude),
        "complete_cases": cases,
        "records_with_recovered_media": len(recovered),
        "records_total": len(items),
        "distinct_domains": len(freq),
        "single_image_domains": singletons,
        "class_totals": dict(sorted(label_totals.items())),
        "mutual_information_nats": round(info["mutual_information"], 4),
        "entropy_domain_nats": round(info["h_domain"], 4),
        "entropy_label_nats": round(info["h_label"], 4),
        "normalised_mutual_information": {
            "sqrt": round(info["nmi_sqrt"], 4),
            "arithmetic": round(info["nmi_arithmetic"], 4),
            "definition": "NMI = I(D;Y) / sqrt(H(D)*H(Y)); arithmetic uses (H(D)+H(Y))/2",
        },
        "domain_only_classifier": {
            "majority_class_baseline": round(base, 4),
            "uniform_class_baseline": round(1 / n_classes, 4),
            "resubstitution_accuracy": round(resub, 4),
            "heldout_accuracy": round(held, 4),
            "folds": folds,
            "seed": seed,
            "lift_over_baseline": round(held - base, 4),
            "note": (
                "resubstitution is inflated by "
                f"{singletons} single-image domains; heldout is the honest figure"
            ),
        },
        "top_domains": top,
        "per_class_most_common_domain": dict(sorted(per_class_top_domain.items())),
    }


def render(result: dict[str, Any]) -> str:
    nmi = result["normalised_mutual_information"]["sqrt"]
    clf = result["domain_only_classifier"]
    cc = result["complete_cases"]
    lines = [
        f"  source filter      : {result['source_filter']}"
        + (f"   excluding {', '.join(result['excluded_domains'])}"
           if result["excluded_domains"] else ""),
        f"  images with media  : {result['records_with_recovered_media']:,} "
        f"of {result['records_total']:,}",
        f"  complete records   : {cc['records_complete']:,} of "
        f"{cc['records_total']:,}  ({cc['retained_overall']:.1%} retained, "
        f"{cc['records_lost']:,} lost)",
        f"  distinct domains   : {result['distinct_domains']:,} "
        f"({result['single_image_domains']:,} appear once)",
        "",
        f"  NMI(domain; label) : {nmi:.4f}   "
        f"(I={result['mutual_information_nats']:.4f} nats, "
        f"H(label)={result['entropy_label_nats']:.4f})",
        "",
        "  domain-only classifier",
        f"    uniform baseline  : {clf['uniform_class_baseline']:.1%}"
        f"   (equal classes)",
        f"    majority baseline : {clf['majority_class_baseline']:.1%}"
        f"   (as recovered)",
        f"    held out ({clf['folds']}-fold) : {clf['heldout_accuracy']:.1%}"
        f"   <-- the honest number",
        f"    resubstitution    : {clf['resubstitution_accuracy']:.1%}"
        f"   (inflated by singleton domains)",
        f"    lift over baseline: {clf['lift_over_baseline']:+.1%}",
        "",
        "  most common domain per class",
    ]
    for label, info in result["per_class_most_common_domain"].items():
        lines.append(f"    {label:<26} {info['domain']:<24} "
                     f"{info['images']:>6,}  {info['share_of_class']:.1%} of class")
    lines += ["", f"  top {min(10, len(result['top_domains']))} domains by image count"]
    for row in result["top_domains"][:10]:
        lines.append(f"    {row['domain']:<28} {row['images']:>6,}  "
                     f"{row['dominant_class']} {row['dominant_share']:.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="factify2", choices=sorted(RESOLVERS),
                        help="dataset to analyse (default factify2)")
    parser.add_argument("--top", type=int, default=20,
                        help="how many domains to tabulate (default 20)")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", default="all", choices=("all", "origin", "wayback"),
                        help="restrict to media from this provenance (default all)")
    parser.add_argument("--exclude-domain", action="append", default=[],
                        dest="exclude_domains", metavar="DOMAIN",
                        help="drop every image from this domain (repeatable)")
    parser.add_argument("--label", default=None,
                        help="suffix for the output filename, to keep variants apart")
    args = parser.parse_args(argv)

    try:
        result = analyse(args.dataset, top_n=args.top, folds=args.folds,
                         seed=args.seed, source=args.source,
                         exclude_domains=args.exclude_domains)
    except FetchError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"\n[{args.dataset}] domain / label confound")
    print(render(result))

    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    out = REPORTS / f"{args.dataset}_domain_label_confound{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  report -> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
