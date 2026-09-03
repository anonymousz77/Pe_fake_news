#!/usr/bin/env python
"""Generate train/val/test splits that a hostname cannot leak the label through.

    python scripts/splits.py --dataset factify2
    python scripts/splits.py --dataset factify2 --disjoint none      # baseline
    python scripts/splits.py --dataset factify2 --disjoint strict    # will refuse

Factify2's Refute class draws its images disproportionately from fact-checking
sites, four of which are 100%% Refute. A model can therefore score well by
recognising the source rather than the content. Splitting so that those domains
never appear on both sides of the train/test boundary removes that option: the
mapping "factly.in -> Refute" cannot be memorised from training data if no
factly.in record is in training data.

**Why not disjoint on every domain?** Because it is impossible here, and aimed
at the wrong thing. Records linked by shared domains form one component of
36,812 of 38,425 complete records (95.8%%), since pbs.twimg.com touches 84.7%%
of records. The only strict split available is 95.8/4.2, whose test set is
99.4%% Refute -- unusable. And pbs.twimg.com is near-uniform (26%%
Support_Multimodal), so it carries no label signal and blocking on it would be
cost without benefit. Constraining only domains whose class purity is at or
above the threshold drops the largest component to 3,367 (8.8%%) and targets
the actual leak.

`--disjoint strict` is kept so that result stays reproducible on demand, not
because anyone should use its output.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import PROJECT_ROOT, SPLITS  # noqa: E402
from scripts.confound import domain_of, majority_map  # noqa: E402
from scripts.fetchlib import FetchError  # noqa: E402
from scripts.hydrate import RESOLVERS, existing_destination  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_PURITY = 0.90
DEFAULT_SEED = 20260901

#: A class holding less than this fraction of its expected share in a split is
#: reported as unusable there.
THIN_CLASS_FRACTION = 0.25

#: A split whose largest constrained component exceeds this share of the corpus
#: cannot be partitioned at the requested ratios.
MODES = ("predictive", "strict", "none")


class SplitError(FetchError):
    """The requested split cannot be produced, or leaks."""


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


def complete_records(name: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    """(record -> its domains, record -> its label) for fully-present records.

    Records missing an image are excluded: a Factify2 row carries two images
    and is not usable multimodally with one of them absent.
    """
    resolver = RESOLVERS[name]
    items, _label_col, _extra = resolver() if name != "fakeddit" else resolver(None)

    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[item.record_id].append(item)

    domains: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for record, parts in grouped.items():
        if all(existing_destination(name, p) is not None for p in parts):
            domains[record] = {domain_of(p.url) for p in parts}
            labels[record] = parts[0].label
    if not domains:
        raise SplitError(
            f"{name}: no complete records on disk. Fetch and hydrate it first."
        )
    return domains, labels


def domain_purity(domains: dict[str, set[str]],
                  labels: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Per domain: how concentrated its records are in a single class."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for record, doms in domains.items():
        for d in doms:
            counts[d][labels[record]] += 1
    out = {}
    for d, c in counts.items():
        total = sum(c.values())
        top_label, top_n = min(c.most_common(), key=lambda kv: (-kv[1], kv[0]))
        out[d] = {
            "records": total,
            "purity": round(top_n / total, 4),
            "dominant_class": top_label,
        }
    return out


def constrained_domains(purity: dict[str, dict[str, Any]], mode: str,
                        threshold: float) -> set[str]:
    """Which domains may not straddle a split boundary."""
    if mode == "strict":
        return set(purity)
    if mode == "none":
        return set()
    return {d for d, info in purity.items() if info["purity"] >= threshold}


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------


def components(domains: dict[str, set[str]],
               constrained: set[str]) -> list[list[str]]:
    """Groups of records that must stay together, largest first.

    Two records sharing a constrained domain cannot be separated, and the
    relation is transitive, so this is a connected-components problem rather
    than a per-domain assignment.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    seen: dict[str, str] = {}
    for record, doms in domains.items():
        find(record)
        for d in doms & constrained:
            if d in seen:
                union(record, seen[d])
            else:
                seen[d] = record

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in domains:
        grouped[find(record)].append(record)
    return sorted((sorted(v) for v in grouped.values()), key=len, reverse=True)


#: Tie-break order when a component fits several splits equally well. The test
#: set is served first because a class absent from test cannot be evaluated at
#: all, whereas a class thin in train merely trains worse.
TIE_BREAK_PRIORITY = ("test", "val", "train")


def assign(groups: list[list[str]], labels: dict[str, str],
           ratios: Sequence[float], seed: int) -> dict[str, str]:
    """Assign whole components to splits, largest first, balancing CLASSES.

    Balancing on total size alone is not enough and fails badly here. Factify2's
    Refute records sit in just two components (3,365 and 1,604 of 4,980), so a
    size-greedy pass puts both in train and leaves the test set with 2 Refute
    records -- a split that cannot evaluate the class the whole exercise is
    about.

    So each component goes to the split whose remaining PER-CLASS capacity it
    overshoots least, with total room and then TIE_BREAK_PRIORITY settling
    ties, and the seed settling anything still tied.
    """
    total = sum(len(g) for g in groups)
    class_totals = Counter(labels.values())
    size_target = {n: r * total for n, r in zip(SPLIT_NAMES, ratios)}
    class_target = {
        n: {c: r * count for c, count in class_totals.items()}
        for n, r in zip(SPLIT_NAMES, ratios)
    }
    sizes = {n: 0 for n in SPLIT_NAMES}
    class_counts: dict[str, Counter] = {n: Counter() for n in SPLIT_NAMES}
    rng = random.Random(seed)

    placement: dict[str, str] = {}
    for group in groups:
        group_classes = Counter(labels[r] for r in group)

        def cost(split: str) -> tuple[float, float, int, float]:
            overshoot = sum(
                max(0.0, n - (class_target[split][c] - class_counts[split][c]))
                for c, n in group_classes.items()
            )
            room = size_target[split] - sizes[split]
            return (overshoot, -room, TIE_BREAK_PRIORITY.index(split), rng.random())

        chosen = min(SPLIT_NAMES, key=cost)
        for record in group:
            placement[record] = chosen
        sizes[chosen] += len(group)
        class_counts[chosen].update(group_classes)
    return placement


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def leakage(placement: dict[str, str], domains: dict[str, set[str]],
            purity: dict[str, dict[str, Any]],
            threshold: float) -> list[dict[str, Any]]:
    """Label-predictive domains appearing in more than one split."""
    where: dict[str, set[str]] = defaultdict(set)
    for record, split in placement.items():
        for d in domains[record]:
            where[d].add(split)

    return [
        {
            "domain": d,
            "splits": sorted(splits),
            "purity": purity[d]["purity"],
            "records": purity[d]["records"],
            "dominant_class": purity[d]["dominant_class"],
        }
        for d, splits in sorted(where.items())
        if len(splits) > 1 and purity[d]["purity"] >= threshold
    ]


def assert_no_leakage(violations: list[dict[str, Any]], mode: str,
                      threshold: float) -> None:
    """Fail loudly. This guard outlives whoever next edits the generator.

    Skipped only for ``--disjoint none``, where leakage is the measurement
    rather than a fault.
    """
    if not violations or mode == "none":
        return
    shown = violations[:10]
    lines = [
        f"    {v['domain']:<32} purity {v['purity']:.0%} "
        f"({v['dominant_class']}, {v['records']} records) in {v['splits']}"
        for v in shown
    ]
    if len(violations) > len(shown):
        lines.append(f"    ... and {len(violations) - len(shown)} more")
    raise SplitError(
        f"LEAKAGE: {len(violations)} domain(s) at or above purity {threshold} "
        f"appear in more than one split:\n" + "\n".join(lines) +
        "\n  A hostname seen in training that predicts the label would let a "
        "model score without reading the image. Refusing to write this split."
    )


# --------------------------------------------------------------------------
# the number that matters
# --------------------------------------------------------------------------


def domain_only_accuracy(placement: dict[str, str], domains: dict[str, set[str]],
                         labels: dict[str, str]) -> dict[str, float]:
    """What a hostname-only classifier scores on THIS test set.

    Fit the majority label per domain on train, score on test, unseen domains
    falling back to the train majority. The corpus-wide 38.2% answers a
    different question; this is the figure to print beside a model's result.
    """
    def pairs(split: str) -> list[tuple[str, str]]:
        return [
            (d, labels[r])
            for r, s in placement.items() if s == split
            for d in sorted(domains[r])
        ]

    train, test = pairs("train"), pairs("test")
    if not train or not test:
        return {"domain_only_test_accuracy": 0.0, "majority_baseline": 0.0,
                "uniform_baseline": 0.0, "test_pairs": len(test)}

    table, fallback = majority_map(train)
    correct = sum(1 for d, y in test if table.get(d, fallback) == y)
    test_labels = Counter(y for _, y in test)
    n_classes = len(set(labels.values()))
    return {
        "domain_only_test_accuracy": round(correct / len(test), 4),
        "majority_baseline": round(test_labels.most_common(1)[0][1] / len(test), 4),
        "uniform_baseline": round(1 / n_classes, 4),
        "unseen_domain_rate": round(
            sum(1 for d, _ in test if d not in table) / len(test), 4),
        "test_pairs": len(test),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build(name: str, mode: str = "predictive", threshold: float = DEFAULT_PURITY,
          ratios: Sequence[float] = DEFAULT_RATIOS,
          seed: int = DEFAULT_SEED) -> dict[str, Any]:
    domains, labels = complete_records(name)
    purity = domain_purity(domains, labels)
    constrained = constrained_domains(purity, mode, threshold)
    groups = components(domains, constrained)
    largest = len(groups[0]) if groups else 0
    total = len(domains)

    if mode == "strict" and largest > max(ratios) * total:
        rest = total - largest
        rest_classes = Counter(labels[r] for g in groups[1:] for r in g)
        dominant = rest_classes.most_common(1)[0] if rest_classes else ("n/a", 0)
        raise SplitError(
            f"{name}: strict source-disjoint splitting is IMPOSSIBLE.\n"
            f"  Largest inseparable component: {largest:,} of {total:,} records "
            f"({largest / total:.1%}).\n"
            f"  Everything else totals {rest:,} records ({rest / total:.1%}), of "
            f"which {dominant[1]:,} are {dominant[0]} "
            f"({dominant[1] / max(1, rest):.1%}).\n"
            f"  The only split available is {largest / total:.1%} / "
            f"{rest / total:.1%} with a near-single-class test set, which cannot "
            "be used for evaluation.\n"
            "  Use --disjoint predictive: it constrains only the domains that "
            "actually predict the label, which is what the rule is for."
        )

    placement = assign(groups, labels, ratios, seed)
    violations = leakage(placement, domains, purity, threshold)
    assert_no_leakage(violations, mode, threshold)

    per_split_classes = {s: Counter() for s in SPLIT_NAMES}
    for record, split in placement.items():
        per_split_classes[split][labels[record]] += 1

    # A class that is nearly absent from a split cannot be evaluated or tuned
    # there. Factify2 hits this unavoidably: 4,969 of 4,980 Refute records sit
    # in two components, so once train and test each take one, val has nothing
    # left. Surfaced rather than left for someone to discover in a results table.
    thin = []
    for split in SPLIT_NAMES:
        for cls, total_c in sorted(Counter(labels.values()).items()):
            got = per_split_classes[split][cls]
            share = got / total_c if total_c else 0.0
            expected = dict(zip(SPLIT_NAMES, ratios))[split]
            if expected > 0 and share < expected * THIN_CLASS_FRACTION:
                thin.append({
                    "split": split, "class": cls, "records": got,
                    "share_of_class": round(share, 4),
                    "expected_share": round(expected, 4),
                })

    size_hist = Counter(len(g) for g in groups)
    return {
        "thin_classes": thin,
        "dataset": name,
        "mode": mode,
        "purity_threshold": threshold,
        "seed": seed,
        "ratios": {n: r for n, r in zip(SPLIT_NAMES, ratios)},
        "records_total": total,
        "split_sizes": {s: sum(1 for v in placement.values() if v == s)
                        for s in SPLIT_NAMES},
        "per_class_by_split": {
            s: dict(sorted(per_split_classes[s].items())) for s in SPLIT_NAMES
        },
        "constrained_domains_count": len(constrained),
        "constrained_domains": [
            {"domain": d, **purity[d]}
            for d in sorted(constrained, key=lambda d: -purity[d]["records"])
        ],
        "component_sizes": {
            "count": len(groups),
            "largest": largest,
            "largest_share": round(largest / total, 4) if total else 0.0,
            "top_10": [len(g) for g in groups[:10]],
            "histogram": {str(k): v for k, v in sorted(size_hist.items(),
                                                       reverse=True)[:15]},
        },
        "leakage": {
            "violations": len(violations),
            "enforced": mode != "none",
            "detail": violations[:50],
        },
        "domain_only": domain_only_accuracy(placement, domains, labels),
        "_placement": placement,
        "_labels": labels,
        "_domains": domains,
    }


def write_outputs(result: dict[str, Any]) -> tuple[Path, list[Path]]:
    name, mode = result["dataset"], result["mode"]
    placement = result["_placement"]
    SPLITS.mkdir(parents=True, exist_ok=True)

    suffix = "" if mode == "predictive" else f"_{mode}"
    written = []
    for split in SPLIT_NAMES:
        path = SPLITS / f"{name}_{split}{suffix}.csv"
        rows = sorted(r for r, s in placement.items() if s == split)
        # record_id and split assignment ONLY. Labels are the annotation that
        # these datasets' licences exist to protect, and per-record domains are
        # derived from URLs inside the gated CSVs. Both stay out; the aggregate
        # split report carries the constrained-domain list that makes the
        # source-disjointness claim checkable. Anyone with legitimate access
        # joins this file on record_id.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "split"])
            for record in rows:
                writer.writerow([record, split])
        written.append(path)

    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    report = SPLITS / f"{name}_split_report{suffix}.json"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report, written


def render(result: dict[str, Any]) -> str:
    comp, dom = result["component_sizes"], result["domain_only"]
    lines = [
        f"  mode              : {result['mode']}  "
        f"(purity threshold {result['purity_threshold']}, seed {result['seed']})",
        f"  complete records  : {result['records_total']:,}",
        f"  constrained domains: {result['constrained_domains_count']:,}",
        f"  components        : {comp['count']:,}, largest {comp['largest']:,} "
        f"({comp['largest_share']:.1%})",
        "",
        "  split sizes",
    ]
    for split in SPLIT_NAMES:
        n = result["split_sizes"][split]
        share = n / result["records_total"] if result["records_total"] else 0
        classes = result["per_class_by_split"][split]
        lines.append(f"    {split:<6} {n:>7,}  {share:>6.1%}   {classes}")
    if result["thin_classes"]:
        lines += ["", "  UNUSABLE CLASS/SPLIT COMBINATIONS"]
        for t in result["thin_classes"]:
            lines.append(
                f"    {t['split']:<6} has {t['records']:>5,} {t['class']} "
                f"({t['share_of_class']:.1%} of the class, expected "
                f"{t['expected_share']:.0%}) -- cannot be evaluated or tuned there")
    lines += [
        "",
        f"  leakage           : {result['leakage']['violations']} violation(s)"
        + ("  (enforced)" if result["leakage"]["enforced"] else "  (baseline: not enforced)"),
        "",
        "  hostname-only classifier ON THIS TEST SET",
        f"    domain-only accuracy : {dom['domain_only_test_accuracy']:.1%}"
        "   <-- print this beside every model result",
        f"    majority baseline    : {dom['majority_baseline']:.1%}",
        f"    uniform baseline     : {dom['uniform_baseline']:.1%}",
        f"    unseen-domain rate   : {dom.get('unseen_domain_rate', 0):.1%}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="factify2", choices=sorted(RESOLVERS))
    parser.add_argument("--disjoint", default="predictive", choices=MODES,
                        dest="mode",
                        help="which domains may not straddle a split boundary "
                             "(default: predictive)")
    parser.add_argument("--purity", type=float, default=DEFAULT_PURITY,
                        help=f"a domain at or above this class purity is "
                             f"label-predictive (default {DEFAULT_PURITY})")
    parser.add_argument("--ratios", default="70,15,15",
                        help="train,val,test percentages (default 70,15,15)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="report without writing split files")
    args = parser.parse_args(argv)

    parts = [float(x) for x in args.ratios.split(",")]
    if len(parts) != 3 or abs(sum(parts) - 100) > 1e-6:
        parser.error("--ratios must be three numbers summing to 100")
    ratios = tuple(p / 100 for p in parts)

    try:
        result = build(args.dataset, mode=args.mode, threshold=args.purity,
                       ratios=ratios, seed=args.seed)
    except SplitError as exc:
        print(f"\n[{args.dataset}] REFUSED\n{exc}\n", file=sys.stderr)
        return 1

    print(f"\n[{args.dataset}] source-disjoint split")
    print(render(result))
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    report, written = write_outputs(result)
    print(f"\n  report -> {report.relative_to(PROJECT_ROOT)}")
    for path in written:
        print(f"  split  -> {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
