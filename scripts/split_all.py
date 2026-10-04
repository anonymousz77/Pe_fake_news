#!/usr/bin/env python
"""One split for every record of every dataset, with duplicates kept together.

Two rules on top of each dataset's own split policy, enforced rather than
checked:

1. **Every duplicate group with five or more copies lands in one split.** An
   image carried by five records is five chances to train on the pixels a
   test record is scored on. "Copies" counts every usable file with that
   sha256, in every corpus.
2. **Every cross-corpus duplicate lands in one split**, whatever its copy
   count. Training on one corpus and evaluating on another must not score a
   model on an image it has seen.

The per-dataset policy underneath:

* factify2 -- generated source-disjoint by scripts/splits.py over ALL 42,500
  records (URL domains are known whether or not the image was recovered, so
  the split no longer moves when recovery does); duplicate groups are extra
  links in the same union-find; records sharing an image with a pinned corpus
  are pinned.
* verite -- every record is test: it is an evaluation benchmark.
* datasets with an official split -- that split.
* datasets without one (welfake, isot, fakenewsnet) -- seeded, class-balanced
  70/15/15 over the same components.

Then reconciliation: a constraint group whose members disagree moves as a
whole -- to factify2's placement if it contains a factify2 record (factify2's
domain constraint is the binding one), otherwise to the most held-out split
among its members (test > val > train), so no record an author held out is
ever moved into training. Every move is counted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from scripts import splits as base

SPLIT_ORDER = ("test", "val", "train")          # most held-out first
DEFAULT_RATIOS = base.DEFAULT_RATIOS
DEFAULT_SEED = base.DEFAULT_SEED
DUPLICATE_MIN_COPIES = 5
#: Corpora whose every record is test. verite is an evaluation benchmark;
#: averimatec is evaluation-only by decision (95% one class), so its official
#: train/val division is kept in meta but no record of it is placed in train.
PINNED_DATASETS = {"verite": "test", "averimatec": "test"}
RANDOM_SPLIT_DATASETS = ("welfake", "isot", "fakenewsnet")

#: Datasets whose records sharing an identical caption must share a split. DGM4
#: swaps one caption onto up to 109 images; a text-only model scores 82.6% vs
#: 79.4% majority on random folds and 79.0% on caption-grouped folds, i.e. the
#: whole text signal is caption memorisation. DGM4's own split already keeps
#: captions disjoint (0 of 7,600 swapped captions straddle); this keeps any
#: reconciliation move from undoing that.
TEXT_GROUPED_DATASETS = frozenset({"dgm4"})

Node = tuple[str, str]  # (dataset, record_id)


class AtomicityError(base.SplitError):
    """A duplicate group straddles a split boundary."""


@dataclass
class SplitInput:
    """What the assigner needs per record; built by scripts/build_processed.py."""

    label: str | None
    official_split: str | None
    hashes: list[str] = field(default_factory=list)   # usable image sha256s
    domains: set[str] = field(default_factory=set)    # factify2 only
    text_key: str | None = None                       # TEXT_GROUPED_DATASETS only


def constraint_groups(records: dict[Node, SplitInput],
                      copies: dict[str, int]) -> list[set[Node]]:
    """Sets of records that must share a split, one per qualifying hash."""
    by_hash: dict[str, set[Node]] = defaultdict(set)
    for node, info in records.items():
        for h in info.hashes:
            by_hash[h].add(node)
    groups = []
    for h, nodes in by_hash.items():
        cross = len({ds for ds, _ in nodes}) > 1
        if len(nodes) > 1 and (copies.get(h, len(nodes)) >= DUPLICATE_MIN_COPIES or cross):
            groups.append(nodes)
    by_text: dict[tuple[str, str], set[Node]] = defaultdict(set)
    for node, info in records.items():
        if info.text_key:
            by_text[(node[0], info.text_key)].add(node)
    groups += [nodes for nodes in by_text.values() if len(nodes) > 1]
    return groups


def _merge(groups: Iterable[set[Node]]) -> list[set[Node]]:
    """Union overlapping groups (an image in two groups chains them)."""
    parent: dict[Node, Node] = {}

    def find(x: Node) -> Node:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        g = list(g)
        if g:
            find(g[0])  # a single-record group must survive the merge
        for other in g[1:]:
            ra, rb = find(g[0]), find(other)
            if ra != rb:
                parent[ra] = rb
    merged: dict[Node, set[Node]] = defaultdict(set)
    for node in parent:
        merged[find(node)].add(node)
    return list(merged.values())


def assign_all(records: dict[Node, SplitInput], copies: dict[str, int], *,
               ratios=DEFAULT_RATIOS, seed: int = DEFAULT_SEED,
               threshold: float = base.DEFAULT_PURITY) -> tuple[dict[Node, str], dict[str, Any]]:
    groups = _merge(constraint_groups(records, copies))
    report: dict[str, Any] = {"constraint_groups": len(groups),
                              "records_in_constraint_groups": sum(len(g) for g in groups)}

    # pins that cross corpora: a group containing a pinned corpus takes its split
    pins: dict[Node, str] = {}
    for g in groups:
        wanted = [PINNED_DATASETS[ds] for ds, _ in g if ds in PINNED_DATASETS]
        if wanted:
            split = min(wanted, key=SPLIT_ORDER.index)
            for node in g:
                pins[node] = split

    placement: dict[Node, str] = {}

    # factify2: source-disjoint generator with duplicate links and pins
    f2 = {n: i for n, i in records.items() if n[0] == "factify2"}
    if f2:
        domains = {rid: i.domains for (_, rid), i in f2.items()}
        labels = {rid: i.label or "?" for (_, rid), i in f2.items()}
        purity = base.domain_purity(domains, labels)
        constrained = base.constrained_domains(purity, "predictive", threshold)
        links = [[rid for ds, rid in g if ds == "factify2"] for g in groups]
        comps = base.components(domains, constrained, [l for l in links if len(l) > 1])
        f2_pins = {rid: s for (ds, rid), s in pins.items() if ds == "factify2"}
        f2_place = base.assign(comps, labels, ratios, seed, pinned=f2_pins)
        violations = base.leakage(f2_place, domains, purity, threshold)
        base.assert_no_leakage(violations, "predictive", threshold)
        placement.update({("factify2", rid): s for rid, s in f2_place.items()})
        report["factify2"] = {
            "constrained_domains": len(constrained), "components": len(comps),
            "largest_component": len(comps[0]) if comps else 0,
            "pinned_records": len(f2_pins), "leakage_violations": len(violations),
            "domain_only": base.domain_only_accuracy(f2_place, domains, labels)}

    # everything else: pinned, official, or seeded random
    for ds in sorted({n[0] for n in records} - {"factify2"}):
        nodes = [n for n in records if n[0] == ds]
        if ds in PINNED_DATASETS:
            for n in nodes:
                placement[n] = PINNED_DATASETS[ds]
        elif ds in RANDOM_SPLIT_DATASETS or all(records[n].official_split is None for n in nodes):
            in_group = {n for g in groups for n in g if n[0] == ds}
            merged = _merge([{n for n in g if n[0] == ds} for g in groups] +
                            [{n} for n in nodes if n not in in_group])
            comps = sorted((sorted(rid for _, rid in c) for c in merged if c),
                           key=len, reverse=True)
            labels = {rid: records[(ds, rid)].label or "?" for _, rid in nodes}
            pinned = {rid: s for (d, rid), s in pins.items() if d == ds}
            for rid, s in base.assign(comps, labels, ratios, seed, pinned=pinned).items():
                placement[(ds, rid)] = s
        else:
            for n in nodes:
                placement[n] = records[n].official_split or "train"

    # reconcile every constraint group onto one split
    moved = Counter()
    for g in groups:
        splits_here = {placement[n] for n in g}
        if len(splits_here) == 1:
            continue
        f2_splits = {placement[n] for n in g if n[0] == "factify2"}
        if any(n in pins for n in g):
            target = min({pins[n] for n in g if n in pins}, key=SPLIT_ORDER.index)
        elif len(f2_splits) == 1:
            target = next(iter(f2_splits))
        else:
            target = min(splits_here, key=SPLIT_ORDER.index)
        for n in g:
            if placement[n] != target:
                moved[(n[0], placement[n], target)] += 1
                placement[n] = target
    report["moved_by_reconciliation"] = {f"{d}:{a}->{b}": c for (d, a, b), c in sorted(moved.items())}

    assert_atomic(placement, groups)
    if f2:  # reconciliation must not have re-opened a domain leak
        f2_place = {rid: s for (ds, rid), s in placement.items() if ds == "factify2"}
        base.assert_no_leakage(base.leakage(f2_place, domains, purity, threshold),
                               "predictive", threshold)
    report["split_sizes"] = {ds: dict(Counter(s for (d, _), s in placement.items() if d == ds))
                             for ds in sorted({n[0] for n in placement})}
    return placement, report


def assert_atomic(placement: dict[Node, str], groups: Iterable[set[Node]]) -> None:
    """Raise if any constraint group has members in more than one split."""
    bad = [g for g in groups if len({placement[n] for n in g}) > 1]
    if bad:
        example = sorted(bad[0])[:6]
        raise AtomicityError(
            f"{len(bad)} duplicate group(s) straddle a split boundary; e.g. "
            f"{[(n, placement[n]) for n in example]}. A model would be trained "
            "on pixels it is later scored on. Refusing to write these splits.")
