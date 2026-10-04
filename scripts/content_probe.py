#!/usr/bin/env python
"""Rank images by how much they look like a registry class, for review by eye.

    python scripts/content_probe.py --status verdict --pool mocheg --pool averimatec
    python scripts/content_probe.py --status furniture --pool fakeddit --top 600

The registry (data/placeholders.yaml) is built by looking. Looking at every
image is impossible at 600,000 images, and looking only at duplicate groups and
tiny files misses the single-copy members of a class -- MOCHEG's verdict
graphics, Fakeddit's logo previews -- that are not small and not repeated.

This ranks the rest. Positives are the registry entries of ``--status``;
negatives are everything already reviewed and judged otherwise
(``--reviewed``, a JSON list of sha256) plus a seeded random sample of the
pool. A logistic-regression probe on the CLIP ViT-B/32 embeddings
(scripts/match_images.py clip-embed) is cross-validated, then scores every
unregistered, unreviewed image; the ranking is written for contact-sheet
review. NOTHING is registered here: every registry entry is a decision made by
eye, recorded with this ranking as how the image was found.

Seed 20260901; numpy only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM  # noqa: E402
from scripts import placeholders  # noqa: E402
from scripts.match_images import EMB_DIR  # noqa: E402

SEED = 20260901
OUT = INTERIM / "content_probe"


def load_pool(names: list[str]):
    import numpy as np

    paths: list[dict[str, str]] = []
    vecs = []
    for name in names:
        listing = json.loads((EMB_DIR / f"{name}.paths.json").read_text(encoding="utf-8"))
        paths += [{**p, "dataset": name} for p in listing]
        vecs.append(np.load(EMB_DIR / f"{name}.npy").astype("float32"))
    return paths, np.concatenate(vecs)


def fit(X, y, l2: float = 1e-3, iters: int = 600, lr: float = 0.5):
    """Class-weighted L2 logistic regression by gradient descent."""
    import numpy as np

    w, b = np.zeros(X.shape[1], dtype="float32"), 0.0
    pos_weight = (len(y) - y.sum()) / max(1.0, y.sum())
    sw = np.where(y == 1, pos_weight, 1.0).astype("float32")
    sw /= sw.mean()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def probe(status: str, pools: list[str], reviewed: set[str], negatives: int = 6000,
          extra_positive: set[str] | None = None, train_pools: list[str] | None = None
          ) -> dict[str, Any]:
    """Train on ``train_pools`` (default: ``pools``), rank the unreviewed images of ``pools``."""
    import numpy as np

    rng = np.random.default_rng(SEED)
    registry = placeholders.registry()
    positive = {h for h, e in registry.items() if e["status"] == status} | set(extra_positive or ())
    every = list(dict.fromkeys((train_pools or []) + pools))
    paths, E = load_pool(every)
    E = E * 10  # unit vectors: scale so the probe's logits have range
    first: dict[str, int] = {}
    for i, p in enumerate(paths):
        first.setdefault(p["sha256"], i)
    pos = [i for h, i in first.items() if h in positive]
    neg_seen = [i for h, i in first.items() if h in reviewed and h not in positive]
    fresh = [i for h, i in first.items() if h not in registry and h not in positive
             and h not in reviewed]
    neg_rand = list(rng.choice(fresh, min(negatives, len(fresh)), replace=False))
    unseen = [i for i in fresh if paths[i]["dataset"] in pools]
    if not pos:
        raise SystemExit(f"no registry entries with status {status!r} in pools {pools}")
    X = E[pos + neg_seen + neg_rand]
    y = np.array([1] * len(pos) + [0] * (len(neg_seen) + len(neg_rand)), dtype="float32")
    order = rng.permutation(len(y))
    folds = np.array_split(order, 5)
    held = np.zeros(len(y))
    for k in range(5):
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        w, b = fit(X[tr], y[tr])
        held[folds[k]] = X[folds[k]] @ w + b
    rank = np.argsort(np.argsort(-held))
    pos_ranks = sorted(int(rank[i]) for i in range(len(pos)))
    cv = {f"positives_in_top_{m}x": sum(r < m * len(pos) for r in pos_ranks) for m in (1, 2, 4)}
    w, b = fit(X, y)
    scores = E[unseen] @ w + b
    ranked = sorted(zip(scores.tolist(), unseen), reverse=True)
    return {"status": status, "pools": pools, "positives": len(pos),
            "reviewed_negatives": len(neg_seen), "random_negatives": len(neg_rand),
            "cv": {"examples": int(len(y)), **cv}, "unreviewed": len(unseen),
            "score_above_0": sum(1 for s, _ in ranked if s > 0),
            "ranked": [{"score": round(s, 3), **paths[i]} for s, i in ranked]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", required=True, choices=sorted(placeholders.MISSING_STATUSES))
    parser.add_argument("--pool", action="append", dest="pools", required=True)
    parser.add_argument("--reviewed", type=Path, default=None,
                        help="JSON list of sha256 already reviewed and judged NOT this status")
    parser.add_argument("--top", type=int, default=2000, help="how many ranked rows to write")
    parser.add_argument("--train-pool", action="append", dest="train_pools", default=None,
                        help="extra pools whose labelled images train the probe but are not ranked")
    parser.add_argument("--positives", type=Path, default=None,
                        help="JSON list of sha256 judged this status but not yet registered")
    args = parser.parse_args(argv)
    reviewed = set(json.loads(args.reviewed.read_text())) if args.reviewed else set()
    extra = set(json.loads(args.positives.read_text())) if args.positives else set()
    out = probe(args.status, args.pools, reviewed, extra_positive=extra, train_pools=args.train_pools)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.status}_{'+'.join(args.pools)}.json"
    out["ranked"] = out["ranked"][:args.top]
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "ranked"}, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
