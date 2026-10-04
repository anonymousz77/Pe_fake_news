#!/usr/bin/env python
"""What Fakeddit's 6-way classes are, measured on our 150,000-row image sample.

    python scripts/fakeddit_classes.py      # -> data/reports/fakeddit_classes.json

Evidence for the reason-code decision, not a model. Every number is computed
on the seeded image sample (seed 20260901), titles from the TSVs:

* class names (verbatim from the paper) and counts; subreddits per class;
  seeded sample titles for codes 2 and 5 and for fakehistoryporn;
* label/subreddit purity -- the paper assigns labels per subreddit;
* text-only multinomial Naive Bayes (unigrams + bigrams, Laplace 1, 5-fold):
  how much of the label, and of the SUBREDDIT, the title alone gives away;
* leave-one-subreddit-out: train "code 2 vs rest" without one code-2
  subreddit and measure recall on that subreddit. If the class were a property
  of the image-text relationship, a model that learned it should transfer to
  an unseen subreddit of the same class; if it learned subreddit style, recall
  on the held-out subreddit collapses.

Naive Bayes is deliberately weak: anything it recovers from titles alone is a
LOWER bound on what the text gives away.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, REPORTS, raw_dir  # noqa: E402
from scripts.reason_codes import FAKEDDIT_6WAY  # noqa: E402

SEED = 20260901
FOLDS = 5
csv.field_size_limit(64 * 1024 * 1024)


def tokens(title: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", (title or "").lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class NaiveBayes:
    def fit(self, docs: Sequence[list[str]], labels: Sequence[str]) -> "NaiveBayes":
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.prior = Counter(labels)
        for toks, y in zip(docs, labels):
            self.counts[y].update(toks)
        self.vocab = set().union(*(c.keys() for c in self.counts.values()))
        self.total = {y: sum(c.values()) for y, c in self.counts.items()}
        self.n = len(labels)
        return self

    def predict(self, toks: list[str]) -> str:
        v = len(self.vocab)
        best, best_lp = None, -math.inf
        for y, c in self.counts.items():
            lp = math.log(self.prior[y] / self.n)
            denom = self.total[y] + v
            for t in toks:
                if t in self.vocab:
                    lp += math.log((c[t] + 1) / denom)
            if lp > best_lp:
                best, best_lp = y, lp
        return best


def folds(n: int) -> list[int]:
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    out = [0] * n
    for k, i in enumerate(idx):
        out[i] = k % FOLDS
    return out


def cross_val(docs, labels) -> list[str]:
    f = folds(len(docs))
    pred = [None] * len(docs)
    for k in range(FOLDS):
        tr = [i for i in range(len(docs)) if f[i] != k]
        model = NaiveBayes().fit([docs[i] for i in tr], [labels[i] for i in tr])
        for i in range(len(docs)):
            if f[i] == k:
                pred[i] = model.predict(docs[i])
    return pred


def scores(labels, pred) -> dict[str, Any]:
    acc = sum(a == b for a, b in zip(labels, pred)) / len(labels)
    per = {}
    for y in sorted(set(labels)):
        tp = sum(1 for a, b in zip(labels, pred) if a == y and b == y)
        fn = sum(1 for a, b in zip(labels, pred) if a == y and b != y)
        fp = sum(1 for a, b in zip(labels, pred) if a != y and b == y)
        per[y] = {"support": tp + fn, "recall": round(tp / max(1, tp + fn), 4),
                  "precision": round(tp / max(1, tp + fp), 4)}
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    return {"accuracy": round(acc, 4), "majority_baseline": round(majority, 4), "per_class": per}


def strip_samples(obj):
    """The report without per-record text (sample_titles), for the tracked copy."""
    if isinstance(obj, dict):
        return {k: (f"{len(v or [])} titles in data/interim/fakeddit_classes_full.json"
                    if k == "sample_titles" else strip_samples(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_samples(x) for x in obj]
    return obj


def load_sample() -> list[dict[str, str]]:
    sample = next((INTERIM / "fakeddit").glob(f"sample_seed{SEED}_n*.csv"))
    ids = {row["id"] for row in csv.DictReader(sample.open(encoding="utf-8"))}
    rows = []
    for fname in ("multimodal_train.tsv", "multimodal_validate.tsv"):
        with (raw_dir("fakeddit") / fname).open(encoding="utf-8", newline="") as h:
            for row in csv.DictReader(h, delimiter="\t"):
                if row["id"] in ids:
                    rows.append({"id": row["id"], "title": row["clean_title"],
                                 "subreddit": row["subreddit"], "label": row["6_way_label"]})
    return rows


def full_corpus_purity() -> dict[str, Any]:
    labels_of: dict[str, set[str]] = defaultdict(set)
    for fname in ("multimodal_train.tsv", "multimodal_validate.tsv"):
        with (raw_dir("fakeddit") / fname).open(encoding="utf-8", newline="") as h:
            for row in csv.DictReader(h, delimiter="\t"):
                labels_of[row["subreddit"]].add(row["6_way_label"])
    mixed = {s: sorted(l) for s, l in labels_of.items() if len(l) > 1}
    return {"subreddits": len(labels_of), "subreddits_with_more_than_one_label": mixed}


def leave_one_subreddit_out(rows, docs, code: str) -> dict[str, Any]:
    """Binary code-vs-rest; hold each of the code's subreddits out in turn."""
    y = ["pos" if r["label"] == code else "neg" for r in rows]
    neg_fold = folds(len(rows))
    out = {}
    for s in sorted({r["subreddit"] for r in rows if r["label"] == code}):
        train = [i for i, r in enumerate(rows) if r["subreddit"] != s and not
                 (y[i] == "neg" and neg_fold[i] == 0)]
        held_pos = [i for i, r in enumerate(rows) if r["subreddit"] == s]
        held_neg = [i for i in range(len(rows)) if y[i] == "neg" and neg_fold[i] == 0]
        model = NaiveBayes().fit([docs[i] for i in train], [y[i] for i in train])
        recall = sum(model.predict(docs[i]) == "pos" for i in held_pos) / len(held_pos)
        fpr = sum(model.predict(docs[i]) == "pos" for i in held_neg) / len(held_neg)
        out[s] = {"held_out_rows": len(held_pos), "recall_on_unseen_subreddit": round(recall, 4),
                  "false_positive_rate": round(fpr, 4)}
    return out


def main() -> int:
    rows = load_sample()
    docs = [tokens(r["title"]) for r in rows]
    rng = random.Random(SEED)
    report: dict[str, Any] = {"sample_rows": len(rows), "seed": SEED, "classes": {}}

    by_code = defaultdict(list)
    for r in rows:
        by_code[r["label"]].append(r)
    for code in sorted(by_code):
        subs = Counter(r["subreddit"] for r in by_code[code])
        report["classes"][code] = {
            "name": FAKEDDIT_6WAY[code]["name"], "rows_in_sample": len(by_code[code]),
            "subreddits": dict(subs.most_common(20)),
            "sample_titles": [r["title"] for r in rng.sample(by_code[code], 5)]
            if code in ("2", "5") else None}
    fhp = [r for r in rows if r["subreddit"] == "fakehistoryporn"]
    report["fakehistoryporn"] = {
        "rows_in_sample": len(fhp),
        "sample_titles": [r["title"] for r in rng.sample(fhp, 12)],
        "share_with_circa": round(sum("circa" in r["title"].split() for r in fhp) / len(fhp), 4),
        "share_with_circa_elsewhere": round(
            sum("circa" in r["title"].split() for r in rows if r["subreddit"] != "fakehistoryporn")
            / max(1, len(rows) - len(fhp)), 4)}
    report["purity"] = full_corpus_purity()

    labels = [r["label"] for r in rows]
    report["text_only_6way"] = scores(labels, cross_val(docs, labels))
    subs = [r["subreddit"] for r in rows]
    report["text_only_subreddit"] = {k: v for k, v in scores(subs, cross_val(docs, subs)).items()
                                     if k != "per_class"}
    for code in ("2", "5"):
        idx = [i for i, r in enumerate(rows) if r["label"] == code]
        sub_c = [rows[i]["subreddit"] for i in idx]
        report[f"text_only_subreddit_within_code_{code}"] = scores(
            sub_c, cross_val([docs[i] for i in idx], sub_c))
        report[f"leave_one_subreddit_out_code_{code}"] = leave_one_subreddit_out(rows, docs, code)
        binary = ["pos" if l == code else "neg" for l in labels]
        pred = cross_val(docs, binary)
        report[f"text_only_binary_code_{code}"] = {
            "in_distribution_recall_by_subreddit": {
                s: round(sum(pred[i] == "pos" for i in range(len(rows))
                             if rows[i]["subreddit"] == s) /
                         sum(1 for r in rows if r["subreddit"] == s), 4)
                for s in sorted({rows[i]["subreddit"] for i in idx})},
            **scores(binary, pred)["per_class"]["pos"]}
    # Sample titles are Fakeddit content and Fakeddit states no licence: they go
    # to data/interim (ignored); the tracked report keeps aggregates only.
    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "fakeddit_classes_full.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "fakeddit_classes.json").write_text(
        json.dumps(strip_samples(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
