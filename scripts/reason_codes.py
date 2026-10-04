#!/usr/bin/env python
"""reason_code and modality_evidence: why a record is flagged, derived from its label.

    python scripts/reason_codes.py            # distribution per dataset per class

``reason_code`` takes one of six values:

    content_refuted     the claim was checked and is false
    content_unverified  the claim was checked and could not be settled
    media_mismatch      real image, real caption, but they do not belong together
    wrong_image         the image is genuine but described falsely
    none                adjudicated as true: checked, and no defect found
    unmapped            the dataset's label scheme carries no reason information

``none`` and ``unmapped`` are different states and are never merged: the first
is a verdict, the second is the absence of one. The first five are SUPERVISED
codes; ``unmapped`` rows -- and rows whose reason_code is null, below -- never
train or evaluate a reason-code head. That is enforced by
scripts/processed_loader.py, not by this docstring.

Null is reserved for a row whose dataset DOES carry reason information but whose
own label fits no code: AVeriTeC/AVerImaTeC "Conflicting
Evidence/Cherrypicking", an unlabelled row, and M4FC (see OPEN).

Provenance of every rule is recorded: SPECIFIED (given in the brief,
2026-10-01), INHERITED (AVerImaTeC takes AVeriTeC's rule because it uses the
identical label scheme -- applied on instruction, flagged in every report).
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPERVISED_CODES = ("content_refuted", "content_unverified", "media_mismatch",
                    "wrong_image", "none")
UNMAPPED = "unmapped"
REASON_CODES = SUPERVISED_CODES + (UNMAPPED,)

SPECIFIED, INHERITED = "specified", "inherited"

_AVERITEC = {
    "Refuted": "content_refuted",
    "Not Enough Evidence": "content_unverified",
    "Supported": "none",
}

#: dataset -> label -> (reason_code, provenance of the rule)
MAPPING: dict[str, dict[str, tuple[str, str]]] = {
    "verite": {
        "out-of-context": ("media_mismatch", SPECIFIED),
        "miscaptioned": ("wrong_image", SPECIFIED),
        "true": ("none", SPECIFIED),
    },
    "mocheg": {
        "refuted": ("content_refuted", SPECIFIED),
        "NEI": ("content_unverified", SPECIFIED),
        "supported": ("none", SPECIFIED),
    },
    "averitec": {label: (code, SPECIFIED) for label, code in _AVERITEC.items()},
    "averimatec": {label: (code, INHERITED) for label, code in _AVERITEC.items()},
    # AVeriTeC's rule applied to M4FC's verdict_coarse labels: confirmed in the
    # brief of 2026-10-01 after M4FC was found in neither list.
    "m4fc": {
        "false": ("content_refuted", SPECIFIED),
        "not enough information": ("content_unverified", SPECIFIED),
        "true": ("none", SPECIFIED),
    },
    "factify2": {
        "Refute": ("content_refuted", SPECIFIED),
        "Insufficient_Text": ("content_unverified", SPECIFIED),
        "Insufficient_Multimodal": ("content_unverified", SPECIFIED),
        "Support_Text": ("none", SPECIFIED),
        "Support_Multimodal": ("none", SPECIFIED),
    },
}

#: DGM4 is a candidate for media_mismatch ONLY through scripts/dgm4_audit.py,
#: whose decision rule (2026-10-02) is: map text_swap iff (a) no publisher is
#: label-pure, (b) a text-only classifier is near the majority baseline, and
#: (c) swaps are within-publisher. As run, the whole class fails (c) -- 50.2%
#: of swaps take another publisher's caption -- and the within-publisher subset
#: fails (a), so DGM4 is UNMAPPED and media_mismatch has no training source.
#: tests/test_reason_codes.py holds this flag to the audit report's decision.
DGM4_AUDIT_PASSED = False

#: Datasets whose label scheme carries no reason information: every row is
#: ``unmapped``, whatever its label.
UNMAPPED_DATASETS = frozenset({"welfake", "liar", "fakenewsnet", "fakeddit", "isot"}
                              | (set() if DGM4_AUDIT_PASSED else {"dgm4"}))

#: The row rule the audit would license: a pure text swap -- a real news photo
#: with another sample's caption -- whose caption came from the image's OWN
#: publisher is media_mismatch, as WEAK supervision (synthetic, not
#: fact-checked): training only, never evaluation. Every other DGM4 row stays
#: unmapped: pristine pairs (`none` is reserved for five adjudicated labels),
#: cross-publisher swaps (detectable from writing-style mismatch alone), face
#: edits (image manipulation, a different construct), text-attribute edits,
#: and rows that are both a face edit and a text swap (construct ambiguous).
DGM4_MEDIA_MISMATCH_LABEL = "text_swap"
WEAK_SUPERVISION_DATASETS = frozenset({"dgm4"})


def dgm4_reason_code(label: str | None, meta: dict[str, Any] | None) -> str:
    if label == DGM4_MEDIA_MISMATCH_LABEL and (meta or {}).get("swap_publisher") == "same":
        return "media_mismatch"
    return UNMAPPED


#: Labels in a reason-bearing scheme that fit no code: reason_code null.
NO_CLEAN_CODE: dict[str, dict[str, str]] = {
    "averitec": {"Conflicting Evidence/Cherrypicking":
                 "evidence both supports and refutes, or the claim is true but cherry-picked: "
                 "neither refuted, unverified, nor clean"},
    "averimatec": {"Conflicting Evidence/Cherrypicking": "as for AVeriTeC"},
}

#: Datasets in neither list of the brief. Left null and flagged, not guessed.
#: (m4fc was here until its mapping was confirmed on 2026-10-01.)
OPEN: dict[str, str] = {}

#: Corpora that may evaluate but never train. Carried into every processed row
#: as ``evaluation_only`` and enforced by scripts/processed_loader.py.
EVALUATION_ONLY: dict[str, str] = {
    "averimatec": "95% one class (897 of 945 Refuted): evaluation only, per the brief",
    "verite": "an evaluation benchmark with no training split: every row is test",
}

#: Fakeddit's 6_way_label, decoded. The TSVs carry only the integer; the
#: names are verbatim from the paper (arXiv 1911.03854, section 3.3 and the
#: subreddit table), and each integer is tied to a name by its subreddits --
#: every subreddit under one integer is listed under one class in the paper.
#: NOT MAPPED to a reason code: shown for decision first.
FAKEDDIT_6WAY: dict[str, dict[str, Any]] = {
    "0": {"name": "True", "subreddits": ["mildlyinteresting", "photoshopbattles", "nottheonion",
                                          "upliftingnews", "neutralnews", "usnews", "usanews", "pic"]},
    "1": {"name": "Satire/Parody", "subreddits": ["fakealbumcovers", "theonion", "satire",
                                                   "waterfordwhispersnews"]},
    "2": {"name": "False Connection", "subreddits": ["pareidolia", "fakehistoryporn",
                                                      "misleadingthumbnails", "confusing_perspective"]},
    "3": {"name": "Imposter Content", "subreddits": ["subredditsimulator", "subsimulatorgpt2"]},
    "4": {"name": "Manipulated Content", "subreddits": ["psbattle_artwork"]},
    "5": {"name": "Misleading Content", "subreddits": ["propagandaposters", "savedyouaclick",
                                                        "fakefacts"]},
}

#: Rulings on Fakeddit, 2026-10-01, recorded as MEASURED findings
#: (data/reports/fakeddit_classes.json; docs/data_card.md, "Fakeddit's labels
#: are its subreddits"). Code 2 (False Connection) stays unmapped: every one of
#: the 22 subreddits is 100% one label, title text recovers the subreddit at
#: 59.3% (baseline 29.9%) and at 79.6% within code 2 (baseline 44.3%), and
#: leave-one-subreddit-out recall collapses to 0.003-0.037.
FAKEDDIT_RULINGS: dict[str, str] = {
    "2": "unmapped: a subreddit-style shortcut, not an image-text mismatch label",
}

#: Rows excluded from reason-code supervision by a property other than their
#: label, so the exclusion survives any later mapping of the dataset's labels.
#: fakehistoryporn: real photos under comedic anachronistic captions -- the joke
#: is detectable in the text alone; a different construct from VERITE
#: miscaptioned (ruling 2026-10-01).
EXCLUDED_FROM_REASON_SUPERVISION: dict[str, dict[str, frozenset[str]]] = {
    "fakeddit": {"subreddit": frozenset({"fakehistoryporn"})},
}

#: Factify2's Text/Multimodal division, where the dataset makes it.
MODALITY_EVIDENCE: dict[str, dict[str, str]] = {
    "factify2": {
        "Support_Text": "text",
        "Support_Multimodal": "multimodal",
        "Insufficient_Text": "text",
        "Insufficient_Multimodal": "multimodal",
    },
}


def reason_code(dataset: str, label: str | None,
                meta: dict[str, Any] | None = None) -> str | None:
    """The code, ``unmapped`` for a reason-less scheme or an excluded row, or None."""
    if dataset in UNMAPPED_DATASETS:
        return UNMAPPED
    if dataset == "dgm4":
        return dgm4_reason_code(label, meta)
    for field, values in EXCLUDED_FROM_REASON_SUPERVISION.get(dataset, {}).items():
        if (meta or {}).get(field) in values:
            return UNMAPPED
    if label is None:
        return None
    hit = MAPPING.get(dataset, {}).get(str(label))
    return hit[0] if hit else None


def is_supervised(code: str | None) -> bool:
    return code in SUPERVISED_CODES


def modality_evidence(dataset: str, label: str | None) -> str | None:
    return MODALITY_EVIDENCE.get(dataset, {}).get(str(label)) if label is not None else None


def evaluation_only(dataset: str) -> bool:
    return dataset in EVALUATION_ONLY


def weak_supervision(dataset: str, code: str | None) -> bool:
    """Synthetic labels that may train a reason-code head but never score one."""
    return dataset in WEAK_SUPERVISION_DATASETS and is_supervised(code)


def classify(dataset: str, label: str | None) -> str:
    """How a label was treated: mapped, unmapped, no_clean_code, open, unlabelled, unexpected."""
    if dataset in UNMAPPED_DATASETS:
        return "unmapped"
    if label is None:
        return "unlabelled"
    if str(label) in MAPPING.get(dataset, {}):
        return "mapped"
    if str(label) in NO_CLEAN_CODE.get(dataset, {}):
        return "no_clean_code"
    if dataset in OPEN:
        return "open"
    return "unexpected"


def bucket(dataset: str, label: str | None) -> str:
    code = reason_code(dataset, label)
    return code if code is not None else f"null:{classify(dataset, label)}"


def report(label_counts: dict[str, Counter]) -> dict[str, Any]:
    """Per dataset per class: reason_code, with none and unmapped kept apart."""
    out: dict[str, Any] = {
        "codes": list(REASON_CODES), "supervised_codes": list(SUPERVISED_CODES),
        "datasets": {}, "not_cleanly_mapped": [],
        "inherited_rules": [{"dataset": ds, "label": lab, "reason_code": code,
                             "from": "averitec"}
                            for ds, m in MAPPING.items()
                            for lab, (code, how) in m.items() if how == INHERITED],
        "evaluation_only": EVALUATION_ONLY, "open": OPEN,
        "fakeddit_6_way": FAKEDDIT_6WAY,
    }
    total_sup = total_sup_train = 0
    for ds, counts in sorted(label_counts.items()):
        per_class, dist = {}, Counter()
        for label, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
            label = None if label in (None, "None") else label
            b = bucket(ds, label)
            per_class[str(label)] = {"reason_code": b, "records": n}
            dist[b] += n
            kind = classify(ds, label)
            if kind in ("no_clean_code", "open", "unexpected", "unlabelled"):
                out["not_cleanly_mapped"].append({
                    "dataset": ds, "label": label, "records": n, "kind": kind,
                    "why": (NO_CLEAN_CODE.get(ds, {}).get(str(label)) or OPEN.get(ds)
                            or ("no label" if kind == "unlabelled" else "unexpected label"))})
        supervised = sum(n for b, n in dist.items() if b in SUPERVISED_CODES)
        total_sup += supervised
        total_sup_train += 0 if evaluation_only(ds) else supervised
        out["datasets"][ds] = {
            "records": sum(counts.values()),
            "reason_code": dict(sorted(dist.items())),
            "none": dist.get("none", 0), "unmapped": dist.get(UNMAPPED, 0),
            "supervised_records": supervised,
            "evaluation_only": evaluation_only(ds),
            "per_class": per_class,
        }
    out["total_supervised_records"] = total_sup
    out["total_supervised_trainable"] = total_sup_train
    out["total_supervised_evaluation_only"] = total_sup - total_sup_train
    return out


def distribution(records, exclude_splits: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """reason_code per dataset per class, computed ROW BY ROW (meta can matter).

    ``records`` yields dicts with dataset, label, meta, official_split. none and
    unmapped are separate columns; supervised totals exclude both evaluation-only
    corpora (reported apart) and nothing else.
    """
    exclude_splits = exclude_splits or {}
    per: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for rec in records:
        ds = rec["dataset"]
        if rec.get("official_split") in exclude_splits.get(ds, set()):
            continue
        code = reason_code(ds, rec["label"], rec.get("meta"))
        b = code if code is not None else f"null:{classify(ds, rec['label'])}"
        per[ds][str(rec["label"])][b] += 1
    out: dict[str, Any] = {"datasets": {}}
    totals = Counter()
    for ds, classes in sorted(per.items()):
        dist = Counter()
        for c in classes.values():
            dist.update(c)
        sup = sum(n for b, n in dist.items() if b in SUPERVISED_CODES)
        kind = ("evaluation_only" if evaluation_only(ds) else
                "weak_training_only" if ds in WEAK_SUPERVISION_DATASETS else "trainable")
        totals[kind] += sup
        out["datasets"][ds] = {"records": sum(dist.values()), "reason_code": dict(sorted(dist.items())),
                               "none": dist.get("none", 0), "unmapped": dist.get(UNMAPPED, 0),
                               "supervised_records": sup, "supervision_use": kind,
                               "per_class": {lab: dict(sorted(c.items())) for lab, c in sorted(classes.items())}}
    out["supervised_records"] = {k: v for k, v in totals.items() if v}
    out["supervised_total"] = sum(totals.values())
    return out


def interim_label_counts(exclude_splits: dict[str, set[str]] | None = None) -> dict[str, Counter]:
    from configs.paths import INTERIM

    exclude_splits = exclude_splits or {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for path in sorted((INTERIM / "records").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("official_split") in exclude_splits.get(rec["dataset"], set()):
                    continue
                counts[rec["dataset"]][rec["label"]] += 1
    return counts


def iter_interim_records():
    from configs.paths import INTERIM

    for path in sorted((INTERIM / "records").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def main() -> int:
    from configs.paths import REPORTS
    from scripts.build_processed import EXCLUDED_SPLITS

    out = {"rules": report(interim_label_counts(EXCLUDED_SPLITS)),
           "distribution": distribution(iter_interim_records(), EXCLUDED_SPLITS)}
    (REPORTS / "reason_codes.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["distribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
