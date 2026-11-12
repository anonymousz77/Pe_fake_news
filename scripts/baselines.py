#!/usr/bin/env python
"""Track B: the baselines the paper needs, per class, with bootstrap intervals.

    python scripts/baselines.py hostname       # Factify2's hostname-only baseline, reproduced, per class
    python scripts/baselines.py text-only      # TF-IDF + logistic regression on the claim, every corpus
    python scripts/baselines.py muse-embed [--with-evidence]   # CLIP L/14 vectors (laptop GPU)
    python scripts/baselines.py muse-fit       # MUSE, adapted: DT, RF, MLP on M4FC train

Each writes its section of data/reports/baselines.json, an aggregate report
with no record id or name in it (consistency.public_report refuses one).

* **Hostname-only, Factify2's source-disjoint test.** Reproduced exactly from
  the data (usable records, the current split files, the hosts of the interim
  image URLs; confound.majority_map, as splits.domain_only_accuracy fits it),
  and asserted equal to data/reports/factify2_domain_only_final.json, which no
  script produced until now. Then per class: by (record, host) pair beside the
  pair majority (56.9%), and by record (the claim image's host) beside the
  record majority (54.4%). Intervals resample whole records.
* **Text-only claim classifier.** m4fc_audit's TF-IDF + logistic regression
  (class-balanced, C=1, nothing tuned), fitted on a corpus's train split and
  scored once on its evaluation split, on the claim text only -- never
  evidence_text, which is MOCHEG's fact-checker text and AVeriTeC's justification.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, REPORTS  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402

SEED = cs.SEED
BOOTSTRAP = 2000
REPORT = REPORTS / "baselines.json"
FINAL_HOSTNAME = REPORTS / "factify2_domain_only_final.json"
INTERIM_FACTIFY2 = INTERIM / "records" / "factify2.jsonl.gz"


# --------------------------------------------------------------------------
# the report: one section per baseline, ids refused
# --------------------------------------------------------------------------


def write_section(name: str, section: dict[str, Any], path: Path = REPORT) -> None:
    """Replace one section of the report, keep the others."""
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    report[name] = {"written_at": utcnow(), **section}
    cs.public_report(dict(sorted(report.items())), path)


# --------------------------------------------------------------------------
# per-class scores with intervals that resample clusters
# --------------------------------------------------------------------------


def scored_clusters(y_true: Sequence[str], y_pred: Sequence[str], clusters: Sequence[str],
                    classes: Sequence[str], seed: int = SEED) -> dict[str, Any]:
    """As m4fc_audit.scored (per-class support, precision, recall, F1;
    accuracy, macro-F1), with 95% percentile intervals over resampled CLUSTERS:
    the rows of one cluster (one record's host pairs) move together."""
    import numpy as np

    from scripts.m4fc_audit import _metrics

    classes = list(classes)
    k = len(classes)
    index = {cl: i for i, cl in enumerate(classes)}
    t = np.array([index[y] for y in y_true])
    p = np.array([index[y] for y in y_pred])
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    members: list[list[int]] = [[] for _ in ids]
    for row, c in enumerate(clusters):
        members[pos[c]].append(row)
    members_np = [np.array(m) for m in members]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOTSTRAP):
        pick = np.concatenate([members_np[i] for i in rng.integers(0, len(ids), size=len(ids))])
        boots.append(_metrics(t[pick], p[pick], k))

    def interval(values) -> list[float] | None:
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        return None if not len(v) else [round(float(np.percentile(v, 2.5)), 4),
                                        round(float(np.percentile(v, 97.5)), 4)]

    acc, macro, prec, rec, f1, support = _metrics(t, p, k)
    out = {"rows": len(t), "clusters": len(ids), "classes": classes,
           "accuracy": {"value": round(acc, 4), "ci": interval([b[0] for b in boots])},
           "macro_f1": {"value": round(macro, 4), "ci": interval([b[1] for b in boots])},
           "per_class": {}}
    for i, cl in enumerate(classes):
        out["per_class"][cl] = {
            "support": int(support[i]),
            "precision": {"value": round(float(prec[i]), 4), "ci": interval([b[2][i] for b in boots])},
            "recall": {"value": None if np.isnan(rec[i]) else round(float(rec[i]), 4),
                       "ci": interval([b[3][i] for b in boots])},
            "f1": {"value": None if np.isnan(f1[i]) else round(float(f1[i]), 4),
                   "ci": interval([b[4][i] for b in boots])}}
    return out


def share_of_most_common(labels: Sequence[str]) -> dict[str, Any]:
    """The applicable majority: the evaluation set's own most common label."""
    label, n = Counter(labels).most_common(1)[0]
    return {"label": label, "share": round(n / len(labels), 4), "n": len(labels)}


# --------------------------------------------------------------------------
# the hostname-only baseline, Factify2
# --------------------------------------------------------------------------


def factify2_records() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], dict[str, str]]:
    """(record -> split, record -> label, record -> its image hosts, record ->
    its claim image's host) over usable records: the processed layer's flag and
    split, the interim layer's image URLs (confound.domain_of)."""
    from scripts.confound import domain_of
    from scripts.processed_loader import iter_rows

    split, label = {}, {}
    for r in iter_rows("factify2"):
        if r["usable"]:
            split[r["record_id"]], label[r["record_id"]] = r["split"], r["label"]
    hosts: dict[str, set[str]] = defaultdict(set)
    claim_host: dict[str, str] = {}
    with gzip.open(INTERIM_FACTIFY2, "rt", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["record_id"] not in split:
                continue
            for img in r.get("images") or []:
                if img.get("url"):
                    hosts[r["record_id"]].add(domain_of(img["url"]))
                    if img.get("role") == "claim":
                        claim_host[r["record_id"]] = domain_of(img["url"])
    missing = [r for r in split if not hosts.get(r)]
    if missing:
        raise SystemExit(f"factify2: {len(missing):,} usable record(s) have no image host")
    return split, label, hosts, claim_host


def random_placement(ids: Sequence[str], seed: int = 20260901) -> dict[str, str]:
    """The random 70/15/15 arm of the final JSON: sorted usable ids, shuffled."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    a, b = round(0.7 * len(order)), round(0.85 * len(order))
    return {r: ("train" if i < a else "val" if i < b else "test") for i, r in enumerate(order)}


def host_pairs(placement: dict[str, str], hosts: dict[str, set[str]], labels: dict[str, str],
               which: str) -> list[tuple[str, str, str]]:
    """(record, host, label) for every distinct host of every record in ``which``."""
    return [(r, h, labels[r]) for r, s in sorted(placement.items()) if s == which
            for h in sorted(hosts[r])]


def hostname_section() -> dict[str, Any]:
    from scripts.confound import majority_map
    from scripts.m4fc_audit import scored
    from scripts.splits import domain_only_accuracy

    split, label, hosts, claim_host = factify2_records()
    final = json.loads(FINAL_HOSTNAME.read_text(encoding="utf-8"))
    arms = {"source_disjoint": split, "random_70_15_15": random_placement(list(split))}
    reproduced = {name: domain_only_accuracy(pl, hosts, label) for name, pl in arms.items()}
    for name, got in reproduced.items():
        if got != final[name]:
            raise SystemExit(f"hostname baseline, {name}: reproduced {got}, the final JSON says "
                             f"{final[name]}; refusing to score a figure that does not reproduce")
    classes = sorted(set(label.values()))
    train = host_pairs(split, hosts, label, "train")
    test = host_pairs(split, hosts, label, "test")
    table, fallback = majority_map([(h, y) for _, h, y in train])
    pair = scored_clusters([y for _, _, y in test], [table.get(h, fallback) for _, h, _ in test],
                           [r for r, _, _ in test], classes)
    pair["applicable_majority"] = share_of_most_common([y for _, _, y in test])
    pair["unseen_host_rate"] = round(sum(1 for _, h, _ in test if h not in table) / len(test), 4)
    train_rec = [r for r, s in split.items() if s == "train"]
    test_rec = sorted(r for r, s in split.items() if s == "test")
    rtable, rfallback = majority_map([(claim_host[r], label[r]) for r in train_rec])
    record = scored([label[r] for r in test_rec], [rtable.get(claim_host[r], rfallback) for r in test_rec],
                    rfallback, classes)
    record["applicable_majority"] = share_of_most_common([label[r] for r in test_rec])
    record["unseen_host_rate"] = round(sum(1 for r in test_rec if claim_host[r] not in rtable)
                                       / len(test_rec), 4)
    return {"dataset": "factify2", "split": "source-disjoint test (data/processed/splits)",
            "population": "usable records", "reproduces": {"file": FINAL_HOSTNAME.name, **reproduced},
            "classifier": "majority label per host, fitted on train; an unseen host gets the train "
                          f"majority ({fallback})",
            "by_pair": pair,
            "by_record": {"host": "the claim image's", **record},
            "train_refute_records": sum(1 for r in train_rec if label[r] == "Refute"),
            "note": "Per class always (CLAUDE.md, Factify2 rule 3). The source-disjoint train split "
                    "holds 15 usable Refute records, so a trained classifier's Refute recall is "
                    "near 0 by construction."}


def cmd_hostname(args) -> int:
    section = hostname_section()
    write_section("hostname_factify2", section)
    p, r = section["by_pair"], section["by_record"]
    print(f"reproduced: {json.dumps(section['reproduces'])}")
    print(f"by pair:   accuracy {p['accuracy']['value']} {p['accuracy']['ci']} beside the pair "
          f"majority {p['applicable_majority']['share']} ({p['applicable_majority']['label']})")
    print(f"by record: accuracy {r['model']['accuracy']['value']} {r['model']['accuracy']['ci']} beside "
          f"the record majority {r['applicable_majority']['share']}")
    print(f"-> {cs.rel(REPORT)}")
    return 0


# --------------------------------------------------------------------------
# the text-only claim classifier
# --------------------------------------------------------------------------

#: Each corpus's labels as the text-side check's three verdicts; a label
#: missing here has none (Conflicting Evidence/Cherrypicking) and is left out.
THREE_WAY = {
    "averitec": {"Refuted": "refuted", "Supported": "supported",
                 "Not Enough Evidence": "not_enough_evidence"},
    "m4fc": {"false": "refuted", "true": "supported", "not enough information": "not_enough_evidence"},
    "mocheg": {"refuted": "refuted", "supported": "supported", "NEI": "not_enough_evidence"},
}
THREE_WAY["averimatec"] = THREE_WAY["averitec"]
#: (train corpus, train split, scored corpus, scored split, label scheme).
#: AVerImaTeC is evaluation-only: it is scored with AVeriTeC's model, as it
#: takes AVeriTeC's reason-code rule. LIAR and Factify2 keep their own labels.
TEXT_ONLY = {
    "averitec": ("averitec", "train", "averitec", "val", "three_way"),
    "averitec_native": ("averitec", "train", "averitec", "val", "native"),
    "averimatec": ("averitec", "train", "averimatec", "test", "three_way"),
    "m4fc": ("m4fc", "train", "m4fc", "val", "three_way"),
    "mocheg": ("mocheg", "train", "mocheg", "test", "three_way"),
    "liar": ("liar", "train", "liar", "test", "native"),
    "factify2": ("factify2", "train", "factify2", "test", "native"),
}


def claim_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    """Usable labelled rows of one split, through the guards: M4FC through
    consistency.rows (its test is never read), and an evaluation-only corpus
    (AVerImaTeC) only ever as the scored side."""
    from scripts.processed_loader import iter_rows

    if dataset == "m4fc":
        rows_ = list(cs.rows("m4fc", (split,)))
    elif dataset == "verite":
        raise SystemExit("VERITE has no train split and is evaluation-only")
    else:
        rows_ = [r for r in iter_rows(dataset, split=split) if r["usable"]]
    return [r for r in rows_ if r.get("label") is not None]


def text_only_one(name: str) -> dict[str, Any]:
    from scripts.m4fc_audit import text_audit

    tr_ds, tr_split, ev_ds, ev_split, scheme = TEXT_ONLY[name]
    if ev_ds == "averimatec" and tr_ds == "averimatec":
        raise SystemExit("AVerImaTeC is evaluation-only: it never fits a model")

    def labelled(rows_, ds):
        if scheme == "native":
            return [(r["text"] or "", r["label"]) for r in rows_]
        return [(r["text"] or "", THREE_WAY[ds][r["label"]]) for r in rows_ if r["label"] in THREE_WAY[ds]]

    train = labelled(claim_rows(tr_ds, tr_split), tr_ds)
    scored_ = labelled(claim_rows(ev_ds, ev_split), ev_ds)
    res = text_audit([t for t, _ in train], [y for _, y in train],
                     [t for t, _ in scored_], [y for _, y in scored_])
    res["applicable_majority"] = share_of_most_common([y for _, y in scored_])
    return {"trained_on": f"{tr_ds} {tr_split}", "scored_on": f"{ev_ds} {ev_split}",
            "labels": scheme, "input": "the claim text only (never evidence_text)", **res}


def cmd_text_only(args) -> int:
    names = args.only or list(TEXT_ONLY)
    for name in names:
        section = text_only_one(name)
        write_section(f"text_only_{name}", section)
        m = section["model"]
        print(f"{name:16} {section['scored_on']:16} rows {section['rows']:>6,}  accuracy "
              f"{m['accuracy']['value']} {m['accuracy']['ci']}  macro-F1 {m['macro_f1']['value']} "
              f"{m['macro_f1']['ci']}  | applicable majority {section['applicable_majority']['share']} "
              f"({section['applicable_majority']['label']})", flush=True)
    print(f"-> {cs.rel(REPORT)}")
    return 0


# --------------------------------------------------------------------------
# MUSE, adapted (github.com/stevejpapad/outcontext-misinfo-progress, Apache-2.0)
# --------------------------------------------------------------------------
#
# MUSE scores a pair by CLIP similarities: image-caption, and four against
# external evidence (MUSE's evidence is web search results: top-1 by
# similarity, 0 when there is none). Trained on NewsCLIPpings. Here, stated as
# an adaptation: NewsCLIPpings is not held (deferred in the register), so it
# trains on M4FC train (true against out-of-context, report C's population);
# VERITE has no date, so it gets no evidence and only the image-caption
# similarity; M4FC's evidence is the text side's dated retrieval (the top
# five documents' TITLES, the closest short text to MUSE's evidence captions),
# so of MUSE's five evidence similarities two exist here: image-evidence text
# and caption-evidence text. No evidence image exists. Class weights are
# balanced (M4FC train is 164 true to the rest; MUSE's data is balanced), and
# the MLP early-stops on balanced accuracy over a 15% holdout of TRAIN.

MUSE_DIR = INTERIM / "textcheck" / "muse"
MUSE_EMB = MUSE_DIR / "clip_l14.npz"
CLIP_BATCH = 16
FEATURES = {"img": ("img_txt",), "img_text_evidence": ("img_txt", "img_X_txt", "txt_X_txt")}
MLP_SETTINGS = {"hidden": 768, "activation": "gelu", "epochs": 50, "early_stop": 10, "batch": 512,
                "lr_options": [1e-4, 5e-5], "weight_decay": 0.01, "eps": 1e-8, "seed": 0,
                "loss": "BCE with logits, pos_weight = negatives / positives",
                "selection": "balanced accuracy on a 15% stratified holdout of train"}


def text_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class ClipL14:
    """CLIP ViT-L/14 at its pinned revision and weights (textcheck.MODEL_PINS),
    fp32, on the laptop's GPU when there is one; images through
    match_images.clip_load, as the consistency block's CLIP reads them."""

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from scripts.textcheck import model_dir

        snap = model_dir("clip_l14")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(str(snap), local_files_only=True,
                                               dtype=torch.float32).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(str(snap), local_files_only=True)

    def config(self) -> dict[str, Any]:
        from scripts.image_text import software_versions
        from scripts.textcheck import MODEL_PINS

        pin = MODEL_PINS["clip_l14"]
        return {"model": pin["repo"], "revision": pin["revision"], "weights_sha256": pin["weights_sha256"],
                "dtype": "float32", "image": "match_images.clip_load, then the processor's 224",
                "text": "cut at CLIP's 77 tokens", "software": software_versions(("torch", "transformers", "pillow"))}

    def _norm(self, out):
        from scripts.clip_pairs import _features  # transformers 5 wraps the projection

        f = _features(out).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    def images(self, paths: Sequence[str]):
        from scripts.match_images import clip_load

        with self.torch.inference_mode():
            inputs = self.proc(images=[clip_load(p) for p in paths], return_tensors="pt").to(self.device)
            return self._norm(self.model.get_image_features(**inputs))

    def texts(self, texts: Sequence[str]):
        with self.torch.inference_mode():
            inputs = self.proc(text=list(texts), return_tensors="pt", padding=True, truncation=True,
                               max_length=77).to(self.device)
            return self._norm(self.model.get_text_features(**inputs))


def m4fc_titles(records: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """M4FC record key -> the titles of its unit's top five distinct documents
    (the text side's retrieval ledger). Refused while a record's unit has no row."""
    from scripts import retrieve
    from scripts import textcheck as tc

    cid = tc.ledger_config("m4fc", start=False)
    held = tc.read_retrieval(tc.retrieval_path("m4fc"), cid)
    out, missing = {}, 0
    for r in records:
        if r["dataset"] != "m4fc":
            continue
        before, _ = retrieve.evidence_bound("m4fc", r["row"].get("meta") or {})
        row = held.get(tc.unit_key(r["claim_key"], before))
        if row is None:
            missing += 1
            continue
        out[r["key"]] = [d.title for d in cs.documents_of(row)]
    if missing:
        raise SystemExit(f"m4fc: {missing:,} record(s) have no retrieval row: run the text side's "
                         "`retrieve --dataset m4fc` to the end first")
    return out


def muse_records() -> list[dict[str, Any]]:
    """Report C's records (consistency.c_records): M4FC train+val and VERITE's 877."""
    return cs.c_records()


def cmd_muse_embed(args) -> int:
    """CLIP L/14 vectors of every distinct image and text MUSE reads: written whole."""
    import numpy as np

    records = muse_records()
    titles = m4fc_titles(records) if args.with_evidence else {}
    images = sorted({(r["sha256"], r["path"]) for r in records if r["sha256"]})
    texts = sorted({r["claim"] for r in records} | {t for ts in titles.values() for t in ts})
    clip = ClipL14()
    print(f"CLIP L/14 on {clip.device}: {len(images):,} images, {len(texts):,} texts", flush=True)
    keys, vecs = [], []
    for i in range(0, len(images), CLIP_BATCH):
        part = images[i:i + CLIP_BATCH]
        vecs.append(clip.images([p for _, p in part]))
        keys += [f"img:{s}" for s, _ in part]
        print(f"  [muse-images] {min(i + CLIP_BATCH, len(images)):,}/{len(images):,}", flush=True)
    for i in range(0, len(texts), 64):
        part = texts[i:i + 64]
        vecs.append(clip.texts(part))
        keys += [f"txt:{text_sha(t)}" for t in part]
    MUSE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MUSE_EMB.with_name(MUSE_EMB.stem + ".part.npz")
    np.savez(tmp, keys=np.array(keys), vecs=np.concatenate(vecs).astype(np.float32))
    tmp.replace(MUSE_EMB)
    cs.write_json(MUSE_DIR / "clip_l14.config.json", {"config": clip.config(), "with_evidence": args.with_evidence,
                                                      "images": len(images), "texts": len(texts),
                                                      "written_at": utcnow()})
    print(f"-> {cs.rel(MUSE_EMB)}")
    return 0


def muse_features(records: Sequence[dict[str, Any]], titles: dict[str, list[str]]) -> None:
    """Add each record's similarities in place: img_txt for every record;
    img_X_txt and txt_X_txt (top-1 over its evidence titles, 0 with none) where
    titles are given."""
    import numpy as np

    emb = np.load(MUSE_EMB)
    vec = dict(zip(emb["keys"].tolist(), emb["vecs"]))
    for r in records:
        img, txt = vec[f"img:{r['sha256']}"], vec[f"txt:{text_sha(r['claim'])}"]
        r["img_txt"] = float(img @ txt)
        if r["key"] in titles:
            ev = [vec[f"txt:{text_sha(t)}"] for t in titles[r["key"]]]
            r["img_X_txt"] = max((float(img @ e) for e in ev), default=0.0)
            r["txt_X_txt"] = max((float(txt @ e) for e in ev), default=0.0)


def fit_mlp(x, y, holdout_x, holdout_y):
    """MUSE's MLP (2 x 768, GELU), as its repo sets it, selected by balanced
    accuracy on the holdout: (the fitted model's scorer, the chosen lr, its
    holdout balanced accuracy)."""
    import numpy as np
    import torch

    torch.use_deterministic_algorithms(True)

    def ba(scores, yy):
        pred = scores >= 0.5
        pos, neg = yy == 1, yy == 0
        return (pred[pos].mean() + (~pred[neg]).mean()) / 2

    best = None
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    xh = torch.tensor(holdout_x, dtype=torch.float32)
    for lr in MLP_SETTINGS["lr_options"]:
        torch.manual_seed(MLP_SETTINGS["seed"])
        h = MLP_SETTINGS["hidden"]
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], h), torch.nn.GELU(),
                                  torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, eps=MLP_SETTINGS["eps"],
                                weight_decay=MLP_SETTINGS["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / max(1, (y == 1).sum())))
        gen = torch.Generator().manual_seed(MLP_SETTINGS["seed"])
        state, best_lr_ba, stale = None, -1.0, 0
        for _ in range(MLP_SETTINGS["epochs"]):
            net.train()
            for idx in torch.randperm(len(xt), generator=gen).split(MLP_SETTINGS["batch"]):
                opt.zero_grad()
                loss_fn(net(xt[idx]).squeeze(1), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                score = torch.sigmoid(net(xh).squeeze(1)).numpy()
            b = ba(score, holdout_y)
            if b > best_lr_ba:
                best_lr_ba, stale = b, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= MLP_SETTINGS["early_stop"]:
                    break
        net.load_state_dict(state)
        if best is None or best_lr_ba > best[2]:
            best = (net, lr, best_lr_ba)
    net = best[0]

    def scorer(z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(z, dtype=torch.float32)).squeeze(1)).numpy()

    return scorer, best[1], round(float(best[2]), 4)


def auroc_ci(records: Sequence[dict[str, Any]], positive: str, negative: str, score: str,
             n: int = BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """One score's AUROC for ``positive`` against ``negative``, with a
    percentile 95% interval from resampling whole clusters (consistency.clusters_of:
    records sharing an image or a claim move together), as report C resamples."""
    part = [r for r in records if r["class"] in (positive, negative)]
    clusters = cs.clusters_of(part)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in part:
        groups[clusters[r["key"]]].append(r)

    def one(rs):
        return cs.auroc([r[score] for r in rs if r["class"] == positive],
                        [r[score] for r in rs if r["class"] == negative])

    value = one(part)
    rng, ids, vals = random.Random(seed), sorted(groups), []
    for _ in range(n):
        v = one([r for c in (rng.choice(ids) for _ in ids) for r in groups[c]])
        if v is not None:
            vals.append(v)
    vals.sort()
    ci = [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)] if vals else None
    return {"positive": positive, "negative": negative, "records": len(part), "clusters": len(ids),
            "auroc": None if value is None else round(value, 4), "ci95": ci, "resamples": len(vals)}


def muse_evaluate(records: Sequence[dict[str, Any]], score: str, threshold: float | None,
                  sets: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for name, recs in sets.items():
        pairs = ([(cs.OOC, "true")] if name.startswith("m4fc") else
                 [("out-of-context", "true"), ("miscaptioned", "true")])
        cell = {"auroc": [auroc_ci(recs, p, q, score) for p, q in pairs], "flagged": {}}
        if threshold is not None:
            classes = sorted({r["class"] for r in recs if r["class"] in {p for p, _ in pairs} | {"true"}})
            for cl in classes:
                rs = [r for r in recs if r["class"] == cl]
                cell["flagged"][cl] = cs.rate(sum(1 for r in rs if r[score] >= threshold), len(rs))
            for p, q in pairs:
                tpr = cell["flagged"][p]["rate"] if cell["flagged"].get(p) else None
                tnr = 1 - cell["flagged"]["true"]["rate"] if cell["flagged"].get("true") else None
                cell[f"balanced_accuracy_{'ooc' if p != 'miscaptioned' else 'miscaptioned'}"] = (
                    round((tpr + tnr) / 2, 4) if tpr is not None and tnr is not None else None)
        out[name] = cell
    return out


def cmd_muse_fit(args) -> int:
    """DT, RF and MLP on M4FC train (true against out-of-context), scored on
    M4FC val and VERITE (img only), with the threshold fitted on M4FC train."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    records = muse_records()
    cfg = json.loads((MUSE_DIR / "clip_l14.config.json").read_text(encoding="utf-8"))
    titles = m4fc_titles(records) if cfg["with_evidence"] else {}
    muse_features(records, titles)
    train = [r for r in records if r["dataset"] == "m4fc" and r["split"] == "train"
             and r["class"] in (cs.OOC, "true")]
    sets = {"m4fc_val": [r for r in records if r["dataset"] == "m4fc" and r["split"] == "val"],
            "verite": [r for r in records if r["dataset"] == "verite"]}
    rng = random.Random(SEED)
    hold = set()
    for cl in (cs.OOC, "true"):
        keys = sorted(r["key"] for r in train if r["class"] == cl)
        hold |= set(rng.sample(keys, max(1, round(0.15 * len(keys)))))
    section: dict[str, Any] = {
        "adaptation": "trained on M4FC train (NewsCLIPpings is not held); VERITE gets image-caption "
                      "similarity only (no date, no evidence); M4FC's evidence is the text side's dated "
                      "retrieval, its top five titles; no evidence image; class weights balanced",
        "encoder": cfg["config"], "train": {"true": sum(1 for r in train if r["class"] == "true"),
                                            "out_of_context": sum(1 for r in train if r["class"] == cs.OOC)},
        "dt": {"max_depth": 7, "random_state": 42, "class_weight": "balanced"},
        "rf": {"random_state": 42, "class_weight": "balanced", "n_estimators": "sklearn default (100)"},
        "mlp": MLP_SETTINGS, "variants": {}}
    plain = "clip_l14_cosine"
    for r in records:
        r[plain] = -r["img_txt"]  # lower similarity, more suspect: the raw signal MUSE-img learns from
    thr = cs.fit_threshold(train, score=plain)
    section["variants"][plain] = {"features": ["img_txt"], "classifier": "none (minus the cosine)",
                                  "threshold": thr, **muse_evaluate(records, plain, thr["threshold"], sets)}
    for variant, feats in FEATURES.items():
        if variant != "img" and not titles:
            continue
        x = np.array([[r[f] for f in feats] for r in train])
        y = np.array([1 if r["class"] == cs.OOC else 0 for r in train])
        fit_idx = np.array([r["key"] not in hold for r in train])
        models = {"dt": DecisionTreeClassifier(max_depth=7, random_state=42, class_weight="balanced").fit(x, y),
                  "rf": RandomForestClassifier(random_state=42, class_weight="balanced").fit(x, y)}
        mlp, lr, hold_ba = fit_mlp(x[fit_idx], y[fit_idx], x[~fit_idx], y[~fit_idx])
        for name, model in models.items():
            score = f"muse_{variant}_{name}"
            usable = [r for r in records if all(f in r for f in feats)]
            for r, s in zip(usable, model.predict_proba(np.array([[r[f] for f in feats] for r in usable]))[:, 1]):
                r[score] = float(s)
            thr = cs.fit_threshold(train, score=score)
            evaluate_on = {k: v for k, v in sets.items() if all(all(f in r for f in feats) for r in v)}
            section["variants"][score] = {"features": list(feats), "threshold": thr,
                                          **muse_evaluate(records, score, thr["threshold"], evaluate_on)}
        score = f"muse_{variant}_mlp"
        usable = [r for r in records if all(f in r for f in feats)]
        for r, s in zip(usable, mlp(np.array([[r[f] for f in feats] for r in usable]))):
            r[score] = float(s)
        thr = cs.fit_threshold(train, score=score)
        evaluate_on = {k: v for k, v in sets.items() if all(all(f in r for f in feats) for r in v)}
        section["variants"][score] = {"features": list(feats), "lr": lr, "holdout_balanced_accuracy": hold_ba,
                                      "threshold": thr, **muse_evaluate(records, score, thr["threshold"], evaluate_on)}
    write_section("muse", section)
    for name, v in section["variants"].items():
        cells = "; ".join(f"{k} " + ", ".join(f"{a['positive'][:14]} {a['auroc']} {a['ci95']}" for a in v[k]["auroc"])
                          for k in ("m4fc_val", "verite") if k in v)
        print(f"{name:34} {cells}")
    print(f"-> {cs.rel(REPORT)}")
    return 0


# --------------------------------------------------------------------------
# M4FC's official test: the baselines, read once (the person's go, 2026-11-12)
# --------------------------------------------------------------------------
#
# The frozen system's own read of the test is the consistency session's
# (`ensemble.py final-test`), and stays its only one. This reads the same
# 1,468 records once more, logged, for the baselines alone: MUSE as fitted on
# M4FC train, and the text-only classifier. Its inputs are the other session's
# live requests, read and never written: the claim, the image, and the
# retrieval its live path ran under the as-of rule.

FINAL_DIR = INTERIM / "ensemble" / "final_test"  # the consistency session's: read only
FINAL_KEYS = FINAL_DIR / "keys.json"
FINAL_SCORED = FINAL_DIR / "scored.json"
SYSTEM_FINAL_REPORT = REPORTS / "ensemble_final_test.json"
TEST_BASELINES = INTERIM / "textcheck" / "test_baselines.jsonl"
TEST_READ_LOG = INTERIM / "textcheck" / "test_baselines.read.json"
TEST_EMB = MUSE_DIR / "clip_l14_test.npz"
M4FC_FALSE = ("false (out-of-context)", "false (manipulated/fake)", "false (ai-generated)")
#: M4FC's verdict task, as the system was scored: true against false; mixture
#: and unverified are not scored.
VP_CLASSES = ("true", *M4FC_FALSE)
TEST_PAIRS = (("false", "true"), (cs.OOC, "true"))


def official_test_rows():
    """M4FC's official test rows, from the interim layer: the consistency
    session's reader (ensemble.test_rows), reproduced. Called once, by the read."""
    from scripts.build_processed import INTERIM_RECORDS

    with gzip.open(INTERIM_RECORDS / "m4fc.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["official_split"] == "test":
                yield row


def fit_muse_models(train: Sequence[dict[str, Any]], with_evidence: bool) -> dict[str, tuple[tuple[str, ...], Any]]:
    """MUSE's DT, RF and MLP per variant, fitted on M4FC train exactly as
    muse-fit fits them (the same holdout draw, seeds and settings): score name
    -> (its features, a function from a feature matrix to P(out-of-context))."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = random.Random(SEED)
    hold = set()
    for cl in (cs.OOC, "true"):
        keys = sorted(r["key"] for r in train if r["class"] == cl)
        hold |= set(rng.sample(keys, max(1, round(0.15 * len(keys)))))
    out: dict[str, tuple[tuple[str, ...], Any]] = {}
    for variant, feats in FEATURES.items():
        if variant != "img" and not with_evidence:
            continue
        x = np.array([[r[f] for f in feats] for r in train])
        y = np.array([1 if r["class"] == cs.OOC else 0 for r in train])
        fit_idx = np.array([r["key"] not in hold for r in train])
        dt = DecisionTreeClassifier(max_depth=7, random_state=42, class_weight="balanced").fit(x, y)
        rf = RandomForestClassifier(random_state=42, class_weight="balanced").fit(x, y)
        mlp, _, _ = fit_mlp(x[fit_idx], y[fit_idx], x[~fit_idx], y[~fit_idx])
        out[f"muse_{variant}_dt"] = (feats, lambda z, m=dt: m.predict_proba(z)[:, 1])
        out[f"muse_{variant}_rf"] = (feats, lambda z, m=rf: m.predict_proba(z)[:, 1])
        out[f"muse_{variant}_mlp"] = (feats, mlp)
    return out


def test_inputs() -> list[dict[str, Any]]:
    """Every live test request: its claim, image and top documents' titles
    (consistency.documents_of over the retrieval its live path ran), by live key."""
    out = []
    for k in sorted(set(json.loads(FINAL_KEYS.read_text(encoding="utf-8")).values())):
        req = cs._read_live(k, "request.json")
        ret = req.get("retrieval")
        docs = cs.documents_of(ret) if ret and not ret.get("error") else ()
        out.append({"live": k, "claim": req["claim"], "claim_key": cs.claim_key(req["claim"]),
                    "sha256": req["sha256"], "path": cs.rel(cs.LIVE_DIR / k / req["image"]),
                    "titles": [d.title for d in docs]})
    return out


def embed_test(items: Sequence[dict[str, Any]]) -> None:
    """CLIP L/14 vectors of the test's images, claims and titles, as muse-embed makes them."""
    import numpy as np

    images = sorted({(it["sha256"], it["path"]) for it in items})
    texts = sorted({it["claim"] for it in items} | {t for it in items for t in it["titles"]})
    clip = ClipL14()
    print(f"CLIP L/14 on {clip.device}: {len(images):,} test images, {len(texts):,} texts", flush=True)
    keys, vecs = [], []
    for i in range(0, len(images), CLIP_BATCH):
        part = images[i:i + CLIP_BATCH]
        vecs.append(clip.images([p for _, p in part]))
        keys += [f"img:{s}" for s, _ in part]
    for i in range(0, len(texts), 64):
        part = texts[i:i + 64]
        vecs.append(clip.texts(part))
        keys += [f"txt:{text_sha(t)}" for t in part]
    tmp = TEST_EMB.with_name(TEST_EMB.stem + ".part.npz")
    np.savez(tmp, keys=np.array(keys), vecs=np.concatenate(vecs).astype(np.float32))
    tmp.replace(TEST_EMB)


def text_only_p_false(texts: Sequence[str]) -> list[float]:
    """The text-only classifier (m4fc_audit's TF-IDF + logistic regression,
    fitted on M4FC train, three-way): P(refuted) per claim."""
    import numpy as np

    from scripts.fakeddit_classes import tokens
    from scripts.m4fc_audit import Tfidf, fit_logreg

    train = [(r["text"] or "", THREE_WAY["m4fc"][r["label"]]) for r in cs.rows("m4fc", ("train",))
             if r.get("label") in THREE_WAY["m4fc"]]
    labels = sorted({y for _, y in train})
    vec = Tfidf().fit([tokens(t) for t, _ in train])
    w, b, _ = fit_logreg(vec.transform([tokens(t) for t, _ in train]), [labels.index(y) for _, y in train],
                         len(labels))
    z = np.asarray(vec.transform([tokens(t) for t in texts]) @ w + b)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return [float(x) for x in p[:, labels.index("refuted")]]


def cmd_test_baselines(args) -> int:
    import numpy as np

    if TEST_READ_LOG.exists():
        raise SystemExit(f"the baselines read M4FC's official test at "
                         f"{json.loads(TEST_READ_LOG.read_text(encoding='utf-8'))['read_at']}: it is read once")
    if not FINAL_SCORED.exists():
        raise SystemExit("the frozen system has not read the official test: its read comes first")
    # -- label-free: inputs, features, the frozen baselines ----------------------
    items = test_inputs()
    if not TEST_EMB.exists() or args.reembed:
        embed_test(items)
    records, train, titles, _cfg = muse_train_set()
    models = fit_muse_models(train, with_evidence=bool(titles))
    published = json.loads(REPORT.read_text(encoding="utf-8"))["muse"]["variants"]
    val = [r for r in records if r["dataset"] == "m4fc" and r["split"] == "val"]
    for name, (feats, fn) in models.items():
        for r, s in zip(val, fn(np.array([[r[f] for f in feats] for r in val]))):
            r[name] = float(s)
        got = auroc_ci(val, cs.OOC, "true", name, n=1)["auroc"]
        want = published[name]["m4fc_val"]["auroc"][0]["auroc"]
        if got != want:
            raise SystemExit(f"{name}: refitted it scores M4FC val {got}, the published {want}: not the "
                             "fitted model; refusing")
    emb = np.load(TEST_EMB)
    vec = dict(zip(emb["keys"].tolist(), emb["vecs"]))
    for it in items:
        img, txt = vec[f"img:{it['sha256']}"], vec[f"txt:{text_sha(it['claim'])}"]
        ev = [vec[f"txt:{text_sha(t)}"] for t in it["titles"]]
        it["img_txt"] = float(img @ txt)
        it["img_X_txt"] = max((float(img @ e) for e in ev), default=0.0)
        it["txt_X_txt"] = max((float(txt @ e) for e in ev), default=0.0)
        it["clip_l14_cosine"] = -it["img_txt"]
    for name, (feats, fn) in models.items():
        for it, s in zip(items, fn(np.array([[it[f] for f in feats] for it in items]))):
            it[name] = float(s)
    for it, p in zip(items, text_only_p_false([it["claim"] for it in items])):
        it["text_only_p_false"] = p
    scores = ["clip_l14_cosine", *models, "text_only_p_false"]
    # -- the one read of the labels, logged at once --------------------------------
    keys = json.loads(FINAL_KEYS.read_text(encoding="utf-8"))
    by_live = {it["live"]: it for it in items}
    rows_ = []
    for row in official_test_rows():
        rk = cs.record_key("m4fc", row)
        k = keys.get(rk)
        if k is None or row["meta"]["verdict"] not in VP_CLASSES:
            continue
        it = by_live[k]
        rows_.append({"key": rk, "live": k, "sha256": it["sha256"], "claim_key": it["claim_key"],
                      "class": row["meta"]["verdict"], **{s: it[s] for s in scores}})
    system_scored = json.loads(FINAL_SCORED.read_text(encoding="utf-8"))
    log = {"read_at": utcnow(), "what": "M4FC's official test verdicts (meta.verdict), once, for the "
                                          "baselines; the frozen system's own read stays its only one",
           "records": len(rows_), "classes": cs.ordered(Counter(r["class"] for r in rows_)),
           "system_read": system_scored, "code_commit": _git_head()}
    cs.write_json(TEST_READ_LOG, log)
    # -- per-record scores (no label), and the aggregates --------------------------
    if len({r["claim_key"] for r in rows_}) != len(rows_):
        print("note: claim keys repeat among the test records; each row also carries its record key")
    TEST_BASELINES.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps({"claim_key": r["claim_key"], "key": r["key"], "live": r["live"],
                               **{s: round(r[s], 6) for s in scores}}) + "\n"
                   for r in sorted(rows_, key=lambda r: (r["claim_key"], r["key"])))
    tmp = TEST_BASELINES.with_name(TEST_BASELINES.name + ".part")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(TEST_BASELINES)
    system = json.loads(SYSTEM_FINAL_REPORT.read_text(encoding="utf-8")) if SYSTEM_FINAL_REPORT.exists() else {}
    section: dict[str, Any] = {"records": len(rows_), "classes": log["classes"], "read": log["read_at"],
                               "system_sha256": system_scored["system_sha256"], "auroc": {}}
    for pos, neg in TEST_PAIRS:
        part = [dict(r, **{"class": "false" if pos == "false" and r["class"] in M4FC_FALSE else r["class"]})
                for r in rows_]
        cell = {}
        for s in scores:
            one = auroc_ci(part, pos, neg, s)
            cell[s] = {**one, "vs_clip_l14": (cs.paired_auroc(part, pos, neg, s, "clip_l14_cosine")
                                              if s != "clip_l14_cosine" else None)}
        cell["system_routed_from_its_report"] = next(
            (a.get("auroc_routed") for a in system.get("auroc", []) if a.get("system") == "routed"
             and a.get("positive") == pos and a.get("negative") == neg), None)
        section["auroc"][f"{pos} vs {neg}"] = cell
    write_section("m4fc_official_test", section)
    log.update(test_baselines_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
               report_written_at=utcnow())
    cs.write_json(TEST_READ_LOG, log)
    for pair, cell in section["auroc"].items():
        print(f"== {pair}  (the system, routed: {cell['system_routed_from_its_report']})")
        for s in scores:
            c = cell[s]
            d = c["vs_clip_l14"]
            print(f"  {s:30} AUROC {c['auroc']} {c['ci95']}" + (f"  vs CLIP L/14 {d['delta']} {d['delta_ci95']}"
                                                                if d else ""))
    print(f"-> {cs.rel(TEST_BASELINES)} ({len(rows_):,} records); {cs.rel(REPORT)}; logged {cs.rel(TEST_READ_LOG)}")
    return 0


# --------------------------------------------------------------------------
# How much of the text-only classifier's M4FC score is the fact-checker?
# (the person's audit, 2026-11-12: train and val only, no test read)
# --------------------------------------------------------------------------

AUDIT_FOLDS = 5
#: An org is large enough to score within when it has at least this many true
#: and this many false records, out of sample, over train and val.
LARGE_ORG_MIN = 10


def m4fc_audit_records() -> list[dict[str, Any]]:
    """M4FC train and val, false or true: claim, fact-checker (processed meta),
    claim language (raw M4FC.json, the only place it is kept: the file holds
    every split, and is filtered to these records' ids on load, before any
    field is used)."""
    from scripts.m4fc_audit import raw_rows

    rows_ = [r for r in cs.rows("m4fc", ("train", "val")) if r["label"] in ("false", "true")]
    ids = {r["record_id"] for r in rows_}
    lang = {r["image_path"]: (r.get("claim_language"), r.get("fc_language")) for r in raw_rows()
            if r["image_path"] in ids}
    if len(lang) != len(ids):
        raise SystemExit(f"m4fc: {len(ids) - len(lang)} train/val record(s) are not in the raw file")
    return [{"key": r["record_id"], "split": r["split"], "text": r["text"] or "",
             "claim_key": cs.claim_key(r["text"] or ""), "class": r["label"],
             "org": (r["meta"] or {}).get("fc_org") or "unknown",
             "claim_language": lang[r["record_id"]][0] or "unknown",
             "fc_language": lang[r["record_id"]][1] or "unknown"} for r in rows_]


def lookup_scores(train: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]], field: str) -> list[float]:
    """P(false | the record's ``field``) from ``train``; a value train never saw
    gets train's overall false rate."""
    by: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        by[r[field]][r["class"]] += 1
    overall = sum(1 for r in train if r["class"] == "false") / len(train)
    return [(by[r[field]]["false"] / sum(by[r[field]].values())) if r[field] in by else overall for r in scored]


def text_p_false(train: Sequence[dict[str, Any]], scored: Sequence[dict[str, Any]]) -> list[float]:
    """The text-only classifier's P(false), fitted on ``train`` (binary here:
    the audit's records are false or true)."""
    import numpy as np

    from scripts.fakeddit_classes import tokens
    from scripts.m4fc_audit import Tfidf, fit_logreg

    labels = ["false", "true"]
    vec = Tfidf().fit([tokens(r["text"]) for r in train])
    w, b, _ = fit_logreg(vec.transform([tokens(r["text"]) for r in train]),
                         [labels.index(r["class"]) for r in train], 2)
    z = np.asarray(vec.transform([tokens(r["text"]) for r in scored]) @ w + b)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return [float(x) for x in (p / p.sum(axis=1, keepdims=True))[:, 0]]


SOURCE_SCORERS = {"text_only": text_p_false,
                  "org_only": lambda tr, sc: lookup_scores(tr, sc, "org"),
                  "language_only": lambda tr, sc: lookup_scores(tr, sc, "claim_language"),
                  "fc_language_only": lambda tr, sc: lookup_scores(tr, sc, "fc_language")}


def out_of_sample(records: Sequence[dict[str, Any]]) -> None:
    """Every scorer's score on every record, never from a fit it helped make:
    train by 5-fold cross-validation within train, val from all of train."""
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    fold = {i: pos % AUDIT_FOLDS for pos, i in enumerate(order)}
    for name, fn in SOURCE_SCORERS.items():
        for f in range(AUDIT_FOLDS):
            held = [train[i] for i in range(len(train)) if fold[i] == f]
            for r, s in zip(held, fn([train[i] for i in range(len(train)) if fold[i] != f], held)):
                r[name] = s
        for r, s in zip(val, fn(train, val)):
            r[name] = s


def within_group_auroc(records: Sequence[dict[str, Any]], score: str, group: str,
                       n: int = BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """AUROC over (false, true) pairs from the SAME group only: what a score
    knows beyond telling the groups apart. Interval from resampling records."""
    def value(rs):
        by: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
        for r in rs:
            by[r[group]][0 if r["class"] == "false" else 1].append(r[score])
        num = den = 0.0
        for pos, neg in by.values():
            if pos and neg:
                a = cs.auroc(pos, neg)
                num += a * len(pos) * len(neg)
                den += len(pos) * len(neg)
        return num / den if den else None

    point = value(records)
    rng = random.Random(seed)
    vals = sorted(v for v in (value([rng.choice(records) for _ in records]) for _ in range(n)) if v is not None)
    ci = [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)] if vals else None
    return {"auroc": None if point is None else round(point, 4), "ci95": ci, "records": len(records)}


def source_audit() -> dict[str, Any]:
    records = m4fc_audit_records()
    out_of_sample(records)
    parts = {"train (5-fold CV)": [r for r in records if r["split"] == "train"],
             "val (fitted on train)": [r for r in records if r["split"] == "val"]}
    parts["train + val (out of sample)"] = records
    out: dict[str, Any] = {"records": {k: cs.ordered(Counter(r["class"] for r in v)) for k, v in parts.items()},
                           "auroc_false_vs_true": {}, "within_org_text_only": {}, "by_org": {},
                           "true_share_by_org": {}}
    for name, rs in parts.items():
        out["auroc_false_vs_true"][name] = {s: auroc_ci(rs, "false", "true", s) for s in SOURCE_SCORERS}
        out["within_org_text_only"][name] = within_group_auroc(rs, "text_only", "org")
    for org in sorted({r["org"] for r in records}):
        rs = [r for r in records if r["org"] == org]
        n_true, n_false = sum(1 for r in rs if r["class"] == "true"), sum(1 for r in rs if r["class"] == "false")
        out["true_share_by_org"][org] = {
            s: cs.rate(sum(1 for r in rs if r["split"] == s and r["class"] == "true"),
                       sum(1 for r in rs if r["split"] == s)) for s in ("train", "val")}
        if n_true >= LARGE_ORG_MIN and n_false >= LARGE_ORG_MIN:
            out["by_org"][org] = {"true": n_true, "false": n_false,
                                  "text_only": auroc_ci(rs, "false", "true", "text_only")}
    out["large_org_rule"] = (f"at least {LARGE_ORG_MIN} true and {LARGE_ORG_MIN} false records, out of sample, "
                             "over train and val")
    out["language_source"] = ("claim_language and fc_language from raw M4FC.json, filtered to the train and val "
                              "records' ids on load; no field of a test row is used")
    return out


def cmd_source_audit(args) -> int:
    rep = source_audit()
    write_section("m4fc_source_audit", rep)
    for part, cell in rep["auroc_false_vs_true"].items():
        print(f"== {part} {json.dumps(rep['records'][part])}")
        for s, a in cell.items():
            print(f"  {s:18} AUROC {a['auroc']} {a['ci95']}")
        w = rep["within_org_text_only"][part]
        print(f"  {'text_only within org':18} AUROC {w['auroc']} {w['ci95']}")
    print("== text-only within each large org (train + val, out of sample)")
    for org, c in rep["by_org"].items():
        print(f"  {org:24} true {c['true']:>3} false {c['false']:>4}  AUROC {c['text_only']['auroc']} "
              f"{c['text_only']['ci95']}")
    print("== share of true claims per org: train | val")
    for org, c in rep["true_share_by_org"].items():
        print(f"  {org:24} {c['train']['k']:>3}/{c['train']['n']:<4} {c['train']['rate']}  |  "
              f"{c['val']['k']:>3}/{c['val']['n']:<4} {c['val']['rate']}")
    print(f"-> {cs.rel(REPORT)}")
    return 0


def muse_train_set():
    """Report C's records with their MUSE features, and M4FC train's true and
    out-of-context records: what muse-fit fits on."""
    records = muse_records()
    cfg = json.loads((MUSE_DIR / "clip_l14.config.json").read_text(encoding="utf-8"))
    titles = m4fc_titles(records) if cfg["with_evidence"] else {}
    muse_features(records, titles)
    train = [r for r in records if r["dataset"] == "m4fc" and r["split"] == "train" and r["class"] in (cs.OOC, "true")]
    return records, train, titles, cfg


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hostname", help="Factify2's hostname-only baseline, per class")
    p.set_defaults(func=cmd_hostname)
    p = sub.add_parser("text-only", help="TF-IDF + logistic regression on the claim")
    p.add_argument("--only", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.set_defaults(func=cmd_text_only)
    p = sub.add_parser("muse-embed", help="CLIP L/14 vectors of MUSE's images and texts (GPU)")
    p.add_argument("--with-evidence", action="store_true", help="also M4FC's evidence titles")
    p.set_defaults(func=cmd_muse_embed)
    p = sub.add_parser("muse-fit", help="MUSE's DT, RF and MLP; scored on M4FC val and VERITE")
    p.set_defaults(func=cmd_muse_fit)
    p = sub.add_parser("source-audit", help="M4FC train and val: how much of the text-only score is the "
                                            "fact-checker (no test read)")
    p.set_defaults(func=cmd_source_audit)
    p = sub.add_parser("test-baselines", help="M4FC's official test, ONCE, logged: MUSE as fitted and "
                                              "the text-only classifier on the frozen system's 1,468 records")
    p.add_argument("--reembed", action="store_true", help="recompute the test's CLIP vectors")
    p.set_defaults(func=cmd_test_baselines)
    args = ap.parse_args(argv)
    if getattr(args, "only", None):
        unknown = sorted(set(args.only) - set(TEXT_ONLY))
        if unknown:
            ap.error(f"unknown: {unknown}; known: {sorted(TEXT_ONLY)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
