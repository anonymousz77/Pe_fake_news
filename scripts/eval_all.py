#!/usr/bin/env python
"""Every required number from every block, in one file: data/reports/eval_all.json.

    python scripts/eval_all.py                   # collect, check the rules, write, print
    python scripts/eval_all.py --allow-missing   # write even while a source is missing (listed)

It recomputes nothing a block reports: each number is READ from the report
that block writes (its sha256 recorded beside it), by a path in ``REQUIRED``.
Two exceptions, computed here because no report holds them: image-to-text's
rates come from its own ledgers through image_text's readers, and VERITE's
evaluation count from consistency.evaluation_rows.

The project's reporting rules are enforced on what is written:

* every Factify2 number stands beside the hostname-only baseline and the
  applicable majority, and a model's Factify2 result comes per class;
* every VERITE number carries the count it was taken on (877: the 37
  cross-corpus records dropped);
* no ISOT number is a headline, and a DGM4-trained number scored on VERITE
  needs its transfer gap (neither is collected today; the checks stand);
* NLI model A is a contrast, never a system number.

A missing source or key is listed under ``missing`` with its reason, never
skipped silently; the run then exits 1 unless --allow-missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, REPORTS  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402

OUT = REPORTS / "eval_all.json"
IMAGE_TEXT_CONFIG = cs.IMAGE_TEXT_CONFIG
IMAGE_TEXT_DATASETS = ("verite", "averimatec", "factify2", "m4fc")
VERITE_EVALUATED = 877


class Missing(Exception):
    """A required number whose source or key is not there."""


def sel(field: str, value: Any) -> tuple[str, str, Any]:
    """A path step into a list: the element whose ``field`` is ``value``."""
    return ("select", field, value)


#: (block, name, report file, path, tags). Tags: "factify2" (needs its
#: baselines beside), "verite" (needs its count), "contrast" (never a system
#: number), "headline" (a result the write-up leads with).
REQUIRED: list[tuple[str, str, str, tuple, tuple[str, ...]]] = [
    # -- the data layer ---------------------------------------------------
    *[("data_layer", f"{ds}_usable_of_rows", "completion.json", ("datasets", ds), ())
      for ds in ("mocheg", "factify2", "averitec", "averimatec", "verite", "fakeddit", "welfake", "liar",
                 "isot", "fakenewsnet", "dgm4", "m4fc")],
    ("data_layer", "factify2_per_class", "completion.json", ("datasets", "factify2", "per_class"),
     ("factify2_limitation",)),
    ("data_layer", "factify2_val_classes", "factify2_domain_only_final.json", ("val_class_counts",),
     ("factify2_limitation",)),
    # -- the gallery --------------------------------------------------------
    ("gallery", "threshold", "gallery_calibration.json", ("sface", "adjudicated", "threshold"), ()),
    ("gallery", "at_threshold", "gallery_calibration.json", ("sface", "adjudicated", "at_threshold"),
     ("headline",)),
    ("gallery", "tar_bootstrap_95", "gallery_calibration.json",
     ("sface", "adjudicated", "bootstrap_95", "tar"), ()),
    ("gallery", "coverage", "gallery_calibration.json", ("sface", "coverage"), ()),
    # -- the evidence store and retrieval -------------------------------------
    ("evidence_store", "redtest_passed", "redtest.json", ("passed",), ()),
    ("evidence_store", "redtest_parity", "redtest.json", ("probe_parity",), ()),
    ("evidence_store", "index_byte_identical", "redtest.json", ("index_byte_identical",), ()),
    ("evidence_store", "default_seconds", "retrieval_bench.json", ("default", "seconds"), ()),
    ("evidence_store", "default_overlap_at_20", "retrieval_bench.json", ("default", "overlap_at_20"), ()),
    # -- the consistency block ------------------------------------------------
    ("consistency", "gate_a_reviewed", "consistency_a.json", ("review", "gate"), ()),
    ("consistency", "gate_a_cleared", "consistency_c.json", ("gate_a_cleared",), ()),
    ("consistency", "m4fc_val_primary_vs_clip", "consistency_c.json",
     ("corpora", "m4fc:extracted:primary", "auroc", sel("positive", "false (out-of-context)")), ("headline",)),
    ("consistency", "verite_ooc_primary_vs_clip", "consistency_c.json",
     ("corpora", "verite:extracted:primary", "auroc", sel("positive", "out-of-context")), ("headline", "verite")),
    ("consistency", "verite_miscaptioned_primary_vs_clip", "consistency_c.json",
     ("corpora", "verite:extracted:primary", "auroc", sel("positive", "miscaptioned")), ("headline", "verite")),
    ("consistency", "thresholds_primary", "consistency_c.json", ("thresholds", "extracted:primary"), ()),
    ("consistency", "fused", "consistency_c.json", ("fused", "beats_clip"), ()),
    # -- Track B: baselines ---------------------------------------------------
    ("baselines", "hostname_factify2_by_pair", "baselines.json", ("hostname_factify2", "by_pair"),
     ("factify2_baseline",)),
    ("baselines", "hostname_factify2_by_record", "baselines.json", ("hostname_factify2", "by_record"),
     ("factify2_baseline",)),
    *[("baselines", f"text_only_{name}", "baselines.json", (f"text_only_{name}",),
       ("factify2",) if name == "factify2" else ())
      for name in ("averitec", "averitec_native", "averimatec", "m4fc", "mocheg", "liar", "factify2")],
    ("baselines", "muse", "baselines.json", ("muse", "variants"), ("verite",)),
    # -- Track B: the text-side check -----------------------------------------
    ("text_side", "system", "textcheck.json", ("system",), ("headline",)),
    ("text_side", "contrast_nli_a", "textcheck.json", ("contrasts", "nli_a"), ("contrast",)),
    ("text_side", "retrieval_coverage", "textcheck.json", ("coverage",), ()),
    ("text_side", "verdicts_file", "textcheck.json", ("verdicts_file",), ()),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(doc: Any, path: Sequence[Any]) -> Any:
    """The value at ``path``; Missing names the first step that is not there."""
    cur = doc
    for i, step in enumerate(path):
        if isinstance(step, tuple) and step[0] == "select":
            _, field, value = step
            if not isinstance(cur, list):
                raise Missing(f"step {i}: not a list")
            hits = [x for x in cur if isinstance(x, dict) and x.get(field) == value]
            if len(hits) != 1:
                raise Missing(f"step {i}: {len(hits)} element(s) with {field} = {value!r}")
            cur = hits[0]
        else:
            if not isinstance(cur, dict) or step not in cur:
                raise Missing(f"no key {step!r} at step {i}")
            cur = cur[step]
    return cur


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------
# what no report holds, computed by the block's own readers
# --------------------------------------------------------------------------


def image_text_rates() -> dict[str, Any]:
    """Per corpus ledger of the evaluated image_text config: images, statuses,
    and the entity answers where the skeleton forced a line -- any line, and
    the ORG line (the answer carried past PERSON)."""
    from collections import Counter

    from scripts.image_text import CATEGORIES, OUT_ROOT, scan_ledger

    out: dict[str, Any] = {}
    for ds in IMAGE_TEXT_DATASETS:
        ledger = OUT_ROOT / IMAGE_TEXT_CONFIG / f"{ds}.jsonl"
        if not ledger.is_file():
            out[ds] = {"missing": f"{cs.rel(ledger)} is not here"}
            continue
        forced: dict[str, tuple[str, ...] | None] = {}

        def keep(r: dict[str, Any]) -> bool:
            block = r.get("entities")
            if block is None:
                forced.pop(r["sha256"], None)
            elif all("forced" in block[c] for c in CATEGORIES):
                forced[r["sha256"]] = tuple(c for c in CATEGORIES if block[c]["forced"])
            else:
                forced[r["sha256"]] = None
            return False

        summaries, _, _ = scan_ledger(ledger, schema=False, keep=keep)
        marked = [v for v in forced.values() if v is not None]
        out[ds] = {"images": len(summaries),
                   "status": dict(Counter(s["status"] for s in summaries.values())),
                   "answers_marked": len(marked),
                   "any_line_forced": cs.rate(sum(1 for v in marked if v), len(marked)),
                   "org_line_forced": cs.rate(sum(1 for v in marked if "ORG" in v), len(marked))}
    return {"config": IMAGE_TEXT_CONFIG, "ledgers": out}


def verite_counts() -> dict[str, Any]:
    from scripts.processed_loader import iter_rows

    usable = sum(1 for r in iter_rows("verite") if r["usable"])
    evaluated = len(cs.evaluation_rows())
    return {"usable": usable, "evaluated": evaluated, "cross_corpus_dropped": usable - evaluated}


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------


def rule_problems(entries: dict[str, dict[str, Any]], verite: dict[str, Any]) -> list[str]:
    out = []
    by_tag: dict[str, list[str]] = {}
    for key, e in entries.items():
        for t in e["tags"]:
            by_tag.setdefault(t, []).append(key)
    baselines_present = all(entries.get(k, {}).get("value") is not None for k in
                            ("baselines.hostname_factify2_by_pair", "baselines.hostname_factify2_by_record"))
    for key in by_tag.get("factify2", []):
        e = entries[key]
        if e.get("value") is None:
            continue
        if not baselines_present:
            out.append(f"{key}: a Factify2 number without the hostname baseline beside it")
        model = (e["value"] or {}).get("model") or e["value"]
        if "per_class" not in model:
            out.append(f"{key}: a Factify2 result that is not per class")
        if "applicable_majority" not in e["value"]:
            out.append(f"{key}: a Factify2 number without its applicable majority")
    if verite.get("evaluated") != VERITE_EVALUATED:
        out.append(f"VERITE is evaluated on {verite.get('evaluated')} rows, not {VERITE_EVALUATED}")
    for key, e in entries.items():
        text = json.dumps(e.get("value"), ensure_ascii=False).casefold()
        if "headline" in e["tags"] and ("isot" in key.casefold() or '"isot"' in text):
            out.append(f"{key}: an ISOT number as a headline (ISOT is a negative control)")
        if '"trained_on": "dgm4' in text and "verite" in text and "transfer_gap" not in text:
            out.append(f"{key}: a DGM4-trained number on VERITE without its transfer gap")
        if "contrast" not in e["tags"] and ("nli_a" in text or "moritzlaurer" in text):
            out.append(f"{key}: NLI model A outside a contrast")
    return out


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def collect(reports: Path = REPORTS) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    docs: dict[str, Any] = {}
    entries: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for block, name, fname, path, tags in REQUIRED:
        key = f"{block}.{name}"
        src = reports / fname
        if fname not in docs:
            if src.is_file():
                docs[fname] = json.loads(src.read_text(encoding="utf-8"))
                sources[fname] = {"sha256": sha256_of(src),
                                  "written_at": docs[fname].get("written_at")
                                  if isinstance(docs[fname], dict) else None}
            else:
                docs[fname] = None
        if docs[fname] is None:
            missing.append({"number": key, "source": fname, "reason": "the report is not there"})
            entries[key] = {"block": block, "source": fname, "tags": list(tags), "value": None}
            continue
        try:
            value = resolve(docs[fname], path)
        except Missing as exc:
            missing.append({"number": key, "source": fname, "reason": str(exc)})
            value = None
        entries[key] = {"block": block, "source": fname,
                        "path": [list(p) if isinstance(p, tuple) else p for p in path],
                        "tags": list(tags), "value": value}
    computed = {"image_to_text": image_text_rates(), "verite_counts": verite_counts()}
    for ds, cell in computed["image_to_text"]["ledgers"].items():
        if "missing" in cell:
            missing.append({"number": f"image_to_text.{ds}", "source": "ledger", "reason": cell["missing"]})
    problems = rule_problems(entries, computed["verite_counts"])
    blocks: dict[str, dict[str, Any]] = {}
    for key, e in entries.items():
        blocks.setdefault(e["block"], {})[key.split(".", 1)[1]] = e
    return {"written_at": utcnow(), "code_commit": git_commit(), "sources": sources,
            "computed": computed, "blocks": blocks, "rule_problems": problems, "missing": missing}


def headline_lines(report: dict[str, Any]) -> list[str]:
    """A compact table of what the write-up leads with."""
    lines = []
    for block, items in report["blocks"].items():
        for name, e in items.items():
            v = e["value"]
            if v is None:
                lines.append(f"  {block:14} {name:40} MISSING")
                continue
            if "factify2" in e["tags"] or "factify2_baseline" in e["tags"]:
                m = v.get("model", v)
                short = {"Insufficient_Multimodal": "IM", "Insufficient_Text": "IT", "Refute": "R",
                         "Support_Multimodal": "SM", "Support_Text": "ST"}
                f1 = ", ".join(f"{short.get(c, c)} {d['f1']['value']}" for c, d in m["per_class"].items())
                lines.append(f"  {block:14} {name:40} per-class F1: {f1} | accuracy "
                             f"{m['accuracy']['value']} beside majority "
                             f"{v.get('applicable_majority', {}).get('share')}")
                continue
            if isinstance(v, dict) and "auroc_score" in v:
                lines.append(f"  {block:14} {name:40} {v['auroc_score']} vs CLIP {v['auroc_clip']}, "
                             f"delta {v['delta']} {v['delta_ci95']} (n={v['records']})")
            elif isinstance(v, dict) and "model" in v and "accuracy" in (v["model"] or {}):
                m = v["model"]
                maj = v.get("applicable_majority", {})
                lines.append(f"  {block:14} {name:40} acc {m['accuracy']['value']} {m['accuracy']['ci']}, "
                             f"macro-F1 {m['macro_f1']['value']} | majority {maj.get('share')} (n={v['rows']})")
            elif isinstance(v, dict) and "accuracy" in v and "per_class" in v:
                maj = v.get("applicable_majority", {})
                lines.append(f"  {block:14} {name:40} acc {v['accuracy']['value']} {v['accuracy']['ci']} | "
                             f"majority {maj.get('share')}")
            elif isinstance(v, (int, float, str, bool)):
                lines.append(f"  {block:14} {name:40} {v}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)
    report = collect()
    if report["rule_problems"]:
        print("REFUSED: the reporting rules fail:\n  " + "\n  ".join(report["rule_problems"]))
        return 2
    if report["missing"] and not args.allow_missing:
        print("MISSING (nothing written; --allow-missing writes and lists them):")
        for m in report["missing"]:
            print(f"  {m['number']}: {m['source']} -- {m['reason']}")
        return 1
    cs.public_report(report, OUT)
    print("\n".join(headline_lines(report)))
    print(f"missing: {len(report['missing'])}; -> {cs.rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
