#!/usr/bin/env python
"""M4FC as a training source for media_mismatch / wrong_image: measured, not trained.

    python scripts/m4fc_audit.py measure            # sections 1-5 and 7: offline, light
    python scripts/m4fc_audit.py evidence --limit 3 # section 6: the timing probe first
    python scripts/m4fc_audit.py evidence           # the 200 claims; resumable
    python scripts/m4fc_audit.py report             # data/reports/m4fc_audit.{md,json}
    python scripts/m4fc_audit.py review-page        # review/review.html: the sheet, in a browser

Why raw. The processed layer keeps six of M4FC's 45 fields
(scripts/records.py, load_m4fc): image_path, claim, verdict_coarse, split,
fc_org and fc_pub_date. The three false subtypes, every claimed and actual
context item, the balanced setting's flag and the whole human-validated file
never reach it. So this audit reads them from data/raw, read-only, joined on
record_id = image_path. Carrying them into the processed layer would take an
interim and a processed rebuild: reported here, not done.

Splits, by the person's decisions of 2026-10-28:

* train and dev are the PROCESSED split, which is what
  ``reason_code_rows("m4fc", split="train")`` would train on. M4FC's own split
  stands beside every count.
* M4FC's official test split enters the counts and the image overlap as
  numbers only. No test text, classifier, query or sheet row touches it.

It writes only to data/interim/m4fc_audit/ (ignored) and to
data/reports/m4fc_audit.* (the .json tracked, aggregates only). It opens no
socket: the evidence store is local SQLite opened read-only, and the NER model
is the gallery's pinned local snapshot. The classifiers are measuring
instruments: settings fixed in advance, fit once, scored once, never saved.
Seed 20260901; 2,000 bootstrap resamples of the scored rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import inspect
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import (BIBLE, DOCS, INTERIM, PROJECT_ROOT, REPORTS,  # noqa: E402
                           dataset, dataset_names, raw_dir)

SEED = 20260901
BOOTSTRAP = 2000
NEI = "not enough information"
#: Values that mean "absent". M4FC writes NEI; two values say the same in other words.
ABSENT = frozenset({NEI, "unspecified", "not specified"})
#: Context items with a claimed_ twin: what the claim says against what the image is.
ITEMS = ("date", "location", "people", "things", "event")
OOC = "false (out-of-context)"
INDIA_ORGS = ("factly.in", "indiatoday.in", "dfrac.org", "youturn.in")
#: M4FC's cutoffs as the paper states them; section 2 checks them against fc_pub_date.
DEV_STARTS, TEST_STARTS, TEST_ENDS = "2022-09-22", "2023-01-18", "2024-09-01"
OFFICIAL_SPLITS = ("train", "dev", "test")
PROCESSED_SPLITS = ("train", "val", "test")
#: scripts/splits.DEFAULT_PURITY: a group at or above this is label-predictive.
PURITY_FLAG = 0.90
SHEET_INSTANCES, SHEET_CAP = 60, 15
EVIDENCE_N, EVIDENCE_K, ENTITY_TOP = 200, 20, 5
#: The text classifier's settings, fixed before any scored row was seen.
LR_C, MIN_DF = 1.0, 2

OUT = INTERIM / "m4fc_audit"
MEASURE = OUT / "measure.json"
EVIDENCE = OUT / "evidence.jsonl"
EVIDENCE_CONFIG = OUT / "evidence_config.json"
REVIEW = OUT / "review"
SHEET = REVIEW / "m4fc_ooc_items.csv"
SHEET_IMAGES = REVIEW / "images"
PRIVATE_MAP = REVIEW / "private_map.json"
REVIEW_PAGE = REVIEW / "review.html"
REPORT_MD = REPORTS / "m4fc_audit.md"
REPORT_JSON = REPORTS / "m4fc_audit.json"
SHEET_COLUMNS = ("sheet_id", "image", "claim", "item", "claimed", "actual", "mismatch", "note")
FORMAT_EXT = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif",
              "BMP": ".bmp", "TIFF": ".tif"}
#: Slug tokens (and token pairs) that state a fact-checker's finding.
SLUG_WORDS = frozenset({"fake", "false", "falsely", "old", "unrelated", "misleading", "edited",
                        "morphed", "doctored", "manipulated", "photoshopped", "fabricated",
                        "hoax", "misattributed"})
SLUG_PHRASES = (("not", "related"), ("passed", "off"), ("ai", "generated"), ("digitally", "altered"))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def ordered(counts: Counter | dict) -> dict[str, int]:
    """Largest first, ties by key: the same counts always serialise the same way."""
    return {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def day(row: dict[str, Any]) -> str:
    return str(row["fc_pub_date"])[:10]


def parts(value: Any) -> list[str]:
    """A value's parts: list elements, or the ';'-separated pieces of a string."""
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [p.strip() for p in str("" if value is None else value).split(";")]


def present(value: Any) -> bool:
    """Some part is non-empty and is not an absent marker."""
    return any(p and p.casefold() not in ABSENT for p in parts(value))


def both(row: dict[str, Any], item: str) -> bool:
    """The claim states this item AND the authors recorded what the image is."""
    return present(row.get(f"claimed_{item}")) and present(row.get(item))


def shown(value: Any) -> str:
    """A value as the sheet shows it: as written; a list joined with '; '."""
    return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)


_DATE_PART = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
_RANGE_PART = re.compile(r"^\d{4}\s*-\s*\d{4}$")


def date_shape(value: Any) -> str | None:
    """None when absent; 'normalised' when every stated part is YYYY, YYYY-MM or
    YYYY-MM-DD; 'range' when some part is a YYYY-YYYY span; else 'other'."""
    stated = [p for p in parts(value) if p and p.casefold() not in ABSENT]
    if not stated:
        return None
    if all(_DATE_PART.match(p) for p in stated):
        return "normalised"
    if all(_DATE_PART.match(p) or _RANGE_PART.match(p) for p in stated):
        return "range"
    return "other"


def scripts_of(text: str) -> Counter:
    """Letters by Unicode script (the first word of each letter's name)."""
    out: Counter = Counter()
    for ch in text or "":
        if ch.isalpha():
            try:
                out[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                out["UNNAMED"] += 1
    return out


def slug_states_finding(record_id: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", Path(record_id).stem.lower()) if t]
    if SLUG_WORDS & set(tokens):
        return True
    return any(tuple(tokens[i:i + 2]) in SLUG_PHRASES for i in range(len(tokens) - 1))


def raw_rows(name: str = "M4FC.json") -> list[dict[str, Any]]:
    return json.loads((raw_dir("m4fc") / "data" / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the corpus: raw rows joined to interim, processed and the resolved image
# --------------------------------------------------------------------------


class Corpus:
    """Raw M4FC, each row's processed split, and its image resolved exactly as
    the processed build resolves it (build_processed.Resolver, read-only)."""

    def __init__(self) -> None:
        from scripts import images
        from scripts.build_processed import Resolver, read_interim
        from scripts.processed_loader import iter_rows

        self.raw = raw_rows()
        self.validated = raw_rows("M4FC_human_validated.json")
        self.by_id = {r["image_path"]: r for r in self.raw}
        self.interim = {r["record_id"]: r for r in read_interim("m4fc")}
        self.processed = {r["record_id"]: r for r in iter_rows("m4fc")}
        resolver = Resolver(["m4fc"])
        index = resolver.index["m4fc"]
        self.image: dict[str, dict[str, Any]] = {}
        for rid, rec in self.interim.items():
            refs = rec.get("images") or []
            if not refs:
                self.image[rid] = {"status": "no_image_ref", "path": None, "sha256": None,
                                   "format": None}
                continue
            status, path, sha = resolver.resolve("m4fc", refs[0], rec.get("label"))
            self.image[rid] = {"status": status, "path": path, "sha256": sha,
                               "format": (index.get(path) or {}).get("format") if path else None}
        self.usable_status = images.USABLE

    def split(self, rid: str) -> str | None:
        """The processed split, or None for a row the register keeps out."""
        row = self.processed.get(rid)
        return row["split"] if row else None

    def usable(self, rid: str) -> bool:
        return self.image[rid]["status"] == self.usable_status

    def groups(self) -> dict[str, list[dict[str, Any]]]:
        """Rows by official split and by processed split, plus all of them."""
        out = {f"official {s}": [r for r in self.raw if r["split"] == s] for s in OFFICIAL_SPLITS}
        out.update({f"processed {s}": [r for r in self.raw if self.split(r["image_path"]) == s]
                    for s in PROCESSED_SPLITS})
        out["all"] = list(self.raw)
        return out


# --------------------------------------------------------------------------
# 1. definitions, and what the sanctioned loader yields today
# --------------------------------------------------------------------------


def definitions(code_counts: dict[str, Any]) -> dict[str, Any]:
    from scripts import processed_loader as pl
    from scripts import reason_codes as rc

    codes = ("media_mismatch", "wrong_image")
    doc = [ln.strip() for ln in (rc.__doc__ or "").splitlines() if ln.strip().startswith(codes)]
    schema = [ln for ln in (DOCS / "processed_schema.md").read_text(encoding="utf-8").splitlines()
              if ln.startswith(tuple(f"| `{c}`" for c in codes))]
    yields: dict[str, Any] = {}
    for name in sorted(n for n in dataset_names() if dataset(n).get("enabled")):
        if not pl.records_path(name).is_file():
            continue
        try:
            got = Counter(r["reason_code"] for r in
                          pl.reason_code_rows(name, split="train", purpose="train"))
            yields[name] = {"yielded": ordered(got)}
        except pl.SupervisionError as exc:
            yields[name] = {"refused": str(exc)}
    return {
        "reason_codes_docstring": doc,
        "processed_schema_rows": schema,
        "mapping": {c: [{"dataset": ds, "label": lab, "provenance": how}
                        for ds, m in sorted(rc.MAPPING.items())
                        for lab, (code, how) in m.items() if code == c] for c in codes},
        "m4fc_mapping": {lab: code for lab, (code, _how) in rc.MAPPING["m4fc"].items()},
        "dgm4": {"rule_source": inspect.getsource(rc.dgm4_reason_code),
                 "audit_passed": rc.DGM4_AUDIT_PASSED,
                 "label": rc.DGM4_MEDIA_MISMATCH_LABEL},
        "fakeddit_rulings": dict(rc.FAKEDDIT_RULINGS),
        "unmapped_datasets": sorted(rc.UNMAPPED_DATASETS),
        "evaluation_only": dict(rc.EVALUATION_ONLY),
        "loader_source": inspect.getsource(pl.reason_code_rows),
        "rows_carrying_the_codes": code_counts,
        "train_yield_of_reason_code_rows": yields,
        "trainable_rows": {c: sum(v.get("yielded", {}).get(c, 0) for v in yields.values())
                           for c in codes},
        "same_string_elsewhere": "images.WRONG_IMAGE == 'wrong_image' is also an image status: "
                                 "a file that is provably not the authors' image",
    }


# --------------------------------------------------------------------------
# one pass over the processed layer: the codes, and M4FC's images elsewhere
# --------------------------------------------------------------------------

_SHA_IN_LINE = re.compile(r'"sha256": "([0-9a-f]{64})"')


def shared_files(m4fc_sha: set[str]) -> dict[str, dict[str, str]]:
    """dataset -> {path: sha256} for every USABLE indexed file whose bytes are an
    M4FC image. Streams each index; only matching lines are parsed."""
    from scripts import images

    out: dict[str, dict[str, str]] = {}
    for name in images.IMAGE_DATASETS:
        if name == "m4fc" or not images.index_path(name).is_file():
            continue
        found: dict[str, str] = {}
        with images.index_path(name).open(encoding="utf-8") as handle:
            for line in handle:
                m = _SHA_IN_LINE.search(line)
                if not m or m.group(1) not in m4fc_sha:
                    continue
                row = json.loads(line)
                if images.status(row) == images.USABLE:
                    found[row["path"]] = row["sha256"]
        out[name] = found
    return out


def processed_scan(shared: dict[str, dict[str, str]]
                   ) -> tuple[dict[str, Any], dict[str, dict[str, set[tuple[str, str]]]]]:
    """(rows carrying media_mismatch / wrong_image, per dataset; partner records:
    dataset -> sha256 -> {(record_id, split)})."""
    from scripts.processed_loader import iter_rows, records_path

    codes: dict[str, Any] = {}
    partners: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for name in sorted(n for n in dataset_names() if dataset(n).get("enabled")):
        if not records_path(name).is_file():
            codes[name] = "no processed records"
            continue
        carried: Counter = Counter()
        paths = shared.get(name, {})
        hits: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in iter_rows(name):
            if row.get("reason_code") in ("media_mismatch", "wrong_image"):
                carried[(row["reason_code"], str(row.get("label")), row["split"],
                         "evaluation_only" if row.get("evaluation_only") else "trainable")] += 1
            if name != "m4fc":
                for p in row.get("image_paths") or []:
                    if p in paths:
                        hits[paths[p]].add((str(row["record_id"]), row["split"]))
        codes[name] = [{"reason_code": c, "label": lab, "split": s, "use": u, "rows": n}
                       for (c, lab, s, u), n in sorted(carried.items())]
        if name != "m4fc":
            partners[name] = dict(hits)
    return codes, partners


# --------------------------------------------------------------------------
# 2. fields
# --------------------------------------------------------------------------

#: Where each surviving raw field lands.
SURVIVES = {
    "image_path": "record_id",
    "claim": "text",
    "verdict_coarse": "label, and reason_code derived from it",
    "split": "meta.official_split (dev -> val); split, after reconciliation",
    "fc_org": "meta.fc_org",
    "fc_pub_date": "meta.fc_pub_date",
}


def fields_section(c: Corpus) -> dict[str, Any]:
    from scripts.records import SPLIT_ALIASES

    raw_keys = list(c.raw[0])
    one_interim = next(iter(c.interim.values()))
    one_processed = next(iter(c.processed.values()))
    checks: Counter = Counter()
    for rid, p in c.processed.items():
        r = c.by_id[rid]
        checks["text == claim"] += p["text"] == r["claim"]
        checks["label == verdict_coarse"] += p["label"] == r["verdict_coarse"]
        checks["meta.fc_org == fc_org"] += p["meta"].get("fc_org") == r["fc_org"]
        checks["meta.fc_pub_date == fc_pub_date"] += p["meta"].get("fc_pub_date") == r["fc_pub_date"]
        checks["meta.official_split == split"] += (p["meta"].get("official_split")
                                                   == SPLIT_ALIASES.get(r["split"]))
        img = c.image[rid]
        same_image = (bool(p["image_paths"]) == c.usable(rid)
                      and (not p["image_paths"] or p["image_paths"][0] == img["path"]))
        checks["image resolved as the build resolved it"] += same_image
        checks["image_status as the build recorded it"] += ([s["status"] for s in p["image_status"]]
                                                            == [img["status"]])
    crosstab = Counter((r["split"], c.split(r["image_path"]) or "not in processed") for r in c.raw)
    moved = sorted(rid for rid, p in c.processed.items()
                   if p["split"] != SPLIT_ALIASES.get(c.by_id[rid]["split"]))
    bounds = {}
    for s in OFFICIAL_SPLITS:
        days = sorted(day(r) for r in c.raw if r["split"] == s)
        bounds[s] = {"rows": len(days), "first": days[0], "last": days[-1]}
    windows = {
        "train after the dev cutoff": sum(1 for r in c.raw if r["split"] == "train" and day(r) > DEV_STARTS),
        "dev before the dev cutoff": sum(1 for r in c.raw if r["split"] == "dev" and day(r) < DEV_STARTS),
        "dev after the test cutoff": sum(1 for r in c.raw if r["split"] == "dev" and day(r) > TEST_STARTS),
        "test before the test cutoff": sum(1 for r in c.raw if r["split"] == "test" and day(r) < TEST_STARTS),
        "test after the paper's end date": sum(1 for r in c.raw if r["split"] == "test" and day(r) > TEST_ENDS),
    }
    boundary = {d: ordered(Counter(r["split"] for r in c.raw if day(r) == d))
                for d in (DEV_STARTS, TEST_STARTS)}
    scripts_seen: Counter = Counter()
    non_latin_rows: list[str] = []
    mostly_non_latin: list[dict[str, str]] = []
    for r in c.raw:
        letters = scripts_of(r["claim"])
        other = sum(n for s, n in letters.items() if s != "LATIN")
        if other:
            non_latin_rows.append(r["image_path"])
            scripts_seen.update({s: 1 for s in letters if s != "LATIN"})
            if other > sum(letters.values()) / 2:
                mostly_non_latin.append({"record_id": r["image_path"], "split": r["split"],
                                         "claim_language": r["claim_language"],
                                         "script": max((s for s in letters if s != "LATIN"),
                                                       key=lambda s: letters[s])})
    fmt = Counter()
    unparseable = 0
    for r in c.raw:
        value = str(r.get("fc_pub_date") or "")
        fmt["YYYY-MM-DD HH:MM:SS" if len(value) == 19 else "YYYY-MM-DD" if len(value) == 10 else
            f"other ({len(value)} chars)"] += 1
        try:
            datetime.date.fromisoformat(value[:10])
        except ValueError:
            unparseable += 1
    return {
        "raw": {"file": "data/M4FC.json", "rows": len(c.raw), "keys": raw_keys,
                "rows_with_other_key_sets": sum(1 for r in c.raw if list(r) != raw_keys)},
        "validated": {"file": "data/M4FC_human_validated.json", "rows": len(c.validated),
                      "same_keys_as_main": list(c.validated[0]) == raw_keys,
                      "read_by_the_pipeline": False},
        "interim": {"rows": len(c.interim), "keys": sorted(one_interim),
                    "meta_keys": sorted(one_interim.get("meta") or {}),
                    "image_ref_keys": sorted((one_interim.get("images") or [{}])[0])},
        "processed": {"rows": len(c.processed), "keys": list(one_processed),
                      "meta_keys": sorted(one_processed.get("meta") or {})},
        "survives": SURVIVES,
        "image_url": "kept in interim (images[].url) for resolution; not a processed field",
        "dropped": [k for k in raw_keys if k not in SURVIVES],
        "rows_dropped": {"official test, excluded by the register (build_processed.EXCLUDED_SPLITS)":
                         sum(1 for r in c.raw if r["split"] == "test")},
        "value_checks": {k: {"equal": v, "of": len(c.processed)} for k, v in checks.items()},
        "split": {
            "crosstab_official_x_processed": {f"{a} -> {b}": n for (a, b), n in sorted(crosstab.items())},
            "moved": len(moved),
            "moved_into_train": sum(1 for rid in moved if c.split(rid) == "train"),
            "bounds": bounds, "outside_the_paper_windows": windows, "boundary_days": boundary,
        },
        "english_claim": {
            "rows_with_a_claim": sum(1 for r in c.raw if present(r["claim"])),
            "rows": len(c.raw),
            "claim_language": ordered(Counter(r["claim_language"] for r in c.raw)),
            "multilingual_claim_absent_when_english": sum(
                1 for r in c.raw if r["claim_language"] == "English" and not present(r["multilingual_claim"])),
            "claims_with_non_latin_letters": len(non_latin_rows),
            "claims_mostly_non_latin": len(mostly_non_latin),
            "mostly_non_latin": mostly_non_latin,
            "non_latin_scripts": ordered(scripts_seen),
            "non_latin_record_ids": non_latin_rows[:20],
            "language_id": "no language-ID library is installed; Latin-script languages are not told apart",
        },
        "fc_pub_date": {"present": sum(1 for r in c.raw if r.get("fc_pub_date")),
                        "unparseable": unparseable, "formats": ordered(fmt)},
    }


def item_shapes(c: Corpus) -> dict[str, Any]:
    """What the paper says of the extracted items, checked value by value."""
    out: dict[str, Any] = {"absent_markers": ordered(Counter(
        p.casefold() for r in c.raw for f in [*ITEMS, *(f"claimed_{i}" for i in ITEMS)]
        for p in parts(r[f]) if p.casefold() in ABSENT))}
    for field in ("claimed_date", "date"):
        shapes = Counter(date_shape(r[field]) or "absent" for r in c.raw)
        examples = sorted({str(r[field]) for r in c.raw if date_shape(r[field]) in ("range", "other")})
        out[field] = {"shapes": ordered(shapes), "multi_valued": sum(
            1 for r in c.raw if len([p for p in parts(r[field]) if p]) > 1), "examples": examples[:12]}
    out["value_types"] = {f: ordered(Counter(type(r[f]).__name__ for r in c.raw))
                          for f in [*ITEMS, *(f"claimed_{i}" for i in ITEMS)]}
    coarse = Counter((r["verdict"], r["verdict_coarse"]) for r in c.raw)
    out["verdict_x_coarse"] = {f"{a} -> {b}": n for (a, b), n in sorted(coarse.items())}
    out["verdict_true_coarse_false"] = [r["image_path"] for r in c.raw
                                        if r["verdict"] == "true" and r["verdict_coarse"] == "false"]
    out["task_verdict_prediction_false"] = ordered(Counter(r["verdict"] for r in c.raw
                                                           if not r["task_verdict_prediction"]))
    return out


# --------------------------------------------------------------------------
# 3. counts per split
# --------------------------------------------------------------------------


def counts(rows: list[dict[str, Any]], c: Corpus) -> dict[str, Any]:
    ooc = [r for r in rows if r["verdict"] == OOC]
    usable = [r for r in rows if c.usable(r["image_path"])]
    return {
        "rows": len(rows),
        "english_claims": sum(1 for r in rows if present(r["claim"])),
        "multilingual_claims": sum(1 for r in rows if r["claim_language"] != "English"
                                   and present(r["multilingual_claim"])),
        "synthetic_true_captions": sum(1 for r in rows if r["use_true_caption"]
                                       and present(r["true_caption"])),
        "rows_with_usable_image": len(usable),
        "distinct_sha256": len({c.image[r["image_path"]]["sha256"] for r in usable}),
        "verdict": ordered(Counter(r["verdict"] for r in rows)),
        "verdict_coarse": ordered(Counter(r["verdict_coarse"] for r in rows)),
        "balanced_flag": sum(1 for r in rows if r["use_true_caption"]),
        "balanced_flag_by_verdict": ordered(Counter(r["verdict"] for r in rows if r["use_true_caption"])),
        "balanced_flag_without_caption": sum(1 for r in rows if r["use_true_caption"]
                                             and not present(r["true_caption"])),
        "ooc_rows": len(ooc),
        "ooc_both_present": {i: sum(1 for r in ooc if both(r, i)) for i in ITEMS},
        "ooc_rows_with_any_item": sum(1 for r in ooc if any(both(r, i) for i in ITEMS)),
        "ooc_both_present_with_usable_image": {i: sum(1 for r in ooc if both(r, i)
                                                      and c.usable(r["image_path"])) for i in ITEMS},
        "fc_org": ordered(Counter(r["fc_org"] for r in rows)),
    }


def _normal(value: Any) -> str:
    return re.sub(r"\s+", " ", shown(value)).strip().casefold()


def validated_section(c: Corpus) -> dict[str, Any]:
    """The authors' human-validated rows against the main file, field by field."""
    per_field: dict[str, Counter] = defaultdict(Counter)
    transitions: Counter = Counter()
    splits = Counter()
    for v in c.validated:
        main = c.by_id[v["image_path"]]
        splits[main["split"]] += 1
        for field in v:
            if v[field] != main[field]:
                per_field[field]["raw:" + main["split"]] += 1
                if _normal(v[field]) != _normal(main[field]):
                    per_field[field]["normalised:" + main["split"]] += 1
        if v["verdict"] != main["verdict"]:
            transitions[(main["verdict"], v["verdict"])] += 1
    fields = {}
    for field, n in sorted(per_field.items(), key=lambda kv: -sum(x for k, x in kv[1].items()
                                                                  if k.startswith("raw:"))):
        fields[field] = {kind: {s: n.get(f"{kind}:{s}", 0) for s in OFFICIAL_SPLITS}
                         for kind in ("raw", "normalised")}
    return {"rows": len(c.validated), "rows_by_split": {s: splits.get(s, 0) for s in OFFICIAL_SPLITS},
            "normalised_means": "case, spacing and list-vs-';' differences ignored",
            "fields_differing": fields,
            "verdict_changes": {f"{a} -> {b}": n for (a, b), n in sorted(transitions.items())}}


# --------------------------------------------------------------------------
# 4. overlap
# --------------------------------------------------------------------------


def overlap_section(c: Corpus, shared: dict[str, dict[str, str]],
                    partners: dict[str, dict[str, set[tuple[str, str]]]]) -> dict[str, Any]:
    from scripts import images
    from scripts.placeholders import cross_corpus_pairs
    from scripts.records import SPLIT_ALIASES

    by_sha: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(c.image):
        if c.usable(rid) and c.image[rid]["sha256"]:
            by_sha[c.image[rid]["sha256"]].append(rid)
    out: dict[str, Any] = {"m4fc_rows_with_usable_image": sum(len(v) for v in by_sha.values()),
                           "m4fc_distinct_sha256": len(by_sha), "datasets": {}}
    for name in sorted(shared):
        rec = partners.get(name, {})
        rows = sorted({rid for sha in rec for rid in by_sha[sha]})
        out["datasets"][name] = {
            "shared_images_on_disk": len(set(shared[name].values())),
            "shared_images_in_processed_records": len(rec),
            "m4fc_rows": len(rows),
            "m4fc_rows_by_official_split": {s: sum(1 for r in rows if c.by_id[r]["split"] == s)
                                            for s in OFFICIAL_SPLITS},
            "m4fc_rows_by_processed_split": ordered(Counter(c.split(r) or "not in processed" for r in rows)),
            "partner_records_by_split": ordered(Counter(s for v in rec.values() for _rid, s in v)),
            "images_by_official_x_partner_split": ordered(Counter(
                f"{a} x {b}" for _sha, a, b in {(sha, c.by_id[r]["split"], s) for sha, v in rec.items()
                                                for r in by_sha[sha] for _rid, s in v})),
            "images_by_processed_x_partner_split": ordered(Counter(
                f"{a} x {b}" for _sha, a, b in {(sha, c.split(r) or "not in processed", s)
                                                for sha, v in rec.items()
                                                for r in by_sha[sha] for _rid, s in v})),
        }
    out["text_only_enabled"] = sorted(n for n in dataset_names() if dataset(n).get("enabled")
                                      and n not in images.IMAGE_DATASETS)
    drop = []
    for name in ("verite", "averimatec"):
        for sha, v in sorted(partners.get(name, {}).items()):
            for rid in by_sha[sha]:
                if c.by_id[rid]["split"] in ("train", "dev"):
                    drop.append({"record_id": rid, "official_split": c.by_id[rid]["split"],
                                 "processed_split": c.split(rid), "sha256": sha[:12],
                                 "partner": name, "partner_record_ids": sorted({r for r, _s in v})})
    out["drop_list"] = drop
    out["on_disk_only"] = {n: len(set(shared.get(n, {}).values()) - set(partners.get(n, {})))
                           for n in ("verite", "averimatec")}
    groups = {sha: rids for sha, rids in by_sha.items() if len(rids) > 1}
    out["within_m4fc"] = {
        "images_on_two_or_more_rows": len(groups),
        "rows_in_them": sum(len(v) for v in groups.values()),
        "copies": {str(k): n for k, n in sorted(Counter(len(v) for v in groups.values()).items())},
        "official_splits_spanned": ordered(Counter(
            " + ".join(sorted({c.by_id[r]["split"] for r in v})) for v in groups.values())),
        "processed_splits_spanned": ordered(Counter(
            " + ".join(sorted({c.split(r) or "not in processed" for r in v})) for v in groups.values())),
    }
    yaml_ids: set[str] = set()
    yaml_groups = 0
    for pair in cross_corpus_pairs():
        for m in pair.get("members", []):
            if m["dataset"] == "m4fc":
                yaml_ids.update(m.get("record_ids") or [])
                yaml_groups += 1
    linked = {sha for name in partners for sha in partners[name]}
    computed = {rid for sha in linked for rid in by_sha[sha] if rid in c.processed}
    out["cross_check_with_duplicates_yaml"] = {
        "yaml_m4fc_records": len(yaml_ids), "yaml_groups_with_m4fc": yaml_groups,
        "computed_m4fc_records": len(computed),
        "computed_groups": len({sha for sha in linked if any(r in c.processed for r in by_sha[sha])}),
        "only_in_yaml": sorted(yaml_ids - computed), "only_computed": sorted(computed - yaml_ids)}
    moved = [rid for rid, p in c.processed.items() if p["split"] != SPLIT_ALIASES.get(c.by_id[rid]["split"])]
    out["moved_rows_by_partners"] = ordered(Counter(
        " + ".join(sorted(n for n in partners if c.image[rid]["sha256"] in partners[n])) or "none"
        for rid in moved))
    out["moved_rows_by_move"] = ordered(Counter(
        f"{c.by_id[rid]['split']} -> {c.split(rid)}" for rid in moved))
    out["fivepils_in_register"] = any("pils" in n.lower() for n in dataset_names())
    return out


# --------------------------------------------------------------------------
# scoring: per-class metrics, bootstrap intervals, the majority baseline
# --------------------------------------------------------------------------


def _metrics(t, p, k: int):
    import numpy as np

    cm = np.bincount(k * t + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.full(k, np.nan), where=support > 0)
    seen = np.nan_to_num(recall)
    f1 = np.divide(2 * precision * seen, precision + seen, out=np.zeros(k),
                   where=(precision + seen) > 0)
    f1[support == 0] = np.nan
    return float(tp.sum() / len(t)), float(np.nanmean(f1)), precision, recall, f1, support


def scored(y_true: Sequence[str], y_pred: Sequence[str], baseline: str,
           classes: Sequence[str], seed: int = SEED) -> dict[str, Any]:
    """Per class (support, precision, recall, F1), accuracy and macro-F1, each
    with a 95% bootstrap interval over the scored rows, for the model and for
    always predicting ``baseline`` (the train majority), on the same draws.

    Precision with no prediction of the class is 0; recall and F1 for a class
    absent from a draw are left out of that draw. Macro-F1 averages the classes
    present in the scored rows."""
    import numpy as np

    classes = list(classes)
    k = len(classes)
    index = {cl: i for i, cl in enumerate(classes)}
    t = np.array([index[y] for y in y_true])
    preds = {"model": np.array([index[y] for y in y_pred]),
             "majority": np.full(len(t), index[baseline])}
    draws = np.random.default_rng(seed).integers(0, len(t), size=(BOOTSTRAP, len(t)))
    out: dict[str, Any] = {"rows": len(t), "classes": classes, "majority_predicts": baseline}

    def interval(values) -> list[float] | None:
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        return None if not len(v) else [round(float(np.percentile(v, 2.5)), 4),
                                        round(float(np.percentile(v, 97.5)), 4)]

    for who, p in preds.items():
        acc, macro, prec, rec, f1, support = _metrics(t, p, k)
        boot = [_metrics(t[d], p[d], k) for d in draws]
        cell = {"accuracy": {"value": round(acc, 4), "ci": interval([b[0] for b in boot])},
                "macro_f1": {"value": round(macro, 4), "ci": interval([b[1] for b in boot])},
                "per_class": {}}
        for i, cl in enumerate(classes):
            cell["per_class"][cl] = {
                "support": int(support[i]),
                "precision": {"value": round(float(prec[i]), 4),
                              "ci": interval([b[2][i] for b in boot])},
                "recall": {"value": None if np.isnan(rec[i]) else round(float(rec[i]), 4),
                           "ci": interval([b[3][i] for b in boot])},
                "f1": {"value": None if np.isnan(f1[i]) else round(float(f1[i]), 4),
                       "ci": interval([b[4][i] for b in boot])}}
        out[who] = cell
    return out


def majority(labels: Iterable[str]) -> str:
    """The most common label; ties broken by name, as confound.majority_map does."""
    return min(Counter(labels).most_common(), key=lambda kv: (-kv[1], kv[0]))[0]


# --------------------------------------------------------------------------
# the text classifier: TF-IDF + logistic regression, numpy and scipy only
# --------------------------------------------------------------------------


class Tfidf:
    """Word unigrams and bigrams (fakeddit_classes.tokens), min_df counted on the
    training rows, smoothed idf, sublinear tf, L2-normalised rows."""

    def __init__(self, min_df: int = MIN_DF):
        self.min_df = min_df

    def fit(self, docs: Sequence[list[str]]) -> "Tfidf":
        import numpy as np

        df: Counter = Counter()
        for toks in docs:
            df.update(set(toks))
        terms = sorted(t for t, n in df.items() if n >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in terms])
        return self

    def transform(self, docs: Sequence[list[str]]):
        import numpy as np
        from scipy import sparse

        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            for term, k in Counter(t for t in toks if t in self.vocab).items():
                j = self.vocab[term]
                rows.append(i)
                cols.append(j)
                vals.append((1 + math.log(k)) * self.idf[j])
        x = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ x


def fit_logreg(x, y: Sequence[int], k: int, c: float = LR_C):
    """Multinomial logistic regression with class-balanced weights and L2 on the
    weights, not the intercepts: minimise c * sum_i w(y_i) * -log p(y_i | x_i)
    + |W|^2 / 2 by L-BFGS. The same objective as a balanced, C=1 classifier."""
    import numpy as np
    from scipy.optimize import minimize

    n, d = x.shape
    y = np.asarray(y)
    counts = np.bincount(y, minlength=k).astype(float)
    sample_weight = (n / (k * counts))[y] * c
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    xt = x.T.tocsr()

    def objective(theta):
        w = theta[:d * k].reshape(d, k)
        z = x @ w + theta[d * k:]
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -(sample_weight * (onehot * logp).sum(axis=1)).sum() + 0.5 * float((w * w).sum())
        g = (np.exp(logp) - onehot) * sample_weight[:, None]
        return loss, np.concatenate([(xt @ g + w).ravel(), g.sum(axis=0)])

    res = minimize(objective, np.zeros(d * k + k), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000})
    w, b = res.x[:d * k].reshape(d, k), res.x[d * k:]
    return w, b, {"converged": bool(res.success), "iterations": int(res.nit),
                  "message": str(res.message)}


def text_audit(train_texts: Sequence[str], train_labels: Sequence[str],
               val_texts: Sequence[str], val_labels: Sequence[str]) -> dict[str, Any]:
    """Fit on train, score on val, once. Nothing is tuned and nothing is kept."""
    import numpy as np

    from scripts.fakeddit_classes import tokens

    fitted = sorted(set(train_labels))
    vec = Tfidf().fit([tokens(t) for t in train_texts])
    w, b, fit = fit_logreg(vec.transform([tokens(t) for t in train_texts]),
                           [fitted.index(lab) for lab in train_labels], len(fitted))
    logits = vec.transform([tokens(t) for t in val_texts]) @ w + b
    pred = [fitted[i] for i in np.asarray(logits).argmax(axis=1)]
    return {"train_rows": len(train_texts), "features": len(vec.vocab), "fit": fit,
            "settings": {"tokens": "fakeddit_classes.tokens: words [a-z0-9'] + bigrams",
                         "min_df": MIN_DF, "idf": "smoothed", "tf": "sublinear",
                         "rows": "L2-normalised", "classifier": "multinomial, class-balanced",
                         "C": LR_C},
            "train_classes": ordered(Counter(train_labels)),
            "val_classes": ordered(Counter(val_labels)),
            **scored(list(val_labels), pred, majority(train_labels),
                     sorted(set(train_labels) | set(val_labels)))}


# --------------------------------------------------------------------------
# 5. shortcut audits
# --------------------------------------------------------------------------


def purity_table(rows: list[dict[str, Any]], key: str, label: str) -> list[dict[str, Any]]:
    import numpy as np

    from scripts.splits import domain_purity

    purity = domain_purity({r["image_path"]: {str(r[key])} for r in rows},
                           {r["image_path"]: str(r[label]) for r in rows})
    by: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by[str(r[key])].append(str(r[label]))
    rng = np.random.default_rng(SEED)
    out = []
    for group, info in sorted(purity.items(), key=lambda kv: (-kv[1]["records"], kv[0])):
        labs = sorted(set(by[group]))
        codes = np.array([labs.index(x) for x in by[group]])
        draws = codes[rng.integers(0, len(codes), size=(BOOTSTRAP, len(codes)))]
        share = np.stack([(draws == j).sum(axis=1) for j in range(len(labs))], axis=1).max(axis=1)
        share = share / len(codes)
        out.append({"group": group, "rows": info["records"], "dominant": info["dominant_class"],
                    "purity": info["purity"],
                    "ci": [round(float(np.percentile(share, 2.5)), 4),
                           round(float(np.percentile(share, 97.5)), 4)],
                    "label_predictive": info["purity"] >= PURITY_FLAG,
                    "classes": ordered(Counter(by[group]))})
    return out


def lookup_audit(train: list[dict[str, Any]], val: list[dict[str, Any]], key: str,
                 label: str) -> dict[str, Any]:
    """A group-only classifier: the train majority of the row's group."""
    from scripts.confound import majority_map

    table, fallback = majority_map([(str(r[key]), str(r[label])) for r in train])
    pred = [table.get(str(r[key]), fallback) for r in val]
    truth = [str(r[label]) for r in val]
    return {"unseen_groups_in_val": sum(1 for r in val if str(r[key]) not in table),
            **scored(truth, pred, majority(str(r[label]) for r in train),
                     sorted({str(r[label]) for r in train} | set(truth)))}


def shortcut_section(c: Corpus) -> dict[str, Any]:
    sub = {s: [r for r in c.raw if c.split(r["image_path"]) == s] for s in ("train", "val")}
    task = {s: [r for r in rows if r["task_verdict_prediction"]] for s, rows in sub.items()}
    out: dict[str, Any] = {
        "rows": {s: len(v) for s, v in task.items()},
        "left_out_not_in_the_verdict_task": {s: len(sub[s]) - len(task[s]) for s in sub},
        "labels": {"coarse": "verdict_coarse (true / false)",
                   "fine": "verdict (out-of-context / manipulated / AI-generated / true)"},
        "purity": {}, "lookup": {}, "text": {}}
    for name, label in (("coarse", "verdict_coarse"), ("fine", "verdict")):
        out["purity"][name] = {key: purity_table(task["train"], key, label)
                               for key in ("fc_org", "claim_language", "fc_language")}
        out["lookup"][name] = {key: lookup_audit(task["train"], task["val"], key, label)
                               for key in ("fc_org", "claim_language")}
        out["text"][name] = text_audit([r["claim"] for r in task["train"]],
                                       [str(r[label]) for r in task["train"]],
                                       [r["claim"] for r in task["val"]],
                                       [str(r[label]) for r in task["val"]])

    def balanced(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        texts, labels = [], []
        for r in rows:
            if r["use_true_caption"] and present(r["true_caption"]):
                texts.append(r["true_caption"])
                labels.append("synthetic_true")
            elif r["verdict_coarse"] == "false" and not r["use_true_caption"]:
                texts.append(r["claim"])
                labels.append("real_false")
        return texts, labels

    tr_x, tr_y = balanced(sub["train"])
    va_x, va_y = balanced(sub["val"])
    out["synthetic_vs_real"] = {
        "rule": "synthetic_true: true_caption of a row flagged use_true_caption; "
                "real_false: claim of a false row not flagged (the balanced setting's two sides)",
        **text_audit(tr_x, tr_y, va_x, va_y)}
    return out


# --------------------------------------------------------------------------
# 7. the review sheet
# --------------------------------------------------------------------------


def allocate(sizes: dict[str, int], total: int = SHEET_INSTANCES,
             cap: int = SHEET_CAP) -> dict[str, int]:
    """Instances per org: one each, the rest in proportion to eligible rows by
    largest remainder, never more than ``cap`` nor more than the org has."""
    orgs = sorted(o for o, n in sizes.items() if n > 0)
    limit = {o: min(cap, sizes[o]) for o in orgs}
    if len(orgs) > total or sum(limit.values()) < total:
        raise ValueError(f"cannot place {total} instances over {len(orgs)} orgs "
                         f"holding {sum(limit.values())} under the cap")
    rest = total - len(orgs)
    weight = sum(sizes[o] for o in orgs)
    share = {o: rest * sizes[o] / weight for o in orgs}
    alloc = {o: min(limit[o], 1 + int(share[o])) for o in orgs}
    order = sorted(orgs, key=lambda o: (-(share[o] - int(share[o])), -sizes[o], o))
    while sum(alloc.values()) < total:
        for o in order:
            if sum(alloc.values()) < total and alloc[o] < limit[o]:
                alloc[o] += 1
    return alloc


def sheet_rows(c: Corpus) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(sheet rows, private map entries, allocation) -- computed, nothing written."""
    eligible: dict[str, list[str]] = defaultdict(list)
    for r in c.raw:
        rid = r["image_path"]
        if (c.split(rid) == "train" and r["verdict"] == OOC and c.usable(rid)
                and any(both(r, i) for i in ITEMS)):
            eligible[r["fc_org"]].append(rid)
    sizes = {o: len(v) for o, v in eligible.items()}
    alloc = allocate(sizes)
    rng = random.Random(SEED)
    picked = []
    for org in sorted(alloc):
        picked += rng.sample(sorted(eligible[org]), alloc[org])
    rng.shuffle(picked)  # sheet order says nothing about the org
    validated = {v["image_path"] for v in c.validated}
    rows, private = [], []
    for n, rid in enumerate(picked, 1):
        r = c.by_id[rid]
        instance = f"m4fc-ooc-{n:03d}"
        img = c.image[rid]
        ext = FORMAT_EXT.get(img["format"] or "", Path(img["path"]).suffix.lower())
        copy = SHEET_IMAGES / f"{instance}{ext}"
        private.append({"instance": instance, "record_id": rid, "fc_org": r["fc_org"],
                        "verdict": r["verdict"], "official_split": r["split"],
                        "processed_split": c.split(rid), "source_image": img["path"],
                        "sha256": img["sha256"], "in_human_validated": rid in validated})
        for item in ITEMS:
            if both(r, item):
                rows.append({"sheet_id": f"{instance}-{item}",
                             "image": copy.relative_to(PROJECT_ROOT).as_posix(),
                             "claim": r["claim"], "item": item,
                             "claimed": shown(r[f"claimed_{item}"]), "actual": shown(r[item]),
                             "mismatch": "", "note": ""})
    summary = {"eligible_rows_by_org": ordered(sizes), "instances_by_org": ordered(alloc),
               "instances": len(picked), "sheet_rows": len(rows),
               "rows_by_item": {i: sum(1 for x in rows if x["item"] == i) for i in ITEMS},
               "instances_in_human_validated": sum(p["in_human_validated"] for p in private)}
    return rows, private, summary


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SHEET_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_sheet(c: Corpus) -> dict[str, Any]:
    """Write the sheet, its neutral image copies and the private map -- never over
    an existing sheet, which may already hold a person's decisions."""
    rows, private, summary = sheet_rows(c)
    summary["sheet"] = SHEET.relative_to(PROJECT_ROOT).as_posix()
    summary["images"] = SHEET_IMAGES.relative_to(PROJECT_ROOT).as_posix()
    if SHEET.is_file():
        with SHEET.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        fixed = [k for k in SHEET_COLUMNS if k not in ("mismatch", "note")]
        same = [{k: x.get(k) for k in fixed} for x in existing] == [{k: x[k] for k in fixed} for x in rows]
        print("  review sheet exists, kept: " + ("it matches this run" if same else
                                                 "IT DIFFERS from this run, left untouched"))
        return summary
    SHEET_IMAGES.mkdir(parents=True, exist_ok=True)
    for entry in private:
        src = PROJECT_ROOT / entry["source_image"]
        ext = FORMAT_EXT.get(c.image[entry["record_id"]]["format"] or "", src.suffix.lower())
        dst = SHEET_IMAGES / f"{entry['instance']}{ext}"
        shutil.copyfile(src, dst)
        if sha256_of(dst) != entry["sha256"]:
            raise RuntimeError(f"{dst}: the copy's bytes are not the image's")
    write_json(PRIVATE_MAP, {"seed": SEED, "instances": private, "allocation": summary["instances_by_org"]})
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(_csv_text(rows), encoding="utf-8-sig", newline="")
    print(f"  review sheet written: {summary['sheet']}")
    return summary


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------


def open_issues(c: Corpus) -> dict[str, Any]:
    test = [r for r in c.raw if c.split(r["image_path"]) == "test"]
    val = [r for r in c.raw if c.split(r["image_path"]) == "val"]
    train = [r for r in c.raw if c.split(r["image_path"]) == "train"]
    return {
        "processed_test_rows": len(test),
        "processed_test_rows_before_the_test_cutoff": sum(1 for r in test if day(r) < TEST_STARTS),
        "processed_test_dates": [min(map(day, test)), max(map(day, test))] if test else None,
        "processed_val_rows_before_the_dev_cutoff": sum(1 for r in val if day(r) < DEV_STARTS),
        "slugs_stating_a_finding": {
            "all": sum(1 for r in c.raw if slug_states_finding(r["image_path"])),
            "processed_train": sum(1 for r in train if slug_states_finding(r["image_path"])),
            "processed_train_rows": len(train),
            "words": sorted(SLUG_WORDS) + [" ".join(p) for p in SLUG_PHRASES]},
    }


def measure() -> dict[str, Any]:
    t0 = time.perf_counter()
    c = Corpus()
    print(f"  corpus: {len(c.raw):,} raw rows, {len(c.processed):,} processed, "
          f"{sum(c.usable(r) for r in c.image):,} usable images", flush=True)
    m4fc_sha = {v["sha256"] for rid, v in c.image.items() if c.usable(rid) and v["sha256"]}
    shared = shared_files(m4fc_sha)
    print(f"  indexes streamed: {', '.join(f'{k} {len(v)}' for k, v in shared.items())}", flush=True)
    code_counts, partners = processed_scan(shared)
    print("  processed layer scanned", flush=True)
    groups = c.groups()
    india = {k: [r for r in v if r["fc_org"] in INDIA_ORGS] for k, v in groups.items()}
    out = {
        "generated_by": "python scripts/m4fc_audit.py measure",
        "seed": SEED, "bootstrap": BOOTSTRAP,
        "decisions": {"train_dev": "processed split", "official_test": "aggregate counts only",
                      "sheet_images": "neutral copies", "decided": "2026-10-28"},
        "definitions": definitions(code_counts),
        "fields": fields_section(c),
        "items": item_shapes(c),
        "counts": {k: counts(v, c) for k, v in groups.items()},
        "india": {k: counts(v, c) for k, v in india.items()},
        "human_validated": validated_section(c),
        "overlap": overlap_section(c, shared, partners),
    }
    print("  sections 1-4 measured", flush=True)
    out["shortcuts"] = shortcut_section(c)
    print("  section 5 measured", flush=True)
    out["sheet"] = write_sheet(c)
    out["open_issues"] = open_issues(c)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(MEASURE, {k: v for k, v in out.items() if k != "seconds"})
    print(f"  wrote {MEASURE.relative_to(PROJECT_ROOT).as_posix()} in {out['seconds']} s")
    return out


# --------------------------------------------------------------------------
# 6. evidence coverage
# --------------------------------------------------------------------------


def evidence_sample() -> list[str]:
    from scripts.processed_loader import iter_rows

    train = sorted(r["record_id"] for r in iter_rows("m4fc", split="train"))
    return random.Random(SEED).sample(train, EVIDENCE_N)


def on_domain(domain: str | None, orgs: Iterable[str]) -> bool:
    d = (domain or "").lower()
    d = d[4:] if d.startswith("www.") else d
    return any(d == o or d.endswith("." + o) for o in orgs)


def entity_terms(spans: list[dict[str, Any]], text: str) -> list[dict[str, str]]:
    """Distinct entity strings (case-folded), every type the model tags."""
    seen, out = set(), []
    for e in spans:
        surface = text[e["start"]:e["end"]].strip()
        key = surface.casefold()
        if sum(ch.isalnum() for ch in surface) < 2 or key in seen:
            continue
        seen.add(key)
        out.append({"type": e["entity_group"], "text": surface})
    return out


def found_in(term: str, haystacks: Sequence[str]) -> bool:
    """A whole-word, case-folded match: 'Modi' is not found in 'Modigliani'."""
    pat = re.compile(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)")
    return any(pat.search(h) for h in haystacks)


def doc_text(doc_id: str) -> str:
    doc = json.loads((BIBLE / "docs" / f"{doc_id}.json").read_text(encoding="utf-8"))
    return f"{doc.get('title') or ''}\n{doc.get('text') or ''}"


def run_evidence(limit: int | None) -> int:
    from scripts import gallery, retrieve

    raw = {r["image_path"]: r for r in raw_rows()}
    sample = evidence_sample()
    config = {"n": EVIDENCE_N, "k": EVIDENCE_K, "entity_top": ENTITY_TOP, "seed": SEED,
              "population": "processed train", "query": "claim", "as_of": "fc_pub_date[:10]",
              "ner": {**gallery.NER_MODEL, "aggregation": "first", "types": "every type tagged",
                      "input": "gallery.clean_claim(claim)"},
              "sample": sample}
    if EVIDENCE_CONFIG.is_file():
        if json.loads(EVIDENCE_CONFIG.read_text(encoding="utf-8")) != config:
            raise SystemExit(f"{EVIDENCE_CONFIG} was written for another sample or setting; "
                             "refusing to mix ledgers")
    else:
        write_json(EVIDENCE_CONFIG, config)
    done = set()
    if EVIDENCE.is_file():
        for line in EVIDENCE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["record_id"])
    todo = [rid for rid in sample if rid not in done][:limit]
    print(f"  evidence: {len(done)} of {EVIDENCE_N} done, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    ner = gallery.Ner(device="cpu")
    ner.load()
    texts = [gallery.clean_claim(raw[rid]["claim"])[0] for rid in todo]
    spans = ner.pipe(texts, batch_size=gallery.NER_BATCH)
    if spans and isinstance(spans[0], dict):  # one text in, one flat list out
        spans = [spans]
    orgs = sorted({r["fc_org"] for r in raw.values()})
    con = retrieve.connect()
    took = []
    try:
        with EVIDENCE.open("a", encoding="utf-8") as ledger:
            for n, (rid, text, found) in enumerate(zip(todo, texts, spans), 1):
                r = raw[rid]
                as_of = day(r)
                start = time.perf_counter()
                hits = retrieve.search(r["claim"], as_of, EVIDENCE_K, con=con)
                searched = time.perf_counter() - start
                entities = entity_terms(found, text)
                top = [doc_text(h.doc_id).casefold() for h in hits[:ENTITY_TOP]]
                matched = [e["text"] for e in entities if found_in(e["text"], top)]
                took.append(time.perf_counter() - start)
                ledger.write(json.dumps({
                    "record_id": rid, "fc_org": r["fc_org"], "india": r["fc_org"] in INDIA_ORGS,
                    "as_of": as_of, "hits": len(hits),
                    "top1_bm25": round(hits[0].score, 5) if hits else None,
                    "after_as_of": sum(1 for h in hits if h.published_at > as_of),
                    "fact_check_domain_hits": sum(1 for h in hits if on_domain(h.source_domain, orgs)),
                    "own_org_hits": sum(1 for h in hits if on_domain(h.source_domain, [r["fc_org"]])),
                    "hit_domains": [h.source_domain for h in hits],
                    "hit_dates": [h.published_at for h in hits],
                    "entities": entities, "matched_top5": matched,
                    "search_seconds": round(searched, 2)}, ensure_ascii=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                print(f"    {len(done) + n}/{EVIDENCE_N}  {searched:6.1f} s  hits {len(hits):2d}  "
                      f"entities {len(entities)} matched {len(matched)}", flush=True)
    finally:
        con.close()
    mean = sum(took) / len(took)
    left = EVIDENCE_N - len(done) - len(todo)
    print(f"  mean {mean:.1f} s per claim; {left} left, about {left * mean / 60:.0f} min")
    return 0


def evidence_summary() -> dict[str, Any] | None:
    import numpy as np

    if not EVIDENCE.is_file():
        return None
    lines = [json.loads(x) for x in EVIDENCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    out: dict[str, Any] = {"queries_run": len(lines), "of": EVIDENCE_N, "k": EVIDENCE_K,
                           "entity_top": ENTITY_TOP,
                           "bm25_sign": "SQLite FTS5 bm25(): lower is a better match; values are negative"}
    for name, rows in (("all", lines), ("india", [x for x in lines if x["india"]])):
        if not rows:
            out[name] = {"queries": 0}
            continue
        n = len(rows)
        ents = [len(x["entities"]) for x in rows]
        hit = [len(x["matched_top5"]) for x in rows]
        top1 = np.array([x["top1_bm25"] for x in rows if x["top1_bm25"] is not None])
        bins = Counter("0" if x["hits"] == 0 else "1-4" if x["hits"] < 5 else "5-9" if x["hits"] < 10
                       else "10-19" if x["hits"] < 20 else "20" for x in rows)
        types: dict[str, Counter] = defaultdict(Counter)
        for x in rows:
            matched = {m.casefold() for m in x["matched_top5"]}
            for e in x["entities"]:
                types[e["type"]]["entities"] += 1
                types[e["type"]]["matched"] += e["text"].casefold() in matched
        out[name] = {
            "queries": n,
            "with_a_hit": sum(1 for x in rows if x["hits"] >= 1),
            "hit_counts": {b: bins.get(b, 0) for b in ("0", "1-4", "5-9", "10-19", "20")},
            "top1_bm25": None if not len(top1) else {
                q: round(float(np.percentile(top1, p)), 3)
                for q, p in (("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75),
                             ("p90", 90), ("max", 100))},
            "claims_with_entities": {"1+": sum(1 for e in ents if e >= 1), "2+": sum(1 for e in ents if e >= 2),
                                     "none": sum(1 for e in ents if e == 0)},
            "entity_matches_top5": {"1+": sum(1 for h in hit if h >= 1), "2+": sum(1 for h in hit if h >= 2)},
            "entities_by_type": {t: dict(v) for t, v in sorted(types.items())},
            "after_as_of": sum(x["after_as_of"] for x in rows),
            "queries_with_a_fact_check_domain_hit": sum(1 for x in rows if x["fact_check_domain_hits"]),
            "queries_with_an_own_org_hit": sum(1 for x in rows if x["own_org_hits"]),
            "median_search_seconds": round(float(np.median([x["search_seconds"] for x in rows])), 2),
        }
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


_NUMERIC = re.compile(r"^[-—\d.]")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A pipe table; a column whose every cell reads as a number is right-aligned."""
    def cell(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    body = [[cell(v) for v in row] for row in rows]
    numeric = [i > 0 and all(not r[i] or _NUMERIC.match(r[i]) for r in body) and any(r[i] for r in body)
               for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---:" if n else "---" for n in numeric) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def pct(a: int, b: int) -> str:
    return f"{a:,} of {b:,} ({100 * a / b:.1f}%)" if b else f"{a:,} of 0"


def ci(cell: dict[str, Any]) -> str:
    v, iv = cell.get("value"), cell.get("ci")
    if v is None:
        return "—"
    return f"{v:.3f} [{iv[0]:.3f}, {iv[1]:.3f}]" if iv else f"{v:.3f}"


def scored_table(res: dict[str, Any]) -> str:
    rows = []
    for cl in res["classes"]:
        m, b = res["model"]["per_class"][cl], res["majority"]["per_class"][cl]
        rows.append([cl, m["support"], ci(m["precision"]), ci(m["recall"]), ci(m["f1"]), ci(b["f1"])])
    rows.append(["**accuracy**", res["rows"], "", "", ci(res["model"]["accuracy"]),
                 ci(res["majority"]["accuracy"])])
    rows.append(["**macro-F1**", res["rows"], "", "", ci(res["model"]["macro_f1"]),
                 ci(res["majority"]["macro_f1"])])
    return table(["class", "support", "precision [95% CI]", "recall [95% CI]",
                  "F1, or the row's metric [95% CI]",
                  f"majority baseline (always {res['majority_predicts']}), same metric"], rows)


def render(m: dict[str, Any], ev: dict[str, Any] | None) -> str:
    d, f, it, cnt = m["definitions"], m["fields"], m["items"], m["counts"]
    ov, sh, sheet, oi, hv = m["overlap"], m["shortcuts"], m["sheet"], m["open_issues"], m["human_validated"]
    cols = [f"official {s}" for s in OFFICIAL_SPLITS] + [f"processed {s}" for s in PROCESSED_SPLITS]
    ptrain = cnt["processed train"]
    L: list[str] = []
    add = L.append
    add("# M4FC as a training source for `media_mismatch` / `wrong_image`: audit")
    add("")
    add(f"Generated {datetime.date.today().isoformat()} by `python scripts/m4fc_audit.py report` "
        "from `data/interim/m4fc_audit/`. Measurement only: no project model trained, nothing rebuilt, "
        "fetched or pushed, nothing written under `data/raw` or `data/processed`. Train and val are the "
        "**processed** split (the person's decision, 2026-10-28); M4FC's official test split appears as "
        "aggregate counts only.")
    add("")
    add("## Summary")
    add("")
    add(summary_lines(m, ev))
    add("")

    # 1 ---------------------------------------------------------------
    add("## 1. Definitions")
    add("")
    add("`scripts/reason_codes.py` (module docstring):")
    add("")
    add("```")
    L.extend(d["reason_codes_docstring"])
    add("```")
    add("")
    add("`docs/processed_schema.md`, reason_code table:")
    add("")
    add("| Value | Meaning | Comes from |")
    add("| --- | --- | --- |")
    L.extend(d["processed_schema_rows"])
    add("")
    add("Which source labels map to each, today (`reason_codes.MAPPING`):")
    add("")
    add(table(["reason code", "dataset", "label", "rule"],
              [[c, x["dataset"], x["label"], x["provenance"]] for c, v in d["mapping"].items() for x in v]))
    add("")
    add(f"- DGM4's `{d['dgm4']['label']}` has a `media_mismatch` rule (same-publisher swaps, weak "
        f"supervision), but `DGM4_AUDIT_PASSED = {d['dgm4']['audit_passed']}`: DGM4 is unmapped.")
    add(f"- Fakeddit code 2 (False Connection): {d['fakeddit_rulings'].get('2')}.")
    add(f"- M4FC today: {', '.join(f'`{k}` → `{v}`' for k, v in d['m4fc_mapping'].items())}. Every false "
        "row, out-of-context included, is `content_refuted`.")
    add(f"- The string `wrong_image` also names an image status ({d['same_string_elsewhere'].split(': ', 1)[1]}).")
    add("")
    carried = [[ds, x["reason_code"], x["label"], x["split"], x["use"], x["rows"]]
               for ds, v in d["rows_carrying_the_codes"].items() if isinstance(v, list) for x in v]
    add("Rows carrying either code in the processed layer, measured now:")
    add("")
    add(table(["dataset", "reason code", "label", "split", "use", "rows"], carried))
    add("")
    add("`processed_loader.reason_code_rows(name, split='train', purpose='train')`, called for every "
        "enabled dataset:")
    add("")
    add(table(["dataset", "yields for training"],
              [[ds, "refused: " + v["refused"] if "refused" in v else
                ", ".join(f"{k} {n:,}" for k, n in v["yielded"].items())]
               for ds, v in d["train_yield_of_reason_code_rows"].items()]))
    add("")
    add(f"Trainable rows today: `media_mismatch` {d['trainable_rows']['media_mismatch']:,}, "
        f"`wrong_image` {d['trainable_rows']['wrong_image']:,}. The loader that enforces it:")
    add("")
    add("```python")
    add(d["loader_source"].rstrip())
    add("```")
    add("")

    # 2 ---------------------------------------------------------------
    add("## 2. Fields")
    add("")
    add(f"Raw `{f['raw']['file']}`: {f['raw']['rows']:,} rows, one per image, {len(f['raw']['keys'])} keys "
        f"(rows with another key set: {f['raw']['rows_with_other_key_sets']}). "
        f"`{f['validated']['file']}`: {f['validated']['rows']} rows, same keys: "
        f"{'yes' if f['validated']['same_keys_as_main'] else 'no'}; read by the pipeline: no.")
    add("")
    add(table(["raw field", "in processed"],
              [[k, f["survives"].get(k, "dropped" if k != "image_url" else
                                     "dropped (" + f["image_url"] + ")")] for k in f["raw"]["keys"]]))
    add("")
    add(f"- Interim (`data/interim/records/m4fc.jsonl.gz`): {f['interim']['rows']:,} rows, test included; "
        f"keys {', '.join(f['interim']['keys'])}; meta {', '.join(f['interim']['meta_keys'])}.")
    add(f"- Processed: {f['processed']['rows']:,} rows; keys {', '.join(f['processed']['keys'])}; "
        f"meta {', '.join(f['processed']['meta_keys'])}.")
    add(f"- Dropped between raw and processed: **{len(f['dropped'])} of {len(f['raw']['keys'])} fields**, "
        f"the human-validated file, and {', '.join(f'{v:,} rows ({k})' for k, v in f['rows_dropped'].items())}. "
        "Bringing the fields in needs an interim + processed rebuild: **not done**. This audit reads them "
        "from raw, joined on record_id = `image_path`.")
    add("")
    add("Value checks, processed against raw:")
    add("")
    add(table(["check", "equal", "of"], [[k, v["equal"], v["of"]] for k, v in f["value_checks"].items()]))
    add("")
    sp = f["split"]
    add(f"**Split.** Official → processed: "
        f"{'; '.join(f'{k} {v:,}' for k, v in sp['crosstab_official_x_processed'].items())}. "
        f"{sp['moved']:,} rows moved, {sp['moved_into_train']} into train; by move: "
        f"{', '.join(f'{k} {v}' for k, v in ov['moved_rows_by_move'].items())}; by the corpus they share an "
        f"image with: {', '.join(f'{k} {v}' for k, v in ov['moved_rows_by_partners'].items())}. "
        "So the split is M4FC's, except where the cross-corpus atomicity rule moved a row to a more "
        "held-out split.")
    add("")
    add(table(["official split", "rows", "first fc_pub_date", "last fc_pub_date"],
              [[s, v["rows"], v["first"], v["last"]] for s, v in sp["bounds"].items()]))
    add("")
    add("The paper's cutoffs (2022-09-22 / 2023-01-18 / 2024-09-01) hold: "
        f"{', '.join(f'{k}: {v}' for k, v in sp['outside_the_paper_windows'].items())}. Boundary days are "
        f"shared: {'; '.join(f'{k}: ' + ', '.join(f'{s} {n}' for s, n in v.items()) for k, v in sp['boundary_days'].items())}.")
    add("")
    ec = f["english_claim"]
    untranslated = "; ".join(f"`{x['record_id']}` (official {x['split']}, claim_language {x['claim_language']}, "
                             f"{x['script']} script)" for x in ec["mostly_non_latin"])
    add(f"**English claim.** Present (non-empty, not \"{NEI}\") in {pct(ec['rows_with_a_claim'], ec['rows'])} rows. "
        f"`claim_language` English {ec['claim_language'].get('English', 0):,}, with `multilingual_claim` "
        f"absent in {ec['multilingual_claim_absent_when_english']:,} of them. Claims with any non-Latin letter: "
        f"{ec['claims_with_non_latin_letters']} (scripts: "
        f"{', '.join(f'{k} {v}' for k, v in ec['non_latin_scripts'].items()) or 'none'}); the others quote a "
        f"short phrase, but {ec['claims_mostly_non_latin']} is mostly non-Latin, i.e. not translated: "
        f"{untranslated or '—'}. {ec['language_id'][0].upper() + ec['language_id'][1:]}.")
    add("")
    fd = f["fc_pub_date"]
    add(f"**FC-article date.** `fc_pub_date` present in {fd['present']:,} rows, unparseable "
        f"{fd['unparseable']}; formats: {', '.join(f'{k} {v:,}' for k, v in fd['formats'].items())}. "
        "Processed keeps it unchanged (value check above).")
    add("")

    # 3 ---------------------------------------------------------------
    add("## 3. Counts per split")
    add("")
    add("Official test is aggregate only. Processed test is the rows moved there by the atomicity rule.")
    add("")
    keys = [("rows (one English claim each)", "rows"), ("multilingual claims", "multilingual_claims"),
            ("synthetic true captions", "synthetic_true_captions"),
            ("rows with a usable image", "rows_with_usable_image"),
            ("distinct images (sha256)", "distinct_sha256"), ("balanced flag set", "balanced_flag"),
            ("… flag set, no caption", "balanced_flag_without_caption"), ("out-of-context rows", "ooc_rows"),
            ("… with ≥ 1 item, both values", "ooc_rows_with_any_item")]
    add(table(["measure", *cols, "all"], [[label, *[cnt[c][k] for c in cols], cnt["all"][k]] for label, k in keys]))
    add("")
    verdicts = list(cnt["all"]["verdict"])
    add(table(["verdict", *cols, "all"], [[v, *[cnt[c]["verdict"].get(v, 0) for c in cols],
                                           cnt["all"]["verdict"][v]] for v in verdicts]))
    add("")
    coarse = list(cnt["all"]["verdict_coarse"])
    add(table(["verdict_coarse", *cols, "all"], [[v, *[cnt[c]["verdict_coarse"].get(v, 0) for c in cols],
                                                  cnt["all"]["verdict_coarse"][v]] for v in coarse]))
    add("")
    add(table(["balanced flag set, by verdict", *cols, "all"],
              [[v, *[cnt[c]["balanced_flag_by_verdict"].get(v, 0) for c in cols],
                cnt["all"]["balanced_flag_by_verdict"].get(v, 0)] for v in verdicts
               if cnt["all"]["balanced_flag_by_verdict"].get(v)]))
    add("")
    add("**Out-of-context rows where both the claimed and the actual value are present** "
        "(a value is present when some `;`-part or list element is not \"not enough information\"; "
        "\"unspecified\" and \"not specified\" count as absent):")
    add("")
    add(table(["item", *cols, "all", "processed train, usable image"],
              [[i, *[cnt[c]["ooc_both_present"][i] for c in cols], cnt["all"]["ooc_both_present"][i],
                ptrain["ooc_both_present_with_usable_image"][i]] for i in ITEMS]))
    add("")
    add("Fact-checking organisations:")
    add("")
    orgs = list(cnt["all"]["fc_org"])
    add(table(["fc_org", *cols, "all"], [[o, *[cnt[c]["fc_org"].get(o, 0) for c in cols],
                                          cnt["all"]["fc_org"][o]] for o in orgs]))
    add("")
    ind = m["india"]
    add(f"**India subset** ({', '.join(INDIA_ORGS)}):")
    add("")
    add(table(["measure", *cols, "all"],
              [[label, *[ind[c][k] for c in cols], ind["all"][k]] for label, k in keys] +
              [[f"OOC both: {i}", *[ind[c]["ooc_both_present"][i] for c in cols],
                ind["all"]["ooc_both_present"][i]] for i in ITEMS] +
              [[f"verdict: {v}", *[ind[c]["verdict"].get(v, 0) for c in cols], ind["all"]["verdict"].get(v, 0)]
               for v in verdicts]))
    add("")
    add("**The paper's claims, checked against the files:**")
    add("")
    add(claims_table(m))
    add("")
    add(f"**Human-validated rows.** `M4FC_human_validated.json` holds {hv['rows']} of the same images "
        f"({', '.join(f'{s} {n}' for s, n in hv['rows_by_split'].items())}) and disagrees with `M4FC.json`. "
        f"Normalised: {hv['normalised_means']}.")
    add("")
    add(table(["field", *[f"raw {s}" for s in OFFICIAL_SPLITS], *[f"normalised {s}" for s in OFFICIAL_SPLITS]],
              [[fld, *[v["raw"][s] for s in OFFICIAL_SPLITS], *[v["normalised"][s] for s in OFFICIAL_SPLITS]]
               for fld, v in hv["fields_differing"].items()]))
    add("")
    add(table(["verdict change (main → validated)", "rows"], list(hv["verdict_changes"].items())))
    add("")

    # 4 ---------------------------------------------------------------
    add("## 4. Overlap (byte-identical images, usable files only)")
    add("")
    add(f"M4FC: {ov['m4fc_rows_with_usable_image']:,} rows with a usable image, "
        f"{ov['m4fc_distinct_sha256']:,} distinct images, every split.")
    add("")
    add(table(["dataset", "shared files on disk", "shared, in processed records", "M4FC rows",
               "M4FC rows by official split (train/dev/test)", "partner records by processed split"],
              [[n, v["shared_images_on_disk"], v["shared_images_in_processed_records"], v["m4fc_rows"],
                " / ".join(str(v["m4fc_rows_by_official_split"][s]) for s in OFFICIAL_SPLITS),
                ", ".join(f"{k} {x}" for k, x in v["partner_records_by_split"].items()) or "—"]
               for n, v in ov["datasets"].items()]))
    add("")
    add(f"Text-only enabled datasets, no images to share: {', '.join(ov['text_only_enabled'])}.")
    add("")
    add(table(["dataset", "images by M4FC official split × partner split",
               "images by M4FC processed split × partner split"],
              [[n, ", ".join(f"{k} {x}" for k, x in v["images_by_official_x_partner_split"].items()) or "—",
                ", ".join(f"{k} {x}" for k, x in v["images_by_processed_x_partner_split"].items()) or "—"]
               for n, v in ov["datasets"].items() if v["shared_images_in_processed_records"]]))
    add("")
    w = ov["within_m4fc"]
    add(f"**Within M4FC.** {w['images_on_two_or_more_rows']} images sit on two or more rows "
        f"({w['rows_in_them']} rows; copies {', '.join(f'{k}: {v}' for k, v in w['copies'].items()) or '—'}). "
        f"Official splits spanned: {', '.join(f'{k} {v}' for k, v in w['official_splits_spanned'].items()) or '—'}. "
        f"Processed splits spanned: {', '.join(f'{k} {v}' for k, v in w['processed_splits_spanned'].items()) or '—'}. "
        "split_all groups an image only at five copies or across corpora, so a pair can straddle.")
    add("")
    add(f"**Drop list: every M4FC official train/dev image that is also in VERITE or AVerImaTeC** "
        f"({len(ov['drop_list'])}; files on disk but in no processed record: "
        f"{', '.join(f'{k} {v}' for k, v in ov['on_disk_only'].items())}). VERITE is named by row index.")
    add("")
    add(table(["M4FC record_id", "official", "processed", "sha256", "partner", "partner record_id"],
              [[x["record_id"], x["official_split"], x["processed_split"], x["sha256"], x["partner"],
                ", ".join(x["partner_record_ids"])] for x in ov["drop_list"]]))
    add("")
    cc = ov["cross_check_with_duplicates_yaml"]
    add(f"Cross-check with `data/cross_corpus_duplicates.yaml`: {cc['yaml_m4fc_records']} M4FC records in "
        f"{cc['yaml_groups_with_m4fc']} groups there, {cc['computed_m4fc_records']} in {cc['computed_groups']} "
        f"computed now; only in the yaml {len(cc['only_in_yaml'])}, only computed {len(cc['only_computed'])}.")
    add("")
    add(f"5Pils in the register (`data/sources.yaml`): {'yes' if ov['fivepils_in_register'] else 'no'}. "
        "Not fetched.")
    add("")

    # 5 ---------------------------------------------------------------
    add("## 5. Shortcut audits")
    add("")
    add(f"Processed train → processed val, verdict-task rows only (`task_verdict_prediction`; left out: "
        f"{', '.join(f'{k} {v}' for k, v in sh['left_out_not_in_the_verdict_task'].items())}, the mixture and "
        f"unverified rows). Rows: train {sh['rows']['train']:,}, val {sh['rows']['val']:,}. Coarse = "
        f"{sh['labels']['coarse']}; fine = {sh['labels']['fine']}. Intervals: {BOOTSTRAP:,} bootstrap resamples "
        "of the scored rows, seed 20260901; the model is fit once, so they cover scoring only. The majority "
        "baseline always predicts the train majority.")
    add("")
    add("### 5a. Organisation and language purity on train")
    add("")
    conc = true_concentration(sh)
    pure = [x["group"] for x in sh["purity"]["fine"]["fc_org"] if x["label_predictive"]]
    add(f"{'Only ' + ', '.join(pure) + ' reaches' if pure else 'No organisation reaches'} {PURITY_FLAG} "
        f"purity on the fine verdict, but the `true` class is concentrated: "
        f"{', '.join(f'{o} {n}' for o, n in conc['top'])} hold {conc['top_rows']} of the {conc['true_rows']} "
        f"train `true` rows ({100 * conc['top_rows'] / conc['true_rows']:.0f}%). An interval on a purity is "
        "the bootstrap distribution of the largest class share, which sits above the estimate when two "
        "classes are close (snopes.com).")
    add("")
    for name in ("fine", "coarse"):
        for key in ("fc_org", "claim_language", "fc_language"):
            rows = sh["purity"][name][key]
            flagged = sum(1 for x in rows if x["label_predictive"])
            add(f"**{key} → {name} verdict** ({flagged} of {len(rows)} groups at purity ≥ {PURITY_FLAG}):")
            add("")
            add(table([key, "rows", "dominant class", "purity [95% CI]", "≥ 0.90", "classes"],
                      [[x["group"], x["rows"], x["dominant"], f"{x['purity']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]",
                        "yes" if x["label_predictive"] else "", ", ".join(f"{k} {v}" for k, v in x["classes"].items())]
                       for x in rows]))
            add("")
    reduced = [f"{key} ({name} verdict)" for name in ("fine", "coarse") for key in ("fc_org", "claim_language")
               if sh["lookup"][name][key]["model"] == sh["lookup"][name][key]["majority"]]
    add(f"Group-only classifiers that reduce exactly to the majority baseline (every group's train majority "
        f"is the overall majority): {', '.join(reduced) or 'none'}.")
    add("")
    for name in ("fine", "coarse"):
        for key in ("fc_org", "claim_language"):
            res = sh["lookup"][name][key]
            add(f"**{key}-only classifier, {name} verdict** (val rows from a group unseen in train: "
                f"{res['unseen_groups_in_val']}):")
            add("")
            add(scored_table(res))
            add("")
    add("### 5b. Claim text only: TF-IDF + logistic regression (imbalanced setting)")
    add("")
    st = sh["text"]["fine"]["settings"]
    add(f"Settings, fixed before val was seen: {'; '.join(f'{k} {v}' for k, v in st.items())}. No tuning; "
        "nothing saved.")
    add("")
    for name in ("coarse", "fine"):
        res = sh["text"][name]
        add(f"**{name} verdict** — {res['train_rows']:,} train rows, {res['features']:,} features; "
            f"fit converged: {res['fit']['converged']} ({res['fit']['iterations']} iterations). "
            f"Train classes: {', '.join(f'{k} {v}' for k, v in res['train_classes'].items())}.")
        add("")
        add(scored_table(res))
        add("")
    add("### 5c. Synthetic true captions against real false claims")
    add("")
    sy = sh["synthetic_vs_real"]
    add(f"{sy['rule']}. {sy['train_rows']:,} train rows, {sy['features']:,} features; train classes "
        f"{', '.join(f'{k} {v}' for k, v in sy['train_classes'].items())}; val "
        f"{', '.join(f'{k} {v}' for k, v in sy['val_classes'].items())}.")
    add("")
    add(scored_table(sy))
    add("")

    # 6 ---------------------------------------------------------------
    add("## 6. Evidence coverage")
    add("")
    if not ev:
        add("Not run yet: `python scripts/m4fc_audit.py evidence`.")
    else:
        add(f"{ev['queries_run']} of {ev['of']} processed-train claims, seed 20260901. Query = the English "
            f"`claim`; as-of = `fc_pub_date[:10]`, inclusive; `retrieve.search(k={ev['k']})` on one read-only "
            "connection. Entities: `dslim/bert-base-NER` at the gallery's pinned revision, every type it tags, "
            f"matched whole-word and case-folded in the title and text of the top {ev['entity_top']} hits.")
        add("")
        add("**Read the hit counts with care:** `retrieve._sanitise` joins every word of the claim with OR, so "
            "a hit means the document shares a word with the claim, not that it is evidence. The entity "
            "matches are the stricter measure. The same OR is why a search takes about "
            f"{ev['all']['median_search_seconds']:.0f} s here: BM25 scores every document that shares any "
            "word with the claim before the 20 best are kept.")
        if ev["queries_run"] < ev["of"]:
            add("")
            add(f"**Partial: {ev['queries_run']} of {ev['of']} claims run.** The rest run with "
                "`python scripts/m4fc_audit.py evidence` (resumable), then `report` again.")
        add("")
        subsets = [s for s in ("all", "india") if ev.get(s, {}).get("queries")]
        def row(label, fn):
            return [label, *[fn(ev[s]) for s in subsets]]
        add(table(["", *[f"{s} ({ev[s]['queries']})" for s in subsets]], [
            row("≥ 1 hit", lambda x: pct(x["with_a_hit"], x["queries"])),
            *[row(f"hits: {b}", lambda x, b=b: x["hit_counts"][b]) for b in ("0", "1-4", "5-9", "10-19", "20")],
            row("claims with ≥ 1 entity", lambda x: pct(x["claims_with_entities"]["1+"], x["queries"])),
            row("claims with ≥ 2 entities", lambda x: pct(x["claims_with_entities"]["2+"], x["queries"])),
            row(f"≥ 1 entity in the top {ev['entity_top']}", lambda x: pct(x["entity_matches_top5"]["1+"], x["queries"])),
            row(f"≥ 2 entities in the top {ev['entity_top']}", lambda x: pct(x["entity_matches_top5"]["2+"], x["queries"])),
            row("… of claims with ≥ 1 entity", lambda x: pct(x["entity_matches_top5"]["1+"], x["claims_with_entities"]["1+"])),
            row("… of claims with ≥ 2 entities", lambda x: pct(x["entity_matches_top5"]["2+"], x["claims_with_entities"]["2+"])),
            row("hits dated after as-of (must be 0)", lambda x: x["after_as_of"]),
            row("queries with a hit on an M4FC fact-checker's domain", lambda x: x["queries_with_a_fact_check_domain_hit"]),
            row("… on the claim's own fact-checker's domain", lambda x: x["queries_with_an_own_org_hit"]),
            row("median seconds per search", lambda x: x["median_search_seconds"]),
        ]))
        add("")
        add(f"Top-1 BM25 score ({ev['bm25_sign']}):")
        add("")
        qs = ("min", "p10", "p25", "median", "p75", "p90", "max")
        add(table(["subset", *qs], [[s, *[ev[s]["top1_bm25"][q] for q in qs]] for s in subsets
                                    if ev[s].get("top1_bm25")]))
        add("")
        add(table(["entity type", *[f"{s}: matched / found" for s in subsets]],
                  [[t, *[f"{ev[s]['entities_by_type'].get(t, {}).get('matched', 0)} / "
                         f"{ev[s]['entities_by_type'].get(t, {}).get('entities', 0)}" for s in subsets]]
                   for t in sorted({t for s in subsets for t in ev[s]["entities_by_type"]})]))
        add("")
        add("Some M4FC organisations are news publishers with a fact-check desk (indiatoday.in, larepublica.pe, "
            "knack.be), so a hit on their domain is not necessarily a fact-check.")
        add("")

    # 7 ---------------------------------------------------------------
    add("## 7. Review sheet")
    add("")
    add(f"**`{sheet['sheet']}`** (never overwritten once written): {sheet['instances']} instances, "
        f"{sheet['sheet_rows']} rows, "
        f"one per item with both values ({', '.join(f'{k} {v}' for k, v in sheet['rows_by_item'].items())}). "
        f"Images are neutral byte-copies in `{sheet['images']}/`; `private_map.json` beside the sheet ties "
        "each instance to its record. Columns: " + ", ".join(f"`{c}`" for c in SHEET_COLUMNS) +
        "; `mismatch` and `note` are empty.")
    add("")
    add("Eligible: processed-train out-of-context rows with a usable image and at least one item with both "
        f"values. Seed 20260901; one instance per org, the rest by largest remainder in proportion to eligible "
        f"rows, at most {SHEET_CAP} per org; sheet order shuffled. {sheet['instances_in_human_validated']} of the "
        "instances are also in the authors' human-validated file (not shown on the sheet).")
    add("")
    add(table(["fc_org", "eligible rows", "instances"],
              [[o, n, sheet["instances_by_org"].get(o, 0)] for o, n in sheet["eligible_rows_by_org"].items()]))
    add("")

    # 8 ---------------------------------------------------------------
    add("## 8. Docs")
    add("")
    add("No sentence in CLAUDE.md, `docs/` (the gallery write-up included) or the status doc said that nobody "
        "does time-filtered evidence or reverse image search, so there was nothing to replace. The paragraph "
        "and the COVE entry are in the new `docs/related_work.md`, with pointers from CLAUDE.md, "
        "`docs/evidence_store.md` and `docs/README.md` (local commit).")
    add("")

    # open issues -----------------------------------------------------
    add("## Open issues (reported, nothing changed)")
    add("")
    add(f"- **M4FC's temporal split is broken in processed test.** {oi['processed_test_rows_before_the_test_cutoff']:,} "
        f"of the {oi['processed_test_rows']:,} processed-test rows have `fc_pub_date` before {TEST_STARTS} "
        f"(dates {oi['processed_test_dates'][0]} to {oi['processed_test_dates'][1]}): they are train-period rows "
        "moved there by the atomicity rule. A model scored on processed test is not scored on a temporal "
        f"holdout. Processed val holds {oi['processed_val_rows_before_the_dev_cutoff']} row(s) dated before {DEV_STARTS}.")
    add(f"- **The fields this audit needs are not in the processed layer** ({len(f['dropped'])} dropped). "
        "A training run would need an interim + processed rebuild.")
    sl = oi["slugs_stating_a_finding"]
    add(f"- **M4FC record ids are article slugs, and many state the finding.** {sl['all']:,} of {f['raw']['rows']:,} "
        f"ids ({sl['processed_train']:,} of {sl['processed_train_rows']:,} in processed train) contain one of: "
        f"{', '.join(sl['words'])}. They are already published in `data/processed/splits/m4fc_*.csv` and the "
        "duplicates yaml. No model may see them.")
    add("- **Human validation changes the labels this block would use**: see section 3.")
    add("")
    return "\n".join(L) + "\n"


def true_concentration(sh: dict[str, Any]) -> dict[str, Any]:
    """How many of the train `true` rows the two largest holders of them carry."""
    holders = Counter({x["group"]: x["classes"].get("true", 0) for x in sh["purity"]["fine"]["fc_org"]})
    top = holders.most_common(2)
    return {"top": top, "top_rows": sum(n for _o, n in top), "true_rows": sum(holders.values())}


def claims_table(m: dict[str, Any]) -> str:
    f, it, cnt = m["fields"], m["items"], m["counts"]["all"]
    b = f["split"]["bounds"]
    ooc = cnt["verdict"].get(OOC, 0)
    flagged_ooc = cnt["balanced_flag_by_verdict"].get(OOC, 0)
    rows = [
        ["temporal split, cutoffs 2022-09-22 / 2023-01-18 / 2024-09-01",
         f"train {b['train']['first']}–{b['train']['last']}; dev {b['dev']['first']}–{b['dev']['last']}; "
         f"test {b['test']['first']}–{b['test']['last']}; boundary days shared", "holds"],
        ["verdicts true / false / mixture / unverified, false in three subtypes",
         ", ".join(f"{k} {v:,}" for k, v in cnt["verdict"].items()), "holds"],
        ["verdict_coarse is the verdict task's label",
         ", ".join(f"{k} {v}" for k, v in it["verdict_x_coarse"].items()),
         f"holds; {len(it['verdict_true_coarse_false'])} row with verdict true has coarse false"],
        ["claimed_{date,location,people,things,event} and the actual items exist",
         f"all {f['raw']['rows']:,} rows carry all of them", "holds (who extracted them cannot be checked from files)"],
        ["dates normalised to YYYY / YYYY-MM / YYYY-MM-DD, ';'-separated",
         "; ".join(f"{fld}: " + ", ".join(f"{k} {v:,}" for k, v in it[fld]["shapes"].items())
                   for fld in ("claimed_date", "date")), "mostly; ranges and free text are the exceptions"],
        ["\"not enough information\" where absent",
         ", ".join(f"\"{k}\" {v:,}" for k, v in it["absent_markers"].items()), "holds, with 2 other wordings"],
        ["people is a ';'-separated value like the others",
         ", ".join(f"{k} {v:,}" for k, v in it["value_types"]["people"].items()), "differs: a list in many rows"],
        ["the balanced setting swaps half the out-of-context claims for synthetic true captions",
         f"flag on {flagged_ooc:,} of {ooc:,} out-of-context rows ({100 * flagged_ooc / ooc:.1f}%), also "
         + ", ".join(f"{k} {v}" for k, v in cnt["balanced_flag_by_verdict"].items() if k != OOC)
         + f"; {cnt['balanced_flag_without_caption']} flagged rows have no caption", "differs"],
        ["an English claim for every row",
         f"{pct(f['english_claim']['rows_with_a_claim'], f['english_claim']['rows'])} present; "
         f"{f['english_claim']['claims_mostly_non_latin']} mostly non-Latin (untranslated)",
         f"holds for {f['english_claim']['rows_with_a_claim'] - f['english_claim']['claims_mostly_non_latin']:,}"],
    ]
    return table(["claim (paper, as relayed)", "measured", "verdict"], rows)


def summary_lines(m: dict[str, Any], ev: dict[str, Any] | None) -> str:
    d, f, cnt, ov, sh, oi, hv = (m["definitions"], m["fields"], m["counts"], m["overlap"], m["shortcuts"],
                                 m["open_issues"], m["human_validated"])
    pt, pv = cnt["processed train"], cnt["processed val"]
    vt = d["rows_carrying_the_codes"].get("verite", [])
    vmm = sum(x["rows"] for x in vt if x["reason_code"] == "media_mismatch")
    vwi = sum(x["rows"] for x in vt if x["reason_code"] == "wrong_image")
    fine = sh["text"]["fine"]
    syn = sh["synthetic_vs_real"]
    org = sh["purity"]["fine"]["fc_org"]
    ooc_all = cnt["all"]["verdict"].get(OOC, 0)
    flag = cnt["all"]["balanced_flag_by_verdict"].get(OOC, 0)
    datasets = ov["datasets"]
    lines = [
        f"1. Trainable rows today: `media_mismatch` {d['trainable_rows']['media_mismatch']}, `wrong_image` "
        f"{d['trainable_rows']['wrong_image']}. Only VERITE carries them ({vmm} / {vwi}), evaluation-only; M4FC "
        "maps every false row, out-of-context included, to `content_refuted`.",
        f"2. Processed M4FC keeps {len(f['raw']['keys']) - len(f['dropped'])} of {len(f['raw']['keys'])} fields: the "
        "false subtypes and every claimed/actual item come from raw here; using them in training needs a rebuild.",
        f"3. The split is M4FC's except {f['split']['moved']} official-train rows moved out of train by atomicity; "
        f"{oi['processed_test_rows_before_the_test_cutoff']} of {oi['processed_test_rows']} processed-test rows "
        f"predate {TEST_STARTS} (open issue).",
        f"4. Out-of-context rows: processed train {pt['ooc_rows']:,}, val {pv['ooc_rows']:,}. The balanced flag "
        f"covers {flag:,} of {ooc_all:,} ({100 * flag / ooc_all:.0f}%), not half.",
        f"5. Org purity ≥ 0.90 on the fine verdict: {sum(x['label_predictive'] for x in org)} of {len(org)} orgs, "
        f"but {true_concentration(sh)['top_rows']} of {true_concentration(sh)['true_rows']} train `true` rows "
        f"are {' and '.join(o for o, _n in true_concentration(sh)['top'])}. Claim text alone, fine verdict: "
        f"macro-F1 {ci(fine['model']['macro_f1'])} against {fine['majority']['macro_f1']['value']:.3f} for the majority.",
        f"6. Synthetic true captions against real false claims: macro-F1 {ci(syn['model']['macro_f1'])}, "
        f"accuracy {ci(syn['model']['accuracy'])} (majority {syn['majority']['accuracy']['value']:.3f}).",
        "7. M4FC rows sharing an image: " + ", ".join(
            f"{n} {v['m4fc_rows']}" for n, v in sorted(datasets.items(), key=lambda kv: -kv[1]["m4fc_rows"])
            if v["m4fc_rows"]) + "; with " + ", ".join(n for n, v in sorted(datasets.items()) if not v["m4fc_rows"])
        + f" none. Drop list (official train/dev in VERITE or AVerImaTeC): {len(ov['drop_list'])}. "
        "5Pils is not in the register.",
    ]
    if ev and ev.get("all", {}).get("queries"):
        a = ev["all"]
        i = ev.get("india", {})
        partial = "" if ev["queries_run"] >= ev["of"] else f" (PARTIAL: {ev['queries_run']} of {ev['of']} run)"
        lines.append(
            f"8. Evidence{partial}, {a['queries']} train claims: ≥ 1 hit {pct(a['with_a_hit'], a['queries'])}; ≥ 1 claim entity "
            f"in the top {ev['entity_top']} {pct(a['entity_matches_top5']['1+'], a['queries'])}, ≥ 2 "
            f"{pct(a['entity_matches_top5']['2+'], a['queries'])}"
            + (f"; India {pct(i['entity_matches_top5']['1+'], i['queries'])} and "
               f"{pct(i['entity_matches_top5']['2+'], i['queries'])}." if i.get("queries") else "."))
    else:
        lines.append("8. Evidence coverage: not run yet.")
    changed = sum(hv["verdict_changes"].values())
    lines.append(f"9. The authors' human validation changed the verdict on {changed} of {hv['rows']} rows; "
                 f"review sheet: {m['sheet']['instances']} instances, {m['sheet']['sheet_rows']} rows, "
                 f"`{m['sheet']['sheet']}`.")
    both_train = pt["ooc_both_present"]
    lines.append("10. Items with both a claimed and an actual value, out-of-context, processed train: "
                 + " · ".join(f"{i} {both_train[i]:,}" for i in ITEMS) + ".")
    return "\n".join(lines)


def report() -> int:
    if not MEASURE.is_file():
        raise SystemExit(f"no {MEASURE}: run `python scripts/m4fc_audit.py measure` first")
    m = json.loads(MEASURE.read_text(encoding="utf-8"))
    ev = evidence_summary()
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render(m, ev), encoding="utf-8")
    write_json(REPORT_JSON, {"measure": m, "evidence": ev})
    print(f"  wrote {REPORT_MD.relative_to(PROJECT_ROOT).as_posix()} and "
          f"{REPORT_JSON.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


# --------------------------------------------------------------------------
# the review page: the sheet, answered in a browser
# --------------------------------------------------------------------------

#: The person's rule, shown verbatim at the top of the page.
REVIEW_RULE = ("yes = the claim's value contradicts the actual one; no = consistent "
               "(same, or one is just more specific); unclear = can't tell.")
EXPORT_NAME = "m4fc_ooc_items_reviewed.csv"
IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}

REVIEW_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M4FC sheet review</title>
<style>
:root {
  --bg: #f5f5f2; --surface: #ffffff; --text: #1c1c1a; --muted: #686862; --line: #deddd7;
  --accent: #2b5c8a; --on-accent: #ffffff; --flash: #c98f00; --warn-bg: #fff3cd; --warn-text: #5c4400;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141413; --surface: #1e1e1c; --text: #ebeae5; --muted: #a2a19a; --line: #3a3935;
    --accent: #86b4e2; --on-accent: #0c1a28; --flash: #e8b931; --warn-bg: #3a2f0b; --warn-text: #f3dd9b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--surface); border-bottom: 1px solid var(--line); }
.bar { max-width: 1120px; margin: 0 auto; padding: 10px 16px; display: flex; flex-wrap: wrap;
       gap: 8px 16px; align-items: center; }
h1 { font-size: 17px; margin: 0; font-weight: 600; }
.progress { display: flex; align-items: center; gap: 8px; font-variant-numeric: tabular-nums; }
.meter { width: 160px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.meter > i { display: block; height: 100%; width: 0; background: var(--accent); }
.spacer { flex: 1; }
.btn { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line);
       background: var(--surface); color: var(--text); cursor: pointer; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
.rule { flex: 1 1 100%; margin: 0; font-size: 14px; font-weight: 600; }
.help { flex: 1 1 100%; margin: 0; color: var(--muted); font-size: 13px; }
.warn { display: none; max-width: 1088px; margin: 12px auto 0; padding: 8px 16px;
        background: var(--warn-bg); color: var(--warn-text); border-radius: 6px; }
main { max-width: 1120px; margin: 0 auto; padding: 4px 16px 64px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
        padding: 14px 16px; margin: 16px 0; }
.card-head { display: flex; gap: 10px; align-items: baseline; color: var(--muted); font-size: 13px; }
.card-head .done { margin-left: auto; }
figure { margin: 10px 0; text-align: center; }
figure img { max-width: 100%; max-height: 60vh; cursor: zoom-in; border-radius: 4px; }
figure.zoom img { max-height: none; cursor: zoom-out; }
.claim { font-size: 16px; margin: 8px 0 12px; overflow-wrap: anywhere; }
.row { display: grid; grid-template-columns: 6.5rem minmax(0, 1fr) minmax(0, 1fr) auto minmax(10rem, 14rem);
       gap: 8px 12px; align-items: start; padding: 10px 0; border-top: 1px solid var(--line); }
.item { font-weight: 600; }
.lbl { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.val { overflow-wrap: anywhere; }
.choice { display: flex; gap: 6px; }
.opt { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line);
       background: transparent; color: var(--text); cursor: pointer; }
.opt[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
textarea { font: inherit; width: 100%; resize: vertical; min-height: 2.1em; padding: 4px 6px;
           border: 1px solid var(--line); border-radius: 6px; background: transparent; color: var(--text); }
.row.flash { outline: 2px solid var(--flash); outline-offset: 4px; border-radius: 4px; }
:focus-visible { outline: 2px solid var(--flash); outline-offset: 2px; }
@media (max-width: 860px) {
  .row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .item, .choice, textarea { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>M4FC sheet review</h1>
    <div class="progress"><span id="progress"></span><span class="meter"><i id="meter"></i></span></div>
    <span class="spacer"></span>
    <button class="btn" id="next" type="button">Next unanswered</button>
    <button class="btn primary" id="export" type="button">Export CSV</button>
    <p class="rule">__RULE__</p>
    <p class="help" id="help">Click a chosen button again to clear it. Answers stay in this browser; Export writes the sheet's eight columns.</p>
  </div>
</header>
<div class="warn" id="warn">This browser does not let the page keep answers: export before you close it.</div>
<noscript><p style="padding:16px">This page needs JavaScript.</p></noscript>
<main id="instances"></main>
<script type="application/json" id="sheet-data">__DATA__</script>
<script id="csv-lib">
// The sheet's own format: a BOM, CRLF line ends, a field quoted only when it
// holds a comma, a quote or a line break -- as Python's csv module writes it.
function buildCsv(columns, rows, answers) {
  var field = function (s) {
    s = s == null ? "" : String(s);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  var lines = [columns.map(field).join(",")];
  rows.forEach(function (r) {
    var a = answers[r.sheet_id] || {};
    lines.push([r.sheet_id, r.image, r.claim, r.item, r.claimed, r.actual, a.m || "", a.n || ""]
      .map(field).join(","));
  });
  return "﻿" + lines.join("\r\n") + "\r\n";
}
</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("sheet-data").textContent);
  var VALUES = ["yes", "no", "unclear"];
  var KEY = "m4fc-review:" + DATA.sheet_sha256;
  var total = DATA.rows.length;
  var rowsById = new Map(DATA.rows.map(function (r) { return [r.sheet_id, r]; }));
  var rowEls = new Map(), cardOf = new Map(), cards = [];
  var answers = {}, storage = null;

  try {
    storage = window.localStorage;
    var saved = JSON.parse(storage.getItem(KEY) || "{}");
    if (saved && typeof saved === "object") answers = saved;
  } catch (e) { storage = null; }
  // keep only this sheet's rows, and only the three answers
  Object.keys(answers).forEach(function (sid) {
    var a = answers[sid];
    if (!rowsById.has(sid) || !a || typeof a !== "object") { delete answers[sid]; return; }
    if (VALUES.indexOf(a.m) < 0) delete a.m;
    if (typeof a.n !== "string" || a.n === "") delete a.n;
    if (a.m === undefined && a.n === undefined) delete answers[sid];
  });

  function warn() { document.getElementById("warn").style.display = "block"; }
  function save() {
    if (!storage) { warn(); return; }
    try { storage.setItem(KEY, JSON.stringify(answers)); } catch (e) { storage = null; warn(); }
  }
  if (!storage) warn();

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function answered(sid) { return !!(answers[sid] && answers[sid].m); }
  function tidy(sid) {
    var a = answers[sid];
    if (a && a.m === undefined && a.n === undefined) delete answers[sid];
  }

  function paintRow(sid) {
    var m = answers[sid] && answers[sid].m;
    rowEls.get(sid).querySelectorAll(".opt").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.v === m ? "true" : "false");
    });
  }
  function paintCard(card) {
    var n = card.items.filter(answered).length;
    card.done.textContent = n === card.items.length ? "done" : n + " of " + card.items.length;
  }
  function paintProgress() {
    var n = DATA.rows.filter(function (r) { return answered(r.sheet_id); }).length;
    document.getElementById("progress").textContent = n + " of " + total;
    document.getElementById("meter").style.width = (100 * n / total) + "%";
  }

  function choose(sid, v) {
    var a = answers[sid] || (answers[sid] = {});
    if (a.m === v) delete a.m; else a.m = v;
    tidy(sid);
    paintRow(sid); paintCard(cardOf.get(sid)); paintProgress(); save();
  }

  function itemRow(r) {
    var row = el("div", "row");
    row.dataset.sid = r.sheet_id;
    row.appendChild(el("div", "item", r.item));
    [["claimed", r.claimed], ["actual", r.actual]].forEach(function (pair) {
      var v = el("div", "val");
      v.appendChild(el("span", "lbl", pair[0]));
      v.appendChild(el("span", "txt", pair[1]));
      row.appendChild(v);
    });
    var choice = el("div", "choice");
    choice.setAttribute("role", "group");
    choice.setAttribute("aria-label", r.item + ": does the claim contradict the actual value?");
    VALUES.forEach(function (v) {
      var b = el("button", "opt", v);
      b.type = "button";
      b.dataset.v = v;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", function () { choose(r.sheet_id, v); });
      choice.appendChild(b);
    });
    row.appendChild(choice);
    var note = document.createElement("textarea");
    note.rows = 1;
    note.placeholder = "note";
    note.setAttribute("aria-label", r.item + ": note");
    note.value = (answers[r.sheet_id] && answers[r.sheet_id].n) || "";
    note.addEventListener("input", function () {
      var a = answers[r.sheet_id] || (answers[r.sheet_id] = {});
      if (note.value) a.n = note.value; else delete a.n;
      tidy(r.sheet_id);
      save();
    });
    row.appendChild(note);
    rowEls.set(r.sheet_id, row);
    return row;
  }

  var main = document.getElementById("instances");
  DATA.instances.forEach(function (inst, i) {
    var sec = el("section", "card");
    sec.id = inst.id;
    var head = el("div", "card-head");
    head.appendChild(el("span", "", (i + 1) + " of " + DATA.instances.length));
    head.appendChild(el("span", "", inst.id));
    var done = el("span", "done");
    head.appendChild(done);
    sec.appendChild(head);
    var fig = el("figure");
    var img = new Image();
    img.decoding = "async";
    img.alt = "image " + inst.id;
    img.src = inst.image;
    img.addEventListener("click", function () { fig.classList.toggle("zoom"); });
    fig.appendChild(img);
    sec.appendChild(fig);
    sec.appendChild(el("p", "claim", inst.claim));
    var card = { items: inst.items, done: done };
    inst.items.forEach(function (sid) {
      sec.appendChild(itemRow(rowsById.get(sid)));
      cardOf.set(sid, card);
    });
    cards.push(card);
    main.appendChild(sec);
  });
  DATA.rows.forEach(function (r) { paintRow(r.sheet_id); });
  cards.forEach(paintCard);
  paintProgress();

  document.getElementById("next").addEventListener("click", function () {
    var next = DATA.rows.find(function (r) { return !answered(r.sheet_id); });
    var help = document.getElementById("help");
    if (!next) { help.textContent = "All " + total + " answered."; return; }
    var row = rowEls.get(next.sheet_id);
    row.scrollIntoView({ block: "center", behavior: "smooth" });
    row.classList.add("flash");
    setTimeout(function () { row.classList.remove("flash"); }, 1600);
    row.querySelector(".opt").focus({ preventScroll: true });
  });

  document.getElementById("export").addEventListener("click", function () {
    var blob = new Blob([buildCsv(DATA.columns, DATA.rows, answers)], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = DATA.export_name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    var n = DATA.rows.filter(function (r) { return answered(r.sheet_id); }).length;
    document.getElementById("help").textContent =
      "Exported " + DATA.export_name + ": " + n + " of " + total + " answered.";
  });
})();
</script>
</body>
</html>
"""


def write_review_page() -> Path:
    """review.html beside the sheet: one self-contained file in which a person
    answers it. It reads the sheet and its neutral image copies and nothing
    else, so no record_id, org or verdict can reach it."""
    import base64
    import html

    raw = SHEET.read_bytes()
    with SHEET.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != SHEET_COLUMNS:
        raise SystemExit(f"{SHEET}: not the sheet's {len(SHEET_COLUMNS)} columns")
    instances: list[dict[str, Any]] = []
    for r in rows:
        iid = r["sheet_id"].rsplit("-", 1)[0]
        if instances and instances[-1]["id"] == iid:
            if (r["image"], r["claim"]) != instances[-1]["_key"]:
                raise SystemExit(f"{r['sheet_id']}: another image or claim inside one instance")
            instances[-1]["items"].append(r["sheet_id"])
            continue
        if any(x["id"] == iid for x in instances):
            raise SystemExit(f"{iid}: its rows are not together in the sheet")
        path = PROJECT_ROOT / r["image"]
        mime = IMAGE_MIME[path.suffix.lower()]
        instances.append({"id": iid, "_key": (r["image"], r["claim"]), "claim": r["claim"],
                          "image": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}",
                          "items": [r["sheet_id"]]})
    data = {"sheet_sha256": hashlib.sha256(raw).hexdigest(), "export_name": EXPORT_NAME,
            "columns": list(SHEET_COLUMNS),
            "rows": [{k: r[k] for k in SHEET_COLUMNS[:6]} for r in rows],
            "instances": [{k: v for k, v in x.items() if k != "_key"} for x in instances]}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = REVIEW_TEMPLATE.replace("__RULE__", html.escape(REVIEW_RULE)).replace("__DATA__", payload)
    REVIEW_PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"  {REVIEW_PAGE.relative_to(PROJECT_ROOT).as_posix()}: {len(instances)} instances, "
          f"{len(rows)} rows, {REVIEW_PAGE.stat().st_size / 2**20:.1f} MiB")
    return REVIEW_PAGE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure", help="sections 1-5 and 7: offline, light")
    e = sub.add_parser("evidence", help="section 6: claims against the evidence store; resumable")
    e.add_argument("--limit", type=int, default=None,
                   help="run at most this many more claims (the timing probe)")
    sub.add_parser("report", help="write data/reports/m4fc_audit.md and .json")
    sub.add_parser("review-page", help="write review/review.html: the sheet, answered in a browser")
    args = p.parse_args(argv)
    if args.cmd == "measure":
        measure()
        return 0
    if args.cmd == "evidence":
        return run_evidence(args.limit)
    if args.cmd == "review-page":
        write_review_page()
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
