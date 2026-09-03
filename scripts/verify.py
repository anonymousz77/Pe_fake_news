#!/usr/bin/env python
"""Measure what we actually hold, and check it against the register.

    python scripts/verify.py --all
    python scripts/verify.py --dataset averitec --evidence-sample 0

Writes ``data/reports/<name>.json`` per dataset: file and row counts, label
distribution, split sizes, date range where a date field exists, and the
fraction of records carrying text only, image only, or both.

For the evidence-bearing corpora (mocheg, averitec, averimatec, factify2) it
also reports how many records carry at least one evidence item and how many
evidence items have a **recoverable date**. That last number matters now rather
than later: time-filtered retrieval is only possible for evidence whose date is
known, so its coverage decides what the evidence store can contain.

A date is counted as either **explicit** (a real date field) or **URL-derived**
(a ``/YYYY-MM-DD/`` or ``/YYYY/MM/DD/`` path segment). The two are counted
separately and never merged, because only the first is authoritative — a date
in a URL is the publisher's filing convention, not a guarantee.

Every measurement is compared against ``expected`` in ``data/sources.yaml``.
Mismatches are collected into a top-level ``mismatches`` list and printed
loudly. Nothing is silently reconciled.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import PROJECT_ROOT, REPORTS, dataset, raw_dir  # noqa: E402
from scripts.fetchlib import GB, enabled_names, utcnow  # noqa: E402
from scripts.hydrate import existing_destination, read_rows  # noqa: E402
from scripts.manifest import data_files  # noqa: E402

csv.field_size_limit(64 * 1024 * 1024)

#: Default number of AVeriTeC knowledge-store claim files to scan. The dev
#: store holds ~1.14M evidence items in a 10.7 GB zip; a bounded scan keeps the
#: report cheap, and the sample size is always recorded so a sample is never
#: presented as a census. 0 means scan everything.
DEFAULT_EVIDENCE_SAMPLE = 100

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_URL_DASH = re.compile(r"/(\d{4})-(\d{2})-(\d{2})(?:/|\b)")
_URL_SLASH = re.compile(r"/(\d{4})/(\d{2})/(\d{2})(?:/|\b)")
_DMY = re.compile(r"\b(\d{1,2})\s+(\w+)\s+(\d{4})\b")

#: AVeriTeC writes claim_date day-first and unpadded: "25-8-2020". Day-first
#: is not an assumption -- 1,712 of its train values have a first component
#: above 12, which can only parse as a day.
_DMY_NUM = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def iso_date(value: Any) -> str | None:
    """Normalise an explicit date value to YYYY-MM-DD, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "-"}:
        return None
    m = _ISO.search(text)
    if m:
        return _valid(*m.groups())
    m = _DMY.search(text)                       # "17 March 2023"
    if m and m.group(2).lower() in _MONTHS:
        return _valid(m.group(3), f"{_MONTHS[m.group(2).lower()]:02d}", m.group(1).zfill(2))
    m = _DMY_NUM.match(text)                    # "25-8-2020", day-first
    if m:
        day, month, year = m.groups()
        return _valid(year, month.zfill(2), day.zfill(2))
    if text.replace(".", "").isdigit():          # unix timestamp (fakeddit)
        try:
            ts = float(text)
            if 10**8 < ts < 4 * 10**9:
                return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return None
    return None


def date_from_url(url: str | None) -> str | None:
    """A date recoverable from a URL path, or None.

    Deliberately kept apart from iso_date: this is the publisher's filing
    convention, not an asserted publication date, and the report never merges
    the two counts.
    """
    if not url:
        return None
    for pattern in (_URL_DASH, _URL_SLASH):
        m = pattern.search(url)
        if m:
            got = _valid(*m.groups())
            if got:
                return got
    return None


def _valid(year: str, month: str, day: str) -> str | None:
    try:
        datetime(int(year), int(month), int(day))
    except ValueError:
        return None
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


class Dates:
    """Explicit and URL-derived date counts, kept separate throughout."""

    def __init__(self, field: str | None = None):
        self.field = field
        self.explicit = 0
        self.url_derived = 0
        self.total = 0
        self.seen: list[str] = []

    def add_explicit(self, value: Any) -> None:
        self.total += 1
        got = iso_date(value)
        if got:
            self.explicit += 1
            self.seen.append(got)

    def add_url(self, url: str | None) -> None:
        self.total += 1
        got = date_from_url(url)
        if got:
            self.url_derived += 1
            self.seen.append(got)

    def to_dict(self) -> dict[str, Any]:
        any_date = self.explicit + self.url_derived
        return {
            "field": self.field,
            "items": self.total,
            "with_explicit_date": self.explicit,
            "with_url_derived_date": self.url_derived,
            "explicit_coverage": round(self.explicit / self.total, 4) if self.total else 0.0,
            "url_derived_coverage": round(self.url_derived / self.total, 4) if self.total else 0.0,
            "any_coverage": round(any_date / self.total, 4) if self.total else 0.0,
            "range": {"min": min(self.seen), "max": max(self.seen)} if self.seen else None,
        }


class Modality:
    """Records by which modalities are actually present on disk."""

    def __init__(self):
        self.text_only = self.image_only = self.both = self.neither = 0

    def add(self, has_text: bool, has_image: bool) -> None:
        if has_text and has_image:
            self.both += 1
        elif has_text:
            self.text_only += 1
        elif has_image:
            self.image_only += 1
        else:
            self.neither += 1

    def to_dict(self) -> dict[str, Any]:
        total = self.text_only + self.image_only + self.both + self.neither
        f = (lambda n: round(n / total, 4)) if total else (lambda n: 0.0)
        return {
            "records": total,
            "text_only": self.text_only,
            "image_only": self.image_only,
            "both": self.both,
            "neither": self.neither,
            "fractions": {
                "text_only": f(self.text_only),
                "image_only": f(self.image_only),
                "both": f(self.both),
                "neither": f(self.neither),
            },
        }


# --------------------------------------------------------------------------
# expected-vs-measured
# --------------------------------------------------------------------------

#: Register `expected` keys that name a label under a different spelling.
EXPECTED_LABEL_ALIASES: dict[str, dict[str, str]] = {
    "verite": {"true": "true", "ooc": "out-of-context", "miscaptioned": "miscaptioned"},
    "welfake": {"real": "0", "fake": "1"},
    "isot": {"real": "real", "fake": "fake"},
}

#: Register `expected` keys that name a row count rather than a label.
ROW_KEYS = {"rows", "usable_rows", "csv_rows"}


def compare_expected(name: str, rows: int,
                     labels: dict[str, int]) -> list[dict[str, Any]]:
    """Every disagreement between the register and what we measured."""
    entry = dataset(name)
    expected = entry.get("expected") or {}
    aliases = EXPECTED_LABEL_ALIASES.get(name, {})
    out: list[dict[str, Any]] = []

    for key, want in expected.items():
        if key in ROW_KEYS:
            got, what = rows, "row count"
        else:
            got, what = labels.get(aliases.get(key, key)), f"label {key!r}"
        if got is None:
            out.append({"field": key, "expected": want, "measured": None,
                        "detail": f"{what} not found in the measured data"})
        elif got != want:
            out.append({"field": key, "expected": want, "measured": got,
                        "difference": got - want,
                        "detail": f"{what} differs by {got - want:+,}"})
    return out


# --------------------------------------------------------------------------
# per-dataset profilers
# --------------------------------------------------------------------------


def _files(name: str) -> tuple[int, int]:
    root = raw_dir(name)
    if not root.is_dir():
        return 0, 0
    paths = list(data_files(root))
    return len(paths), sum(p.stat().st_size for p in paths)


def _blank(value: Any) -> bool:
    return not str(value or "").strip()


def profile_liar(name: str, _sample: int) -> dict[str, Any]:
    """Headerless 14-column TSV; label is column 1, statement column 2."""
    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    for split, fname in (("train", "train.tsv"), ("val", "valid.tsv"), ("test", "test.tsv")):
        path = raw_dir(name) / fname
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="", errors="replace") as fh:
            rows = [r for r in csv.reader(fh, delimiter="\t") if r]
        by_split[split] = len(rows)
        for r in rows:
            labels[r[1]] += 1
            modality.add(has_text=not _blank(r[2] if len(r) > 2 else ""), has_image=False)
    return {"rows": sum(by_split.values()), "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(),
            "notes": ["headerless TSV; label is column 1, statement column 2",
                      "text-only corpus: no image field exists"]}


def profile_welfake(name: str, _sample: int) -> dict[str, Any]:
    path = next(raw_dir(name).rglob("*.csv"))
    header, rows = read_rows(path)
    labels = Counter(r["label"] for r in rows)
    modality = Modality()
    for r in rows:
        modality.add(has_text=not (_blank(r.get("title")) and _blank(r.get("text"))),
                     has_image=False)
    return {"rows": len(rows), "by_split": {"all": len(rows)},
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(), "columns": header,
            "notes": ["single undivided CSV: no train/val/test split shipped",
                      "label 0 = real, 1 = fake", "text-only corpus"]}


def profile_verite(name: str, _sample: int) -> dict[str, Any]:
    path = next(raw_dir(name).rglob("VERITE.csv"))
    header, rows = read_rows(path)
    labels = Counter(r["label"] for r in rows)
    root = path.parent
    modality = Modality()
    referenced = on_disk = 0
    distinct = set()
    for r in rows:
        img = (r.get("image_path") or "").strip()
        present = bool(img) and (root / img).is_file()
        if img:
            distinct.add(img)
        referenced += bool(img)
        on_disk += present
        modality.add(has_text=not _blank(r.get("caption")), has_image=present)
    return {"rows": len(rows), "by_split": {"all": len(rows)},
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(), "columns": header,
            "images": {"referenced": referenced, "on_disk": on_disk,
                       "coverage": round(on_disk / referenced, 4) if referenced else 0.0,
                       "distinct_images": len(distinct),
                       "note": "one image serves two classes: `true` and "
                               "`miscaptioned` share the same true_N.jpg, so "
                               "their coverage is identical by construction"},
            "notes": [
                "single undivided CSV; image_path resolved relative to it",
                "images are NOT shipped with the corpus. The register points at "
                "github.com/stevejpapad/relevant-evidence-detection (RED-DOT), "
                "which carries VERITE's CSVs and precomputed evidence only. The "
                "images were hydrated from true_url/false_url in "
                "VERITE_articles.csv -- see data/reports/hydration_verite.json "
                "for per-class recovery.",
            ]}


def profile_fakenewsnet(name: str, _sample: int) -> dict[str, Any]:
    """Four CSVs; the label is carried by the filename, not a column."""
    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    columns: list[str] = []
    for path in sorted(raw_dir(name).rglob("dataset/*.csv")):
        header, rows = read_rows(path)
        columns = header
        label = "fake" if "fake" in path.stem else "real"
        source = path.stem.split("_")[0]
        by_split[path.stem] = len(rows)
        labels[label] += len(rows)
        for r in rows:
            modality.add(has_text=not _blank(r.get("title")), has_image=False)
        del source
    return {"rows": sum(by_split.values()), "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(), "columns": columns,
            "notes": ["label comes from the filename (politifact/gossipcop x fake/real)",
                      "rows are IDs plus a news_url; article text and images are "
                      "NOT included and require the upstream crawler"]}


def _fakeddit_images_on_disk(name: str) -> set[str]:
    """Post ids whose sampled image actually landed.

    Only a seeded 150,000-image sample was fetched, so image presence cannot be
    read off the TSV's hasImage column -- that describes what Reddit held, not
    what we have.
    """
    folder = raw_dir(name) / "media" / "images"
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.is_file() and p.stat().st_size > 0}


def profile_fakeddit(name: str, _sample: int) -> dict[str, Any]:
    with_image = _fakeddit_images_on_disk(name)
    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    dates = Dates("created_utc")
    columns: list[str] = []
    declared_image = 0
    for split, pattern in (("train", "multimodal_train.tsv"),
                           ("validate", "multimodal_validate.tsv")):
        path = next(raw_dir(name).rglob(pattern), None)
        if path is None:
            continue
        header, rows = read_rows(path, delimiter="\t")
        columns = header
        by_split[split] = len(rows)
        for r in rows:
            labels[r.get("6_way_label", "?")] += 1
            dates.add_explicit(r.get("created_utc"))
            if str(r.get("hasImage", "")).strip().lower() in {"true", "1"}:
                declared_image += 1
            modality.add(has_text=not _blank(r.get("clean_title") or r.get("title")),
                         has_image=r.get("id", "") in with_image)
    fetched = len(with_image)
    return {"rows": sum(by_split.values()), "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": dates.to_dict(), "columns": columns,
            "images": {"declared_hasImage": declared_image,
                       "fetched_on_disk": fetched,
                       "note": "images are a seeded 150,000-row sample, not the "
                               "full set: 'text_only' means the row was not "
                               "sampled, NOT that Reddit had no image"},
            "notes": ["created_utc is a unix timestamp, converted to UTC dates",
                      "modality reflects images ON DISK; declared_hasImage is "
                      "what the TSV claims Reddit held, a different quantity"]}


def profile_factify2(name: str, _sample: int) -> dict[str, Any]:
    """Two images per row; evidence is the `document` text, and there are no dates."""
    from scripts.hydrate import resolve_factify2

    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    with_evidence = 0
    columns: list[str] = []
    rows_total = 0
    for path in sorted(raw_dir(name).rglob("*.csv")):
        if "test" in path.name.lower():
            continue
        header, rows = read_rows(path)
        columns = header
        by_split[path.stem] = len(rows)
        rows_total += len(rows)
        for r in rows:
            labels[r.get("Category", "?")] += 1
            if not _blank(r.get("document")):
                with_evidence += 1

    items, _label_col, _extra = resolve_factify2()
    by_record: dict[str, list] = {}
    for it in items:
        by_record.setdefault(it.record_id, []).append(it)
    for parts in by_record.values():
        present = [existing_destination(name, p) is not None for p in parts]
        modality.add(has_text=True, has_image=all(present))

    return {"rows": rows_total, "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(), "columns": columns,
            "evidence": {
                "kind": "document text field (not discrete evidence items)",
                "records_with_evidence": with_evidence,
                "records_total": rows_total,
                "coverage": round(with_evidence / rows_total, 4) if rows_total else 0.0,
                "evidence_items": 0,
                "with_explicit_date": 0,
                "with_url_derived_date": 0,
                "date_coverage": 0.0,
                "note": "NO DATE FIELD EXISTS in factify2. Columns are claim, "
                        "claim_image, document, document_image, Category, and the "
                        "two OCR fields. Its evidence therefore cannot participate "
                        "in time-filtered retrieval at all.",
            },
            "notes": ["a record carries TWO images (claim + document); modality "
                      "'both' requires both to be on disk",
                      "test.csv is excluded: it is the unlabelled shared-task split"]}


def _averitec_like(name: str, files: dict[str, str], date_field: str,
                   text_field: str, image_field: str | None) -> dict[str, Any]:
    """AVeriTeC and AVerImaTeC share a claim/questions/answers shape."""
    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    claim_dates = Dates(date_field)
    ev_items = 0
    ev_dates = Dates(None)
    with_evidence = 0
    total = 0

    for split, rel in files.items():
        path = raw_dir(name) / rel
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        by_split[split] = len(records)
        total += len(records)
        for rec in records:
            labels[rec.get("label", "?")] += 1
            claim_dates.add_explicit(rec.get(date_field))
            imgs = rec.get(image_field) if image_field else None
            modality.add(has_text=not _blank(rec.get(text_field)),
                         has_image=bool(imgs))
            answers = [a for q in (rec.get("questions") or [])
                       for a in (q.get("answers") or [])]
            if answers:
                with_evidence += 1
            ev_items += len(answers)
            for a in answers:
                ev_dates.add_url(a.get("source_url"))

    return {"rows": total, "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": claim_dates.to_dict(),
            "evidence": {
                "kind": "questions -> answers, each with a source_url",
                "records_with_evidence": with_evidence,
                "records_total": total,
                "coverage": round(with_evidence / total, 4) if total else 0.0,
                "evidence_items": ev_items,
                **{k: v for k, v in ev_dates.to_dict().items()
                   if k in {"with_explicit_date", "with_url_derived_date",
                            "explicit_coverage", "url_derived_coverage",
                            "any_coverage", "range"}},
                "note": "answers carry NO date field; any date here is derived "
                        "from the source URL path and is the publisher's filing "
                        "convention, not an asserted publication date.",
            }}


def profile_averitec(name: str, sample: int) -> dict[str, Any]:
    out = _averitec_like(
        name, {"train": "data/train.json", "dev": "data/dev.json",
               "test": "data/test.json"},
        date_field="claim_date", text_field="claim", image_field=None)
    out["notes"] = [
        "test.json is the unlabelled shared-task split; its claims are counted "
        "in rows but carry no usable label",
        "claim_date is day-first and unpadded (25-8-2020), parsed as such "
        "because 1,712 train values have a first component above 12",
        "claim_date is explicit; evidence dates are not",
    ]
    out["knowledge_store"] = _averitec_knowledge_store(name, sample)
    return out


def _averitec_knowledge_store(name: str, sample: int) -> dict[str, Any]:
    """Scan the dev knowledge store inside its zip, without extracting."""
    path = raw_dir(name) / "data_store/knowledge_store/dev_knowledge_store.zip"
    if not path.is_file():
        return {"present": False}
    dates = Dates(None)
    scanned = 0
    unparsable = 0
    with zipfile.ZipFile(path) as zf:
        members = sorted(n for n in zf.namelist() if n.endswith(".json"))
        chosen = members if sample <= 0 else members[:sample]
        for member in chosen:
            scanned += 1
            with zf.open(member) as fh:
                text = fh.read().decode("utf-8", "replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    unparsable += 1
                    continue
                if not isinstance(rec, dict):
                    # the store is not strictly one object per line; some lines
                    # parse as bare scalars and are not evidence records
                    unparsable += 1
                    continue
                dates.add_url(rec.get("url"))
    return {
        "present": True,
        "claim_files_total": len(members),
        "claim_files_scanned": scanned,
        "is_sample": sample > 0 and scanned < len(members),
        "unparsable_lines": unparsable,
        **dates.to_dict(),
        "note": "knowledge-store records are claim_id/type/query/url/url2text — "
                "there is NO date field, so every date here is URL-derived.",
    }


def profile_averimatec(name: str, _sample: int) -> dict[str, Any]:
    out = _averitec_like(
        name, {"train": "train.json", "val": "val.json"},
        date_field="date", text_field="claim_text", image_field="claim_images")
    out["notes"] = [
        "the held-out test archive is forbidden by the register and is not on disk",
        "claim images ship inside images.zip; presence counted from the record's "
        "claim_images field rather than unpacked files",
    ]
    return out


def profile_mocheg(name: str, _sample: int) -> dict[str, Any]:
    labels: Counter = Counter()
    by_split: dict[str, int] = {}
    modality = Modality()
    ev_dates = Dates(None)
    claims: set[str] = set()
    columns: list[str] = []
    rows_total = 0
    for split in ("train", "val", "test"):
        path = raw_dir(name) / "mocheg_back_up_without_image" / split / "Corpus2.csv"
        if not path.is_file():
            continue
        header, rows = read_rows(path)
        columns = header
        by_split[split] = len(rows)
        rows_total += len(rows)
        for r in rows:
            labels[r.get("cleaned_truthfulness", "?")] += 1
            claims.add(r.get("claim_id", ""))
            modality.add(has_text=not _blank(r.get("Claim")), has_image=False)
            ev_dates.add_url(r.get("Commoncrawl URL") or r.get("Snopes URL"))
    return {"rows": rows_total, "by_split": by_split,
            "labels": dict(labels.most_common()), "modality": modality.to_dict(),
            "dates": Dates(None).to_dict(), "columns": columns,
            "unique_claims": len(claims - {""}),
            "evidence": {
                "kind": "one Corpus2 row per evidence item, keyed to a claim_id",
                "records_with_evidence": len(claims - {""}),
                "records_total": len(claims - {""}),
                "coverage": 1.0 if claims else 0.0,
                "evidence_items": rows_total,
                **{k: v for k, v in ev_dates.to_dict().items()
                   if k in {"with_explicit_date", "with_url_derived_date",
                            "explicit_coverage", "url_derived_coverage",
                            "any_coverage", "range"}},
                "note": "Corpus2 has NO date column; dates are derived from the "
                        "Commoncrawl/Snopes URL only.",
            },
            "notes": ["this release is mocheg_v1_without_image and carries NO images",
                      "rows are evidence items, not claims; unique_claims is the "
                      "claim count"]}


PROFILERS: dict[str, Callable[[str, int], dict[str, Any]]] = {
    "liar": profile_liar,
    "welfake": profile_welfake,
    "verite": profile_verite,
    "fakenewsnet": profile_fakenewsnet,
    "fakeddit": profile_fakeddit,
    "factify2": profile_factify2,
    "averitec": profile_averitec,
    "averimatec": profile_averimatec,
    "mocheg": profile_mocheg,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def verify(name: str, sample: int = DEFAULT_EVIDENCE_SAMPLE) -> dict[str, Any]:
    entry = dataset(name)
    count, size = _files(name)

    if count == 0:
        return {
            "dataset": name, "generated_at": utcnow(), "status": "NOT FETCHED",
            "files": {"count": 0, "bytes": 0},
            "licence": entry.get("licence"),
            "redistributable": entry.get("redistributable"),
            "access": {"method": entry.get("method"), "url": entry.get("url")},
            "expected": entry.get("expected"),
            "mismatches": [],
            "notes": [f"{name} has never been fetched; nothing to measure."],
        }

    profile = PROFILERS[name](name, sample)
    mismatches = compare_expected(name, profile.get("rows", 0),
                                  profile.get("labels", {}))
    return {
        "dataset": name,
        "generated_at": utcnow(),
        "status": "measured",
        "files": {"count": count, "bytes": size, "gb": round(size / GB, 4)},
        "licence": entry.get("licence"),
        "redistributable": entry.get("redistributable"),
        "access": {"method": entry.get("method"), "url": entry.get("url"),
                   "repo_type": entry.get("repo_type")},
        "expected": entry.get("expected"),
        "mismatches": mismatches,
        **profile,
    }


def render(report: dict[str, Any]) -> str:
    if report["status"] == "NOT FETCHED":
        return f"  NOT FETCHED — {report['notes'][0]}"
    lines = [
        f"  files      : {report['files']['count']:,}  ({report['files']['gb']:.3f} GB)",
        f"  rows       : {report['rows']:,}   splits: {report['by_split']}",
        f"  labels     : {report['labels']}",
    ]
    mod = report["modality"]["fractions"]
    lines.append(f"  modality   : text-only {mod['text_only']:.1%}  "
                 f"image-only {mod['image_only']:.1%}  both {mod['both']:.1%}  "
                 f"neither {mod['neither']:.1%}")
    d = report["dates"]
    if d["field"]:
        rng = d["range"] or {}
        lines.append(f"  dates      : field {d['field']!r}  coverage "
                     f"{d['explicit_coverage']:.1%}  range "
                     f"{rng.get('min','—')} .. {rng.get('max','—')}")
    else:
        lines.append("  dates      : NO DATE FIELD")
    ev = report.get("evidence")
    if ev:
        lines.append(
            f"  evidence   : {ev['records_with_evidence']:,} of "
            f"{ev['records_total']:,} records ({ev['coverage']:.1%}) carry "
            f"evidence  [{ev['kind']}]")
        if ev["evidence_items"]:
            lines.append(
                f"               {ev['evidence_items']:,} items; explicit dates "
                f"{ev.get('with_explicit_date', 0):,}  URL-derived "
                f"{ev.get('with_url_derived_date', 0):,} "
                f"({ev.get('url_derived_coverage', 0):.1%})")
        else:
            lines.append("               NO DISCRETE EVIDENCE ITEMS, so there "
                         "are no evidence dates to recover")
    ks = report.get("knowledge_store")
    if ks and ks.get("present"):
        tag = " (SAMPLE)" if ks["is_sample"] else " (all)"
        lines.append(f"  kn. store  : {ks['claim_files_scanned']:,} of "
                     f"{ks['claim_files_total']:,} claim files{tag}, "
                     f"{ks['items']:,} items, URL dates "
                     f"{ks['url_derived_coverage']:.1%}")
    if report["mismatches"]:
        lines.append(f"  MISMATCHES : {len(report['mismatches'])}")
        for m in report["mismatches"]:
            lines.append(f"    {m['field']}: expected {m['expected']!r}, "
                         f"measured {m['measured']!r} — {m['detail']}")
    elif report.get("expected"):
        lines.append("  expected   : all fields match")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", action="append", dest="datasets", metavar="NAME")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--evidence-sample", type=int, default=DEFAULT_EVIDENCE_SAMPLE,
                        help="AVeriTeC knowledge-store claim files to scan "
                             f"(default {DEFAULT_EVIDENCE_SAMPLE}; 0 = all)")
    args = parser.parse_args(argv)

    names = enabled_names() if args.all else args.datasets
    REPORTS.mkdir(parents=True, exist_ok=True)

    total_mismatches = 0
    for name in names:
        try:
            report = verify(name, sample=args.evidence_sample)
        except Exception as exc:
            print(f"\n[{name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            total_mismatches += 1
            continue
        path = REPORTS / f"{name}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                        encoding="utf-8")
        print(f"\n[{name}]")
        print(render(report))
        print(f"  report -> {path.relative_to(PROJECT_ROOT)}")
        total_mismatches += len(report.get("mismatches", []))

    print(f"\n{total_mismatches} mismatch(es) across {len(names)} dataset(s)")
    return 1 if total_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
