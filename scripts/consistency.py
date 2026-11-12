#!/usr/bin/env python
"""The consistency block, v0 zero-shot: does a claim agree with what is known about its image?

    python scripts/consistency.py snapshot-processed            # before the M4FC rebuild
    python scripts/consistency.py snapshot-processed --compare  # after it: what moved
    python scripts/consistency.py m4fc-files                    # what image_text reads on the server
    python scripts/consistency.py gold                          # stage A: the sheet and the true rows
    python scripts/consistency.py prompts --task judge --stage a    # id-checked input for the server
    python scripts/consistency.py llm --task judge --input F    # server: text-only Qwen; resumable
    python scripts/consistency.py llm --task judge --input F --recheck 3   # fresh process, identical?
    python scripts/consistency.py score --stage a               # data/reports/consistency_a.json
    python scripts/consistency.py faces --sample 100            # stage B: time a spread sample
    python scripts/consistency.py faces                         # query faces, M4FC + VERITE; resumable
    python scripts/consistency.py verify-people                 # every name, at the certified threshold
    python scripts/consistency.py retrieve --dataset m4fc --sample 30   # time a spread sample
    python scripts/consistency.py retrieve --dataset m4fc       # evidence per claim; resumable
    python scripts/consistency.py retrieve --dataset m4fc --check   # the ledger against both rules
    python scripts/consistency.py prompts --task predict --stage b --part pilot  # 30 M4FC train images
    python scripts/consistency.py prompts --task predict --stage b --part full   # every prediction
    python scripts/consistency.py llm --task predict --input F  # server: the context predictor
    python scripts/consistency.py prompts --task judge --stage b    # B.4: the scoring judge's pairs
    python scripts/consistency.py score --stage b               # data/reports/consistency_b.json
    python scripts/consistency.py prompts --task extract --stage c --part pilot  # C.1: 30 M4FC train claims
    python scripts/consistency.py prompts --task extract --stage c --part full   # C.1: every claim C reads
    python scripts/consistency.py clip --sample 100             # C.5: time the CLIP baseline (CPU)
    python scripts/consistency.py clip                          # C.5: image-claim cosine; resumable
    python scripts/consistency.py prompts --task judge --stage c    # C.2: claim values against predictions
    python scripts/consistency.py verdicts                      # C.2: every record's verdicts
    python scripts/consistency.py score --stage c               # data/reports/consistency_c.json
    python scripts/consistency.py demo --rehearse               # D: pages from the outputs there are now
    python scripts/consistency.py demo --from-run KEY           # D: one computed record as a local page

For each claim and its image, and each item -- date, location, people, event --
a verdict: consistent, contradicted or unsupported, with a score, the evidence
it used and a reason code. The verdicts aggregate into one media_mismatch
score. v0 trains nothing.

Three rules hold in every step, and tests/test_consistency.py holds them:

* A model sees a ``ModelView`` and nothing else. A view is built from
  whitelisted content, so no record id, path, URL or meta value can reach a
  prompt, a retrieval query or a feature. M4FC's record ids are slugs of its
  fact-checks' URLs, and many state the finding. M4FC publishes them itself
  (CC-BY-SA), so they may stand in published files, but never in a model input.
* M4FC's test split is never read: ``rows`` refuses a processed test row and a
  row of the authors' own test split alike.
* VERITE is read only through ``evaluation_rows``, which drops the 37 records
  byte-identical to Factify2, M4FC or MOCHEG and then asserts that none is left.

It writes to data/interim/consistency/ (ignored) and opens no socket: retrieval
reads the local index.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import (CONFIGS, CROSS_CORPUS_YAML, FINGERPRINT_REJECTIONS_YAML,  # noqa: E402
                           INTERIM, PLACEHOLDERS_YAML, PROCESSED, PROJECT_ROOT, REPORTS,
                           SOURCES_YAML, SPLITS, raw_dir)
from scripts.fetchlib import utcnow  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SEED = 20260901
#: The items a verdict is given for. M4FC also annotates "things", which no
#: verdict uses.
ITEMS = ("date", "location", "people", "event")
#: The processed splits this block reads of M4FC. Its test is never read.
M4FC_SPLITS = ("train", "val")
#: Corpora VERITE shares images with: their records leave VERITE's evaluation.
VERITE_AGAINST = ("factify2", "m4fc", "mocheg")
#: The image_text run every view reads: the configuration VERITE, AVerImaTeC
#: and Factify2 were processed under, and M4FC is run under (--expect-config).
IMAGE_TEXT_CONFIG = "93cd9d679720"

OUT = INTERIM / "consistency"
SHIP = OUT / "ship"
M4FC_FILES = SHIP / "m4fc_files.txt"
SNAPSHOT = OUT / "processed_snapshot"
SNAPSHOT_JSON = SNAPSHOT / "snapshot.json"
COMPARE_JSON = SNAPSHOT / "compare.json"
PROCESSED_RECORDS = PROCESSED / "records"
INTERIM_M4FC = INTERIM / "records" / "m4fc.jsonl.gz"
PROCESSED_REPORT = REPORTS / "processed.json"
#: processed.json keys that say when the build ran, not what it built.
CLOCK_KEYS = ("built_at", "started_at", "images_verified_at")
#: Tracked files the image gates read: shipped with the code, copied into the
#: empty root the server's work list is rehearsed in.
REGISTRY_FILES = (SOURCES_YAML, PLACEHOLDERS_YAML, FINGERPRINT_REJECTIONS_YAML, CROSS_CORPUS_YAML)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gz_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def gz_digest(path: Path) -> dict[str, Any]:
    """sha256 of the DECOMPRESSED bytes, and the row count: a gzip header
    carries its write time, so the compressed file changes on every build."""
    h, n = hashlib.sha256(), 0
    with gzip.open(path, "rb") as handle:
        for line in handle:
            h.update(line)
            n += 1
    return {"sha256": h.hexdigest(), "rows": n}


def ordered(counts: Counter | dict) -> dict[str, int]:
    """Largest first, ties by key: the same counts always serialise the same way."""
    return {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def rel(path: Path) -> str:
    """A path as the project names it: relative to the root where it lies under it."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------
# guards: which rows may be read, and what may reach a model
# --------------------------------------------------------------------------


class GuardError(RuntimeError):
    """A step asked for rows this block may not read, or a model input carries an id."""


def rows(dataset: str, splits: Sequence[str], *, usable: bool = True) -> Iterator[dict[str, Any]]:
    """Processed rows of ``splits``: how this block reads every corpus but VERITE.

    M4FC's test is refused whichever way it comes: asked for by name, or as a
    row of the authors' own test split. Processed holds no such row, so one
    that appears means the layer changed under this block, and the step stops.
    """
    from scripts.processed_loader import iter_rows

    if dataset == "verite":
        raise GuardError("VERITE is read through evaluation_rows() only")
    wanted = set(splits)
    if dataset == "m4fc" and wanted - set(M4FC_SPLITS):
        raise GuardError(f"M4FC: this block reads {M4FC_SPLITS} only, never "
                         f"{sorted(wanted - set(M4FC_SPLITS))}")
    for row in iter_rows(dataset):
        if dataset == "m4fc" and (row.get("meta") or {}).get("official_split") == "test":
            raise GuardError("M4FC: processed holds a row of the authors' test split; stopping")
        if row["split"] in wanted and (row["usable"] or not usable):
            yield row


def verite_dropped() -> frozenset[str]:
    """VERITE record_ids byte-identical to a record of Factify2, M4FC or MOCHEG."""
    from scripts import placeholders

    return frozenset().union(*(placeholders.excluded_keys("verite", against=a)
                               for a in VERITE_AGAINST))


def evaluation_rows() -> list[dict[str, Any]]:
    """VERITE's evaluation rows: usable, minus every record byte-identical to a
    corpus on the other side -- then asserted, so that a change to the list
    cannot let one through silently."""
    from scripts import placeholders
    from scripts.processed_loader import iter_rows

    drop = verite_dropped()
    out = [r for r in iter_rows("verite") if r["usable"] and r["record_id"] not in drop]
    for against in VERITE_AGAINST:
        placeholders.assert_no_cross_corpus_leak("verite", [r["record_id"] for r in out],
                                                 against=against)
    return out


#: Shapes an identifier has, whatever row it comes from: VERITE's label-bearing
#: file names (true_N, false_N), a path into data/raw, and M4FC's record-id shape.
ID_SHAPES = (re.compile(r"(?<![a-z0-9])(?:true|false)_\d+(?![0-9])"),
             re.compile(r"data/raw/"),
             re.compile(r"(?<![a-z0-9])images/[^\s\"']+?\.(?:png|jpe?g|webp|gif|bmp)(?![a-z0-9])"))
#: Shorter forms are too common to mean anything (VERITE's record ids are row numbers).
MIN_FORM = 8


def id_forms(row: dict[str, Any]) -> frozenset[str]:
    """Every form in which a row's identity could leak, case-folded: its record
    id, its image paths and file names, the slug inside each, and that slug
    with its separators read as spaces."""
    out: set[str] = set()
    for value in (row.get("record_id"), *(row.get("image_paths") or ())):
        if not value:
            continue
        path = Path(str(value))
        out.update({str(value), path.name, path.stem,
                    re.sub(r"[\s_.\-]+", " ", path.stem).strip()})
    return frozenset(f.casefold() for f in out if len(f) >= MIN_FORM)


def ids_in(text: str, forms: Iterable[str] = (), *, explained: str = "") -> list[str]:
    """The identifier forms and shapes ``text`` holds that ``explained`` does not.

    ``explained`` is the whitelisted content the text was built from: a claim,
    OCR, a retrieved excerpt. A form that occurs there as often as in the text
    came in with that content -- a claim may say what its own slug says -- and
    not from an id.
    """
    low, src = text.casefold(), explained.casefold()
    found = [f for f in forms if low.count(f) > src.count(f)]
    found += [s.pattern for s in ID_SHAPES if len(s.findall(low)) > len(s.findall(src))]
    return found


def refuse_ids(text: str, forms: Iterable[str] = (), *, explained: str = "",
               what: str = "a model input") -> str:
    """``text``, or GuardError if it carries an identifier its content does not explain."""
    found = ids_in(text, forms, explained=explained)
    if found:
        raise GuardError(f"{what} carries {len(found)} identifier form(s) its content does not "
                         "explain; refusing")
    return text


# --------------------------------------------------------------------------
# the model view: the only thing a model sees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """One of the VLM's entity items, as image_text wrote it."""

    category: str
    name: str
    basis: str
    confidence: float | None
    forced: bool


@dataclass(frozen=True)
class Person:
    """A name the gallery accepted at its certified threshold."""

    name: str
    qid: str
    score: float


@dataclass(frozen=True)
class Document:
    """A retrieved document, never its URL."""

    doc_id: str
    title: str
    published_at: str
    host: str
    excerpt: str


@dataclass(frozen=True)
class ModelView:
    """Everything a model, a query or a feature may see of one claim and its image.

    Every field is whitelisted content. There is no field for a record id, a
    path, a URL or meta, so none can reach a prompt. ``image_sha256`` is the
    join key: it is carried beside a prompt, never written into one.
    """

    image_sha256: str | None
    claim: str | None = None
    ocr: str | None = None
    description: str | None = None
    entities: tuple[Entity, ...] = ()
    people: tuple[Person, ...] = ()
    documents: tuple[Document, ...] = ()

    def content(self) -> str:
        """The view's text: what every input built from it is checked against."""
        parts = [self.claim or "", self.ocr or "", self.description or ""]
        parts += [e.name for e in self.entities] + [p.name for p in self.people]
        parts += [f"{d.doc_id}\n{d.title}\n{d.published_at}\n{d.host}\n{d.excerpt}"
                  for d in self.documents]
        return "\n".join(parts)


def image_part(image_row: dict[str, Any] | None, sha256: str | None
               ) -> tuple[str | None, str | None, tuple[Entity, ...]]:
    """OCR text, description and entity items of an image_text row, and nothing
    else of it: never its record_ids, paths or provenance. A component that
    failed is missing (None), never empty."""
    if image_row is None:
        return None, None, ()
    if image_row.get("sha256") != sha256:
        raise GuardError("an image_text row was joined to another image")
    from scripts.image_text import entities_of

    ocr, desc = image_row.get("ocr") or {}, image_row.get("description") or {}
    return ((None if ocr.get("error") else ocr.get("text")),
            (None if desc.get("error") else desc.get("text")),
            tuple(Entity(e["category"], e["name"], e["basis"], e["confidence"], e["forced"])
                  for e in entities_of(image_row, forced="mark")))


def view_of(row: dict[str, Any], *, sha_of: Callable[[str], str | None],
            image_rows: dict[str, dict[str, Any]], people: Sequence[Person] = (),
            documents: Sequence[Document] = ()) -> ModelView:
    """The view of one processed row. Of the row itself it reads the claim text
    and, through ``sha_of``, its image's sha256 -- nothing else."""
    paths = row.get("image_paths") or []
    sha = sha_of(paths[0]) if paths else None
    ocr, desc, ents = image_part(image_rows.get(sha) if sha else None, sha)
    return ModelView(sha, row.get("text"), ocr, desc, ents, tuple(people), tuple(documents))


def view_json(view: ModelView) -> str:
    return json.dumps(asdict(view), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------
# stage 0: the processed layer before and after the M4FC rebuild
# --------------------------------------------------------------------------


def report_without_clock(path: Path = PROCESSED_REPORT) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in report.items() if k not in CLOCK_KEYS}


def digests() -> dict[str, Any]:
    return {"records": {p.name.split(".")[0]: gz_digest(p)
                        for p in sorted(PROCESSED_RECORDS.glob("*.jsonl.gz"))},
            "splits": {p.name: sha256_file(p) for p in sorted(SPLITS.glob("*.csv"))},
            "report": hashlib.sha256(canonical(report_without_clock())).hexdigest(),
            "interim_m4fc": gz_digest(INTERIM_M4FC)}


def snapshot_processed() -> dict[str, Any]:
    """Digests of the processed layer, and copies of what they digest, taken
    once before the rebuild: a moved row can then be counted, not just seen."""
    if SNAPSHOT_JSON.is_file():
        raise SystemExit(f"{rel(SNAPSHOT_JSON)} exists: a snapshot is taken once, before the "
                         "rebuild. Compare against it (--compare), or delete "
                         f"{rel(SNAPSHOT)} to take a new one.")
    snap = {"taken_at": utcnow(), **digests()}
    shutil.copytree(PROCESSED_RECORDS, SNAPSHOT / "records", dirs_exist_ok=True)
    shutil.copytree(SPLITS, SNAPSHOT / "splits", dirs_exist_ok=True)
    shutil.copy2(PROCESSED_REPORT, SNAPSHOT / "processed.json")
    shutil.copy2(INTERIM_M4FC, SNAPSHOT / "interim_m4fc.jsonl.gz")
    write_json(SNAPSHOT_JSON, snap)  # last: it means the copies are complete
    n = sum(d["rows"] for d in snap["records"].values())
    print(f"snapshot: {len(snap['records'])} record files ({n:,} rows), "
          f"{len(snap['splits'])} split files -> {rel(SNAPSHOT)}")
    return snap


def row_diff(before: Path, after: Path) -> dict[str, Any]:
    """Counts of what differs between two record files, by record_id."""
    a = {r["record_id"]: r for r in gz_rows(before)}
    b = {r["record_id"]: r for r in gz_rows(after)}
    changed, by_field = 0, Counter()
    for rid in a.keys() & b.keys():
        if a[rid] != b[rid]:
            changed += 1
            by_field.update(k for k in set(a[rid]) | set(b[rid]) if a[rid].get(k) != b[rid].get(k))
    return {"only_before": len(a.keys() - b.keys()), "only_after": len(b.keys() - a.keys()),
            "changed": changed, "changed_fields": ordered(by_field)}


def meta_only(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]],
              raw: dict[str, dict[str, Any]], added: Sequence[str]) -> dict[str, Any]:
    """M4FC rows after the rebuild against before: the same rows in the same
    order, every field as it was except ``meta``, whose old keys keep their
    values and whose only new keys are ``added``, each verbatim from raw."""
    problems: Counter = Counter()
    if [r["record_id"] for r in before] != [r["record_id"] for r in after]:
        problems["the record ids, or their order, differ"] += 1
    for old, new in zip(before, after):
        for k in set(old) | set(new):
            if k != "meta" and old.get(k) != new.get(k):
                problems[f"field {k} changed"] += 1
        m_old, m_new = old.get("meta") or {}, new.get("meta") or {}
        if {k: v for k, v in m_new.items() if k not in added} != m_old:
            problems["an earlier meta value changed or went"] += 1
        if set(m_new) - set(m_old) != set(added):
            problems["the new meta keys are not exactly the ones added"] += 1
        src = raw.get(new["record_id"])
        if src is None or any(m_new.get(k) != src.get(k) for k in added):
            problems["a new meta value is not M4FC.json's"] += 1
    return {"rows_before": len(before), "rows_after": len(after), "problems": ordered(problems)}


def compare_snapshot() -> int:
    """The layer now against the snapshot. Exit 0 only if nothing but M4FC's
    new meta keys moved; anything else is counted and the exit is 2."""
    from scripts.records import M4FC_META

    if not SNAPSHOT_JSON.is_file():
        raise SystemExit("no snapshot: run snapshot-processed before the rebuild")
    snap = json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    now = digests()
    raw = {r["image_path"]: r
           for r in json.loads((raw_dir("m4fc") / "data" / "M4FC.json").read_text(encoding="utf-8"))}
    out: dict[str, Any] = {"compared_at": utcnow(), "snapshot_taken_at": snap["taken_at"],
                           "datasets": {}, "splits": {}}
    for ds in sorted(set(snap["records"]) | set(now["records"])):
        if ds == "m4fc":
            continue
        if snap["records"].get(ds) == now["records"].get(ds):
            out["datasets"][ds] = "identical"
        else:
            out["datasets"][ds] = row_diff(SNAPSHOT / "records" / f"{ds}.jsonl.gz",
                                           PROCESSED_RECORDS / f"{ds}.jsonl.gz")
    for name in sorted(set(snap["splits"]) | set(now["splits"])):
        out["splits"][name] = ("identical" if snap["splits"].get(name) == now["splits"].get(name)
                               else "DIFFERS")
    before = json.loads((SNAPSHOT / "processed.json").read_text(encoding="utf-8"))
    after = report_without_clock()
    out["report"] = ("identical" if snap["report"] == now["report"] else
                     {"differing_keys": sorted(k for k in set(before) | set(after)
                                               if k not in CLOCK_KEYS and before.get(k) != after.get(k))})
    new_rows = list(gz_rows(PROCESSED_RECORDS / "m4fc.jsonl.gz"))
    out["m4fc_processed"] = meta_only(list(gz_rows(SNAPSHOT / "records" / "m4fc.jsonl.gz")),
                                      new_rows, raw, M4FC_META)
    out["m4fc_processed"]["splits"] = ordered(Counter(r["split"] for r in new_rows))
    out["m4fc_interim"] = meta_only(list(gz_rows(SNAPSHOT / "interim_m4fc.jsonl.gz")),
                                    list(gz_rows(INTERIM_M4FC)), raw, M4FC_META)
    passed = (all(v == "identical" for v in out["datasets"].values())
              and all(v == "identical" for v in out["splits"].values())
              and out["report"] == "identical"
              and not out["m4fc_processed"]["problems"] and not out["m4fc_interim"]["problems"])
    out["passed"] = passed
    write_json(COMPARE_JSON, out)
    moved = {ds: v for ds, v in out["datasets"].items() if v != "identical"}
    print(f"other datasets: {len(out['datasets']) - len(moved)} identical"
          + (f"; MOVED: {json.dumps(moved)}" if moved else ""))
    split_moved = [n for n, v in out["splits"].items() if v != "identical"]
    print(f"split files: {len(out['splits']) - len(split_moved)} identical"
          + (f"; DIFFER: {split_moved}" if split_moved else ""))
    print(f"processed.json without its clock: {json.dumps(out['report'])}")
    for part in ("m4fc_processed", "m4fc_interim"):
        p = out[part]
        print(f"{part}: {p['rows_after']:,} rows (before {p['rows_before']:,}); problems "
              f"{json.dumps(p['problems']) if p['problems'] else 'none'}"
              + (f"; splits {json.dumps(p['splits'])}" if "splits" in p else ""))
    print(("PASSED" if passed else "FAILED: bring the counts above back before going on")
          + f" -> {rel(COMPARE_JSON)}")
    return 0 if passed else 2


# --------------------------------------------------------------------------
# stage 0: the files image_text reads on the server
# --------------------------------------------------------------------------


def rehearse_worklist(listed: Sequence[str]) -> list[str]:
    """The M4FC work list built in an EMPTY root holding only ``listed`` and the
    registry files: what the server will build from what is shipped to it."""
    with tempfile.TemporaryDirectory(prefix="m4fc_ship_") as tmp:
        root = Path(tmp)
        for path in [*listed, *(p.relative_to(PROJECT_ROOT).as_posix() for p in REGISTRY_FILES)]:
            dst = root / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / path, dst)
        code = (f"import json, sys; sys.path.insert(0, {str(REPO)!r}); "
                "from scripts import image_text; "
                "print(json.dumps(sorted(it.path for it in image_text.build_worklist('m4fc')[0])))")
        env = {**os.environ, "PE_FAKE_NEWS_ROOT": str(root), "PYTHONIOENCODING": "utf-8"}
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                              encoding="utf-8", cwd=root)
    if done.returncode:
        raise SystemExit(f"the work list fails in a root holding only the listed files:\n"
                         f"{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def m4fc_files() -> dict[str, Any]:
    """The files image_text reads on the server for M4FC train+val, checked
    before they are listed: every image is an M4FC file on disk with the bytes
    its index row describes, and the work list built from the listed files
    alone, in an empty root, is the one built here."""
    from scripts import image_text, images
    from scripts.processed_loader import records_path

    index = images.load_index("m4fc")
    paths = sorted({p for r in rows("m4fc", M4FC_SPLITS) for p in r["image_paths"]})
    bad, total = Counter(), 0
    for p in paths:
        entry, f = index.get(p), PROJECT_ROOT / p
        if images.dataset_of(p) != "m4fc":
            bad["not an M4FC file"] += 1
        elif not entry or not f.is_file():
            bad["not on disk or not indexed"] += 1
        elif sha256_file(f) != entry["sha256"]:
            bad["bytes differ from the index"] += 1
        else:
            total += f.stat().st_size
    if bad:
        raise SystemExit(f"refusing to list: {ordered(bad)}")
    # strictly relative: tar runs from the root and reads these names as they are
    listed = [p.relative_to(PROJECT_ROOT).as_posix()
              for p in (records_path("m4fc"), images.index_path("m4fc"))]
    for p in listed:
        total += (PROJECT_ROOT / p).stat().st_size
    expected = sorted(it.path for it in image_text.build_worklist("m4fc")[0])
    if expected != paths:
        raise SystemExit(f"the work list holds {len(expected):,} images, the train+val rows "
                         f"{len(paths):,}: they must be the same set")
    if rehearse_worklist(listed) != expected:
        raise SystemExit("the work list built from the listed files alone is not the one built here")
    listed += paths
    SHIP.mkdir(parents=True, exist_ok=True)
    M4FC_FILES.write_bytes(("\n".join(listed) + "\n").encode("utf-8"))  # LF: tar -T reads it
    print(f"{len(listed):,} files ({len(paths):,} images), {total / 2**30:.2f} GiB -> "
          f"{rel(M4FC_FILES)}")
    print(f"  checked: every image's bytes match its index row; the work list rebuilt from "
          f"these files alone, in an empty root, is the same {len(expected):,} images")
    print(f"  tar it from {PROJECT_ROOT} (in WSL: /mnt/{PROJECT_ROOT.drive[0].lower()}"
          f"{PROJECT_ROOT.as_posix()[2:]})" if PROJECT_ROOT.drive else f"  tar it from {PROJECT_ROOT}")
    return {"files": len(listed), "images": len(paths), "bytes": total}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """The Wilson score interval for k of n (95% by default)."""
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict[str, Any]:
    """k of n, its rate and Wilson interval: the form every proportion is reported in."""
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": None if n == 0 else round(k / n, 4),
            "wilson95": None if n == 0 else [round(lo, 4), round(hi, 4)]}


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

CONTRADICTS, CONSISTENT, UNCLEAR = "contradicts", "consistent", "unclear"
COMPARATOR_LABELS = (CONTRADICTS, CONSISTENT, UNCLEAR)


def outcome(label: str, why: str, score: float | None, comparator: str, **extra) -> dict[str, Any]:
    """A comparator's answer for one claimed value against one actual value.
    ``score`` is P(contradicts): 1 or 0 from a rule, a probability from the
    judge, None where nothing was decided."""
    if label not in COMPARATOR_LABELS:
        raise ValueError(label)
    return {"label": label, "why": why, "score": score, "comparator": comparator, **extra}


def stated_parts(value: Any) -> list[str]:
    """A value's parts that state something: no empty part, no absent marker."""
    from scripts.m4fc_audit import ABSENT, parts

    return [p for p in parts(value) if p and p.casefold() not in ABSENT]


# --------------------------------------------------------------------------
# the date rule
# --------------------------------------------------------------------------

_YMD = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")
_SPAN = re.compile(r"(\d{4})\s*[-–]\s*(\d{4})")
_DECADE = re.compile(r"(\d{3})0s")
_AD = re.compile(r"(\d{1,4})\s*(?:AD|CE)", re.I)


def date_interval(part: str) -> tuple[date, date] | None:
    """A date part as the days it covers, at its own granularity: YYYY,
    YYYY-MM, YYYY-MM-DD, a YYYY-YYYY span, a decade (1990s) or a year AD.
    Nothing else is read as a date, and an impossible one is no date."""
    p = part.strip()
    try:
        if m := _YMD.fullmatch(p):
            y = int(m[1])
            if m[3]:
                d = date(y, int(m[2]), int(m[3]))
                return d, d
            if m[2]:
                month = int(m[2])
                return date(y, month, 1), date(y, month, monthrange(y, month)[1])
            return date(y, 1, 1), date(y, 12, 31)
        if m := _SPAN.fullmatch(p):
            a, b = int(m[1]), int(m[2])
            return (date(a, 1, 1), date(b, 12, 31)) if a <= b else None
        if m := _DECADE.fullmatch(p):
            y = int(m[1]) * 10
            return date(y, 1, 1), date(y + 9, 12, 31)
        if m := _AD.fullmatch(p):
            y = int(m[1])
            return (date(y, 1, 1), date(y, 12, 31)) if y >= 1 else None
    except ValueError:  # month 13, 30 February
        return None
    return None


def date_rule(claimed: Any, actual: Any) -> dict[str, Any]:
    """Contradicts if no claimed date overlaps any actual one; consistent if one
    does; unclear if a side states nothing, or holds a part that is no date
    (that part might have overlapped). Overlap is at each part's own
    granularity: "2019" and "2019-03-15" can both be true."""
    c, a = stated_parts(claimed), stated_parts(actual)
    if not c or not a:
        return outcome(UNCLEAR, "absent", None, "rule")
    ci, ai = [date_interval(p) for p in c], [date_interval(p) for p in a]
    if any(x[0] <= y[1] and y[0] <= x[1] for x in ci if x for y in ai if y):
        return outcome(CONSISTENT, "date_overlap", 0.0, "rule")
    if None in ci or None in ai:
        return outcome(UNCLEAR, "date_unparsed", None, "rule")
    return outcome(CONTRADICTS, "date_disjoint", 1.0, "rule")


# --------------------------------------------------------------------------
# the people QID rule (scored in A only; C decides people by the gallery)
# --------------------------------------------------------------------------


def people_qid_rule(claimed: Any, actual: Any, resolve: Callable[[str], Sequence[str]]
                    ) -> dict[str, Any]:
    """Both sides through the gallery's name resolution. A shared QID ->
    consistent. Every name on both sides one person, none shared ->
    contradicts: M4FC's ``people`` are the people in the image. Any name that
    resolves to no one or to several -> unclear, unless a QID is shared."""
    c, a = stated_parts(claimed), stated_parts(actual)
    if not c or not a:
        return outcome(UNCLEAR, "absent", None, "qid_rule")
    cr, ar = [list(resolve(n)) for n in c], [list(resolve(n)) for n in a]
    cq = {q[0] for q in cr if len(q) == 1}
    aq = {q[0] for q in ar if len(q) == 1}
    if cq & aq:
        return outcome(CONSISTENT, "same_qid", 0.0, "qid_rule")
    if all(len(q) == 1 for q in cr + ar):
        return outcome(CONTRADICTS, "different_qids", 1.0, "qid_rule")
    why = "ambiguous_name" if any(len(q) > 1 for q in cr + ar) else "name_not_in_gallery"
    return outcome(UNCLEAR, why, None, "qid_rule")


# --------------------------------------------------------------------------
# stage A: the gold sets
# --------------------------------------------------------------------------

A_DIR = OUT / "a"
GOLD = A_DIR / "gold.jsonl"
GOLD_PRIVATE = A_DIR / "gold_private.json"
SHEET_DIR = INTERIM / "m4fc_audit" / "review"
SHEET_REVIEWED = SHEET_DIR / "m4fc_ooc_items_reviewed.csv"
ASSISTANT_ROWS = SHEET_DIR / "assistant_rows.json"
SHEET_PRIVATE = SHEET_DIR / "private_map.json"
#: The sheet's answers: "yes" said the claim's value contradicts the actual one.
SHEET_LABELS = {"yes": CONTRADICTS, "no": CONSISTENT, "unclear": UNCLEAR}
SOURCES = ("sheet", "true_row")


def sheet_gold() -> list[dict[str, Any]]:
    """The person's answers on the four items, without the rows the assistant
    answered. Refused if the assistant-rows file was written for another sheet."""
    body = SHEET_REVIEWED.read_bytes()
    marks = json.loads(ASSISTANT_ROWS.read_text(encoding="utf-8-sig"))
    if marks.get("sheet_sha256") != hashlib.sha256(body).hexdigest():
        raise SystemExit(f"{rel(ASSISTANT_ROWS)} was written for another sheet "
                         f"(its sheet_sha256 is not {rel(SHEET_REVIEWED)}'s): refusing")
    drop = set(marks["sheet_ids"])
    out = []
    for r in csv.DictReader(io.StringIO(body.decode("utf-8-sig"))):
        if r["sheet_id"] in drop or r["item"] not in ITEMS:
            continue
        label = SHEET_LABELS.get(r["mismatch"].strip().casefold())
        if label is None:
            raise SystemExit(f"{r['sheet_id']}: the sheet's answer {r['mismatch']!r} is not "
                             f"one of {sorted(SHEET_LABELS)}")
        out.append({"gold_id": r["sheet_id"], "source": "sheet", "item": r["item"],
                    "claimed": r["claimed"], "actual": r["actual"], "label": label})
    return out


def true_row_gold() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """M4FC train ``true`` rows with both values of an item: consistent BY
    CONSTRUCTION, not by a person. Instances are numbered in image-sha256
    order (label-free); their record ids stay in the private map."""
    from scripts import images
    from scripts.m4fc_audit import both, shown

    index = images.load_index("m4fc")
    picked = []
    for row in rows("m4fc", ("train",)):
        meta = row["meta"]
        if meta.get("verdict") == "true" and any(both(meta, i) for i in ITEMS):
            sha = (index.get(row["image_paths"][0]) or {}).get("sha256")
            picked.append((sha, row))
    out, private = [], {}
    for n, (sha, row) in enumerate(sorted(picked, key=lambda x: x[0]), 1):
        instance = f"true-{n:03d}"
        private[instance] = {"record_id": row["record_id"], "image_path": row["image_paths"][0],
                             "sha256": sha}
        for item in ITEMS:
            if both(row["meta"], item):
                out.append({"gold_id": f"{instance}-{item}", "source": "true_row", "item": item,
                            "claimed": shown(row["meta"][f"claimed_{item}"]),
                            "actual": shown(row["meta"][item]), "label": CONSISTENT})
    return out, private


def cmd_gold() -> int:
    sheet = sheet_gold()
    true_rows, private = true_row_gold()
    A_DIR.mkdir(parents=True, exist_ok=True)
    with GOLD.open("w", encoding="utf-8", newline="\n") as handle:
        for g in sheet + true_rows:
            handle.write(json.dumps(g, ensure_ascii=False) + "\n")
    write_json(GOLD_PRIVATE, {"sheet": rel(SHEET_REVIEWED),
                              "sheet_sha256": sha256_file(SHEET_REVIEWED),
                              "true_row_instances": private})
    for source, gs in (("sheet", sheet), ("true_row", true_rows)):
        counts = {i: ordered(Counter(g["label"] for g in gs if g["item"] == i)) for i in ITEMS}
        print(f"{source}: {len(gs)} pairs  {json.dumps(counts)}")
    print(f"-> {rel(GOLD)} (and the private map {rel(GOLD_PRIVATE)})")
    return 0


def read_gold() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line]


def gold_id_forms() -> dict[str, frozenset[str]]:
    """gold_id -> the id forms of the M4FC row behind it, for the prompt check."""
    by_rid = {r["record_id"]: r for r in rows("m4fc", ("train",), usable=False)}
    instance_rid = {e["instance"]: e["record_id"]
                    for e in json.loads(SHEET_PRIVATE.read_text(encoding="utf-8"))["instances"]}
    instance_rid.update({k: v["record_id"] for k, v in json.loads(
        GOLD_PRIVATE.read_text(encoding="utf-8"))["true_row_instances"].items()})
    out = {}
    for g in read_gold():
        rid = instance_rid[g["gold_id"].rsplit("-", 1)[0]]
        out[g["gold_id"]] = id_forms(by_rid.get(rid) or {"record_id": rid})
    return out


# --------------------------------------------------------------------------
# the judge: Qwen, text-only, few-shot, scored in one forward pass
# --------------------------------------------------------------------------

FEWSHOT_YAML = CONFIGS / "consistency_fewshot.yaml"
#: The judge's answers to "Can both be true at once?", and what each means.
JUDGE_ANSWERS = {"yes": CONSISTENT, "no": CONTRADICTS, "unclear": UNCLEAR}
#: An answer whose three label tokens hold less of the first token's
#: probability than this is no answer: the model wanted to say something else.
MIN_LABEL_MASS = 0.5
ITEM_WORDS = {"date": "date", "location": "place", "people": "people shown", "event": "event"}
JUDGE_SYSTEM = "You compare two statements about one photograph. Answer with one word."
#: The sheet's rule as the person put it ("contradicts only if both can't be
#: true at once") and as the sheet's page wrote it (m4fc_audit.REVIEW_RULE).
JUDGE_TEMPLATE = ("Two statements about the {what} of one photograph.\n"
                  "Statement A, from its caption: {claimed}\n"
                  "Statement B, from what is known about the photograph: {actual}\n"
                  "They contradict only if both can't be true at once. If they say the same, or "
                  "one is just more specific than the other, both can be true. If you can't "
                  "tell, the answer is unclear.\n"
                  "Can both be true at once? Answer yes, no or unclear.")


def fewshot() -> dict[str, Any]:
    import yaml

    doc = yaml.safe_load(FEWSHOT_YAML.read_text(encoding="utf-8"))
    for shot in doc["judge"]:
        if shot["item"] not in ITEMS or shot["answer"] not in JUDGE_ANSWERS:
            raise SystemExit(f"{FEWSHOT_YAML.name}: a judge exemplar with item {shot['item']!r} "
                             f"or answer {shot['answer']!r}")
    return doc


def _text(content: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": content}]


def judge_messages(item: str, claimed: str, actual: str, shots: Sequence[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    """The judge's chat: the system line, every exemplar as a user turn and the
    one-word answer, then the pair. The exemplars are the same, in the same
    order, for every pair."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": _text(JUDGE_SYSTEM)}]
    for s in shots:
        msgs.append({"role": "user", "content": _text(JUDGE_TEMPLATE.format(
            what=ITEM_WORDS[s["item"]], claimed=s["claimed"], actual=s["actual"]))})
        msgs.append({"role": "assistant", "content": _text(s["answer"])})
    msgs.append({"role": "user", "content": _text(JUDGE_TEMPLATE.format(
        what=ITEM_WORDS[item], claimed=claimed, actual=actual))})
    return msgs


def judge_prompt_config(shots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"system": JUDGE_SYSTEM, "template": JUDGE_TEMPLATE, "item_words": ITEM_WORDS,
            "shots": list(shots), "answers": JUDGE_ANSWERS, "min_label_mass": MIN_LABEL_MASS}


def messages_text(messages: Sequence[dict[str, Any]]) -> str:
    return "\n".join(part["text"] for m in messages for part in m["content"] if "text" in part)


def shot_overlap(shots: Sequence[dict[str, Any]], pairs: Iterable[tuple[str, str]]) -> list[int]:
    """Exemplars whose (claimed, actual) equals a gold pair, either way round."""
    seen = {(a.strip().casefold(), b.strip().casefold()) for a, b in pairs}
    seen |= {(b, a) for a, b in seen}
    return [i for i, s in enumerate(shots)
            if (s["claimed"].strip().casefold(), s["actual"].strip().casefold()) in seen]


def neutral_key(prefix: str, value: str) -> str:
    """A key that says nothing about what it keys: a hash, so a row's source
    (sheet or true row, its class) is not in its name."""
    return prefix + hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()[:15]


def write_prompts(path: Path, task: str, rows_: list[dict[str, Any]], prompt_config: dict[str, Any],
                  keys: dict[str, str]) -> dict[str, Any]:
    """An input file for the server's ``llm`` step, seeded-shuffled, with its
    sidecar: the file's sha256, the prompt configuration it was built under,
    and the record that every row passed the id check. ``keys`` (neutral key
    -> what it keys) stays here."""
    random.Random(SEED).shuffle(rows_)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows_)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    meta = {"task": task, "rows": len(rows_), "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "prompt_config": prompt_config,
            "prompt_config_sha256": hashlib.sha256(canonical(prompt_config)).hexdigest(),
            "id_check": {"passed": True, "rows": len(rows_)}, "written_at": utcnow()}
    write_json(sidecar(path), meta)
    write_json(path.with_name(path.stem + ".keys.json"), keys)
    return meta


def sidecar(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


def cmd_prompts_judge_a() -> int:
    """Stage A's judge input: every gold pair of every item, refused if any
    prompt holds an id form its content does not explain, or if an exemplar
    is a gold pair."""
    doc = fewshot()
    shots = doc["judge"]
    gold = read_gold()
    clash = shot_overlap(shots, ((g["claimed"], g["actual"]) for g in gold))
    if clash:
        raise SystemExit(f"exemplars {clash} are gold pairs: an exemplar never comes from the test set")
    forms = gold_id_forms()
    out, keys = [], {}
    for g in gold:
        msgs = judge_messages(g["item"], g["claimed"], g["actual"], shots)
        content = f"{g['claimed']}\n{g['actual']}"
        refuse_ids(messages_text(msgs), forms[g["gold_id"]],
                   explained=content + "\n" + messages_text(msgs[:-1]), what="a judge prompt")
        key = neutral_key("a", g["gold_id"])
        keys[key] = g["gold_id"]
        out.append({"key": key, "task": "judge", "messages": msgs, "content": content})
    path = A_DIR / "judge_in.jsonl"
    meta = write_prompts(path, "judge", out, judge_prompt_config(shots), keys)
    print(f"{meta['rows']} judge prompts, every one id-checked -> {rel(path)} "
          f"(prompt config {meta['prompt_config_sha256'][:12]})")
    return 0


# --------------------------------------------------------------------------
# the runner: text-only Qwen on the server, resumable, one config per task
# --------------------------------------------------------------------------

#: Bump when what a task computes from a prompt changes: it is in the config.
RUNNER_VERSION = 1
LLM_SOFTWARE = ("torch", "transformers", "bitsandbytes", "accelerate")
PILOT_N = 20


class TextLLM:
    """Qwen2.5-VL-7B-Instruct with no image: image_text's pinned NF4 load and
    its checks (QwenVLM), greedy, batch size 1."""

    kind = "qwen"

    def __init__(self) -> None:
        from scripts.image_text import QwenVLM

        self.vlm = QwenVLM()

    def prepare(self) -> None:
        self.vlm.prepare()

    def load(self) -> None:
        self.vlm.load()

    def config(self) -> dict[str, Any]:
        c = self.vlm.config()
        return {k: c[k] for k in ("id", "revision", "licence", "dtype", "attn_implementation",
                                  "quantization")}

    def first_tokens(self, labels: Sequence[str]) -> list[int]:
        """Each label's first token; refused unless all differ, so one forward
        pass tells them apart."""
        tok = self.vlm.processor.tokenizer
        ids = [tok.encode(label, add_special_tokens=False)[0] for label in labels]
        if len(set(ids)) != len(ids):
            raise GuardError(f"the labels {list(labels)} share a first token")
        return ids

    def label_probs(self, messages: list[dict[str, Any]], labels: Sequence[str]
                    ) -> tuple[dict[str, float], float]:
        """P(label) over ``labels`` at the answer's first token, renormalised
        among them, and the mass they held before renormalising."""
        import torch

        text = self.vlm.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
        inputs = self.vlm.processor(text=[text], return_tensors="pt").to(self.vlm.model.device)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            logits = self.vlm.model(**inputs).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)[self.first_tokens(labels)]
        probs = torch.softmax(logp, dim=-1).tolist()
        return dict(zip(labels, probs)), float(logp.exp().sum())

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
        """Greedy text, every token checked to be its step's argmax
        (image_text.check_greedy, image_text's own recorder): the text, whether
        it ended or was cut ("eos" | "length"), its length, and the geometric-mean
        probability of its tokens."""
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        from scripts.image_text import check_greedy

        vlm = self.vlm
        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm.processor(text=[text], return_tensors="pt").to(vlm.model.device)
        recorder = vlm._recorder(None)
        config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                                  eos_token_id=vlm.eos, pad_token_id=vlm.pad,
                                  max_new_tokens=max_new_tokens)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            out = vlm.model.generate(**inputs, generation_config=config,
                                     logits_processor=LogitsProcessorList([recorder]))
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        check_greedy(new, torch.stack(recorder.ids).view(-1).tolist() if recorder.ids else [])
        lps = torch.stack(recorder.lps).view(-1).tolist() if recorder.lps else []
        return {"text": vlm.processor.tokenizer.decode(new, skip_special_tokens=True),
                "finish": "eos" if new and new[-1] in vlm.eos else "length", "n_tokens": len(new),
                "confidence": round(math.exp(sum(lps) / len(lps)), 6) if lps else None}


class StubLLM:
    """Deterministic stand-in: checks everything but the model."""

    kind = "stub"

    def prepare(self) -> None:
        pass

    def load(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"id": "stub", "revision": "0"}

    def label_probs(self, messages: list[dict[str, Any]], labels: Sequence[str]
                    ) -> tuple[dict[str, float], float]:
        digest = hashlib.sha256(canonical(messages)).digest()
        raw = [1 + b for b in digest[:len(labels)]]
        return {lab: x / sum(raw) for lab, x in zip(labels, raw)}, 0.4 + digest[-1] / 425

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
        """A predictor's answer: a place citing the prompt's first evidence id,
        no date or event. An extractor's: the caption's first word as the
        place, nothing else."""
        if messages_text(messages[:1]) == EXTRACT_SYSTEM:
            caption = messages_text(messages[-1:]).split("\n", 1)[0].removeprefix("Caption: ").split()
            answer = json.dumps({"date": None, "location": caption[0] if caption else None, "event": None})
            return {"text": answer, "finish": "eos", "n_tokens": len(answer.split()), "confidence": 0.5}
        ids = re.findall(r"^\[([A-Z]+\d*)\]", messages_text(messages[-1:]), re.M)
        place = {"value": "Somewhere", "evidence": ids[:1]} if ids else {"value": None, "evidence": []}
        answer = json.dumps({"date": {"value": None, "evidence": []}, "location": place,
                             "event": {"value": None, "evidence": []}})
        return {"text": answer, "finish": "eos", "n_tokens": len(answer.split()), "confidence": 0.5}


def run_judge(llm, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    probs, mass = llm.label_probs(row["messages"], tuple(JUDGE_ANSWERS))
    probs = {k: round(v, 6) for k, v in probs.items()}
    answer = max(JUDGE_ANSWERS, key=lambda a: (probs[a], -list(JUDGE_ANSWERS).index(a)))
    if mass < MIN_LABEL_MASS:
        return "off_format", {"p": probs, "label_mass": round(mass, 6), "answer": None,
                              "label": UNCLEAR, "score": None}
    return "ok", {"p": probs, "label_mass": round(mass, 6), "answer": answer,
                  "label": JUDGE_ANSWERS[answer], "score": probs["no"]}


TASKS: dict[str, Callable[[Any, dict[str, Any]], tuple[str, dict[str, Any]]]] = {"judge": run_judge}
TASK_DECODING = {"judge": {"method": "first-token probabilities over the answers, one forward pass",
                           "answers": list(JUDGE_ANSWERS), "batch_size": 1}}


def output_path(inp: Path) -> Path:
    if not inp.stem.endswith("_in"):
        raise SystemExit(f"{inp.name}: an llm input is named <task>_in.jsonl")
    return inp.with_name(inp.stem[:-3] + "_out.jsonl")


def llm_config(llm, task: str, meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from scripts.image_text import software_versions

    config = {"runner_version": RUNNER_VERSION, "task": task, "model": llm.config(),
              "decoding": TASK_DECODING[task], "seed": SEED,
              "prompt_config_sha256": meta["prompt_config_sha256"],
              "software": software_versions(LLM_SOFTWARE) if llm.kind != "stub" else {}}
    return hashlib.sha256(canonical(config)).hexdigest()[:12], config


def read_input(inp: Path, task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The input rows, refused unless the file is the one its sidecar
    describes, for this task, id-checked when it was written -- and no prompt
    holds an id shape its own content does not explain."""
    meta = json.loads(sidecar(inp).read_text(encoding="utf-8"))
    body = inp.read_bytes()
    if hashlib.sha256(body).hexdigest() != meta["sha256"]:
        raise SystemExit(f"{inp.name} is not the file its sidecar describes: refusing")
    if meta["task"] != task or not (meta.get("id_check") or {}).get("passed"):
        raise SystemExit(f"{inp.name} was written for task {meta['task']!r}, or not id-checked")
    out = [json.loads(line) for line in body.decode("utf-8").splitlines() if line]
    for r in out:
        refuse_ids(messages_text(r["messages"]), explained=r.get("content", ""),
                   what=f"input row {r['key']}")
    if len({r["key"] for r in out}) != len(out):
        raise SystemExit(f"{inp.name}: a key appears twice")
    return out, meta


def read_output(path: Path, cid: str) -> dict[str, dict[str, Any]]:
    """key -> row of a resumable output file. A torn last line (a killed run)
    is cut off; a row of another configuration refuses the run."""
    if not path.is_file():
        return {}
    body = path.read_bytes()
    good = body.rfind(b"\n") + 1
    if good < len(body):
        with path.open("r+b") as handle:
            handle.truncate(good)
    done = {}
    for line in body[:good].decode("utf-8").splitlines():
        r = json.loads(line)
        if r["config_id"] != cid:
            raise SystemExit(f"{path.name} holds rows of config {r['config_id']}, not {cid}: "
                             "refusing to mix runs")
        done[r["key"]] = r
    return done


def result_sha(result: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(result)).hexdigest()


def cmd_llm(args) -> int:
    inp = Path(args.input)
    inp = inp if inp.is_absolute() else PROJECT_ROOT / inp
    rows_, meta = read_input(inp, args.task)
    if not args.stub:  # as image_text: nothing past fetch-models may reach the network
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    llm = StubLLM() if args.stub else TextLLM()
    llm.prepare()
    cid, config = llm_config(llm, args.task, meta)
    out_path = output_path(inp)
    cfg_path = out_path.with_name(out_path.stem + ".config.json")
    if cfg_path.is_file():
        if json.loads(cfg_path.read_text(encoding="utf-8"))["config_id"] != cid:
            raise SystemExit(f"{cfg_path.name} is another configuration: refusing to mix runs")
    else:  # written once: it records the code the run started under
        write_json(cfg_path, {"config_id": cid, **config, "code_commit": _code_commit(),
                              "input": inp.name, "input_sha256": meta["sha256"]})
    done = read_output(out_path, cid)
    run = TASKS[args.task]
    if args.recheck:
        return _recheck(llm, run, rows_, done, args.recheck)
    todo = [r for r in rows_ if r["key"] not in done]
    print(f"config {cid}: {len(rows_):,} rows, {len(done):,} done, {len(todo):,} to do", flush=True)
    if not todo:
        return 0
    t = time.perf_counter()
    llm.load()
    print(f"model loaded in {time.perf_counter() - t:.0f}s", flush=True)
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="\n") as handle:
        for i, r in enumerate(todo, 1):
            t = time.perf_counter()
            status, result = run(llm, r)
            secs = time.perf_counter() - t
            handle.write(json.dumps({"key": r["key"], "config_id": cid, "status": status,
                                     "result": result, "result_sha256": result_sha(result),
                                     "seconds": round(secs, 3)}, ensure_ascii=False) + "\n")
            handle.flush()
            per = (time.perf_counter() - started) / i
            print(f"  [{args.task}] {i:,}/{len(todo):,} {status} {secs:.2f}s "
                  f"eta {per * (len(todo) - i) / 3600:.2f}h", flush=True)
            if i == PILOT_N:
                print(f"  [{args.task}] pilot: {per:.2f} s/item over {PILOT_N}; "
                      f"{len(todo):,} items -> {per * len(todo) / 3600:.2f} h", flush=True)
    return 0


def _code_commit() -> str | None:
    """The commit the code was shipped from: code_commit.txt beside the code."""
    path = REPO / "code_commit.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _recheck(llm, run, rows_: list[dict[str, Any]], done: dict[str, dict[str, Any]], n: int) -> int:
    """A FRESH process re-runs finished rows: their results must be byte-identical."""
    by_key = {r["key"]: r for r in rows_}
    picked = sorted(k for k, d in done.items() if d["status"] == "ok")[:n]
    if not picked:
        print("no finished rows to recheck")
        return 1
    llm.load()
    same = [result_sha(run(llm, by_key[k])[1]) == done[k]["result_sha256"] for k in picked]
    print(json.dumps({"checked": len(same), "identical": sum(same)}))
    return 0 if all(same) else 2


def read_results(inp: Path) -> dict[str, dict[str, Any]]:
    """The finished output of an llm input, keyed by its neutral key."""
    out_path = output_path(inp)
    if not out_path.is_file():
        return {}
    cfg = json.loads(out_path.with_name(out_path.stem + ".config.json").read_text(encoding="utf-8"))
    return read_output(out_path, cfg["config_id"])


# --------------------------------------------------------------------------
# stage A: scoring
# --------------------------------------------------------------------------

#: The comparator each item uses in C, fixed before A ran: A measures it,
#: never chooses it. People in C are the gallery's (not measured here).
#: Date is the rule alone (the person's decision, 2026-10-30): the judge
#: called "the 2010s" against "2014-02" a contradiction on the server (no
#: 0.554, yes 0.336), so a date the rule cannot read stays unsupported and is
#: never sent to the judge. Until then it was "rule, then the judge where the
#: rule says unclear"; on the gold pairs the rule decided every one.
FIXED = {"date": "rule", "location": "judge", "event": "judge"}
#: Every comparator a verdict in C may reach, per item, in order. The rule's
#: unclear on a date is the end of the line.
ROUTES = {"date": ("rule",), "location": ("judge",), "event": ("judge",), "people": ("gallery",)}


def date_verdict(claimed: Any, actual: Any) -> dict[str, Any]:
    """A date verdict as C gives it: the rule's, and nothing else. Its
    unclear is unsupported (``comparator_unclear``), never a question for the
    judge."""
    out = date_rule(claimed, actual)
    verdict = {CONTRADICTS: "contradicted", CONSISTENT: "consistent", UNCLEAR: "unsupported"}
    return {**out, "verdict": verdict[out["label"]],
            "why": out["why"] if out["label"] != UNCLEAR else f"comparator_unclear:{out['why']}"}
#: Gate A, as the person set it (2026-10-30).
GATE = {"precision": 0.80, "precision_lower": 0.60, "recall": 0.60, "date_recall": 0.80,
        "false_contradiction": 0.10, "false_contradiction_upper": 0.20}
REPORT_A = REPORTS / "consistency_a.json"
REPORT_A_MD = REPORTS / "consistency_a.md"


def a_predictions(gold: Sequence[dict[str, Any]], judged: dict[str, dict[str, Any]],
                  resolve: Callable[[str], Sequence[str]] | None) -> dict[str, dict[str, dict]]:
    """gold_id -> component -> outcome, for every component A measures."""
    out: dict[str, dict[str, dict]] = {}
    for g in gold:
        comp: dict[str, dict] = {}
        j = judged.get(g["gold_id"])
        if j is not None:
            res = j["result"]
            comp["judge"] = outcome(res["label"], "judge_" + (res["answer"] or "off_format"),
                                    res["score"], "judge")
        if g["item"] == "date":
            comp["rule"] = date_rule(g["claimed"], g["actual"])
        if g["item"] == "people" and resolve is not None:
            comp["qid_rule"] = people_qid_rule(g["claimed"], g["actual"], resolve)
        comp["always_contradicts"] = outcome(CONTRADICTS, "constant", 1.0, "constant")
        comp["always_consistent"] = outcome(CONSISTENT, "constant", 0.0, "constant")
        out[g["gold_id"]] = comp
    return out


def a_metrics(gold: Sequence[dict[str, Any]], preds: dict[str, dict[str, dict]],
              component: str) -> dict[str, Any] | None:
    """One component on one item's gold pairs: the sheet's confusion, precision
    and recall of contradicts, consistent recall, abstention; the true rows'
    false-contradiction rate. Strict precision counts a contradiction the
    person marked unclear as a false one; the lenient figure leaves those out."""
    scored = [(g, preds[g["gold_id"]][component]) for g in gold if component in preds[g["gold_id"]]]
    if not scored:
        return None
    out: dict[str, Any] = {}
    for source in SOURCES:
        part = [(g, p) for g, p in scored if g["source"] == source]
        conf = Counter((g["label"], p["label"]) for g, p in part)
        said = sum(v for (_, pl), v in conf.items() if pl == CONTRADICTS)
        tp = conf[(CONTRADICTS, CONTRADICTS)]
        gold_c = sum(v for (gl, _), v in conf.items() if gl == CONTRADICTS)
        gold_k = sum(v for (gl, _), v in conf.items() if gl == CONSISTENT)
        out[source] = {
            "pairs": len(part),
            "confusion": {f"{gl}->{pl}": conf[(gl, pl)] for gl in COMPARATOR_LABELS
                          for pl in COMPARATOR_LABELS if conf[(gl, pl)]},
            "contradicts_precision": rate(tp, said),
            "contradicts_precision_lenient": rate(tp, said - conf[(UNCLEAR, CONTRADICTS)]),
            "contradicts_recall": rate(tp, gold_c),
            "consistent_recall": rate(conf[(CONSISTENT, CONSISTENT)], gold_k),
            "abstention": rate(sum(v for (_, pl), v in conf.items() if pl == UNCLEAR), len(part)),
            "false_contradiction": rate(conf[(CONSISTENT, CONTRADICTS)], gold_k),
        }
    return out


def gate_a(item: str, m: dict[str, Any] | None) -> dict[str, Any]:
    """Whether an item's fixed configuration clears the person's bars."""
    if m is None:
        return {"passed": None, "why": "not run"}
    sheet, true_ = m["sheet"], m["true_row"]
    fc = true_["false_contradiction"]
    checks = {"false_contradiction": fc["rate"] is not None and fc["rate"] <= GATE["false_contradiction"],
              "false_contradiction_upper": fc["wilson95"] is not None
              and fc["wilson95"][1] <= GATE["false_contradiction_upper"]}
    rec = sheet["contradicts_recall"]
    if item == "date":
        checks["recall"] = rec["rate"] is not None and rec["rate"] >= GATE["date_recall"]
    else:
        pr = sheet["contradicts_precision"]
        checks["precision"] = pr["rate"] is not None and pr["rate"] >= GATE["precision"]
        checks["precision_lower"] = pr["wilson95"] is not None and pr["wilson95"][0] >= GATE["precision_lower"]
        checks["recall"] = rec["rate"] is not None and rec["rate"] >= GATE["recall"]
    return {"passed": all(checks.values()), "checks": checks}


def resolver() -> Callable[[str], Sequence[str]] | None:
    """The gallery's name resolution, or None where this machine has no gallery."""
    from scripts import gallery as gl

    try:
        g = gl.Gallery(gl.face_config_id(gl.make_backend("sface")))
    except Exception as exc:  # no gallery here: the QID rule is not scored
        print(f"  no gallery ({type(exc).__name__}): the people QID rule is not scored")
        return None
    return lambda name: g.resolve(name, "PERSON")


def cmd_score_a() -> int:
    gold = read_gold()
    inp = A_DIR / "judge_in.jsonl"
    judged: dict[str, dict[str, Any]] = {}
    if inp.is_file():
        keys = json.loads(inp.with_name(inp.stem + ".keys.json").read_text(encoding="utf-8"))
        judged = {keys[k]: r for k, r in read_results(inp).items()}
    preds = a_predictions(gold, judged, resolver())
    report: dict[str, Any] = {"written_at": utcnow(), "judge_rows": len(judged),
                              "fixed_configuration": FIXED, "gate_bars": GATE,
                              **score_a(gold, preds)}
    report["judge_status"] = ordered(Counter(r["status"] for r in judged.values()))
    report["contested_true_rows"] = contested_true_rows(gold, preds)
    decisions = true_row_decisions(gold)
    review = {"contested": len(report["contested_true_rows"]), "decided": len(decisions),
              "relabelled": ordered(Counter(f"consistent->{lab}" for lab in decisions.values()
                                            if lab != CONSISTENT))}
    if decisions:
        review.update(score_a([dict(g, label=decisions.get(g["gold_id"], g["label"])) for g in gold],
                              preds))
    report["review"] = review
    if report["contested_true_rows"]:
        write_contested_review(report["contested_true_rows"], gold)
    public = {k: v for k, v in report.items() if k != "contested_true_rows"}
    public["contested_true_rows"] = len(report["contested_true_rows"])
    public_report(public, REPORT_A)
    write_json(A_DIR / "score_a_local.json", report)
    REPORT_A_MD.write_text(render_a(report), encoding="utf-8", newline="\n")
    print(render_a(report))
    return 0


def score_a(gold: Sequence[dict[str, Any]], preds: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """Every component's metrics, and the fixed configuration's gate, per item."""
    out: dict[str, Any] = {"gold": {}, "items": {}, "gate": {}}
    for item in ITEMS:
        g_item = [g for g in gold if g["item"] == item]
        out["gold"][item] = {s: ordered(Counter(g["label"] for g in g_item if g["source"] == s))
                             for s in SOURCES}
        comps = sorted({c for g in g_item for c in preds[g["gold_id"]]})
        out["items"][item] = {c: a_metrics(g_item, preds, c) for c in comps}
        if item in FIXED:
            out["gate"][item] = gate_a(item, out["items"][item].get(FIXED[item]))
    return out


def contested_true_rows(gold: Sequence[dict[str, Any]], preds: dict[str, dict[str, dict]]) -> list[str]:
    """True-row pairs some comparator (never a constant) calls contradictions."""
    return sorted(g["gold_id"] for g in gold if g["source"] == "true_row"
                  and any(p["label"] == CONTRADICTS for c, p in preds[g["gold_id"]].items()
                          if not c.startswith("always_")))


# --------------------------------------------------------------------------
# stage A: the person's review of contested true-row pairs
# --------------------------------------------------------------------------

A_REVIEW = A_DIR / "review"
#: Not "images/": "images/<name>.png" is M4FC's record-id shape, which the
#: page's own id check refuses.
A_REVIEW_COPIES = A_REVIEW / "copies"
CONTESTED_SHEET = A_REVIEW / "contested_true_rows.csv"
#: What the page exports, and where the person puts it for scoring to read.
CONTESTED_REVIEWED = A_REVIEW / "contested_true_rows_reviewed.csv"
CONTESTED_PAGE = A_REVIEW / "review.html"
CONTESTED_TITLE = "Contested true-row pairs"


def contested_rows(contested: Sequence[str], gold: Sequence[dict[str, Any]],
                   private: dict[str, dict[str, Any]], claims: dict[str, str],
                   ext_of: Callable[[str], str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The contested pairs as the M4FC sheet's eight columns, instance by
    instance, and instance -> its neutral image copy's name. The person
    answers the sheet's own question; no comparator's answer is shown."""
    from scripts.m4fc_audit import SHEET_COLUMNS

    by_id = {g["gold_id"]: g for g in gold}
    order = {item: i for i, item in enumerate(ITEMS)}
    picked = sorted(contested, key=lambda gid: (gid.rsplit("-", 1)[0], order[by_id[gid]["item"]]))
    rows_, names = [], {}
    for gid in picked:
        g, instance = by_id[gid], gid.rsplit("-", 1)[0]
        names[instance] = instance + ext_of(private[instance]["image_path"])
        rows_.append(dict(zip(SHEET_COLUMNS, (
            gid, (A_REVIEW_COPIES / names[instance]).relative_to(PROJECT_ROOT).as_posix(),
            claims[private[instance]["record_id"]], g["item"], g["claimed"], g["actual"], "", ""))))
    return rows_, names


def write_contested_review(contested: Sequence[str], gold: Sequence[dict[str, Any]]) -> Path:
    """The sheet, its neutral image copies (checked byte for byte) and the
    page, from m4fc_audit's review page: the person's answers come back as a
    CSV in the sheet's columns. Refused if the page would hold any id form of
    its rows that their claims and values do not."""
    import base64

    from scripts import images
    from scripts.m4fc_audit import (FORMAT_EXT, IMAGE_MIME, REVIEW_RULE, REVIEW_TEMPLATE,
                                    SHEET_COLUMNS, _csv_text)

    private = json.loads(GOLD_PRIVATE.read_text(encoding="utf-8"))["true_row_instances"]
    train = {r["record_id"]: r for r in rows("m4fc", ("train",))}
    index = images.load_index("m4fc")
    rows_, names = contested_rows(
        contested, gold, private, {rid: r["text"] or "" for rid, r in train.items()},
        lambda p: FORMAT_EXT.get((index.get(p) or {}).get("format") or "", Path(p).suffix.lower()))
    A_REVIEW_COPIES.mkdir(parents=True, exist_ok=True)
    for instance, name in names.items():
        src, dst = PROJECT_ROOT / private[instance]["image_path"], A_REVIEW_COPIES / name
        shutil.copyfile(src, dst)
        if sha256_file(dst) != private[instance]["sha256"]:
            raise SystemExit(f"{dst.name}: the copy's bytes are not the image's")
    body = _csv_text(rows_)
    CONTESTED_SHEET.write_bytes(("﻿" + body).encode("utf-8"))
    instances: list[dict[str, Any]] = []
    for r in rows_:
        iid = r["sheet_id"].rsplit("-", 1)[0]
        if instances and instances[-1]["id"] == iid:
            instances[-1]["items"].append(r["sheet_id"])
            continue
        path = PROJECT_ROOT / r["image"]
        mime = IMAGE_MIME.get(path.suffix.lower(), "application/octet-stream")
        instances.append({"id": iid, "claim": r["claim"], "items": [r["sheet_id"]],
                          "image": f"data:{mime};base64,"
                                   f"{base64.b64encode(path.read_bytes()).decode('ascii')}"})
    data = {"sheet_sha256": hashlib.sha256(CONTESTED_SHEET.read_bytes()).hexdigest(),
            "export_name": CONTESTED_REVIEWED.name, "columns": list(SHEET_COLUMNS),
            "rows": [{k: r[k] for k in SHEET_COLUMNS[:6]} for r in rows_], "instances": instances}
    shown_text = json.dumps({**data, "instances": [{k: v for k, v in x.items() if k != "image"}
                                                   for x in instances]}, ensure_ascii=False)
    forms = frozenset().union(*(id_forms(train[private[i]["record_id"]]) for i in names))
    refuse_ids(shown_text, forms, explained="\n".join(
        f"{r['claim']}\n{r['claimed']}\n{r['actual']}" for r in rows_), what="the review page")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (REVIEW_TEMPLATE.replace("M4FC sheet review", CONTESTED_TITLE)
            .replace("__RULE__", html.escape(REVIEW_RULE)).replace("__DATA__", payload))
    CONTESTED_PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"  review page: {rel(CONTESTED_PAGE)} ({len(instances)} images, {len(rows_)} pairs, "
          f"{CONTESTED_PAGE.stat().st_size / 2**20:.1f} MiB). Its export goes to "
          f"{rel(CONTESTED_REVIEWED)}")
    return CONTESTED_PAGE


def true_row_decisions(gold: Sequence[dict[str, Any]], path: Path | None = None) -> dict[str, str]:
    """gold_id -> the person's label of a true-row pair, from the page's
    export: yes -> contradicts, no -> consistent, unclear -> unclear. An empty
    answer keeps the label the pair was built with. Refused if a row is not
    the gold pair its id names."""
    path = path or CONTESTED_REVIEWED
    if not path.is_file():
        return {}
    by_id = {g["gold_id"]: g for g in gold if g["source"] == "true_row"}
    out = {}
    for r in csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))):
        g = by_id.get(r["sheet_id"])
        if g is None or (r["item"], r["claimed"], r["actual"]) != (g["item"], g["claimed"], g["actual"]):
            raise SystemExit(f"{path.name}: {r['sheet_id']} is not that true-row pair of the gold set: "
                             "refusing")
        answer = r["mismatch"].strip().casefold()
        if answer:
            if answer not in SHEET_LABELS:
                raise SystemExit(f"{path.name}: {r['sheet_id']}: the answer {r['mismatch']!r} is not one "
                                 f"of {sorted(SHEET_LABELS)}")
            out[r["sheet_id"]] = SHEET_LABELS[answer]
    return out


def public_report(obj: dict[str, Any], path: Path) -> None:
    """Write an aggregate report, refused if it holds any id form of an M4FC
    train/val row (the only M4FC rows this block reads) or any id shape --
    which is what VERITE's identifiers are: its record ids are row numbers,
    and its file names (true_N, false_N) are a shape."""
    body = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    forms: set[str] = set()
    for r in rows("m4fc", M4FC_SPLITS, usable=False):
        forms |= id_forms(r)
    refuse_ids(body, forms, what=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _cell(r: dict[str, Any] | None) -> str:
    if not r or r["n"] == 0:
        return "–"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} [{lo:.2f}, {hi:.2f}]"


def _mark(ok: bool | None) -> str:
    return "–" if ok is None else ("pass" if ok else "FAIL")


def gate_table(scored: dict[str, Any]) -> list[str]:
    """The fixed configuration of each gated item against the person's bars."""
    lines = ["| item | comparator | sheet precision, strict (bar ≥ 0.80, lower ≥ 0.60) | lenient | "
             "sheet recall (bar ≥ 0.60; date ≥ 0.80) | true-row false contradiction "
             "(bar ≤ 0.10, upper ≤ 0.20) | gate |", "|---|---|---|---|---|---|---|"]
    for item, gate in scored["gate"].items():
        m = scored["items"][item].get(FIXED[item])
        if m is None:
            lines.append(f"| {item} | {FIXED[item]} | – | – | – | – | not run |")
            continue
        s, c = m["sheet"], gate["checks"]
        prec = ("not gated (one class)" if item == "date" else
                f"{_cell(s['contradicts_precision'])} {_mark(c['precision'] and c['precision_lower'])}")
        lines.append(f"| {item} | {FIXED[item]} | {prec} | "
                     f"{'–' if item == 'date' else _cell(s['contradicts_precision_lenient'])} | "
                     f"{_cell(s['contradicts_recall'])} {_mark(c['recall'])} | "
                     f"{_cell(m['true_row']['false_contradiction'])} "
                     f"{_mark(c['false_contradiction'] and c['false_contradiction_upper'])} | "
                     f"**{'PASS' if gate['passed'] else 'FAIL'}** |")
    return lines


def render_a(report: dict[str, Any]) -> str:
    lines = ["# Consistency block, stage A: the comparator on gold pairs", "",
             f"Written {report['written_at']}. Judge rows read: {report['judge_rows']} "
             f"({json.dumps(report['judge_status'])}).", "",
             "Sheet = the person's 110 rows of the four items (assistant rows excluded). "
             "True rows = M4FC train `true` rows, consistent by construction, not by a person.",
             "Strict precision counts a contradiction on a row the person marked unclear as "
             "false; the gate reads the strict figure.", "", "## Gate A, as built", ""]
    lines += gate_table(report) + [""]
    review = report.get("review") or {}
    if review.get("decided"):
        lines += [f"## Gate A after the person's review of contested true-row pairs", "",
                  f"{review['decided']} of {review['contested']} contested pairs decided; relabelled "
                  f"{json.dumps(review['relabelled'])}. Pairs nobody looked at keep their label "
                  "by construction.", ""] + gate_table(review) + [""]
    else:
        lines += [f"Not yet reviewed: {review.get('contested', len(report['contested_true_rows']))} "
                  "contested true-row pairs wait on the person's page; the gate above reads the "
                  "labels by construction.", ""]
    for item in ITEMS:
        lines += [f"## {item}", "", f"Gold: {json.dumps(report['gold'][item])}", ""]
        lines += ["| component | sheet: precision (strict) | (lenient) | sheet: recall | "
                  "sheet: abstain | true rows: false contradiction | true rows: abstain |",
                  "|---|---|---|---|---|---|---|"]
        for comp, m in report["items"][item].items():
            if m is None:
                continue
            s, t = m["sheet"], m["true_row"]
            lines.append(f"| {comp} | {_cell(s['contradicts_precision'])} | "
                         f"{_cell(s['contradicts_precision_lenient'])} | "
                         f"{_cell(s['contradicts_recall'])} | {_cell(s['abstention'])} | "
                         f"{_cell(t['false_contradiction'])} | {_cell(t['abstention'])} |")
        if item in report["gate"]:
            lines += ["", f"Gate ({FIXED[item]}): {json.dumps(report['gate'][item])}"]
        lines.append("")
    lines += [f"Contested true-row pairs (a comparator said contradicts): "
              f"{len(report['contested_true_rows'])}. They go to the person's review page.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# stage B: the rows and images it reads, and where a long run goes
# --------------------------------------------------------------------------

B_DIR = OUT / "b"
#: The corpora stage B reads: M4FC's train and val, VERITE's evaluation rows.
B_DATASETS = ("m4fc", "verite")
#: The person's rule: a run of this many minutes or more goes to their terminal.
TERMINAL_MINUTES = 20
#: ...and so does one projected within a quarter of it: a sample's rate is an
#: estimate, and two estimates of 15 and 12 minutes ran 22 and 60 (2026-10-28/29).
TERMINAL_MARGIN = 0.75


def b_rows(dataset: str) -> list[dict[str, Any]]:
    """Stage B's rows of one corpus, through the guards: M4FC's train and val
    (``rows``), VERITE's evaluation rows (``evaluation_rows``)."""
    if dataset not in B_DATASETS:
        raise GuardError(f"stage B reads {B_DATASETS} only, not {dataset!r}")
    return evaluation_rows() if dataset == "verite" else list(rows(dataset, M4FC_SPLITS))


def sha_lookup() -> Callable[[str], str | None]:
    """path -> sha256, from the index of the dataset that owns the FILE, as
    image_text reads it (two VERITE rows point at a MOCHEG file)."""
    from scripts.image_text import _index_row

    cache: dict[str, dict[str, dict[str, Any]]] = {}
    return lambda path: (_index_row(path, cache) or {}).get("sha256")


def b_images(dataset: str, sha_of: Callable[[str], str | None] | None = None
             ) -> list[tuple[str, str]]:
    """(sha256, path) of each distinct image stage B's rows are views of -- a
    row's first image, as ``view_of`` reads it -- in sha256 order. Refused if
    one is not indexed: a missing image is not an image without faces."""
    sha_of = sha_of or sha_lookup()
    paths: dict[str, set[str]] = defaultdict(set)
    for row in b_rows(dataset):
        first = (row.get("image_paths") or [None])[0]
        sha = sha_of(first) if first else None
        if sha is None:
            raise SystemExit(f"{dataset}: a stage B row's image is not in the image index; "
                             "rebuild it with scripts/images.py index")
        paths[sha].add(first)
    return [(sha, min(p)) for sha, p in sorted(paths.items())]


def spread(items: Sequence[Any], n: int) -> list[Any]:
    """n of ``items`` at an even stride: a timing sample that is neither the
    first n nor one corpus."""
    if n <= 0 or n >= len(items):
        return list(items)
    return [items[i * len(items) // n] for i in range(n)]


def projection(seconds: float, done: int, rest: int, overhead: float = 0.0) -> dict[str, Any]:
    """What a timed sample says about the rest, and where the rest should run."""
    if not done:
        return {"sample": 0, "rest": rest, "minutes": None, "where": None}
    per = seconds / done
    minutes = (per * rest + overhead) / 60
    return {"sample": done, "seconds": round(seconds, 1), "per_item": round(per, 3), "rest": rest,
            "minutes": round(minutes, 1),
            "where": ("the person's terminal" if minutes >= TERMINAL_MARGIN * TERMINAL_MINUTES
                      else "in session")}


def _print_projection(p: dict[str, Any], unit: str, units: str) -> None:
    if not p["sample"]:
        print("nothing left to time")
        return
    print(f"sample: {p['sample']} {units} spread over what was left, {p['seconds']} s "
          f"({p['per_item']} s/{unit}); the other {p['rest']:,} project to {p['minutes']} min "
          f"-> {p['where']}")


# --------------------------------------------------------------------------
# stage B.1: query faces, and the gallery's answer for every name
# --------------------------------------------------------------------------

B_FACES = B_DIR / "faces"
B_PEOPLE = B_DIR / "people.jsonl"
B_PEOPLE_SUMMARY = B_DIR / "people_summary.json"
#: The recogniser of the face configuration the gallery's threshold certifies.
FACE_BACKEND = "sface"
#: Images per call of gallery.embed_images: one progress line each.
FACES_CHUNK = 100
#: Who named a name: the claim (the gallery's claim NER), or the VLM on a free
#: or a forced line.
SIDES = ("claim", "vlm_free", "vlm_forced")


class ImageChanged(GuardError):
    """An image's bytes are not the ones its index row describes."""


def checked_read(path: str, sha256: str) -> bytes:
    """The bytes at ``path``, refused unless they are the indexed ones: faces
    are of the image the index names, or of nothing."""
    from scripts.image_text import read_bytes

    body = read_bytes(path)
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ImageChanged(f"{path}: not the bytes its index row describes")
    return body


def faces_path(cid: str, dataset: str) -> Path:
    return B_FACES / cid / f"{dataset}.jsonl"


def read_faces(path: Path) -> dict[str, dict[str, Any]]:
    """sha256 -> its latest faces row."""
    from scripts import gallery as gl

    return gl.latest(gl.read_jsonl(path), lambda r: r["sha256"]) if path.is_file() else {}


def embed_faces(backend, cid: str, todo: Sequence[tuple[str, str, str]], *,
                chunk: int = FACES_CHUNK) -> float:
    """Faces of each (dataset, sha256, path), by the gallery's own detector and
    recogniser at its query floor, into the corpus's ledger; the seconds it
    took. An image that cannot be read, or whose bytes changed, is a row with an
    error and no faces: missing evidence, which verify-people never reads as
    "no face"."""
    from scripts import gallery as gl

    started = time.perf_counter()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        for ds in dict.fromkeys(d for d, _s, _p in part):
            gl.embed_images(backend, [(s, (lambda p=p, s=s: checked_read(p, s)))
                                      for d, s, p in part if d == ds],
                            faces_path(cid, ds), gl.MIN_FACE["query"])
        n = i + len(part)
        per = (time.perf_counter() - started) / n
        print(f"  [faces] {n:,}/{len(todo):,} ok {per:.2f}s/image "
              f"eta {per * (len(todo) - n) / 3600:.2f}h", flush=True)
    return time.perf_counter() - started


def cmd_faces(args) -> int:
    from scripts import gallery as gl

    t = time.perf_counter()
    backend = gl.make_backend(FACE_BACKEND)
    cid = gl.face_config_id(backend)
    overhead = time.perf_counter() - t
    images_ = {ds: b_images(ds) for ds in args.dataset}
    todo = []
    for ds, imgs in images_.items():
        done = read_faces(faces_path(cid, ds))
        todo += [(ds, s, p) for s, p in imgs if s not in done]
    total = sum(len(v) for v in images_.values())
    print(f"faces, config {cid}: {total:,} images, {total - len(todo):,} done, "
          f"{len(todo):,} to do", flush=True)
    picked = spread(todo, args.sample) if args.sample else todo
    seconds = embed_faces(backend, cid, picked) if picked else 0.0
    if args.sample:
        _print_projection(projection(seconds, len(picked), len(todo) - len(picked), overhead),
                          "image", "images")
    for ds, imgs in images_.items():
        held = read_faces(faces_path(cid, ds))
        got = [held[s] for s, _p in imgs if s in held]
        errors = ordered(Counter(r["error"] for r in got if r["error"]))
        print(f"  {ds}: {len(got):,} of {len(imgs):,} images have a row; "
              f"{sum(1 for r in got if r['faces']):,} with a face of {gl.MIN_FACE['query']} px or "
              f"more; errors {json.dumps(errors) if errors else 'none'}")
    return 0


def claim_ner(dataset: str) -> dict[str, dict[str, Any]]:
    """record_id -> the gallery's cached claim NER of one corpus. Read only:
    nothing here loads a model, so a record the cache lacks is counted, never read."""
    from scripts import gallery as gl

    directory = gl.CLAIM_NER_DIR / gl.config_id_of(gl.ner_config())
    config = directory / "config.json"
    if config.is_file() and json.loads(config.read_text(encoding="utf-8")) != gl.ner_config():
        raise SystemExit(f"{rel(config)} is another NER configuration: refusing to read its cache")
    path = directory / f"{dataset}.jsonl"
    return gl.latest(gl.read_jsonl(path), lambda r: r["record_id"]) if path.is_file() else {}


def image_text_rows(dataset: str) -> dict[str, dict[str, Any]] | None:
    """sha256 -> the image_text row every view reads (IMAGE_TEXT_CONFIG), or
    None where this machine holds no ledger for the corpus."""
    from scripts.image_text import OUT_ROOT, scan_ledger

    path = OUT_ROOT / IMAGE_TEXT_CONFIG / f"{dataset}.jsonl"
    if not path.is_file():
        return None
    _, kept, _ = scan_ledger(path, dataset=dataset, config_id=IMAGE_TEXT_CONFIG,
                             keep=lambda r: True)
    return {r["sha256"]: r for r in kept}


def vlm_people(image_row: dict[str, Any]) -> dict[str, str]:
    """name -> "vlm_free" | "vlm_forced": the VLM's PERSON items, read the
    sanctioned way (entities_of, forced lines kept and marked). A name on a
    free line and on a forced one is free."""
    from scripts.image_text import entities_of

    out: dict[str, str] = {}
    for e in entities_of(image_row, forced="mark"):
        name = e["name"].strip()
        if e["category"] == "PERSON" and name and out.get(name) != "vlm_free":
            out[name] = "vlm_forced" if e["forced"] else "vlm_free"
    return out


def claim_people(entry: dict[str, Any]) -> list[str]:
    """The PERSON spans of one record's cached claim NER."""
    return sorted({s["text"].strip() for s in entry.get("spans", [])
                   if s["type"] == "PERSON" and s["text"].strip()})


def b_names(dataset: str, sha_of: Callable[[str], str | None]
            ) -> tuple[dict[str, dict[str, set[str]]], dict[str, str | None], Counter]:
    """(sha256 -> name -> the sides that named it, sha256 -> M4FC split, what
    is missing). A record without cached NER and an image without an image_text
    row are counted as missing: their names are unknown, not absent."""
    named: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    split_of: dict[str, str | None] = {}
    missing: Counter = Counter()
    spans = claim_ner(dataset)
    for row in b_rows(dataset):
        sha = sha_of(row["image_paths"][0])
        split_of[sha] = row["split"] if dataset == "m4fc" else None
        entry = spans.get(str(row["record_id"]))
        if entry is None:
            missing["records_without_claim_ner"] += 1
            continue
        for name in claim_people(entry):
            named[sha][name].add("claim")
    by_sha = image_text_rows(dataset)
    if by_sha is None:
        missing["images_without_an_image_text_ledger"] = len(split_of)
    else:
        for sha in split_of:
            if sha not in by_sha:
                missing["images_without_an_image_text_row"] += 1
                continue
            for name, side in vlm_people(by_sha[sha]).items():
                named[sha][name].add(side)
    return named, split_of, missing


def verdict_of(answer: dict[str, Any]) -> dict[str, Any]:
    """What stage B keeps of one Gallery.verify answer: the decision, its
    reason's code, and what a person needs to check it."""
    return {"qids": [e["qid"] for e in answer["entities"]], "match": answer["match"],
            "score": answer["score"], "reason": answer["reason"].split(":", 1)[0],
            "in_gallery": answer["in_gallery"], "faces_in_image": answer["faces_in_image"],
            "matched_entity": answer["matched_entity"], "query_face": answer["query_face"],
            "matched_reference": answer["matched_reference"],
            "same_photo": answer.get("same_photo")}


def verify_names(dataset: str, named: dict[str, dict[str, set[str]]],
                 faces: dict[str, dict[str, Any]], gallery, threshold: float,
                 split_of: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
    """One row per (image, name): the sides that named it, and the gallery's
    answer at the certified threshold. An image whose faces could not be
    computed is faces_error, never no_face_in_image: its faces are unknown,
    not absent."""
    out = []
    for sha in sorted(named):
        frow = faces.get(sha)
        if frow is None:
            raise SystemExit(f"{dataset}: an image has no faces row; run `faces` first")
        for name in sorted(named[sha]):
            if frow["error"]:
                v = {"qids": gallery.resolve(name, "PERSON"), "match": None, "score": None,
                     "reason": "faces_error", "in_gallery": None, "faces_in_image": None,
                     "matched_entity": None, "query_face": None, "matched_reference": None,
                     "same_photo": None}
            else:
                v = verdict_of(gallery.verify(frow["faces"], name, threshold))
            out.append({"dataset": dataset, "sha256": sha, "split": (split_of or {}).get(sha),
                        "name": name, "sides": sorted(named[sha][name]), **v})
    return out


def people_counts(part: Sequence[dict[str, Any]]) -> dict[str, Any]:
    resolve = Counter("one" if len(r["qids"]) == 1 else "several" if r["qids"] else "none"
                      for r in part)
    return {"pairs": len(part), "names": len({r["name"] for r in part}),
            "resolve": {k: resolve[k] for k in ("one", "several", "none")},
            "accepted": sum(1 for r in part if r["match"] is True),
            "reasons": ordered(Counter(r["reason"] for r in part))}


def cmd_verify_people(args) -> int:
    """Every name stage B knows of an image -- the claim's and the VLM's --
    through Gallery.verify at the certified threshold. Rewritten whole on every
    run: rerun it when the M4FC image_text ledger comes back."""
    from scripts import gallery as gl

    cid = gl.face_config_id(gl.make_backend(FACE_BACKEND))
    calib = gl.CALIBRATION_DIR / f"{FACE_BACKEND}.json"
    threshold = gl.certified_threshold(
        json.loads(calib.read_text(encoding="utf-8")) if calib.is_file() else None, cid)
    if threshold is None:
        raise SystemExit(f"no certified threshold for face config {cid}: verify-people decides at "
                         "the gallery's certified threshold or not at all")
    gallery = gl.Gallery(cid)
    sha_of = sha_lookup()
    out, summary = [], {"face_config": cid, "threshold": threshold, "written_at": utcnow(),
                        "datasets": {}}
    for ds in args.dataset:
        faces = read_faces(faces_path(cid, ds))
        images_ = [s for s, _p in b_images(ds, sha_of)]
        if any(s not in faces for s in images_):
            raise SystemExit(f"{ds}: {sum(s not in faces for s in images_):,} of {len(images_):,} "
                             "images have no faces row; run `faces` first")
        named, split_of, missing = b_names(ds, sha_of)
        got = verify_names(ds, named, faces, gallery, threshold, split_of)
        out += got
        frows = [faces[s] for s in images_]
        summary["datasets"][ds] = {
            "images": len(images_), "with_a_face": sum(1 for r in frows if r["faces"]),
            "faces_errors": sum(1 for r in frows if r["error"]), "missing": ordered(missing),
            "sides": {side: people_counts([r for r in got if side in r["sides"]]) for side in SIDES}}
        if ds == "m4fc":
            summary["datasets"][ds]["by_split"] = {
                sp: {side: people_counts([r for r in got if side in r["sides"] and r["split"] == sp])
                     for side in SIDES} for sp in M4FC_SPLITS}
    gl.write_jsonl(B_PEOPLE, out)
    write_json(B_PEOPLE_SUMMARY, summary)
    print(f"verify-people at the certified {threshold} (face config {cid})")
    for ds, s in summary["datasets"].items():
        print(f"{ds}: {s['images']:,} images, {s['with_a_face']:,} with a face, faces errors "
              f"{s['faces_errors']}; missing {json.dumps(s['missing']) if s['missing'] else 'nothing'}")
        print(f"  {'side':<11}{'pairs':>7}{'names':>7}{'one':>7}{'several':>9}{'no one':>8}"
              f"{'accepted':>10}  reasons")
        for side, c in s["sides"].items():
            r = c["resolve"]
            print(f"  {side:<11}{c['pairs']:>7,}{c['names']:>7,}{r['one']:>7,}{r['several']:>9,}"
                  f"{r['none']:>8,}{c['accepted']:>10,}  {json.dumps(c['reasons'])}")
    print(f"-> {rel(B_PEOPLE)} (local: it holds names) and {rel(B_PEOPLE_SUMMARY)}")
    return 0


# --------------------------------------------------------------------------
# stage B.2: retrieval, for M4FC -- the one corpus here whose claims have a bound
# --------------------------------------------------------------------------

B_RETRIEVAL = B_DIR / "retrieval"
RETRIEVE_K = 10
#: An excerpt is at most this many characters of a document's text.
EXCERPT_CHARS = 600
#: Bump when what a retrieval row holds, or how it is made, changes: it is in the config.
#: 2: the densest window, entities decoded (1 anchored on a word's first occurrence).
RETRIEVE_VERSION = 2


def retrieval_query(view: ModelView) -> str:
    """What retrieval searches by: the claim, and nothing else of the view."""
    return view.claim or ""


def retrieval_config() -> dict[str, Any]:
    from configs.paths import AS_OF_YAML, FACTCHECKERS_YAML
    from scripts import retrieve

    return {"version": RETRIEVE_VERSION, "k": RETRIEVE_K, "query": "the claim (retrieval_query)",
            "terms": f"retrieve.query_terms: the {retrieve.RAREST} rarest words outside half the store",
            "rarest": retrieve.RAREST, "df_sql": retrieve.DF_SQL,
            "search_sql_sha256": hashlib.sha256(retrieve.SEARCH_SQL.encode("utf-8")).hexdigest(),
            "bound": "retrieve.evidence_bound, strictly before", "excerpt_chars": EXCERPT_CHARS,
            "as_of_sha256": sha256_file(AS_OF_YAML),
            "factcheckers_sha256": sha256_file(FACTCHECKERS_YAML),
            # size and mtime change with every rebuild and cost nothing to read;
            # the sha256 of 13 GB is the red-test's and retrieval_bench's identity
            "index": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in (retrieve.BM25_DB, retrieve.TEMPORAL_DB)}}


def excerpt(text: str | None, title: str | None, terms: Sequence[str],
            n: int = EXCERPT_CHARS) -> tuple[str, str]:
    """(at most n characters of a document's text, what anchors them).

    The store keeps a page's text as extracted: its title, then the page's
    menus, so its opening is mostly furniture, and a word the query was
    searched by can first occur in a menu or a footer. So the excerpt is the
    window holding the most distinct search words, as FTS5's snippet picks a
    fragment, starting a quarter-window before the first of them; ties go to
    the earliest ("term"). A text holding none as written -- the index matches
    stems, this does not -- gives its opening ("opening"). Entities are
    decoded, whitespace collapsed, a cut end marked."""
    body = " ".join(html.unescape(text or "").split())
    head = " ".join(html.unescape(title or "").split())
    if head and body.startswith(head):
        body = body[len(head):].lstrip()
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)"
    found = [(m.start(), m.group(0).casefold()) for m in re.finditer(pattern, body, re.I)] if terms else []

    def window(i: int) -> tuple[int, int]:
        lo = max(0, found[i][0] - n // 4)
        return len({w for p, w in found if lo <= p and p + len(w) <= lo + n}), -i

    start = 0
    if found:
        at = found[max(range(len(found)), key=window)][0]
        if at > n // 4:
            space = body.find(" ", at - n // 4, at)
            start = space + 1 if space >= 0 else at
    end = min(len(body), start + n)
    if end < len(body):
        space = body.rfind(" ", start, end)
        end = space if space > start else end
    return (("…" if start else "") + body[start:end] + ("…" if end < len(body) else ""),
            "term" if found else "opening")


def read_doc(doc_id: str) -> dict[str, Any] | None:
    """A document of the evidence store, or None if its file is gone or torn."""
    from configs.paths import BIBLE

    try:
        return json.loads((BIBLE / "docs" / f"{doc_id}.json").read_bytes())
    except (OSError, ValueError):
        return None


def hit_row(hit, terms: Sequence[str], doc: dict[str, Any] | None) -> dict[str, Any]:
    """A hit as the ledger keeps it: never its URL (a Document has none). Its
    title is decoded as its excerpt is: the store keeps both as the page wrote them."""
    text, anchor = (excerpt(doc.get("text"), doc.get("title") or hit.title, terms) if doc
                    else (None, "doc_missing"))
    return {"doc_id": hit.doc_id, "title": html.unescape(hit.title or ""),
            "published_at": hit.published_at,
            "host": hit.source_domain, "score": round(hit.score, 5), "excerpt": text,
            "anchor": anchor}


def plan_retrieval(dataset: str, rows_: Iterable[dict[str, Any]],
                   sha_of: Callable[[str], str | None]) -> list[dict[str, Any]]:
    """One query per row, in image-sha256 order: the claim, through the view
    and the id check, and the bound its dataset's as-of rule gives (None: no
    retrieval). Keyed by image, so two rows sharing one are refused."""
    from scripts import retrieve

    out = []
    for row in rows_:
        view = view_of(row, sha_of=sha_of, image_rows={})
        query = refuse_ids(retrieval_query(view), id_forms(row), explained=view.content(),
                           what="a retrieval query")
        try:
            before, bound = retrieve.evidence_bound(dataset, row.get("meta") or {})
        except retrieve.RetrievalError:
            before, bound = None, None
        out.append({"sha256": view.image_sha256, "query": query, "before": before, "bound": bound})
    if len({x["sha256"] for x in out}) != len(out):
        raise SystemExit(f"{dataset}: two rows share an image, and retrieval rows are keyed by image")
    return sorted(out, key=lambda x: x["sha256"])


def read_retrieval(path: Path, cid: str | None = None) -> dict[str, dict[str, Any]]:
    """sha256 -> its retrieval row. A row of another configuration refuses."""
    from scripts import gallery as gl

    out: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for r in gl.read_jsonl(path):
            if cid is not None and r["config_id"] != cid:
                raise SystemExit(f"{path.name} holds rows of retrieval config {r['config_id']}, not "
                                 f"{cid}: refusing to mix runs")
            out[r["sha256"]] = r
    return out


def run_retrieval(items: Sequence[dict[str, Any]], path: Path, cid: str,
                  search: Callable[[str, str], tuple[list[Any], list[str], bool]],
                  doc: Callable[[str], dict[str, Any] | None] = read_doc) -> float:
    """Each item's hits and excerpts, one ledger row each, fsynced; the seconds
    it took. A row with no bound or no searchable word is written with that as
    its error and no hits: retrieval refused it, which is not "nothing found"."""
    from scripts import gallery as gl
    from scripts.retrieve import words

    ledger = gl.JsonlLedger(path)
    ledger.open()
    started = time.perf_counter()
    try:
        for i, it in enumerate(items, 1):
            t = time.perf_counter()
            row = {"sha256": it["sha256"], "config_id": cid, "before": it["before"],
                   "bound": it["bound"], "terms": None, "fell_back": None, "hits": [], "error": None,
                   "query_sha256": hashlib.sha256(it["query"].encode("utf-8")).hexdigest()}
            if it["before"] is None:
                row["error"] = "no_bound"
            elif not words(it["query"]):
                row["error"] = "empty_query"
            else:
                hits, terms, fell_back = search(it["query"], it["before"])
                row.update(terms=terms, fell_back=fell_back,
                           hits=[hit_row(h, terms, doc(h.doc_id)) for h in hits])
            row["seconds"] = round(time.perf_counter() - t, 3)
            ledger.append(row)
            per = (time.perf_counter() - started) / i
            print(f"  [retrieve] {i:,}/{len(items):,} {row['error'] or 'ok'} {row['seconds']:.2f}s "
                  f"hits {len(row['hits'])} eta {per * (len(items) - i) / 3600:.2f}h", flush=True)
    finally:
        ledger.close()
    return time.perf_counter() - started


def store_pages(doc_ids: Iterable[str]) -> dict[str, tuple[str, str | None, str | None]]:
    """doc_id -> (published_at, source_domain, url) as the temporal table holds
    them: the very values the statement judged."""
    import sqlite3

    from scripts import retrieve

    ids, out = sorted(set(doc_ids)), {}
    con = sqlite3.connect(f"file:{retrieve.TEMPORAL_DB}?mode=ro", uri=True)
    try:
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            out.update((d, (p, s, u)) for d, p, s, u in con.execute(
                "SELECT doc_id, published_at, source_domain, url FROM docs WHERE doc_id IN "
                f"({','.join('?' * len(part))})", part))
    finally:
        con.close()
    return out


def check_retrieval(items: Sequence[dict[str, Any]], held: dict[str, dict[str, Any]],
                    pages: Callable[[Iterable[str]], dict[str, tuple[str, str | None, str | None]]]
                    = store_pages) -> dict[str, Any]:
    """The ledger against what it must be -- the strict twin of the statement.
    One row per claim, at its claim's own bound and query; every hit strictly
    before that bound by the temporal table's date, and allowed by the
    fact-checker rules re-judged in Python (factcheckers.blocker) on the page as
    the index stores it. Problems are counted; an empty count is a pass."""
    from scripts import factcheckers, retrieve

    problems: Counter = Counter()
    want = {it["sha256"]: it for it in items}
    problems["claims without a row"] = sum(1 for s in want if s not in held)
    problems["rows of no claim"] = sum(1 for s in held if s not in want)
    rows_ = [held[s] for s in want if s in held]
    for r in rows_:
        it = want[r["sha256"]]
        if (r["before"], r["bound"]) != (it["before"], it["bound"]):
            problems["a row's bound is not its claim's"] += 1
        if r["query_sha256"] != hashlib.sha256(it["query"].encode("utf-8")).hexdigest():
            problems["a row's query is not its claim"] += 1
        ids = [h["doc_id"] for h in r["hits"]]
        problems["more than k hits"] += len(ids) > RETRIEVE_K
        problems["a document twice in one row"] += len(ids) != len(set(ids))
        problems["a hit at or after its bound (ledger)"] += sum(
            1 for h in r["hits"] if not h["published_at"] < r["before"])
        problems["an excerpt longer than the cap"] += sum(
            1 for h in r["hits"] if h["excerpt"] and len(h["excerpt"]) > EXCERPT_CHARS + 2)
    stored = pages(h["doc_id"] for r in rows_ for h in r["hits"])
    why = factcheckers.blocker(retrieve.load_blocklist())
    blocked: Counter = Counter()
    for r in rows_:
        for h in r["hits"]:
            page = stored.get(h["doc_id"])
            if page is None:
                problems["a hit the index does not hold"] += 1
                continue
            published, domain, url = page
            problems["a hit's date differs from the index's"] += published != h["published_at"]
            problems["a hit at or after its bound (index)"] += not published < r["before"]
            reason = why(h["doc_id"], *factcheckers.page(domain, url))
            if reason:
                blocked[reason] += 1
    problems.update({f"a hit the fact-checker rules block ({k})": v for k, v in blocked.items()})
    hits = [h for r in rows_ for h in r["hits"]]
    return {"problems": ordered(Counter({k: v for k, v in problems.items() if v})),
            "rows": len(rows_), "hits": len(hits),
            "hits_per_row": ordered(Counter(len(r["hits"]) for r in rows_)),
            "anchors": ordered(Counter(h["anchor"] for h in hits)),
            "errors": ordered(Counter(r["error"] for r in rows_ if r["error"]))}


def retrieval_ledger(splits: Sequence[str] = M4FC_SPLITS, sha_of: Callable[[str], str | None] | None = None,
                     *, start: bool = False) -> tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """(config id, the claims, the ledger's rows): refused if the ledger was
    written under another configuration than the one computed now. Only a run
    (``start``) writes the configuration of a ledger that has none."""
    config = retrieval_config()
    cid = hashlib.sha256(canonical(config)).hexdigest()[:12]
    cfg_path = B_RETRIEVAL / "m4fc.config.json"
    if cfg_path.is_file():
        if json.loads(cfg_path.read_text(encoding="utf-8"))["config_id"] != cid:
            raise SystemExit(f"{rel(cfg_path)} is another retrieval configuration (the index, a rule or "
                             "a setting changed): refusing to mix runs")
    elif start:
        write_json(cfg_path, {"config_id": cid, **config})
    else:
        raise SystemExit("no retrieval ledger here: run `retrieve --dataset m4fc` first")
    items = plan_retrieval("m4fc", rows("m4fc", splits), sha_of or sha_lookup())
    return cid, items, read_retrieval(B_RETRIEVAL / "m4fc.jsonl", cid)


def cmd_retrieve(args) -> int:
    from scripts import retrieve

    if args.dataset != "m4fc":
        raise SystemExit(f"{args.dataset}: its claims carry no date (configs/as_of.yaml), so they get "
                         "no evidence: retrieval refuses it")
    cid, items, done = retrieval_ledger(args.splits, start=not args.check)
    path = B_RETRIEVAL / "m4fc.jsonl"
    if args.check:
        report = check_retrieval(items, done)
        write_json(B_RETRIEVAL / "m4fc.check.json", {"config_id": cid, "checked_at": utcnow(), **report})
        print(json.dumps(report, indent=1))
        print(("PASSED" if not report["problems"] else "FAILED") + f" -> "
              f"{rel(B_RETRIEVAL / 'm4fc.check.json')}")
        return 0 if not report["problems"] else 2
    todo = [it for it in items if it["sha256"] not in done]
    print(f"retrieval, config {cid}: {len(items):,} M4FC {'+'.join(args.splits)} claims, "
          f"{len(items) - len(todo):,} done, {len(todo):,} to do", flush=True)
    picked = spread(todo, args.sample) if args.sample else todo
    seconds = 0.0
    if picked:
        t = time.perf_counter()
        con = retrieve.connect()
        overhead = time.perf_counter() - t
        try:
            seconds = run_retrieval(picked, path, cid, lambda q, before: retrieve.search_with_terms(
                q, before, RETRIEVE_K, con=con))
        finally:
            con.close()
        if args.sample:
            _print_projection(projection(seconds, len(picked), len(todo) - len(picked), overhead),
                              "query", "queries")
    held = read_retrieval(path, cid)
    got = [held[it["sha256"]] for it in items if it["sha256"] in held]
    ok = [r for r in got if not r["error"]]
    secs = sorted(r["seconds"] for r in ok)
    errors = ordered(Counter(r["error"] for r in got if r["error"]))
    print(f"  ledger: {len(got):,} of {len(items):,} claims; errors "
          f"{json.dumps(errors) if errors else 'none'}; "
          f"fewer than {RETRIEVE_K} hits {sum(1 for r in ok if len(r['hits']) < RETRIEVE_K)}, "
          f"none {sum(1 for r in ok if not r['hits'])}; fell back to one word "
          f"{sum(1 for r in ok if r['fell_back'])}")
    if secs:
        print(f"  seconds per query: median {secs[len(secs) // 2]:.2f}, mean "
              f"{sum(secs) / len(secs):.2f}, max {secs[-1]:.2f}; excerpts anchored "
              f"{json.dumps(ordered(Counter(h['anchor'] for r in ok for h in r['hits'])))}")
    return 0


# --------------------------------------------------------------------------
# stage B.3: the context predictor -- what the image's evidence says of its
# date, place and event. It never sees the claim.
# --------------------------------------------------------------------------

B_PREDICT_PILOT = B_DIR / "predict_pilot_in.jsonl"
B_PREDICT = B_DIR / "predict_in.jsonl"
PREDICT_ITEMS = ("date", "location", "event")
#: The seeded slice of M4FC train a predictor prompt is piloted on: the only
#: rows it is developed on. Val and VERITE never are.
PILOT_SLICE = 30
#: A prediction sees the top of its claim's retrieval; a description, OCR
#: text and a document's title are cut at these lengths (4 of the 15,105 top
#: titles of M4FC train+val exceed 300 characters, the longest 505).
PREDICT_DOCS = 5
TEXT_CHARS = 1000
TITLE_CHARS = 200
PREDICT_MAX_NEW_TOKENS = 200
#: What a prediction is made of, per variant: M4FC train+val with its claim's
#: retrieval, M4FC val again without it (the echo rate's contrast), VERITE
#: from the image alone (no date, no retrieval).
VARIANTS = ("retrieval", "no_retrieval", "image_only")
#: Values meaning "no value", read as null and flagged.
NULL_WORDS = frozenset({"", "null", "none", "unknown", "n/a", "na", "not known", "not stated",
                        "unclear"})
PREDICT_SYSTEM = ("You work out the context of a news photograph from evidence about it. "
                  "Answer with one JSON object and nothing else.")
PREDICT_TEMPLATE = (
    "Evidence about one news photograph, each piece with its id in brackets:\n"
    "\n"
    "{evidence}\n"
    "\n"
    "From this evidence alone, say when the photograph was taken, where, and what event it shows.\n"
    "- date: when it was taken, as YYYY, YYYY-MM or YYYY-MM-DD, no more precise than the evidence.\n"
    "- location: where it was taken, as precisely as the evidence allows; name a country only "
    "where the evidence gives one.\n"
    "- event: what is happening in it, in a few words.\n"
    "Write null where the evidence does not tell. Do not guess from what is usual.\n"
    "Documents were found by a text search and may be about something else: use one only where "
    "it is about what this photograph shows.\n"
    "A document's date is when it was published, not when this photograph was taken: give a date "
    "only where the evidence says when the photograph itself was taken.\n"
    "A line marked weaker was written to fill the image model's answer form, not freely.\n"
    "For each item, first list the ids of the evidence it rests on, each as a quoted string, then "
    "the value. A value that rests on none of the evidence is not allowed: write null instead.\n"
    "Answer with this JSON object and nothing else:\n"
    '{{"date": {{"evidence": [...], "value": ...}}, "location": {{"evidence": [...], "value": ...}}, '
    '"event": {{"evidence": [...], "value": ...}}}}')
ENTITY_WORDS = {"PERSON": "a person", "ORG": "an organisation", "GPE": "a place",
                "EVENT": "an event", "DATE": "a date"}
BASIS_WORDS = {"text": "written in the photograph", "seen": "recognised from how it looks",
               "unstated": "how it knows is not stated"}


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text.rfind(" ", 0, n)
    return text[:cut if cut > 0 else n] + "…"


def predict_evidence(view: ModelView) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The evidence lines of a view, each under a short id the answer cites,
    and what each id stands for. Never the claim: a prediction is of the image."""
    lines: list[str] = []
    ev: dict[str, dict[str, Any]] = {}
    if view.description:
        lines.append(f"[DESC] An image model's description of the photograph: "
                     f"{_clip(view.description, TEXT_CHARS)}")
        ev["DESC"] = {"kind": "description"}
    if view.ocr:
        lines.append(f"[OCR] Text read in the photograph: {_clip(view.ocr, TEXT_CHARS)}")
        ev["OCR"] = {"kind": "ocr"}
    for i, e in enumerate(view.entities, 1):
        how = BASIS_WORDS.get(e.basis, e.basis)
        conf = "" if e.confidence is None else f", confidence {e.confidence:.2f}"
        weak = "; weaker" if e.forced else ""
        lines.append(f"[E{i}] The image model names {ENTITY_WORDS[e.category]}: {e.name} "
                     f"({how}{conf}{weak})")
        ev[f"E{i}"] = {"kind": "entity", "category": e.category, "name": e.name, "forced": e.forced}
    for i, p in enumerate(view.people, 1):
        lines.append(f"[P{i}] A face in the photograph matches the reference photographs of {p.name}")
        ev[f"P{i}"] = {"kind": "person", "name": p.name, "qid": p.qid, "score": p.score}
    for i, d in enumerate(view.documents[:PREDICT_DOCS], 1):
        lines.append(f"[D{i}] A document published {d.published_at} on {d.host}: "
                     f"{_clip(d.title, TITLE_CHARS)}\n     {d.excerpt}")
        ev[f"D{i}"] = {"kind": "document", "doc_id": d.doc_id, "published_at": d.published_at,
                       "host": d.host}
    return lines, ev


def predict_shot() -> tuple[ModelView, dict[str, Any]]:
    """The predictor's one worked example (configs/consistency_fewshot.yaml):
    an invented view and its answer."""
    return _predict_shot(FEWSHOT_YAML.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _predict_shot(text: str) -> tuple[ModelView, dict[str, Any]]:
    import yaml

    shot = yaml.safe_load(text)["predict"]
    v = shot["view"]
    view = ModelView(None, None, v.get("ocr"), v.get("description"),
                     tuple(Entity(**e) for e in v.get("entities", [])),
                     tuple(Person(**p) for p in v.get("people", [])),
                     tuple(Document(**d) for d in v.get("documents", [])))
    return view, shot["answer"]


def _predict_user(view: ModelView) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lines, ev = predict_evidence(view)
    return {"role": "user", "content": _text(PREDICT_TEMPLATE.format(evidence="\n".join(lines)))}, ev


def predict_messages(view: ModelView, shot: tuple[ModelView, dict[str, Any]] | None = None
                     ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The predictor's chat for one image -- the worked example as a prior
    turn, then the image's evidence -- and the image's own evidence ids."""
    shot_view, answer = shot or predict_shot()
    example, _ = _predict_user(shot_view)
    user, ev = _predict_user(view)
    return ([{"role": "system", "content": _text(PREDICT_SYSTEM)}, example,
             {"role": "assistant", "content": _text(json.dumps(answer, ensure_ascii=False))}, user],
            ev)


def predict_content(view: ModelView) -> str:
    """What a predictor prompt is built from: the view's content without its claim."""
    return replace(view, claim=None).content()


def predict_prompt_config() -> dict[str, Any]:
    return {"system": PREDICT_SYSTEM, "template": PREDICT_TEMPLATE, "shot": fewshot()["predict"],
            "entity_words": ENTITY_WORDS,
            "basis_words": BASIS_WORDS, "documents": PREDICT_DOCS,
            "documents_distinct": "by title and excerpt, the higher rank kept", "text_chars": TEXT_CHARS,
            "title_chars": TITLE_CHARS, "excerpt_chars": EXCERPT_CHARS, "items": list(PREDICT_ITEMS),
            "null_words": sorted(NULL_WORDS), "max_new_tokens": PREDICT_MAX_NEW_TOKENS}


def parse_item(item: str, raw: Any, allowed: set[str], flags: list[str]) -> dict[str, Any]:
    """One item of an answer: a value with the ids it rests on, or null. A value
    citing no evidence, an id the prompt did not give, or a date not in one of
    the three forms is an error: the item is then no prediction."""
    if not isinstance(raw, dict) or "value" not in raw:
        return {"value": None, "evidence": [], "error": "bad_item"}
    value, cited = raw["value"], raw.get("evidence", [])
    if item == "date" and isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
        flags.append("number_date")
    if isinstance(value, str) and value.strip().casefold() in NULL_WORDS:
        if value.strip():
            flags.append("null_word")
        value = None
    if value is None:
        return {"value": None, "evidence": [], "error": None}
    if not isinstance(value, str):
        return {"value": None, "evidence": [], "error": "bad_value"}
    value = " ".join(value.split())
    if not isinstance(cited, list) or not all(isinstance(x, str) for x in cited):
        return {"value": value, "evidence": [], "error": "bad_evidence"}
    ids = [x.strip() for x in cited]
    if any(x.startswith("[") or x.endswith("]") for x in ids):
        flags.append("bracketed_ids")
        ids = [x.strip("[] ") for x in ids]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {"value": value, "evidence": [], "error": "no_evidence"}
    if any(x not in allowed for x in ids):
        return {"value": value, "evidence": ids, "error": "unknown_evidence"}
    if item == "date" and not (_YMD.fullmatch(value) and date_interval(value)):
        return {"value": value, "evidence": ids, "error": "bad_date"}
    return {"value": value, "evidence": ids, "error": None}


_EVIDENCE_OPEN = re.compile(r'"evidence"\s*:\s*\[')
_BARE_ID = re.compile(r"\[?\s*[A-Za-z]+\d*\s*\]?")


def quote_bare_ids(text: str) -> str:
    """``text`` with every unquoted id inside an "evidence" array quoted:
    [E3, D1] -> ["E3", "D1"]. The first pilot (2026-10-30) wrote 26 of 30
    answers so, the ids as its prompt shows them, and the meaning is not in
    doubt. Anything else in an array is left as it is, to fail as JSON."""
    out, pos = [], 0
    for m in _EVIDENCE_OPEN.finditer(text):
        if m.start() < pos:
            continue
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            i += 1
        if depth:
            break
        inner, fixed, changed = text[m.end():i - 1], [], False
        for part in inner.split(","):
            s = part.strip()
            if s and s[0] != '"' and s not in ("null", "true", "false") and _BARE_ID.fullmatch(s):
                fixed.append(part.replace(s, json.dumps(s.strip("[] "))))
                changed = True
            else:
                fixed.append(part)
        out += [text[pos:m.end()], ",".join(fixed) if changed else inner, "]"]
        pos = i
    return "".join(out) + text[pos:]


def parse_prediction(text: str, allowed: Iterable[str], finish: str) -> tuple[str, dict[str, Any]]:
    """(status, parsed) of one answer. ``error``: cut at max_new_tokens
    (``truncated``), no JSON object, or not the three items -- nothing of it is
    read. ``partial``: some item is an error. ``ok``: every item is a value
    with known evidence, or null. Ids left unquoted in an evidence array are
    read as the ids they are, and flagged (``bare_ids``)."""
    flags: list[str] = []
    if finish == "length":
        return "error", {"error": "truncated", "items": None, "flags": flags}
    body = text.strip()
    if body.startswith("```"):
        flags.append("code_fence")
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
    quoted = quote_bare_ids(body)
    if quoted != body:
        flags.append("bare_ids")
        body = quoted
    try:
        obj = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        try:
            obj = json.loads(body[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            obj = None
        if obj is None:
            return "error", {"error": "not_json", "items": None, "flags": flags}
        flags.append("extra_text")
    if not isinstance(obj, dict) or not set(PREDICT_ITEMS) <= set(obj):
        return "error", {"error": "missing_items", "items": None, "flags": flags}
    known = set(allowed)
    items = {item: parse_item(item, obj[item], known, flags) for item in PREDICT_ITEMS}
    status = "partial" if any(v["error"] for v in items.values()) else "ok"
    return status, {"error": None, "items": items, "flags": sorted(set(flags))}


def run_predict(llm, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gen = llm.generate(row["messages"], PREDICT_MAX_NEW_TOKENS)
    status, parsed = parse_prediction(gen["text"], row["evidence"], gen["finish"])
    return status, {"raw": gen["text"], "finish": gen["finish"], "n_tokens": gen["n_tokens"],
                    "confidence": gen["confidence"], **parsed}


TASKS["predict"] = run_predict
TASK_DECODING["predict"] = {"method": "greedy generation, each token checked to be the argmax",
                            "max_new_tokens": PREDICT_MAX_NEW_TOKENS, "repetition_penalty": 1.0,
                            "batch_size": 1}


def pilot_slice(rows_: Iterable[dict[str, Any]], sha_of: Callable[[str], str | None]
                ) -> list[dict[str, Any]]:
    """The PILOT_SLICE M4FC train rows first by a seeded hash of their image:
    label-free, and the same slice every time."""
    from scripts.image_text import _priority

    train = [r for r in rows_ if r["split"] == "train"]
    return sorted(train, key=lambda r: _priority(SEED, sha_of(r["image_paths"][0])))[:PILOT_SLICE]


def vlm_accepts(rows_: Iterable[dict[str, Any]], dataset: str) -> dict[str, list[Person]]:
    """sha256 -> the people the gallery accepted among the VLM's own names.
    Never a name only the claim gave: a prediction never sees the claim."""
    out: dict[str, list[Person]] = defaultdict(list)
    for r in rows_:
        if (r["dataset"] == dataset and r["match"] is True
                and {"vlm_free", "vlm_forced"} & set(r["sides"])):
            out[r["sha256"]].append(Person(r["name"], r["matched_entity"], r["score"]))
    return out


def accepted_people(dataset: str) -> dict[str, list[Person]]:
    """``vlm_accepts`` of verify-people's output, refused if it ran under
    another face configuration or threshold, or without the corpus's
    image_text ledger (the VLM's names were then not read)."""
    from scripts import gallery as gl

    if not B_PEOPLE_SUMMARY.is_file():
        raise SystemExit("no verify-people output: run `verify-people` first")
    summary = json.loads(B_PEOPLE_SUMMARY.read_text(encoding="utf-8"))
    cid = gl.face_config_id(gl.make_backend(FACE_BACKEND))
    calib = gl.CALIBRATION_DIR / f"{FACE_BACKEND}.json"
    threshold = gl.certified_threshold(
        json.loads(calib.read_text(encoding="utf-8")) if calib.is_file() else None, cid)
    if (summary["face_config"], summary["threshold"]) != (cid, threshold):
        raise SystemExit("verify-people ran under another face configuration or threshold: rerun it")
    s = summary["datasets"].get(dataset)
    gaps = {k: v for k, v in ((s or {}).get("missing") or {}).items() if "image_text" in k}
    if s is None or gaps:
        raise SystemExit(f"{dataset}: verify-people has not read the VLM's names "
                         f"({json.dumps(gaps) if gaps else 'not run'}): rerun it once the image_text "
                         "ledger is here")
    return vlm_accepts(gl.read_jsonl(B_PEOPLE), dataset)


def documents_of(row: dict[str, Any]) -> tuple[Document, ...]:
    """The top PREDICT_DOCS hits of a retrieval row as a view holds them,
    leaving out a hit whose title and excerpt repeat a higher-ranked one's:
    the store holds one story under several doc ids (380 of M4FC's 3,021
    claims had a repeat in their top 5, 7 one story five times), and a repeat
    adds weight to a source, not evidence. Lower ranks fill its place."""
    seen: set[tuple[str, str]] = set()
    out: list[Document] = []
    for h in row["hits"]:
        key = (" ".join(h["title"].split()).casefold(), " ".join((h["excerpt"] or "").split()).casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(Document(h["doc_id"], h["title"], h["published_at"], h["host"], h["excerpt"] or ""))
        if len(out) == PREDICT_DOCS:
            break
    return tuple(out)


def predict_rows(dataset: str, rows_: Iterable[dict[str, Any]], variant: str, *,
                 sha_of: Callable[[str], str | None], image_rows: dict[str, dict[str, Any]],
                 people: dict[str, Sequence[Person]],
                 documents: dict[str, tuple[Document, ...]] | None = None
                 ) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    """One predictor prompt per image of ``rows_``, id-checked against every row
    of that image, and refused if it holds a claim its content does not.
    An image with no evidence at all gets no prompt, and is counted."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        by_sha[sha_of(row["image_paths"][0])].append(row)
    out, keys, tally = [], {}, Counter()
    for sha in sorted(by_sha):
        group = by_sha[sha]
        if sha not in image_rows:
            raise SystemExit(f"{dataset}: an image has no image_text row; the ledger is incomplete")
        view = view_of(group[0], sha_of=sha_of, image_rows=image_rows, people=people.get(sha, ()),
                       documents=(documents or {}).get(sha, ()))
        msgs, evidence = predict_messages(view)
        if not evidence:
            tally["no_evidence"] += 1
            continue
        check_predict_prompt(msgs, view, group)
        name = f"{dataset}:{sha}:{variant}"
        key = neutral_key("b", name)
        keys[key] = name
        out.append({"key": key, "task": "predict", "messages": msgs,
                    "content": predict_content(view), "evidence": evidence})
        tally[f"{dataset}:{variant}"] += 1
    return out, keys, tally


def check_predict_prompt(msgs: list[dict[str, Any]], view: ModelView,
                         group: Sequence[dict[str, Any]]) -> None:
    """Refuse a predictor prompt holding an id form of any row of its image,
    or any of their claims, more often than its content does: an image's own
    text may quote a claim, and an excerpt may too, but the prompt never adds one."""
    text, content = messages_text(msgs), predict_content(view)
    refuse_ids(text, frozenset().union(*(id_forms(r) for r in group)),
               explained=f"{content}\n{PREDICT_SYSTEM}\n{PREDICT_TEMPLATE}\n{messages_text(msgs[1:3])}",
               what="a predictor prompt")
    flat, source = " ".join(text.split()), " ".join(content.split())
    for r in group:
        claim = " ".join((r.get("text") or "").split())
        if claim and flat.count(claim) > source.count(claim):
            raise GuardError("a predictor prompt holds its claim")


def prompt_tokens(rows_: Sequence[dict[str, Any]]) -> list[int] | None:
    """Each prompt's length in the model's tokens, or None where this machine
    has no copy of its tokenizer."""
    try:
        from scripts.image_text import QwenVLM

        vlm = QwenVLM()
        vlm.prepare()
    except Exception as exc:  # noqa: BLE001 -- a length report, not a step
        print(f"  (no token counts: {type(exc).__name__})")
        return None
    return sorted(len(vlm.processor.tokenizer(vlm.processor.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=True))["input_ids"]) for r in rows_)


def cmd_prompts_predict_b(args) -> int:
    """Stage B.3's input: the pilot slice, or every prediction the stage
    scores. Refused unless every input it reads is complete -- the image_text
    ledgers, verify-people on the VLM's names, a retrieval ledger that passes
    its check -- because an incomplete input would read as missing evidence."""
    m4fc_images = image_text_rows("m4fc")
    if m4fc_images is None:
        raise SystemExit("M4FC's image_text ledger is not here yet: fetch it from the server, "
                         "rerun `verify-people`, then build the prompts")
    sha_of = sha_lookup()
    m4fc_rows = list(rows("m4fc", M4FC_SPLITS))
    _cid, items, held = retrieval_ledger(M4FC_SPLITS, sha_of)
    check = check_retrieval(items, held)
    if check["problems"]:
        raise SystemExit(f"the retrieval ledger fails its check: {json.dumps(check['problems'])}")
    docs = {s: documents_of(r) for s, r in held.items()}
    built: list[dict[str, Any]] = []
    keys: dict[str, str] = {}
    tally: Counter = Counter()
    m4fc_people = accepted_people("m4fc")
    parts = ([("m4fc", pilot_slice(m4fc_rows, sha_of), "retrieval")] if args.part == "pilot" else
             [("m4fc", m4fc_rows, "retrieval"),
              ("m4fc", [r for r in m4fc_rows if r["split"] == "val"], "no_retrieval"),
              ("verite", evaluation_rows(), "image_only")])
    for ds, rows_, variant in parts:
        out, k, t = predict_rows(
            ds, rows_, variant, sha_of=sha_of,
            image_rows=m4fc_images if ds == "m4fc" else image_text_rows(ds) or {},
            people=m4fc_people if ds == "m4fc" else accepted_people(ds),
            documents=docs if variant == "retrieval" else None)
        built += out
        keys.update(k)
        tally.update(t)
    path = B_PREDICT_PILOT if args.part == "pilot" else B_PREDICT
    meta = write_prompts(path, "predict", built, predict_prompt_config(), keys)
    print(f"{meta['rows']:,} predictor prompts ({json.dumps(ordered(tally))}), every one id-checked "
          f"-> {rel(path)} (prompt config {meta['prompt_config_sha256'][:12]})")
    lengths = prompt_tokens(built)
    if lengths:
        print(f"  prompt tokens: min {lengths[0]:,}, median {lengths[len(lengths) // 2]:,}, "
              f"max {lengths[-1]:,}")
    return 0


# --------------------------------------------------------------------------
# stage B.4: the predictor scored against M4FC's annotated context
# --------------------------------------------------------------------------

B_JUDGE = B_DIR / "judge_in.jsonl"
REPORT_B = REPORTS / "consistency_b.json"
REPORT_B_MD = REPORTS / "consistency_b.md"
#: The items the judge compares in B. A date is the rule's (ROUTES); people are
#: the gallery's, scored by QID.
JUDGED_ITEMS = ("location", "event")
#: What B scores, per split: the predictor with its claim's retrieval; on val
#: again without it (the echo rate's contrast); and the floor, the image
#: model's own free DATE, GPE and EVENT lines.
B_VARIANTS = {"train": ("retrieval", "floor"), "val": ("retrieval", "no_retrieval", "floor")}
FLOOR_LINES = {"date": "DATE", "location": "GPE", "event": "EVENT"}
OOC = "false (out-of-context)"
CLASS_NAMES = {"true": "true", OOC: "out-of-context", "false (manipulated/fake)": "manipulated",
               "false (ai-generated)": "ai-generated", "mixture": "mixture", "unverified": "unverified"}
#: Words that say a document is about a false claim: counted, never filtered.
DEBUNK = re.compile(r"(?<!\w)(fake|falsely|misleading|hoax|debunk\w*|fact[- ]?check\w*|morphed|"
                    r"doctored|photoshopped|old (?:photo|image|video|picture)s?|"
                    r"viral (?:photo|image|video|claim)s?)(?!\w)", re.I)


class _Refused:
    """A prediction the predictor gave and the parser refused: not a null."""

    def __repr__(self) -> str:
        return "ERROR"


ERROR = _Refused()


def predicted(result: dict[str, Any] | None, built: bool = True) -> dict[str, Any]:
    """item -> the predicted value; None where the answer is null, or where no
    prompt was built (an image with no evidence at all); ERROR where the
    answer, or that item of it, was refused."""
    if not built:
        return dict.fromkeys(PREDICT_ITEMS)
    if result is None or result["status"] == "error":
        return dict.fromkeys(PREDICT_ITEMS, ERROR)
    items = result["result"]["items"]
    return {i: ERROR if items[i]["error"] else items[i]["value"] for i in PREDICT_ITEMS}


def floor_values(image_row: dict[str, Any] | None) -> dict[str, Any]:
    """item -> the image model's own free DATE, GPE or EVENT items, joined
    with '; '; None where it named none; ERROR where its entity answer is not
    usable (failed, cut, no free line). Forced lines are left out."""
    from scripts.image_text import entities_of, entity_answer_error

    ents = (image_row or {}).get("entities")
    if (not ents or image_row.get("status") not in ("ok", "partial") or ents.get("error")
            or entity_answer_error(ents)):
        return dict.fromkeys(FLOOR_LINES, ERROR)
    by: dict[str, list[str]] = defaultdict(list)
    for e in entities_of(image_row, forced="exclude"):
        by[e["category"]].append(e["name"].strip())
    return {item: ("; ".join(by[c]) if by[c] else None) for item, c in FLOOR_LINES.items()}


_EVIDENCE_LINE = re.compile(r"^\[([A-Z]+\d*)\] (.*)$")
_DOC_HEADER = re.compile(r"^A document published \d{4}-\d{2}-\d{2} on [^:]*: ")


def evidence_lines(messages: Sequence[dict[str, Any]]) -> dict[str, str]:
    """id -> the text of its evidence line in a predictor prompt (the image's
    turn, never the worked example's), a document's publication date left
    out: a year read from it is the echo the prompt warns against."""
    text = messages[-1]["content"][0]["text"].split("\n\nFrom this evidence alone")[0]
    out: dict[str, str] = {}
    cur = None
    for line in text.splitlines():
        m = _EVIDENCE_LINE.match(line)
        if m:
            cur = m.group(1)
            out[cur] = _DOC_HEADER.sub("", m.group(2))
        elif cur and line.startswith("     "):
            out[cur] += " " + line.strip()
    return out


def date_grounded(value: str, cited: Sequence[str], lines: dict[str, str]) -> bool:
    """Whether a predicted date's year is written in the evidence it cites.
    The third pilot's dates showed why: nine of eighteen took a day and a
    month ("September 12") and added a year no evidence gives."""
    return any(re.search(rf"(?<!\d){re.escape(value[:4])}(?!\d)", lines.get(e, "")) for e in cited)


def b_results(inp: Path | None = None) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """(the predictions a prompt was built for, "dataset:sha256:variant" ->
    the predictor's output row). Refused if the run is incomplete."""
    inp = inp or B_PREDICT
    keys = json.loads(inp.with_name(inp.stem + ".keys.json").read_text(encoding="utf-8"))
    results = read_results(inp)
    missing = [k for k in keys if k not in results]
    if missing:
        raise SystemExit(f"{len(missing):,} of {len(keys):,} predictions are not in {output_path(inp).name}: "
                         "fetch the complete run")
    return set(keys.values()), {keys[k]: r for k, r in results.items()}


def b4_rows(built: set[str], results: dict[str, dict[str, Any]],
            m4fc_rows: Iterable[dict[str, Any]], sha_of: Callable[[str], str | None],
            images: dict[str, dict[str, Any]],
            lines: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """One record per M4FC train/val row: its class, the claimed and actual
    value of each item (as the sheet shows them; None where absent), each
    variant's predictions, and whether each predicted date's year is written
    in the evidence it cites (``lines``: prediction -> its evidence lines)."""
    from scripts.m4fc_audit import present, shown

    out = []
    for row in m4fc_rows:
        sha, meta = sha_of(row["image_paths"][0]), row["meta"]
        preds, grounded = {}, {}
        for v in B_VARIANTS[row["split"]]:
            if v == "floor":
                preds[v] = floor_values(images.get(sha))
                continue
            name = f"m4fc:{sha}:{v}"
            preds[v] = predicted(results.get(name), name in built)
            if isinstance(preds[v]["date"], str):
                cited = results[name]["result"]["items"]["date"]["evidence"]
                grounded[v] = date_grounded(preds[v]["date"], cited, (lines or {}).get(name, {}))
        out.append({"sha256": sha, "split": row["split"], "class": meta["verdict"], "row": row,
                    "claimed": {i: shown(meta[f"claimed_{i}"]) if present(meta.get(f"claimed_{i}"))
                                else None for i in ITEMS},
                    "actual": {i: shown(meta[i]) if present(meta.get(i)) else None for i in ITEMS},
                    "preds": preds, "date_grounded": grounded})
    return out


def grounded_only(recs: Sequence[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    """The records with a predicted date whose year its evidence does not
    write read as having no date: the strict twin of the date figures."""
    out = []
    for r in recs:
        if isinstance(r["preds"][variant]["date"], str) and not r["date_grounded"].get(variant):
            r = {**r, "preds": {**r["preds"], variant: {**r["preds"][variant], "date": None}}}
        out.append(r)
    return out


def judge_pairs(b4: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """(item, statement A, statement B) -> the records it decides: every pair
    B.4 needs the judge for, each asked once. The judge's frozen template
    (stage A's) puts a claim in A and what is known about the photograph in B:
    a claim against the actual value (does the claim contradict the image?),
    a prediction against the actual value (is it right?), and on
    out-of-context rows the claim against a prediction (does it echo the claim?)."""
    pairs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in b4:
        for item in JUDGED_ITEMS:
            c, a = r["claimed"][item], r["actual"][item]
            if c and a:
                pairs[(item, c, a)].append(r)
            for p in r["preds"].values():
                value = p[item]
                if not isinstance(value, str):
                    continue
                if a:
                    pairs[(item, value, a)].append(r)
                if c and r["class"] == OOC:
                    pairs[(item, c, value)].append(r)
    return pairs


def _b4_inputs() -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, Any]]]:
    images = image_text_rows("m4fc")
    if images is None:
        raise SystemExit("M4FC's image_text ledger is not here")
    built, results = b_results()
    keys = json.loads(B_PREDICT.with_name(B_PREDICT.stem + ".keys.json").read_text(encoding="utf-8"))
    lines = {}
    with B_PREDICT.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if keys[r["key"]].startswith("m4fc:"):
                lines[keys[r["key"]]] = evidence_lines(r["messages"])
    return (b4_rows(built, results, rows("m4fc", M4FC_SPLITS), sha_lookup(), images, lines),
            built, results)


def cmd_prompts_judge_b() -> int:
    """The scoring judge's input: every location and event pair B.4 needs,
    once each, id-checked against every record it decides."""
    b4, _built, _results = _b4_inputs()
    shots = fewshot()["judge"]
    out, keys = [], {}
    for (item, a, b), recs in sorted(judge_pairs(b4).items()):
        msgs = judge_messages(item, a, b, shots)
        content = f"{a}\n{b}"
        refuse_ids(messages_text(msgs), frozenset().union(*(id_forms(r["row"]) for r in recs)),
                   explained=content + "\n" + messages_text(msgs[:-1]), what="a scoring-judge prompt")
        key = neutral_key("bj", f"{item}\n{a}\n{b}")
        keys[key] = {"item": item, "a": a, "b": b}
        out.append({"key": key, "task": "judge", "messages": msgs, "content": content})
    meta = write_prompts(B_JUDGE, "judge", out, judge_prompt_config(shots), keys)
    print(f"{meta['rows']:,} scoring-judge prompts ({json.dumps(ordered(Counter(k['item'] for k in keys.values())))}), "
          f"every one id-checked -> {rel(B_JUDGE)} (prompt config {meta['prompt_config_sha256'][:12]}, "
          "stage A's)")
    return 0


def contains(value: str, text: str, item: str) -> bool:
    """Whether case-folded ``text`` holds a value: a date by any of its years,
    anything else by any of its parts of three characters or more."""
    if item == "date":
        years = {y for p in stated_parts(value) for y in re.findall(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", p)}
        return any(re.search(rf"(?<!\d){y}(?!\d)", text) for y in years)
    return any(len(p) >= 3 and p.casefold() in text for p in stated_parts(value))


_OCR_LINE = "Text read in the photograph: "
#: An entity line the image model says is written in the photograph, as
#: predict_evidence writes it: its name is the group.
_TEXT_ENTITY = re.compile(
    r"^The image model names (?:%s): (.+) \(%s(?:, confidence [0-9.]+)?(?:; weaker)?\)$"
    % ("|".join(map(re.escape, ENTITY_WORDS.values())), re.escape(BASIS_WORDS["text"])))


#: A place line the image model says it recognised by how it looks.
_SEEN_PLACE = re.compile(
    r"^The image model names %s: (.+) \(%s(?:, confidence [0-9.]+)?(?:; weaker)?\)$"
    % (re.escape(ENTITY_WORDS["GPE"]), re.escape(BASIS_WORDS["seen"])))
#: image_carries_claim's two forms. "narrow", as fixed before C ran: the OCR
#: line and entity lines written in the photograph. "widened" at Gate C (the
#: person's decision, 2026-11-10, a rule defect found on train): for a place,
#: also a place line recognised by how it looks whose name is written in the
#: OCR text -- the text it rests on is then that OCR text.
CARRY_RULES = ("narrow", "widened")
CARRY_RULE = "widened"


def ocr_text(lines: dict[str, str]) -> str:
    """The OCR text of a prompt's evidence lines, or ""."""
    line = lines.get("OCR", "")
    return line[len(_OCR_LINE):] if line.startswith(_OCR_LINE) else ""


def text_evidence(cited: Sequence[str], lines: dict[str, str], item: str | None = None,
                  rule: str = "narrow") -> str | None:
    """The text a predicted value rests on, when it rests on text read in the
    image alone: the OCR line, or entity lines the image model says are written
    in the photograph; under the widened rule, for a place, also a place line
    recognised by how it looks whose name the OCR text holds (its text is the
    OCR's). None when it cites nothing, or anything else as well (a
    description, a face, a document, any other line recognised by how it looks)."""
    if rule not in CARRY_RULES:
        raise ValueError(f"unknown image_carries_claim rule {rule!r}")
    ocr, parts = ocr_text(lines), []
    for e in cited:
        line = lines.get(e, "")
        if e == "OCR" and ocr:
            parts.append(ocr)
        elif re.fullmatch(r"E\d+", e) and (m := _TEXT_ENTITY.match(line)):
            parts.append(m.group(1))
        elif (rule == "widened" and item == "location" and re.fullmatch(r"E\d+", e)
              and (m := _SEEN_PLACE.match(line)) and ocr and contains(m.group(1), ocr.casefold(), "location")):
            parts.append(ocr)
        else:
            return None
    return "\n".join(parts) if parts else None


def image_carries_claim(item: str, claimed: str | None, cited: Sequence[str], lines: dict[str, str],
                        rule: str = CARRY_RULE) -> bool:
    """Whether a predicted value rests only on text in the image (OCR, or an
    entity the image model read as written there; widened: a place it says it
    recognised whose name the OCR holds) that holds the claimed value, by
    ``contains``: the image carries the claim -- a screenshot of it -- and is
    no independent evidence. The person's rule (2026-11-10), fixed before C
    ran and widened at Gate C; each form measured on M4FC train and applied
    unchanged to val and VERITE."""
    text = text_evidence(cited, lines, item, rule)
    return bool(text) and isinstance(claimed, str) and contains(claimed, text.casefold(), item)


def accuracy_cell(recs: Sequence[dict[str, Any]], variant: str, item: str,
                  decide: Callable[[str, str, str], str]) -> dict[str, Any]:
    """Against the actual value: how many records a variant predicts the item
    for, and of those the share the comparator calls consistent (right),
    contradicted (wrong) or unclear (undecided)."""
    with_actual = [r for r in recs if r["actual"][item]]
    valued = [r for r in with_actual if isinstance(r["preds"][variant][item], str)]
    labels = Counter(decide(item, r["preds"][variant][item], r["actual"][item]) for r in valued)
    k = len(valued)
    return {"rows_with_actual": len(with_actual),
            "refused": sum(1 for r in with_actual if r["preds"][variant][item] is ERROR),
            "predicted": rate(k, len(with_actual)), "right": rate(labels[CONSISTENT], k),
            "wrong": rate(labels[CONTRADICTS], k), "undecided": rate(labels[UNCLEAR], k)}


def echo_cell(recs: Sequence[dict[str, Any]], variant: str, item: str,
              decide: Callable[[str, str, str], str]) -> dict[str, Any]:
    """On out-of-context records whose claim contradicts the actual value: the
    share of predictions that agree with the claim and contradict the actual
    value -- the predictor repeating the claim-keyed evidence."""
    base = [r for r in recs if r["class"] == OOC and r["claimed"][item] and r["actual"][item]
            and decide(item, r["claimed"][item], r["actual"][item]) == CONTRADICTS]
    valued = [r for r in base if isinstance(r["preds"][variant][item], str)]
    agrees = {id(r) for r in valued if decide(item, r["claimed"][item], r["preds"][variant][item]) == CONSISTENT}
    against = {id(r) for r in valued if decide(item, r["preds"][variant][item], r["actual"][item]) == CONTRADICTS}
    return {"rows": len(base), "predicted": rate(len(valued), len(base)),
            "echo": rate(len(agrees & against), len(valued)),
            "agrees_with_claim": rate(len(agrees), len(valued)),
            "contradicts_actual": rate(len(against), len(valued))}


def people_cell(recs: Sequence[dict[str, Any]], accepts: dict[str, Sequence[Person]],
                resolve: Callable[[str], Sequence[str]]) -> dict[str, Any]:
    """The gallery's accepts of the VLM's names against M4FC's people, by QID:
    right where an accepted person is one of them; wrong where every M4FC name
    resolves to one person and none is accepted; otherwise undecided."""
    with_actual = [r for r in recs if r["actual"]["people"]]
    tally: Counter = Counter()
    for r in with_actual:
        accepted = {p.qid for p in accepts.get(r["sha256"], ())}
        if not accepted:
            continue
        resolved = [list(resolve(n)) for n in stated_parts(r["actual"]["people"])]
        single = {q[0] for q in resolved if len(q) == 1}
        tally["right" if accepted & single else
              "wrong" if resolved and all(len(q) == 1 for q in resolved) else "undecided"] += 1
    covered = sum(tally.values())
    return {"rows_with_actual": len(with_actual), "covered": rate(covered, len(with_actual)),
            "right": rate(tally["right"], covered), "wrong": rate(tally["wrong"], covered),
            "undecided": rate(tally["undecided"], covered),
            "precision": rate(tally["right"], tally["right"] + tally["wrong"])}


def statuses(results: dict[str, dict[str, Any]], dataset: str, variant: str) -> dict[str, Any]:
    picked = [r for name, r in results.items() if name.startswith(f"{dataset}:") and name.endswith(f":{variant}")]
    items = Counter(f"{i}:{v['error']}" for r in picked for i, v in (r["result"]["items"] or {}).items()
                    if v["error"])
    return {"rows": len(picked), "status": ordered(Counter(r["status"] for r in picked)),
            "row_errors": ordered(Counter(r["result"]["error"] for r in picked if r["result"]["error"])),
            "item_errors": ordered(items),
            "flags": ordered(Counter(f for r in picked for f in r["result"]["flags"]))}


def stage_a_figures() -> dict[str, Any]:
    """The fixed comparator's own error, as stage A measured it, stated beside
    B: as built, and after the person's review of the contested true rows
    where there is one (None where not)."""
    rep = json.loads(REPORT_A.read_text(encoding="utf-8"))
    review = rep.get("review") or {}

    def figures(block: dict[str, Any], item: str, comp: str) -> dict[str, Any]:
        m = block["items"][item].get(comp) or {}
        s, t = m.get("sheet", {}), m.get("true_row", {})
        return {"sheet_precision_strict": s.get("contradicts_precision"), "sheet_recall": s.get("contradicts_recall"),
                "true_row_false_contradiction": t.get("false_contradiction"),
                "gate": (block["gate"].get(item) or {}).get("passed")}

    out: dict[str, Any] = {item: {"comparator": comp, "as_built": figures(rep, item, comp),
                                  "after_review": figures(review, item, comp) if review.get("decided") else None}
                           for item, comp in FIXED.items()}
    out["review"] = {k: review.get(k) for k in ("contested", "decided", "relabelled")}
    return out


def cmd_score_b() -> int:
    from scripts import gallery as gl

    b4, built, results = _b4_inputs()
    judged: dict[tuple[str, str, str], dict[str, Any]] = {}
    if B_JUDGE.is_file():
        keys = json.loads(B_JUDGE.with_name(B_JUDGE.stem + ".keys.json").read_text(encoding="utf-8"))
        done = read_results(B_JUDGE)  # once: the output holds a row per pair
        judged = {(v["item"], v["a"], v["b"]): done[k] for k, v in keys.items() if k in done}
    need = judge_pairs(b4)
    if missing := [p for p in need if p not in judged]:
        raise SystemExit(f"the scoring judge has not decided {len(missing):,} of {len(need):,} pairs: "
                         "run it, and fetch its complete output")

    def decide(item: str, a: str, b: str) -> str:
        return date_rule(a, b)["label"] if item == "date" else judged[(item, a, b)]["result"]["label"]

    resolve = resolver() or (lambda name: [])
    accepts = vlm_accepts(gl.read_jsonl(B_PEOPLE), "m4fc")
    _cid, _items, held = retrieval_ledger(M4FC_SPLITS)
    texts: dict[str, str] = {}

    def doc_text(sha: str) -> str:
        if sha not in texts:
            parts = []
            for d in documents_of(held[sha]):
                doc = read_doc(d.doc_id) or {}
                parts.append(f"{doc.get('title') or ''}\n{doc.get('text') or ''}")
            texts[sha] = "\n".join(parts).casefold()
        return texts[sha]

    report: dict[str, Any] = {
        "written_at": utcnow(),
        "predictor": {"config": json.loads(output_path(B_PREDICT).with_name(
            output_path(B_PREDICT).stem + ".config.json").read_text(encoding="utf-8"))["config_id"],
                      "statuses": {f"{ds}:{v}": statuses(results, ds, v) for ds, v in
                                   (("m4fc", "retrieval"), ("m4fc", "no_retrieval"), ("verite", "image_only"))},
                      "no_prompt_no_evidence": len({f"m4fc:{r['sha256']}:{v}" for r in b4 for v in r["preds"]
                                                    if v != "floor"} - built)},
        "judge_pairs": len(need), "judge_statuses": ordered(Counter(judged[p]["status"] for p in need)),
        "comparator_from_stage_a": stage_a_figures(),
        "accuracy": {}, "echo": {}, "date_grounding": {}, "people": {}, "evidence_recall_at_5": {},
        "debunk_words": {}}
    for split, variants in B_VARIANTS.items():
        recs = [r for r in b4 if r["split"] == split]
        classes = sorted({r["class"] for r in recs}, key=lambda c: list(CLASS_NAMES).index(c))
        report["accuracy"][split] = {v: {item: {"all": accuracy_cell(recs, v, item, decide), "by_class": {
            CLASS_NAMES[c]: accuracy_cell([r for r in recs if r["class"] == c], v, item, decide)
            for c in classes}} for item in PREDICT_ITEMS} for v in variants}
        report["echo"][split] = {v: {item: echo_cell(recs, v, item, decide) for item in PREDICT_ITEMS}
                                 for v in variants}
        report["date_grounding"][split] = {}
        for v in variants:
            if v == "floor":
                continue
            dated = [r for r in recs if isinstance(r["preds"][v]["date"], str)]
            strict = grounded_only(recs, v)
            report["date_grounding"][split][v] = {
                "dates": len(dated), "grounded": rate(sum(bool(r["date_grounded"].get(v)) for r in dated),
                                                      len(dated)),
                "accuracy_grounded_only": accuracy_cell(strict, v, "date", decide),
                "echo_grounded_only": echo_cell(strict, v, "date", decide)}
        report["people"][split] = {"all": people_cell(recs, accepts, resolve), "by_class": {
            CLASS_NAMES[c]: people_cell([r for r in recs if r["class"] == c], accepts, resolve)
            for c in classes}}
        report["evidence_recall_at_5"][split] = {}
        for item in ITEMS:
            paired = [r for r in recs if r["actual"][item] and r["claimed"][item]]
            report["evidence_recall_at_5"][split][item] = {
                "rows": len(paired),
                "actual_value": rate(sum(contains(r["actual"][item], doc_text(r["sha256"]), item)
                                         for r in paired), len(paired)),
                "claimed_value": rate(sum(contains(r["claimed"][item], doc_text(r["sha256"]), item)
                                          for r in paired), len(paired))}
        report["debunk_words"][split] = {CLASS_NAMES[c]: rate(
            sum(bool(DEBUNK.search(doc_text(r["sha256"]))) for r in recs if r["class"] == c),
            sum(1 for r in recs if r["class"] == c)) for c in classes}
    public_report(report, REPORT_B)
    REPORT_B_MD.write_text(render_b(report), encoding="utf-8", newline="\n")
    print(render_b(report))
    return 0


def render_b(report: dict[str, Any]) -> str:
    lines = ["# Consistency block, stage B: the context predictor against M4FC's context", "",
             f"Written {report['written_at']}. Predictor config {report['predictor']['config']}; "
             f"{report['judge_pairs']:,} judge pairs ({json.dumps(report['judge_statuses'])}).", "",
             "Right = the fixed comparator calls the prediction consistent with M4FC's actual value "
             "(the same, or one more specific); wrong = contradicted; undecided = it cannot tell. "
             "Dates by the rule alone, location and event by the judge, people by gallery QID. "
             "Floor = the image model's own free DATE, GPE and EVENT lines, scored the same way.", "",
             "## The comparator's own error (stage A)", ""]
    review = report["comparator_from_stage_a"]["review"]
    lines.append(f"The person's review of contested true-row pairs: {review['decided'] or 0} of "
                 f"{review['contested']} decided; relabelled {json.dumps(review['relabelled'] or {})}.")
    for item, f in report["comparator_from_stage_a"].items():
        if item == "review":
            continue
        for name in ("as_built", "after_review"):
            g = f[name]
            if g is None:
                continue
            lines.append(f"- {item} ({f['comparator']}), {name.replace('_', ' ')}: sheet precision "
                         f"{_cell(g['sheet_precision_strict'])}, recall {_cell(g['sheet_recall'])}, true-row false "
                         f"contradiction {_cell(g['true_row_false_contradiction'])}; Gate A "
                         f"{'passed' if g['gate'] else 'not passed'}")
    lines += ["", "## Predictor statuses", ""]
    for name, s in report["predictor"]["statuses"].items():
        lines.append(f"- {name}: {s['rows']:,} rows, {json.dumps(s['status'])}; item errors "
                     f"{json.dumps(s['item_errors'])}; flags {json.dumps(s['flags'])}")
    for split in report["accuracy"]:
        lines += ["", f"## {split}: against the actual value", "",
                  "| item | variant | rows with actual | predicted | right | wrong | undecided |",
                  "|---|---|---|---|---|---|---|"]
        for v, items in report["accuracy"][split].items():
            for item, cell in items.items():
                a = cell["all"]
                lines.append(f"| {item} | {v} | {a['rows_with_actual']:,} | {_cell(a['predicted'])} | "
                             f"{_cell(a['right'])} | {_cell(a['wrong'])} | {_cell(a['undecided'])} |")
        lines += ["", f"### {split}: by class (retrieval)", "",
                  "| item | class | rows with actual | predicted | right | wrong |", "|---|---|---|---|---|---|"]
        for item, cell in report["accuracy"][split]["retrieval"].items():
            for c, a in cell["by_class"].items():
                lines.append(f"| {item} | {c} | {a['rows_with_actual']:,} | {_cell(a['predicted'])} | "
                             f"{_cell(a['right'])} | {_cell(a['wrong'])} |")
        lines += ["", f"### {split}: echo (out-of-context rows whose claim contradicts the actual value)", "",
                  "| item | variant | rows | predicted | echo | agrees with claim | contradicts actual |",
                  "|---|---|---|---|---|---|---|"]
        for v, items in report["echo"][split].items():
            for item, e in items.items():
                lines.append(f"| {item} | {v} | {e['rows']:,} | {_cell(e['predicted'])} | {_cell(e['echo'])} | "
                             f"{_cell(e['agrees_with_claim'])} | {_cell(e['contradicts_actual'])} |")
        lines += ["", f"### {split}: dates whose year their cited evidence writes (the strict twin)", ""]
        for v, g in report["date_grounding"][split].items():
            a = g["accuracy_grounded_only"]
            lines.append(f"- {v}: {g['dates']:,} dates predicted, grounded {_cell(g['grounded'])}. Grounded "
                         f"only: predicted {_cell(a['predicted'])}, right {_cell(a['right'])}, wrong "
                         f"{_cell(a['wrong'])}; echo {_cell(g['echo_grounded_only']['echo'])}")
        p = report["people"][split]["all"]
        lines += ["", f"### {split}: people (the gallery's accepts of the VLM's names)", "",
                  f"Rows with M4FC people {p['rows_with_actual']:,}; covered {_cell(p['covered'])}; "
                  f"precision {_cell(p['precision'])}; undecided {_cell(p['undecided'])}.", "",
                  f"### {split}: evidence recall@5 (rows with both values)", "",
                  "| item | rows | actual value in the top 5 | claimed value in the top 5 |", "|---|---|---|---|"]
        for item, e in report["evidence_recall_at_5"][split].items():
            lines.append(f"| {item} | {e['rows']:,} | {_cell(e['actual_value'])} | {_cell(e['claimed_value'])} |")
        lines += ["", f"Debunk words in the top 5, by class: " + "; ".join(
            f"{c} {_cell(r)}" for c, r in report["debunk_words"][split].items()), ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# stage C.1: what the claim says of its photograph -- read from its text alone
# --------------------------------------------------------------------------

C_DIR = OUT / "c"
C_EXTRACT_PILOT = C_DIR / "extract_pilot_in.jsonl"
C_EXTRACT = C_DIR / "extract_in.jsonl"
EXTRACT_MAX_NEW_TOKENS = 120
EXTRACT_SYSTEM = ("You read what a news caption says about its photograph. "
                  "Answer with one JSON object and nothing else.")
EXTRACT_TEMPLATE = (
    "Caption: {claim}\n"
    "\n"
    "What does this caption say about the photograph: when it was taken, where, and what event it "
    "shows?\n"
    "- date: as YYYY, YYYY-MM or YYYY-MM-DD, no more precise than the caption; a date the caption "
    "does not write is null.\n"
    "- location: the place, in the caption's own words.\n"
    "- event: what the caption says is happening, in a few words.\n"
    "Write null for anything the caption does not say. Add nothing it does not say.\n"
    "Answer with this JSON object and nothing else:\n"
    '{{"date": ..., "location": ..., "event": ...}}')


def extract_messages(claim: str, shots: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The extractor's chat: every exemplar as a prior turn, then the claim.
    It reads the claim's text and nothing else -- no image, no id."""
    shots = fewshot()["extract"] if shots is None else shots
    msgs: list[dict[str, Any]] = [{"role": "system", "content": _text(EXTRACT_SYSTEM)}]
    for s in shots:
        msgs.append({"role": "user", "content": _text(EXTRACT_TEMPLATE.format(claim=s["claim"]))})
        msgs.append({"role": "assistant", "content": _text(json.dumps(s["answer"], ensure_ascii=False))})
    msgs.append({"role": "user", "content": _text(EXTRACT_TEMPLATE.format(claim=" ".join(claim.split())))})
    return msgs


def extract_prompt_config() -> dict[str, Any]:
    return {"system": EXTRACT_SYSTEM, "template": EXTRACT_TEMPLATE, "shots": fewshot()["extract"],
            "items": list(PREDICT_ITEMS), "null_words": sorted(NULL_WORDS),
            "max_new_tokens": EXTRACT_MAX_NEW_TOKENS,
            "date_rule": "YYYY, YYYY-MM or YYYY-MM-DD, its year written in the claim"}


#: How an extraction's answer is read, beyond the prompt its run's config
#: holds. 2 (the person's decision, 2026-11-10): a decade or a span is a date,
#: read as the interval the date rule gives it, when the claim writes its
#: years. C.1's answers are read again by it; nothing was rerun.
EXTRACT_PARSER = 2


def extracted_date_error(value: str, claim: str) -> str | None:
    """Why an extracted date is refused, or None. It must be YYYY, YYYY-MM or
    YYYY-MM-DD, a decade ("1980s") or a span ("1951-1967"), each a form the
    date rule reads, and every year in it must be written in the claim: the
    claim is the only evidence, so a year it does not write was added."""
    if not (any(p.fullmatch(value) for p in (_YMD, _SPAN, _DECADE)) and date_interval(value)):
        return "bad_date"
    if not all(re.search(rf"(?<!\d){y}(?!\d)", claim) for y in re.findall(r"\d{4}", value)):
        return "date_not_in_claim"
    return None


def parse_extraction(text: str, claim: str, finish: str) -> tuple[str, dict[str, Any]]:
    """(status, parsed) of one extraction, read as strictly as a prediction:
    cut, not JSON or not the three items is an ``error``; an item that is no
    string, or a date the date rule cannot read or with a year the claim does
    not write (``extracted_date_error``), is that item's error (``partial``)."""
    flags: list[str] = []
    if finish == "length":
        return "error", {"error": "truncated", "items": None, "flags": flags}
    body = text.strip()
    if body.startswith("```"):
        flags.append("code_fence")
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body).strip()
    try:
        obj = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        try:
            obj = json.loads(body[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            obj = None
        if obj is None:
            return "error", {"error": "not_json", "items": None, "flags": flags}
        flags.append("extra_text")
    if not isinstance(obj, dict) or not set(PREDICT_ITEMS) <= set(obj):
        return "error", {"error": "missing_items", "items": None, "flags": flags}
    items = {}
    for item in PREDICT_ITEMS:
        value = obj[item]
        if item == "date" and isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
            flags.append("number_date")
        if isinstance(value, str) and value.strip().casefold() in NULL_WORDS:
            if value.strip():
                flags.append("null_word")
            value = None
        if value is not None and not isinstance(value, str):
            items[item] = {"value": None, "error": "bad_value"}
        elif value is None:
            items[item] = {"value": None, "error": None}
        else:
            value = " ".join(value.split())
            error = extracted_date_error(value, claim) if item == "date" else None
            items[item] = {"value": value, "error": error}
    status = "partial" if any(v["error"] for v in items.values()) else "ok"
    return status, {"error": None, "items": items, "flags": sorted(set(flags))}


def run_extract(llm, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gen = llm.generate(row["messages"], EXTRACT_MAX_NEW_TOKENS)
    status, parsed = parse_extraction(gen["text"], row["content"], gen["finish"])
    return status, {"raw": gen["text"], "finish": gen["finish"], "n_tokens": gen["n_tokens"],
                    "confidence": gen["confidence"], **parsed}


TASKS["extract"] = run_extract
TASK_DECODING["extract"] = {"method": "greedy generation, each token checked to be the argmax",
                            "max_new_tokens": EXTRACT_MAX_NEW_TOKENS, "repetition_penalty": 1.0,
                            "batch_size": 1}


def claim_key(claim: str) -> str:
    """A claim's identity: the sha256 of its text, whitespace collapsed."""
    return hashlib.sha256(" ".join((claim or "").split()).encode("utf-8")).hexdigest()


def extract_rows(dataset: str, rows_: Iterable[dict[str, Any]], shots: Sequence[dict[str, Any]]
                 ) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """One extraction prompt per distinct claim (VERITE's out-of-context rows
    reuse a true row's caption), id-checked against every row that carries it."""
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        if (row.get("text") or "").strip():
            by_claim[claim_key(row["text"])].append(row)
    out, keys = [], {}
    for ck in sorted(by_claim):
        group = by_claim[ck]
        claim = " ".join(group[0]["text"].split())
        msgs = extract_messages(claim, shots)
        refuse_ids(messages_text(msgs), frozenset().union(*(id_forms(r) for r in group)),
                   explained=f"{claim}\n{messages_text(msgs[:-1])}\n{EXTRACT_TEMPLATE}",
                   what="an extraction prompt")
        name = f"{dataset}:{ck}"
        key = neutral_key("c", name)
        keys[key] = name
        out.append({"key": key, "task": "extract", "messages": msgs, "content": claim})
    return out, keys


def cmd_prompts_extract_c(args) -> int:
    """C.1's input: the seeded pilot slice of M4FC train's claims, or every
    claim C reads -- M4FC train+val and VERITE's evaluation rows."""
    from scripts.image_text import _priority

    shots = fewshot()["extract"]
    m4fc_rows = list(rows("m4fc", M4FC_SPLITS))
    if args.part == "pilot":
        train = sorted((r for r in m4fc_rows if r["split"] == "train"),
                       key=lambda r: _priority(SEED, claim_key(r["text"])))
        parts = [("m4fc", train[:PILOT_SLICE])]
        path = C_EXTRACT_PILOT
    else:
        parts = [("m4fc", m4fc_rows), ("verite", evaluation_rows())]
        path = C_EXTRACT
    built, keys = [], {}
    for ds, rows_ in parts:
        out, k = extract_rows(ds, rows_, shots)
        built += out
        keys.update(k)
    meta = write_prompts(path, "extract", built, extract_prompt_config(), keys)
    print(f"{meta['rows']:,} extraction prompts ({json.dumps(ordered(Counter(v.split(':')[0] for v in keys.values())))}), "
          f"every one id-checked -> {rel(path)} (prompt config {meta['prompt_config_sha256'][:12]})")
    lengths = prompt_tokens(built)
    if lengths:
        print(f"  prompt tokens: min {lengths[0]:,}, median {lengths[len(lengths) // 2]:,}, "
              f"max {lengths[-1]:,}")
    return 0


# --------------------------------------------------------------------------
# stage C.2-C.4: a verdict per item, and the record's media_mismatch score
# --------------------------------------------------------------------------

C_VERDICTS = C_DIR / "verdicts.jsonl"
OTHER_PERSON_DECISIONS = C_DIR / "review" / "other_person_decisions.json"
#: A verdict's reason code, from the processed layer's vocabulary.
REASON_CODES = {"consistent": "none", "contradicted": "media_mismatch", "unsupported": "content_unverified"}
VERDICT_OF = {CONSISTENT: "consistent", CONTRADICTS: "contradicted", UNCLEAR: "unsupported"}
#: Why a claimed name the gallery did not accept decides nothing: verify's
#: reason, a reject renamed for what it is (not evidence of absence).
PEOPLE_WHY = {"below_threshold": "gallery_reject"}
#: The person's answers on the other_person page; only "contradicts" counts.
OTHER_PERSON_ANSWERS = ("contradicts", "not_a_contradiction", "unsure")
#: VERITE's classes and M4FC's, as C reports them.
VERITE_CLASSES = ("true", "miscaptioned", "out-of-context")


def gate_cleared(report: dict[str, Any] | None = None) -> dict[str, bool]:
    """item -> whether Gate A cleared it: after the person's review where they
    decided any contested pair, else as built. An item it did not clear stays
    unsupported in C, and is never re-tuned on the sheet. People are the
    gallery's, which Gate A does not measure."""
    rep = report or json.loads(REPORT_A.read_text(encoding="utf-8"))
    review = rep.get("review") or {}
    gate = review["gate"] if review.get("decided") and review.get("gate") else rep["gate"]
    return {item: bool((gate.get(item) or {}).get("passed")) for item in FIXED}


def _verdict(rec: dict[str, Any], verdict: str, why: str, score: float | None) -> dict[str, Any]:
    return {**rec, "verdict": verdict, "why": why, "score": score, "reason_code": REASON_CODES[verdict]}


def item_verdict(item: str, claimed: str | None, predicted: Any, cleared: bool,
                 judged: Callable[[str, str, str], dict[str, Any] | None] | None = None,
                 evidence: Sequence[dict[str, Any]] = (), *, carried: bool = False,
                 left_out: str | None = None, grounded: bool | None = None) -> dict[str, Any] | None:
    """One item's verdict record. No claimed value: no verdict. No predicted
    value, or a refused one: unsupported. Otherwise the fixed comparator
    decides -- the rule for a date, the judge for a place or an event -- and
    its outcome is kept on the record whatever the verdict. Then, in order, it
    is unsupported where:
    - Gate A did not clear the item (``gate_a_not_cleared``);
    - a person's decision leaves the item out (``left_out``: its why);
    - a predicted date's year is in none of its cited evidence (``grounded``
      False: ``date_not_grounded``);
    - the evidence is text in the image that holds the claim (``carried``:
      ``image_carries_claim``);
    - the comparator cannot tell (``comparator_unclear``)."""
    if not claimed:
        return None
    rec = {"item": item, "claimed": claimed, "predicted": predicted if isinstance(predicted, str) else None,
           "evidence": list(evidence), "comparator": ROUTES[item][0], "gate_a_cleared": cleared,
           "image_carries_claim": bool(carried), "outcome": None}
    if predicted is ERROR:
        return _verdict(rec, "unsupported", "prediction_refused", None)
    if predicted is None:
        return _verdict(rec, "unsupported", "no_image_evidence", None)
    if item == "date":
        out = date_verdict(claimed, predicted)
        rec["outcome"] = {"label": out["label"], "why": out["why"], "score": out["score"]}
    else:
        j = judged(item, claimed, predicted) if judged else None
        if j is None:
            return _verdict(rec, "unsupported", "comparator_not_run", None)
        res = j["result"]
        rec["outcome"] = {"label": res["label"], "why": "judge_" + (res["answer"] or "off_format"),
                          "score": res["score"]}
    if not cleared:
        return _verdict(rec, "unsupported", "gate_a_not_cleared", None)
    if left_out:
        return _verdict(rec, "unsupported", left_out, None)
    if grounded is False:
        return _verdict(rec, "unsupported", "date_not_grounded", None)
    if carried:
        return _verdict(rec, "unsupported", "image_carries_claim", None)
    if rec["outcome"]["label"] == UNCLEAR:
        return _verdict(rec, "unsupported", f"comparator_unclear:{rec['outcome']['why']}", None)
    return _verdict(rec, VERDICT_OF[rec["outcome"]["label"]], rec["outcome"]["why"], rec["outcome"]["score"])


def _commons_text(value: Any, *keys: str) -> str | None:
    """A Commons licence or attribution as plain text: the gallery keeps them
    as Commons states them (dicts), and a page shows their words."""
    if value is None or isinstance(value, str):
        return value
    words = [str(value[k]).strip() for k in keys if isinstance(value, dict) and value.get(k)]
    return ", ".join(dict.fromkeys(w for w in words if w)) or None


def gallery_evidence(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("matched_reference") or {}
    return {"kind": "gallery", "name": row["name"], "qid": row["matched_entity"], "score": row["score"],
            "reference": ref.get("page_url"), "licence": _commons_text(ref.get("licence"), "short_name"),
            "attribution": _commons_text(ref.get("attribution"), "artist", "credit")}


def other_person_key(sha: str, claimed: str, qid: str) -> str:
    return f"{sha}|{claimed}|{qid}"


def people_verdict(sha: str, claim_names: Sequence[str], verified: dict[tuple[str, str], dict[str, Any]],
                   vlm_rows: Sequence[dict[str, Any]], decisions: dict[str, str] | None = None
                   ) -> dict[str, Any] | None:
    """C's people rule, by the gallery at its certified threshold:
    - a claimed name accepted -> consistent;
    - else a VLM-named other person accepted on the image's only face ->
      unsupported until the person checks it (``other_person_unchecked``), and
      contradicted only on their "contradicts". That check is the filter: a
      face in a screenshot's interface (an avatar) is the person's to set
      aside there, by no size or position rule (the person's decision,
      2026-11-10: such a rule would be fitted to their own checks);
    - else unsupported, for verify's own reason: a reject is not evidence of
      absence (``gallery_reject``). With ``decisions`` None, no check counts:
      the figures before the person's check."""
    if not claim_names:
        return None
    claimed = "; ".join(claim_names)
    rec = {"item": "people", "claimed": claimed, "predicted": None, "evidence": [], "comparator": "gallery",
           "gate_a_cleared": None, "outcome": None}
    rows_ = [verified.get((sha, n)) for n in claim_names]
    accepted = [r for r in rows_ if r and r["match"] is True]
    if accepted:
        rec.update(predicted=accepted[0]["name"], evidence=[gallery_evidence(accepted[0])])
        return _verdict(rec, "consistent", "claimed_person_accepted", 0.0)
    claimed_qids = {q for r in rows_ if r for q in r["qids"]}
    others = [v for v in vlm_rows if v["match"] is True and v["faces_in_image"] == 1
              and v["matched_entity"] not in claimed_qids and v["name"] not in claim_names]
    if others:
        o = others[0]
        rec.update(predicted=o["name"], evidence=[gallery_evidence(o)])
        answer = (decisions or {}).get(other_person_key(sha, claimed, o["matched_entity"]))
        if answer == "contradicts":
            return _verdict(rec, "contradicted", "other_person_confirmed", 1.0)
        return _verdict(rec, "unsupported", "other_person_not_confirmed" if answer else "other_person_unchecked",
                        None)
    first = next((r for r in rows_ if r), None)
    why = "not_verified" if first is None else PEOPLE_WHY.get(first["reason"], first["reason"])
    return _verdict(rec, "unsupported", why, None)


def record_score(verdicts: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    """media_mismatch = the largest P(contradicts) of the items decided. A
    record with no decided item abstains: it scores 0 and is counted."""
    decided = [v for v in verdicts if v and v["verdict"] in ("consistent", "contradicted")]
    if not decided:
        return {"score": 0.0, "abstained": True, "by": None}
    top = max(decided, key=lambda v: (v["score"], v["item"]))
    return {"score": top["score"], "abstained": False, "by": top["item"]}


#: What the scores count (the person's decisions at Gates A and B,
#: 2026-11-10). The primary score counts location and people. Event did not
#: clear Gate A (8 of 78 true-row false contradictions, against at most 0.10),
#: and Gate B left dates out: the image side's dates agreed with M4FC's actual
#: date 5-8% of the time. Each variant puts one item back, and is reported as
#: a variant, never as the score: event as the comparator decides it, and
#: dates only where their year is written in the evidence they cite.
SCORE_SETS = ("primary", "with_event", "with_grounded_date")
VARIANT_OF = {"with_event": "event", "with_grounded_date": "date"}
GATE_B_LEFT_OUT = "gate_b_left_out"


def record_verdicts(values: dict[str, Any], pred: dict[str, Any], cited: dict[str, list[dict[str, Any]]],
                    ev: dict[str, Any], judged: Callable[[str, str, str], dict[str, Any] | None] | None,
                    cleared: dict[str, bool], *, rule: str | None = CARRY_RULE
                    ) -> tuple[dict[str, dict[str, Any] | None], dict[str, dict[str, Any] | None]]:
    """(the date, location and event verdicts as the primary score reads them,
    the variants' event and date verdicts) of one record under one claim
    variant. ``ev`` holds the prediction's cited ids per item and its evidence
    lines; ``rule`` is image_carries_claim's form (CARRY_RULES), None to leave
    it out and measure it."""
    ids, lines = ev.get("ids") or {}, ev.get("lines") or {}

    def one(item: str, cleared_: bool, **kw) -> dict[str, Any] | None:
        c = values.get(item) if isinstance(values.get(item), str) else None
        p = pred.get(item)
        carried = rule is not None and isinstance(p, str) and image_carries_claim(item, c, ids.get(item, []),
                                                                                    lines, rule)
        return item_verdict(item, c, p, cleared_, judged, cited.get(item, []), carried=carried, **kw)

    p = pred.get("date")
    grounded = date_grounded(p, ids.get("date", []), lines) if isinstance(p, str) else None
    items = {"date": one("date", cleared.get("date", False), left_out=GATE_B_LEFT_OUT),
             "location": one("location", cleared.get("location", False)),
             "event": one("event", cleared.get("event", False))}
    variants = {"event": one("event", True), "date": one("date", cleared.get("date", False), grounded=grounded)}
    return items, variants


def counted_items(items: dict[str, Any], variants: dict[str, Any], score_set: str) -> dict[str, Any]:
    """The verdicts a score reads: the primary items, with the variant's one
    item in place of its primary verdict."""
    which = VARIANT_OF.get(score_set)
    return {**items, which: variants.get(which)} if which else dict(items)


def scores_of(items: dict[str, Any], variants: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s: record_score(counted_items(items, variants, s).values()) for s in SCORE_SETS}


def other_person_decisions(path: Path | None = None) -> dict[str, str]:
    """The person's answers on the other_person page; an unknown answer refuses."""
    path = path or OTHER_PERSON_DECISIONS
    if not path.is_file():
        return {}
    out = {}
    for key, answer in json.loads(path.read_text(encoding="utf-8")).items():
        if answer is None:
            continue
        if answer not in OTHER_PERSON_ANSWERS:
            raise SystemExit(f"{path.name}: {answer!r} is not one of {OTHER_PERSON_ANSWERS}")
        out[key] = answer
    return out


# --------------------------------------------------------------------------
# stage C.6: metrics -- a threshold fitted on M4FC train only, AUROC, a paired
# cluster bootstrap against CLIP
# --------------------------------------------------------------------------

BOOTSTRAP = 2000


def fit_threshold(records: Sequence[dict[str, Any]], score: str = "score") -> dict[str, Any]:
    """The threshold that maximises balanced accuracy, true against
    out-of-context, on M4FC TRAIN and nothing else: a val or VERITE record
    refuses. Ties go to the lowest threshold."""
    bad = [r for r in records if r["dataset"] != "m4fc" or r["split"] != "train"]
    if bad:
        raise GuardError(f"a threshold is fitted on M4FC train only; {len(bad)} record(s) are not")
    pos = [r[score] for r in records if r["class"] == OOC]
    neg = [r[score] for r in records if r["class"] == "true"]
    if not pos or not neg:
        return {"threshold": None, "balanced_accuracy": None, "true": len(neg), "out_of_context": len(pos)}
    best = None
    for t in sorted(set(pos) | set(neg)):
        ba = (sum(s >= t for s in pos) / len(pos) + sum(s < t for s in neg) / len(neg)) / 2
        if best is None or ba > best[1]:
            best = (t, ba)
    return {"threshold": best[0], "balanced_accuracy": round(best[1], 4), "true": len(neg),
            "out_of_context": len(pos)}


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """P(a mismatched record outscores a true one); ties count half."""
    from scripts.clip_pairs import auc

    return None if not pos or not neg else auc(list(pos), list(neg))


def clusters_of(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """record key -> its cluster: records joined by a shared image or a shared
    claim (VERITE's triplets: a true row and its miscaptioned row share an
    image, its out-of-context row the caption). A bootstrap resamples whole
    clusters."""
    parent = {r["key"]: r["key"] for r in records}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for field in ("sha256", "claim_key"):
        first: dict[str, str] = {}
        for r in records:
            if r.get(field) in first:
                parent[find(r["key"])] = find(first[r[field]])
            elif r.get(field):
                first[r[field]] = r["key"]
    return {k: find(k) for k in parent}


def paired_auroc(records: Sequence[dict[str, Any]], positive: str, negative: str, a: str, b: str,
                 n: int = BOOTSTRAP, seed: int = SEED) -> dict[str, Any]:
    """AUROC of score ``a`` and of score ``b`` for ``positive`` against
    ``negative`` records, and their difference with a percentile 95% interval
    from a paired bootstrap that resamples whole clusters."""
    part = [r for r in records if r["class"] in (positive, negative)]
    clusters = clusters_of(part)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in part:
        groups[clusters[r["key"]]].append(r)

    def both(rs):
        pos = [r for r in rs if r["class"] == positive]
        neg = [r for r in rs if r["class"] == negative]
        ra, rb = auroc([r[a] for r in pos], [r[a] for r in neg]), auroc([r[b] for r in pos], [r[b] for r in neg])
        return ra, rb

    ra, rb = both(part)
    rng, ids, deltas = random.Random(seed), sorted(groups), []
    for _ in range(n):
        sample = [r for c in (rng.choice(ids) for _ in ids) for r in groups[c]]
        x, y = both(sample)
        if x is not None and y is not None:
            deltas.append(x - y)
    deltas.sort()
    ci = [round(deltas[int(0.025 * len(deltas))], 4), round(deltas[int(0.975 * len(deltas)) - 1], 4)] \
        if deltas else None
    return {"records": len(part), "clusters": len(ids), "positive": positive, "negative": negative,
            f"auroc_{a}": None if ra is None else round(ra, 4), f"auroc_{b}": None if rb is None else round(rb, 4),
            "delta": None if ra is None or rb is None else round(ra - rb, 4), "delta_ci95": ci,
            "resamples": len(deltas), "excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0))}


def flag_cell(records: Sequence[dict[str, Any]], threshold: float | None, score: str = "score",
              abstained: str | None = "abstained") -> dict[str, Any]:
    """The share flagged at the train threshold, and the share that abstained
    (never flagged). ``abstained`` None for a score that never abstains: CLIP's."""
    held = (lambda r: bool(r.get(abstained))) if abstained else (lambda r: False)
    flagged = 0 if threshold is None else sum(1 for r in records if r[score] >= threshold and not held(r))
    out = {"records": len(records), "flagged": rate(flagged, len(records))}
    if abstained:
        out["abstained"] = rate(sum(1 for r in records if held(r)), len(records))
    return out


def evidence_cell(items: dict[str, Any]) -> str:
    """Which evidence the verdicts a score reads had: image-side only, +
    retrieved documents, + a gallery-scored person (C.6's second breakdown)."""
    vs = [v for v in items.values() if v]
    docs = any(e.get("kind") == "document" for v in vs for e in v["evidence"])
    person = any(v["item"] == "people" and (v["why"] in ("claimed_person_accepted", "gallery_reject")
                                            or v["why"].startswith("other_person")) for v in vs)
    return "image only" + (" + documents" if docs else "") + (" + a gallery-scored person" if person else "")


# --------------------------------------------------------------------------
# stage C: the judge's pairs, the verdicts, the report
# --------------------------------------------------------------------------

C_JUDGE = C_DIR / "judge_in.jsonl"
REPORT_C = REPORTS / "consistency_c.json"
REPORT_C_MD = REPORTS / "consistency_c.md"
#: B's prediction C reads, per corpus: M4FC with its claim's retrieval, VERITE
#: from the image alone.
C_PREDICTION = {"m4fc": "retrieval", "verite": "image_only"}


def extracted_values(status: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """item -> an extraction's value, None (the claim does not say), or ERROR."""
    items = parsed["items"] if status != "error" else None
    return {i: ERROR if items is None or items[i]["error"] else items[i]["value"] for i in PREDICT_ITEMS}


def _extraction_run(inp: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, str]]:
    keys = json.loads(inp.with_name(inp.stem + ".keys.json").read_text(encoding="utf-8"))
    results = read_results(inp)
    if missing := [k for k in keys if k not in results]:
        raise SystemExit(f"{len(missing):,} of {len(keys):,} extractions are not in {output_path(inp).name}")
    return keys, results, {r["key"]: r["content"] for r in read_input(inp, "extract")[0]}


def c_extractions(inp: Path | None = None) -> dict[str, dict[str, Any]]:
    """"dataset:claim_key" -> item -> the extracted value, None (the claim
    does not say), or ERROR (refused). Every answer is read again by today's
    parser (EXTRACT_PARSER), from the raw text the run kept: a parser change
    needs no rerun. Refused if the run is incomplete."""
    keys, results, claims = _extraction_run(inp or C_EXTRACT)
    return {keys[k]: extracted_values(*parse_extraction(r["result"]["raw"], claims[k], r["result"]["finish"]))
            for k, r in results.items()}


def extraction_changes(inp: Path | None = None) -> dict[str, Any]:
    """What reading C.1's answers by today's parser changed against what the
    run wrote: per dataset, rows by status both ways, and each changed item."""
    keys, results, claims = _extraction_run(inp or C_EXTRACT)
    out: dict[str, Any] = {"parser": EXTRACT_PARSER, "datasets": {}}
    for k, r in sorted(results.items(), key=lambda kv: keys[kv[0]]):
        ds = keys[k].split(":", 1)[0]
        d = out["datasets"].setdefault(ds, {"rows": 0, "as_run": Counter(), "as_read": Counter(), "changed": []})
        status, parsed = parse_extraction(r["result"]["raw"], claims[k], r["result"]["finish"])
        d["rows"] += 1
        d["as_run"][r["status"]] += 1
        d["as_read"][status] += 1
        before = extracted_values(r["status"], r["result"])
        after = extracted_values(status, parsed)
        for item in PREDICT_ITEMS:
            if before[item] is not after[item] and before[item] != after[item]:
                d["changed"].append({"item": item, "as_run": "refused" if before[item] is ERROR else before[item],
                                     "as_read": "refused" if after[item] is ERROR else after[item]})
    for d in out["datasets"].values():
        d["as_run"], d["as_read"] = ordered(d["as_run"]), ordered(d["as_read"])
    return out


def claim_values(record: dict[str, Any], extracted: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """variant -> item -> the claim's value: the extractor's, and for M4FC
    also its own annotation (the gold-claim variant, which separates the
    extractor's error from the rest)."""
    from scripts.m4fc_audit import present, shown

    out = {"extracted": extracted.get(f"{record['dataset']}:{record['claim_key']}")
           or dict.fromkeys(PREDICT_ITEMS)}
    if record["dataset"] == "m4fc":
        meta = record["row"]["meta"]
        out["gold"] = {i: shown(meta[f"claimed_{i}"]) if present(meta.get(f"claimed_{i}")) else None
                       for i in PREDICT_ITEMS}
    return out


def c_judge_pairs(records: Sequence[dict[str, Any]], preds: dict[str, dict[str, Any]],
                  extracted: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """(item, statement A, statement B) -> the records it decides, asked once:
    each claim value against B's prediction (A = the claim, B = what is known
    about the photograph, as the judge's template has it), and on M4FC the
    record's own claimed value against the extractor's (C.1's score)."""
    pairs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        pred = preds.get(f"{r['dataset']}:{r['sha256']}:{C_PREDICTION[r['dataset']]}") or {}
        claims = claim_values(r, extracted)
        for item in JUDGED_ITEMS:
            p = pred.get(item)
            for values in claims.values():
                c = values.get(item)
                if isinstance(c, str) and isinstance(p, str):
                    pairs[(item, c, p)].append(r)
            g, e = claims.get("gold", {}).get(item), claims["extracted"].get(item)
            if isinstance(g, str) and isinstance(e, str):
                pairs[(item, g, e)].append(r)
    return pairs


def rule_changes(with_rule: dict[str, Any], without: dict[str, Any]) -> dict[str, Any]:
    """item -> the verdict image_carries_claim replaced, where it changed one."""
    return {k: {"was": without[k]["verdict"], "score_was": without[k]["score"]}
            for k, v in with_rule.items() if v and without.get(k) and v["verdict"] != without[k]["verdict"]}


def assemble_verdicts(records: Sequence[dict[str, Any]], preds: dict[str, dict[str, Any]],
                      cites: dict[str, dict[str, list[dict[str, Any]]]], extracted: dict[str, dict[str, Any]],
                      judged: Callable[[str, str, str], dict[str, Any] | None], cleared: dict[str, bool],
                      verified: dict[tuple[str, str], dict[str, Any]], vlm: dict[str, list[dict[str, Any]]],
                      names: Callable[[dict[str, Any]], list[str]], decisions: dict[str, str],
                      evidence: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One verdict record per record and claim variant: its items' verdicts
    as the primary score reads them, the variants' items, and every score
    (SCORE_SETS) four ways -- after the person's check of other_person
    contradictions, before it (none counts), under image_carries_claim's
    narrow form (before Gate C widened it), and without the rule (to measure
    it on train). ``rule_changed`` is what the rule changed; ``widening_changed``
    what widening it changed."""
    out = []

    def tagged(items_, variants_):
        return {**items_, **{f"variant:{k}": v for k, v in variants_.items()}}

    for r in records:
        name = f"{r['dataset']}:{r['sha256']}:{C_PREDICTION[r['dataset']]}"
        pred, cited = preds.get(name) or dict.fromkeys(PREDICT_ITEMS), cites.get(name) or {}
        ev = (evidence or {}).get(name) or {}
        claim_names = names(r)
        people = {when: people_verdict(r["sha256"], claim_names, verified, vlm.get(r["sha256"], []), d)
                  for when, d in (("before", None), ("after", decisions))}
        for variant, values in claim_values(r, extracted).items():
            items, variants = record_verdicts(values, pred, cited, ev, judged, cleared)
            narrow, narrow_variants = record_verdicts(values, pred, cited, ev, judged, cleared, rule="narrow")
            bare, bare_variants = record_verdicts(values, pred, cited, ev, judged, cleared, rule=None)
            out.append({"key": r["key"], "dataset": r["dataset"], "split": r["split"], "class": r["class"],
                        "variant": variant, "sha256": r["sha256"], "claim_key": r["claim_key"],
                        "items": {**items, "people": people["after"]}, "variants": variants,
                        "scores": scores_of({**items, "people": people["after"]}, variants),
                        "scores_before_check": scores_of({**items, "people": people["before"]}, variants),
                        "scores_narrow_rule": scores_of({**narrow, "people": people["after"]}, narrow_variants),
                        "scores_without_rule": scores_of({**bare, "people": people["after"]}, bare_variants),
                        "rule_changed": rule_changes(tagged(items, variants), tagged(bare, bare_variants)),
                        "widening_changed": rule_changes(tagged(items, variants),
                                                         tagged(narrow, narrow_variants))})
    return out


def flat_scores(verdicts: Sequence[dict[str, Any]], clip: dict[str, float | None]) -> list[dict[str, Any]]:
    """One row per verdict record and score (SCORE_SETS): its variant is
    "<claim variant>:<score>", and CLIP's score is the negated similarity, so
    that for both a higher score means more of a mismatch."""
    out = []
    for v in verdicts:
        for s in SCORE_SETS:
            sc, before, bare = v["scores"][s], v["scores_before_check"][s], v["scores_without_rule"][s]
            narrow = v["scores_narrow_rule"][s]
            out.append({"key": v["key"], "dataset": v["dataset"], "split": v["split"], "class": v["class"],
                        "variant": f"{v['variant']}:{s}", "claim_variant": v["variant"], "score_set": s,
                        "sha256": v["sha256"], "claim_key": v["claim_key"],
                        "score": sc["score"], "abstained": sc["abstained"],
                        "score_before_check": before["score"], "abstained_before_check": before["abstained"],
                        "score_narrow_rule": narrow["score"], "abstained_narrow_rule": narrow["abstained"],
                        "score_without_rule": bare["score"], "abstained_without_rule": bare["abstained"],
                        "cell": evidence_cell(counted_items(v["items"], v["variants"], s)),
                        "clip": None if clip.get(v["key"]) is None else -clip[v["key"]]})
    return out


def corpus_of(v: dict[str, Any]) -> str:
    return "verite" if v["dataset"] == "verite" else f"m4fc {v['split']}"


def item_counts(verdicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per corpus and claim variant, per item (the variants' two as well), per
    class: the records, those whose claim has the item, and their verdicts and
    reasons."""
    out: dict[str, Any] = {}
    for v in verdicts:
        cls = CLASS_NAMES.get(v["class"], v["class"])
        corpus = out.setdefault(f"{corpus_of(v)}:{v['variant']}", {})
        for item, x in [*v["items"].items(), *((f"{k} (variant)", x) for k, x in v["variants"].items())]:
            cell = corpus.setdefault(item, {}).setdefault(cls, {"records": 0, "claimed": 0, "verdict": Counter(),
                                                                "why": Counter()})
            cell["records"] += 1
            if x:
                cell["claimed"] += 1
                cell["verdict"][x["verdict"]] += 1
                cell["why"][x["why"].split(":", 1)[0]] += 1
    for corpus in out.values():
        for by_class in corpus.values():
            for cell in by_class.values():
                cell["verdict"], cell["why"] = ordered(cell["verdict"]), ordered(cell["why"])
    return out


def carry_effect(verdicts: Sequence[dict[str, Any]], flat: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """image_carries_claim measured on M4FC train, extracted claims: the
    verdicts it made unsupported (by item, class and what they had been), what
    widening it at Gate C changed, and per score its AUROC (true against
    out-of-context) and balanced accuracy at its own train threshold -- widened
    ("with"), narrow, and without the rule. On val and VERITE, where it is
    applied unchanged and never measured for a choice: what it changed, and
    the primary score's AUROC under each form, reported."""
    def changed(vs, field="rule_changed"):
        c: dict[str, Counter] = defaultdict(Counter)
        for v in vs:
            for item, ch in v[field].items():
                c[item][f"{CLASS_NAMES.get(v['class'], v['class'])}: was {ch['was']}"] += 1
        return {k: ordered(x) for k, x in sorted(c.items())}

    def pick(sel):
        return [v for v in verdicts if v["variant"] == "extracted" and sel(v)]

    forms = (("with", "score", "abstained"), ("narrow", "score_narrow_rule", "abstained_narrow_rule"),
             ("without", "score_without_rule", "abstained_without_rule"))
    train = pick(lambda v: v["dataset"] == "m4fc" and v["split"] == "train")
    out: dict[str, Any] = {"rule": CARRY_RULE,
                           "train": {"records": len(train), "changed": changed(train),
                                     "widening_changed": changed(train, "widening_changed"), "scores": {}},
                           "applied_unchanged": {}}
    for s in SCORE_SETS:
        rows_ = [f for f in flat if f["claim_variant"] == "extracted" and f["score_set"] == s
                 and f["dataset"] == "m4fc" and f["split"] == "train"]
        cell = {}
        for tag, key, ab in forms:
            fitted = fit_threshold([dict(f, score=f[key]) for f in rows_])
            a = auroc([f[key] for f in rows_ if f["class"] == OOC], [f[key] for f in rows_ if f["class"] == "true"])
            cell[tag] = {"auroc": None if a is None else round(a, 4), "threshold": fitted["threshold"],
                         "balanced_accuracy": fitted["balanced_accuracy"],
                         "abstained": rate(sum(1 for f in rows_ if f[ab]), len(rows_))}
        out["train"]["scores"][s] = cell
    pairs = {"m4fc val": [(OOC, "true")], "verite": [("out-of-context", "true"), ("miscaptioned", "true")]}
    for name, sel in (("m4fc val", lambda v: v["dataset"] == "m4fc" and v["split"] == "val"),
                      ("verite", lambda v: v["dataset"] == "verite")):
        vs = pick(sel)
        rows_ = [f for f in flat if f["claim_variant"] == "extracted" and f["score_set"] == "primary"
                 and corpus_of(f) == name]
        out["applied_unchanged"][name] = {
            "records": len(vs), "changed": changed(vs), "widening_changed": changed(vs, "widening_changed"),
            "primary_auroc": {f"{p} vs {n}": {tag: _round(auroc([f[key] for f in rows_ if f["class"] == p],
                                                                 [f[key] for f in rows_ if f["class"] == n]))
                                              for tag, key, _ab in forms}
                              for p, n in pairs[name]}}
    return out


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(x, n)


# --------------------------------------------------------------------------
# Gate C's system score: CLIP and the consistency score, fused
# --------------------------------------------------------------------------

#: The system's score from Gate C on (the person's decision, 2026-11-10): one
#: logistic regression on [CLIP's image-claim similarity, the primary
#: consistency score, whether the block decided], fitted on M4FC train only,
#: true against out-of-context. An abstaining record's consistency score and
#: flag are both 0, so its fused score is a function of CLIP's similarity
#: alone: CLIP's, on the regression's scale. Fixed before it was fitted: the
#: features, the claims as extracted, each class weighted to half the total,
#: an L2 penalty of 1 on the standardised features (not the intercept),
#: Newton's method from zero. Explanations still come from the verdicts alone.
FUSED_FEATURES = ("clip_similarity", "consistency_score", "decided")
FUSED_FROM = "extracted:primary"
FUSED_L2 = 1.0
C_FUSED = C_DIR / "fused.json"


def fused_features(similarity: float, score: float, abstained: bool) -> list[float]:
    return [float(similarity), 0.0 if abstained else float(score), 0.0 if abstained else 1.0]


def fit_logistic(X: Sequence[Sequence[float]], y: Sequence[int], l2: float = FUSED_L2,
                 iterations: int = 100) -> dict[str, Any]:
    """A class-balanced logistic regression on standardised features, by
    Newton's method from zero: deterministic, no library beyond numpy."""
    import numpy as np

    x, t = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if not 0 < t.sum() < len(t):
        raise ValueError("a logistic regression needs both classes")
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    z = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
    w = np.where(t == 1, len(t) / (2 * t.sum()), len(t) / (2 * (len(t) - t.sum())))
    pen = np.full(z.shape[1], float(l2))
    pen[0] = 0.0
    beta, converged, n = np.zeros(z.shape[1]), False, 0
    for n in range(1, iterations + 1):
        p = 1.0 / (1.0 + np.exp(-(z @ beta)))
        grad = z.T @ (w * (p - t)) + pen * beta
        hess = (z * (w * p * (1 - p))[:, None]).T @ z + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if float(np.max(np.abs(step))) < 1e-10:
            converged = True
            break
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std], "intercept": float(beta[0]),
            "coef": [float(v) for v in beta[1:]], "iterations": n, "converged": converged,
            "coef_raw_units": [float(b / s) for b, s in zip(beta[1:], std)]}


def fused_score(model: dict[str, Any], features: Sequence[float]) -> float:
    z = model["intercept"] + sum(c * (x - m) / s for c, x, m, s in
                                 zip(model["coef"], features, model["mean"], model["std"]))
    return round(1.0 / (1.0 + math.exp(-z)), 6)


def fit_fused(flat: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The fused score's regression, from M4FC train records of FUSED_FROM
    that CLIP scored, true against out-of-context; a val or VERITE record
    refuses."""
    train = [f for f in flat if f["variant"] == FUSED_FROM and f["dataset"] == "m4fc" and f["split"] == "train"
             and f["class"] in (OOC, "true") and f["clip"] is not None]
    if any(f["dataset"] != "m4fc" or f["split"] != "train" for f in train):
        raise GuardError("the fused score is fitted on M4FC train only")
    model = fit_logistic([fused_features(-f["clip"], f["score"], f["abstained"]) for f in train],
                         [int(f["class"] == OOC) for f in train])
    return {"features": list(FUSED_FEATURES), "from": FUSED_FROM, "fitted_on": "m4fc train, true vs out-of-context",
            "class_weight": "balanced", "l2": FUSED_L2,
            "records": {"true": sum(f["class"] == "true" for f in train),
                        "out_of_context": sum(f["class"] == OOC for f in train)}, **model}


def fused_report(flat: Sequence[dict[str, Any]], clip_threshold: float | None) -> dict[str, Any]:
    """The fused score against CLIP: its regression and train threshold, then
    on M4FC val and VERITE the paired cluster bootstrap AUROC (every record,
    and the records the block decided), and per class the flag rates at each
    score's train threshold, with the share of the fused flags that rest on
    similarity alone (the block abstained)."""
    model = fit_fused(flat)
    rows_ = [dict(f, fused=fused_score(model, fused_features(-f["clip"], f["score"], f["abstained"])))
             for f in flat if f["variant"] == FUSED_FROM and f["clip"] is not None]
    train = [f for f in rows_ if f["dataset"] == "m4fc" and f["split"] == "train"]
    fitted = fit_threshold(train, "fused")
    t = fitted["threshold"]
    out: dict[str, Any] = {
        "model": model, "threshold": fitted,
        "train_auroc": {"fused": _round(auroc([f["fused"] for f in train if f["class"] == OOC],
                                              [f["fused"] for f in train if f["class"] == "true"])),
                        "clip": _round(auroc([f["clip"] for f in train if f["class"] == OOC],
                                             [f["clip"] for f in train if f["class"] == "true"]))},
        "corpora": {}}
    pairs_of = {"m4fc": [(OOC, "true")], "verite": [("out-of-context", "true"), ("miscaptioned", "true")]}
    for ds, pairs in pairs_of.items():
        part = [f for f in rows_ if f["dataset"] == ds and (ds == "verite" or f["split"] == "val")]
        decided = [f for f in part if not f["abstained"]]
        entry: dict[str, Any] = {"records": len(part), "coverage": rate(len(decided), len(part)),
                                 "auroc": [], "auroc_decided": [], "by_class": {}}
        for pos, neg in pairs:
            both = [f for f in part if f["class"] in (pos, neg)]
            cov = rate(sum(1 for f in both if not f["abstained"]), len(both))
            entry["auroc"].append({**paired_auroc(part, pos, neg, "fused", "clip"), "coverage": cov})
            entry["auroc_decided"].append({**paired_auroc(decided, pos, neg, "fused", "clip"), "coverage": cov})
        for c in sorted({f["class"] for f in part}):
            sub = [f for f in part if f["class"] == c]
            flagged = [f for f in sub if t is not None and f["fused"] >= t]
            entry["by_class"][CLASS_NAMES.get(c, c)] = {
                "fused": flag_cell(sub, t, "fused", None), "clip": flag_cell(sub, clip_threshold, "clip", None),
                "fused_flags_on_similarity_alone": rate(sum(1 for f in flagged if f["abstained"]), len(flagged))}
        out["corpora"][ds] = entry
    out["beats_clip"] = {f"{ds}: {a['positive']} vs {a['negative']}": bool(a["excludes_zero"] and a["delta"] > 0)
                         for ds, e in out["corpora"].items() for a in e["auroc"]}
    return out


def c_report(verdicts: Sequence[dict[str, Any]], clip: dict[str, float | None]) -> dict[str, Any]:
    """Report C: thresholds fitted on M4FC train (the system's per claim
    variant and score, and CLIP's), then on M4FC val and VERITE, per claim
    variant and score: coverage, the flag and abstention rates per class and
    per evidence cell beside CLIP's, AUROC against CLIP with a paired cluster
    bootstrap (on every record, and on the records the system decided), and
    the people check's before and after. Then the items' verdicts by class,
    and image_carries_claim's measurement."""
    pairs_of = {"m4fc": [(OOC, "true")],
                "verite": [("out-of-context", "true"), ("miscaptioned", "true")]}
    flat = flat_scores(verdicts, clip)
    report: dict[str, Any] = {"thresholds": {}, "corpora": {},
                              "clip_missing": sum(1 for f in flat if f["clip"] is None) // len(SCORE_SETS)}
    for variant in sorted({f["variant"] for f in flat}):
        train = [f for f in flat if f["variant"] == variant and f["dataset"] == "m4fc" and f["split"] == "train"]
        report["thresholds"][variant] = {
            "system": fit_threshold(train), "clip": fit_threshold([f for f in train if f["clip"] is not None],
                                                                  "clip")}
        t, tc = report["thresholds"][variant]["system"]["threshold"], report["thresholds"][variant]["clip"]["threshold"]
        for ds, pairs in pairs_of.items():
            part = [f for f in flat if f["variant"] == variant and f["dataset"] == ds
                    and (ds == "verite" or f["split"] == "val") and f["clip"] is not None]
            if not part:
                continue
            decided = [f for f in part if not f["abstained"]]
            entry = {"records": len(part), "coverage": rate(len(decided), len(part)), "by_class": {},
                     "by_evidence": {}, "auroc": [], "auroc_decided": [],
                     "people_check": {"before": flag_cell(part, t, "score_before_check", "abstained_before_check"),
                                      "after": flag_cell(part, t)}}
            for c in sorted({f["class"] for f in part}):
                sub = [f for f in part if f["class"] == c]
                entry["by_class"][CLASS_NAMES.get(c, c)] = {"system": flag_cell(sub, t),
                                                            "clip": flag_cell(sub, tc, "clip", None)}
            for cell in sorted({f["cell"] for f in part}):
                sub = [f for f in part if f["cell"] == cell]
                entry["by_evidence"][cell] = {"system": flag_cell(sub, t), "clip": flag_cell(sub, tc, "clip", None)}
            for pos, neg in pairs:
                both = [f for f in part if f["class"] in (pos, neg)]
                cov = rate(sum(1 for f in both if not f["abstained"]), len(both))
                entry["auroc"].append({**paired_auroc(part, pos, neg, "score", "clip"), "coverage": cov})
                entry["auroc_decided"].append({**paired_auroc(decided, pos, neg, "score", "clip"), "coverage": cov})
            report["corpora"][f"{ds}:{variant}"] = entry
    report["items"] = item_counts(verdicts)
    report["image_carries_claim"] = carry_effect(verdicts, flat)
    if FUSED_FROM in report["thresholds"]:
        report["fused"] = fused_report(flat, report["thresholds"][FUSED_FROM]["clip"]["threshold"])
    return report


def prediction_evidence(row: dict[str, Any], result: dict[str, Any] | None,
                        titles: dict[str, str] | None = None
                        ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """(item -> the evidence a prediction cites, documents with their titles;
    {"ids": item -> its cited ids, "lines": id -> the prompt's evidence line})
    of one predictor input row and its answer."""
    items = ((result or {}).get("result") or {}).get("items") or {}
    ids = {i: [e for e in (items.get(i) or {}).get("evidence", []) if e in row["evidence"]] for i in PREDICT_ITEMS}
    cites = {i: [{**row["evidence"][e], **({"title": (titles or {}).get(row["evidence"][e]["doc_id"])}
                                           if row["evidence"][e]["kind"] == "document" else {})} for e in ids[i]]
             for i in PREDICT_ITEMS}
    return cites, {"ids": ids, "lines": evidence_lines(row["messages"])}


def c_inputs(with_cites: bool = True) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]],
                                               dict[str, dict[str, list[dict[str, Any]]]],
                                               dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """(C's records, B's predictions by name, each prediction's cited
    evidence by item -- documents with their titles --, its cited ids and
    evidence lines, and the extractions)."""
    records = c_records()
    built, results = b_results()
    preds = {name: predicted(r) for name, r in results.items()}
    cites: dict[str, dict[str, list[dict[str, Any]]]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    if with_cites:
        _cid, _items, held = retrieval_ledger(M4FC_SPLITS)
        titles = {h["doc_id"]: h["title"] for r in held.values() for h in r["hits"]}
        keys = json.loads(B_PREDICT.with_name(B_PREDICT.stem + ".keys.json").read_text(encoding="utf-8"))
        with B_PREDICT.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                name = keys[row["key"]]
                cites[name], evidence[name] = prediction_evidence(row, results[name], titles)
    return records, preds, cites, evidence, c_extractions()


def cmd_prompts_judge_c() -> int:
    """C.2's judge input: every claim-value / prediction pair of location and
    event, and C.1's scoring pairs, asked once each and id-checked."""
    records, preds, _cites, _evidence, extracted = c_inputs(with_cites=False)
    shots = fewshot()["judge"]
    out, keys = [], {}
    for (item, a, b), recs in sorted(c_judge_pairs(records, preds, extracted).items()):
        msgs = judge_messages(item, a, b, shots)
        content = f"{a}\n{b}"
        refuse_ids(messages_text(msgs), frozenset().union(*(id_forms(r["row"]) for r in recs)),
                   explained=content + "\n" + messages_text(msgs[:-1]), what="a C judge prompt")
        key = neutral_key("cj", f"{item}\n{a}\n{b}")
        keys[key] = {"item": item, "a": a, "b": b}
        out.append({"key": key, "task": "judge", "messages": msgs, "content": content})
    meta = write_prompts(C_JUDGE, "judge", out, judge_prompt_config(shots), keys)
    print(f"{meta['rows']:,} C judge prompts, every one id-checked -> {rel(C_JUDGE)} "
          f"(prompt config {meta['prompt_config_sha256'][:12]}, stage A's)")
    return 0


def cmd_verdicts() -> int:
    """C.2: every record's verdicts, from B's predictions, the extractions,
    the C judge, the gallery and the person's other_person decisions."""
    from scripts import gallery as gl

    records, preds, cites, evidence, extracted = c_inputs()
    keys = json.loads(C_JUDGE.with_name(C_JUDGE.stem + ".keys.json").read_text(encoding="utf-8"))
    done = read_results(C_JUDGE)
    judged = {(v["item"], v["a"], v["b"]): done[k] for k, v in keys.items() if k in done}
    need = c_judge_pairs(records, preds, extracted)
    if missing := [p for p in need if p not in judged]:
        raise SystemExit(f"the C judge has not decided {len(missing):,} of {len(need):,} pairs")
    people = list(gl.read_jsonl(B_PEOPLE))
    verified = {(r["sha256"], r["name"]): r for r in people if "claim" in r["sides"]}
    vlm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in people:
        if {"vlm_free", "vlm_forced"} & set(r["sides"]):
            vlm[r["sha256"]].append(r)
    spans = {ds: claim_ner(ds) for ds in B_DATASETS}

    def names(rec: dict[str, Any]) -> list[str]:
        return claim_people(spans[rec["dataset"]].get(str(rec["row"]["record_id"])) or {"spans": []})

    verdicts = assemble_verdicts(records, preds, cites, extracted, lambda i, a, b: judged.get((i, a, b)),
                                 gate_cleared(), verified, vlm, names, other_person_decisions(), evidence)
    gl.write_jsonl(C_VERDICTS, verdicts)
    print(f"{len(verdicts):,} verdict records -> {rel(C_VERDICTS)}")
    for s in SCORE_SETS:
        print(f"  {s}: " + json.dumps(ordered(Counter(
            f"{v['variant']}:{'abstained' if v['scores'][s]['abstained'] else 'decided'}" for v in verdicts))))
    unchecked = {other_person_key(v["sha256"], v["items"]["people"]["claimed"], v["items"]["people"]["evidence"][0]["qid"])
                 for v in verdicts if v["items"]["people"] and v["items"]["people"]["why"] == "other_person_unchecked"}
    print(f"  other_person contradictions waiting for the person's check: {len(unchecked)}")
    return 0


#: The person's decisions report C is built under (2026-11-10), as it states them.
C_DECISIONS = {
    "primary_score": "location and people",
    "event": "Gate A not cleared (true-row false contradictions 8/78 = 0.1026 > 0.10): unsupported in the "
             "primary score; reported as the variant with_event",
    "date": "left out of the primary score at Gate B (B's dates right 5-8% of the time); reported as the "
            "variant with_grounded_date: a date counts only where its year is written in the evidence it cites",
    "image_carries_claim": "a predicted value resting only on OCR or text-basis entity evidence that contains "
                           "the claimed value is unsupported; fixed before C ran, measured on M4FC train, "
                           "applied unchanged to val and VERITE. Widened at Gate C (a rule defect found on "
                           "train): for a place, a place line recognised by how it looks whose name the OCR "
                           "text holds rests on that OCR text",
    "predictor_not_json": "the 19 predictor answers short of their final brace are not repaired: they stay errors",
    "extraction_parser": "decades and spans whose years the claim writes are dates (parser 2); no rerun",
    "gate_c": "accepted as measured: zero-shot context-then-compare does not beat CLIP",
    "fused_score": "the system's score from Gate C: one logistic regression on [CLIP similarity, the primary "
                   "consistency score, decided], fitted on M4FC train only; where the block abstains it is a "
                   "function of CLIP's similarity alone. Explanations come from the verdicts only",
    "other_person_face": "no size or position rule for the face other_person scores (a face rule was built and "
                         "reverted: it would be fitted to the person's own checks); every other_person "
                         "contradiction needs the person's review before it counts, and that review is the filter",
}


def cmd_score_c() -> int:
    """Report C from the verdicts and CLIP's ledger."""
    from scripts import gallery as gl

    if not C_VERDICTS.is_file() or not C_CLIP.is_file():
        raise SystemExit("report C needs the verdicts (`verdicts`) and CLIP's ledger (`clip`)")
    verdicts = list(gl.read_jsonl(C_VERDICTS))
    clip = {r["key"]: r["similarity"] for r in gl.read_jsonl(C_CLIP)}
    changes = extraction_changes()
    decisions = other_person_decisions()
    report = {"written_at": utcnow(), "decisions": C_DECISIONS, "gate_a_cleared": gate_cleared(),
              "score_sets": list(SCORE_SETS), "extraction": changes,
              "people_check": {"decisions": len(decisions)}, **c_report(verdicts, clip)}
    public_report(report, REPORT_C)
    REPORT_C_MD.write_text(render_c(report), encoding="utf-8", newline="\n")
    if report.get("fused"):
        write_json(C_FUSED, {"model": report["fused"]["model"], "threshold": report["fused"]["threshold"],
                             "clip_threshold": report["thresholds"][FUSED_FROM]["clip"]["threshold"],
                             "written_at": report["written_at"]})
    for k, v in report["corpora"].items():
        print(k, json.dumps([(a["positive"], a["auroc_score"], a["auroc_clip"], a["delta"], a["delta_ci95"])
                             for a in v["auroc"]]))
    for ds, e in (report.get("fused") or {}).get("corpora", {}).items():
        print(f"fused {ds}", json.dumps([(a["positive"], a["auroc_fused"], a["auroc_clip"], a["delta"],
                                          a["delta_ci95"]) for a in e["auroc"]]))
    print(f"-> {rel(REPORT_C)} and {rel(REPORT_C_MD)} (local)")
    return 0


def _r3(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _pct(cell: dict[str, Any] | None) -> str:
    return "–" if not cell or cell["rate"] is None else f"{cell['k']}/{cell['n']} = {cell['rate']:.2f}"


def _auroc_row(label: str, entry: dict[str, Any], a: dict[str, Any], score: str = "score") -> str:
    ci = a["delta_ci95"]
    delta = "–" if a["delta"] is None else f"{a['delta']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
    return (f"| {label} | {a['positive']} vs {a['negative']} | {a['records']} | {_pct(a.get('coverage'))} | "
            f"{_r3(a[f'auroc_{score}'])} | {_r3(a['auroc_clip'])} | {delta} | {'yes' if a['excludes_zero'] else 'no'} |")


def render_fused(fused: dict[str, Any]) -> list[str]:
    """The fused score's section of report C."""
    m, t = fused["model"], fused["threshold"]
    head = ("| corpus | AUROC of | records | decided (of the pair) | fused | CLIP | Δ fused − CLIP [95%] | "
            "excludes 0 |\n| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    beats = [k for k, ok in fused["beats_clip"].items() if ok]
    lines = ["## The system score: CLIP and the consistency score, fused", "",
             f"**{'It beats CLIP on ' + ', '.join(beats) + '.' if beats else 'It does not beat CLIP: no interval excludes 0 in its favour.'}**",
             "", f"One logistic regression on [{', '.join(m['features'])}], fitted on {m['fitted_on']} "
             f"({m['records']['true']} true, {m['records']['out_of_context']} out-of-context; classes balanced, "
             f"L2 {m['l2']} on standardised features; {m['iterations']} Newton steps, converged "
             f"{m['converged']}). Coefficients per raw unit: "
             + ", ".join(f"{f} {c:+.3f}" for f, c in zip(m["features"], m["coef_raw_units"]))
             + f"; intercept on the standardised scale {m['intercept']:+.3f}. Where the block abstains the "
             "fused score is a function of CLIP's similarity alone.", "",
             f"Train threshold {_r3(t['threshold'])} (balanced accuracy {_r3(t['balanced_accuracy'])}). Train "
             f"AUROC, in sample: fused {_r3(fused['train_auroc']['fused'])}, CLIP "
             f"{_r3(fused['train_auroc']['clip'])}.", "", head]
    for ds, e in fused["corpora"].items():
        for a in e["auroc"]:
            lines.append(_auroc_row("M4FC val" if ds == "m4fc" else "VERITE", e, a, "fused"))
    lines += ["", "On the records the block decided (where the fused score can differ from CLIP's ranking):", "",
              head]
    for ds, e in fused["corpora"].items():
        for a in e["auroc_decided"]:
            lines.append(_auroc_row("M4FC val" if ds == "m4fc" else "VERITE", e, a, "fused"))
    lines += ["", "Per class, at each score's own M4FC-train threshold:", "",
              "| corpus | class | records | fused flags | of them on similarity alone | CLIP flags |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for ds, e in fused["corpora"].items():
        for c, cell in e["by_class"].items():
            lines.append(f"| {'M4FC val' if ds == 'm4fc' else 'VERITE'} | {c} | {cell['fused']['records']} | "
                         f"{_pct(cell['fused']['flagged'])} | {_pct(cell['fused_flags_on_similarity_alone'])} | "
                         f"{_pct(cell['clip']['flagged'])} |")
    return lines + [""]


def render_c(report: dict[str, Any]) -> str:
    """Report C for a person to read; the JSON beside it is the record."""
    corp = report["corpora"]
    head = ("| score | AUROC of | records | decided (of the pair) | system | CLIP | Δ system − CLIP [95%] | "
            "excludes 0 |\n| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    lines = ["# Report C: end to end, against CLIP", "",
             "Local. `data/reports/consistency_c.json` is the record. The decisions it is built under "
             "(the person's, 2026-11-10):", ""]
    lines += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in report["decisions"].items()]
    lines += ["", "Every AUROC is a paired cluster bootstrap against CLIP (2,000 resamples; VERITE by "
              "triplet). A record no decided item scores abstains at 0; coverage is the share decided. "
              "Thresholds are fitted on M4FC train only.", ""]
    if report.get("fused"):
        lines += render_fused(report["fused"])
    lines += ["# The consistency score alone (Gate C: accepted as measured)", ""]
    for claims, what in (("extracted", "claims as the extractor reads them"),
                         ("gold", "M4FC's own claimed values (a diagnostic: separates the extractor's error)")):
        keys = [(f"{ds}:{claims}:{s}", ds, s) for s in SCORE_SETS for ds in ("m4fc", "verite")
                if f"{ds}:{claims}:{s}" in corp]
        if not keys:
            continue
        lines += [f"## AUROC against CLIP: {what}", "", "M4FC is val; VERITE is its 877 evaluation records.", "",
                  head]
        for k, ds, s in keys:
            for a in corp[k]["auroc"]:
                lines.append(_auroc_row(f"{s} ({'M4FC val' if ds == 'm4fc' else 'VERITE'})", corp[k], a))
        lines += ["", "On the records the system decided (coverage-matched: CLIP on the same records):", "", head]
        for k, ds, s in keys:
            for a in corp[k]["auroc_decided"]:
                lines.append(_auroc_row(f"{s} ({'M4FC val' if ds == 'm4fc' else 'VERITE'})", corp[k], a))
        lines.append("")
    lines += ["## Per class, at the M4FC-train threshold (extracted claims)", ""]
    for s in SCORE_SETS:
        t = report["thresholds"].get(f"extracted:{s}", {})
        lines += [f"**{s}**: system threshold {_r3((t.get('system') or {}).get('threshold'))} (train balanced "
                  f"accuracy {_r3((t.get('system') or {}).get('balanced_accuracy'))}); CLIP's "
                  f"{_r3((t.get('clip') or {}).get('threshold'))} (as minus similarity).", "",
                  "| corpus | class | records | system flags | system abstains | CLIP flags |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for ds in ("m4fc", "verite"):
            e = corp.get(f"{ds}:extracted:{s}")
            for c, cell in (e or {}).get("by_class", {}).items():
                lines.append(f"| {'M4FC val' if ds == 'm4fc' else 'VERITE'} | {c} | {cell['system']['records']} | "
                             f"{_pct(cell['system']['flagged'])} | {_pct(cell['system']['abstained'])} | "
                             f"{_pct(cell['clip']['flagged'])} |")
        lines.append("")
    lines += ["## Which evidence the decided items had (primary, extracted claims)", "",
              "| corpus | evidence | records | system flags | CLIP flags |", "| --- | --- | ---: | ---: | ---: |"]
    for ds in ("m4fc", "verite"):
        for cell, c in corp.get(f"{ds}:extracted:primary", {}).get("by_evidence", {}).items():
            lines.append(f"| {'M4FC val' if ds == 'm4fc' else 'VERITE'} | {cell} | {c['system']['records']} | "
                         f"{_pct(c['system']['flagged'])} | {_pct(c['clip']['flagged'])} |")
    lines += ["", "## Items: verdicts by class (extracted claims)", "",
              "| corpus | item | class | records | claim has it | verdicts | why |", "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for corpus, by_item in report["items"].items():
        if not corpus.endswith(":extracted"):
            continue
        for item, by_class in by_item.items():
            for c, cell in by_class.items():
                lines.append(f"| {corpus.split(':')[0]} | {item} | {c} | {cell['records']} | {cell['claimed']} | "
                             f"{json.dumps(cell['verdict'])} | {json.dumps(cell['why'])} |")
    eff = report["image_carries_claim"]
    lines += ["", f"## image_carries_claim ({eff.get('rule', 'narrow')}), measured on M4FC train (extracted claims)",
              "", f"Verdicts it made unsupported, of {eff['train']['records']} train records: "
              f"`{json.dumps(eff['train']['changed'])}`", "",
              f"Of those, what widening it at Gate C added: `{json.dumps(eff['train'].get('widening_changed', {}))}`",
              "", "| score | AUROC widened | narrow | without | balanced accuracy widened | narrow | without | "
              "abstains widened | narrow | without |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s, c in eff["train"]["scores"].items():
        nar = c.get("narrow") or {}
        lines.append(f"| {s} | {_r3(c['with']['auroc'])} | {_r3(nar.get('auroc'))} | {_r3(c['without']['auroc'])} | "
                     f"{_r3(c['with']['balanced_accuracy'])} | {_r3(nar.get('balanced_accuracy'))} | "
                     f"{_r3(c['without']['balanced_accuracy'])} | {_pct(c['with']['abstained'])} | "
                     f"{_pct(nar.get('abstained'))} | {_pct(c['without']['abstained'])} |")
    lines += ["", "Applied unchanged to val and VERITE:", ""]
    for k, v in eff["applied_unchanged"].items():
        lines.append(f"- {k}: {v['records']} records; changed `{json.dumps(v['changed'])}`; by widening "
                     f"`{json.dumps(v.get('widening_changed', {}))}`; primary AUROC (widened / narrow / without) "
                     + "; ".join(f"{p}: {_r3(a['with'])} / {_r3(a['narrow'])} / {_r3(a['without'])}"
                                 for p, a in (v.get("primary_auroc") or {}).items()))
    lines.append("")
    lines += ["## The people check", "",
              f"Decisions recorded: {report['people_check']['decisions']}. Before the check no other_person "
              "contradiction counts; after it, only the person's \"contradicts\" does.", "",
              "| corpus | score | flagged before | flagged after |", "| --- | --- | ---: | ---: |"]
    for k, e in corp.items():
        if ":extracted:" in k:
            lines.append(f"| {k.split(':')[0]} | {k.split(':')[2]} | {_pct(e['people_check']['before']['flagged'])} | "
                         f"{_pct(e['people_check']['after']['flagged'])} |")
    ex = report["extraction"]
    lines += ["", f"## Extraction, read by parser {ex['parser']}", ""]
    for ds, d in ex["datasets"].items():
        lines.append(f"- {ds}: {d['rows']} claims; as run {json.dumps(d['as_run'])}, as read "
                     f"{json.dumps(d['as_read'])}; {len(d['changed'])} items changed: "
                     + ", ".join(f"{c['as_run']} -> {c['as_read']}" for c in d["changed"]))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# stage C.5: the CLIP baseline -- cosine of the image and the claim
# --------------------------------------------------------------------------

C_CLIP = C_DIR / "clip.jsonl"
CLIP_MODEL = "openai/clip-vit-base-patch32"
#: The local cache's main, pinned: the earlier VERITE run had no revision.
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_BATCH = 32
#: How far a live similarity may sit from the batch's: the live path embeds
#: one image at a time, the batch 32, and a batch changes the float order.
CLIP_TOLERANCE = 1e-4


def record_key(dataset: str, row: dict[str, Any]) -> str:
    """A record's key in C's files: a hash of its dataset and record id, so a
    file names no record (M4FC's ids are label-bearing slugs)."""
    return neutral_key("r", f"{dataset}:{row['record_id']}")


def c_records(sha_of: Callable[[str], str | None] | None = None) -> list[dict[str, Any]]:
    """Every record C scores: M4FC train+val and VERITE's 877, each with its
    key, class, image, claim and the claim's key. Paths stay in memory."""
    sha_of = sha_of or sha_lookup()
    out = []
    for ds, rows_ in (("m4fc", rows("m4fc", M4FC_SPLITS)), ("verite", evaluation_rows())):
        for row in rows_:
            path = row["image_paths"][0]
            out.append({"key": record_key(ds, row), "dataset": ds, "split": row["split"],
                        "class": row["meta"]["verdict"] if ds == "m4fc" else row["label"],
                        "sha256": sha_of(path), "path": path, "claim": row["text"] or "",
                        "claim_key": claim_key(row["text"] or ""), "row": row})
    return out


class ClipScorer:
    """CLIP ViT-B/32 at the pinned revision, on the CPU, offline."""

    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        torch.manual_seed(SEED)
        self.torch = torch
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION,
                                               local_files_only=True).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION, local_files_only=True)

    def config(self) -> dict[str, Any]:
        from scripts.image_text import software_versions

        return {"model": CLIP_MODEL, "revision": CLIP_REVISION, "device": "cpu",
                "image": "match_images.clip_load (short side >= 448, then the processor's 224)",
                "text": "the claim, cut at CLIP's 77 tokens", "batch": CLIP_BATCH,
                "software": software_versions(("torch", "transformers", "pillow"))}

    def _norm(self, out):
        from scripts.clip_pairs import _features

        f = _features(out)
        return (f / f.norm(dim=-1, keepdim=True)).float().numpy()

    def images(self, paths: Sequence[str]):
        from scripts.match_images import clip_load

        with self.torch.inference_mode():
            return self._norm(self.model.get_image_features(
                **self.proc(images=[clip_load(p) for p in paths], return_tensors="pt")))

    def texts(self, texts: Sequence[str]):
        with self.torch.inference_mode():
            inputs = self.proc(text=list(texts), return_tensors="pt", padding=True, truncation=True,
                               max_length=77)
            return self._norm(self.model.get_text_features(input_ids=inputs["input_ids"],
                                                           attention_mask=inputs["attention_mask"]))


def clip_similarities(records: Sequence[dict[str, Any]], scorer, out_path: Path, cid: str) -> float:
    """Each record's image-claim cosine, one ledger row each (resumable), each
    distinct image and claim embedded once; the seconds it took. An image that
    cannot be read is a row with an error, never a similarity."""
    from scripts import gallery as gl

    import numpy as np

    started = time.perf_counter()
    todo = list(records)
    shas = list(dict.fromkeys(r["sha256"] for r in todo))
    path_of = {r["sha256"]: r["path"] for r in todo}
    claims = list(dict.fromkeys(r["claim_key"] for r in todo))
    text_of = {r["claim_key"]: r["claim"] for r in todo}
    img_vec, txt_vec, errors = {}, {}, {}
    for i in range(0, len(shas), CLIP_BATCH):
        part = shas[i:i + CLIP_BATCH]
        try:
            vecs = scorer.images([path_of[s] for s in part])
            img_vec.update(zip(part, vecs))
        except Exception:  # noqa: BLE001 -- one at a time, to name the image that fails
            for s in part:
                try:
                    img_vec[s] = scorer.images([path_of[s]])[0]
                except Exception as exc:  # noqa: BLE001
                    errors[s] = type(exc).__name__
        n = min(i + CLIP_BATCH, len(shas))
        per = (time.perf_counter() - started) / n
        print(f"  [clip] {n:,}/{len(shas):,} images ok {per:.2f}s/image eta "
              f"{per * (len(shas) - n) / 3600:.2f}h", flush=True)
    for i in range(0, len(claims), 64):
        part = claims[i:i + 64]
        txt_vec.update(zip(part, scorer.texts([text_of[c] for c in part])))
    ledger = gl.JsonlLedger(out_path)
    ledger.open()
    try:
        ledger.append_many([{"key": r["key"], "config_id": cid, "dataset": r["dataset"],
                             "error": errors.get(r["sha256"]),
                             "similarity": None if r["sha256"] in errors else
                             round(float(np.dot(img_vec[r["sha256"]], txt_vec[r["claim_key"]])), 6)}
                            for r in todo])
    finally:
        ledger.close()
    return time.perf_counter() - started


def cmd_clip(args) -> int:
    """C.5's baseline: every record's image-claim cosine. With --sample N, N
    records spread over what is left, timed, and the rest projected."""
    from scripts import gallery as gl

    t = time.perf_counter()
    scorer = ClipScorer()
    overhead = time.perf_counter() - t
    config = scorer.config()
    cid = hashlib.sha256(canonical(config)).hexdigest()[:12]
    cfg = C_CLIP.with_name("clip.config.json")
    if cfg.is_file():
        if json.loads(cfg.read_text(encoding="utf-8"))["config_id"] != cid:
            raise SystemExit(f"{rel(cfg)} is another CLIP configuration: refusing to mix runs")
    else:
        write_json(cfg, {"config_id": cid, **config})
    records = c_records()
    done = {r["key"] for r in gl.read_jsonl(C_CLIP)} if C_CLIP.is_file() else set()
    todo = [r for r in records if r["key"] not in done]
    print(f"clip, config {cid}: {len(records):,} records, {len(records) - len(todo):,} done, "
          f"{len(todo):,} to do", flush=True)
    picked = spread(todo, args.sample) if args.sample else todo
    if picked:
        seconds = clip_similarities(picked, scorer, C_CLIP, cid)
        if args.sample:
            _print_projection(projection(seconds, len(picked), len(todo) - len(picked), overhead),
                              "record", "records")
    held = list(gl.read_jsonl(C_CLIP)) if C_CLIP.is_file() else []
    errors = ordered(Counter(r["error"] for r in held if r["error"]))
    print(f"  ledger: {len(held):,} of {len(records):,} records; errors "
          f"{json.dumps(errors) if errors else 'none'}")
    return 0


# --------------------------------------------------------------------------
# stage D: the demonstration -- a record's verdicts, explained from its fields
# --------------------------------------------------------------------------

DEMO_DIR = C_DIR / "demo"
ITEM_NOUNS = {"date": "date", "location": "place", "event": "event", "people": "people"}
#: Why the gallery cannot check a claimed name, in words.
GALLERY_CANNOT = {"name_not_in_gallery": "the name is not in the gallery",
                  "ambiguous_name": "the name is more than one person's",
                  "no_usable_reference": "the gallery has no usable reference photograph of them",
                  "no_face_in_image": "no face in the image is large enough to check",
                  "no_usable_face": "the face that best matches is too small to score",
                  "faces_error": "the image's faces could not be read",
                  "not_verified": "the name was not verified"}
_SAID = "The claim's {noun} is {claimed}; the image's evidence says {predicted}{sources}."
#: One template per verdict's why. A sentence is a template filled with the
#: verdict record's own fields and nothing else: no model writes any of it.
EXPLAIN = {
    "date_disjoint": "The claim dates the photograph {claimed}; the image's evidence dates it {predicted}{sources}. "
                     "These dates cannot both be true.",
    "date_overlap": "The claim dates the photograph {claimed}; the image's evidence dates it {predicted}{sources}. "
                    "These dates can both be true.",
    "judge_no": _SAID + " The comparator finds they cannot both be true (P = {score}).",
    "judge_yes": _SAID + " The comparator finds they can both be true (P of a contradiction = {score}).",
    "comparator_unclear": _SAID + " The comparator cannot tell whether both can be true, so this is unsupported.",
    "comparator_not_run": _SAID + " The comparator has not been run on this pair, so this is unsupported.",
    "gate_a_not_cleared": _SAID + " {outcome} This item has not cleared Gate A, so it counts as unsupported.",
    "gate_b_left_out": "The claim dates the photograph {claimed}; the image's evidence dates it {predicted}{sources}. "
                       "{outcome} Gate B left dates out of the primary score, so this counts as unsupported.",
    "date_not_grounded": "The claim dates the photograph {claimed}; the image's evidence dates it {predicted}"
                         "{sources}, but none of the evidence it cites writes that year, so this is unsupported.",
    "image_carries_claim": _SAID + " That evidence is text in the image that already says what the claim says: "
                                   "the image carries the claim, so it is no independent evidence and this is "
                                   "unsupported.",
    "no_image_evidence": "The claim's {noun} is {claimed}; nothing known about the image says anything of its "
                         "{noun}, so this is unsupported.",
    "prediction_refused": "The claim's {noun} is {claimed}; the answer about the image failed its checks, so this "
                          "is unsupported.",
    "claimed_person_accepted": "The claim names {claimed}; a face in the image matches the gallery's reference "
                               "photographs of {predicted}{sources}.",
    "gallery_reject": "The claim names {claimed}; the gallery does not accept that name on this image. A reject is "
                      "not evidence that the person is absent, so this is unsupported.",
    "other_person_unchecked": "The claim names {claimed}; a different person was recognised: {predicted}{sources}. "
                              "Not yet checked by a person, so this is unsupported.",
    "other_person_not_confirmed": "The claim names {claimed}; a different person was recognised: {predicted}"
                                  "{sources}. A person checked it and did not confirm a contradiction, so this is "
                                  "unsupported.",
    "other_person_confirmed": "The claim names {claimed}; the person in the image is {predicted}{sources}, and a "
                              "person has confirmed it. The claim's people are contradicted.",
    "gallery_cannot": "The claim names {claimed}; the gallery cannot check it here ({cannot}), so this is "
                      "unsupported.",
}
OUTCOME_WORDS = {CONTRADICTS: "The {comparator} finds they cannot both be true.",
                 CONSISTENT: "The {comparator} finds they can both be true.",
                 UNCLEAR: "The {comparator} cannot tell."}


def _source(e: dict[str, Any]) -> str:
    kind = e.get("kind")
    if kind == "document":
        return f"{e['host']}, {e['published_at']}"
    if kind == "gallery":
        return f"gallery score {e['score']:.2f}" if e.get("score") is not None else "the gallery"
    if kind == "entity":
        return f"the image model read \"{e['name']}\""
    return {"ocr": "text read in the image", "description": "the image's description",
            "person": "a gallery-verified face"}.get(kind, kind or "")


def explain(v: dict[str, Any]) -> str:
    """One plain sentence for one verdict, from its template and the record's
    own fields. A why with no template refuses: no sentence is ever free text."""
    why = v["why"].split(":", 1)[0]
    if why in GALLERY_CANNOT:
        why = "gallery_cannot"
    if why not in EXPLAIN:
        raise KeyError(f"no explanation template for {v['why']!r}")
    sources = "; ".join(dict.fromkeys(s for s in (_source(e) for e in v["evidence"]) if s))  # each source once
    outcome = v.get("outcome") or {}
    return EXPLAIN[why].format(
        noun=ITEM_NOUNS[v["item"]], claimed=v["claimed"], predicted=v["predicted"],
        sources=f" (source: {sources})" if sources else "",
        score="–" if v.get("score") is None else f"{v['score']:.2f}",
        cannot=GALLERY_CANNOT.get(v["why"], ""),
        outcome=OUTCOME_WORDS[outcome["label"]].format(comparator=v["comparator"]) if outcome else "")


#: The fused flag's basis, in words: a template per case, filled with the
#: flag's own numbers. Nothing about why comes from the regression; the why is
#: the verdicts'.
FUSED_BASIS = {
    "decided": "The flag combines the image–text similarity ({similarity}) with the consistency verdicts above "
               "(consistency score {consistency}).",
    "abstained": "No item is decided, so the flag rests on image–text similarity alone ({similarity}).",
}
FUSED_FLAG = {True: "Flagged: fused score {score} is at or above the M4FC-train threshold {threshold}.",
              False: "Not flagged: fused score {score} is below the M4FC-train threshold {threshold}."}


def read_fused() -> dict[str, Any] | None:
    """The fused score's regression and threshold, as report C fitted them."""
    return json.loads(C_FUSED.read_text(encoding="utf-8")) if C_FUSED.is_file() else None


def fused_flag(fitted: dict[str, Any], similarity: float, score: dict[str, Any]) -> dict[str, Any]:
    """One record's fused score and flag, and their sentences, from CLIP's
    similarity and the record's primary score."""
    value = fused_score(fitted["model"], fused_features(similarity, score["score"], score["abstained"]))
    t = fitted["threshold"]["threshold"]
    flagged = t is not None and value >= t
    words = {"similarity": f"{similarity:.3f}", "consistency": f"{score['score']:.2f}", "score": f"{value:.3f}",
             "threshold": "–" if t is None else f"{t:.3f}"}
    return {"similarity": similarity, "consistency_score": score["score"], "decided": not score["abstained"],
            "score": value, "threshold": t, "flagged": flagged,
            "flag": FUSED_FLAG[flagged].format(**words),
            "basis": FUSED_BASIS["abstained" if score["abstained"] else "decided"].format(**words)}


def demo_bundle(key: str, image_path: str | None, image_sha: str | None, claim: str, date_given: str | None,
                bound: str | None, items: dict[str, dict[str, Any] | None], notes: Sequence[str] = (),
                variants: dict[str, dict[str, Any] | None] | None = None, similarity: float | None = None,
                fused: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a demo page shows: the claim, its image, the system's flag (the
    fused score, where CLIP's similarity and the fitted regression are given),
    each item's verdict and explanation, the record's primary score, each
    variant's item and score (labelled as variants), and how evidence was
    bounded."""
    kept = {i: v for i, v in items.items() if v}
    var = {i: v for i, v in (variants or {}).items() if v}
    bundle = {"key": key, "image": {"path": image_path, "sha256": image_sha}, "claim": claim,
              "date": date_given, "retrieval": {"bound": bound}, "items": kept,
              "explanations": {i: explain(v) for i, v in kept.items()}, "score": record_score(kept.values()),
              "notes": list(notes)}
    if similarity is not None and fused is not None:
        bundle["fused"] = fused_flag(fused, similarity, bundle["score"])
    if variants is not None:
        bundle["variants"] = var
        bundle["variant_explanations"] = {i: explain(v) for i, v in var.items()}
        bundle["variant_scores"] = {s: record_score(counted_items(kept, var, s).values())
                                    for s in SCORE_SETS if s in VARIANT_OF}
    return bundle


DEMO_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Consistency check</title>
<style>
:root { --bg:#f6f6f3; --surface:#fff; --text:#1c1c1a; --muted:#66655f; --line:#deddd7;
        --bad:#9b2c1f; --good:#22663a; --unsure:#6b5a12; }
@media (prefers-color-scheme: dark) { :root { --bg:#141413; --surface:#1e1e1c; --text:#ebeae5; --muted:#a3a29b;
        --line:#3a3935; --bad:#f0907f; --good:#8fd3a4; --unsure:#e3cf7e; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 8px 0 4px; } h2 { font-size: 16px; margin: 20px 0 8px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }
.muted { color: var(--muted); font-size: 13px; }
figure { margin: 0; text-align: center; } img { max-width: 100%; max-height: 60vh; border-radius: 6px; }
.claim { font-size: 17px; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; } th, td { text-align: left; padding: 8px 6px;
        border-top: 1px solid var(--line); vertical-align: top; overflow-wrap: anywhere; }
th { color: var(--muted); font-weight: 600; font-size: 13px; }
.v-contradicted { color: var(--bad); font-weight: 600; } .v-consistent { color: var(--good); font-weight: 600; }
.v-unsupported { color: var(--unsure); font-weight: 600; }
ul { padding-left: 20px; } li { margin: 6px 0; }
@media (max-width: 640px) { th:nth-child(5), td:nth-child(5) { display: none; } }
</style></head><body><main>
__BODY__
</main></body></html>
"""


def demo_html(bundle: dict[str, Any], image_body: bytes | None) -> str:
    """The bundle as a static local page: no script, nothing fetched."""
    import base64

    esc = html.escape
    rows_html, sources = [], []
    for item, v in bundle["items"].items():
        score = "–" if v["score"] is None else f"{v['score']:.2f}"
        rows_html.append(
            f"<tr><td>{esc(ITEM_NOUNS[item])}</td><td>{esc(v['claimed'])}</td><td>{esc(v['predicted'] or '–')}</td>"
            f"<td class=\"v-{v['verdict']}\">{esc(v['verdict'])}</td><td>{score}</td>"
            f"<td>{esc(v['reason_code'])}</td></tr>")
        sources += [e for e in v["evidence"] if e.get("kind") == "document"]
    s = bundle["score"]
    parts = ["<h1>Does the claim fit its image?</h1>",
             "<p class=\"muted\">Each verdict comes from the pipeline's own outputs; each sentence below is a "
             "template filled from them. No model wrote any of this text.</p>"]
    if image_body:
        mime = "image/png" if image_body[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        parts.append(f"<div class=\"card\"><figure><img alt=\"the claim's image\" src=\"data:{mime};base64,"
                     f"{base64.b64encode(image_body).decode('ascii')}\"></figure></div>")
    parts.append(f"<div class=\"card\"><div class=\"muted\">Claim</div><div class=\"claim\">{esc(bundle['claim'])}"
                 "</div>")
    bound = bundle["retrieval"]["bound"]
    parts.append(f"<p class=\"muted\">{'Documents were retrieved from before ' + esc(bound) + '.' if bound else 'No date bounds this claim, so no documents were retrieved: evidence comes from the image alone (the as-of rule).'}</p></div>")
    if sysf := bundle.get("system"):  # the ensemble's flag (scripts/ensemble.py fills it from templates)
        parts.append(f"<div class=\"card\"><h2>The system's flag</h2><p class=\"{'v-contradicted' if sysf['flagged'] else 'v-consistent'}\">"
                     f"{esc(sysf['flag'])}</p><p>{esc(sysf['basis'])}</p><p class=\"muted\">{esc(sysf['note'])}</p></div>")
    if ts := bundle.get("text_side"):
        parts.append(f"<div class=\"card\"><h2>Text-side check</h2><p>{esc(ts['sentence'])}</p>"
                     f"<p class=\"muted\">{esc(ts['note'])}</p></div>")
    if (f := bundle.get("fused")) and not bundle.get("system"):
        parts.append(f"<div class=\"card\"><h2>The system's flag</h2><p class=\"{'v-contradicted' if f['flagged'] else 'v-consistent'}\">"
                     f"{esc(f['flag'])}</p><p>{esc(f['basis'])}</p><p class=\"muted\">The fused score is one logistic "
                     "regression on the image–text similarity and the consistency score below, fitted on M4FC "
                     "train. It decides the flag; the reasons below come from the consistency verdicts alone.</p>"
                     "</div>")
    parts.append("<div class=\"card\"><h2>Per item</h2><table><thead><tr><th>item</th><th>the claim says</th>"
                 "<th>the image's evidence says</th><th>verdict</th><th>P(contradiction)</th><th>reason code</th>"
                 "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>")
    parts.append(f"<p>Mismatch score <strong>{s['score']:.2f}</strong>"
                 + (" (no item decided: the record abstains)" if s["abstained"] else f", from {esc(s['by'])}")
                 + ".</p></div>")
    parts.append("<div class=\"card\"><h2>Why</h2><ul>"
                 + "".join(f"<li>{esc(t)}</li>" for t in bundle["explanations"].values()) + "</ul></div>")
    if bundle.get("variants") is not None:
        names = {"with_event": "with event", "with_grounded_date": "with dates grounded in their evidence"}
        rows_v = "".join(
            f"<tr><td>{esc(ITEM_NOUNS[i])}</td><td>{esc(v['claimed'])}</td><td>{esc(v['predicted'] or '–')}</td>"
            f"<td class=\"v-{v['verdict']}\">{esc(v['verdict'])}</td>"
            f"<td>{'–' if v['score'] is None else format(v['score'], '.2f')}</td></tr>"
            for i, v in bundle["variants"].items())
        scores_v = "; ".join(f"{names[s]}: {sc['score']:.2f}" + (" (abstains)" if sc["abstained"] else "")
                             for s, sc in bundle["variant_scores"].items())
        parts.append("<div class=\"card\"><h2>Variants, not the score</h2><p class=\"muted\">Each puts one item "
                     "back: the event, which did not clear Gate A, as the comparator decides it; and the date, "
                     "only where its year is written in the evidence it cites.</p>"
                     + (f"<table><thead><tr><th>item</th><th>the claim says</th><th>the image's evidence says</th>"
                        f"<th>verdict</th><th>P(contradiction)</th></tr></thead><tbody>{rows_v}</tbody></table>"
                        if rows_v else "<p class=\"muted\">Neither item is in the claim.</p>")
                     + f"<p>Variant scores: {esc(scores_v)}.</p><ul>"
                     + "".join(f"<li>{esc(t)}</li>" for t in bundle["variant_explanations"].values())
                     + "</ul></div>")
    if sources:
        seen = {}
        for e in sources:
            seen.setdefault(e.get("doc_id"), e)
        parts.append("<div class=\"card\"><h2>Sources</h2><ul>" + "".join(
            f"<li>{esc(e['host'])}, {esc(e['published_at'])}: {esc(e.get('title') or '')}</li>"
            for e in seen.values()) + "</ul></div>")
    if bundle["notes"]:
        parts.append("<div class=\"card muted\">" + "".join(f"<p>{esc(n)}</p>" for n in bundle["notes"]) + "</div>")
    return DEMO_PAGE.replace("__BODY__", "\n".join(parts))


def write_demo(bundle: dict[str, Any], forms: Iterable[str] = (), out_dir: Path | None = None) -> Path:
    """The bundle's JSON and page, refused if the page shows an id form (or a
    path, or a VERITE file name) that its own content does not explain."""
    out_dir = out_dir or DEMO_DIR
    body = None
    if bundle["image"]["path"]:
        from scripts.image_text import read_bytes

        body = read_bytes(bundle["image"]["path"])
    page = demo_html(bundle, body)
    shown = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", page)
    shown_items = [*bundle["items"].values(), *(bundle.get("variants") or {}).values()]
    content = "\n".join([bundle["claim"], *(f"{v['claimed']}\n{v['predicted'] or ''}\n"
                                            + "\n".join(str(e.get(k) or '') for e in v["evidence"]
                                                        for k in ("host", "title", "name"))
                                            for v in shown_items)])
    refuse_ids(shown, forms, explained=content, what="a demo page")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{bundle['key']}.json", bundle)
    path = out_dir / f"{bundle['key']}.html"
    path.write_text(page, encoding="utf-8", newline="\n")
    return path


def rehearsal_bundles() -> list[tuple[dict[str, Any], frozenset[str]]]:
    """Demo pages from outputs that exist now: the third pilot's predictions
    for its 30 M4FC train images, against each record's own claimed values
    (the gold-claim variant), dates by the rule, people by the gallery. The
    judge has not compared places or events for C, so those say so. Train
    rows: a rehearsal of the page, not the meeting's examples."""
    from scripts import gallery as gl
    from scripts.m4fc_audit import present, shown

    inp = B_PREDICT_PILOT
    out = inp.with_name("predict_pilot3_out.jsonl")
    keys = json.loads(inp.with_name(inp.stem + ".keys.json").read_text(encoding="utf-8"))
    ins = {json.loads(line)["key"]: json.loads(line) for line in inp.read_text(encoding="utf-8").splitlines()
           if line}
    results = {json.loads(line)["key"]: json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
               if line}
    sha_of = sha_lookup()
    by_sha = {sha_of(r["image_paths"][0]): r for r in rows("m4fc", M4FC_SPLITS)}
    _cid, _items, held = retrieval_ledger(M4FC_SPLITS, sha_of)
    titles = {h["doc_id"]: h["title"] for r in held.values() for h in r["hits"]}
    people = list(gl.read_jsonl(B_PEOPLE))
    verified = {(r["sha256"], r["name"]): r for r in people if r["dataset"] == "m4fc" and "claim" in r["sides"]}
    vlm = defaultdict(list)
    for r in people:
        if r["dataset"] == "m4fc" and {"vlm_free", "vlm_forced"} & set(r["sides"]):
            vlm[r["sha256"]].append(r)
    spans = claim_ner("m4fc")
    cleared = gate_cleared()
    decisions = other_person_decisions()
    bundles = []
    for key, name in sorted(keys.items(), key=lambda kv: kv[1]):
        sha = name.split(":")[1]
        row, res = by_sha[sha], results[key]
        meta, evidence = row["meta"], ins[key]["evidence"]
        pred, answered = predicted(res), res["result"]["items"] or {}
        items = {}
        for item in PREDICT_ITEMS:
            claimed = shown(meta[f"claimed_{item}"]) if present(meta.get(f"claimed_{item}")) else None
            cited = (answered.get(item) or {}).get("evidence", [])
            ev = [{**evidence[e], **({"title": titles.get(evidence[e]["doc_id"])}
                                     if evidence[e]["kind"] == "document" else {})} for e in cited if e in evidence]
            items[item] = item_verdict(item, claimed, pred[item], cleared.get(item, False), None, ev)
        names = claim_people(spans.get(str(row["record_id"])) or {"spans": []})
        items["people"] = people_verdict(sha, names, verified, vlm.get(sha, []), decisions)
        bound = retrieval_items_bound(meta)
        bundles.append((demo_bundle(neutral_key("d", f"m4fc:{row['record_id']}"), row["image_paths"][0], sha,
                                    row["text"] or "", None, bound, items,
                                    ["Rehearsal on M4FC train, from the third predictor pilot. Claimed values are "
                                     "M4FC's own annotation (the gold-claim variant), not yet the extractor's.",
                                     "Gate A waits for the person's review of 46 contested true-row pairs: until "
                                     "then an item it has not cleared counts as unsupported."]),
                        id_forms(row)))
    return bundles


def retrieval_items_bound(meta: dict[str, Any]) -> str | None:
    """The date M4FC's evidence is bounded by, or None."""
    from scripts import retrieve

    try:
        return retrieve.evidence_bound("m4fc", meta)[0]
    except retrieve.RetrievalError:
        return None


def cmd_demo(args) -> int:
    if args.rehearse:
        written = [write_demo(b, forms, DEMO_DIR / "rehearsal") for b, forms in rehearsal_bundles()]
        print(f"{len(written)} rehearsal pages -> {rel(DEMO_DIR / 'rehearsal')} (local; open any .html)")
        return 0
    if args.record:
        bundle = json.loads(Path(args.record).read_text(encoding="utf-8"))
        print(f"-> {rel(write_demo(bundle))}")
        return 0
    for source in (C_VERDICTS, *sorted((DEMO_DIR / "rehearsal").glob("*.json"))):
        if not source.is_file():
            continue
        found = None
        if source == C_VERDICTS:
            row = next((json.loads(l) for l in source.read_text(encoding="utf-8").splitlines()
                        if l and json.loads(l)["key"] == args.from_run
                        and json.loads(l)["variant"] == "extracted"), None)
            rec = next((r for r in c_records() if r["key"] == args.from_run), None) if row else None
            if rec:
                bound = retrieval_items_bound(rec["row"]["meta"]) if rec["dataset"] == "m4fc" else None
                sim = next((json.loads(l)["similarity"] for l in C_CLIP.read_text(encoding="utf-8").splitlines()
                            if l and json.loads(l)["key"] == rec["key"]), None) if C_CLIP.is_file() else None
                found = demo_bundle(rec["key"], rec["path"], rec["sha256"], rec["claim"], None, bound, row["items"],
                                    variants=row.get("variants"), similarity=sim, fused=read_fused())
        elif source.stem == args.from_run:
            found = json.loads(source.read_text(encoding="utf-8"))
        if found:
            if found.get("fused"):
                print(f"  flag: {found['fused']['flag']} {found['fused']['basis']}")
            for item, text in found["explanations"].items():
                print(f"  {item}: {found['items'][item]['verdict']} -- {text}")
            print(f"-> {rel(write_demo(found))}")
            return 0
    raise SystemExit(f"no record {args.from_run!r} in {rel(C_VERDICTS)} or the rehearsal pages")


# --------------------------------------------------------------------------
# stage C.3: the person's check of the other_person contradictions
# --------------------------------------------------------------------------

OTHER_PERSON_PAGE = OTHER_PERSON_DECISIONS.with_name("other_person.html")
OTHER_PERSON_QUESTION = (
    "The claim names the people listed. The gallery matched the image's only face to another person, at its "
    "certified threshold. Look at the face (box red; eyes green, nose yellow, mouth corners blue) and the "
    "gallery's reference. Answer \"contradicts\" only if the face is that person, and so none of the people "
    "the claim names. \"Not a contradiction\" if it is one of them, or the match is wrong. \"Unsure\" "
    "otherwise. Only \"contradicts\" makes the claim's people contradicted; anything else leaves them "
    "unsupported.")
OTHER_PERSON_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Other-person check</title>
<style>
:root { --bg:#f6f6f3; --surface:#fff; --text:#1c1c1a; --muted:#66655f; --line:#deddd7; --accent:#2b5c8a; }
@media (prefers-color-scheme: dark) { :root { --bg:#141413; --surface:#1e1e1c; --text:#ebeae5; --muted:#a3a29b;
  --line:#3a3935; --accent:#8fb8de; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, "Segoe UI", sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 16px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }
.row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
img { max-width:100%; border-radius:6px; } .muted { color:var(--muted); font-size:13px; }
a { color: var(--accent); overflow-wrap:anywhere; } label { margin-right: 14px; white-space: nowrap; }
button { font: inherit; padding: 8px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--accent);
  color: #fff; cursor: pointer; }
</style></head><body><main>
<h1>Other-person check</h1>
<p>__QUESTION__</p>
<p class="muted">Answers stay in this browser until exported. Export writes __EXPORT__; put it in the folder that
holds this page.</p>
<div id="cases"></div>
<div class="card"><button id="export">Export __EXPORT__</button> <span id="count" class="muted"></span></div>
<script>
const DATA = __DATA__;
const store = (() => { try { return JSON.parse(localStorage.getItem(DATA.store) || "{}"); } catch (e) { return {}; } })();
const save = () => { try { localStorage.setItem(DATA.store, JSON.stringify(store)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const box = document.getElementById("cases");
DATA.cases.forEach((c, i) => {
  const div = document.createElement("div"); div.className = "card";
  div.innerHTML = `<div class="row"><img alt="the image, the scored face boxed" src="${c.whole}">
    <img alt="the scored face, landmarks drawn" src="${c.crop}"></div>
    <p><strong>${i + 1}.</strong> The claim names: <strong>${esc(c.claimed)}</strong></p>
    <p>The gallery matched the face to: <strong>${esc(c.person)}</strong> (${esc(c.qid)}), score ${c.score.toFixed(3)}.
    Reference: <a href="${esc(c.reference || "")}" rel="noreferrer">${esc(c.reference || "none")}</a>
    <span class="muted">${esc(c.licence || "")}${c.attribution ? "; " + esc(c.attribution) : ""}</span></p>
    <p>${DATA.answers.map(a => `<label><input type="radio" name="a${i}" value="${a}"> ${a.replace(/_/g, " ")}</label>`).join("")}</p>`;
  div.querySelectorAll("input").forEach(inp => {
    if (store[c.key] === inp.value) inp.checked = true;
    inp.addEventListener("change", () => { store[c.key] = inp.value; save(); count(); });
  });
  box.appendChild(div);
});
const count = () => { document.getElementById("count").textContent =
  `${DATA.cases.filter(c => store[c.key]).length} of ${DATA.cases.length} answered`; };
count();
document.getElementById("export").addEventListener("click", () => {
  const out = {}; DATA.cases.forEach(c => { out[c.key] = store[c.key] || null; });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1) + "\n"], {type: "application/json"}));
  a.download = DATA.export; a.click();
});
</script>
</main></body></html>
"""


def other_person_cases(verdicts: Iterable[dict[str, Any]], people: Iterable[dict[str, Any]]
                       ) -> list[dict[str, Any]]:
    """Every other_person contradiction the people rule raised, once per
    image, claimed names and accepted person (the key the answer is filed
    under), with the face the gallery scored."""
    rows_ = {(r["sha256"], r["name"]): r for r in people}
    out: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        p = v["items"].get("people")
        if not p or not p["why"].startswith("other_person"):
            continue
        ev = p["evidence"][0]
        key = other_person_key(v["sha256"], p["claimed"], ev["qid"])
        out.setdefault(key, {"key": key, "sha256": v["sha256"], "claimed": p["claimed"], "person": ev["name"],
                             "qid": ev["qid"], "score": ev["score"], "reference": ev["reference"],
                             "licence": ev["licence"], "attribution": ev["attribution"],
                             "face": rows_[(v["sha256"], p["predicted"])]["query_face"]})
    return [out[k] for k in sorted(out)]


def cmd_review_people() -> int:
    """C.3's page: every other_person contradiction, for the person to check.
    It shows the image and the scored face, the claim's names, the accepted
    person and the gallery reference it matched -- no label, record id, slug or
    file name -- and exports the answers as other_person_decisions.json."""
    from scripts import gallery as gl
    from scripts.image_text import read_bytes

    if not C_VERDICTS.is_file():
        raise SystemExit("run `verdicts` first")
    cases = other_person_cases(gl.read_jsonl(C_VERDICTS), gl.read_jsonl(B_PEOPLE))
    if not cases:
        print("no other_person contradiction to check: no page written")
        return 0
    records = c_records()
    path_of = {r["sha256"]: r["path"] for r in records}
    shown = []
    for c in cases:
        body = read_bytes(path_of[c["sha256"]])
        if hashlib.sha256(body).hexdigest() != c["sha256"]:
            raise SystemExit("an image's bytes are not the ones its verdict read: refusing")
        shown.append({k: v for k, v in c.items() if k not in ("face", "sha256")}
                     | {"whole": gl._whole(body, c["face"]["box"]), "crop": gl._crop(body, c["face"], gl.REVIEW_CROP)})
    data = {"cases": shown, "answers": list(OTHER_PERSON_ANSWERS), "export": OTHER_PERSON_DECISIONS.name,
            "store": "other-person-check"}
    text = json.dumps({**data, "cases": [{k: v for k, v in c.items() if k not in ("whole", "crop")}
                                         for c in shown]}, ensure_ascii=False)
    keys = {c["sha256"] for c in cases}
    forms = frozenset().union(*(id_forms(r["row"]) for r in records if r["sha256"] in keys))
    refuse_ids(text, forms, explained="\n".join(f"{c['key']}\n{c['claimed']}\n{c['person']}\n{c['reference']}\n"
                                                f"{c['licence']}\n{c['attribution']}" for c in cases),
               what="the other_person page")
    page = (OTHER_PERSON_TEMPLATE.replace("__QUESTION__", html.escape(OTHER_PERSON_QUESTION))
            .replace("__EXPORT__", OTHER_PERSON_DECISIONS.name)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    OTHER_PERSON_PAGE.parent.mkdir(parents=True, exist_ok=True)
    OTHER_PERSON_PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"{len(cases)} other_person contradictions -> {rel(OTHER_PERSON_PAGE)} (local). Its export goes to "
          f"{rel(OTHER_PERSON_DECISIONS)}")
    return 0


# --------------------------------------------------------------------------
# stage D: the live path -- a new image and claim through every stage. The
# laptop does the gallery, retrieval and the verdicts; the server the image
# model and the text model (scripts/live_demo.sh drives both).
# --------------------------------------------------------------------------

LIVE_DIR = C_DIR / "live"
#: The configurations the live path runs under: the evaluated pipeline's, so
#: a live verdict is one report C measured. Each server stage refuses before
#: it computes anything if its own is another.
LIVE_CONFIGS = {"image_text": IMAGE_TEXT_CONFIG, "predict": "44010bda9178", "extract": "d1d7aa2f42d4",
                "judge": "007d326ccf33"}
#: What an image_text row's computation is, without where it ran or for which record.
IMAGE_TEXT_PARTS = ("status", "reason", "image", "model_input", "ocr", "description", "entities", "confidence")


def live_key(sha: str, claim: str, date_given: str | None) -> str:
    return neutral_key("l", f"{sha}\n{' '.join(claim.split())}\n{date_given or ''}")


def live_keys(text: str) -> list[str]:
    keys = [k for k in re.split(r"[,\s]+", text or "") if k]
    if not keys or any(not re.fullmatch(r"l[0-9a-f]{15}", k) for k in keys):
        raise SystemExit("--keys: the request keys `live prepare` printed, comma-separated")
    return keys


def live_requests(args) -> list[dict[str, Any]]:
    """--image, --claim and --date, or a JSONL file of {image, claim, date}."""
    if args.requests:
        reqs = [json.loads(line) for line in Path(args.requests).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        reqs = [{"image": args.image, "claim": args.claim, "date": args.date}]
    for r in reqs:
        if not r.get("image") or not (r.get("claim") or "").strip():
            raise SystemExit("a live request needs an image and a claim")
        if r.get("date"):
            try:
                r["date"] = date.fromisoformat(r["date"]).isoformat()
            except ValueError:
                raise SystemExit(f"a date is YYYY-MM-DD, not {r['date']!r}") from None
        else:
            r["date"] = None
    return reqs


def _read_live(key: str, name: str) -> Any:
    path = LIVE_DIR / key / name
    if not path.is_file():
        raise SystemExit(f"{rel(path)} is not here: run the step that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_live_prepare(args) -> int:
    """The laptop's first part, per request: the image through the three
    gates and copied under a neutral name; its faces, by the gallery's own
    detector and recogniser; the claim's people, by the gallery's NER; and,
    where a date bounds the claim, retrieval strictly before it (no date, no
    retrieval). Prints the requests' keys, comma-separated, last."""
    from scripts import gallery as gl
    from scripts import images, retrieve
    from scripts.m4fc_audit import FORMAT_EXT

    backend = gl.make_backend(FACE_BACKEND)
    fcid = gl.face_config_id(backend)
    ner, con, keys = gl.Ner(), None, []
    try:
        for req in live_requests(args):
            src = Path(req["image"])
            src = src if src.is_absolute() else PROJECT_ROOT / src
            body = src.read_bytes()
            examined = images.examine_bytes(body)
            status = images.status({**examined, "path": rel(src)})
            if status != images.USABLE:
                raise SystemExit(f"{src.name}: the image is not usable evidence ({status}): refusing")
            sha = examined["sha256"]
            key = live_key(sha, req["claim"], req["date"])
            d = LIVE_DIR / key
            d.mkdir(parents=True, exist_ok=True)
            name = "image" + FORMAT_EXT.get(examined["format"] or "", ".img")
            (d / name).write_bytes(body)
            gl.embed_images(backend, [(sha, lambda b=body: b)], d / "faces.jsonl", gl.MIN_FACE["query"])
            spans = ner([gl.clean_claim(req["claim"])[0]])[0]
            retrieval = None
            if req["date"]:
                if not retrieve.words(req["claim"]):
                    retrieval = {"before": req["date"], "error": "empty_query", "hits": []}
                else:
                    con = con or retrieve.connect()
                    hits, terms, fell_back = retrieve.search_with_terms(req["claim"], req["date"], RETRIEVE_K, con=con)
                    retrieval = {"before": req["date"], "error": None, "terms": terms, "fell_back": fell_back,
                                 "config_id": hashlib.sha256(canonical(retrieval_config())).hexdigest()[:12],
                                 "hits": [hit_row(h, terms, read_doc(h.doc_id)) for h in hits]}
            write_json(d / "request.json", {"key": key, "sha256": sha, "image": name, "claim": req["claim"],
                                            "date": req["date"], "face_config": fcid, "claim_ner": spans,
                                            "claim_people": claim_people({"spans": spans}),
                                            "retrieval": retrieval})
            print(f"  [live] {key}: faces {fcid}, claim people {len(claim_people({'spans': spans}))}, "
                  f"documents {'none (no date)' if retrieval is None else len(retrieval['hits'])}", flush=True)
            keys.append(key)
    finally:
        if con is not None:
            con.close()
    print(",".join(keys))
    return 0


def cmd_live_vlm(args) -> int:
    """The server's first part: image_text's own computation of each
    request's image, under the evaluated configuration or not at all."""
    from scripts import image_text as it

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vlm, ocr = (it.StubVLM(), it.StubOCR()) if args.stub else (it.QwenVLM(), it.EasyOcrBackend("cuda"))
    vlm.prepare()
    ocr.prepare()
    ctx = it.make_context(vlm, ocr, out_root=LIVE_DIR)
    if not args.stub and ctx.config_id != LIVE_CONFIGS["image_text"]:
        print(f"REFUSED: image_text's config here is {ctx.config_id}, not the evaluated "
              f"{LIVE_CONFIGS['image_text']}. Nothing written.", file=sys.stderr)
        return 3
    keys = live_keys(args.keys)
    todo = [k for k in keys if not (LIVE_DIR / k / "image_text.json").is_file()]
    print(f"live image_text, config {ctx.config_id}: {len(keys)} requests, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    vlm.load()
    ocr.load()
    it.warm_up()
    for i, k in enumerate(todo, 1):
        req = _read_live(k, "request.json")
        row = it.process(it.WorkItem("live", req["sha256"], str(LIVE_DIR / k / req["image"]), (), 1),
                         vlm, ocr, ctx)
        write_json(LIVE_DIR / k / "image_text.json", row)
        print(f"  [live-vlm] {i}/{len(todo)} {row['status']} {row['run']['seconds'].get('total', 0.0):.1f}s",
              flush=True)
    return 0


def live_prompts(key: str, gallery, threshold: float) -> dict[str, Any]:
    """One request's model inputs, as the batch builds them: the gallery's
    answers for the claim's names and the image model's, then the predictor's
    prompt (the image's evidence, the VLM's accepted names, any documents;
    never the claim) and the extractor's (the claim alone)."""
    d = LIVE_DIR / key
    req, row = _read_live(key, "request.json"), _read_live(key, "image_text.json")
    sha = req["sha256"]
    if row["sha256"] != sha:
        raise GuardError("an image_text row was joined to another image")
    named: dict[str, set[str]] = defaultdict(set)
    for n in req["claim_people"]:
        named[n].add("claim")
    for n, side in vlm_people(row).items():
        named[n].add(side)
    people = verify_names("live", {sha: named}, read_faces(d / "faces.jsonl"), gallery, threshold)
    write_json(d / "people.json", people)
    retrieval = req["retrieval"]
    docs = documents_of(retrieval) if retrieval and not retrieval["error"] else ()
    ocr, desc, ents = image_part(row, sha)
    view = ModelView(sha, req["claim"], ocr, desc, ents, tuple(vlm_accepts(people, "live").get(sha, ())), docs)
    msgs, evidence = predict_messages(view)
    predict = None
    if evidence:
        check_predict_prompt(msgs, view, [{"text": req["claim"]}])
        predict = {"key": "predict", "task": "predict", "messages": msgs, "content": predict_content(view),
                   "evidence": evidence}
    claim = " ".join(req["claim"].split())
    emsgs = extract_messages(claim)
    refuse_ids(messages_text(emsgs), explained=f"{claim}\n{messages_text(emsgs[:-1])}\n{EXTRACT_TEMPLATE}",
               what="an extraction prompt")
    out = {"predict": predict, "extract": {"key": "extract", "task": "extract", "messages": emsgs, "content": claim}}
    write_json(d / "llm_in.json", out)
    return out


def cmd_live_prompts(args) -> int:
    from scripts import gallery as gl

    cid = gl.face_config_id(gl.make_backend(FACE_BACKEND))
    calib = gl.CALIBRATION_DIR / f"{FACE_BACKEND}.json"
    threshold = gl.certified_threshold(json.loads(calib.read_text(encoding="utf-8")) if calib.is_file() else None,
                                       cid)
    if threshold is None:
        raise SystemExit(f"no certified threshold for face config {cid}: the gallery decides there or not at all")
    gallery = gl.Gallery(cid)
    for k in live_keys(args.keys):
        out = live_prompts(k, gallery, threshold)
        print(f"  [live] {k}: predictor prompt {'built' if out['predict'] else 'none (no evidence)'}, "
              f"extraction prompt built", flush=True)
    return 0


def live_llm_configs(llm) -> dict[str, str]:
    """The config id each text task gets here, from its prompt configuration."""
    configs = {"predict": predict_prompt_config(), "extract": extract_prompt_config(),
               "judge": judge_prompt_config(fewshot()["judge"])}
    return {t: llm_config(llm, t, {"prompt_config_sha256": hashlib.sha256(canonical(c)).hexdigest()})[0]
            for t, c in configs.items()}


def cmd_live_llm(args) -> int:
    """The server's second part: per request, the extraction and the
    prediction, then the judge on each claimed place and event against the
    prediction -- under the evaluated configurations or not at all."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    llm = StubLLM() if args.stub else TextLLM()
    llm.prepare()
    cids = live_llm_configs(llm)
    if not args.stub and (bad := {t: c for t, c in cids.items() if c != LIVE_CONFIGS[t]}):
        print(f"REFUSED: the text tasks' configs here are {bad}, not the evaluated ones. Nothing written.",
              file=sys.stderr)
        return 3
    keys = live_keys(args.keys)
    todo = [k for k in keys if not (LIVE_DIR / k / "llm_out.json").is_file()]
    print(f"live text model, configs {cids}: {len(keys)} requests, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    llm.load()
    shots = fewshot()["judge"]
    for k in todo:
        inp, t = _read_live(k, "llm_in.json"), time.perf_counter()
        es, er = run_extract(llm, inp["extract"])
        ps, pr = run_predict(llm, inp["predict"]) if inp["predict"] else (None, None)
        claimed = extracted_values(es, er)
        pred = predicted({"status": ps, "result": pr}) if ps else dict.fromkeys(PREDICT_ITEMS)
        judged = {}
        for item in JUDGED_ITEMS:
            a, b = claimed[item], pred[item]
            if isinstance(a, str) and isinstance(b, str):
                msgs = judge_messages(item, a, b, shots)
                refuse_ids(messages_text(msgs), explained=f"{a}\n{b}\n{messages_text(msgs[:-1])}",
                           what="a live judge prompt")
                js, jr = run_judge(llm, {"messages": msgs})
                judged[f"{item}\n{a}\n{b}"] = {"item": item, "a": a, "b": b, "status": js, "result": jr,
                                               "result_sha256": result_sha(jr)}
        out = {"config_ids": cids, "code_commit": _code_commit(),
               "extract": {"status": es, "result": er, "result_sha256": result_sha(er)},
               "predict": None if ps is None else {"status": ps, "result": pr, "result_sha256": result_sha(pr)},
               "judge": judged, "seconds": round(time.perf_counter() - t, 3)}
        write_json(LIVE_DIR / k / "llm_out.json", out)
        print(f"  [live-llm] {k}: extract {es}, predict {ps or 'none'}, judge {len(judged)}, "
              f"{out['seconds']:.1f}s", flush=True)
    return 0


def live_record(key: str, *, expect: dict[str, str] | None = LIVE_CONFIGS) -> dict[str, Any]:
    """One live run's verdicts, by the batch's own functions: the extraction
    read by today's parser, the prediction and its cited evidence, the judge's
    answers, the gallery rule, Gate A's clearance and the person's decisions.
    Refused unless the server ran the evaluated configurations."""
    req, inp, out = _read_live(key, "request.json"), _read_live(key, "llm_in.json"), _read_live(key, "llm_out.json")
    if expect and (bad := {t: c for t, c in out["config_ids"].items() if c != expect.get(t)}):
        raise SystemExit(f"{key}: the server ran {bad}, not the evaluated configurations: refusing")
    people = _read_live(key, "people.json")
    sha = req["sha256"]
    er = out["extract"]["result"]
    claimed = extracted_values(*parse_extraction(er["raw"], inp["extract"]["content"], er["finish"]))
    pred = predicted(out["predict"], inp["predict"] is not None)
    titles = {h["doc_id"]: h["title"] for h in ((req["retrieval"] or {}).get("hits") or [])}
    cites, ev = prediction_evidence(inp["predict"], out["predict"], titles) if inp["predict"] else ({}, {})
    items, variants = record_verdicts(claimed, pred, cites, ev, lambda i, a, b: out["judge"].get(f"{i}\n{a}\n{b}"),
                                      gate_cleared())
    verified = {(sha, r["name"]): r for r in people if "claim" in r["sides"]}
    vlm = [r for r in people if {"vlm_free", "vlm_forced"} & set(r["sides"])]
    items["people"] = people_verdict(sha, req["claim_people"], verified, vlm, other_person_decisions())
    return {"request": req, "items": items, "variants": variants, "scores": scores_of(items, variants)}


def live_similarity(scorer: "ClipScorer", path: str, claim: str) -> float:
    """CLIP's image-claim cosine for one live request, as C.5 computes it."""
    import numpy as np

    return round(float(np.dot(scorer.images([path])[0], scorer.texts([claim])[0])), 6)


def cmd_live_finish(args) -> int:
    """The laptop's last part: each request's CLIP similarity (laptop CPU),
    verdicts, fused flag and local page."""
    fitted = read_fused()
    scorer = ClipScorer() if fitted else None
    clip_cid = hashlib.sha256(canonical(scorer.config())).hexdigest()[:12] if scorer else None
    for k in live_keys(args.keys):
        rec = live_record(k)
        req, d = rec["request"], LIVE_DIR / k
        sim = live_similarity(scorer, str(d / req["image"]), req["claim"]) if scorer else None
        if scorer:
            write_json(d / "clip.json", {"config_id": clip_cid, "similarity": sim})
        write_json(d / "verdicts.json", {k2: v for k2, v in rec.items() if k2 != "request"})
        bundle = demo_bundle(k, str(d / req["image"]), req["sha256"], req["claim"], req["date"], req["date"],
                             rec["items"], [f"A live run. The image model and the text model ran on the server under "
                                            f"the evaluated configurations ({', '.join(f'{t} {c}' for t, c in LIVE_CONFIGS.items())}); "
                                            "the gallery, retrieval, CLIP and the verdicts ran on the laptop."],
                             rec["variants"], similarity=sim, fused=fitted)
        path = write_demo(bundle, (), d)
        s = bundle["score"]
        flag = (f"; fused {bundle['fused']['score']:.3f} {'flagged' if bundle['fused']['flagged'] else 'not flagged'}"
                if bundle.get("fused") else "; no fused score (report C has not fitted it)")
        print(f"{k}: primary score {s['score']:.2f}" + (" (abstains)" if s["abstained"] else f" from {s['by']}")
              + flag + "; " + "; ".join(f"{i} {v['verdict']} ({v['why'].split(':')[0]})"
                                        for i, v in bundle["items"].items())
              + f" -> {rel(path)}")
    return 0


def _same(a: Any, b: Any) -> bool:
    return canonical(a) == canonical(b)


def cmd_live_check(args) -> int:
    """Each live run whose image and claim are a C record's, against the batch
    run, stage by stage: the image model's computation, the faces, the
    claim's people, the predictor's prompt and answer, the extraction's
    answer, every judge answer and the verdicts. Identical, or exit 2."""
    from scripts import gallery as gl

    records = {(r["sha256"], r["claim_key"]): r for r in c_records()}
    verdicts = {(v["key"], v["variant"]): v for v in gl.read_jsonl(C_VERDICTS)}
    built, b_out = b_results()
    b_keys = json.loads(B_PREDICT.with_name(B_PREDICT.stem + ".keys.json").read_text(encoding="utf-8"))
    b_in = {b_keys[json.loads(line)["key"]]: json.loads(line)
            for line in B_PREDICT.read_text(encoding="utf-8").splitlines() if line}
    e_keys, e_out, _claims = _extraction_run(C_EXTRACT)
    e_by = {e_keys[k]: r for k, r in e_out.items()}
    j_keys = json.loads(C_JUDGE.with_name(C_JUDGE.stem + ".keys.json").read_text(encoding="utf-8"))
    j_out = read_results(C_JUDGE)
    j_by = {(v["item"], v["a"], v["b"]): j_out[k] for k, v in j_keys.items() if k in j_out}
    fcid = gl.face_config_id(gl.make_backend(FACE_BACKEND))
    clip_of = {r["key"]: r["similarity"] for r in gl.read_jsonl(C_CLIP)} if C_CLIP.is_file() else {}
    failed = 0
    for k in live_keys(args.keys):
        req, d = _read_live(k, "request.json"), LIVE_DIR / k
        r = records.get((req["sha256"], claim_key(req["claim"])))
        if r is None:
            print(f"{k}: no C record has this image and claim: nothing to compare")
            continue
        ds, sha = r["dataset"], r["sha256"]
        variant = "retrieval" if req["date"] else ("image_only" if ds == "verite" else "no_retrieval")
        name = f"{ds}:{sha}:{variant}"
        row, inp, out = _read_live(k, "image_text.json"), _read_live(k, "llm_in.json"), _read_live(k, "llm_out.json")
        batch_row = (image_text_rows(ds) or {}).get(sha) or {}
        live_faces, batch_faces = read_faces(d / "faces.jsonl")[sha], read_faces(faces_path(fcid, ds)).get(sha) or {}
        ner = claim_ner(ds).get(str(r["row"]["record_id"])) or {"spans": []}
        checks = {
            "image model": _same({p: row.get(p) for p in IMAGE_TEXT_PARTS}, {p: batch_row.get(p) for p in IMAGE_TEXT_PARTS}),
            "faces": _same(live_faces["faces"], batch_faces.get("faces")),
            "claim's people": req["claim_people"] == claim_people(ner),
            "predictor prompt": inp["predict"] is not None and name in b_in
                                and _same(inp["predict"]["messages"], b_in[name]["messages"]),
            "predictor answer": out["predict"] is not None and name in b_out
                                and out["predict"]["result_sha256"] == b_out[name]["result_sha256"],
            "extraction answer": _same({x: out["extract"]["result"][x] for x in ("raw", "finish")},
                                       {x: e_by[f"{ds}:{r['claim_key']}"]["result"][x] for x in ("raw", "finish")}),
            "judge answers": all(j_by.get((j["item"], j["a"], j["b"]), {}).get("result_sha256") == j["result_sha256"]
                                 for j in out["judge"].values()),
        }
        rec, batch = live_record(k), verdicts.get((r["key"], "extracted"))
        checks["verdicts"] = batch is not None and _same(rec["items"], batch["items"]) \
            and _same(rec["variants"], batch["variants"])
        if (d / "clip.json").is_file():
            live_sim, batch_sim = _read_live(k, "clip.json")["similarity"], clip_of.get(r["key"])
            checks[f"CLIP similarity (within {CLIP_TOLERANCE:g})"] = (
                live_sim is not None and batch_sim is not None and abs(live_sim - batch_sim) <= CLIP_TOLERANCE)
        failed += sum(1 for ok in checks.values() if not ok)
        print(f"{k} ({ds}, {len(out['judge'])} judge answers): "
              + ", ".join(f"{c} {'identical' if ok else 'DIFFERS'}" for c, ok in checks.items()))
    return 0 if not failed else 2
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot-processed",
                       help="record the processed layer before the M4FC rebuild")
    s.add_argument("--compare", action="store_true",
                   help="compare the layer now against the snapshot; exit 2 if anything moved")
    sub.add_parser("m4fc-files", help="check and list the files image_text reads on the server")
    sub.add_parser("gold", help="stage A's gold pairs: the person's sheet and the true rows")
    p = sub.add_parser("prompts", help="an id-checked input file for the server's llm step")
    p.add_argument("--task", choices=("judge", "predict", "extract"), required=True)
    p.add_argument("--stage", choices=("a", "b", "c"), required=True)
    p.add_argument("--part", choices=("pilot", "full"),
                   help="predict, extract: the pilot slice of M4FC train, or every input the stage scores")
    q = sub.add_parser("llm", help="run a task's prompts through text-only Qwen (server); resumable")
    q.add_argument("--task", choices=sorted(TASKS), required=True)
    q.add_argument("--input", required=True)
    q.add_argument("--recheck", type=int, default=0, metavar="N",
                   help="re-run N finished rows in this fresh process; they must be byte-identical")
    q.add_argument("--stub", action="store_true", help="deterministic stand-in model (tests)")
    c = sub.add_parser("score", help="score a stage")
    c.add_argument("--stage", choices=("a", "b", "c"), required=True)
    sub.add_parser("verdicts", help="stage C.2: every record's verdicts and media_mismatch score")
    f = sub.add_parser("faces", help="stage B: query faces of every image it reads; resumable")
    f.add_argument("--dataset", type=_b_datasets, default=list(B_DATASETS),
                   help="comma-separated, of " + ", ".join(B_DATASETS))
    f.add_argument("--sample", type=int, default=0, metavar="N",
                   help="only N images spread over what is left, timed; the rest projected")
    v = sub.add_parser("verify-people",
                       help="stage B: the gallery's answer for every name, at its certified threshold")
    v.add_argument("--dataset", type=_b_datasets, default=list(B_DATASETS))
    r = sub.add_parser("retrieve", help="stage B: evidence for each M4FC claim, strictly before "
                                        "its bound; resumable")
    r.add_argument("--dataset", choices=B_DATASETS, required=True)
    r.add_argument("--splits", type=_m4fc_splits, default=list(M4FC_SPLITS),
                   help="comma-separated, of " + ", ".join(M4FC_SPLITS))
    r.add_argument("--sample", type=int, default=0, metavar="N",
                   help="only N claims spread over what is left, timed; the rest projected")
    r.add_argument("--check", action="store_true",
                   help="check the ledger against its claims and both retrieval rules; exit 2 on a problem")
    k = sub.add_parser("clip", help="stage C.5: the CLIP baseline, image-claim cosine (laptop CPU); resumable")
    k.add_argument("--sample", type=int, default=0, metavar="N",
                   help="only N records spread over what is left, timed; the rest projected")
    sub.add_parser("review-people", help="stage C.3: a local page of the other_person contradictions to check")
    lv = sub.add_parser("live", help="stage D's live path, laptop side (scripts/live_demo.sh runs it all)")
    lv.add_argument("stage", choices=("prepare", "prompts", "finish", "check"))
    lv.add_argument("--image", help="prepare: the image")
    lv.add_argument("--claim", help="prepare: the claim")
    lv.add_argument("--date", help="prepare: YYYY-MM-DD bounding retrieval; none, no retrieval")
    lv.add_argument("--requests", help="prepare: a JSONL file of {image, claim, date}")
    lv.add_argument("--keys", default="", help="prompts, finish, check: the keys prepare printed")
    ls = sub.add_parser("live-server", help="stage D's live path, server side: the image model, or the text model")
    ls.add_argument("stage", choices=("vlm", "llm"))
    ls.add_argument("--keys", required=True)
    ls.add_argument("--stub", action="store_true", help="deterministic stand-in models (tests)")
    m = sub.add_parser("demo", help="stage D: a record's verdicts as a local page")
    how = m.add_mutually_exclusive_group(required=True)
    how.add_argument("--from-run", metavar="KEY", help="a record already computed (C's verdicts, or a rehearsal)")
    how.add_argument("--record", metavar="FILE", help="a demo bundle written as JSON")
    how.add_argument("--rehearse", action="store_true",
                     help="pages from the outputs that exist now: the third pilot's 30 M4FC train images")
    args = parser.parse_args(argv)
    if args.cmd == "snapshot-processed":
        if args.compare:
            return compare_snapshot()
        snapshot_processed()
        return 0
    if args.cmd == "m4fc-files":
        m4fc_files()
        return 0
    commands: dict[str, Callable[[], int]] = {
        "gold": cmd_gold, "prompts": lambda: cmd_prompts(args), "llm": lambda: cmd_llm(args),
        "score": lambda: {"a": cmd_score_a, "b": cmd_score_b, "c": cmd_score_c}[args.stage](),
        "verdicts": cmd_verdicts,
        "faces": lambda: cmd_faces(args),
        "verify-people": lambda: cmd_verify_people(args), "retrieve": lambda: cmd_retrieve(args),
        "clip": lambda: cmd_clip(args), "demo": lambda: cmd_demo(args),
        "review-people": cmd_review_people,
        "live": lambda: {"prepare": cmd_live_prepare, "prompts": cmd_live_prompts, "finish": cmd_live_finish,
                         "check": cmd_live_check}[args.stage](args),
        "live-server": lambda: {"vlm": cmd_live_vlm, "llm": cmd_live_llm}[args.stage](args)}
    return commands[args.cmd]()


def cmd_prompts(args) -> int:
    if (args.task, args.stage) == ("judge", "a"):
        return cmd_prompts_judge_a()
    if (args.task, args.stage) == ("judge", "b"):
        return cmd_prompts_judge_b()
    if (args.task, args.stage) == ("judge", "c"):
        return cmd_prompts_judge_c()
    if (args.task, args.stage) in (("predict", "b"), ("extract", "c")):
        if args.part is None:
            raise SystemExit(f"{args.task} prompts: say --part pilot or --part full")
        return cmd_prompts_predict_b(args) if args.task == "predict" else cmd_prompts_extract_c(args)
    raise SystemExit(f"there are no {args.task} prompts at stage {args.stage}")


def _listed(text: str, allowed: Sequence[str]) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    unknown = sorted(set(names) - set(allowed))
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(allowed)}; not {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def _b_datasets(text: str) -> list[str]:
    return _listed(text, B_DATASETS)


def _m4fc_splits(text: str) -> list[str]:
    return _listed(text, M4FC_SPLITS)


if __name__ == "__main__":
    raise SystemExit(main())
