#!/usr/bin/env python
"""Every dataset as one kind of record: the input to splits and the processed layer.

A Record is one labelled unit of a corpus -- a claim, a post, an image-text
pair -- with the text it carries, the images it points at, its label, and the
split its authors assigned (if any). Loaders read data/raw only.

Images are referenced, not resolved, here: ``ImageRef.ref`` is either a
hydration key (resolved by scripts/recover.resolve_image(), so a file only
counts if it passed all three gates) or a path inside a shipped archive or
directory (resolved through the image index). scripts/build_processed.py does
the resolving, at build time, so "usable" is checked rather than assumed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.paths import INTERIM, raw_dir  # noqa: E402

csv.field_size_limit(64 * 1024 * 1024)

#: Official split names normalised to the three the project uses.
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validate": "val", "validation": "val", "dev": "val", "test": "test"}


@dataclass(frozen=True)
class ImageRef:
    role: str             # claim | document | post | evidence | pair
    kind: str             # "hydrated" (key/url) or "file" (index path)
    ref: str              # hydration key, or PROJECT_ROOT-relative index path
    url: str | None = None


@dataclass
class Record:
    dataset: str
    record_id: str
    text: str | None
    label: str | None
    official_split: str | None
    images: list[ImageRef] = field(default_factory=list)
    evidence_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _rel(path: Path) -> str:
    from configs.paths import PROJECT_ROOT
    return path.relative_to(PROJECT_ROOT).as_posix()


def _dict_rows(path: Path, delimiter: str | None = None) -> Iterator[dict[str, str]]:
    from scripts.hydrate import sniff_delimiter
    with path.open(encoding="utf-8", newline="", errors="replace") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter or sniff_delimiter(path))


# --------------------------------------------------------------------------
# text-only corpora
# --------------------------------------------------------------------------


def load_liar() -> Iterator[Record]:
    for stem, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        with (raw_dir("liar") / f"{stem}.tsv").open(encoding="utf-8", newline="") as h:
            for row in csv.reader(h, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) >= 3:
                    yield Record("liar", row[0], row[2], row[1], split,
                                 meta={"speaker": row[4] if len(row) > 4 else None})


def load_welfake() -> Iterator[Record]:
    for row in _dict_rows(raw_dir("welfake") / "WELFake_Dataset.csv", ","):
        text = "\n\n".join(p for p in (row.get("title"), row.get("text")) if p)
        label = {"0": "real", "1": "fake"}.get(row.get("label", ""), row.get("label"))
        yield Record("welfake", row.get("") or row.get("Unnamed: 0"), text, label, None)


def load_isot() -> Iterator[Record]:
    """Record ids are a content hash: ISOT ships one file per label, so a
    file-and-row id (the old ``fake_12``) would publish the label in every
    split file. The hash covers subject, date, title and text; 196 groups of
    rows are byte-identical, always within one file, and take ``-2``, ``-3``
    in order of appearance."""
    seen: dict[str, int] = {}
    for fname, label in (("True.csv", "real"), ("Fake.csv", "fake")):
        for row in _dict_rows(raw_dir("isot") / fname, ","):
            text = "\n\n".join(p for p in (row.get("title"), row.get("text")) if p)
            digest = hashlib.sha1("\x00".join(row.get(k) or "" for k in
                                              ("subject", "date", "title", "text")).encode()).hexdigest()[:16]
            seen[digest] = seen.get(digest, 0) + 1
            rid = digest if seen[digest] == 1 else f"{digest}-{seen[digest]}"
            yield Record("isot", rid, text, label, None,
                         meta={"subject": row.get("subject"), "date": row.get("date")})


def load_fakenewsnet() -> Iterator[Record]:
    """Record ids are FakeNewsNet's own news ids, which carry no label (the
    label is the file a row sits in). Two ids, politifact14940 and
    politifact14920, sit in both politifact_fake.csv and politifact_real.csv
    -- the first byte-identical in both -- so each is ONE record with label
    None and ``label_conflict``; the processed build makes it unusable."""
    rows: dict[str, list[tuple[str, str, dict]]] = {}
    for source in ("politifact", "gossipcop"):
        for label in ("fake", "real"):
            for row in _dict_rows(raw_dir("fakenewsnet") / "dataset" / f"{source}_{label}.csv", ","):
                rows.setdefault(row["id"], []).append((source, label, row))
    for news_id, found in rows.items():
        source, label, row = found[0]
        labels = sorted({lab for _s, lab, _r in found})
        conflict = len(labels) > 1
        yield Record("fakenewsnet", news_id, row.get("title"), None if conflict else label, None,
                     meta={"source": source, "label_conflict": conflict,
                           **({"conflicting_labels": labels} if conflict else {})})


def load_averitec() -> Iterator[Record]:
    for stem, split in (("train", "train"), ("dev", "val"), ("test", "test")):
        rows = json.loads((raw_dir("averitec") / "data" / f"{stem}.json").read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            yield Record("averitec", f"{stem}_{i}", row.get("claim"), row.get("label"), split,
                         evidence_text=row.get("justification"),
                         meta={"claim_date": row.get("claim_date")})


# --------------------------------------------------------------------------
# corpora whose images were hydrated from URLs
# --------------------------------------------------------------------------


def load_verite() -> Iterator[Record]:
    from scripts.hydrate import RESOLVERS
    items, _label, _extra = RESOLVERS["verite"]()
    url_of = {it.key: it.url for it in items}
    path = next(raw_dir("verite").rglob("VERITE.csv"))
    for row in _dict_rows(path, ","):
        key = (row.get("image_path") or "").strip()
        images = [ImageRef("pair", "hydrated", key, url_of.get(key))] if key else []
        # VERITE is an evaluation benchmark with no training split: every row is test.
        yield Record("verite", row.get("") or row.get("id"), row.get("caption"),
                     row.get("label"), "test", images)


#: VERITE names its images true_N / false_N, so an image path published in a
#: tracked file publishes the label. Tracked files use row indices (the
#: record_id) instead; this private map, gitignored, ties them back.
VERITE_PRIVATE_MAP = INTERIM / "verite_private_map.json"
VERITE_IMAGE_NAME = re.compile(r"^(true|false)_(?:(direct|inverse)_)?(\d+)")


def verite_rows_by_image() -> dict[str, list[str]]:
    """Image stem (``<kind>_<n>``) -> the record_ids (row indices) that use it."""
    out: dict[str, list[str]] = {}
    path = next(raw_dir("verite").rglob("VERITE.csv"))
    for row in _dict_rows(path, ","):
        key = (row.get("image_path") or "").strip()
        if key:
            out.setdefault(Path(key).stem, []).append(row.get("") or row.get("id"))
    return out


def verite_alias(name: str, rows_by_image: dict[str, list[str]]) -> str | None:
    """The label-free public name of a VERITE file named after an image:
    ``<kind>_<n>.v2.jpg`` -> ``rows-<r>.v2.jpg``; the shipped evidence files
    ``<kind>_direct_<n>.json`` -> ``rows-<r1>-<r2>_direct.json``. None for any other
    name."""
    m = VERITE_IMAGE_NAME.match(name)
    if not m:
        return None
    stem = f"{m.group(1)}_{m.group(3)}"
    if stem not in rows_by_image:
        return None
    variant = f"_{m.group(2)}" if m.group(2) else ""
    return "rows-" + "-".join(rows_by_image[stem]) + variant + name[m.end():]


def write_verite_private_map() -> Path:
    rows = verite_rows_by_image()
    VERITE_PRIVATE_MAP.parent.mkdir(parents=True, exist_ok=True)
    VERITE_PRIVATE_MAP.write_text(json.dumps(
        {"image_to_rows": rows,
         "row_to_image": {r: f"images/{stem}" for stem, rs in rows.items() for r in rs}},
        indent=1) + "\n", encoding="utf-8")
    return VERITE_PRIVATE_MAP


def load_factify2() -> Iterator[Record]:
    root = raw_dir("factify2")
    for path in sorted(root.rglob("*.csv")):
        if "test" in path.name.lower():
            continue
        for index, row in enumerate(_dict_rows(path)):
            row_id = f"{path.stem}_{row.get('Id') or row.get('id') or index}"
            images = []
            for column, role in (("claim_image", "claim"), ("document_image", "document")):
                url = (row.get(column) or "").strip()
                if url.startswith("http"):
                    images.append(ImageRef(role, "hydrated", f"{row_id}_{column}", url))
            # The official train/val division is NOT used: factify2's split is
            # generated source-disjoint by scripts/splits.py.
            yield Record("factify2", row_id, row.get("claim"), row.get("Category"), None,
                         images, evidence_text=row.get("document"),
                         meta={"official_split": path.stem})


def load_fakeddit() -> Iterator[Record]:
    from scripts.hydrate import RESOLVERS
    items, _label, _extra = RESOLVERS["fakeddit"](None)
    sampled = {it.key: it.url for it in items}
    for fname, split in (("multimodal_train.tsv", "train"), ("multimodal_validate.tsv", "val")):
        for row in _dict_rows(raw_dir("fakeddit") / fname, "\t"):
            rid = row.get("id")
            images = ([ImageRef("post", "hydrated", rid, sampled[rid])] if rid in sampled else [])
            yield Record("fakeddit", rid, row.get("clean_title"), row.get("6_way_label"), split,
                         images, meta={"image_sampled": rid in sampled,
                                       "subreddit": row.get("subreddit"),
                                       "created_utc": row.get("created_utc")})


def load_m4fc() -> Iterator[Record]:
    rows = json.loads((raw_dir("m4fc") / "data" / "M4FC.json").read_text(encoding="utf-8"))
    for row in rows:
        key = row["image_path"]
        yield Record("m4fc", key, row.get("claim"), row.get("verdict_coarse"),
                     SPLIT_ALIASES.get(row.get("split", ""), None),
                     [ImageRef("claim", "hydrated", key, row.get("image_url"))],
                     meta={"fc_org": row.get("fc_org"), "fc_pub_date": row.get("fc_pub_date")})


# --------------------------------------------------------------------------
# corpora that ship their images
# --------------------------------------------------------------------------


def load_averimatec() -> Iterator[Record]:
    archive = _rel(raw_dir("averimatec") / "images.zip")
    for stem, split in (("train", "train"), ("val", "val")):
        rows = json.loads((raw_dir("averimatec") / f"{stem}.json").read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            images = [ImageRef("claim", "file", f"{archive}::images/{name}")
                      for name in (row.get("claim_images") or [])]
            yield Record("averimatec", f"{stem}_{i}", row.get("claim_text"), row.get("label"),
                         split, images, evidence_text=row.get("justification"),
                         meta={"date": row.get("date")})


def _mocheg_stored_names() -> dict[str, str]:
    """Archive member file name -> stored relative path (Windows-safe encoded)."""
    listing = INTERIM / "mocheg_archive_members.jsonl"
    out: dict[str, str] = {}
    if listing.is_file():
        for line in listing.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kept") and row.get("stored_as") and row["name"].startswith("mocheg/images/"):
                out[row["name"].rsplit("/", 1)[1]] = row["stored_as"]
    return out


def load_mocheg() -> Iterator[Record]:
    base = raw_dir("mocheg") / "mocheg_back_up_without_image"
    stored = _mocheg_stored_names()
    image_root = raw_dir("mocheg") / "with_tweet_2023_03"
    for split in ("train", "val", "test"):
        claims: dict[str, dict[str, Any]] = {}
        for row in _dict_rows(base / split / "Corpus2.csv", ","):
            c = claims.setdefault(row["claim_id"], {"claim": row.get("Claim"),
                                                    "label": row.get("cleaned_truthfulness"),
                                                    "evidence": []})
            if row.get("Evidence"):
                c["evidence"].append(row["Evidence"])
        relevant: dict[str, list[str]] = {}
        for row in _dict_rows(base / split / "img_evidence_qrels.csv", ","):
            if row.get("RELEVANCY") == "1":
                relevant.setdefault(row["TOPIC"], []).append(row["DOCUMENT#"])
        for claim_id, c in claims.items():
            images = []
            for name in sorted(set(relevant.get(claim_id, []))):
                rel = stored.get(name)
                if rel is None:
                    from scripts.stream_archive import windows_safe
                    rel = f"images/{windows_safe(name)}"
                images.append(ImageRef("evidence", "file", _rel(image_root / rel)))
            yield Record("mocheg", f"{split}_{claim_id}", c["claim"], c["label"], split, images,
                         evidence_text="\n\n".join(c["evidence"][:20]) or None,
                         meta={"evidence_rows": len(c["evidence"])})


def dgm4_record_id(row: dict[str, Any]) -> str:
    """DGM4's ``id`` is the VisualNews source id, shared by every manipulation
    of the same original (64,869 ids over 230,310 rows). A record is the
    (id, image, text) triple, so the id carries a digest of the other two."""
    import hashlib
    digest = hashlib.sha1(f"{row['image']}\n{row.get('text') or ''}".encode("utf-8")).hexdigest()
    return f"{row['id']}-{digest[:10]}"


def load_dgm4() -> Iterator[Record]:
    """train.json repeats 51,015 rows byte for byte; each is yielded once.

    meta carries what the media_mismatch rule needs (scripts/reason_codes.py):
    the image's publisher (its origin folder, or for a face-edited image the
    origin folder of the same source id), and for a text swap whether the
    swapped caption came from that same publisher -- found by matching the
    caption against DGM4's own pristine captions.
    """
    root = raw_dir("dgm4")
    splits = {stem: json.loads((root / "metadata" / f"{stem}.json").read_text(encoding="utf-8"))
              for stem in ("train", "val")}
    publisher_of_id: dict[Any, str] = {}
    pristine_publishers: dict[str, set[str]] = {}
    for rows in splits.values():
        for row in rows:
            parts = row["image"].split("/")
            if parts[1] == "origin":
                publisher_of_id.setdefault(row["id"], parts[2])
                if row.get("fake_cls") == "orig":
                    pristine_publishers.setdefault(row.get("text") or "", set()).add(parts[2])
    for stem, split in (("train", "train"), ("val", "val")):
        rows = splits[stem]
        seen: set[str] = set()
        for row in rows:
            rid = dgm4_record_id(row)
            if rid in seen:
                continue
            seen.add(rid)
            # 'DGM4/origin/usa_today/0048/120.jpg' lives in origin/usa_today.zip
            parts = row["image"].split("/")
            archive = _rel(root / parts[1] / f"{parts[2]}.zip")
            member = "/".join(parts[2:])
            publisher = parts[2] if parts[1] == "origin" else publisher_of_id.get(row["id"])
            swap_publisher = None
            if "text_swap" in (row.get("fake_cls") or ""):
                src = pristine_publishers.get(row.get("text") or "", set())
                swap_publisher = ("same" if src == {publisher} else
                                  "cross" if len(src) == 1 else
                                  "ambiguous" if src else "unknown")
            yield Record("dgm4", rid, row.get("text"), row.get("fake_cls"), split,
                         [ImageRef("pair", "file", f"{archive}::{member}")],
                         meta={"source_id": row["id"], "publisher": publisher,
                               "swap_publisher": swap_publisher,
                               "fake_image_box": bool(row.get("fake_image_box")),
                               "fake_text_pos": bool(row.get("fake_text_pos"))})


LOADERS: dict[str, Callable[[], Iterator[Record]]] = {
    "mocheg": load_mocheg, "factify2": load_factify2, "averitec": load_averitec,
    "averimatec": load_averimatec, "verite": load_verite, "fakeddit": load_fakeddit,
    "welfake": load_welfake, "liar": load_liar, "isot": load_isot,
    "fakenewsnet": load_fakenewsnet, "dgm4": load_dgm4, "m4fc": load_m4fc,
}
