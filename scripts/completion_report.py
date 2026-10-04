#!/usr/bin/env python
"""Assemble the data-layer completion report from the reports that measured it.

    python scripts/completion_report.py      # -> data/reports/completion.json
                                             #    data/reports/completion_tables.md

Nothing here is measured afresh: every figure is read from a report a pipeline
stage wrote (processed.json, dead_images.json, the placeholder registry, the
register), so the completion report cannot disagree with the stages it
summarises. A missing input is reported as missing, never filled in.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import PLACEHOLDERS_YAML, REPORTS  # noqa: E402
from scripts import placeholders  # noqa: E402


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "—"


def image_reasons(counts: dict[str, int]) -> dict[str, int]:
    """image_<status> counters -> {status: n}, usable excluded."""
    return {k[len("image_"):]: v for k, v in counts.items()
            if k.startswith("image_") and k != "image_coverage"}


def register_rows() -> list[dict[str, Any]]:
    import yaml

    from configs.paths import DATA

    doc = yaml.safe_load((DATA / "sources.yaml").read_text(encoding="utf-8"))
    entries = doc["datasets"] if isinstance(doc, dict) else doc
    return [{"name": e["name"], "enabled": e.get("enabled"), "licence": e.get("licence"),
             "redistributable": e.get("redistributable")} for e in entries]


def build() -> dict[str, Any]:
    processed = _load("processed.json")
    dead = _load("dead_images.json")
    registry = placeholders.registry()
    out: dict[str, Any] = {
        "inputs": {"processed": bool(processed), "dead_images": bool(dead),
                   "placeholders": PLACEHOLDERS_YAML.name},
        "registry": {"version": placeholders.registry_version(), "entries": len(registry),
                     "by_status": dict(Counter(e["status"] for e in registry.values()))},
        "register": register_rows(),
        "datasets": {},
    }
    if processed:
        out["built_at"] = processed.get("built_at")
        out["images_verified_at"] = processed.get("images_verified_at")
        out["reverified"] = processed.get("reverified")
        out["excluded_by_register"] = processed.get("excluded_by_register")
        for name, d in sorted(processed["datasets"].items()):
            t = d["total"]
            classes = {}
            for label, c in d["per_class"].items():
                classes[label] = {
                    "rows": c.get("rows", 0), "usable": c.get("usable", 0),
                    "images_expected": c.get("images_expected", 0),
                    "images_usable": c.get("images_usable", 0),
                    "not_usable_by_status": image_reasons(c),
                    "image_coverage": c.get("image_coverage")}
            out["datasets"][name] = {
                "rows": t.get("rows", 0), "usable": t.get("usable", 0),
                "images_expected": t.get("images_expected", 0),
                "images_usable": t.get("images_usable", 0),
                "not_usable_by_status": image_reasons(t),
                "splits": {k[len("split_"):]: v for k, v in t.items() if k.startswith("split_")},
                "per_class": classes}
    if dead:
        out["dead"] = {n: {"dead": v["dead"], "not_closed": v["not_closed"],
                           "by_origin": v["by_origin"], "by_archive": v["by_archive"],
                           "by_class": v["by_class"]}
                       for n, v in dead["datasets"].items()}
    return out


def tables(rep: dict[str, Any]) -> str:
    lines = [f"Registry version {rep['registry']['version']} "
             f"({rep['registry']['entries']} entries: "
             + ", ".join(f"{k} {v}" for k, v in sorted(rep['registry']['by_status'].items())) + ")", ""]
    lines += ["| Dataset | Class | Rows | Usable | Images expected | Images usable | Coverage | Not usable, by status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, d in rep["datasets"].items():
        for label, c in sorted(d["per_class"].items()):
            why = ", ".join(f"{k} {v:,}" for k, v in sorted(c["not_usable_by_status"].items(),
                                                            key=lambda kv: -kv[1]) if k != "usable")
            lines.append(f"| {name} | {label} | {c['rows']:,} | {c['usable']:,} | "
                         f"{c['images_expected']:,} | {c['images_usable']:,} | "
                         f"{_pct(c['images_usable'], c['images_expected'])} | {why or '—'} |")
        lines.append(f"| **{name}** | **all** | **{d['rows']:,}** | **{d['usable']:,}** | "
                     f"**{d['images_expected']:,}** | **{d['images_usable']:,}** | "
                     f"**{_pct(d['images_usable'], d['images_expected'])}** | |")
    if rep.get("dead"):
        lines += ["", "| Dataset | Dead images | Not closed | Origin outcome | Archive outcome |",
                  "| --- | --- | --- | --- | --- |"]
        for name, d in rep["dead"].items():
            o = ", ".join(f"{k} {v:,}" for k, v in list(d["by_origin"].items())[:6])
            a = ", ".join(f"{k} {v:,}" for k, v in list(d["by_archive"].items())[:6])
            lines.append(f"| {name} | {d['dead']:,} | {d['not_closed']:,} | {o} | {a} |")
    lines += ["", "| Dataset | Enabled | Licence | Redistributable |", "| --- | --- | --- | --- |"]
    for r in rep["register"]:
        lines.append(f"| {r['name']} | {r['enabled']} | {r['licence']} | {r['redistributable']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rep = build()
    (REPORTS / "completion.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "completion_tables.md").write_text(tables(rep), encoding="utf-8")
    print(f"-> {REPORTS / 'completion.json'}\n-> {REPORTS / 'completion_tables.md'}")
    missing = [k for k, v in rep["inputs"].items() if v is False]
    if missing:
        print(f"MISSING INPUTS: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
