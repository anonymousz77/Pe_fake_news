#!/usr/bin/env python
"""Track B, the text-side check: does what was published before a claim's date
support it, refute it, or say nothing about it?

    python scripts/textcheck.py units --dataset m4fc               # query units and bounds; offline
    python scripts/textcheck.py retrieve --dataset m4fc --sample 30   # time a spread sample
    python scripts/textcheck.py retrieve --dataset m4fc            # evidence per unit; resumable
    python scripts/textcheck.py retrieve --dataset m4fc --check    # the ledger against both rules
    python scripts/textcheck.py fetch-models                       # pinned NLI A, NLI B, CLIP L/14 (network)
    python scripts/textcheck.py prompts --task tc_judge --dataset m4fc --part pilot   # 30 train units
    python scripts/textcheck.py prompts --task tc_nli --dataset m4fc --part full      # NLI pairs
    python scripts/textcheck.py run --input F                      # server: Qwen; resumable
    python scripts/textcheck.py run --input F --model nli_b        # server: NLI B (nli_a: the contrast)
    python scripts/textcheck.py run --input F --recheck 3          # fresh process, identical?
    python scripts/textcheck.py verdicts                           # the ensemble's file, once the runs exist

The check is the system's source of the reason codes ``content_refuted`` and
``content_unverified``. Four rules hold in every step, and
tests/test_textcheck.py holds them:

* A model sees a ``ClaimView`` and nothing else: the claim's text, and its
  documents' titles, dates, hosts and excerpts. Never a record id, a path, a
  URL, a meta value, a label, a split -- nor the claim's own bound.
* Evidence comes only through ``retrieve.search_with_terms``, strictly before
  ``retrieve.evidence_bound``: a corpus with no date gets none (MOCHEG, LIAR,
  VERITE, Factify2). M4FC's test is never read (``consistency.rows``), and
  VERITE only through ``consistency.evaluation_rows``.
* A query unit is one (claim, bound). Records sharing a claim's text and its
  bound share its evidence, and every record's evidence is strictly before its
  own bound.
* The ensemble's file, ``verdicts.jsonl``, holds exactly ``VERDICT_KEYS`` per
  row, one row per (claim_key, judge), from the judges ``qwen`` and ``nli``
  (model B) only. Model A is trained partly on ANLI (CC BY-NC 4.0): it is a
  reported contrast and never enters that file.

It writes to data/interim/textcheck/ (ignored). Only ``fetch-models`` opens a
socket; retrieval reads the local index.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from configs.paths import INTERIM, MODELS, free_gb  # noqa: E402
from scripts import consistency as cs  # noqa: E402
from scripts.fetchlib import utcnow  # noqa: E402

SEED = cs.SEED
OUT = INTERIM / "textcheck"
RETRIEVAL = OUT / "retrieval"
VERDICTS = OUT / "verdicts.jsonl"
VERDICTS_META = OUT / "verdicts.meta.json"

#: The corpora whose claims carry a date (configs/as_of.yaml) and a
#: claim-level verdict, and the processed splits read of each. AVerImaTeC is
#: evaluation-only: every row of it is processed ``test``.
DATASETS: dict[str, tuple[str, ...]] = {"m4fc": ("train", "val"),
                                        "averitec": ("train", "val"),
                                        "averimatec": ("test",)}
#: Where each is scored. Every other split is train: fitting only.
EVAL_SPLIT = {"m4fc": "val", "averitec": "val", "averimatec": "test"}
#: Bump when what a retrieval row holds, or how a unit is formed, changes.
TEXTCHECK_VERSION = 1
#: The excerpt recipe is the consistency block's, imported unchanged so both
#: blocks see one claim's evidence alike. Its source is hashed into the
#: config: a change there is a new configuration, never silently a mixed run.
SHARED_FUNCTIONS = ("excerpt", "hit_row", "read_doc", "documents_of")


# --------------------------------------------------------------------------
# claims and their bounds
# --------------------------------------------------------------------------


def unit_key(claim_key: str, before: str | None) -> str:
    """A unit's key: a hash, so it says nothing of the claim or its corpus."""
    return "u" + hashlib.sha256(f"{SEED}:{claim_key}:{before}".encode("utf-8")).hexdigest()[:15]


def units(dataset: str, *, with_forms: bool = False) -> list[dict[str, Any]]:
    """One query unit per distinct (claim, bound) of ``dataset``'s splits, in
    key order. A unit carries its query (the claim, through the id check), its
    bound (None: the as-of rule gives none) and how many records share it.
    ``with_forms`` adds the id forms of every record it covers, for checking a
    prompt built from it; they are never written anywhere."""
    from scripts import retrieve

    if dataset not in DATASETS:
        raise SystemExit(f"{dataset}: the text-side check reads {sorted(DATASETS)} only. A corpus "
                         "whose claims carry no date (configs/as_of.yaml) gets no evidence")
    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row in cs.rows(dataset, DATASETS[dataset]):
        claim = row.get("text") or ""
        try:
            before, bound = retrieve.evidence_bound(dataset, row.get("meta") or {})
        except retrieve.RetrievalError:
            before, bound = None, None
        ck = cs.claim_key(claim)
        g = groups.setdefault((ck, before), {
            "key": unit_key(ck, before), "claim_key": ck, "query": claim, "before": before,
            "bound": bound, "records": 0, "splits": set(), "forms": set()})
        g["records"] += 1
        g["splits"].add(row["split"])
        g["forms"] |= cs.id_forms(row)
    out = []
    for g in groups.values():
        view = ClaimView(g["query"])
        forms = frozenset(g.pop("forms"))
        query = cs.refuse_ids(view.claim, forms, explained=view.content(), what="a retrieval query")
        out.append({**g, "query": query, "splits": sorted(g["splits"]),
                    **({"forms": forms} if with_forms else {})})
    if len({u["key"] for u in out}) != len(out):
        raise SystemExit(f"{dataset}: two query units share a key")
    return sorted(out, key=lambda u: u["key"])


def unit_summary(dataset: str, us: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What a corpus's units are: how many, how many claims, how many claims
    carry several bounds, how many units carry no bound."""
    per_claim = Counter(u["claim_key"] for u in us)
    return {"dataset": dataset, "units": len(us), "claims": len(per_claim),
            "records": sum(u["records"] for u in us),
            "claims_with_several_bounds": sum(1 for n in per_claim.values() if n > 1),
            "units_without_bound": sum(1 for u in us if u["before"] is None),
            "units_by_split": cs.ordered(Counter("+".join(u["splits"]) for u in us))}


# --------------------------------------------------------------------------
# the view: the only thing a model sees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimView:
    """Everything a judge may see of one claim: its text and its documents.

    There is no field for a record id, a path, a URL, meta, a label, a split or
    the claim's bound, so none can reach a prompt. A Document carries its store
    doc_id for the join back; prompts number documents instead of showing it.
    """

    claim: str
    documents: tuple[cs.Document, ...] = ()

    def content(self) -> str:
        """The view's text: what every input built from it is checked against."""
        parts = [self.claim or ""]
        parts += [f"{d.doc_id}\n{d.title}\n{d.published_at}\n{d.host}\n{d.excerpt}"
                  for d in self.documents]
        return "\n".join(parts)


# --------------------------------------------------------------------------
# retrieval: the consistency block's recipe, per query unit
# --------------------------------------------------------------------------


def retrieval_path(dataset: str) -> Path:
    return RETRIEVAL / f"{dataset}.jsonl"


def retrieval_config() -> dict[str, Any]:
    """The consistency block's retrieval configuration (index, rules, k, the
    excerpt), and what this block adds: its unit, the hit's store source, and
    the source of every function it imports to build a hit."""
    fns = {name: hashlib.sha256(inspect.getsource(getattr(cs, name)).encode("utf-8")).hexdigest()
           for name in SHARED_FUNCTIONS}
    return {**cs.retrieval_config(), "textcheck_version": TEXTCHECK_VERSION,
            "unit": "(claim_key, bound): records sharing a claim's text and bound share it",
            "hit_source": "the document's provenance.source", "functions_sha256": fns}


def config_id(config: dict[str, Any]) -> str:
    return hashlib.sha256(cs.canonical(config)).hexdigest()[:12]


def read_retrieval(path: Path, cid: str | None = None) -> dict[str, dict[str, Any]]:
    """key -> its retrieval row. A row of another configuration refuses."""
    from scripts import gallery as gl

    out: dict[str, dict[str, Any]] = {}
    for r in gl.read_jsonl(path):
        if cid is not None and r["config_id"] != cid:
            raise SystemExit(f"{path.name} holds rows of retrieval config {r['config_id']}, not "
                             f"{cid}: refusing to mix runs")
        out[r["key"]] = r
    return out


def ledger_config(dataset: str, *, start: bool) -> str:
    """The configuration id of ``dataset``'s ledger, refused if the ledger was
    written under another one. Only a run (``start``) writes it."""
    config = retrieval_config()
    cid = config_id(config)
    path = RETRIEVAL / f"{dataset}.config.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8"))["config_id"] != cid:
            raise SystemExit(f"{path.name} is another retrieval configuration (the index, a rule, a "
                             "setting or a shared function changed): refusing to mix runs")
    elif start:
        cs.write_json(path, {"config_id": cid, **config})
    else:
        raise SystemExit(f"no retrieval ledger for {dataset}: run `retrieve --dataset {dataset}` first")
    return cid


def hit_with_source(hit, terms: Sequence[str], doc: dict[str, Any] | None) -> dict[str, Any]:
    """A hit as the consistency ledger keeps it, plus where the store got the page."""
    row = cs.hit_row(hit, terms, doc)
    row["source"] = ((doc or {}).get("provenance") or {}).get("source")
    return row


def run_retrieval(items: Sequence[dict[str, Any]], path: Path, cid: str,
                  search: Callable[[str, str], tuple[list[Any], list[str], bool]],
                  doc: Callable[[str], dict[str, Any] | None] = cs.read_doc) -> float:
    """Each unit's hits and excerpts, one ledger row each, fsynced; the seconds
    it took. A unit with no bound or no searchable word is written with that as
    its error and no hits: retrieval refused it, which is not "nothing found"."""
    from scripts import gallery as gl
    from scripts.retrieve import words

    ledger = gl.JsonlLedger(path)
    ledger.open()
    started = time.perf_counter()
    try:
        for i, it in enumerate(items, 1):
            t = time.perf_counter()
            row = {"key": it["key"], "config_id": cid, "before": it["before"], "bound": it["bound"],
                   "terms": None, "fell_back": None, "hits": [], "error": None,
                   "query_sha256": hashlib.sha256(it["query"].encode("utf-8")).hexdigest()}
            if it["before"] is None:
                row["error"] = "no_bound"
            elif not words(it["query"]):
                row["error"] = "empty_query"
            else:
                hits, terms, fell_back = search(it["query"], it["before"])
                row.update(terms=terms, fell_back=fell_back,
                           hits=[hit_with_source(h, terms, doc(h.doc_id)) for h in hits])
            row["seconds"] = round(time.perf_counter() - t, 3)
            ledger.append(row)
            per = (time.perf_counter() - started) / i
            print(f"  [retrieve] {i:,}/{len(items):,} {row['error'] or 'ok'} {row['seconds']:.2f}s "
                  f"hits {len(row['hits'])} eta {per * (len(items) - i) / 3600:.2f}h", flush=True)
    finally:
        ledger.close()
    return time.perf_counter() - started


def consistency_agreement(mine: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """M4FC only: this ledger against the consistency block's, read and never
    written. Rows are matched by query and bound; the same recipe on the same
    index must give the same documents in the same order."""
    from scripts import gallery as gl

    theirs_path = cs.B_RETRIEVAL / "m4fc.jsonl"
    if not theirs_path.is_file():
        return {"compared": 0, "note": f"{cs.rel(theirs_path)} is not here"}
    theirs = {(r["query_sha256"], r["before"]): [h["doc_id"] for h in r["hits"]]
              for r in gl.read_jsonl(theirs_path)}
    compared = same = 0
    for r in mine.values():
        got = theirs.get((r["query_sha256"], r["before"]))
        if got is None:
            continue
        compared += 1
        same += got == [h["doc_id"] for h in r["hits"]]
    return {"compared": compared, "identical": same, "different": compared - same,
            "units_unmatched": len(mine) - compared}


def check_ledger(dataset: str, items: Sequence[dict[str, Any]],
                 held: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The consistency block's strict twin of the statement (every hit strictly
    before its bound by the index's own date, every hit allowed by the
    fact-checker rules re-judged in Python), run on this ledger; then the id
    shapes no document may carry, the hits' store sources, and for M4FC the
    agreement with the consistency block's ledger."""
    report = cs.check_retrieval([{**it, "sha256": it["key"]} for it in items],
                                {k: {**r, "sha256": k} for k, r in held.items()})
    # A document whose own text holds an id SHAPE (a page's image URL under images/) is content,
    # not an identifier: every prompt is checked with its documents as explained content. Counted.
    report["documents_with_an_id_shape"] = sum(
        1 for r in held.values() for h in r["hits"]
        if any(s.search(f"{h['title']}\n{h['excerpt'] or ''}".casefold()) for s in cs.ID_SHAPES))
    report["sources"] = cs.ordered(Counter(h.get("source") or "unknown"
                                           for r in held.values() for h in r["hits"]))
    if dataset == "m4fc":
        report["consistency_ledger"] = consistency_agreement(held)
    return report


def cmd_units(args) -> int:
    us = units(args.dataset)
    print(json.dumps(unit_summary(args.dataset, us), indent=1))
    return 0


def cmd_retrieve(args) -> int:
    from scripts import retrieve

    items = units(args.dataset)
    path = retrieval_path(args.dataset)
    cid = ledger_config(args.dataset, start=not args.check)
    done = read_retrieval(path, cid)
    if args.check:
        report = check_ledger(args.dataset, items, done)
        out = {"config_id": cid, "checked_at": utcnow(), "dataset": args.dataset, **report}
        cs.write_json(RETRIEVAL / f"{args.dataset}.check.json", out)
        print(json.dumps(report, indent=1))
        passed = not report["problems"]
        print(("PASSED" if passed else "FAILED") + f" -> {cs.rel(RETRIEVAL / f'{args.dataset}.check.json')}")
        return 0 if passed else 2
    todo = [it for it in items if it["key"] not in done]
    print(f"retrieval, config {cid}: {len(items):,} {args.dataset} units, "
          f"{len(items) - len(todo):,} done, {len(todo):,} to do", flush=True)
    if args.pilot:
        keys = set(pilot_keys(args.dataset))
        picked = [it for it in todo if it["key"] in keys]
    else:
        picked = cs.spread(todo, args.sample) if args.sample else todo
    if picked:
        t = time.perf_counter()
        con = retrieve.connect()
        overhead = time.perf_counter() - t
        try:
            seconds = run_retrieval(picked, path, cid, lambda q, before: retrieve.search_with_terms(
                q, before, cs.RETRIEVE_K, con=con))
        finally:
            con.close()
        if args.sample:
            cs._print_projection(cs.projection(seconds, len(picked), len(todo) - len(picked), overhead),
                                 "query", "queries")
    held = read_retrieval(path, cid)
    got = [held[it["key"]] for it in items if it["key"] in held]
    ok = [r for r in got if not r["error"]]
    secs = sorted(r["seconds"] for r in ok)
    errors = cs.ordered(Counter(r["error"] for r in got if r["error"]))
    print(f"  ledger: {len(got):,} of {len(items):,} units; errors "
          f"{json.dumps(errors) if errors else 'none'}; fewer than {cs.RETRIEVE_K} hits "
          f"{sum(1 for r in ok if len(r['hits']) < cs.RETRIEVE_K)}, none "
          f"{sum(1 for r in ok if not r['hits'])}; fell back to one word "
          f"{sum(1 for r in ok if r['fell_back'])}")
    if secs:
        print(f"  seconds per query: median {secs[len(secs) // 2]:.2f}, mean "
              f"{sum(secs) / len(secs):.2f}, max {secs[-1]:.2f}")
    return 0


# --------------------------------------------------------------------------
# the ensemble's file: verdicts.jsonl, a contract with the consistency block
# --------------------------------------------------------------------------

#: Exactly these keys, in this order, and nothing else: the ensemble joins on
#: claim_key (consistency.claim_key) and reads the rest by these names.
VERDICT_KEYS = ("claim_key", "judge", "verdict", "reason_code", "p_supported", "p_refuted", "p_nei",
                "doc_ids", "status")
#: The judges whose verdicts the ensemble reads: Qwen, and NLI model B. Model
#: A never: it is trained partly on ANLI, CC BY-NC 4.0 (the person's decision,
#: 2026-11-10), and is a reported contrast only.
ENSEMBLE_JUDGES = ("qwen", "nli")
VERDICT_LABELS = ("supported", "refuted", "not_enough_evidence")
#: A verdict's reason code (docs/processed_schema.md): "none" is adjudicated true.
REASON_OF = {"supported": "none", "refuted": "content_refuted",
             "not_enough_evidence": "content_unverified"}
#: ok: a verdict in format. partial: a verdict whose citations are not all the
#: prompt's documents (doc_ids keeps those that are). uncited: supported or
#: refuted with no document cited (doc_ids empty). off_format: the judge's
#: answer was not one of the labels (no verdict). no_documents: the store holds
#: nothing before the bound, so not enough evidence, with no judge called.
#: no_bound: the claim has no date, so retrieval never ran and the text side
#: abstains (VERITE).
STATUSES = ("ok", "partial", "uncited", "off_format", "no_documents", "no_bound")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def verdict_problems(row: dict[str, Any]) -> list[str]:
    """Every way ``row`` breaks the contract; empty when it keeps it."""
    out: list[str] = []
    if tuple(row) != VERDICT_KEYS:
        return [f"keys {list(row)} are not exactly {list(VERDICT_KEYS)}"]
    if not isinstance(row["claim_key"], str) or not _HEX64.match(row["claim_key"]):
        out.append("claim_key is not a claim's sha256")
    if row["judge"] not in ENSEMBLE_JUDGES:
        out.append(f"judge {row['judge']!r} is not one the ensemble reads")
    if row["status"] not in STATUSES:
        out.append(f"status {row['status']!r}")
    verdict, ps = row["verdict"], (row["p_supported"], row["p_refuted"], row["p_nei"])
    if verdict is not None and verdict not in VERDICT_LABELS:
        out.append(f"verdict {verdict!r}")
    if row["reason_code"] != (REASON_OF.get(verdict) if verdict else None):
        out.append(f"reason_code {row['reason_code']!r} does not follow verdict {verdict!r}")
    ids = row["doc_ids"]
    if not isinstance(ids, list) or not all(isinstance(d, str) and _HEX64.match(d) for d in ids):
        out.append("doc_ids is not a list of store doc_ids")
    elif len(set(ids)) != len(ids):
        out.append("a doc_id twice")
    status = row["status"]
    if status in ("off_format", "no_bound"):
        if verdict is not None or any(p is not None for p in ps) or ids:
            out.append(f"{status}: verdict, probabilities and doc_ids must be empty")
    elif status == "no_documents":
        if verdict != "not_enough_evidence" or any(p is not None for p in ps) or ids:
            out.append("no_documents: not enough evidence, no probabilities, no doc_ids")
    elif status in ("ok", "partial", "uncited"):
        if verdict is None:
            out.append(f"{status}: no verdict")
        if not all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in ps):
            out.append(f"{status}: a probability is missing or outside [0, 1]")
        elif abs(sum(ps) - 1.0) > 1e-4:
            out.append(f"{status}: the probabilities sum to {sum(ps):.6f}")
        if status == "uncited" and (verdict == "not_enough_evidence" or ids):
            out.append("uncited: a supported or refuted verdict that cites nothing")
    return out


def verdict_row(claim_key: str, judge: str, status: str, verdict: str | None = None,
                p: dict[str, float] | None = None, doc_ids: Sequence[str] = ()) -> dict[str, Any]:
    """One row of the ensemble's file, refused unless it keeps the contract.
    ``p`` maps the three VERDICT_LABELS to probabilities."""
    p = p or {}
    row = {"claim_key": claim_key, "judge": judge, "verdict": verdict,
           "reason_code": REASON_OF.get(verdict) if verdict else None,
           "p_supported": p.get("supported"), "p_refuted": p.get("refuted"),
           "p_nei": p.get("not_enough_evidence"), "doc_ids": list(doc_ids), "status": status}
    problems = verdict_problems(row)
    if problems:
        raise cs.GuardError(f"a verdict row breaks the contract: {problems}")
    return row


def file_problems(rows_: Sequence[dict[str, Any]]) -> list[str]:
    """The whole file's contract: every row's, and one row per (claim_key, judge)."""
    out = [f"row {i}: {p}" for i, r in enumerate(rows_) for p in verdict_problems(r)]
    pairs = Counter((r.get("claim_key"), r.get("judge")) for r in rows_)
    out += [f"{n} rows for one (claim_key, judge)" for n in pairs.values() if n > 1]
    return out


def write_verdicts(rows_: Sequence[dict[str, Any]], about: dict[str, Any],
                   path: Path = VERDICTS) -> dict[str, Any]:
    """Write the ensemble's file whole, atomically, refused unless every row
    keeps the contract; its sidecar (a separate file) records what it holds."""
    problems = file_problems(rows_)
    if problems:
        raise cs.GuardError(f"{path.name}: {len(problems)} contract problem(s), e.g. {problems[:3]}")
    ordered_rows = sorted(rows_, key=lambda r: (r["claim_key"], r["judge"]))
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    meta = {"written_at": utcnow(), "rows": len(ordered_rows),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "keys": list(VERDICT_KEYS),
            "join": "consistency.claim_key", "by_judge_status": {
                j: cs.ordered(Counter(r["status"] for r in ordered_rows if r["judge"] == j))
                for j in ENSEMBLE_JUDGES}, **about}
    cs.write_json(path.with_name(path.stem + ".meta.json"), meta)
    return meta


def read_verdicts(path: Path = VERDICTS) -> list[dict[str, Any]]:
    """The ensemble's file, refused if it does not keep the contract."""
    rows_ = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    problems = file_problems(rows_)
    if problems:
        raise cs.GuardError(f"{path.name}: {len(problems)} contract problem(s), e.g. {problems[:3]}")
    return rows_


def verite_claim_keys() -> list[str]:
    """Every distinct claim of VERITE's evaluation rows (the 877, the 37
    cross-corpus records dropped and the leak asserted)."""
    return sorted({cs.claim_key(r.get("text") or "") for r in cs.evaluation_rows()})


def verite_rows() -> list[dict[str, Any]]:
    """VERITE's rows of the ensemble's file: it carries no date, so the as-of
    rule gives its claims no evidence, and the text side abstains (no_bound)."""
    return [verdict_row(k, j, "no_bound") for k in verite_claim_keys() for j in ENSEMBLE_JUDGES]


# --------------------------------------------------------------------------
# models: pinned by revision, the weights by sha256, the licence as read
# --------------------------------------------------------------------------

NLI_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "spm.model")
CLIP_FILES = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
#: Read at source on 2026-11-10 (docs/textcheck.md quotes each licence).
#: card_licence is the model card's front matter, checked again before every
#: fetch; a change refuses the fetch until a person has read it.
MODEL_PINS: dict[str, dict[str, Any]] = {
    "nli_b": {"repo": "cross-encoder/nli-deberta-v3-large",
              "revision": "bab4bc7178836f731dcfd18c06ca9def0a137712",
              "card_licence": "apache-2.0", "files": NLI_FILES,
              "weights_sha256": "01cb91b046c0c88c538c06c6d2a8ab8b11f9b9c5f9c2ef0c8e20c4086710063c",
              "weights_bytes": 1740312740,
              "role": "the system's NLI (the person's decision, 2026-11-10)"},
    "nli_a": {"repo": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
              "revision": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
              "card_licence": "mit", "files": NLI_FILES,
              "weights_sha256": "c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b",
              "weights_bytes": 870182606,
              "role": "a reported contrast only: trained partly on ANLI (CC BY-NC 4.0); never in "
                      "the system or the ensemble"},
    "clip_l14": {"repo": "openai/clip-vit-large-patch14",
                 "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
                 "card_licence": None, "files": CLIP_FILES,
                 "weights_sha256": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                 "weights_bytes": 1710540580,
                 "role": "MUSE's encoder (scripts/baselines.py); the card states no licence, the "
                         "code's (openai/CLIP) is MIT, and the card puts any deployed use out of scope"},
}


def model_dir(which: str) -> Path:
    """The pinned snapshot of ``which``, its weights re-hashed: refused if absent or changed."""
    from scripts.image_text import HF_CACHE, ModelMissing
    from scripts.manifest import sha256

    pin = MODEL_PINS[which]
    snap = HF_CACHE / f"models--{pin['repo'].replace('/', '--')}" / "snapshots" / pin["revision"]
    weights = snap / "model.safetensors"
    if not weights.is_file():
        raise ModelMissing(f"{pin['repo']}@{pin['revision'][:12]} is not in {HF_CACHE}; run: "
                           f"python scripts/textcheck.py fetch-models --only {which}")
    if weights.stat().st_size != pin["weights_bytes"] or sha256(weights) != pin["weights_sha256"]:
        raise ModelMissing(f"{pin['repo']}@{pin['revision'][:12]}: the weights are not the pinned "
                           "bytes; refusing")
    return snap


def cmd_fetch_models(args) -> int:
    """The only step that opens a socket: each model at its pinned revision,
    refused if the Hub resolves it elsewhere or its card's licence changed;
    every file checked against the Hub's metadata, the weights against the pin."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    from scripts.image_text import HF_CACHE, log
    from scripts.manifest import sha256

    MODELS.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for which in args.only:
        pin = MODEL_PINS[which]
        info = api.model_info(pin["repo"], revision=pin["revision"], files_metadata=True)
        if info.sha != pin["revision"]:
            raise SystemExit(f"the Hub resolved {pin['repo']}@{pin['revision']} to {info.sha}; refusing")
        card = getattr(info.card_data, "license", None) if info.card_data else None
        if card != pin["card_licence"]:
            raise SystemExit(f"{pin['repo']} now states licence {card!r}, not {pin['card_licence']!r}; "
                             "read it before fetching")
        wanted = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in pin["files"])]
        if {s.rfilename for s in wanted} != set(pin["files"]):
            raise SystemExit(f"{pin['repo']}: the pinned files are not all at this revision")
        need = sum(s.size or 0 for s in wanted) / 1024 ** 3
        if free_gb(MODELS) < 1.5 * need:
            raise SystemExit(f"refusing: {free_gb(MODELS):.1f} GiB free, {1.5 * need:.2f} needed")
        snap = Path(snapshot_download(pin["repo"], revision=pin["revision"], cache_dir=str(HF_CACHE),
                                      allow_patterns=list(pin["files"])))
        bad = [s.rfilename for s in wanted
               if not (snap / s.rfilename).is_file()
               or (snap / s.rfilename).stat().st_size != s.size
               or (s.lfs is not None and sha256(snap / s.rfilename) != s.lfs.sha256)]
        if bad:
            raise SystemExit(f"{pin['repo']}: downloaded files do not match the Hub's metadata: {bad}")
        model_dir(which)
        log(f"{pin['repo']}@{pin['revision'][:12]} ({card}): {len(wanted)} files verified, "
            f"weights at the pinned sha256")
    return 0


# --------------------------------------------------------------------------
# the Qwen judge: text-only, few-shot, invented exemplars, greedy
# --------------------------------------------------------------------------

FEWSHOT_YAML = Path(__file__).resolve().parent.parent / "configs" / "textcheck_fewshot.yaml"
JUDGE_DIR = OUT / "judge"
NLI_DIR = OUT / "nli"
#: The judge's answers, in tie order, and the verdict each is. Their first
#: tokens differ (supported / ref / not), so one step's logits tell them apart.
ANSWERS = {"supported": "supported", "refuted": "refuted", "not enough evidence": "not_enough_evidence"}
#: Labels holding less of the first token's probability than this are no
#: answer (the consistency block's rule).
MIN_LABEL_MASS = cs.MIN_LABEL_MASS
#: The longest answer: the label, a newline, "Documents:" and five numbers.
JUDGE_MAX_NEW_TOKENS = 24
#: A judge sees its unit's top documents, distinct by title and excerpt (the
#: consistency block's documents_of: five), titles cut as the predictor cuts them.
TITLE_CHARS = cs.TITLE_CHARS
#: The seeded slice of each corpus's TRAIN units a prompt is piloted on. Never
#: an evaluation unit.
PILOT_N = 30
JUDGE_SYSTEM = (
    "You check a claim against news documents published before the claim was made. Use only "
    "the documents.\n"
    "- supported: a document states what the claim says.\n"
    "- refuted: a document states something that cannot be true at the same time as the claim.\n"
    "- not enough evidence: no document settles the claim either way, including when the "
    "documents are about something else.\n"
    "Answer on two lines: first supported, refuted or not enough evidence; then \"Documents:\" "
    "with the numbers of the documents a supported or refuted answer rests on, or "
    "\"Documents: none\".")
JUDGE_TEMPLATE = "Claim: {claim}\n\nDocuments:\n{documents}\n\nSupported, refuted or not enough evidence?"
NODOCS_SYSTEM = (
    "You judge a claim from what you already know.\n"
    "- supported: you know it to be true.\n"
    "- refuted: you know it to be false.\n"
    "- not enough evidence: you do not know, or general knowledge cannot tell.\n"
    "Answer with supported, refuted or not enough evidence, and nothing else.")
NODOCS_TEMPLATE = "Claim: {claim}\n\nSupported, refuted or not enough evidence?"
_DOCS_LINE = re.compile(r"^documents?\s*:\s*(none|\d+(?:\s*,\s*\d+)*)\s*\.?$", re.I)


def fewshot() -> dict[str, Any]:
    import yaml

    doc = yaml.safe_load(FEWSHOT_YAML.read_text(encoding="utf-8"))
    for shot in doc["judge"]:
        n = len(shot["documents"])
        if shot["answer"] not in ANSWERS or not all(1 <= c <= n for c in shot["cite"]):
            raise SystemExit(f"{FEWSHOT_YAML.name}: a judge exemplar answers {shot['answer']!r} "
                             f"or cites {shot['cite']} of {n} documents")
        if (shot["answer"] == "not enough evidence") != (not shot["cite"]):
            raise SystemExit(f"{FEWSHOT_YAML.name}: only a supported or refuted exemplar cites")
    for shot in doc["nodocs"]:
        if shot["answer"] not in ANSWERS:
            raise SystemExit(f"{FEWSHOT_YAML.name}: a nodocs exemplar answers {shot['answer']!r}")
    return doc


def documents_text(docs: Sequence[Any]) -> str:
    """Numbered documents, as a judge reads them: never a doc_id or a URL."""
    def field(d, name):
        return d[name] if isinstance(d, dict) else getattr(d, name)

    return "\n\n".join(
        f"[{i}] {' '.join(field(d, 'title').split())[:TITLE_CHARS]} ({field(d, 'published_at')}, "
        f"{field(d, 'host')})\n{field(d, 'excerpt')}" for i, d in enumerate(docs, 1))


def answer_text(answer: str, cite: Sequence[int]) -> str:
    return f"{answer}\nDocuments: {', '.join(str(c) for c in cite) if cite else 'none'}"


def judge_messages(view: ClaimView, shots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The system line, every exemplar as a user turn and its answer, then the
    claim and its documents. The same exemplars, in the same order, every time."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": cs._text(JUDGE_SYSTEM)}]
    for s in shots:
        msgs.append({"role": "user", "content": cs._text(JUDGE_TEMPLATE.format(
            claim=s["claim"], documents=documents_text(s["documents"])))})
        msgs.append({"role": "assistant", "content": cs._text(answer_text(s["answer"], s["cite"]))})
    msgs.append({"role": "user", "content": cs._text(JUDGE_TEMPLATE.format(
        claim=view.claim, documents=documents_text(view.documents)))})
    return msgs


def nodocs_messages(view: ClaimView, shots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The probe of what the model already knows: the claim alone."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": cs._text(NODOCS_SYSTEM)}]
    for s in shots:
        msgs.append({"role": "user", "content": cs._text(NODOCS_TEMPLATE.format(claim=s["claim"]))})
        msgs.append({"role": "assistant", "content": cs._text(s["answer"])})
    msgs.append({"role": "user", "content": cs._text(NODOCS_TEMPLATE.format(claim=view.claim))})
    return msgs


TASKS = {"tc_judge": {"messages": judge_messages, "shots": "judge", "cites": True},
         "tc_nodocs": {"messages": nodocs_messages, "shots": "nodocs", "cites": False}}


def prompt_config(task: str, shots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    spec = TASKS[task]
    return {"task": task, "system": JUDGE_SYSTEM if spec["cites"] else NODOCS_SYSTEM,
            "template": JUDGE_TEMPLATE if spec["cites"] else NODOCS_TEMPLATE,
            "answers": ANSWERS, "shots": list(shots), "min_label_mass": MIN_LABEL_MASS,
            "max_new_tokens": JUDGE_MAX_NEW_TOKENS, "docs": cs.PREDICT_DOCS,
            "title_chars": TITLE_CHARS}


def parse_answer(text: str, n_docs: int, cites: bool) -> dict[str, Any]:
    """The answer's lines, read strictly. Line one must be an answer as
    written; with documents, line two must be "Documents:" and numbers the
    prompt gave, or none."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0].casefold().rstrip(".").strip() if lines else ""
    out: dict[str, Any] = {"answer": first if first in ANSWERS else None, "cited": [],
                           "cite_problem": None}
    if not cites:
        if len(lines) > 1:
            out["cite_problem"] = "extra_lines"
        return out
    m = _DOCS_LINE.match(lines[1]) if len(lines) > 1 else None
    if len(lines) > 2:
        out["cite_problem"] = "extra_lines"
    if m is None:
        out["cite_problem"] = out["cite_problem"] or "no_documents_line"
        return out
    if m.group(1).casefold() != "none":
        nums = [int(x) for x in m.group(1).split(",")]
        good = sorted({x for x in nums if 1 <= x <= n_docs})
        out["cited"] = good
        if len(good) != len(nums):
            out["cite_problem"] = "a number the prompt did not give, or one twice"
    return out


def judge_status(result: dict[str, Any]) -> str:
    """ok, partial, uncited or off_format, from a judge result."""
    if result["label_mass"] < MIN_LABEL_MASS or result["answer"] is None:
        return "off_format"
    if result.get("cites") is False:
        return "partial" if result["cite_problem"] else "ok"
    if result["answer"] != "not enough evidence" and not result["cited"]:
        return "uncited"
    return "partial" if result["cite_problem"] else "ok"


class JudgeLLM:
    """The pinned Qwen through the consistency block's TextLLM (image_text's
    NF4 load and checks), answering in ONE greedy generate call: the first
    step's logits give P(answer) over the answers' first tokens; that step is
    steered to the most probable answer's first token (recorded as ``forced``
    when the free argmax was another token); every later token is the greedy
    argmax, checked."""

    kind = "qwen"

    def __init__(self) -> None:
        self.base = cs.TextLLM()

    def prepare(self) -> None:
        self.base.prepare()

    def load(self) -> None:
        self.base.load()

    def config(self) -> dict[str, Any]:
        return self.base.config()

    def answer(self, messages: list[dict[str, Any]], labels: Sequence[str],
               max_new_tokens: int) -> dict[str, Any]:
        import torch
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

        from scripts.image_text import FatalError, check_greedy

        vlm = self.base.vlm
        firsts = self.base.first_tokens(labels)
        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm.processor(text=[text], return_tensors="pt").to(vlm.model.device)

        class Steer(LogitsProcessor):
            def __init__(self) -> None:
                self.argmax: list[int] = []
                self.chosen: list[int] = []
                self.p: list[float] = []
                self.mass = 0.0

            def __call__(self, input_ids, scores):
                logp = torch.log_softmax(scores[0].float(), dim=-1)
                want = int(logp.argmax())
                if not self.chosen:
                    lab = logp[firsts]
                    self.mass = float(lab.exp().sum())
                    self.p = torch.softmax(lab, dim=-1).tolist()
                    pick = firsts[int(lab.argmax())]
                else:
                    pick = want
                self.argmax.append(want)
                self.chosen.append(pick)
                if pick == want:
                    return scores
                steered = torch.full_like(scores, float("-inf"))
                steered[0, pick] = 0.0
                return steered

        steer = Steer()
        config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                                  eos_token_id=vlm.eos, pad_token_id=vlm.pad,
                                  max_new_tokens=max_new_tokens)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            out = vlm.model.generate(**inputs, generation_config=config,
                                     logits_processor=LogitsProcessorList([steer]))
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        replaced = check_greedy(new, steer.argmax, steer.chosen)
        if any(i != 0 for i in replaced):
            raise FatalError("only the first step may be steered")
        return {"p": dict(zip(labels, steer.p)), "label_mass": steer.mass, "forced": bool(replaced),
                "text": vlm.processor.tokenizer.decode(new, skip_special_tokens=True),
                "finish": "eos" if new and new[-1] in vlm.eos else "length", "n_tokens": len(new),
                "prompt_tokens": int(inputs["input_ids"].shape[1])}


class StubJudge:
    """Deterministic stand-in: checks everything but the model."""

    kind = "stub"

    def prepare(self) -> None:
        pass

    def load(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"id": "stub", "revision": "0"}

    def answer(self, messages, labels, max_new_tokens) -> dict[str, Any]:
        digest = hashlib.sha256(cs.canonical(messages)).digest()
        raw = [1 + b for b in digest[:len(labels)]]
        p = {lab: x / sum(raw) for lab, x in zip(labels, raw)}
        best = max(labels, key=lambda a: (p[a], -list(labels).index(a)))
        cites = "Documents:" in cs.messages_text(messages[:1])
        n = len(re.findall(r"^\[\d+\]", cs.messages_text(messages[-1:]), re.M))
        line2 = ("\nDocuments: " + ("1" if n and best != "not enough evidence" else "none")) if cites else ""
        return {"p": p, "label_mass": 0.4 + digest[-1] / 425, "forced": False, "text": best + line2,
                "finish": "eos", "n_tokens": 3, "prompt_tokens": 0}


def run_judge(llm, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One prompt: the answer's probabilities and label mass, its strict
    parse, and a status. The probabilities are kept even off format."""
    spec = TASKS[row["task"]]
    got = llm.answer(row["messages"], tuple(ANSWERS), JUDGE_MAX_NEW_TOKENS)
    p = {k: round(v, 6) for k, v in got["p"].items()}
    best = max(ANSWERS, key=lambda a: (p[a], -list(ANSWERS).index(a)))
    parsed = parse_answer(got["text"], row["n_docs"], spec["cites"])
    if parsed["answer"] is not None and parsed["answer"] != best:
        parsed["answer"] = None  # the text says other than the steered first token: no answer
    result = {"p": p, "label_mass": round(got["label_mass"], 6), "forced": got["forced"],
              "text": got["text"], "finish": got["finish"], "n_tokens": got["n_tokens"],
              "answer": parsed["answer"], "cited": parsed["cited"],
              "cite_problem": parsed["cite_problem"], "cites": spec["cites"]}
    return judge_status(result), result


# --------------------------------------------------------------------------
# NLI: DeBERTa-v3, one pair per (claim, document); B is the system's
# --------------------------------------------------------------------------

NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_MAX_LENGTH = 512
NLI_SOFTWARE = ("torch", "transformers", "tokenizers")


def nli_pairs(view: ClaimView) -> list[dict[str, str]]:
    """Premise: a document's title and excerpt. Hypothesis: the claim."""
    return [{"premise": f"{' '.join(d.title.split())[:TITLE_CHARS]}. {d.excerpt}",
             "hypothesis": view.claim} for d in view.documents]


def pairs_text(pairs: Sequence[dict[str, str]]) -> str:
    """Every word an NLI model reads of a unit: what its id check reads."""
    return "\n".join(f"{p['premise']}\n{p['hypothesis']}" for p in pairs)


def label_index(id2label: dict[Any, str]) -> dict[str, int]:
    """Each NLI label's logit index, by the label's NAME in the model's config:
    the two models order them differently, so an index is never assumed."""
    by_name = {str(name).casefold(): int(i) for i, name in id2label.items()}
    if set(by_name) != set(NLI_LABELS):
        raise cs.GuardError(f"an NLI config labels its logits {sorted(by_name)}, not {list(NLI_LABELS)}")
    return {lab: by_name[lab] for lab in NLI_LABELS}


class NLIModel:
    """A pinned DeBERTa-v3 NLI cross-encoder, fp32, deterministic, one unit's
    pairs per batch (so a recheck sees the same batches)."""

    kind = "nli"

    def __init__(self, which: str) -> None:
        if which not in ("nli_a", "nli_b"):
            raise SystemExit(f"{which} is not an NLI model")
        self.which, self.pin = which, MODEL_PINS[which]
        self.tok = self.model = self.index = None

    def prepare(self) -> None:
        from transformers import AutoConfig, AutoTokenizer

        snap = model_dir(self.which)
        self.tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        self.index = label_index(AutoConfig.from_pretrained(str(snap), local_files_only=True).id2label)

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        from scripts.image_text import ModelMissing, deterministic_torch

        if not torch.cuda.is_available():
            raise ModelMissing("the NLI runs need CUDA; none is visible")
        deterministic_torch(torch)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir(self.which)), local_files_only=True, dtype=torch.float32).eval().to("cuda")

    def config(self) -> dict[str, Any]:
        return {"which": self.which, "repo": self.pin["repo"], "revision": self.pin["revision"],
                "weights_sha256": self.pin["weights_sha256"], "card_licence": self.pin["card_licence"],
                "role": self.pin["role"], "dtype": "float32", "max_length": NLI_MAX_LENGTH,
                "truncation": "only_first", "batch": "one unit's pairs", "label_index": self.index}

    def probs(self, pairs: Sequence[dict[str, str]]) -> list[dict[str, float]]:
        import torch

        enc = self.tok([p["premise"] for p in pairs], [p["hypothesis"] for p in pairs],
                       truncation="only_first", max_length=NLI_MAX_LENGTH, padding=True,
                       return_tensors="pt").to("cuda")
        torch.manual_seed(SEED)
        with torch.inference_mode():
            pr = torch.softmax(self.model(**enc).logits.float(), dim=-1).tolist()
        return [{lab: round(row[i], 6) for lab, i in self.index.items()} for row in pr]


class StubNLI:
    kind = "stub"

    def __init__(self, which: str = "nli_b") -> None:
        self.which = which

    def prepare(self) -> None:
        pass

    def load(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"which": self.which, "repo": "stub", "revision": "0"}

    def probs(self, pairs):
        out = []
        for p in pairs:
            d = hashlib.sha256(cs.canonical(p)).digest()
            raw = [1 + b for b in d[:3]]
            out.append({lab: round(x / sum(raw), 6) for lab, x in zip(NLI_LABELS, raw)})
        return out


# --------------------------------------------------------------------------
# input files for the server: id-checked, sidecar-sealed (consistency's format)
# --------------------------------------------------------------------------


def docs_of(row: dict[str, Any], variant: str = "docs") -> tuple[cs.Document, ...]:
    """A unit's top documents; ``noenrich`` first leaves out every page the
    store took from the annotators' cited evidence (source enrich)."""
    if variant == "noenrich":
        row = {**row, "hits": [h for h in row["hits"] if h.get("source") != "enrich"]}
    return cs.documents_of(row)


def noenrich_keys(dataset: str) -> list[str]:
    """Evaluation units whose top documents change without the enrich pages."""
    us, views, held = views_of(dataset)
    return sorted(u["key"] for u in us if EVAL_SPLIT[dataset] in u["splits"]
                  and docs_of(held[u["key"]], "noenrich") != views[u["key"]].documents)


def views_of(dataset: str, keys: Iterable[str] | None = None, variant: str = "docs"
             ) -> tuple[list[dict[str, Any]], dict[str, ClaimView], dict[str, dict[str, Any]]]:
    """(units, key -> view, key -> retrieval row) for units with a ledger row.
    Refused while a requested unit has no row."""
    us = units(dataset, with_forms=True)
    cid = ledger_config(dataset, start=False)
    held = read_retrieval(retrieval_path(dataset), cid)
    want = set(keys) if keys is not None else {u["key"] for u in us}
    missing = [u["key"] for u in us if u["key"] in want and u["key"] not in held]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing):,} unit(s) have no retrieval row yet: run "
                         f"`retrieve --dataset {dataset}` to the end first")
    views = {u["key"]: ClaimView(u["query"], docs_of(held[u["key"]], variant)) for u in us
             if u["key"] in want}
    return [u for u in us if u["key"] in want], views, held


def pilot_keys(dataset: str) -> list[str]:
    """PILOT_N seeded TRAIN units with a bound: the only units a prompt is developed on."""
    import random

    train = [u["key"] for u in units(dataset) if u["splits"] == ["train"] and u["before"]]
    return sorted(random.Random(SEED).sample(train, min(PILOT_N, len(train))))


def build_prompts(task: str, dataset: str, part: str) -> Path:
    """The id-checked input file for one task over one corpus. A unit with no
    document is left out: it is no_documents, and no judge is called for it.
    ``nodocs`` covers the evaluation split only (and the pilot)."""
    doc = fewshot()
    shots = doc[TASKS[task]["shots"]]
    if part == "noenrich" and task == "tc_nodocs":
        raise SystemExit("noenrich changes documents; the no-documents probe has none")
    keys = pilot_keys(dataset) if part == "pilot" else noenrich_keys(dataset) if part == "noenrich" else None
    us, views, _ = views_of(dataset, keys, "noenrich" if part == "noenrich" else "docs")
    if task == "tc_nodocs" and part == "full":
        us = [u for u in us if EVAL_SPLIT[dataset] in u["splits"]]
    shots_text = "\n".join(cs.messages_text(TASKS[task]["messages"](ClaimView(""), shots)[1:-1]))
    rows_, keymap = [], {}
    for u in us:
        view = views[u["key"]]
        if TASKS[task]["cites"] and not view.documents:
            continue
        msgs = TASKS[task]["messages"](view, shots)
        explained = f"{view.content()}\n{shots_text}"
        cs.refuse_ids(cs.messages_text(msgs), u["forms"], explained=explained,
                      what=f"a {task} prompt")
        rows_.append({"key": u["key"], "task": task, "messages": msgs, "content": explained,
                      "n_docs": len(view.documents)})
        keymap[u["key"]] = {"dataset": dataset, "claim_key": u["claim_key"], "before": u["before"]}
    path = JUDGE_DIR / f"{dataset}_{part}_{task}_in.jsonl"
    meta = cs.write_prompts(path, task, rows_, prompt_config(task, shots), keymap)
    print(f"{task} {dataset} {part}: {meta['rows']:,} prompts -> {cs.rel(path)} "
          f"(prompt config {meta['prompt_config_sha256'][:12]})")
    return path


def build_nli(dataset: str, part: str) -> Path:
    """The id-checked NLI input: one row per unit with documents, its pairs."""
    keys = pilot_keys(dataset) if part == "pilot" else noenrich_keys(dataset) if part == "noenrich" else None
    us, views, _ = views_of(dataset, keys, "noenrich" if part == "noenrich" else "docs")
    rows_, keymap = [], {}
    for u in us:
        view = views[u["key"]]
        if not view.documents:
            continue
        pairs = nli_pairs(view)
        # the claim stands in every pair, so the content explains it once per pair: a claim may
        # say what its own slug says (consistency.ids_in), and that is content, not an id
        explained = view.content() + "".join("\n" + view.claim for _ in pairs)
        cs.refuse_ids(pairs_text(pairs), u["forms"], explained=explained, what="an NLI pair")
        rows_.append({"key": u["key"], "task": "tc_nli", "pairs": pairs, "content": explained})
        keymap[u["key"]] = {"dataset": dataset, "claim_key": u["claim_key"], "before": u["before"]}
    path = NLI_DIR / f"{dataset}_{part}_nli_in.jsonl"
    cfg = {"task": "tc_nli", "premise": "title. excerpt", "hypothesis": "the claim",
           "docs": cs.PREDICT_DOCS, "title_chars": TITLE_CHARS, "labels": list(NLI_LABELS)}
    meta = cs.write_prompts(path, "tc_nli", rows_, cfg, keymap)
    print(f"tc_nli {dataset} {part}: {meta['rows']:,} units, {sum(len(r['pairs']) for r in rows_):,} "
          f"pairs -> {cs.rel(path)}")
    return path


def read_nli_input(inp: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """As consistency.read_input, for pair rows: the sealed file, id-checked."""
    meta = json.loads(cs.sidecar(inp).read_text(encoding="utf-8"))
    body = inp.read_bytes()
    if hashlib.sha256(body).hexdigest() != meta["sha256"]:
        raise SystemExit(f"{inp.name} is not the file its sidecar describes: refusing")
    if meta["task"] != "tc_nli" or not (meta.get("id_check") or {}).get("passed"):
        raise SystemExit(f"{inp.name} was written for task {meta['task']!r}, or not id-checked")
    out = [json.loads(line) for line in body.decode("utf-8").splitlines() if line]
    for r in out:
        cs.refuse_ids(pairs_text(r["pairs"]), explained=r.get("content", ""),
                      what=f"input row {r['key']}")
    if len({r["key"] for r in out}) != len(out):
        raise SystemExit(f"{inp.name}: a key appears twice")
    return out, meta


# --------------------------------------------------------------------------
# the server's runs: resumable, one config per output, rechecked
# --------------------------------------------------------------------------

RUNNER_VERSION = 1
LLM_SOFTWARE = cs.LLM_SOFTWARE


def run_config(model, task: str, meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from scripts.image_text import software_versions

    names = NLI_SOFTWARE if task == "tc_nli" else LLM_SOFTWARE
    config = {"runner_version": RUNNER_VERSION, "task": task, "model": model.config(), "seed": SEED,
              "decoding": ({"method": "softmax over the NLI labels, one forward pass per unit"}
                           if task == "tc_nli" else
                           {"method": "one greedy generate call; the first step steered to the most "
                                      "probable answer's first token, every later token the argmax",
                            "answers": list(ANSWERS), "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
                            "batch_size": 1}),
              "prompt_config_sha256": meta["prompt_config_sha256"],
              "software": software_versions(names) if model.kind != "stub" else {}}
    return hashlib.sha256(cs.canonical(config)).hexdigest()[:12], config


def out_path_for(inp: Path, which: str | None) -> Path:
    """<stem>_in.jsonl -> <stem>_out.jsonl; for NLI, <stem>_<model>_out.jsonl."""
    base = cs.output_path(inp)
    return base.with_name(base.stem[:-4] + f"_{which}_out.jsonl") if which else base


def run_rows(model, task: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if task == "tc_nli":
        return "ok", {"probs": model.probs(row["pairs"])}
    return run_judge(model, row)


def cmd_run(args) -> int:
    """The server's step: an input file through one model, resumable."""
    from configs.paths import PROJECT_ROOT

    inp = Path(args.input)
    inp = inp if inp.is_absolute() else PROJECT_ROOT / inp
    task = json.loads(cs.sidecar(inp).read_text(encoding="utf-8"))["task"]
    if task == "tc_nli":
        rows_, meta = read_nli_input(inp)
        model = StubNLI(args.model) if args.stub else NLIModel(args.model)
        which = args.model
    elif task in TASKS:
        rows_, meta = cs.read_input(inp, task)
        model = StubJudge() if args.stub else JudgeLLM()
        which = None
    else:
        raise SystemExit(f"{inp.name}: task {task!r} is not this block's")
    if not args.stub:  # as image_text: nothing past fetch-models reaches the network
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model.prepare()
    cid, config = run_config(model, task, meta)
    out_path = out_path_for(inp, which)
    cfg_path = out_path.with_name(out_path.stem + ".config.json")
    if cfg_path.is_file():
        if json.loads(cfg_path.read_text(encoding="utf-8"))["config_id"] != cid:
            raise SystemExit(f"{cfg_path.name} is another configuration: refusing to mix runs")
    else:
        cs.write_json(cfg_path, {"config_id": cid, **config, "code_commit": cs._code_commit(),
                                 "input": inp.name, "input_sha256": meta["sha256"]})
    done = cs.read_output(out_path, cid)
    if args.recheck:
        return cs._recheck(model, lambda m, r: run_rows(m, task, r), rows_, done, args.recheck)
    todo = [r for r in rows_ if r["key"] not in done]
    print(f"config {cid}: {len(rows_):,} rows, {len(done):,} done, {len(todo):,} to do", flush=True)
    if not todo:
        return 0
    t = time.perf_counter()
    model.load()
    print(f"model loaded in {time.perf_counter() - t:.0f}s", flush=True)
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="\n") as handle:
        for i, r in enumerate(todo, 1):
            t = time.perf_counter()
            status, result = run_rows(model, task, r)
            secs = time.perf_counter() - t
            handle.write(json.dumps({"key": r["key"], "config_id": cid, "status": status,
                                     "result": result, "result_sha256": cs.result_sha(result),
                                     "seconds": round(secs, 3)}, ensure_ascii=False) + "\n")
            handle.flush()
            per = (time.perf_counter() - started) / i
            print(f"  [{task}] {i:,}/{len(todo):,} {status} {secs:.2f}s "
                  f"eta {per * (len(todo) - i) / 3600:.2f}h", flush=True)
            if i == cs.PILOT_N:
                print(f"  [{task}] pilot: {per:.2f} s/item over {cs.PILOT_N}; {len(todo):,} items -> "
                      f"{per * len(todo) / 3600:.2f} h", flush=True)
    return 0


def read_run(inp: Path, which: str | None = None) -> dict[str, dict[str, Any]]:
    """The finished output of an input file (for NLI, of one model), by key."""
    out = out_path_for(inp, which)
    if not out.is_file():
        return {}
    cfg = json.loads(out.with_name(out.stem + ".config.json").read_text(encoding="utf-8"))
    return cs.read_output(out, cfg["config_id"])


def cmd_prompts(args) -> int:
    if args.task == "tc_nli":
        build_nli(args.dataset, args.part)
    else:
        build_prompts(args.task, args.dataset, args.part)
    return 0


# --------------------------------------------------------------------------
# verdicts: from the runs to the ensemble's file
# --------------------------------------------------------------------------

#: Each corpus's gold label as a 3-way verdict; a label missing here has none
#: (AVeriTeC's and AVerImaTeC's Conflicting Evidence/Cherrypicking).
GOLD = {"m4fc": {"false": "refuted", "true": "supported",
                 "not enough information": "not_enough_evidence"},
        "averitec": {"Refuted": "refuted", "Supported": "supported",
                     "Not Enough Evidence": "not_enough_evidence"}}
GOLD["averimatec"] = GOLD["averitec"]


def qwen_verdict(result: dict[str, Any] | None, status: str | None, docs: Sequence[cs.Document]
                 ) -> tuple[str, str | None, dict[str, float] | None, list[str]]:
    """(status, verdict, p, doc_ids) of one unit's Qwen run, as the contract wants it."""
    if not docs:
        return "no_documents", "not_enough_evidence", None, []
    if result is None or status == "off_format":
        return "off_format", None, None, []
    verdict = ANSWERS[result["answer"]]
    p = {ANSWERS[k]: float(v) for k, v in result["p"].items()}
    if verdict == "not_enough_evidence":
        ids = [d.doc_id for d in docs]
    else:
        ids = [docs[c - 1].doc_id for c in result["cited"]]
    return status, verdict, p, ids


def nli_decisive(probs: Sequence[dict[str, float]], mask: Sequence[bool] | None = None
                 ) -> tuple[int, float] | None:
    """The decisive document among those that count (``mask``; all by
    default): the largest of P(entailment) and P(contradiction), ties to the
    higher rank; and that largest value. None when no document counts."""
    idx = [i for i in range(len(probs)) if mask is None or mask[i]]
    if not idx:
        return None
    best = max(idx, key=lambda i: (max(probs[i]["entailment"], probs[i]["contradiction"]), -i))
    return best, max(probs[best]["entailment"], probs[best]["contradiction"])


def nli_verdict(probs: Sequence[dict[str, float]] | None, docs: Sequence[cs.Document], tau: float,
                mask: Sequence[bool] | None = None
                ) -> tuple[str, str | None, dict[str, float] | None, list[str]]:
    """(status, verdict, p, doc_ids) of one unit's NLI run at threshold ``tau``:
    the decisive document's three probabilities are the claim's; below tau the
    verdict is not enough evidence and rests on every document that counts.
    With a relevance ``mask``, a unit none of whose documents counts has no
    usable document: no_documents, as when the store holds none."""
    if not docs:
        return "no_documents", "not_enough_evidence", None, []
    if probs is None:
        return "off_format", None, None, []
    dec = nli_decisive(probs, mask)
    if dec is None:
        return "no_documents", "not_enough_evidence", None, []
    i, top = dec
    d = probs[i]
    total = d["entailment"] + d["neutral"] + d["contradiction"]
    p = {"supported": d["entailment"] / total, "refuted": d["contradiction"] / total,
         "not_enough_evidence": d["neutral"] / total}
    if top < tau:
        return "ok", "not_enough_evidence", p, [x.doc_id for j, x in enumerate(docs)
                                                if mask is None or mask[j]]
    verdict = "supported" if d["entailment"] >= d["contradiction"] else "refuted"
    return "ok", verdict, p, [docs[i].doc_id]


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Macro-F1 over the classes present in the gold, as m4fc_audit.scored averages."""
    f1s = []
    for lab in [x for x in VERDICT_LABELS if x in set(gold)]:
        tp = sum(1 for g, q in zip(gold, pred) if g == lab and q == lab)
        fp = sum(1 for g, q in zip(gold, pred) if g != lab and q == lab)
        fn = sum(1 for g, q in zip(gold, pred) if g == lab and q != lab)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def _sweep(items: Sequence[tuple[str, float | None, str | None]]) -> tuple[float, float]:
    """(tau, macro-F1) maximising macro-F1 over (gold, decisive value or None,
    the side above the threshold). A unit with no value is always not enough
    evidence. One sorted pass: as tau rises past a value, its units turn to not
    enough evidence. Candidates are 0 and every observed value; ties go to the
    lowest threshold."""
    present = {g for g, _, _ in items}
    labs = [x for x in VERDICT_LABELS if x in present] or list(VERDICT_LABELS)
    tp, fp, fn = Counter(), Counter(), Counter()

    def add(g, q, w):
        if g == q:
            tp[q] += w
        else:
            fp[q] += w
            fn[g] += w

    for g, v, side in items:
        add(g, side if v is not None else "not_enough_evidence", 1)

    def f1() -> float:
        return sum((2 * tp[x] / (2 * tp[x] + fp[x] + fn[x])) if tp[x] else 0.0 for x in labs) / len(labs)

    order = sorted((v, g, side) for g, v, side in items if v is not None)
    best, i = (0.0, f1()), 0  # tau = 0: every unit with a value keeps its side
    while i < len(order):
        v0 = order[i][0]
        while i < len(order) and order[i][0] == v0:  # below any tau above v0, these turn to NEI
            _, g, side = order[i]
            add(g, side, -1)
            add(g, "not_enough_evidence", 1)
            i += 1
        if i < len(order):  # the next candidate: every value under it is now NEI
            score = f1()
            if score > best[1] + 1e-12:
                best = (order[i][0], score)
    return best


def fit_tau(items: Sequence[tuple[str, Sequence[dict[str, float]] | None, Sequence[cs.Document]]],
            masks: Sequence[Sequence[bool] | None] | None = None) -> dict[str, Any]:
    """The NLI threshold, fitted by macro-F1 on TRAIN units only: (gold verdict,
    probabilities, documents) each, and optionally each unit's relevance mask."""
    masks = masks or [None] * len(items)
    rows_ = []
    for (g, pr, docs), mask in zip(items, masks):
        if pr is None or not docs:
            continue
        dec = nli_decisive(pr, mask)
        if dec is None:
            rows_.append((g, None, None))
            continue
        d = pr[dec[0]]
        rows_.append((g, dec[1], "supported" if d["entailment"] >= d["contradiction"] else "refuted"))
    tau, score = _sweep(rows_)
    return {"tau": round(tau, 6), "train_macro_f1": round(score, 6), "train_units": len(rows_)}


# --------------------------------------------------------------------------
# relevance: a document counts only if it shares an entity of the claim's, or
# passes a BM25 floor -- the floor fitted with the threshold, on train only
# --------------------------------------------------------------------------

ENTITIES_DIR = OUT / "entities"
ENTITY_LABELS = ("PER", "ORG", "LOC", "MISC")
MIN_ENTITY_CHARS = 2
#: Candidate floors: these percentiles of the train units' document scores,
#: and the two ends (no document passes by BM25: entities only; every
#: document passes: no filter).
FLOOR_PERCENTILES = tuple(range(0, 101, 5))


def entity_config() -> dict[str, Any]:
    from scripts import gallery as gl

    return {"model": dict(gl.NER_MODEL), "labels": list(ENTITY_LABELS), "aggregation": "first",
            "clean": gl.CLEAN_RULE, "max_chars": gl.NER_MAX_CHARS, "min_chars": MIN_ENTITY_CHARS,
            "key": "consistency.claim_key"}


class EntityNer:
    """The gallery's pinned NER model, every label it gives (PER, ORG, LOC, MISC):
    the gallery keeps people and organisations, a document's relevance needs
    places and events too."""

    def __init__(self) -> None:
        self.pipe = None

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        from scripts import gallery as gl

        snap = gl.snapshot_dir(gl.NER_MODEL["id"], gl.NER_MODEL["revision"])
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(snap), local_files_only=True)
        self.pipe = pipeline("token-classification", model=model, tokenizer=tok,
                             aggregation_strategy="first", device=0 if torch.cuda.is_available() else -1)

    def __call__(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if self.pipe is None:
            self.load()
        out: list[list[dict[str, Any]]] = [[] for _ in texts]
        todo = [i for i, t in enumerate(texts) if t.strip()]
        found = self.pipe([texts[i] for i in todo], batch_size=64) if todo else []
        for i, ents in zip(todo, found):
            out[i] = [{"type": e["entity_group"], "text": texts[i][e["start"]:e["end"]].strip(),
                       "score": round(float(e["score"]), 4)}
                      for e in ents if e["entity_group"] in ENTITY_LABELS
                      and len(texts[i][e["start"]:e["end"]].strip()) >= MIN_ENTITY_CHARS]
        return out


def claim_entities(dataset: str, ner: Callable[[Sequence[str]], list[list[dict[str, Any]]]] | None = None,
                   claims: dict[str, str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """claim_key -> its entities, from the cache where it has them; the rest
    read once and appended. ``claims`` (claim_key -> text) defaults to every
    unit's claim of ``dataset``."""
    from scripts import gallery as gl

    if claims is None:
        claims = {u["claim_key"]: u["query"] for u in units(dataset)}
    cfg_path = ENTITIES_DIR / "config.json"
    if cfg_path.is_file():
        if json.loads(cfg_path.read_text(encoding="utf-8")) != entity_config():
            raise SystemExit(f"{cfg_path} is another entity configuration: refusing to mix")
    else:
        cs.write_json(cfg_path, entity_config())
    ledger = gl.JsonlLedger(ENTITIES_DIR / f"{dataset}.jsonl")
    done = {r["claim_key"]: r["entities"] for r in ledger.open()}
    try:
        todo = sorted(k for k in claims if k not in done)
        if todo:
            ner = ner or EntityNer()
            for i in range(0, len(todo), 1024):
                part = todo[i:i + 1024]
                found = ner([gl.clean_claim(claims[k])[0] for k in part])
                ledger.append_many([{"claim_key": k, "entities": e} for k, e in zip(part, found)])
                done.update(zip(part, found))
                print(f"  [entities] {dataset} {min(i + 1024, len(todo)):,}/{len(todo):,}", flush=True)
    finally:
        ledger.close()
    return {k: done[k] for k in claims}


def entity_terms(entities: Sequence[dict[str, Any]]) -> list[str]:
    """A claim's distinct entity strings, case-folded."""
    return sorted({e["text"].casefold() for e in entities if len(e["text"]) >= MIN_ENTITY_CHARS})


def shares_entity(doc: cs.Document, terms: Sequence[str]) -> bool:
    """Whether a document's title or excerpt (what NLI reads) holds one of the
    claim's entities, whole word and case-folded (m4fc_audit.found_in)."""
    from scripts.m4fc_audit import found_in

    text = f"{doc.title}\n{doc.excerpt}".casefold()
    return any(found_in(t, [text]) for t in terms)


def relevance(docs: Sequence[cs.Document], scores: Sequence[float], terms: Sequence[str],
              floor: float) -> list[bool]:
    """Which documents count: those sharing an entity of the claim's, or whose
    BM25 score (FTS5 rank: lower is more relevant) is at or under the floor."""
    return [shares_entity(d, terms) or s <= floor for d, s in zip(docs, scores)]


def fit_filter(units_: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The BM25 floor and the threshold, fitted jointly by macro-F1 on TRAIN
    units: each {gold, probs, docs, scores, terms}. Candidates run from the
    strictest (BM25 admits nothing: entities alone) to no filter; ties go to
    the stricter."""
    import numpy as np

    pool = [s for u in units_ for s in u["scores"]]
    floors = [float("-inf")] + sorted({float(np.percentile(pool, q)) for q in FLOOR_PERCENTILES}) + [float("inf")]
    ent = [[shares_entity(d, u["terms"]) for d in u["docs"]] for u in units_]
    best = None
    for floor in floors:
        masks = [[e or s <= floor for e, s in zip(es, u["scores"])] for es, u in zip(ent, units_)]
        fit = fit_tau([(u["gold"], u["probs"], u["docs"]) for u in units_], masks)
        if best is None or fit["train_macro_f1"] > best["train_macro_f1"] + 1e-12:
            kept = sum(sum(m) for m in masks)
            best = {**fit, "floor": floor, "documents_kept": kept, "documents": sum(len(m) for m in masks),
                    "units_with_none_kept": sum(1 for m in masks if not any(m))}
    best["floor"] = None if best["floor"] == float("inf") else (
        "entities only" if best["floor"] == float("-inf") else round(best["floor"], 5))
    best["candidates"] = len(floors)
    return best


def floor_value(fit: dict[str, Any]) -> float:
    f = fit["floor"]
    return float("inf") if f is None else float("-inf") if f == "entities only" else float(f)


def nli_units(dataset: str, which: str, part: str = "full") -> list[dict[str, Any]]:
    """Every unit with documents and an NLI output from ``which``: its key,
    splits, gold verdict, the model's probabilities, the documents, their BM25
    scores (as retrieved) and the claim's entity terms."""
    us, views, held = views_of(dataset)
    gold = unit_gold(dataset)
    ents = claim_entities(dataset)
    done = nli_run(run_files(dataset, part)["nli"], which)[0]
    out = []
    for u in us:
        k, docs = u["key"], views[u["key"]].documents
        if not docs or k not in done:
            continue
        score_of = {h["doc_id"]: h["score"] for h in held[k]["hits"]}
        out.append({"key": k, "splits": u["splits"], "gold": gold.get(k), "probs": done[k]["result"]["probs"],
                    "docs": docs, "scores": [score_of[d.doc_id] for d in docs],
                    "terms": entity_terms(ents[u["claim_key"]])})
    return out


VARIANTS = ("unfiltered", "filtered")


def fit_variant(variant: str, train: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if variant == "unfiltered":
        return {**fit_tau([(u["gold"], u["probs"], u["docs"]) for u in train]), "floor": None}
    return fit_filter(train)


def predict_variant(fit: dict[str, Any], u: dict[str, Any]) -> str:
    mask = (None if fit["floor"] is None else
            relevance(u["docs"], u["scores"], u["terms"], floor_value(fit)))
    return nli_verdict(u["probs"], u["docs"], fit["tau"], mask)[1]


def cross_validated(units_: Sequence[dict[str, Any]], variant: str, folds: int = 5,
                    seed: int = SEED) -> list[str]:
    """Each TRAIN unit's verdict from a fit on the other folds: a train figure
    that does not score a unit on the fit it helped make."""
    import random

    order = list(range(len(units_)))
    random.Random(seed).shuffle(order)
    fold_of = {i: pos % folds for pos, i in enumerate(order)}
    pred: list[str | None] = [None] * len(units_)
    for f in range(folds):
        fit = fit_variant(variant, [u for i, u in enumerate(units_) if fold_of[i] != f])
        for i, u in enumerate(units_):
            if fold_of[i] == f:
                pred[i] = predict_variant(fit, u)
    return pred  # type: ignore[return-value]


def nli_adoption(dataset: str) -> dict[str, Any]:
    """The figures the NLI adoption is decided on: TRAIN only. For model B
    without and with the relevance filter, and model A (a contrast): the fit
    on all of train, and 5-fold cross-validated per-class scores within train
    beside the train majority."""
    from scripts.m4fc_audit import majority, scored

    out: dict[str, Any] = {"dataset": dataset, "split": "train (5-fold cross-validated within it)"}
    for which, variants in (("nli_b", VARIANTS), ("nli_a", ("unfiltered",))):
        units_ = [u for u in nli_units(dataset, which) if u["splits"] == ["train"] and u["gold"]]
        golds = [u["gold"] for u in units_]
        for variant in variants:
            fit = fit_variant(variant, units_)
            pred = cross_validated(units_, variant)
            cell = {"fit_on_all_train": fit, **scored(golds, pred, majority(golds), list(VERDICT_LABELS))}
            if variant == "filtered":
                masks = [relevance(u["docs"], u["scores"], u["terms"], floor_value(fit)) for u in units_]
                kept = [(p["contradiction"] >= 0.9) for u, m in zip(units_, masks)
                        for p, keep in zip(u["probs"], m) if keep]
                dropped = [(p["contradiction"] >= 0.9) for u, m in zip(units_, masks)
                           for p, keep in zip(u["probs"], m) if not keep]
                cell["contradiction_0_9"] = {"kept": cs.rate(sum(kept), len(kept)),
                                             "dropped": cs.rate(sum(dropped), len(dropped))}
            out[f"{which}_{variant}"] = cell
    return out


def cmd_nli_adoption(args) -> int:
    report = {ds: nli_adoption(ds) for ds in args.dataset}
    cs.write_json(OUT / "nli_adoption.json", {"written_at": utcnow(), **report})
    for ds, rep in report.items():
        print(f"== {ds} train ({rep['nli_b_unfiltered']['rows']:,} units), 5-fold CV within train")
        for name in ("nli_b_unfiltered", "nli_b_filtered", "nli_a_unfiltered"):
            c = rep[name]
            m = c["model"]
            pcs = "  ".join(f"{lab[:3]} P {d['precision']['value']} R {d['recall']['value']}"
                            for lab, d in m["per_class"].items())
            fit = c["fit_on_all_train"]
            print(f"  {name:18} macro-F1 {m['macro_f1']['value']} {m['macro_f1']['ci']}  acc "
                  f"{m['accuracy']['value']}  | {pcs} | tau {fit['tau']} floor {fit['floor']}")
        print(f"  {'majority':18} macro-F1 {rep['nli_b_unfiltered']['majority']['macro_f1']['value']}  acc "
              f"{rep['nli_b_unfiltered']['majority']['accuracy']['value']}")
    print(f"-> {cs.rel(OUT / 'nli_adoption.json')}")
    return 0


def cmd_entities(args) -> int:
    for ds in args.dataset:
        ents = claim_entities(ds)
        with_any = sum(1 for e in ents.values() if entity_terms(e))
        print(f"{ds}: {len(ents):,} claims, {with_any:,} with at least one entity "
              f"({with_any / max(1, len(ents)):.1%})")
    return 0


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------


def unit_gold(dataset: str) -> dict[str, str | None]:
    """unit key -> its gold verdict; None where its records disagree, or their
    label has no 3-way verdict (Conflicting)."""
    from scripts import retrieve

    labels: dict[str, set] = {}
    for row in cs.rows(dataset, DATASETS[dataset]):
        try:
            before, _ = retrieve.evidence_bound(dataset, row.get("meta") or {})
        except retrieve.RetrievalError:
            before = None
        key = unit_key(cs.claim_key(row.get("text") or ""), before)
        labels.setdefault(key, set()).add(GOLD[dataset].get(row["label"]))
    return {k: (next(iter(v)) if len(v) == 1 else None) for k, v in labels.items()}


def run_files(dataset: str, part: str = "full") -> dict[str, Path]:
    return {"judge": JUDGE_DIR / f"{dataset}_{part}_tc_judge_in.jsonl",
            "nodocs": JUDGE_DIR / f"{dataset}_{part}_tc_nodocs_in.jsonl",
            "nli": NLI_DIR / f"{dataset}_{part}_nli_in.jsonl"}


def nli_run(inp: Path, which: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """One NLI model's finished run, refused unless it is that model at its pin."""
    out = out_path_for(inp, which)
    cfg = json.loads(out.with_name(out.stem + ".config.json").read_text(encoding="utf-8"))
    pin = MODEL_PINS[which]
    model = cfg["model"]
    if (model.get("which"), model.get("repo"), model.get("revision")) != (which, pin["repo"], pin["revision"]):
        raise cs.GuardError(f"{out.name} is not {which} at its pinned revision: refusing")
    return cs.read_output(out, cfg["config_id"]), cfg


def run_config_id(inp: Path, which: str | None = None) -> str | None:
    out = out_path_for(inp, which)
    cfg = out.with_name(out.stem + ".config.json")
    return json.loads(cfg.read_text(encoding="utf-8"))["config_id"] if cfg.is_file() else None


def unit_verdicts(dataset: str, part: str = "full",
                  taus: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every unit's Qwen and NLI verdicts (both models, where run), and the
    NLI thresholds: fitted on the corpus's own TRAIN units, or given (AVerImaTeC
    inherits AVeriTeC's, as its reason codes do). Refused while a unit with
    documents has no output in a run."""
    files = run_files(dataset, part)
    us, views, _ = views_of(dataset)
    gold = unit_gold(dataset)
    judge = read_run(files["judge"])
    nli = {w: nli_run(files["nli"], w) for w in ("nli_b", "nli_a")
           if out_path_for(files["nli"], w).is_file()}
    if "nli_b" not in nli:
        raise SystemExit(f"{dataset}: NLI-B has not run (no {out_path_for(files['nli'], 'nli_b').name})")
    with_docs = [u["key"] for u in us if views[u["key"]].documents]
    for name, done in [("qwen", judge)] + [(w, nli[w][0]) for w in nli]:
        missing = [k for k in with_docs if k not in done]
        if missing:
            raise SystemExit(f"{dataset}: {name} has no output for {len(missing):,} unit(s) with "
                             "documents: the run is not finished")
    if taus is None:
        if "train" not in DATASETS[dataset]:
            raise SystemExit(f"{dataset} has no train split: give it a corpus's thresholds")
        taus = {w: fit_tau([(gold[u["key"]], done[u["key"]]["result"]["probs"] if u["key"] in done
                             else None, views[u["key"]].documents) for u in us
                            if u["splits"] == ["train"] and gold.get(u["key"])])
                for w, (done, _) in nli.items()}
    per_unit = {}
    for u in us:
        k, docs = u["key"], views[u["key"]].documents
        q = judge.get(k)
        row = {"claim_key": u["claim_key"], "before": u["before"], "splits": u["splits"],
               "gold": gold.get(k),
               "qwen": qwen_verdict(q["result"] if q else None, q["status"] if q else None, docs)}
        for w, (done, _) in nli.items():
            row[w] = nli_verdict(done[k]["result"]["probs"] if k in done else None, docs,
                                 taus[w]["tau"])
        per_unit[k] = row
    return {"units": per_unit, "taus": taus,
            "configs": {"qwen": run_config_id(files["judge"]),
                        **{w: cfg["config_id"] for w, (_, cfg) in nli.items()}}}


def m4fc_rows(uv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """M4FC's rows of the ensemble's file: per claim, the unit with the earliest
    bound (so no record of it sees evidence from after its own date); judges
    qwen and nli (model B) only."""
    by_claim: dict[str, tuple[str, dict[str, Any]]] = {}
    for k, u in uv["units"].items():
        cur = by_claim.get(u["claim_key"])
        if cur is None or (u["before"] or "") < (cur[1]["before"] or ""):
            by_claim[u["claim_key"]] = (k, u)
    rows_ = []
    for ck_, (_, u) in sorted(by_claim.items()):
        for judge, src in (("qwen", u["qwen"]), ("nli", u["nli_b"])):
            status, verdict, p, ids = src
            rows_.append(verdict_row(ck_, judge, status, verdict, p, ids))
    several = sum(1 for n in Counter(u["claim_key"] for u in uv["units"].values()).values() if n > 1)
    return rows_, {"claims": len(by_claim), "claims_with_several_bounds": several}


# --------------------------------------------------------------------------
# score: the evaluation sets, per class, beside the baselines -> textcheck.json
# --------------------------------------------------------------------------

#: Which NLI-B variant is the system's: the person's adoption decision, on
#: train figures (nli-adoption), 2026-11-11: B WITHOUT the relevance filter
#: (5-fold CV macro-F1 0.324 / 0.393 against 0.322 / 0.386 with it; the filter
#: kept every M4FC document). Both variants stay reported side by side.
NLI_SYSTEM_VARIANT: str | None = "unfiltered"
#: Corpora whose thresholds a corpus takes (AVerImaTeC has no train split).
FITS_FROM = {"m4fc": "m4fc", "averitec": "averitec", "averimatec": "averitec"}


def eval_records(dataset: str) -> list[dict[str, Any]]:
    """The evaluation split's usable labelled records, each with its unit key
    and gold verdict (None: Conflicting, kept apart)."""
    from scripts import retrieve

    out = []
    for row in cs.rows(dataset, (EVAL_SPLIT[dataset],)):
        if row.get("label") is None:
            continue
        try:
            before, _ = retrieve.evidence_bound(dataset, row.get("meta") or {})
        except retrieve.RetrievalError:
            before = None
        out.append({"text": row["text"] or "", "label": row["label"], "gold": GOLD[dataset].get(row["label"]),
                    "unit": unit_key(cs.claim_key(row["text"] or ""), before)})
    return out


def text_only_predictions(dataset: str, records: Sequence[dict[str, Any]]) -> list[str]:
    """The text-only classifier's three-way prediction per record (baselines.py's
    setting: m4fc_audit's TF-IDF + logistic regression on the claim, fitted on
    the train split; AVerImaTeC takes AVeriTeC's)."""
    import numpy as np

    from scripts.fakeddit_classes import tokens
    from scripts.m4fc_audit import Tfidf, fit_logreg

    src = FITS_FROM[dataset]
    train = [(r["text"] or "", GOLD[src][r["label"]]) for r in cs.rows(src, ("train",))
             if r.get("label") in GOLD[src]]
    labels = sorted({y for _, y in train})
    vec = Tfidf().fit([tokens(t) for t, _ in train])
    w, b, _ = fit_logreg(vec.transform([tokens(t) for t, _ in train]), [labels.index(y) for _, y in train],
                         len(labels))
    logits = vec.transform([tokens(r["text"]) for r in records]) @ w + b
    return [labels[i] for i in np.asarray(logits).argmax(axis=1)]


def paired_delta(gold: Sequence[str], a: Sequence[str], b: Sequence[str], seed: int = SEED) -> dict[str, Any]:
    """Macro-F1(a) - macro-F1(b) on the same records, with a percentile 95%
    interval from resampling records (the same draws for both)."""
    import numpy as np

    from scripts.m4fc_audit import BOOTSTRAP as N, _metrics

    labs = list(VERDICT_LABELS)
    t = np.array([labs.index(g) for g in gold])
    pa, pb = np.array([labs.index(x) for x in a]), np.array([labs.index(x) for x in b])
    draws = np.random.default_rng(seed).integers(0, len(t), size=(N, len(t)))
    deltas = sorted(_metrics(t[d], pa[d], 3)[1] - _metrics(t[d], pb[d], 3)[1] for d in draws)
    point = _metrics(t, pa, 3)[1] - _metrics(t, pb, 3)[1]
    return {"delta_macro_f1": round(float(point), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)]}


def unit_predictions(dataset: str, fits: dict[str, dict[str, Any]]) -> dict[str, dict[str, str | None]]:
    """variant -> unit key -> verdict, for every run that exists. A unit
    without documents is not enough evidence; an off-format answer is None."""
    us, views, held = views_of(dataset)
    files = run_files(dataset)
    out: dict[str, dict[str, str | None]] = {}
    judge = read_run(files["judge"])
    if judge:
        out["qwen"] = {u["key"]: qwen_verdict(judge[u["key"]]["result"] if u["key"] in judge else None,
                                              judge[u["key"]]["status"] if u["key"] in judge else None,
                                              views[u["key"]].documents)[1] for u in us}
        noenrich = read_run(run_files(dataset, "noenrich")["judge"])
        if noenrich:
            ne = {**out["qwen"]}
            _, ne_views, _ = views_of(dataset, noenrich_keys(dataset), "noenrich")
            for k, r in noenrich.items():
                ne[k] = qwen_verdict(r["result"], r["status"], ne_views[k].documents)[1]
            out["qwen_noenrich"] = ne
    nodocs = read_run(files["nodocs"])
    if nodocs:
        out["qwen_nodocs"] = {k: (ANSWERS[r["result"]["answer"]] if r["status"] != "off_format" else None)
                              for k, r in nodocs.items()}
    ents = claim_entities(dataset)
    for which, variants in (("nli_b", VARIANTS), ("nli_a", ("unfiltered",))):
        if not out_path_for(files["nli"], which).is_file():
            continue
        done = nli_run(files["nli"], which)[0]
        for variant in variants:
            fit = fits[f"{which}_{variant}"]
            pred = {}
            for u in us:
                k, docs = u["key"], views[u["key"]].documents
                score_of = {h["doc_id"]: h["score"] for h in held[k]["hits"]}
                mask = (None if fit["floor"] is None else
                        relevance(docs, [score_of[d.doc_id] for d in docs],
                                  entity_terms(ents[u["claim_key"]]), floor_value(fit)))
                pred[k] = nli_verdict(done[k]["result"]["probs"] if k in done else None, docs,
                                      fit["tau"], mask)[1]
            out[f"{which}_{variant}"] = pred
        # without the enrich pages: the changed units from the no-enrich run, the rest as they
        # are, at the same threshold (fitted on train, no filter)
        ne_inp = run_files(dataset, "noenrich")["nli"]
        if out_path_for(ne_inp, which).is_file():
            ne_done = nli_run(ne_inp, which)[0]
            _, ne_views, _ = views_of(dataset, noenrich_keys(dataset), "noenrich")
            ne = dict(out[f"{which}_unfiltered"])
            tau = fits[f"{which}_unfiltered"]["tau"]
            for k, r in ne_done.items():
                ne[k] = nli_verdict(r["result"]["probs"], ne_views[k].documents, tau)[1]
            out[f"{which}_unfiltered_noenrich"] = ne
    return out


def coverage(dataset: str) -> dict[str, Any]:
    """The evaluation units' evidence, with its strict twins: a hit at all;
    the top documents holding one, or two, of the claim's entities; and the
    share of top documents the store took from the annotators' cited pages."""
    us, views, held = views_of(dataset)
    ents = claim_entities(dataset)
    ev = [u for u in us if EVAL_SPLIT[dataset] in u["splits"]]
    n_ent = []
    for u in ev:
        terms = entity_terms(ents[u["claim_key"]])
        text = "\n".join(f"{d.title}\n{d.excerpt}" for d in views[u["key"]].documents).casefold()
        from scripts.m4fc_audit import found_in
        n_ent.append(sum(1 for t in terms if found_in(t, [text])))
    top = [h for u in ev for h in held[u["key"]]["hits"]
           if h["doc_id"] in {d.doc_id for d in views[u["key"]].documents}]
    return {"units": len(ev), "with_a_hit": cs.rate(sum(1 for u in ev if views[u["key"]].documents), len(ev)),
            "top_holds_1_entity": cs.rate(sum(1 for n in n_ent if n >= 1), len(ev)),
            "top_holds_2_entities": cs.rate(sum(1 for n in n_ent if n >= 2), len(ev)),
            "top_documents_from_enrich": cs.rate(sum(1 for h in top if h.get("source") == "enrich"), len(top)),
            "units_changed_without_enrich": len(noenrich_keys(dataset))}


def score_dataset(dataset: str, fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from scripts.m4fc_audit import majority, scored

    recs = eval_records(dataset)
    three = [r for r in recs if r["gold"]]
    conflicting = [r for r in recs if not r["gold"]]
    gold = [r["gold"] for r in three]
    src = FITS_FROM[dataset]
    train_major = majority([GOLD[src][r["label"]] for r in cs.rows(src, ("train",)) if r.get("label") in GOLD[src]])
    text_pred = text_only_predictions(dataset, three)
    out: dict[str, Any] = {"records": len(three), "conflicting_kept_apart": len(conflicting),
                           "text_only": scored(gold, text_pred, train_major, list(VERDICT_LABELS)),
                           "variants": {}}
    base_label, base_n = Counter(gold).most_common(1)[0]
    out["base_rate"] = {"label": base_label, **cs.rate(base_n, len(gold))}
    for variant, by_unit in unit_predictions(dataset, fits).items():
        pred_raw = [by_unit.get(r["unit"], "not_enough_evidence") for r in three]
        pred = [p or "not_enough_evidence" for p in pred_raw]
        out["variants"][variant] = {
            **scored(gold, pred, train_major, list(VERDICT_LABELS)),
            "no_verdict_scored_as_not_enough_evidence": sum(1 for p in pred_raw if p is None),
            # the person's request (2026-11-11): how often the judge abstains, overall and per
            # true class, read beside the corpus's base rate
            "abstention": {"overall": cs.rate(sum(1 for p in pred if p == "not_enough_evidence"), len(pred)),
                           "by_true_class": {lab: cs.rate(sum(1 for g, p in zip(gold, pred)
                                                              if g == lab and p == "not_enough_evidence"),
                                                          sum(1 for g in gold if g == lab))
                                             for lab in VERDICT_LABELS if lab in gold},
                           "base_rate": {"label": base_label, **cs.rate(base_n, len(gold))}},
            "vs_text_only": paired_delta(gold, pred, text_pred),
            "conflicting_predicted": cs.ordered(Counter(by_unit.get(r["unit"]) or "no_verdict"
                                                        for r in conflicting))}
    out["coverage"] = coverage(dataset)
    return out


# --------------------------------------------------------------------------
# the pilot: format, labels against gold, copying from the exemplars
# --------------------------------------------------------------------------

#: The person's stop rule (2026-11-11): a label above this share of a pilot's
#: predictions stops the work before the full runs, for the person to see.
PILOT_MAX_LABEL_SHARE = 0.70
#: Names and numbers that exist only in the exemplars.
EXEMPLAR_NAMES = ("velmora", "dunholt", "karst bay", "olsterra", "pellow", "torvale", ".example")
EXEMPLAR_NUMBERS = ("61,300", "57,900", "4,000", "34c", "20,000", "2 hours 14", "19-year-old", "6 percent",
                    "50,000")


def exemplar_shingles(n: int = 6) -> set[str]:
    doc = fewshot()
    texts = [s["claim"] for s in doc["judge"] + doc["nodocs"]]
    texts += [f"{d['title']} {d['excerpt']}" for s in doc["judge"] for d in s["documents"]]
    out: set[str] = set()
    for t in texts:
        words = " ".join(t.casefold().split()).split()
        out |= {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return out


def copies_exemplar(text: str, shingles: set[str], n: int = 6) -> list[str]:
    """What of an answer's text comes from the exemplars: their invented names,
    their numbers, or six words in a row."""
    low = " ".join(text.casefold().split())
    words = low.split()
    found = [x for x in EXEMPLAR_NAMES + EXEMPLAR_NUMBERS if x in low]
    found += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)
              if " ".join(words[i:i + n]) in shingles]
    return found


def pilot_report() -> dict[str, Any]:
    from statistics import median

    shingles = exemplar_shingles()
    out: dict[str, Any] = {"stop_rule": f"any label above {PILOT_MAX_LABEL_SHARE:.0%} of a pilot's predictions",
                           "tasks": {}, "stop": []}
    for task in ("tc_judge", "tc_nodocs"):
        pooled: Counter = Counter()
        for ds in ("m4fc", "averitec"):
            inp = JUDGE_DIR / f"{ds}_pilot_{task}_in.jsonl"
            done = read_run(inp)
            gold = unit_gold(ds)
            keys = pilot_keys(ds)
            rows_ = [done[k] for k in keys if k in done]
            pred = {k: (ANSWERS[done[k]["result"]["answer"]] if done[k]["result"]["answer"] else None)
                    for k in keys if k in done}
            dist = Counter(v or "no_answer" for v in pred.values())
            pooled.update(dist)
            copies = {k: copies_exemplar(done[k]["result"]["text"], shingles) for k in pred}
            cell = {
                "rows": len(rows_), "missing": len(keys) - len(rows_),
                "status": cs.ordered(Counter(r["status"] for r in rows_)),
                "label_mass": {"min": round(min(r["result"]["label_mass"] for r in rows_), 4),
                               "median": round(median(r["result"]["label_mass"] for r in rows_), 4)},
                "forced_first_token": sum(1 for r in rows_ if r["result"]["forced"]),
                "predicted": cs.ordered(dist),
                "gold": cs.ordered(Counter(gold[k] or "conflicting" for k in pred)),
                "gold_vs_predicted": cs.ordered(Counter(f"{gold[k] or 'conflicting'} -> {v or 'no_answer'}"
                                                        for k, v in pred.items())),
                "cited": cs.ordered(Counter(str(r["result"]["cited"]) for r in rows_)) if task == "tc_judge" else None,
                "copies_from_exemplars": {"answers": sum(1 for v in copies.values() if v),
                                          "found": cs.ordered(Counter(x for v in copies.values() for x in v))},
                "seconds": {"median": round(median(r["seconds"] for r in rows_), 3),
                            "mean": round(sum(r["seconds"] for r in rows_) / len(rows_), 3)},
                "texts_off_format": [done[k]["result"]["text"] for k in pred
                                     if done[k]["status"] == "off_format"][:5]}
            top, n = dist.most_common(1)[0]
            cell["largest_label_share"] = round(n / len(pred), 4)
            if n / len(pred) > PILOT_MAX_LABEL_SHARE:
                out["stop"].append(f"{task} {ds}: {top} is {n} of {len(pred)} predictions")
            out["tasks"].setdefault(task, {})[ds] = cell
        top, n = pooled.most_common(1)[0]
        out["tasks"][task]["pooled"] = {"predicted": cs.ordered(pooled),
                                        "largest_label_share": round(n / sum(pooled.values()), 4)}
        if n / sum(pooled.values()) > PILOT_MAX_LABEL_SHARE:
            out["stop"].append(f"{task} pooled: {top} is {n} of {sum(pooled.values())} predictions")
    return out


def cmd_pilot_report(args) -> int:
    rep = pilot_report()
    cs.write_json(OUT / "pilot_report.json", {"written_at": utcnow(), **rep})
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print("STOP: " + "; ".join(rep["stop"]) if rep["stop"] else "no label above the stop share")
    return 3 if rep["stop"] else 0


def cmd_score(args) -> int:
    from configs.paths import REPORTS

    adoption = json.loads((OUT / "nli_adoption.json").read_text(encoding="utf-8"))
    fits = {ds: {k: v["fit_on_all_train"] for k, v in adoption[ds].items() if isinstance(v, dict) and
                 "fit_on_all_train" in v} for ds in ("m4fc", "averitec") if ds in adoption}
    report: dict[str, Any] = {"written_at": utcnow(), "nli_system_variant": NLI_SYSTEM_VARIANT,
                              "fits": fits, "datasets": {}}
    for ds in args.dataset:
        report["datasets"][ds] = score_dataset(ds, fits[FITS_FROM[ds]])
    contrasts, system = {"nli_a": {}, "nli_a_noenrich": {}, "qwen_nodocs": {}}, {}
    for ds, rep in report["datasets"].items():
        v = rep["variants"]
        contrasts["nli_a"][ds] = v.get("nli_a_unfiltered")
        contrasts["nli_a_noenrich"][ds] = v.get("nli_a_unfiltered_noenrich")
        contrasts["qwen_nodocs"][ds] = v.get("qwen_nodocs")
        system[ds] = {"qwen": v.get("qwen"), "nli": v.get(f"nli_b_{NLI_SYSTEM_VARIANT}") if NLI_SYSTEM_VARIANT
                      else None, "text_only": rep["text_only"]}
    report["system"] = system
    report["contrasts"] = contrasts
    report["coverage"] = {ds: rep["coverage"] for ds, rep in report["datasets"].items()}
    if VERDICTS_META.is_file():
        report["verdicts_file"] = json.loads(VERDICTS_META.read_text(encoding="utf-8"))
    for rep in report["datasets"].values():  # model A lives under contrasts only
        rep["variants"].pop("nli_a_unfiltered", None)
        rep["variants"].pop("nli_a_unfiltered_noenrich", None)
    cs.public_report(report, REPORTS / "textcheck.json")
    for ds, rep in report["datasets"].items():
        print(f"== {ds} ({rep['records']} records): text-only macro-F1 "
              f"{rep['text_only']['model']['macro_f1']['value']}, majority "
              f"{rep['text_only']['majority']['macro_f1']['value']}")
        for name, c in rep["variants"].items():
            ab = c["abstention"]
            per = "  ".join(f"{lab[:3]} {v['rate']}" for lab, v in ab["by_true_class"].items())
            print(f"  {name:22} macro-F1 {c['model']['macro_f1']['value']} {c['model']['macro_f1']['ci']}  "
                  f"vs text-only {c['vs_text_only']['delta_macro_f1']} {c['vs_text_only']['ci95']}  | "
                  f"abstains {ab['overall']['rate']} ({per}) beside base rate "
                  f"{ab['base_rate']['rate']} {ab['base_rate']['label']}")
    print(f"-> {cs.rel(REPORTS / 'textcheck.json')}")
    return 0


def cmd_verdicts(args) -> int:
    """The ensemble's file, written once, whole: M4FC train and val from the
    Qwen and NLI-B runs, and VERITE's no_bound rows. Refused while either run
    is unfinished."""
    uv = unit_verdicts("m4fc")
    rows_, counts = m4fc_rows(uv)
    verite = verite_rows()
    # A claim whose text both corpora carry gets one row per judge: VERITE's no_bound wins, so no
    # VERITE record is given evidence (it has no date); the M4FC record abstains on the text side.
    shared = {r["claim_key"] for r in verite} & {r["claim_key"] for r in rows_}
    rows_ = [r for r in rows_ if r["claim_key"] not in shared]
    counts["claims_shared_with_verite_left_no_bound"] = len(shared)
    print(f"M4FC: {counts['claims']:,} claims x 2 judges; VERITE: {len(verite) // 2:,} claims x 2 "
          f"judges (no_bound); NLI-B threshold {uv['taus']['nli_b']['tau']} fitted on M4FC train")
    if args.dry_run:
        print(json.dumps({"problems": file_problems(rows_ + verite)[:5]}, indent=1))
        return 0
    meta = write_verdicts(rows_ + verite, {
        "holds": ["m4fc train", "m4fc val", "verite"], "m4fc": counts,
        "verite": "no_bound: VERITE carries no date, so the as-of rule gives it no evidence",
        "qwen": {"config_id": uv["configs"]["qwen"], "model": "Qwen/Qwen2.5-VL-7B-Instruct"},
        "nli": {"config_id": uv["configs"]["nli_b"], "model": MODEL_PINS["nli_b"]["repo"],
                "revision": MODEL_PINS["nli_b"]["revision"], "threshold": uv["taus"]["nli_b"]},
        "rules": {"doc_ids": "supported/refuted: the documents the verdict rests on; not enough "
                             "evidence: every document shown",
                  "nli_p": "the decisive document's three probabilities",
                  "earliest_bound": "a claim with several bounds takes its earliest unit",
                  "shared_with_verite": "a claim text in both M4FC and VERITE keeps VERITE's no_bound "
                                        "row (VERITE has no date); its M4FC record abstains"}})
    print(f"-> {cs.rel(VERDICTS)} ({meta['rows']:,} rows, sha256 {meta['sha256'][:12]})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("units", help="query units and bounds of a corpus; offline")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_units)
    p = sub.add_parser("retrieve", help="evidence per query unit; resumable")
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample", type=int, default=0, help="time this many units spread over the rest")
    p.add_argument("--pilot", action="store_true", help="only the seeded train units a prompt is piloted on")
    p.add_argument("--check", action="store_true", help="the ledger against both rules")
    p.set_defaults(func=cmd_retrieve)
    p = sub.add_parser("fetch-models", help="pinned NLI A, NLI B and CLIP L/14 (network)")
    p.add_argument("--only", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=list(MODEL_PINS))
    p.set_defaults(func=cmd_fetch_models)
    p = sub.add_parser("entities", help="the claims' entities (pinned NER, all labels); cached")
    p.add_argument("--dataset", nargs="+", required=True, choices=sorted(DATASETS))
    p.set_defaults(func=cmd_entities)
    p = sub.add_parser("nli-adoption", help="train figures for the NLI adoption: B with and without "
                                            "the relevance filter, A as the contrast")
    p.add_argument("--dataset", nargs="+", default=["m4fc", "averitec"], choices=["m4fc", "averitec"])
    p.set_defaults(func=cmd_nli_adoption)
    p = sub.add_parser("prompts", help="an id-checked input file for the server")
    p.add_argument("--task", required=True, choices=sorted(TASKS) + ["tc_nli"])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--part", required=True, choices=("pilot", "full", "noenrich"))
    p.set_defaults(func=cmd_prompts)
    p = sub.add_parser("run", help="server: an input file through a model; resumable")
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="nli_b", choices=("nli_b", "nli_a"), help="NLI inputs only")
    p.add_argument("--recheck", type=int, default=0, help="fresh process: N finished rows identical?")
    p.add_argument("--stub", action="store_true", help="no model: checks everything else")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("pilot-report", help="the Qwen pilot: format, labels against gold, copying, "
                                            "the person's 70% stop rule")
    p.set_defaults(func=cmd_pilot_report)
    p = sub.add_parser("score", help="the evaluation sets, per class, beside the baselines")
    p.add_argument("--dataset", nargs="+", default=["m4fc", "averitec", "averimatec"],
                   choices=sorted(DATASETS))
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("verdicts", help="the ensemble's file (once the Qwen and NLI-B runs exist)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_verdicts)
    args = ap.parse_args(argv)
    if getattr(args, "only", None):
        unknown = sorted(set(args.only) - set(MODEL_PINS))
        if unknown:
            ap.error(f"unknown model(s) {unknown}; known: {sorted(MODEL_PINS)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
