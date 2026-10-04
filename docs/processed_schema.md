# The processed layer — one schema for every corpus

Built by `scripts/build_processed.py` in two stages, so derivation is strictly
`data/raw` → `data/interim` → `data/processed`:

```
python scripts/build_processed.py interim --all          # raw -> data/interim/records/<dataset>.jsonl.gz
python scripts/build_processed.py processed --all --reverify
```

Stage `interim` is the only step that reads `data/raw`, through the per-dataset
loaders in `scripts/records.py`. Stage `processed` reads only the interim
records, the image index (`data/interim/image_index/`), the recovery ledgers
and matches, and the placeholder registry. `--reverify` re-decodes every image
file before anything is resolved, so an image counted usable was decoded by the
build that counted it.

## One row per record

`data/processed/records/<dataset>.jsonl.gz` — gitignored; it carries text,
labels and paths into licensed data.

| Field | Type | Meaning |
| --- | --- | --- |
| `dataset` | str | register name |
| `record_id` | str | stable id within the dataset (below) |
| `text` | str \| null | the claim, post, caption, statement or article |
| `evidence_text` | str \| null | the dataset's own evidence, where it ships one: factify2 `document`, mocheg evidence rows (first 20), averitec/averimatec `justification` |
| `label` | str \| null | the dataset's label, unmapped (labels are not harmonised across corpora: their schemes are not commensurable) |
| `reason_code` | `content_refuted` \| `content_unverified` \| `media_mismatch` \| `wrong_image` \| `none` \| `unmapped` \| null | derived from `label` by `scripts/reason_codes.py`, never re-annotated (table below) |
| `modality_evidence` | `text` \| `multimodal` \| null | Factify2's own Text/Multimodal division of Support and Insufficient; null for Refute and for every other corpus |
| `evaluation_only` | bool | true for averimatec (95% one class) and verite (a benchmark); such rows may score a model, never fit one |
| `weak_supervision` | bool | true where the reason_code is synthetic, not fact-checked; such rows may fit a model, never score one. **False everywhere at present**: the one candidate, DGM4's same-publisher text swaps, failed its shortcut audit (docs/data_card.md, "DGM4 text-swap") |
| `split` | `train` \| `val` \| `test` | assigned by `scripts/split_all.py`; the author's split is kept in `meta.official_split` |
| `image_paths` | list[str] | images that passed all three gates at build time; `PROJECT_ROOT`-relative, `archive.zip::member` for zip members |
| `images_expected` | int | images the record references |
| `images_usable` | int | `len(image_paths)` |
| `image_status` | list[{role, status}] | per referenced image: `usable`, a registry status (`placeholder`, `furniture`, `blank`, `verdict` -- the image shows the fact-checker's ruling), `wrong_image` (decodes, but is provably not the image the authors used: `data/fingerprint_rejections.yaml`), `undecodable`, `not_on_disk`, or `dead:<terminal reason>` from the dead-images manifest |
| `modality` | `text+image` \| `text_only` \| `image_only` \| `none` | what is actually present, not what is promised |
| `usable` | bool | label present, not conflicting, text present, and every REQUIRED image usable. MOCHEG's `evidence` images are an optional set: one removed by the registry or rejected as the wrong image is excluded from the set (and listed in `image_status`), not counted missing; a claim / document / pair / post image is a required slot |
| `usable_reason` | str \| null | why not, `;`-joined: `no_label`, `label_conflict`, `no_text`, `images_missing:N` |
| `meta` | dict | dataset-specific extras (dates, source, sampling flags) |

### Record ids

| Dataset | `record_id` | Note |
| --- | --- | --- |
| factify2 | `<csv stem>_<row index>` | matches the published split files and the hydration keys |
| fakeddit | Reddit post id | image only if the post was in the seeded 150,000 sample (`meta.image_sampled`) |
| verite | row index of `VERITE.csv` | |
| m4fc | `image_path` from `M4FC.json` | test rows excluded, as the register requires |
| dgm4 | `<source id>-<sha1(image, text)[:10]>` | DGM4's `id` is shared by every manipulation of one original |
| mocheg | `<split>_<claim_id>` | one row per CLAIM; images are its RELEVANCY=1 evidence images |
| averitec / averimatec | `<split file>_<index>` | averitec test is unlabelled: `usable: false`, `no_label` |
| liar | statement id (`2635.json`) | one record per line (`QUOTE_NONE`) |
| fakenewsnet | FakeNewsNet's news id (`gossipcop-123`, `politifact456`) | the label is the file a row sits in, never part of the id; two ids appear in both label files and are ONE record each, label null, `label_conflict`, unusable |
| welfake | source row index | WELFake is redistributable (CC-BY-4.0) |
| isot | `sha1(subject, date, title, text)[:16]`, `-2`, `-3` for byte-identical repeats | ISOT ships one file per label, so a file-and-row id would publish the label |

**No record id, and no published file name, may encode the label.** VERITE names its
images `true_N` / `false_N`, so every tracked file names a VERITE image by its
row-index alias (`rows-108-109.jpg`, `scripts/records.verite_alias`); the real
paths are in `data/interim/verite_private_map.json` (gitignored).
`scripts/prepush_check.py` refuses a staged file that names a VERITE image by its
label, and a split file of a non-redistributable dataset whose ids carry a label
word.

## Splits

`data/processed/splits/<dataset>_{train,val,test}.csv` carry `record_id,split`
and nothing else — no label, text or URL — and are published.
`scripts/split_all.py` assigns them:

- **factify2** — source-disjoint on label-predictive domains (purity ≥ 0.90),
  generated over all 42,500 records from URL domains, so recovering an image
  never moves a record.
- **verite** — all `test`: it is an evaluation benchmark.
- **official splits** are kept where a dataset ships them (liar, mocheg,
  averitec, averimatec, fakeddit, dgm4, m4fc).
- **welfake, isot, fakenewsnet** — seeded (20260901), class-balanced 70/15/15.

Two constraints override all of the above and are enforced, not checked: every
image with **five or more copies** anywhere, and every image appearing in **more
than one corpus**, keeps all its records in one split. A group that straddles
moves whole to the most held-out split among its members, or to factify2's
placement when it contains a factify2 record. `assert_atomic()` raises if any
group still straddles; `tests/test_split_all.py` holds the contract.

## reason_code

| Value | Meaning | Comes from |
| --- | --- | --- |
| `content_refuted` | checked, false | MOCHEG `refuted`; AVeriTeC `Refuted`; Factify2 `Refute`; M4FC `false`; AVerImaTeC `Refuted` (inherited) |
| `content_unverified` | checked, unsettled | MOCHEG `NEI`; AVeriTeC `Not Enough Evidence`; Factify2 `Insufficient_Text`, `Insufficient_Multimodal`; M4FC `not enough information`; AVerImaTeC (inherited) |
| `media_mismatch` | real image, real caption, wrong pairing | VERITE `out-of-context` |
| `wrong_image` | genuine image described falsely | VERITE `miscaptioned` |
| `none` | adjudicated true, no defect | VERITE `true`; MOCHEG `supported`; AVeriTeC `Supported`; Factify2 `Support_Text`, `Support_Multimodal`; M4FC `true`; AVerImaTeC (inherited) |
| `unmapped` | the dataset's labels carry no reason information | every row of WELFake, LIAR, FakeNewsNet, Fakeddit, ISOT, DGM4 |
| null | a reason-bearing scheme, but this label fits no code | AVeriTeC/AVerImaTeC `Conflicting Evidence/Cherrypicking`; unlabelled rows |

`none` and `unmapped` are never merged: the first is a verdict, the second the
absence of one. **Only the five supervised codes may train or evaluate a
reason-code head**, and `scripts/processed_loader.reason_code_rows()` enforces
it: it refuses an all-`unmapped` dataset outright, never yields an `unmapped` or
null row, raises if an `evaluation_only` row is asked for training, and never
yields a `weak_supervision` row for evaluation.

Fakeddit is `unmapped` throughout, and its `fakehistoryporn` rows are excluded
from reason-code supervision under any future mapping
(`EXCLUDED_FROM_REASON_SUPERVISION`): the subreddit is a joke format and its
class is a source identifier (docs/data_card.md, "Fakeddit").

### Cross-dataset transfer is always reported

A reason-code head trained on one corpus and scored on another reports the
**in-domain** number beside the **cross-dataset** number, and the gap between
them, every time. Above all: **never train on DGM4 and evaluate on VERITE
without the transfer gap.** DGM4's mismatches are synthetic (a caption moved
between real news photos by an algorithm); VERITE's are real-world (curated
from fact-checks). Synthetic-to-real is the weakest transfer case there is.
Published out-of-context work reports drops of roughly 30 points across that
boundary, and this project's own WELFake↔ISOT text transfer gave macro F1
0.251 (figure stated by the project lead on 2026-10-02; the run is not in this
repository). A VERITE number reported alone, after DGM4 training, reads as
"detects real out-of-context media", which nothing has shown.
