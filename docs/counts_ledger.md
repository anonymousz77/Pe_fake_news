# Counts ledger

Row and file counts at every pipeline stage. The point of this file is
arithmetic that reconciles: if `raw` minus `dropped` does not equal `interim`,
something is wrong and this file is where it becomes visible.

**Status: nine of ten enabled datasets acquired** (2026-09-01/02); Part B
closed 2026-09-03. Only `isot` is outstanding, blocked on Kaggle credentials.
Counts below are measured from what actually landed, not from any publication.

## Stage definitions

| Stage | Location | Meaning |
| --- | --- | --- |
| raw | `data/raw/<name>/` | Exactly as downloaded. Immutable. Never edited in place. |
| interim | `data/interim/<name>/` | Parsed into a common schema; nothing dropped yet |
| processed | `data/processed/` | Cleaned, deduplicated, filtered |
| splits | `data/processed/splits/` | Final train / dev / test partitions |

## Per-dataset counts

`expected` comes from `data/sources.yaml` and is the number to reproduce.
Everything else is filled in on acquisition.

| Dataset | expected | raw rows | interim rows | dropped | reason | processed rows |
| --- | --- | --- | --- | --- | --- | --- |
| mocheg | — | **21,184 claims / 43,148 text evidence** | — | — | — | — |
| factify2 | — | **85,000 image refs from 42,500 rows** | — | — | — | — |
| averitec | — | train/dev/test JSON + dev KS | — | — | — | — |
| averimatec | UNVERIFIED | 6 files | — | — | — | — |
| verite | 338 / 324 / 338 | 1,377 files | — | — | — | — |
| fakeddit | — | **621,635 candidate rows** | — | — | — | — |
| welfake | 72,134 usable | 1 CSV | — | — | — | — |
| liar | — | **12,791** | — | — | — | — |
| isot | 21,417 / 23,481 | NOT FETCHED | — | — | — | — |
| fakenewsnet | — | 17 files | — | — | — | — |

`visualnews` and `newsclippings` are deferred (`enabled: false`) and are
deliberately absent from this table.

### Measured counts, all ten enabled datasets

Every figure below was read off the files on disk on 2026-09-03.

`averimatec` is 0.207 GB / 4 files after its held-out `test_data.zip` (30 MB)
was forbidden and removed — see [data_card.md](data_card.md).

| Dataset | Measured GB | Files | Rows / claims | Against `expected` |
| --- | --- | --- | --- | --- |
| mocheg | 1.900 | 167 | 21,184 claims / 43,148 text evidence | matches Zenodo exactly |
| factify2 | 9.760 | 79,257 | 42,500 rows, 38,425 complete (90.4%) | no `expected` declared |
| averitec | 10.758 | 4 | train 3,068 / dev 500 / test 2,215 claims | no `expected` declared |
| averimatec | 0.207 | 4 | train 793 / val 152 claims | test archive now forbidden + removed |
| verite | 0.162 | 1,870 | 1,001 rows; 721 images (72.0%) | investigated — genuine, register corrected |
| fakeddit | 15.072 | 142,436 | 621,635 candidates, 150,000 sampled, 142,434 images | no `expected` declared |
| welfake | 0.228 | 1 | 72,134 rows (real 35,028 / fake 37,106) | matches `expected` exactly |
| liar | 0.004 | 5 | 12,791 rows | no `expected` declared |
| isot | — | 0 | NOT FETCHED | untested |
| fakenewsnet | 0.041 | 17 | 17 files (crawler index only) | no `expected` declared |

**Total measured: 38.06 GB** across 223,269 manifest entries.

### verite — image hydration, 2026-09-03

Images are not shipped with the corpus; 663 distinct images were fetched from
`true_url`/`false_url` in `VERITE_articles.csv`. Recovered **721 of 1,001 rows
(72.0%)**: out-of-context 263/325 (80.9%), true 229/338 (67.8%), miscaptioned
229/338 (67.8%). `true` and `miscaptioned` share one image per index, so their
rates match by construction. Loss is blocking rather than deletion — 198 of 285
first-pass failures were HTTP 403, 127 of them `media.snopes.com`.

### verite — investigated: the extra row is genuine, and it is kept

`data/VERITE/VERITE.csv` holds **1,001** rows against a published 1,000:

| Class | Published | Measured |
| --- | --- | --- |
| true | 338 | **338** ✓ |
| miscaptioned | 338 | **338** ✓ |
| out-of-context | 324 | **325** — one more |

Checked for every artefact that could fake an extra row, on 2026-09-03:

- **Duplicate row?** No. Zero exact duplicates.
- **Trailing blank line?** No. 1,002 physical lines = header + 1,001; no blank
  lines anywhere; the file ends with a single newline.
- **Header off-by-one?** No. `csv.DictReader` yields 1,001 rows and the header
  parses as 4 columns (an unnamed index, caption, image_path, label).
- **Index integrity?** The index column runs **contiguously 0..1000**, 1,001
  unique values, no gaps and no repeats.
- **Independent confirmation?** All three shipped files —
  `VERITE.csv`, `VERITE_with_evidence.csv` and
  `VERITE_ranked_evidence_clip_ViTB32.csv` — agree on 338 / 338 / 325.

(`image_path` and `caption` do repeat, 338 and 326 times, but that is the
corpus design: a true image is re-paired with miscaptioned and out-of-context
variants. Those are not duplicate records.)

**Conclusion: a genuine 1,001st record.** The register's `expected` now reads
325 because it must describe what we have; the published 324 is recorded in the
entry's notes. **The row is kept — data is not dropped to match a paper.**

### welfake — `expected` corrected to describe the download

Measured **72,134 rows: label 0 (real) 35,028, label 1 (fake) 37,106**, summing
exactly. All three figures matched the register already.

The one wrong field was `csv_rows: 78098`. That is the source's *pre-filtering*
count and describes no file we hold — the Zenodo distribution already ships the
filtered subset, so the 5,964-row drop happened upstream and is not
reproducible here. It has been removed from `expected` and moved to the entry's
notes, and `usable_rows` renamed to `rows` since there is no unfiltered variant
to distinguish it from.

The rule this establishes: **`expected` describes what the download contains.**
A figure that describes the source rather than the file belongs in notes.

### MOCHEG — the version question, answered

The register recorded two conflicting count sets. Measured from the Zenodo
release (`mocheg_v1_without_image.tar.gz`, record 6653772):

| Quantity | Measured | Zenodo reported | SIGIR paper |
| --- | --- | --- | --- |
| Unique claims | **21,184** | 21,184 | 15,601 |
| Text evidence rows (Corpus2) | **43,148** | 43,148 | 33,880 |
| Image evidence qrels | **88,040** rows, 16,515 RELEVANCY=1 | 15,373 | 12,112 |

Per split: train 36,358 evidence / 18,583 claims; val 1,562 / 600;
test 5,228 / 2,001.

**This release matches the Zenodo figures exactly** for claims and text
evidence, so it is the Zenodo version, not the smaller one the SIGIR paper
describes. The article-level text qrels total 33,756, which is close to the
paper's 33,880 — the two figure sets appear to count different views of
overlapping data. Both are recorded here; neither is discarded.

Two gaps, recorded rather than worked around:

- The tarball is named `_without_image` and **contains no images**, so MOCHEG's
  image evidence is not present in this distribution at all.
- **No CSV carries a tweet-id column** (`tweet_id`, `tweetid`, `id`,
  `twitter_id` all absent), and only 7 rows in train mention Twitter in their
  Source field. The "2,916 ID-only tweets" are not in this release, so tweet
  hydration has no target here. This is a distribution gap, not a fetch failure.

### Fakeddit — sampling and image recovery

- Candidate rows with an `image_url`: **621,635** (train + validate; the test
  TSVs are held out by `forbidden_patterns`).
- Seeded stratified sample: **150,000** (seed 20260901), per 6-way class
  0=58,925  1=8,909  2=28,503  3=3,140  4=44,801  5=5,722.
- Images recovered: **142,434 / 150,000 = 94.96%**, per class 92.3%–99.3%.
  Failures are dominated by 7,350 HTTP 404s — genuinely deleted Reddit content,
  spread across classes rather than concentrated in one.

### Measured on acquisition

**liar**, 2026-09-01: train 10,240 + valid 1,284 + test 1,267 = **12,791** rows,
14 columns each. Nothing dropped yet — these are raw counts straight from the
TSVs, before any parsing or cleaning.

### Reconciliations that must hold

- **welfake**: `csv_rows` 78,098 − dropped 5,964 = `usable_rows` 72,134. The
  drop is expected; the *reason* for each dropped row still has to be recorded.
  Separately, real 35,028 + fake 37,106 = 72,134 — the class counts must sum to
  the usable rows, not the CSV rows.
- **verite**: 338 + 324 + 338 = 1,000 total.
- **isot**: 21,417 + 23,481 = 44,898 total.

If any of these fails to reconcile on ingest, stop and investigate before
building anything downstream.

## Split sizes

| Split | Rows | Sources | Notes |
| --- | --- | --- | --- |
| train | — | — | — |
| dev | — | — | — |
| test | — | — | — |

## Rules

- Every dropped row needs a **reason**, and the reasons must sum to the drop
  count. "Cleaning" is not a reason.
- Update this file in the **same commit** as the code that changed a count. A
  ledger reconstructed later is a guess wearing a table's clothes.
- Held-out test rows never appear in the train or dev counts. If the arithmetic
  says otherwise, stop and check the register's `forbidden_patterns` before
  changing anything else.
- **isot rows never contribute to a headline number.** It is a negative control
  (see [data_card.md](data_card.md)); count it separately and report it
  separately.
- Record absolute dates for each recount.
