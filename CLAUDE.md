# Pe_Fake_News_Dec

Multimodal fake-news / claim-verification research scaffold. This file is the
standing brief for anyone — human or agent — working in this repo.

**The data layer is closed (2026-10-03).** What it holds, what it lost and why,
and what remains open are in `docs/completion_report.md`. Read that before
changing anything under `data/`.

## The four invariants

1. **No hardcoded paths, anywhere.** `configs/paths.py` is the only module that
   knows what the filesystem looks like. Import from it:

   ```python
   from configs.paths import RAW, BIBLE, raw_dir, free_gb, ensure_tree
   ```

   The project root comes from `PE_FAKE_NEWS_ROOT` (environment first, then a
   `.env` at the repo root, then the repo root itself). Data may live on a
   different volume than the code; code that assumes otherwise is broken.

2. **`data/sources.yaml` is the register, and it is the source of truth.** It
   holds exactly fourteen datasets, twelve of them enabled — the test asserts
   the exact *name set*, not the count, because the right number of wrong
   datasets is a real failure mode this project has already hit once. Nothing
   gets downloaded, parsed, or referenced by a name that is not in the register;
   `raw_dir()` raises `KeyError` on an unknown name specifically to stop that.

3. **Test data does not enter the pipeline.** Every register entry declares
   `forbidden_patterns`, *required even when empty*, because an absent key means
   nobody considered leakage. `include_patterns` is optional and its absence
   means "take everything" — which counts as an implicit `*` for the guard, so
   an entry with no includes may not forbid anything. Includes and forbidden
   globs must not overlap in either direction. `tests/test_sources.py` enforces
   all of this and fails the build if a glob is widened into a held-out split.

4. **A fetch that validates on status code instead of content is not a fetch.**
   This is the same mistake as counting datasets instead of naming them: the
   right *number* of wrong datasets passes a cardinality check, and the right
   *status* on the wrong bytes passes an HTTP check. Both gates measure a proxy
   and report it as the thing.

   It has cost this project repeatedly. Accepting any non-empty 200 wrote 97
   HTML error pages to disk as `.jpg`. Accepting any *decodable* body let
   thousands of well-formed images through that depict nothing: imgur's "this
   image does not exist" bitmap, tip-line banners, flat colours. And accepting
   any image that is not a placeholder let through **the answer itself**:
   fact-check articles illustrate themselves with their own ruling (a
   Truth-O-Meter, a FALSE stamp, a fact-checker's seal), and MOCHEG and Factify2
   take their images from those articles. On MOCHEG an image-only
   nearest-neighbour vote scores 91.7% against a 36.0% majority on the records
   that carry one.

   So there are three gates, and each catches what the one before it cannot:

   ```
   status 200          →  says the host answered, nothing more
   looks_like_an_image →  says the bytes decode        (catches the error pages)
   not_a_placeholder   →  says the bytes are evidence  (catches placeholders,
                                                         logos, blanks, verdicts)
   ```

   `data/placeholders.yaml` is the content-hash registry behind the third gate,
   versioned (v13: 4,135 hashes) because the list grows. Statuses `placeholder`,
   `furniture`, `blank` and `verdict` are a **missing image**: rejected at fetch
   time as a retryable failure, and not counted in any coverage number;
   `generic_stock` is recorded and stays usable. `data/fingerprint_rejections.yaml`
   adds what a hash cannot express — real images that are provably not the one
   the authors used (checked against fingerprints a dataset ships), keyed by path
   *and* hash. `scripts/images.status()` applies both; everything that counts
   an image goes through it. `tests/test_placeholders.py` holds the contract.

   The general rule: **validate the artefact you actually wanted, not the
   transport that delivered it.** When a check is cheap and its proxy is
   cheaper, the proxy is what silently rots the numbers. (It recurred inside
   the analysis code: `scripts/confound.py` counted any file that *existed* as
   recovered media until 2026-10-03.)

## The fourteen datasets

| Dataset | Method | GiB on disk | Files | Acquired | Usable rows / rows |
| --- | --- | ---: | ---: | --- | --- |
| mocheg | zenodo + image release | 40.352 | 137,830 | 2026-09-02; images 2026-10-01 | 21,184 / 21,184 |
| factify2 | gdrive | 11.042 | 84,653 | 2026-09-02 | 37,807 / 42,500 |
| averitec | hf (`repo_type: model`) | 10.758 | 11 | 2026-09-02 | 3,567 / 5,783 (test unlabelled) |
| averimatec | hf (`repo_type: dataset`) | 0.209 | 14 | 2026-09-02 | 944 / 945 |
| verite | git | 0.213 | 2,023 | 2026-09-02 | 914 / 1,001 |
| fakeddit | gdrive | 15.763 | 148,815 | 2026-09-02 | 615,822 / 623,342 |
| welfake | direct (Zenodo) | 0.228 | 1 | 2026-09-02 | 72,134 / 72,134 |
| liar | direct | 0.004 | 5 | 2026-09-01 | 12,836 / 12,836 |
| isot | direct (UVic) | 0.149 | 3 | 2026-10-01 | 44,898 / 44,898 |
| fakenewsnet | git | 0.059 | 49 | 2026-09-02 | 23,192 / 23,194 |
| dgm4 | hf | 10.205 | 35 | 2026-10-01 | 179,295 / 179,295 (train+val) |
| m4fc | git + hydration | 1.125 | 4,980 | 2026-10-01 | 3,441 / 3,493 (test excluded) |
| visualnews | git | — | — | deferred | — |
| newsclippings | git | — | — | deferred | — |

`data/raw` holds 96.75 GB. The register's `est_gb` values are measurements in
GiB (the originals were wrong by up to 200x and are preserved in each entry's
notes) and sum to 89.86 against the 120 budget.

`averitec` is an HF **model**-type repo despite holding a dataset. Fetching it
as `repo_type: dataset` returns 404; do not "correct" it.

### Rules that are not negotiable

- **isot is a negative control.** All its real news is Reuters, so a model can
  score well by identifying the publisher rather than detecting deception.
  Never report an ISOT number as a headline result.
- **verite is load-bearing.** It is the only real-world out-of-context
  *evaluation* set (914 of 1,001 rows usable). It is evaluation-only, always.
- **37 VERITE records share an image with Factify2, M4FC or MOCHEG.** Drop them
  from any VERITE evaluation of a model trained on those corpora;
  `placeholders.assert_no_cross_corpus_leak()` raises otherwise.
- **Never train on DGM4 and evaluate on VERITE without the transfer gap** —
  the in-domain number, the cross-dataset number, and the difference, every
  time. Synthetic-to-real is the weakest transfer case there is.
- **No published identifier or file name carries a label.** VERITE's image
  names (`true_N` / `false_N`) are its labels: tracked files use the row-index
  alias, the real paths stay in `data/interim/verite_private_map.json`. ISOT and
  FakeNewsNet record ids are label-free. `scripts/prepush_check.py` enforces both.
- **Fakeddit's labels are its subreddits.** All 22 are 100% one label, and the
  title identifies the subreddit. Read `docs/data_card.md` before reporting any
  Fakeddit number.

## Factify2: standing rules

These are rules for all later work, not commentary. Image recovery was
**exhausted** in the closeout pass (2026-10-01..03) on an explicit instruction
that superseded the earlier decision to stop at 1,278 archive images; every
remaining image has a terminal reason in `data/dead/factify2.csv`. The two
defects below are controlled in the experimental design, not in the data.

**1. Every Factify2 result is reported beside the applicable baseline.**
A hostname-only classifier — no pixels, no text, just the image URL's host —
scores **40.3%** on a random split against a 20.0% uniform baseline. On the
source-disjoint split it scores 12.7%, but that split's test set is 54.4%
Refute, so there the comparator is the **56.9% test majority**. A model not
clearly above the applicable figure is reading hostnames or class priors, not
content. Print it next to every number.

**2. Splits are source-disjoint on LABEL-PREDICTIVE domains, not on all
domains.** Threshold: class purity ≥ 0.90. Generated by `scripts/split_all.py`
(which calls `scripts/splits.py`), enforced, failing loudly on a violation.

Do not "strengthen" this to all domains. It was tried and it is impossible:
records linked by shared domains form a single component of 95.8% of records,
because `pbs.twimg.com` touches 84.7% of records and carries no label signal
(26% Support_Multimodal). The leak is `factly.in`, `boomlive.in`,
`images.thequint.com` and `snopes.com`, each 100% Refute. `scripts/splits.py
--disjoint strict` still exists and still refuses, so the finding stays
reproducible.

**3. Per-class metrics always. Never a single averaged Factify2 number.**
Coverage, completeness and class balance all differ by class.

**4. Permanent limitations, stated in any write-up:**
- Refute **image** coverage 81.6%; every other class 92.3–99.7%.
- Refute **row** completeness 66.3% (a row needs both its images); others
  85.1–99.4%.
- **`val` contains 0 Refute records.** Refute sits in a few indivisible domain
  components; Refute model selection must use cross-validation within train,
  never val.
- **1,486 Refute records carried a verdict image** (BOOM's seal, Snopes's badge)
  — the hostname confound in pixels. Those images are removed by the registry,
  and the rows they leave incomplete are not usable.

## Reason codes

`reason_code` is derived from each dataset's label by `scripts/reason_codes.py`
and never re-annotated: `content_refuted`, `content_unverified`,
`media_mismatch`, `wrong_image`, `none` (adjudicated true), `unmapped` (the
label scheme carries no reason information), or null. The rules live in
`docs/processed_schema.md`. **`scripts/processed_loader.reason_code_rows()` is
the only sanctioned way to feed a reason-code head**: it refuses unmapped
datasets outright, never yields unmapped or null rows, refuses evaluation-only
corpora (VERITE, AVerImaTeC) for training, and never yields weak-supervision
rows for evaluation. `media_mismatch` and `wrong_image` have **no training
source**: Fakeddit and DGM4 both failed their shortcut audits.

## Layout

```
configs/paths.py           path resolution; the only module that names directories
scripts/manifest.py        register audit + sha256 integrity manifest
scripts/fetchlib.py        shared acquisition machinery: disk model, guards, markers
scripts/fetch.py           network acquisition, one handler per method
scripts/stream_archive.py  selective streaming extraction from a remote tar.gz
scripts/hydrate.py         URL-referenced media + recovery reports
scripts/recover.py         image recovery: origin retry, Wayback, alternates; dead manifests
scripts/images.py          image index: every file through the three gates
scripts/placeholders.py    placeholder registry + cross-corpus duplicates and leak guard
scripts/match_images.py    offline recovery: URL identity, CLIP fingerprints (VERITE)
scripts/content_probe.py   CLIP probe that RANKS images for review; never registers
scripts/ocr_audit.py       OCR over images, flags error cards for review
scripts/records.py         per-dataset loaders into one record shape
scripts/build_processed.py raw -> interim -> processed, images re-verified
scripts/split_all.py       splits for every dataset; atomicity enforced
scripts/splits.py          source-disjoint generator + leakage guard (Factify2)
scripts/reason_codes.py    label -> reason_code, per dataset
scripts/processed_loader.py the processed layer, supervision rules enforced
scripts/confound.py        domain/label shortcut diagnostic
scripts/fakeddit_classes.py / dgm4_audit.py / clip_pairs.py   shortcut audits
scripts/verify.py          measure what we hold, against the register
scripts/completion_report.py  assemble the completion report from stage reports
scripts/bible.py           evidence-store build: fetch, date, quarantine, index, report
scripts/retrieve.py        time-filtered retrieval; the date predicate lives in the SQL
scripts/redtest.py         red-teams that predicate against the REAL index, then rolls back
tests/                     contract tests, offline
data/sources.yaml          the register (tracked)
data/placeholders.yaml     content hashes of evidence-free images (tracked, versioned)
data/fingerprint_rejections.yaml  real images that are not the authors' image (tracked)
data/cross_corpus_duplicates.yaml byte-identical images across corpora (tracked)
data/dead/<name>.csv       dead-image manifest: key + terminal reason (tracked)
data/MANIFEST.sha256       integrity record for raw data (tracked)
data/FETCH_LOG.jsonl       append-only acquisition log (tracked — it is provenance)
data/raw/<name>/           immutable downloads, one dir per register entry (ignored)
data/raw/<name>/media/provenance.jsonl  every recovery attempt, URL/status/host/outcome
data/interim/              intermediate artefacts: image index, OCR, CLIP, records (ignored)
data/interim/<name>_rejected/  quarantined non-images (ignored, nothing deleted)
data/processed/records/    the processed layer (ignored)
data/processed/splits/     <name>_{train,val,test}.csv, record_id,split only
data/reports/              generated tables and figures
data/bible/                the evidence store (ignored)
docs/                      hand-written prose (tracked)
```

### `data/bible/` is not documentation

The "bible" is the **evidence store**: the time-filtered corpus of dated
documents that claims get checked against. It is data, it is gitignored, and it
is expected to reach tens of GB. Prose that a human wrote — the data card, the
counts ledger, provenance notes — goes in `docs/` and is tracked in git. Do not
put writing in `data/bible/`, and do not put evidence in `docs/`.

## Budgets

- Enabled datasets must estimate **≤ 120 GB** (`sum(est_gb)` over
  `enabled: true`). Enforced by `tests/test_sources.py` and `manifest.py audit`.
- The machine runs under a hard **160 GB** ceiling overall. Call `free_gb()`
  before any acquisition.
- Enabling visualnews + newsclippings (68 GB together) requires raising
  `budget_gb` in the register **and** `BUDGET_GB` in `scripts/manifest.py` in
  the same change. It is a budget decision, not a flag flip.
- This machine has 16 GB of RAM. Claude Code reaps background jobs under memory
  pressure: run long jobs (crawls, GPU passes, full builds) **one at a time**,
  and never restart a reaped job without asking.

### The committed-disk model

`fetch.py` gates every dataset twice before a byte moves:

```
projected = raw_on_disk_now + remaining_est_gb + 30 (data/bible) + 20 (interim/processed)
refuse if projected > 140 GB          # names which dataset to cut
refuse if free_gb() < 1.5 * est_gb    # per dataset, before it starts
```

A third gate catches what patterns cannot: after enumerating a fetch, the
resolved selection is compared to `est_gb`. More than 3x the estimate aborts as
a wrong fetch scope rather than a stale number.

## Acquisition

- Never fetch an HF repo unfiltered when its entry declares
  `forbidden_patterns` — `allow_patterns_for()` refuses, because "take
  everything" reaches the forbidden paths too.
- `forbidden_patterns` are enforced **at fetch time**, not just in tests. Every
  method enumerates before downloading (`list_repo_files`, `git ls-tree` on a
  blobless fetch, `SevenZipFile.getnames()`, `download_folder(skip_download)`),
  and a match aborts the whole dataset.
- `--dry-run` is entirely offline and never opens a socket.
- Fakeddit images are **sampled, never taken wholesale**: a seeded stratified
  sample across the 6-way label, index written to `data/interim/fakeddit/`.
- Recovery runs **one worker, one request per 5 s per host** (at 12 workers we
  measured 53.3% errors; at 1 req/5s, 6.0%). It is resumable from the
  provenance ledger; an archive index that is *unreachable* is retried on every
  run, one that *refuses* (HTTP 404) is final.
- Re-crawling refused items does not pay: the archive returns the same refused
  content re-encoded (13 genuine images among 453 recovered, 2026-10-03).
- Hydration reports must carry `recovery_rate_per_class`. If the label column
  cannot be found, `hydrate.py` aborts rather than writing a report without it —
  an overall 85% hides a class that recovered 20%.
- Kaggle credentials: this version (2.2.4) does **not** read
  `~/.kaggle/kaggle.json` first. It wants `kaggle auth login`,
  `KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`.
- New corpora: acquire only under an explicit licence; skip and report anything
  behind a form, a login, an access request or terms. Never sign, email or
  accept terms on the project's behalf.

## What has been verified

Every enabled dataset is acquired and measured (`scripts/verify.py --all`; one
known mismatch: M4FC claims 6,789 against 6,980 stated). Every licence was read
at source and is quoted verbatim in `docs/licences.md`; `redistributable` is
true only where that text grants it, and split files publish `record_id,split`
only. `docs/provenance.md` records what was checked by hand — keep it honest.
**Do not invent counts, versions, or licence terms.** Where a value is unknown,
write `null` or `UNVERIFIED — confirm before download`.

## Working rules

- **Offline by default.** Tests must never hit the network. `manifest.py audit`
  runs with no data on disk.
- **Raw data is immutable.** Downloads land in `data/raw/<name>/` and are never
  edited in place; derived artefacts go to `interim/` or `processed/`.
  Rejected files are quarantined, never deleted.
- **The registry is curated by eye.** `scripts/content_probe.py` and OCR rank
  candidates; a person (or an agent, viewing the image at full size) decides.
  Thumbnail-only calls over-called by 19 in 799 when re-checked.
- Regenerate the integrity manifest after any acquisition:
  `python scripts/manifest.py build`, then commit the diff.
- Before committing register changes: `python scripts/manifest.py audit && pytest -q`.

## Commands

```bash
python scripts/manifest.py audit                    # register self-check; no data required
python scripts/manifest.py build                    # hash all of data/raw/
python scripts/manifest.py build --dataset liar     # rehash just one, merge in
python scripts/manifest.py verify                   # check disk against the manifest
pytest -q                                           # all contract tests, offline

python scripts/fetch.py --all --dry-run             # plan + disk projection, no network
python scripts/fetch.py --dataset liar --yes        # cheapest real fetch; the smoke test
python scripts/verify.py --all                      # measure what we hold vs the register

python scripts/images.py index --all                # three-gate index of every image
python scripts/recover.py run --all --budget-gb 2   # resumable recovery crawl
python scripts/recover.py dead                      # dead-image manifests
python scripts/match_images.py clip-verify          # VERITE images vs the authors' fingerprints
python scripts/content_probe.py --status verdict --pool mocheg   # rank images for review
python scripts/placeholders.py cross-corpus         # regenerate cross-corpus duplicates

python scripts/build_processed.py interim --all     # raw -> interim records
python scripts/build_processed.py processed --all --reverify   # interim -> processed, splits
python scripts/reason_codes.py                      # reason_code distribution
python scripts/completion_report.py                 # data/reports/completion.json + tables

python scripts/confound.py --dataset factify2       # domain/label shortcut, usable images
python scripts/dgm4_audit.py                        # DGM4 shortcut audit + decision rule
python scripts/fakeddit_classes.py                  # Fakeddit subreddit confound

python scripts/bible.py --source ccnews --warc-limit 40   # resumable; skips done WARCs
python scripts/bible.py --requarantine --dry-run          # plausibility window, no writes
python scripts/bible.py --requarantine                    # apply it to the existing store
python scripts/bible.py --index                           # rebuild bm25 + temporal
python scripts/bible.py --scan                            # resumable; repeat until done
python scripts/bible.py --report                          # data/reports/bible.json
python scripts/retrieve.py --query "vaccine safety" --as-of 2020-01-01 -k 10
python scripts/redtest.py                                 # time filter vs the real index
python scripts/redtest.py --skip-corpus                   # same, without the slow probe
```

### `--scan` before `--report`, on a small machine

The report needs four fields per document, and reading them meant reading all
8.6 GB of the store. On this 16 GB machine that fills the file cache until the
OS reports a few hundred MB free, and the report was killed for memory three
times while itself using 217 MB — the process was never the problem. Free memory
returns as soon as a process exits, so `--scan` reads the store in resumable
slices and writes ~90 bytes per document to `data/interim/`; `--report` then
reads 120 MB instead of 8.6 GB and takes seconds.

Run `--scan` repeatedly until it prints `done` — it resumes from its own line
count and exits 3 while there is more to do. `--report` falls back to walking
the store when there is no complete, current scan, so the scan is an
optimisation and never a source of truth. Delete the files in `data/interim/`
to force a fresh one.

### The evidence store's one invariant

**A document with no publication date is quarantined and never indexed**, and
**a date that cannot be one does not count as having a date.** A document
stamped `1970-01-01` satisfies `published_at <= :as_of` for every claim ever
made, so a page whose real date is unknown gets served as evidence for all of
them -- including claims it actually postdates. That is the same hindsight leak
the quarantine rule exists to prevent, so `make_document` refuses any date
outside 1900..today, and refuses the sentinel values that sit *inside* that
window: the Unix epoch and 32-bit `INT_MIN`. The rejected value is kept on the
record as `raw_date_rejected`; nothing is destroyed.

Two carve-outs, both deliberate. Jalali and Buddhist-era years are real dates in
another calendar, so they are kept, flagged and reported as a known issue rather
than deleted. And an isolated old date is **not** a broken one: every pre-1996
date in this store that is not a sentinel was checked by hand and is a genuine
archival article whose URL carries the same date. Do not tighten the window to
make the per-year table look tidy -- it would delete correct data.

The time filter itself is a predicate *inside* `retrieve.SEARCH_SQL`, not a pass
over the results. `scripts/redtest.py` proves it on the real 1,368,613-document index by
planting a future-dated and an undated document, sweeping `k` past the corpus
size, and rolling the plants back in a `finally`. It never commits. Re-run it
after any change to the retrieval SQL or the index build.

Re-running any fetch is safe: completed datasets skip via a `.done` marker
holding a tree content-hash, and interrupted downloads resume.
