# Provenance

Where each corpus actually came from, and what a human has verified by hand.

Scaffold created 2026-09-01. **Nine of the ten enabled datasets have been
acquired** (2026-09-01/02); only `isot` is outstanding, blocked on Kaggle
credentials. A row that says "fetched" means the endpoint resolved and served
data — it does **not** mean anyone has read the licence.

Except where the ledger below says otherwise, the URLs in `data/sources.yaml`
have **not** been fetched or confirmed. Treat them as leads, not facts, until
the corresponding row says otherwise. Licence fields carrying "UNVERIFIED" mean
nobody has read the terms — including for `liar`, whose file downloaded fine but
whose terms nobody has read.

`redistributable: false` is set on every entry as the conservative default. It
is not a finding — it is the absence of one. Flip an entry to `true` only after
reading the licence and recording it here.

## Ledger

| Dataset | Method | Acquired (date) | URL confirmed | Licence read | Counts match `expected` | Verified by |
| --- | --- | --- | --- | --- | --- | --- |
| mocheg | zenodo | **2026-09-02** | **confirmed** | **CC-BY-4.0 (read from Zenodo API)** | see counts_ledger | automated fetch |
| factify2 | gdrive | **2026-09-02** | **confirmed** | unverified | see counts_ledger | automated fetch |
| averitec | hf | **2026-09-02** | **confirmed** | **CC-BY-NC-4.0, accepted via --accept-licence** | n/a (expected null) | automated fetch |
| averimatec | hf | **2026-09-02** | **confirmed** | unverified | n/a (counts UNVERIFIED) | automated fetch |
| verite | git | **2026-09-02** | **confirmed** | unverified | not yet checked | automated fetch |
| fakeddit | gdrive | **2026-09-02** | **confirmed** | unverified | see counts_ledger | automated fetch |
| welfake | direct | **2026-09-02** | **confirmed** | CC-BY (register) | not yet checked | automated fetch |
| liar | direct | **2026-09-01** | **confirmed** | unverified | n/a (expected null) | automated fetch |
| isot | kaggle | — | unverified | unverified | — | — |
| fakenewsnet | git | **2026-09-02** | **confirmed** | unverified | n/a (expected null) | automated fetch |
| visualnews | git | — (deferred) | unverified | unverified | n/a | — |
| newsclippings | git | — (deferred) | unverified | unverified | n/a | — |

## Counts to check on ingest

Three entries carry `expected` counts in the register. These are the only
numbers in this project that came with the specification; reproduce them
exactly or investigate the difference before proceeding.

| Dataset | Expected |
| --- | --- |
| verite | `true` 338, `ooc` 324, `miscaptioned` 338 |
| welfake | usable_rows 72,134 of csv_rows 78,098; real 35,028, fake 37,106 |
| isot | real 21,417, fake 23,481 |

`averimatec` explicitly has **no** expected counts — the register records
"Counts UNVERIFIED — confirm on download". Fill them in here when they are
known; do not guess them beforehand.

## What acquisition has actually established

**liar** — fetched 2026-09-01 from
`www.cs.ucsb.edu/~william/data/liar_dataset.zip` by `scripts/fetch.py`. The URL
resolved and served a valid zip, so the URL is confirmed. Five files landed
(the zip plus train/valid/test TSVs and README), hashes recorded in
`data/MANIFEST.sha256`, and the fetch is logged in `data/FETCH_LOG.jsonl`.

Row counts, measured not assumed: train 10,240 / valid 1,284 / test 1,267 =
**12,791** rows, 14 columns each. That confirms the register's "12.8k" note.

The **licence remains unverified**: the file downloaded, but nobody has read
Wang (ACL 2017)'s terms. Downloading a file is not reading its licence.

**averitec** — the repo id `chenxwh/AVeriTeC`, `repo_type: model`, and the
CC-BY-NC-4.0 licence were supplied and confirmed by the user on 2026-09-01, not
fetched. `repo_type: model` is deliberate: the repo holds a dataset but is a
model-type repo, and fetching it as `dataset` returns 404.

**fakeddit** — `est_gb` corrected 42 -> 8 on 2026-09-01. The 42 GB figure
assumed the full ~1M image set; only a ~150,000-image stratified sample is
fetched. Recompute if `--sample-size` changes.

### A note on the fetch log

`data/FETCH_LOG.jsonl` contains one `status: aborted` entry for `liar` dated
2026-09-01. That was a **deliberate red-test** of the fetch-time forbidden-path
guard: `liar_dataset.zip` was temporarily added to the entry's
`forbidden_patterns` to confirm the whole dataset aborts before downloading. The
register was reverted byte-identically afterwards. The entry is left in the log
because the log is append-only and an edited provenance record is worth nothing.

## Source corrections made on 2026-09-02

Three of the original URLs were wrong, and all three failed loudly rather than
silently — the fetch layer refused rather than downloading something plausible.

| Dataset | Was | Now | Why |
| --- | --- | --- | --- |
| mocheg | `git github.com/VT-NLP/Mocheg` | `zenodo zenodo.org/records/6653772` + aux git `github.com/PLUM-Lab/Mocheg` | The GitHub repos are **code only**; include patterns matched 0 of 148 files. Data is on Zenodo. |
| fakeddit | `git github.com/entitize/Fakeddit` | `gdrive .../folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm` | The repo holds only `image_downloader.py` and a README (5 files). **URL taken from that README**, not guessed. |
| factify2 | `direct aiisc.ai/defactify2/` | `gdrive .../folders/13JwnIBzDfe8a5E1anPkt7J90r4NBIYES` | The old value was the shared-task landing page; gdown read "defactify2" as a folder id and 404'd. |

`averitec` also had its `include_patterns` corrected: the dev knowledge store is
a single file, `data_store/knowledge_store/dev_knowledge_store.zip` (10.74 GB),
not a `dev/` directory. The old `dev/*` glob matched nothing and **failed
silently** — 12 MB arrived against a 12 GB estimate and the fetch reported
success. `reconcile_estimate()` in `scripts/fetchlib.py` now flags any fetch
under 25% of its estimate for exactly this reason.

## Media provenance: origin vs archive

Factify2 images come from two different places and are **never** mixed:

- `data/raw/factify2/media/images/` — fetched from the original host.
- `data/raw/factify2/media/images_wayback/` — recovered from web.archive.org
  after the origin refused. An archived snapshot may differ from what the
  dataset authors fetched in 2022.

The split is by directory so that `data/MANIFEST.sha256` distinguishes them by
path, and `data/raw/factify2/media/provenance.jsonl` records per file which
source it came from plus the Wayback snapshot timestamp. Any analysis that
treats the two as interchangeable is making an assumption it must state.

## Procedure for filling a row

1. Fetch into `data/raw/<name>/` using the entry's `method` and `url`,
   honouring `include_patterns` and never fetching anything matched by
   `forbidden_patterns`.
2. Read the licence at the source. Record the actual terms here, fix the
   register's `licence` field, and drop the "UNVERIFIED" marker.
3. Check the row counts against `expected`. A mismatch is a finding, not a
   rounding error — write down what differed.
4. `python scripts/manifest.py build` — commit the `MANIFEST.sha256` diff.
5. `python scripts/manifest.py audit` — this also fails if a forbidden file
   landed on disk.
6. Fill the row above with absolute dates and your name.

## Corpora needing extra provenance detail

**fakenewsnet** is distributed as IDs plus a crawler. Two crawls on different
dates produce different corpora because of link rot and deletion. Record the
crawl start date, the crawler commit hash, and the resolved article count — a
checksum alone does not make that acquisition reproducible.

**factify2** requires accepting shared-task terms through a registration form.
Record who accepted, on what date, and under which terms.

**isot** comes via the Kaggle API and needs credentials. Record the Kaggle
dataset version, since Kaggle datasets can be updated in place under the same
slug.

**averimatec** restricts evidence to sources predating each claim's date. On
ingest, confirm that property holds in the data rather than assuming it — it is
the same invariant `data/bible/` depends on.
