# Provenance

Where each corpus actually came from, and what a human has verified by hand.

Scaffold created 2026-09-01. **All ten originally enabled datasets are now
acquired**: nine on 2026-09-01/02, `isot` on 2026-10-01 from its creators'
own distribution. Two more, `dgm4` and `m4fc`, were added on 2026-10-01 by the
new-corpus survey (`docs/data_card.md`). A row that says "fetched" means the
endpoint resolved and served data.

**Every enabled dataset's licence has now been read at its source**
(2026-10-01) and is quoted verbatim in `docs/licences.md`. `redistributable`
is `true` only where that text grants redistribution (mocheg, welfake,
averitec, dgm4, m4fc); `tests/test_sources.py` refuses `true` on any entry
whose licence does not name a granting licence and that `docs/licences.md`
does not list as redistributable.

## Ledger

| Dataset | Method | Acquired (date) | URL confirmed | Licence read | Counts match `expected` | Verified by |
| --- | --- | --- | --- | --- | --- | --- |
| mocheg | zenodo + image release | 2026-09-02; **images 2026-10-01** | confirmed | **CC-BY-4.0, verbatim (2026-10-01)** | see counts_ledger | automated fetch; stream_archive |
| factify2 | gdrive | 2026-09-02 | confirmed | **none stated (2026-10-01)** | see counts_ledger | automated fetch |
| averitec | hf | 2026-09-02 | confirmed | **CC-BY-NC-4.0, verbatim (2026-10-01)** | n/a (expected null) | automated fetch |
| averimatec | hf | 2026-09-02 | confirmed | **none stated (2026-10-01)** | n/a | automated fetch |
| verite | git | 2026-09-02 | confirmed | **Apache-2.0 repo; research-only dataset (2026-10-01)** | measured 338/338/325 | automated fetch |
| fakeddit | gdrive | 2026-09-02 | confirmed | **none stated (2026-10-01)** | see counts_ledger | automated fetch |
| welfake | direct | 2026-09-02 | confirmed | **CC-BY-4.0, verbatim (2026-10-01)** | **exact** | automated fetch |
| liar | direct | 2026-09-01 | confirmed | **research use only (README, 2026-10-01)** | n/a; 12,836 measured | automated fetch |
| isot | direct (UVic) | **2026-10-01** | **confirmed** | **none stated (2026-10-01)** | **exact: real 21,417, fake 23,481** | automated fetch |
| fakenewsnet | git | 2026-09-02 | confirmed | **none stated (2026-10-01)** | n/a | automated fetch |
| dgm4 | hf | **2026-10-01** | confirmed | **Apache-2.0 card; S-Lab code licence** | not reproducible (test forbidden) | automated fetch |
| m4fc | git | **2026-10-01** | confirmed | **CC-BY-SA-4.0, verbatim** | images 4,982 exact; claims 6,789 vs 6,980 | automated fetch |
| visualnews | — | — (deferred) | listing seen 2026-10-01 | **none stated** | n/a | — |
| newsclippings | — | — (deferred) | repo read 2026-10-01 | **none stated** | n/a | — |

## Counts to check on ingest

Three entries carry `expected` counts in the register. These are the only
numbers in this project that came with the specification; reproduce them
exactly or investigate the difference before proceeding.

| Dataset | Expected |
| --- | --- |
| verite | `true` 338, `ooc` 324, `miscaptioned` 338 |
| welfake | usable_rows 72,134 of csv_rows 78,098; real 35,028, fake 37,106 |
| isot | real 21,417, fake 23,481 — **reproduced exactly 2026-10-01** |
| m4fc | images 4,982 — reproduced; claims 6,980 — measured 6,789 (−191, unresolved) |

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


## Acquisitions of 2026-10-01

**isot** — from `onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip`,
linked from the ISOT lab page; no account needed, unlike the Kaggle mirror the
register used to name. True.csv 21,417, Fake.csv 23,481 — the register's
`expected` exactly. The lab's ReadMe PDF states no licence, only a citation
request, and contradicts itself on size ("more than 12,600 articles" per file
in prose; 21,417 / 23,481 in its own table).

**mocheg images** — the Zenodo tarball's own README points to
`nlplab1.cs.vt.edu/.../MOCHEG/dataset/`, an open directory holding
`latest_dataset/mocheg_with_tweet_2023_03.tar.gz` (74,304,681,291 bytes). The
GitHub README routes through a Google Form instead; it was not filled in.
Streamed through `scripts/stream_archive.py`, writing only `mocheg/images/*`
(137,619 images) and metadata under 100 MB. All 16,291 relevant evidence
images are present. The release is a later version than Zenodo v1 (its qrels
differ); records are built from v1.

**dgm4** — `rshaojimmy/DGM4` on the Hugging Face Hub, `metadata/test.json`
forbidden; 12 files, 10,689,659,069 bytes; all 206,362 zipped images decode.

**m4fc** — `github.com/UKPLab/M4FC`, annotations only; images are hydrated from
the URLs in `data/M4FC.json` through the three gates, with the authors'
`wayback_image_url` as the declared alternate.

## Image recovery: closeout pass, 2026-10-01..03

Pass id `closeout-2026-10`. Every HTTP request is a row in
`data/raw/<name>/media/provenance.jsonl` (URL, host, HTTP status, outcome,
reason, pass); the ledgers are the resume state and the record.
68,464 requests, 3.98 GB in, one worker, 5 s per host. Routes in order: the
dataset's URL with fresh headers and a different user agent; the Wayback
Machine (CDX prefix lookup, up to three snapshots); every alternate URL the
dataset's own metadata declares; then offline matches against files already on
disk (URL identity; CLIP fingerprint for VERITE). Recovered files are written
under `images_retry/`, `images_wayback/` and `images_alternate/`, never over an
origin file, so an archived copy is always distinguishable by path.

Usable from this pass: VERITE 122, Factify2 4,178, Fakeddit 4,069, M4FC 4,920.
Every image still missing has a terminal reason in `data/dead/<name>.csv`
(VERITE 56, Factify2 4,994, Fakeddit 7,520, M4FC 61; none open).

**Checked by hand:** every VERITE image counted as usable was compared with the
CLIP embedding the authors ship for its row; the five that did not match were
inspected and excluded (`data/placeholders.yaml` v6, `data/fingerprint_rejections.yaml`).
`FETCH_LOG.jsonl` has one line per *completed* run; runs reaped for memory left
no line, and their requests are in the ledgers.
