# Data card — Pe_Fake_News_Dec

A record of what this project holds, how it was obtained, and what is wrong
with it. Written to be read cold: every number below was **measured from the
files on disk**, not copied from a paper. Reproduce any of them with
`python scripts/verify.py --all`, which writes one JSON report per dataset to
`data/reports/` and compares each measurement against the `expected` figures
declared in `data/sources.yaml`.

Measured 2026-09-03. Ten datasets are enabled; nine were acquired on
2026-09-01/02, and one (`isot`) never was.

Where this card says "not declared", the register carries no published figure
for that quantity, so there is nothing to check our measurement against — that
is a gap in what we recorded, and it is stated rather than filled with a guess.

## Summary

| Dataset | Measured | Files | Rows | Licence | Redist. | Access | Obtained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mocheg | 1.900 GB | 166 | 43,148 evid. / 21,184 claims | CC-BY-4.0 | no | Zenodo 6653772 (+ aux git) | 2026-09-02 |
| factify2 | 9.760 GB | 79,257 | 42,500 | research, registration | no | Google Drive, password | 2026-09-02 |
| averitec | 10.758 GB | 4 | 5,783 claims | CC-BY-NC-4.0 | no | HF `chenxwh/AVeriTeC` (model repo) | 2026-09-02 |
| averimatec | 0.209 GB | 5 | 945 claims | research | no | HF `Rui4416/AVerImaTeC` (dataset) | 2026-09-02 |
| verite | 0.162 GB | 1,870 | 1,001 | repo terms, UNVERIFIED | no | GitHub RED-DOT + image hydration | 2026-09-02/03 |
| fakeddit | 15.072 GB | 142,436 | 623,342 | UNRESOLVED | no | Google Drive (from repo README) | 2026-09-02 |
| welfake | 0.228 GB | 1 | 72,134 | CC-BY | no | Zenodo 4561253 | 2026-09-02 |
| liar | 0.004 GB | 5 | 12,791 | research, UNVERIFIED | no | direct zip | 2026-09-01 |
| **isot** | — | 0 | **NOT FETCHED** | see Kaggle, UNVERIFIED | no | Kaggle (no credentials) | — |
| fakenewsnet | 0.041 GB | 17 | 23,196 | research, UNVERIFIED | no | GitHub (index only) | 2026-09-02 |

**38.16 GB, 223,931 manifest entries.** `redistributable: no` on every row is a
conservative default, not a finding: it means nobody has read the licence and
confirmed otherwise. Licences marked UNVERIFIED have not been read at all.

## Measured against declared

`scripts/verify.py` compares every measurement to `expected` in the register.
**Current result: 0 mismatches across 10 datasets.** Three entries declare
`expected` figures; the rest declare none.

| Dataset | Declared | Measured | Agrees |
| --- | --- | --- | --- |
| welfake | rows 72,134; real 35,028; fake 37,106 | identical | yes |
| verite | true 338; ooc 325; miscaptioned 338 | identical | yes |
| isot | real 21,417; fake 23,481 | not fetched | untested |

Two of those `expected` values were themselves corrected against measurement
rather than the other way round — see *Two corrections* below.

## Time coverage — what can enter a time-filtered evidence store

Retrieval in this project is meant to be time-filtered: evidence must predate
the claim it is offered for. That is only possible where a date is recoverable,
so coverage was measured now rather than discovered later. An **explicit** date
is a real field; a **URL-derived** date is a `/YYYY-MM-DD/` path segment, which
is the publisher's filing convention and *not* an asserted publication date.
The two are counted separately and never merged.

| Dataset | Claim dates | Range | Evidence items | Evidence dated |
| --- | --- | --- | --- | --- |
| averitec | **98.2%** explicit (`claim_date`) | 1919-02-19 – 2023-08-13 | 9,878 | 9.2%, URL-derived only |
| averimatec | **100%** explicit (`date`) | 2005-06-30 – 2023-07-31 | 2,709 | 6.8%, URL-derived only |
| fakeddit | **100%** explicit (`created_utc`) | 2008-06-01 – 2019-11-15 | n/a | n/a |
| mocheg | none — no date field | — | 43,148 | **0.0%** |
| factify2 | none — no date field | — | **none** | **n/a** |
| liar, welfake, verite, fakenewsnet | none | — | n/a | n/a |

Three consequences, and they are the point of measuring this:

- **AVeriTeC's knowledge store carries no date field.** Its records are
  `claim_id / type / query / url / url2text`. A scan of 100 of its 500 dev
  claim files (102,325 evidence items) recovered a date from the URL for
  **10.8%**. Nine items in ten cannot be time-filtered at all.
- **MOCHEG's evidence has no recoverable date whatsoever.** Its Commoncrawl URL
  column is empty and its Snopes URLs are `/fact-check/<slug>/` with no date
  segment. 0 of 43,148.
- **Factify2 has no dates and no discrete evidence items.** Its columns are
  `claim, claim_image, document, document_image, Category, Claim OCR,
  Document OCR`. The `document` field is evidence-shaped text, but it is one
  blob per record with no source and no date, so it cannot participate in
  time-filtered retrieval in any form.

Only `averitec` and `averimatec` have usable claim dates, and even for those
the *evidence* side is under 10% dated.

## Modality — what is actually present, not what is promised

Fractions are over records, counting files **on disk**, not fields in a CSV.

| Dataset | Text only | Image only | Both | Note |
| --- | --- | --- | --- | --- |
| factify2 | 9.6% | 0% | **90.4%** | a record needs *both* its images |
| averimatec | 0% | 0% | **100%** | images ship inside `images.zip` |
| fakeddit | 77.1% | 0% | **22.9%** | only a 150,000-row image sample was fetched |
| verite | 28.0% | 0% | **72.0%** | images hydrated 2026-09-03; see below |
| mocheg, liar, welfake, fakenewsnet, averitec | 100% | 0% | 0% | text-only as held |

## Per-dataset detail

### mocheg — 1.900 GB, 166 files
21,184 claims and 43,148 evidence rows (train 36,358 / val 1,562 / test 5,228).
Labels: refuted 19,179, supported 12,407, NEI 11,562 — these count evidence
rows, not claims.

**Version resolved.** The register recorded two conflicting figure sets. Our
measurement matches the Zenodo figures exactly (21,184 claims / 43,148 text
evidence) and not the SIGIR paper's smaller ones (15,601 / 33,880), so this is
the Zenodo release. Image-evidence qrels hold 88,040 rows of which 16,515 are
RELEVANCY=1, against a reported 15,373.

**Two gaps.** The tarball is `mocheg_v1_without_image` and contains **no
images**, so the multimodal half of this corpus is absent. And **no CSV carries
a tweet-id column**, so the 2,916 ID-only tweets described in the literature
have no hydration target in this distribution. Both are distribution gaps, not
fetch failures. The GitHub repo is fetched as an aux source for code only.

### factify2 — 9.760 GB, 79,257 files
42,500 rows (train 35,000 / val 7,500), five classes of exactly 8,500 each.
Images are hydrated from URLs, not shipped. **Two serious defects, documented
at length below**: Refute recovered 70.0% of its images against 96.7–99.9% for
every other class, and a classifier using only the image URL's hostname beats
the majority baseline by 15.5 points. Neither is fixed by discarding data.

Access is a Google Drive folder whose archives are password-protected; the
passwords are the registration gate and are **not stored in this repository**.

### averitec — 10.758 GB, 4 files
5,783 claims: train 3,068, dev 500, test 2,215. Labels Refuted 2,047, Supported
971, Not Enough Evidence 317, Conflicting Evidence/Cherrypicking 233; the 2,215
test claims are unlabelled by design and appear as `?`.

`claim_date` is day-first and unpadded (`25-8-2020`), 98.2% covered, spanning
1919–2023. Evidence is `questions -> answers`, 9,878 items, of which 9.2% have
a URL-derived date and **none** an explicit one.

The repo is 110.70 GB in total; we take `data/*` plus the dev knowledge store
only (10.74 GB). `test/` (superseded 15 Nov 2024) and `test_updated/` (the live
held-out shard) are both forbidden. The dev knowledge store is a single ZIP,
**not** a `dev/` directory — an earlier glob assumed otherwise, matched nothing,
and silently fetched 12 MB against a 12 GB estimate.

### averimatec — 0.209 GB, 5 files
945 claims (train 793 / val 152). Labels Refuted 897, Not Enough Evidence 24,
Supported 17, Conflicting 7 — heavily skewed. `date` is fully covered,
2005–2023. Evidence 2,709 items, 6.8% URL-dated.

Its held-out `test_data.zip` was fetched on the first pass because the entry
declared no `include_patterns` and forbade nothing, so it took the whole repo.
It is now forbidden and **removed from disk**; nothing had read it.

### verite — 0.162 GB, 1,870 files
1,001 rows: true 338, miscaptioned 338, out-of-context 325.

**Images are not shipped with this corpus and had to be fetched.** The register
points at `github.com/stevejpapad/relevant-evidence-detection` (RED-DOT), which
carries VERITE's CSVs and precomputed evidence but no images. They were
hydrated on 2026-09-03 from `true_url` / `false_url` in `VERITE_articles.csv`,
which is keyed by the same index the image paths use: `images/true_N.jpg` comes
from row N's `true_url`, `images/false_N.jpg` from its `false_url`. All 1,001
rows mapped to a URL; none were unmatched.

**663 distinct images serve 1,001 rows**, because `true` and `miscaptioned`
share the same `true_N.jpg` and differ only in caption. Their recovery rates
are therefore identical by construction, not by coincidence.

| Class | Rows | Recovered | Rate |
| --- | --- | --- | --- |
| out-of-context | 325 | 263 | **80.9%** |
| true | 338 | 229 | **67.8%** |
| miscaptioned | 338 | 229 | **67.8%** |
| **Overall** | 1,001 | 721 | **72.0%** |

**Loss is concentrated in blocking, not deletion**, the same pattern as
Factify2: of 285 first-pass failures, 198 were HTTP 403 and only 43 were 404.
`media.snopes.com` alone accounts for 127, all 403, and the circuit breaker
wrote it off after 25 consecutive refusals. Facebook's CDN
(`scontent.*.fbcdn.net`) contributes a further 40, which are genuinely
access-controlled. A browser-header retry recovered 5 more images — 71.5% to
72.0% — confirming that Cloudflare-fronted hosts do not yield to headers here
either.

**This matters more than the dataset's size suggests: verite is the only
enabled out-of-context corpus.** That cell is no longer empty, but it holds
80.9% of its out-of-context rows rather than all of them, and the true /
miscaptioned contrast — the pairing the benchmark is built on — is available for
only 67.8% of pairs. An archive fallback was not attempted; on Factify2 it
recovered roughly a fifth of blocked images at ~1 request/second, so the same
route is open here if the cell needs to be fuller.
### fakeddit — 15.072 GB, 142,436 files
623,342 rows (train 564,000 / validate 59,342) across six classes.
`created_utc` gives full date coverage, 2008–2019.

Images are a **seeded stratified sample of 150,000** (seed 20260901), of which
142,434 were recovered — **94.96%**, per class 92.3–99.3%. Loss is even across
classes and dominated by 7,350 genuine 404s: deleted Reddit content, not
blocking. In the modality table, "text only" means the row was not sampled, not
that Reddit had no image.

Both public test TSVs are held out by `forbidden_patterns`. **Its licence
restriction is unresolved** and it is treated as non-redistributable.

### welfake — 0.228 GB, 1 file
72,134 rows: real (label 0) 35,028, fake (label 1) 37,106 — matching the
register exactly. The source reports 78,098 rows *before* filtering, but the
Zenodo distribution already ships the filtered subset, so that 5,964-row drop
happened upstream and is not reproducible here.

### liar — 0.004 GB, 5 files
12,791 rows (train 10,240 / valid 1,284 / test 1,267) across six labels, from
half-true 2,627 down to pants-fire 1,047. Headerless TSV. Test labels are
public, so nothing is held back. This is the pipeline's smoke test.

### isot — NOT FETCHED
Blocked on Kaggle credentials; `kaggle` 2.2.4 wants `kaggle auth login`,
`KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`, none of which are present.
Nothing depends on it: **isot is a negative control only.** All its real news is
Reuters, so a model can score well by identifying the publisher rather than
detecting deception. Never report an ISOT number as a headline result.

### fakenewsnet — 0.041 GB, 17 files
23,196 rows across four files (gossipcop fake 5,323 / real 16,817; politifact
fake 432 / real 624). The label is carried by the filename, not a column.

**These rows are IDs and a `news_url` only** — article text, images and social
context require the upstream crawler, which we did not run. Two crawls on
different dates yield different corpora, so this dataset is not reproducible
from a checksum alone.

## Deferred

`visualnews` (60 GB) and `newsclippings` (8 GB) are `enabled: false`.
NewsCLIPpings ships only index files over VisualNews image IDs, so it is inert
without it; enable both together or neither, and raise `budget_gb` deliberately
when you do. Until then `verite` is the only out-of-context corpus — and it
currently has no images.

## Two corrections made against measurement

Both cases where a published figure and the file disagreed, resolved in favour
of the file:

- **verite holds 1,001 rows, not the published 1,000**, the extra one being
  out-of-context (325 vs 324). Investigated before recording: no duplicate rows,
  no blank line, no header off-by-one, the index runs contiguously 0–1000 with
  1,001 unique values, and all three shipped files agree on 338/338/325. **The
  row is kept.** Data is not dropped to match a paper.
- **welfake's `expected.csv_rows: 78098` described no file we hold.** Removed
  from `expected` and moved to notes as the source's pre-filter count.

## What is published, and what is not

The public repository at `github.com/anonymousz77/Pe_fake_news` carries the
code, the register, the manifests, the aggregate reports, the split membership
lists and this documentation. It carries **no dataset content**.

Every entry in the register is `redistributable: false`. AVeriTeC is CC-BY-NC,
Factify2 arrives password-protected behind a shared-task registration, and
Fakeddit's restriction is unresolved. So the rule is:

> **Published per-record files carry identifiers and split assignment only.**

| Published | Withheld |
| --- | --- |
| code, `data/sources.yaml`, `docs/` | anything under `data/raw/` |
| `data/MANIFEST.sha256` (hashes and paths) | image, archive and columnar files |
| `data/FETCH_LOG.jsonl` (source URLs, timings) | per-record labels |
| `data/reports/*.json` (aggregate statistics) | `hydration_*_failures.jsonl` (labels **and** image URLs) |
| `data/processed/splits/*.csv` as `record_id,split` | the Factify2 archive passwords |

Aggregate domain statistics — per-domain purity and counts — **are** published.
They are facts about public web properties rather than anyone's annotation, and
they are what makes the 37.5% → 22.2% result below checkable by a reader who
has no access to the data.

**To reproduce the splits:** obtain Factify2 yourself through the shared-task
registration, then join `data/processed/splits/factify2_{train,val,test}.csv`
on `record_id`. The labels come from your copy, not ours.

This is enforced mechanically, not by memory: `scripts/prepush_check.py` refuses
any staged path that is under a data directory, carries a media or archive
extension, exposes a per-record label or URL column for a non-redistributable
dataset, or contains a gate credential. It reads the gated list from the
register's `redistributable` field, so a new dataset is covered automatically.

---

# Known issue: Factify2 image recovery and the domain/label confound

Two defects, measured on 2026-09-02. Neither is fixed by discarding data, and
neither should be smoothed over in write-up. **Both constrain what this project
may claim from Factify2.**

## 1. Class-imbalanced image recovery

Factify2 ships image URLs, not images. Hydrating all 85,000 references
(42,500 rows x claim_image + document_image) recovered 77,973 = **91.7%**, but
the loss is almost entirely in one class:

| Class | Attempted | Recovered | Rate |
| --- | --- | --- | --- |
| Insufficient_Multimodal | 17,000 | 16,983 | 99.9% |
| Insufficient_Text | 17,000 | 16,936 | 99.6% |
| Support_Multimodal | 17,000 | 16,902 | 99.4% |
| Support_Text | 17,000 | 16,441 | 96.7% |
| **Refute** | 17,000 | **10,711** | **63.0%** |

The corpus is balanced by construction — 17,000 references per class — so this
is purely an acquisition artefact. Any model trained on the recovered subset
sees roughly two-thirds as many Refute images as anything else.

### It is one host

Grouping the 7,052 original failures by domain:

| Domain | Failures | Attempts | Rate | Dominant class |
| --- | --- | --- | --- | --- |
| snopes.com | 5,814 | 5,814 | **100.0%** | Refute 5,810 |
| i0.wp.com | 457 | 535 | 85.4% | Support_Text 380 |
| empirenews.net | 74 | 75 | 98.7% | Support_Text 57 |
| gannett-cdn.com | 64 | 114 | 56.1% | Refute 63 |
| factcheck.afp.com | 30 | 32 | 93.8% | Refute 30 |
| i.ytimg.com | 27 | 227 | 11.9% | Refute 20 |
| cdn.cnn.com | 24 | 1,269 | 1.9% | Insufficient_MM 12 |
| washingtonpost.com | 24 | 42 | 57.1% | Refute 22 |
| api.time.com | 23 | 23 | 100.0% | Refute 17 |
| thelogicalindian.com | 21 | 21 | 100.0% | Refute 12 |

The top 10 cover 93% of failures; 268 further domains supply the remaining 494.
**snopes.com alone is 82% of all failures and 92% of all Refute loss**, and it
refused every single one of its 5,814 requests.

Failure histogram (first pass, 7,052 failures): `http_403` 6,470,
`http_404` 302, `connectionerror` 78, `http_406` 68, `sslerror` 37,
`http_400` 34, `readtimeout` 20, `http_502` 15, `empty_body` 7, remainder <5
each. **403 is a refusal, not a disappearance.**

### What the retry established

A retry with a browser User-Agent, a Referer set to each URL's own origin, the
standard image `Accept` header, concurrency 4 and a 0.5 s per-host delay
recovered **25 of 7,052**. Refute moved 62.9% -> 63.0%.

snopes.com sits behind Cloudflare and returns 403 to browser headers exactly as
it does to a plain client — verified directly before the run. Six hosts were
written off by the circuit breaker after 25 consecutive refusals each:
snopes.com, i0.wp.com, empirenews.net, gannett-cdn.com, factcheck.afp.com,
i.ytimg.com.

**Header-based recovery of the Refute class is not possible.**

### What the archive recovery established

A web.archive.org pass at ~1 req/s recovered **1,278 images**, of which
**1,192 are Refute**. It was stopped after 5,600 of 6,792 candidates (82%), so
these figures are a **partial** recovery, not a ceiling — resuming
`hydrate.py --dataset factify2 --wayback` would continue from here, since
anything already on disk is skipped.

| Class | Origin only | + archive | Change |
| --- | --- | --- | --- |
| Insufficient_Multimodal | 99.9% | 99.9% | +0.0 |
| Insufficient_Text | 99.6% | 99.7% | +0.1 |
| Support_Multimodal | 99.4% | 99.5% | +0.1 |
| Support_Text | 96.7% | 97.0% | +0.3 |
| **Refute** | **63.0%** | **70.0%** | **+7.0** |
| Overall | 91.7% | 93.2% | +1.5 |

Snapshot availability on a sampled probe was 4/5, with timestamps in 2022 —
the same period the dataset authors would have fetched. Success across the run
averaged ~23%, so most blocked URLs have no usable snapshot.

**Refute remains ~27 points below every other class.** The imbalance is
reduced, not resolved.

### Provenance is tracked, never merged

Archived copies are a different artefact from what the dataset authors fetched
in 2022, so they are kept apart structurally:

- `media/images/` — origin fetches
- `media/images_wayback/` — archive recoveries
- `media/provenance.jsonl` — per file: source, Wayback snapshot timestamp, URL

`data/MANIFEST.sha256` therefore distinguishes them **by path**, so code that
never reads the ledger still cannot conflate them.

## 2. Image provenance predicts the label

This is the more serious defect, and it is independent of recovery.

Measured over the 77,948 recovered images (`scripts/confound.py`, full output in
`data/reports/factify2_domain_label_confound.json`):

| Measure | Origin only (77,948 imgs) | Current (79,251 imgs) |
| --- | --- | --- |
| Normalised mutual information NMI(domain; label) | 0.3055 | **0.3118** |
| Majority-class baseline | 21.8% | 21.4% |
| Domain-only classifier, held out (5-fold) | 37.3% | **38.2%** |
| Domain-only classifier, resubstitution | 38.0% | 38.9% |
| Lift over baseline | +15.5 points | **+16.8 points** |

A classifier that sees **only the hostname of the image URL** — not one pixel —
beats the majority baseline by 15.5 points. The held-out figure is the honest
one; resubstitution is inflated by the 433 domains that appear exactly once.

### Three domains are 100% one class

| Domain | Images | Class purity |
| --- | --- | --- |
| factly.in | 3,563 | **100% Refute** |
| boomlive.in | 3,209 | **100% Refute** |
| images.thequint.com | 1,808 | **100% Refute** |
| snopes.com (archive-recovered) | 1,123 | **100% Refute** |
| static01.nyt.com | 248 | 88% Refute |
| i.ytimg.com | 200 | 86% Refute |
| pbs.twimg.com | 60,837 | 26% Support_Multimodal (spread) |

`pbs.twimg.com` is 77% of the corpus and is spread across classes, which is
what keeps NMI at 0.31 rather than higher. The signal comes from the
fact-checking sites: **9,703 images whose domain alone determines the label
with certainty** — 8,580 from the three original pure domains plus 1,123
archive-recovered snopes images.

Most-common domain per class:

| Class | Top domain | Share of class |
| --- | --- | --- |
| Insufficient_Multimodal | pbs.twimg.com | 85.1% |
| Insufficient_Text | pbs.twimg.com | 91.5% |
| Support_Multimodal | pbs.twimg.com | 95.2% |
| Support_Text | pbs.twimg.com | 89.8% |
| Refute | factly.in | 33.3% |

### Why this matters

This is **ISOT's Reuters shortcut in the image modality**. There, every real
article comes from Reuters, so a model learns the publisher instead of
deception. Here, Refute images disproportionately come from fact-checking sites
— which is unsurprising, since a fact-check article is where a refuted claim's
image lives — and a model can exploit that without looking at content.

The consequence is concrete: a Factify2 score is not evidence of multimodal
reasoning unless the evaluation controls for source domain. Any headline number
from this corpus must be reported alongside the domain-only baseline of 37.3%.

### The two defects pull against each other — measured, not predicted

Recovering Refute images **improves class balance and worsens the confound**,
because the images being recovered come from near-pure domains. Measured before
and after the archive pass:

| Measure | Origin only | + archive | Change |
| --- | --- | --- | --- |
| Refute recovery | 63.0% | 70.0% | **+7.0 points (better)** |
| NMI(domain; label) | 0.3055 | **0.3118** | +0.0063 (worse) |
| Domain-only held-out accuracy | 37.3% | **38.2%** | +0.9 points (worse) |
| Lift over baseline | +15.5 | **+16.8** | +1.3 points (worse) |

snopes.com has now entered the top-10 domains at 1,123 images and **100%
Refute** — a *fourth* perfectly label-predictive domain alongside factly.in,
boomlive.in and images.thequint.com.

This is the trade-off in numbers: every image recovered for the under-served
class arrives from a source that makes the class more identifiable by
provenance alone. Neither figure is optimised at the other's expense, and both
are reported.

## What is explicitly not being done

- **No rebalancing by discarding images from other classes.** Throwing away
  ~6,300 images each from four classes to match Refute would destroy a quarter
  of the corpus to hide a number.
- **No dropping of the leaky domains.** Removing factly.in, boomlive.in and
  thequint would remove a third of the Refute class.

Both are measurement problems, not data problems. They belong to the
experimental design: source-disjoint splits, a domain-only baseline reported
alongside every result, or a domain-adversarial objective. Recorded here so the
design can address them deliberately.


---

## Does further recovery help or hurt? A controlled answer

Three runs of `scripts/confound.py` over the same corpus, differing only in
which images are counted. 5-fold held-out, seed 20260901.

| | (a) origin only | (b) origin + wayback | (c) (b) minus snopes.com |
| --- | --- | --- | --- |
| Images counted | 77,973 | 79,251 | 78,128 |
| **NMI(domain; label)** | **0.3053** | **0.3118** | **0.3042** |
| **Domain-only, held out** | **37.3%** | **38.2%** | **37.3%** |
| Domain-only, resubstitution | 38.0% | 38.9% | 38.0% |
| Uniform 5-class baseline | 20.0% | 20.0% | 20.0% |
| Majority baseline (as recovered) | 21.8% | 21.4% | 21.7% |

**(c) lands back on (a) almost exactly.** Removing the 1,123 archive-recovered
snopes images from (b) returns both NMI and held-out accuracy to their
origin-only values. That is a controlled result, not an inference: the entire
(a) -> (b) increase is attributable to snopes and nothing else.

### The answer: stronger. Further recovery degrades the dataset.

Projecting forward — the domain and label of every *unrecovered* item are
already known from the CSVs, so this is arithmetic on real labels, not a guess:

| Scenario | Images | NMI | Domain-only held out |
| --- | --- | --- | --- |
| (b) current | 79,251 | 0.3118 | 38.2% |
| + all 4,691 remaining snopes | 83,942 | **0.3476** | **41.7%** |
| + all 5,749 remaining images | 85,000 | 0.3417 | **42.2%** |

Of the 4,691 snopes images still missing, **4,688 are Refute** — 99.94%.

Completing snopes recovery would take the domain-only classifier from 37.3%
(origin only) to **41.7%**, a **+4.4 point** increase, while the uniform
baseline stays at 20%. In other words, finishing the recovery would roughly
**double the headroom a model can win by ignoring the image and reading the
hostname** — from +17.3 points over uniform to +21.7.

So the two objectives are not merely in tension; on this corpus they are close
to directly opposed. **Recovery buys class balance and pays for it in
construct validity.** Recovering the remaining snopes images would raise Refute
image recovery from 70% toward ~97% while making the dataset measurably easier
to game.

(Recovering *everything* gives a marginally lower NMI than snopes alone — 0.3417
vs 0.3476 — because the 1,058 non-snopes stragglers are spread across many
domains and dilute the association slightly. Held-out accuracy still rises,
to 42.2%, because those domains are individually small and mostly pure.)

## The alternative: complete cases only

Dropping every row with a missing image, instead of recovering. A Factify2 row
carries two images (claim + document), so one absent image discards the row's
text, its other image and its label as well.

| Class | Rows | (a) origin only | (b) + wayback | (c) minus snopes |
| --- | --- | --- | --- | --- |
| Insufficient_Multimodal | 8,500 | 8,484 (99.8%) | 8,488 (99.9%) | 8,488 (99.9%) |
| Insufficient_Text | 8,500 | 8,444 (99.3%) | 8,450 (99.4%) | 8,450 (99.4%) |
| Support_Multimodal | 8,500 | 8,414 (99.0%) | 8,427 (99.1%) | 8,427 (99.1%) |
| Support_Text | 8,500 | 8,041 (94.6%) | 8,080 (95.1%) | 8,080 (95.1%) |
| **Refute** | 8,500 | **4,310 (50.7%)** | **4,980 (58.6%)** | **4,311 (50.7%)** |
| **Total kept** | 42,500 | 37,693 (88.7%) | **38,425 (90.4%)** | 37,756 (88.8%) |

The overall cost looks mild — 4,075 rows, 9.6% — and that is exactly the trap.
**Refute loses 41.4% of its rows while no other class loses more than 5.4%.**
A complete-cases design on the current state yields roughly 4,980 Refute rows
against ~8,450 for every other class: a 1.7:1 imbalance in a corpus that was
built balanced.

Note the paired-image effect: Refute has 70.0% of its *images* but only 58.6%
of its *rows*, because a row needs both.

## What this means for the design

Three options, none of them free:

1. **Resume recovery.** Refute images 70% -> ~97%, rows 58.6% -> ~97%. Domain-only
   accuracy 38.2% -> ~41.7%. Best balance, worst confound.
2. **Complete cases only.** No new confound, no archive provenance to reason
   about, but a 1.7:1 class imbalance and 4,075 rows discarded — 3,520 of them
   Refute.
3. **Stop here and control for it in the design.** Keep the current state,
   report the domain-only baseline of 38.2% alongside every Factify2 result,
   and use source-disjoint splits so a model cannot learn the mapping in the
   first place.

Option 3 is the only one that does not trade one defect for another, and it is
the reason these numbers are recorded rather than optimised. Whichever is
chosen, **a Factify2 score is uninterpretable without the domain-only baseline
printed next to it.**


---

# Factify2: the decision, and the rules that follow from it

**Recovery is stopped permanently as of 2026-09-02.** The remaining images are
not pursued, because pursuing them would make the dataset easier to game:
completing snopes recovery raises corpus-wide domain-only accuracy from 38.2% to
a projected 41.7%, and 4,688 of the 4,691 remaining snopes images are Refute.

The defects are therefore controlled in the design. 
These are rules for all later work, not commentary. Recovery of Factify2's
missing images was **stopped permanently on 2026-09-02**; the two defects below
are controlled in the experimental design instead of fixed in the data.

**1. Every Factify2 result is reported beside the domain-only baseline.**
A hostname-only classifier — no pixels, no text, just the image URL's host —
scores **37.5%** on a random split and **22.2%** on the source-disjoint split,
against a 20.0% uniform baseline. A model not clearly above the applicable
figure is reading hostnames, not content. Print it next to every number.

**2. Splits are source-disjoint on LABEL-PREDICTIVE domains, not on all
domains.** Threshold: class purity ≥ 0.90. Generated by `scripts/splits.py`,
which enforces the constraint and fails loudly if it is violated.

Do not "strengthen" this to all domains. It was tried and it is impossible:
records linked by shared domains form a single component of 36,812 of 38,425
records (95.8%), because `pbs.twimg.com` touches 84.7% of records. The only
strict split available is 95.8/4.2 with a 99.4%-Refute test set. And
`pbs.twimg.com` is near-uniform (26% Support_Multimodal) — it carries no label
signal, so constraining it is cost without benefit. The leak is `factly.in`,
`boomlive.in`, `images.thequint.com` and `snopes.com`, each 100% Refute.
`--disjoint strict` still exists and still refuses, so the finding stays
reproducible.

**3. Per-class metrics always. Never a single averaged Factify2 number.**
Recovery, completeness and class balance all differ by class; an average hides
every one of them.

**4. Permanent limitations, stated in any write-up:**
- Refute **image** recovery 70.0%; every other class 97.0–99.9%.
- Refute **row** completeness 58.6% (a row needs both its images); others
  95.1–99.9%.
- **`val` contains 4 Refute records.** 4,969 of 4,980 Refute records sit in just
  two indivisible domain components, so once train and test each take one there
  is nothing left. Refute model selection must use cross-validation within
  train, or train/test only — never val.

**5. Recovery was stopped deliberately**, at 1,278 archive-recovered images.
Completing it would raise the corpus-wide domain-only accuracy from 38.2% to a
projected **41.7%**, because **4,688 of the 4,691 remaining snopes images are
Refute**. Recovery buys class balance and pays for it in construct validity.
Do not restart it without re-reading `docs/data_card.md`.


## What the source-disjoint split achieves

`scripts/splits.py`, seed 20260901, 70/15/15 requested, purity threshold 0.90,
249 constrained domains, **0 leakage violations**.

| | random baseline (`--disjoint none`) | source-disjoint (`--disjoint predictive`) |
| --- | --- | --- |
| **Hostname-only accuracy on test** | **37.5%** | **22.2%** |
| Majority baseline on test | 23.9% | 21.7% |
| Uniform baseline | 20.0% | 20.0% |
| Lift available from hostname alone | **+17.5 pts** | **+2.2 pts** |
| Leakage violations | 35 (not enforced) | **0 (enforced)** |

**The constraint removes almost the whole shortcut**: from +17.5 points of free
accuracy down to +2.2, essentially the majority baseline. The gap between the
two columns — 15.3 points — is a measurement of what the confound was worth,
and is itself a reportable result.

### The cost: split shape

| Split | Records | Share | Refute |
| --- | --- | --- | --- |
| train | 26,784 | 69.7% | 3,372 |
| val | 5,021 | 13.1% | **4** |
| test | 6,620 | 17.2% | 1,604 |

Ratios drift from the requested 70/15/15 because whole domain components must
move together, and the largest is 3,367 records.

**`val` has 4 Refute records.** This is structural, not a bug: 4,969 of 4,980
Refute records lie in two indivisible components (3,365 and 1,604), so once
train and test have each taken one, none remains. The generator reports this as
an UNUSABLE CLASS/SPLIT COMBINATION rather than letting it pass quietly. Refute
model selection must use cross-validation within train, or train/test only.

All of it is recorded in `data/processed/splits/factify2_split_report.json`:
mode, threshold, seed, the full constrained-domain list with per-domain purity
and counts, split sizes, per-class distribution per split, component-size
distribution, leakage detail, and the domain-only accuracies.
