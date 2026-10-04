# Data card — Pe_Fake_News_Dec

A record of what this project holds, how it was obtained, and what is wrong
with it. Written to be read cold: every number below was **measured from the
files on disk**, not copied from a paper. Reproduce any of them with
`python scripts/verify.py --all`, which writes one JSON report per dataset to
`data/reports/` and compares each measurement against the `expected` figures
declared in `data/sources.yaml`.

Measured 2026-09-03. Ten datasets are enabled; nine were acquired on
2026-09-01/02, and one (`isot`) never was.

> **Closed 2026-10-03.** Twelve datasets are enabled and all are acquired. The
> sections below are dated records and keep the figures measured on their date;
> where the closeout pass changed a figure, the current value is in
> `docs/completion_report.md` section 9 (corrected values), and the per-class
> tables there supersede any coverage figure quoted here. The defects found in
> the closeout pass are recorded at the end of this file: Fakeddit's subreddit
> labels, DGM4's text-swap audit, and verdict-bearing evidence images.

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

> **MOCHEG cannot participate in time-filtered retrieval at all.**
>
> It ships **no claim-date field and no evidence-date field**. Its evidence
> cannot be dated (0 of 43,148) and, because there is no claim date either,
> it cannot even be placed *relative* to the claim it supports. There is no
> ordering to recover, so no time filter can be applied to any part of it.
>
> This matters out of proportion to its 1.9 GB: mocheg is **43,148 of the
> 55,168 bundled evidence URLs in this project — 78%** — and it is the source
> of nearly all the snopes and politifact volume. The evidence store is
> therefore built from `averitec` and `averimatec` only, an 11,725-URL pool
> (8,482 distinct) with an entirely different domain shape: 38.8%
> `web.archive.org`, fact-check domains almost absent.
>
> **This is a property of the distributed dataset, not of this pipeline.**
> Nothing in the acquisition or hydration layer discarded these dates; they
> were never released. No amount of re-fetching recovers them.
>
> It is pinned in code by `ENRICH_DATASETS` in `scripts/bible.py` and by
> `tests/test_bible.py::test_mocheg_is_not_an_enrichment_source`, so re-adding
> mocheg as an evidence source requires deliberately deleting a test.


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
  segment. 0 of 43,148 — and with no claim date either, it is excluded from the
  evidence store outright rather than partially covered.
- **Factify2 has no dates and no discrete evidence items.** Its columns are
  `claim, claim_image, document, document_image, Category, Claim OCR,
  Document OCR`. The `document` field is evidence-shaped text, but it is one
  blob per record with no source and no date, so it cannot participate in
  time-filtered retrieval in any form.

Only `averitec` and `averimatec` have usable claim dates, and even for those
the *evidence* side is under 10% dated.

## The evidence store (`data/bible/`)

The store is the corpus that claims get checked against. Its one hard rule:

> **A document with no publication date is quarantined to `data/bible/undated/`
> and is never admitted to the retrieval index.** It is counted and reported,
> never indexed. An undated document cannot be placed relative to a claim, so
> admitting it would silently reintroduce the leakage the time filter exists to
> prevent.

The filter is enforced *inside* the retrieval query — a JOIN against
`temporal.sqlite` with `published_at IS NOT NULL AND published_at <= :as_of` —
not by trimming results afterwards. A post-filter is a promise that every future
caller remembers to apply it, and it silently degrades `k`. `tests/test_bible.py`
plants a document dated one day after the `as_of` date and one with a null date,
and asserts neither ever surfaces at a `k` large enough to return the whole
corpus.

That test runs on a four-row fixture, which proves the SQL and nothing about the
store. `scripts/redtest.py` runs the same predicate against the **real** index —
1,368,613 documents — by planting the same three documents into `bm25.sqlite` and
`temporal.sqlite` inside a transaction it always rolls back, sweeping `k` up past
the corpus size, and verifying afterwards that both row counts, a `quick_check`
and a 50-deep ranking fingerprint are unchanged.

It passed on 2026-09-11. At its widest, k=2,000,000 against a term matching
973,757 documents, it returned 703,362 in one result set with nothing dated
after the `as_of` date and neither plant present, while the document dated one
day earlier came back at every k tried. Full result in
`data/reports/redtest.json`.

### What counts as a publication date

| `date_source` | Confidence | What it actually asserts |
| --- | --- | --- |
| `jsonld` | high | the publisher's own structured claim |
| `meta_published` | high | the publisher's own meta tag |
| `url_path` | low | the publisher's filing convention, e.g. `/2019/05/03/` |
| `wayback_capture` | low | when *someone archived* the page — an upper bound |
| `http_last_modified` | **none — quarantined** | when the file changed on the server |

**`Last-Modified` is not a publication date.** A probe of 70 non-archive URLs
found 15 returning a usable `Last-Modified`, of which **9 (60%) were dated 2025
or later — several dated the day of the probe** — for claims from 2018–2020.
It is a file mtime. It is recorded for provenance and then quarantines the
document; it never sets `published_at`.

**A wayback capture stamp is not a filing path**, and the two were measured
pulling in opposite directions over 8,482 enrichment URLs:

| Signal | Present | Predates its claim | Usable |
| --- | --- | --- | --- |
| filing path, archive slice | 10.3% | 85.0% | 8.7% |
| filing path, other hosts | 9.6% | 88.0% | 8.5% |
| capture stamp | 85.7% of archive URLs | **3.6%** | 3.1% |

A capture stamp postdates its claim 95.4% of the time, because evidence is
archived at annotation time. It is still *sound* — a capture before the claim
guarantees the content existed before it — so it is kept, but strictly as a
last resort behind any filing path in the same URL. Merging the two into one
`url_path` count, as an earlier version did, hid a good signal inside a bad one.

Note the two questions this separates. **Admission** to the index needs only a
date; **usability as evidence for its own claim** needs that date to predate the
claim. The store holds more documents than are usable for their originating
claim, which is correct — they remain valid evidence for other claims.

### Sources

| Source | Role | Dates from |
| --- | --- | --- |
| **Wikinews** (D) | fixture; the test corpus, ungated | native `{{date\|...}}` — 100% explicit |
| **enrich** (A) | the evidence actually cited by claims | publisher metadata, then URL |
| **CC-NEWS** (B) | the backbone: contemporaneous news at volume | WARC record date |

Measured yields, from `data/reports/bible.json` over the store as built:

| Source | Attempted | Indexed | Quarantined | High-confidence date |
| --- | --- | --- | --- | --- |
| Wikinews (D) | 22,394 pages | 21,457 | 937 | 21,457 — **100%** of indexed |
| enrich (A) | 8,482 URLs | 4,894 | 1,539 | 3,183 — **37.5%** of URLs |
| CC-NEWS (B) | 40 WARCs | 1,342,262 | 738 | 1,071,949 — **79.9%** of indexed |
| **store** | | **1,368,613** | **3,214** | **1,096,589 — 80.1%** |

Three different denominators, on purpose, because the three sources fail in
three different places.

**Wikinews** is the fixture and behaves like one: every page that parses carries
an explicit `{{date|...}}`, so its high-confidence rate is 100% by construction
and its 937 quarantines are pages with no date template at all. It measures the
pipeline, not the world.

**enrich** loses documents twice before dating even arises. Of 8,482 evidence
URLs, 2,049 (24.2%) returned no document — dead links, paywalls, the archive
throttle — and a further 1,539 returned a page with no usable date. So 4,894 are
indexed, and 3,183 carry a publisher's own claim about when they were published.
The three rates are 57.7% indexed over URLs, 65.0% high-confidence over indexed,
and **37.5% high-confidence over URLs**, which is the one that matters: it is the
share of the evidence a claim actually cites that can be placed on a timeline.

**CC-NEWS** almost never quarantines, because a WARC record always has a crawl
date and that is a real upper bound on publication. Its 738 quarantines are all
from the plausibility window — timestamps that failed to convert — not from
missing dates. So its interesting number is not admission but *quality*: 79.9%
of its documents carry publisher metadata, and the remaining 270,313 fall back
to `warc_date` or `url_path`.

**CC-NEWS begins 2016-08.** Probed, not assumed: every month before it returns
HTTP 404 for `warc.paths.gz`. The 40-WARC allocation spans 11 months chosen so
that 98.9% of the 6,623 dated claims have a selected month within the 12 months
*before* the claim; **75 claims (1.1%) predate the crawl entirely and can never
be covered by it.** An earlier allocation aimed 2 of its 40 WARCs at 2016-01 and
2016-04, which do not exist — the builder skips an unavailable month rather than
failing, so those WARCs would have been lost with nothing in the output to say
so. Pinned by `test_no_allocated_month_predates_the_crawl`.

### Two text extractors, on purpose

The store uses **bs4 for per-URL sources and a regex extractor for CC-NEWS**.
Every document records which one produced it in `provenance.extractor`, so no
reader has to guess.

This is a throughput decision with a measured, and small, quality cost. bs4
spends its time in `decompose()` over `nav/footer/header/aside/form`, not in
parsing — which is why switching its parser to lxml changed nothing:

| Extractor | Per page | Per WARC (~53k pages) | 38 WARCs |
| --- | --- | --- | --- |
| bs4 + `html.parser` | 56 ms | ~50 min | ~33 h |
| bs4 + `lxml` | 56 ms | ~50 min | ~33 h |
| regex (`fast`) | **4.0 ms** | ~10 min | **~6.5 h** |

Fidelity, measured over 400 real CC-NEWS records against bs4 as reference:

- share of bs4's words the regex path also finds: **p50 1.00, p10 0.96**
- documents passing the 200-character rule: **358 vs 357**
- extracted length: 1.42x bs4's, the excess being residual boilerplate

So it keeps essentially all of the article and a little more chrome. For a BM25
index that is noise, not loss.

**The 6,433 enrich documents stay on bs4 and were not rebuilt.** Documents store
extracted text only, never raw HTML, so re-extracting them means re-fetching all
8,482 URLs — 5–6 hours, dominated by the 1 req/5 s archive throttle — to make
6,433 documents match ~2M on the *cheaper* extractor. Uniformity is not worth
degrading the smaller, more carefully fetched half of the store to buy it. The
mix is recorded per document instead.

### Budgets

Hard caps, refused rather than truncated: **30 GB stored, 60 GB transferred**,
`--warc-limit` 40. Cumulative transfer is logged to `data/reports/bible.json`
across runs, so the cap holds over a resumed build rather than per-invocation.

### Date coverage of the indexed store

97.0% of the store falls in 2016–2023, which is the CC-NEWS window; 2020 alone
is 34.7%. The distribution is a fact about the WARC allocation, not about the
world, and it should not be read as news volume.

| Year | Documents | Share |
| --- | --- | --- |
| ≤ 2015 | 40,801 | 2.98% |
| 2016 | 59,003 | 4.31% |
| 2017 | 45,757 | 3.34% |
| 2018 | 88,937 | 6.50% |
| 2019 | 293,516 | 21.45% |
| 2020 | 474,680 | 34.68% |
| 2021 | 167,333 | 12.23% |
| 2022 | 175,929 | 12.85% |
| 2023 | 22,120 | 1.62% |
| ≥ 2024 | 537 | 0.04% |

**The reported range is `1395-06-03` to `2562-02-02`, and both ends are
calendars rather than errors** — a Solar Hijri year and a Buddhist-era year. In
Gregorian terms the store runs from a New York Times archive page dated
1970-04-17 to 2026.

### Dates that could not be dates: 742 documents quarantined

**An earlier version of this section said these were harmless. That was wrong,
and it is worth recording why rather than quietly deleting it.** The reasoning
was that a wrong date in the past merely admits a document earlier than it
deserves, costing recall rather than validity. That is false. The filter admits
a document when `published_at <= :as_of`, so a document stamped `1970-01-01`
satisfies it for **every claim in the corpus**. Its real date is unknown — the
Unix epoch is what a failed timestamp conversion produces, not a date — and it
may well be *after* the claim it is now being served as evidence for. The leak
the store exists to prevent comes straight back, wearing the opposite sign: not
a document that is visibly too new, but one that is invisibly undated.

So they are quarantined, on the same rule and for the same reason as a document
with no date at all. Each keeps its original value as `raw_date_rejected`.

| Cluster | Count | Rule | What it is |
| --- | --- | --- | --- |
| 1970-01-01 | 575 | sentinel | the Unix epoch: a timestamp of zero |
| 1969-12-31 | 75 | sentinel | the same zero in a negative UTC offset |
| 1901-12-14 | 1 | sentinel | 32-bit signed `INT_MIN` |
| year 1 | 86 | window | a null field reaching `strftime` |
| 1879, 1899 | 3 | window | below the 1900 floor |
| 2026-09-21, 2121-01-01 | 2 | window | after today |

**650 of the 742 are the epoch cluster, and a window alone would have missed
every one of them**, because 1970 sits comfortably inside 1900…today. That is
the whole reason the rule has two parts rather than one.

By origin: 738 from CC-NEWS, 4 from enrich. The store's quarantine total rises
from 2,472 to 3,214, a rate of 0.23%.

### Two things the window deliberately leaves alone

**65 documents carry a real date in another calendar** — 39 Solar Hijri (1395,
1398) and 26 Buddhist-era (2561, 2562). They fall outside the window at opposite
ends, so they are tested before it and kept, flagged in `provenance.calendar`
and reported. Converting them is a separate job and a wrong conversion is worse
than a flagged one. The cost is stated rather than hidden: until they are
converted, the 39 Solar Hijri documents carry the same admit-everywhere property
described above. That is a bounded exposure of 39 documents against 650 removed,
not zero.

**15 isolated pre-1996 dates are genuine and are kept.** Every one was checked
by hand and is a real archival article whose URL carries the same date:

| Date | Source |
| --- | --- |
| 1970-04-17, 1971-06-23 | New York Times archives |
| 1982-04-12 | Christian Science Monitor |
| 1984-05-03 | Washington Post archive |
| 1985-01-20 | Los Angeles Times archive |
| 1992-03-05 | Chicago Tribune |
| 1994-01-10 ×2, 1994-01-30 | The New Republic |
| 1995-01-31 | Congressional Record |

An earlier draft of this document called them typos. That was an assertion, not
a measurement, and it was wrong. **Do not tighten the floor to make the table
above look tidy** — it would delete correct data to improve the appearance of a
summary.

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

**A third gap, and the consequential one: no dates, on either side.** There is
no claim-date column and no evidence-date column, and the evidence URLs carry
no date path (Commoncrawl column empty; Snopes URLs are `/fact-check/<slug>/`).
So mocheg is **excluded from the evidence store entirely** — see the callout
under *Time coverage*. Despite being 78% of the project's bundled evidence, it
contributes nothing to time-filtered retrieval, and cannot be made to.

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

## Corrections made against measurement

The first two are cases where a published figure and the file disagreed,
resolved in favour of the file:

- **verite holds 1,001 rows, not the published 1,000**, the extra one being
  out-of-context (325 vs 324). Investigated before recording: no duplicate rows,
  no blank line, no header off-by-one, the index runs contiguously 0–1000 with
  1,001 unique values, and all three shipped files agree on 338/338/325. **The
  row is kept.** Data is not dropped to match a paper.
- **welfake's `expected.csv_rows: 78098` described no file we hold.** Removed
  from `expected` and moved to notes as the source's pre-filter count.

The third is a defect in our own tooling, which made several date measurements
too low before it was caught:

- **`iso_date` silently rejected every ISO *datetime*.** Its pattern required a
  word boundary after the day, and `2019-07-01T10:00:00Z` has none — the day is
  followed directly by `T`. That is the single commonest format in `meta` and
  JSON-LD publication tags, so dates in that form returned `None` and the
  documents fell through to the `Last-Modified` branch or to no date at all.
  Every high-confidence date yield measured before this fix is an **undercount**,
  and the correction is not marginal. Over the same 8,482 enrichment URLs, the
  survey reported **3.3%** publisher metadata; the corrected build measures
  **37.53%** — 3,183 URLs, of which 2,252 are JSON-LD and 931 are `meta` tags.
  That is an **11.4x** undercount, and it was the number the store's usefulness
  was being judged on. The bug was found by a unit test asserting a `meta` tag
  parsed, not by inspecting output — an aggregate yield of "3.3%" looks plausible
  and reveals nothing, which is exactly why it survived. Fixed to
  `(?!\d)`, which still refuses a digit run like `2019-07-011`, with a
  regression test over five datetime forms.

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

---

# New corpora: the 2026-10-01 survey

Our multimodal coverage was thin — verite 71% of rows, factify2 Refute 67% of
images — and both are load-bearing. Fourteen candidates were examined. The
rule, applied mechanically: **acquire** if the corpus fills a real gap, its
licence explicitly permits research use, and it fits the budget; **skip** if
the licence is absent or restrictive, or access needs a request, a form, an
institutional login or acceptance of terms. Nothing was signed, no one was
emailed, no terms were accepted.

Counts below are what each distribution states, retrieved on the day; where a
figure could not be retrieved from the distribution it says so. Licence text is
verbatim in `docs/licences.md` for the two acquired; the raw pages for every
candidate are in `data/interim/candidates/`.

| Candidate | Paper / venue | Authors' counts | Images as | Size | Licence (source) | Access | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DGM4** | Shao et al., *Detecting and Grounding Multi-Modal Media Manipulation*, CVPR 2023 | 230k pairs: 77,426 pristine, 152,574 manipulated (face swap 66,722, face attribute 56,411, text swap 43,546, text attribute 18,588; 32,693 mixed) | files (zips) | 9.99 GB | `apache-2.0` (HF card); S-Lab 1.0 non-commercial (code repo) | open | **ACQUIRED** — candidate out-of-context training source; **failed its shortcut audit, unmapped** (see "DGM4 text-swap") |
| **M4FC** | Geng, Tonglet et al., arXiv 2510.23508; README: EMNLP 2026 Findings | 4,982 images, 6,980 claims, 22 fact-checking organisations | URLs + authors' Wayback URLs | 13 MB annotations; images hydrated | CC-BY-SA-4.0 (README, data/README, data/license) | open | **ACQUIRED** — real-world image+claim verification |
| NewsCLIPpings | Luo et al., arXiv 2104.05893 | not stated in the README retrieved | VisualNews image ids | depends on VisualNews | none in repo (GitHub `license: null`) | needs VisualNews | skip — no licence; images are VisualNews's |
| VisualNews | Liu et al., arXiv 2010.03743 | "more than one million news images" | files | 4.2 GB + 91 GB (directory listing) | none stated | open directory | skip — no licence; 95 GB exceeds the pass |
| COSMOS | Aneja et al. (repo shivangi-aneja/COSMOS) | train 161,752 images / 360,749 captions; val 41,006 / 90,036; test 1,700 / 3,400 | via script after form | not stated | MIT (code repo) | **Google Form** | skip — access request |
| MMFakeBench | liuxuannan/MMFakeBench | not retrieved | files | 6.95 GB (HF listing) | `cc-by-4.0` card, but gated terms | **gated: name, affiliation, PI, terms** | skip — access request, terms |
| Factify3M | Chakraborty et al., EMNLP 2023 (arXiv 2306.05523) | "3 million samples" (abstract) | — | — | arXiv CC BY 4.0 covers the paper only | no download located | skip — not obtainable |
| MediaEval VMU | MKLab-ITI/image-verification-corpus | README states none; measured 42,018 tweets over 360 distinct images, 399 verified image ids | files (rar) + URLs | 757 MB repo | Apache-2.0 (repo) | open | skip — **no gap**: one image carries up to 3,760 tweets, 43.9% of tweets sit on 10 images, and the image alone fixes the label (the unimodal bias VERITE's authors measured on it) |
| Weibo (Jin et al. 2017) | — | — | — | — | the repo search found (GYXie/weibo_dataset) is an unrelated microblog crawl with no licence | — | skip — no licensed distribution located |
| X-POSE | Papadopoulos et al., ECCV 2026 (arXiv 2606.31367) | not stated in the README retrieved | HF, gated | — | Apache-2.0 (code repo) | **institutional e-mail verification** | skip — access request |
| RW-Post | Xu et al., CVPR-W 2026 | not stated; repo holds dev jsonl (988 KB), empty test file | Google Drive / IEEE DataPort | — | custom "Dataset Usage Policy" (DATA_LICENSE.md) | "By accessing or using this dataset, you agree…" | skip — terms accepted by access |
| XFacta | arXiv 2508.09999 | not stated | Google Drive | — | MIT covers the code repo; the dataset states none | open link | skip — no data licence |
| MiRAGeNews | Huang et al., EMNLP 2024 Findings | train 10,000, validation 2,500, five test sets of 500 (HF card) | files (parquet) | 2.08 GB | none on the HF card | open | skip — no licence |
| MultiCaption | Frade et al., arXiv 2601.11220 | "11,088 visual claims in 64 languages" (abstract) | — | — | arXiv CC BY 4.0 covers the paper only | no repository located | skip — not obtainable |
| VERITE images | stefpapad/VERITE (HF) | 1,000 samples (card) | files | — | Apache-2.0 card, gated: research only, no redistribution | **institutional e-mail + terms** | skip — but see below |

**VERITE's own images now exist.** Since July 2026 the authors host the full
image set on Hugging Face behind an institutional-email gate. That would close
verite's 29% gap outright. It requires a human with an institutional login to
request it; this pass does not.

## What the two acquisitions change

**DGM4** was acquired to fill the gap the register has carried since
NewsCLIPpings was deferred: the project had no out-of-context *training* data
at all — VERITE is an evaluation benchmark of 1,001 rows. DGM4's text-swap
pairs keep a real news photograph and swap in another sample's caption, which
is the construction NewsCLIPpings uses. **It did not pass the shortcut audit
that gates that use** (below), so it is held unmapped and the gap stays open.
VERITE stays the only real-world out-of-context *evaluation* set. DGM4's face-swap and face-attribute classes are image
*manipulation*, a task with no block in the current diagram: they are held, and
they change the diagram only if that block is added.

**M4FC** slots into the image+text verification block beside Factify2 and
MOCHEG, as real fact-checked images with verdicts. Its image-contextualisation
and location tasks have no block in the diagram; using them would add one.

**Leakage risk.** M4FC's single largest image host is `factly.in` — 1,005 of
its 4,982 images — the domain that is 100% Refute in Factify2. Byte-identical
images across the two are found by content hash after hydration and kept in one
split by `scripts/split_all.py`; the count is in the completion report.

---

# Known issue: Fakeddit's labels are its subreddits

Measured 2026-10-01 on the seeded 150,000-row image sample (seed 20260901);
full output `data/reports/fakeddit_classes.json`, script
`scripts/fakeddit_classes.py`. This is a confound in a widely used benchmark,
reported with the same weight as the Factify2 hostname result, and it decided
how Fakeddit may be used here.

**The label is a deterministic function of the subreddit.** The authors say so:
"we do not manually label each sample and instead label our samples based on
their respective subreddit's theme. By doing this, we employ distant
supervision". Measured: all **22** subreddits carry exactly one 6-way label —
100% purity for every source, where Factify2 had four 100%-pure domains among
hundreds and its dominant domain carried no label signal.

**The subreddit is recoverable from the text a model must read.** A text-only
multinomial Naive Bayes on titles (unigrams + bigrams, 5-fold) — a weak model,
so a lower bound:

| Target | Accuracy | Majority baseline |
| --- | --- | --- |
| subreddit, all 22 | **59.3%** | 29.9% |
| subreddit, within code 2 (False Connection) | **79.6%** | 44.3% |
| — fakehistoryporn | 96.3% recall | |
| — pareidolia | 94.8% recall | |

A hostname is metadata and can be withheld from a model; a subreddit's writing
style is in the input and cannot.

**The authors cleaned for exactly this, and a token survived.** They "removed
all punctuation, numbers, and revealing words such as “PsBattle” and
“colorized” that automatically reveal the subreddit source." The word
**"circa"** survived: it is in **12.1%** of fakehistoryporn titles and **0.08%**
of all other titles.

**What the text detects does not transfer across subreddits.** Train "code 2 vs
rest" with one code-2 subreddit held out, then measure recall on that
subreddit:

| Held-out subreddit | Recall, seen in training (5-fold) | Recall, held out |
| --- | --- | --- |
| fakehistoryporn | 0.499 | **0.003** |
| pareidolia | 0.258 | **0.005** |
| misleadingthumbnails | 0.093 | **0.037** |
| confusing_perspective | 0.038 | **0.033** |

Code 5 (Misleading Content) collapses the same way: held-out recall 0.000 for
all three of its subreddits. A model that had learned "the image does not
support the text" would carry it to an unseen subreddit of the same class; one
that learned subreddit idiom does not.

**Decisions that follow, recorded as consequences of the measurement:**
- Fakeddit code 2 is `unmapped`, not `media_mismatch`. Training on 28,503
  Reddit rows from four communities to evaluate on 325 VERITE news fact-checks
  would teach Reddit idiom, not mismatch.
- fakehistoryporn is excluded from reason-code supervision whatever Fakeddit's
  labels are later mapped to (`reason_codes.EXCLUDED_FROM_REASON_SUPERVISION`).
  Its titles are anachronism jokes (a present-day event captioned as a historical
  one, usually ending in "circa") detectable from the text alone: comedy, not
  deception, and a different construct from VERITE miscaptioned. Sample titles
  are in `data/interim/fakeddit_classes_full.json`, which stays out of git:
  Fakeddit states no licence.
- The paper's own manual check of its labels reached Cohen's kappa 0.54 on 150
  pairs, "showing moderate agreement and that some samples may represent more
  than one label."

For scale: the paper's text-only BERT scores 76.8% on 6-way test against 85.9%
for BERT+ResNet50 (its Table 4). Most of what Fakeddit's 6-way measures is
available from the title alone.

---

# DGM4 text-swap: shortcut audit before any media_mismatch mapping

Measured 2026-10-01, re-run 2026-10-02 with the decision rule built in, over
the 179,295 distinct train+val records; `data/reports/dgm4_audit.json`, script
`scripts/dgm4_audit.py`. Same questions as the Fakeddit audit.
**Outcome: DGM4 is unmapped** (decision at the end of this section).

1. **No publisher is label-pure.** Per-publisher purity 0.33–0.34; text-swap
   share by publisher BBC 5.0%, Guardian 8.2%, Washington Post 10.9%,
   USA Today 13.5%. NMI(publisher; label) = **0.005** (Factify2's
   domain/label NMI was 0.31).
2. **Text recovers the label only by memorising captions.** Text-only, text-swap
   vs pristine: **82.6%** vs a 79.4% majority on random folds. DGM4 reuses
   swapped captions — 15,682 swap rows draw on 7,600 captions, one of them
   swapped onto 109 images ("powerful people": 218 swaps, 2 pristine). With
   folds grouped by caption the lift vanishes: **79.0%** vs 79.4% majority,
   swap recall 0.032. DGM4's own train/val split already keeps captions
   disjoint (0 of 7,600 straddle), and `scripts/split_all.py` now enforces
   caption grouping so no reconciliation can undo it.
3. **Text recovers the publisher (80.9% vs 52.5% majority), but the publisher
   does not track the label** (NMI 0.005).
4. **Half the swaps cross publishers**: of 15,682 swapped captions, all found
   among DGM4's own pristine captions, 7,814 come from the image's own
   publisher, 7,867 from another, 1 ambiguous. The rule "caption style ≠
   image's publisher" flags 43.0% of swaps and 15.3% of pristine pairs — a
   style channel open to any model that can recognise a publisher from the
   photograph. Restricting to the **7,814 same-publisher swaps** (6,826 train,
   988 val, 4,671 captions) closes it by construction; on that subset,
   caption-grouped text-only accuracy is 88.46% vs 88.57% majority.
5. **Face-edit rows separate exactly from pure text swaps by metadata**: every
   one of the 17,448 combined rows has a non-empty `fake_image_box` and a
   `manipulation/` image; no pure text-swap row has either. Text alone cannot
   tell them apart (51.9% vs 52.7% majority), as expected — the caption swap is
   the same procedure. Only pure text-swap rows are candidates; the 17,448
   rows that are both a face edit and a text swap (`face_swap&text_swap`
   9,480, `face_attribute&text_swap` 7,968) are excluded outright, construct
   ambiguous.
6. **Deduplication comes first, and nothing straddles train/val.**
   `train.json` holds 208,184 rows of which 157,169 are distinct: **51,015
   byte-identical repeats**, every one of them a pristine (`orig`) row, at
   most 3 copies of a row (89,780 train rows sit in a repeated group).
   `val.json` has none. Every count in this section, and every record in the
   processed layer (`scripts/records.load_dgm4`), is over distinct rows. No
   row, source id or image path appears in both train and val, and no swapped
   caption does; 4 caption strings recur across the split, all on non-swap
   rows (12 pristine, 11 face edit, 2 text attribute). **No internal
   train/val leakage.** The repeats are a defect of a different kind: left in,
   they triple the weight of some pristine pairs in any model trained on the
   shipped file.

Every text-swap image is a real news photograph; 11,428 of the 15,682 also
appear in DGM4 as a pristine row with their true caption, always in the same
split (no source id or image crosses train/val), so the corpus carries
VERITE-style matched true/false pairs.

### The image side: CLIP image-text similarity (measured 2026-10-02)

`scripts/clip_pairs.py`, CLIP ViT-B/32 zero-shot, cosine(image, caption), AUC
of that score for telling true pairs from defective ones, seeded bootstrap 95%
interval; `data/reports/clip_pairs.json`. Nothing is trained, so any separation
comes from the pair itself.

| True vs defective | n | mean cosine | AUC [95% CI] |
| --- | --- | --- | --- |
| DGM4 pristine vs all text swaps | 8,000 / 15,682 | 0.314 / 0.233 | 0.894 [0.889, 0.898] |
| DGM4 pristine vs same-publisher swaps | 8,000 / 7,814 | 0.314 / 0.239 | 0.879 [0.874, 0.884] |
| **DGM4 matched: same image, true vs same-publisher swapped caption** | 6,106 pairs | 0.323 / 0.242 | **0.907 [0.902, 0.912]**; true caption higher in **94.3%** |
| VERITE true vs out-of-context | 231 / 261 | 0.332 / 0.290 | 0.764 [0.723, 0.803] |
| VERITE true vs miscaptioned | 231 / 231 | 0.332 / 0.320 | 0.587 [0.538, 0.640] |

**The swap is a property of the pair.** With the image held fixed, only the
caption's relation to it can move the score, and it separates at 0.907. That
is the opposite of Fakeddit's False Connection, whose text-detectable signal
was subreddit idiom.

**DGM4's swaps are easier than VERITE's.** The same zero-shot score separates
VERITE out-of-context at 0.764: VERITE's out-of-context images were chosen to
be plausible for the caption, DGM4's swaps are drawn by text similarity alone.
A model trained on DGM4 can learn "caption unrelated to image" that VERITE's
curated cases defeat; VERITE as held-out evaluation is what would show it.
VERITE miscaptioned barely separates (0.587) — a genuine image with a subtly
false caption keeps its topic, so similarity is the wrong tool for it.

### The decision (rule of 2026-10-02, applied by the script)

Map text-swap to `media_mismatch` iff **(a)** no publisher is label-pure,
**(b)** a text-only classifier is near the majority baseline, and **(c)** swaps
are within-publisher; otherwise leave DGM4 unmapped. "Near" was fixed at
within 1 point and "label-pure" at class purity ≥ 0.90 (the Factify2
threshold) before the run.

| | (a) max publisher purity | (b) text-only − majority | (c) swaps within-publisher | Passes |
| --- | --- | --- | --- | --- |
| Whole text-swap class | 0.342 ✓ | +3.15 pts random folds ✗; −0.42 caption-grouped | **50.2% cross ✗** | **no** |
| Same-publisher subset (7,814 swaps vs 60,550 pristine) | **0.960 (BBC) ✗**, Washington Post 0.910 | −0.08 pts caption-grouped ✓; +0.15 random | by construction ✓ | **no** |

**DGM4 stays unmapped. `media_mismatch` has no training source**: its only
supervision is VERITE out-of-context, evaluation-only — 301 of its 325 rows
have a usable image at the 2026-10-02 build (261 before the archive pass
recovered 116 VERITE images; final figure in the completion report).
That is a stated limitation of the reason-code head: it can be *scored* on
media mismatch and cannot be *trained* on it.

The subset's (a) failure needs reading with care, and is recorded rather than
argued away. Over pristine + same-publisher swaps, 88.6% of rows are pristine,
so purity sits near 0.886 for a publisher that carries no information at all.
BBC's 0.960 is a low swap rate (4.0% against 11.4% overall; Guardian 14.5%).
A publisher-only classifier on the subset scores **88.57% against an 88.57%
majority and never predicts a swap** (NMI 0.014). A base-rate-corrected reading
of (a) would pass the subset. That reading was not the rule fixed before the
run; the whole class fails (c) regardless; and the subset was this project's
extension, not part of the rule as given. Flipping it is one constant,
`reason_codes.DGM4_AUDIT_PASSED`, which `tests/test_reason_codes.py` holds to
this report. Should it be flipped, the subset is weak supervision (synthetic,
not fact-checked), training only, and two further cautions apply. `none` is
reserved for five adjudicated labels, so DGM4's pristine pairs would stay
unmapped and every trainable `media_mismatch` row would come from one corpus —
"this row is DGM4" would predict the code perfectly among training rows. And
DGM4's swaps are easier than VERITE's (AUC 0.907 vs 0.764 zero-shot).

**What is publishable either way:** DGM4's text-swap class as shipped is
partly solvable from text alone, but only by memorising reused captions
(+3.15 points on random folds, gone on caption-grouped folds), and half of its
swaps cross publishers, which opens a writing-style channel (43.0% of swaps
vs 15.3% of pristine pairs flagged by a caption-reads-like-another-publisher
rule). Both are properties of how the benchmark was constructed, like
Fakeddit's subreddit labels, and any DGM4 result should be read beside them.

### Cross-dataset transfer is always reported

Never train on DGM4 and evaluate on VERITE without reporting the transfer gap
beside the in-domain number (`docs/processed_schema.md`). Synthetic-to-real is
the weakest transfer case: published out-of-context work reports drops of
roughly 30 points across it, and this project's WELFake↔ISOT text transfer gave
macro F1 0.251 (figure stated by the project lead 2026-10-02; the run is not in
this repository).

---

# Known issue: the evidence images show the verdict (MOCHEG, Factify2)

Found 2026-10-03 while reviewing duplicate-image groups; measured over the
processed layer; `data/reports/verdict_images.json`. Reported with the weight of
the Factify2 hostname result, because it is the same failure in a worse form:
the answer is in the input.

A fact-check article illustrates itself with its own ruling. PolitiFact pages
carry the Truth-O-Meter (MOSTLY TRUE, HALF TRUE, PANTS ON FIRE); Snopes article
images carry the rating badge (FALSE, MISCAPTIONED, LABELED SATIRE); Lead
Stories stamps its verdict across a screenshot of the claim; BOOM, Factly, The
Quint and AFP stamp the claim image FAKE / MISLEADING or set their red FACT CHECK
seal in the corner. Two corpora take their images from those articles. A model
shown the image is shown the label.

**MOCHEG** -- evidence images are the article's images. A 10-nearest-neighbour
vote on the CLIP ViT-B/32 embedding of a record's evidence images (no text,
nothing trained beyond the vote, 5-fold, seed 20260901):

| MOCHEG records | n | Image-only accuracy | Majority baseline | Recall NEI / refuted / supported |
| --- | --- | --- | --- | --- |
| with a verdict image | 1,199 | **91.7%** | 36.0% | 0.861 / 0.982 / 0.911 |
| without one | 11,470 | 48.5% | 45.7% | 0.160 / 0.743 / 0.342 |

The leak is spread across classes (NEI 432, refuted 395, supported 372 records),
so class balance does not reveal it; only the pixels do. A balanced leak teaches
"read the badge".

**Factify2** -- the Refute class's claim and document images were taken from
the fact-checkers' own articles. **1,486 of 8,500 Refute records (17.5%) carry a
verdict image, against 15 of the other 34,000.** The rule "a verdict image is
present -> Refute" is 99.0% precise. By host: boomlive.in 984 image
references, snopes.com 446, factly.in 39 -- the domains that are 100% Refute in
the hostname analysis. The hostname confound has a pixel-level counterpart, and
withholding the URL does not remove it.

**Controlled.** Every verdict image found is in `data/placeholders.yaml` with
status `verdict` (registry v8-v12): it fails the third fetch gate and never
reaches `image_paths`. 2,477 distinct images: Factify2 1,565, MOCHEG 926, M4FC
14, Fakeddit 3. For Factify2 a claim or document image is a required slot, so a
record whose image is a verdict becomes `usable: false` (`images_missing`): the
Refute class loses rows, recorded in the completion report. For MOCHEG the
evidence images are an optional set and the verdict image is dropped from it.

How they were found: 50 from the duplicate-group review, 122 by CLIP
nearest-neighbour expansion, a linear probe on CLIP embeddings
(`scripts/content_probe.py`) that ranked every image of MOCHEG, AVerImaTeC,
Factify2, M4FC and Fakeddit, and a search of the OCR text for rating words.
Every entry was confirmed by eye. Thumbnail-size review over-called: 799 entries
that rested on a thumbnail alone were re-reviewed at full size and **19 were
removed** (a claim graphic with its own red markings, a news banner).

**Not exhaustive; the residual is estimated, not assumed.** Factify2 + M4FC:
every image the probe scored above 0 was reviewed at full size; random samples
below that found 1 of 60 in the next 5,319 (~89, 95% CI 0-266) and 0 of 100 in
the remaining 46,415 (rule-of-three bound 1,392). MOCHEG: estimates in the
report. Fakeddit: 0 of the probe's top 60.
