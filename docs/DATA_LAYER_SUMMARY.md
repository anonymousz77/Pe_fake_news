# Data layer — closing summary

Measured 2026-09-11; **updated 2026-10-03 at the close of the closeout pass**
(`docs/completion_report.md` holds the per-class tables, dead-image counts and
every corrected figure). Every number here comes from a generated report in
`data/reports/` or `data/processed/splits/`, not from a paper or an estimate.
The data layer is closed: nothing below is scheduled to change.

Detail and reasoning live in `docs/data_card.md` and `docs/evidence_store.md`.
This page is the index to them.

## The twelve enabled datasets

| Dataset | GiB | Files | Rows | Usable | Date field | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| mocheg | 40.352 | 137,830 | 21,184 claims | 21,184 | none | image release acquired 2026-10-01; evidence cannot be time-filtered, below |
| fakeddit | 15.763 | 148,815 | 623,342 | 615,822 | `created_utc`, 2008–2019 | images are a seeded sample of 150,000 |
| factify2 | 11.042 | 84,653 | 42,500 | 37,807 | none | 5 classes, 8,500 each as shipped |
| averitec | 10.758 | 11 | 5,783 | 3,567 | `claim_date`, 1919–2023 | HF **model**-type repo; test (2,215) unlabelled |
| dgm4 | 10.205 | 35 | 179,295 | 179,295 | none | train+val, deduplicated; test forbidden; acquired 2026-10-01 |
| m4fc | 1.125 | 4,980 | 3,493 | 3,441 | none | test excluded; acquired 2026-10-01 |
| welfake | 0.228 | 1 | 72,134 | 72,134 | none | 37,106 fake / 35,028 real |
| verite | 0.213 | 2,023 | 1,001 | 914 | none | the only real-world out-of-context set; evaluation-only |
| averimatec | 0.209 | 14 | 945 | 944 | `date`, 2005–2023 | 897 of 945 are Refuted; evaluation-only |
| isot | 0.149 | 3 | 44,898 | 44,898 | none | negative control; acquired 2026-10-01 |
| fakenewsnet | 0.059 | 49 | 23,194 | 23,192 | none | 17,439 real / 5,753 fake / 2 conflicting |
| liar | 0.004 | 5 | 12,836 | 12,836 | none | 6-way truth scale |

All twelve acquired. `verify.py --all` reports **one mismatch across 12
datasets**: M4FC claims 6,789 against the 6,980 its README states, unresolved.
Two further datasets, `visualnews` and `newsclippings`, are deliberately
disabled: no licence stated, 68 GB together, and enabling them is a budget
decision requiring `budget_gb` and `BUDGET_GB` to move in the same change.

**isot is a negative control and nothing depends on it.** All its real news is
Reuters, so a model can score well by identifying the publisher rather than
detecting deception. No ISOT number is ever a headline result.

## The evidence store

1,368,613 documents indexed, 3,214 quarantined — a 0.23% quarantine rate.

| Source | Attempted | Indexed | Quarantined | High-confidence date |
| --- | --- | --- | --- | --- |
| CC-NEWS | 40 WARCs | 1,342,262 | 738 | 1,071,949 — 79.9% |
| Wikinews | 22,394 pages | 21,457 | 937 | 21,457 — 100% |
| enrich | 8,482 URLs | 4,894 | 1,539 | 3,183 — **37.5% of URLs** |
| **store** | | **1,368,613** | **3,214** | **1,096,589 — 80.1%** |

Three denominators on purpose, because the three sources fail in three different
places. The enrich figure is the one that matters: it is the evidence claims
actually cite, and 37.5% of those URLs can be placed on a timeline at all. It
replaces a previously reported 3.3%, which was measured with a regex that
silently rejected every ISO *datetime* — an 11.4x undercount.

**What dated the store**, whole store including quarantine:

| `date_source` | Documents | Confidence |
| --- | --- | --- |
| `jsonld` | 686,719 | high |
| `meta_published` | 388,413 | high |
| `warc_date` | 245,650 | low |
| `url_path` | 24,792 | low |
| `explicit_field` | 21,457 | high |
| `none` | 2,078 | quarantined |
| `wayback_capture` | 1,582 | low |
| `implausible_date` | 742 | quarantined |
| `http_last_modified` | 394 | quarantined |

**Date coverage.** 97.0% of the store is 2016–2023, the CC-NEWS window, with
2020 alone at 34.7% and 2019 at 21.5%. That distribution is a fact about the
WARC allocation, not about news volume. Below 2016 there are 40,801 documents
(2.98%) and from 2024 onward 537 (0.04%). In Gregorian terms the store runs from
a New York Times archive page dated 1970-04-17 to 2026.

**Budgets**: the September pass stored 21.72 of 30 GB and transferred 43.49 of 60 GB; the closeout pass
(2026-10-01..03) stored **58.18 of 60 GB** and transferred about **102.6 of 120 GB** (`docs/completion_report.md`).

## Known defects, each with its number

**5,057 images on disk were not images of anything.** A fetch that accepts any
HTTP 200 accepts the host's "image unavailable" bitmap, its logo and its
loading spinner, and every one of those was counted as a recovered image.
An audit of all 225,629 files on 2026-09-21 found 97 that do not decode at all
(94 HTML error pages, 2 HTML fragments, 1 truncated PNG) and **5,057 that
decode perfectly and carry no evidence**: 2,959 placeholders, 2,062 pieces of
article furniture, 36 single-flat-colour images including five 1x1 pixels.
They are listed by content hash in `data/placeholders.yaml` (versioned) and
counted as missing, not present. Corrected coverage is below. The files are
marked, not deleted: `data/interim/<dataset>_placeholders.jsonl`. The registry has
since grown to **v13: 4,135 hashes**, and 19,851 files on disk count as missing, after
review of every duplicate group, every small single-copy image, OCR of every
single-copy image, and the verdict sweep below.

**The placeholder shortcut is worth +1.85 points on fakeddit.** Measured the
same way as the hostname confound: one binary feature, "is this image in the
placeholder registry", 5-fold held-out, seed 20260901. On fakeddit it scores
**40.30%** against a 38.45% majority baseline and 16.7% uniform. On factify2
it scores **22.22%** against 21.44% majority and 20.0% uniform. Those lifts
are small because the feature fires on only ~2.2% of items -- the danger is
conditional, not marginal: **when it fires on fakeddit it is label 4 with
86.88% purity**, because 2,899 copies of one imgur placeholder all sit in that
class. The same bitmap in one class is a free 2,901-record giveaway to any
model that memorises it.

**Fakeddit's labels are its subreddits** — measured 2026-10-01, reported with
the weight of the hostname result. All 22 subreddits are 100% one 6-way label
(the authors label "based on their respective subreddit's theme"). Title text
alone identifies the subreddit at **59.3%** against a 29.9% baseline, and at
**79.6%** within False Connection against 44.3%. The authors stripped
"PsBattle" and "colorized" because they "automatically reveal the subreddit
source"; **"circa" survived**, in 12.1% of fakehistoryporn titles against 0.08%
elsewhere. Hold one False-Connection subreddit out of training and text-only
recall on it collapses from 0.04–0.50 to **0.003–0.037**. Consequence: Fakeddit
stays `unmapped` for reason codes and fakehistoryporn is excluded from
reason-code supervision. Detail in `docs/data_card.md`.

**The evidence images show the verdict** -- measured 2026-10-03. Fact-check
articles illustrate themselves with their own ruling (Truth-O-Meter, rating
badge, FAKE stamp, fact-checker seal), and MOCHEG and Factify2 take their images
from those articles. On MOCHEG, a CLIP nearest-neighbour vote on the images
alone scores **91.7%** against a 36.0% majority on the 1,199 records that carry
one, and 48.5% against 45.7% on the rest. On Factify2, **1,486 of 8,500 Refute
records** carry one against 15 of the other 34,000: "verdict image present ->
Refute" is 99.0% precise, and the images come from boomlive.in and snopes.com,
the hostname confound in pixels. 2,477 such images are registered with status
`verdict` and count as missing; the sweep's residual is estimated in
`data/reports/verdict_images.json`. Detail in `docs/data_card.md`.

**DGM4's text-swap class failed its shortcut audit, so `media_mismatch` has no
training source** — measured 2026-10-02 over 179,295 distinct rows. No
publisher is label-pure (max 0.342; NMI 0.005), but **50.2% of swaps take a
caption from another publisher**, a writing-style channel: a
caption-reads-like-another-publisher rule flags 43.0% of swaps and 15.3% of
pristine pairs. Text alone beats the majority by **+3.15 points** on random
folds, all of it caption memorisation (15,682 swaps reuse 7,600 captions, one
of them 109 times); on caption-grouped folds it is −0.42. The same-publisher
subset fails the pre-fixed purity test (BBC 0.960) on base rate alone — a
publisher-only classifier scores exactly the 88.57% majority — and is held
unmapped as the rule was written. `train.json` repeats **51,015 rows byte for
byte** (all pristine); dropped before every count. Nothing straddles train/val:
no row, id, image or swapped caption. VERITE out-of-context remains
`media_mismatch`'s only supervision, evaluation-only.

**Factify2 image recovery is class-imbalanced.** After the closeout pass
exhausted every route, Refute holds **81.6%** of its images against 92.3–99.7%
for every other class, and **66.3%** of its *rows* (5,635 of 8,500), since a
row needs both its images; other classes keep 85.1–99.4% of rows. Recovery was
first stopped at 1,278 archive images on the grounds that completing it would
raise domain-only accuracy; it was then completed on explicit instruction, and
the measured corpus-wide domain-only accuracy on usable images is **40.4%**
(it was 38.2%). Every remaining image is dead with a terminal reason in
`data/dead/factify2.csv`.

**Factify2's `val` holds 0 Refute records.** Refute sits in a few indivisible
domain components, and once train and test take them there is nothing left. Refute model selection must use cross-validation within train, or
train/test only — never val.

**Factify2 image provenance predicts the label.** `factly.in`, `boomlive.in` and
`snopes.com` are each 100% Refute. This is controlled in the split, not the
data; see the guarantee below.

**verite has a usable image for 914 of 1,001 rows (91.3%)**: true 307,
miscaptioned 307, out-of-context 300. It was 71.3% before the archive pass
recovered 120 images and a CLIP fingerprint match found one more in MOCHEG.
Every counted image was checked against the CLIP embedding the authors ship for
each row: five that decoded were not the authors' image (three publisher logo
cards, two wrong archive snapshots) and are excluded. The remaining 87 rows are
dead with terminal reasons; the authors' own gated image release would close
them and needs a human request.

**fakeddit images are a sample, not the corpus.** 150,000 seeded stratified
draws, **142,480 usable (95.0%)** after the archive pass, per class 94.3–98.4%.
The placeholder registry counts imgur's removed-image bitmap and the linked
sites' logos (Reddit link previews) as missing. Loss is otherwise even across
classes and dominated by genuine 404s. "Text only" in the modality table means "not
sampled", not "no image".

**mocheg cannot enter the evidence store at all.** Its 43,148 evidence URLs are
78% of all bundled evidence in the project, and the corpus ships no claim-date
field, so nothing in it can be placed on a timeline relative to its claim. That
is a property of the dataset, not of this pipeline.

**742 documents carried dates that could not be dates**, and were quarantined:
650 at the Unix epoch, 86 in year 1, and a handful below 1900 or after today.
These were a hole in the time filter, not untidy data — a document stamped
1970-01-01 satisfies `published_at <= :as_of` for every claim ever made. Each
keeps its original value as `raw_date_rejected`.

**65 documents carry a real date in another calendar** — 39 Solar Hijri, 26
Buddhist-era — and are deliberately left unconverted. Until they are converted
the 39 Solar Hijri ones sort before every claim and so are admitted for all of
them: the same property the epoch quarantine removes, at 39 documents rather
than 650. Recorded, bounded, and accepted.

**Three images are byte-identical across factify2 and verite.** Two are
genuine shared source photographs -- the same viral image entered both corpora
because both draw on the same fact-checked events -- and one is the 1x1 white
pixel. Any experiment training on one and evaluating on the other must drop
them or it scores itself on pixels it trained on. Recorded in
`data/cross_corpus_duplicates.yaml` and enforced by
`placeholders.assert_no_cross_corpus_leak()`, which raises; it is not a comment.

**No licence has been verified.** Every `UNVERIFIED` marking means nobody has
read the terms, and `redistributable: false` everywhere is a conservative
default rather than a finding. `docs/provenance.md` tracks what has actually
been checked by hand.

## The guarantees that hold

**The time filter cannot be bypassed.** It is a predicate inside
`retrieve.SEARCH_SQL` — a JOIN with `published_at IS NOT NULL AND published_at
<= :as_of` — not a pass over the results, so an undated or future-dated document
cannot enter the candidate set whatever the ranking does, and `k` still means k.
Undated documents are never indexed at all, and `published_at` is `NOT NULL` in
the schema as a third line of defence.

This is proven against the real index, not a fixture. `scripts/redtest.py`
plants three documents into the 1,368,613-document store — one dated the day
before the as-of date, one the day after, one absent from the temporal table —
inside a transaction it always rolls back, and sweeps k past the corpus size.
Result, 2026-09-11:

| Probe | k | Returned | Future plant | Undated plant | Day-earlier plant |
| --- | --- | --- | --- | --- | --- |
| nonce, k = 1 … 2,000,000 | up to 2,000,000 | 1 | never | never | always |
| bounded (`hospital`) | 880,160 | 62,380 of 88,016 matched | no | no | yes |
| whole corpus (`a`) | 2,000,000 | 703,362 of 973,757 matched | no | no | yes |

703,362 documents came back in a single result set and not one was dated after
the as-of date; the newest was the as-of date itself, the inclusive boundary.
Afterwards both row counts, a `quick_check` and a 50-deep ranking fingerprint
were unchanged, and a byte-for-byte comparison against a copy taken beforehand
found no difference in either database.

**Factify2 splits are source-disjoint on label-predictive domains**, purity
threshold ≥ 0.90, 780 domains constrained over all 42,500 records, seed
20260901, **0 leakage violations, enforced rather than checked**. Usable records
split 22,747 / 4,735 / 10,325. A hostname-only classifier scores **40.3%** on a
random split against 20.0% uniform; on this split it scores 12.7%, but the test
set is 54.4% Refute, so the figure to beat there is the **56.9% majority**.
Every Factify2 result must be reported beside the applicable figure.

Disjointness on *all* domains is impossible and `--disjoint strict` still
refuses, so the finding stays reproducible: shared domains link 95.8% of
records into one component because `pbs.twimg.com` touches 84.7% of them, and
that domain carries no label signal anyway.

**Test data cannot enter the pipeline.** Every register entry declares
`forbidden_patterns`, required even when empty, and they are enforced at fetch
time — each method enumerates before downloading and a match aborts the whole
dataset — not only in tests. `tests/test_sources.py` additionally refuses any
overlap between include and forbidden globs in either direction.

**Raw data is immutable and its integrity is recorded.** Downloads land in
`data/raw/<name>/` and are never edited in place; derived artefacts go to
`interim/` or `processed/`. `data/MANIFEST.sha256` carries 378,252 file hashes (rebuilt 2026-10-04; every hash recorded before the closeout pass is unchanged except the append-only provenance ledgers, and the 97 removed paths are byte-identical in quarantine)
and `manifest.py verify` checks disk against it.

**Nothing published carries data.** `scripts/prepush_check.py` refuses any
staged path that holds data bytes, a per-record label or URL for a
non-redistributable dataset, or a gate credential. Published per-record files
carry identifiers and split assignment only. Aggregate statistics are published
deliberately — per-domain purity and counts are facts about public web
properties, and they are what make the results above checkable by a reader with
no access to the data.

**The suite is offline and data-free.** 445 tests pass (2026-10-04) with no network and no
store on disk; the one skip is the real-index red-test, which is opt-in behind
`PE_FAKE_NEWS_REDTEST` precisely so the default run stays that way.
