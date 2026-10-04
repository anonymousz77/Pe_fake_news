# The evidence store (`data/bible/`)

## What it is

The time-filtered corpus of dated documents that claims are checked against.
When the pipeline verifies a claim, it retrieves from here — and only from
here.

It lives at `data/bible/`, resolved through `configs.paths.BIBLE`. It is
gitignored and expected to reach tens of gigabytes.

## What it is not

It is not documentation. The name invites that misreading, so it is worth
stating plainly: no hand-written prose belongs in `data/bible/`, and no
evidence documents belong in `docs/`. Prose is tracked in git and written by a
person; the evidence store is data, ignored by git, and assembled by scripts.

## The dating invariant

Every document in the store carries a publication date, and retrieval for a
claim is filtered to documents published **before that claim's date**.

This is the whole reason the store exists as a separate thing rather than "the
union of the corpora in `data/raw/`". A retrieval corpus that contains
documents published after a claim was made will confirm that claim with
hindsight the model should not have. That failure is invisible in aggregate
metrics — accuracy simply looks better than it is.

Consequences:

- A document with no reliable publication date **does not enter the store.**
  An estimated date is worse than an exclusion, because it silently converts a
  known gap into an unknown error.
- **A date that cannot be one does not count as having a date.** See the
  plausibility window below: the rule is the same rule, applied to values that
  look like dates and are not.
- Dates are stored as UTC dates, not localised strings.
- The filter is applied at retrieval time and cannot be disabled by a flag. If
  you need an unfiltered retrieval for an ablation, build a separate index and
  name it something that cannot be mistaken for this one.

## Relationship to the register

The evidence store is *derived*: it is built from corpora declared in
`data/sources.yaml`, but it is not itself a register entry and gets no
`raw_dir()`. `forbidden_patterns` therefore constrain it transitively — a
document that could not be ingested into `data/raw/` cannot reach the store.

The averitec knowledge store is the case that matters most. Its `test/` shard
is forbidden, which means test-claim retrieval evidence never enters
`data/raw/averitec/`, and therefore never enters the evidence store, and
therefore cannot be retrieved for any claim.

## Layout

```
data/bible/docs/<doc_id>.json      dated documents — the indexable store
data/bible/undated/<doc_id>.json   quarantined; counted, reported, never indexed
data/bible/index/bm25.sqlite       FTS5 over title + text
data/bible/index/temporal.sqlite   doc_id -> published_at, source_domain, url
data/bible/_src/                   upstream dumps kept for re-parsing
data/bible/_state.json             transfer ledger and per-source resume markers
data/interim/bible_scan_*.jsonl    derived: the report's fields, one line per doc
```

`doc_id` is `sha256(url)`, so the filename says nothing about the date. The
requirement that the date be readable without opening the document is met by
`temporal.sqlite` — the sidecar index — which is also what the time filter joins
against, so the auditable thing and the enforcing thing are the same thing.

`_state.json` splits into two halves with different owners. The transfer ledger
and the per-source resume markers are written by the running build and nothing
else may touch them. The `notes` array is hand-written commentary — `bible.py`
never writes it — so it is the one part that can go stale, and correcting it is
a person's job. It went stale exactly once: a note saying the CC-NEWS run
stopped at 2 of 40 WARCs survived the resume that took it to 40 of 40, and was
corrected on 2026-09-10. `warcs_done` was right the whole time.

## Settled questions

**Deduplication is by URL, and only by URL.** `doc_id = sha256(url)` and
`write_document` refuses to overwrite, so the same URL reached from two sources
yields one document and the first writer wins. The same *article* at two
different URLs is two documents. That is deliberate: collapsing them needs a
content-similarity judgement, and a wrong merge silently drops the copy whose
date was the usable one.

**Documents store extracted text, never raw HTML**, capped at 20,000 characters
for CC-NEWS. The cost of that choice is that re-extracting a document means
re-fetching it, which is why the enrichment half of the store stays on `bs4`
rather than being rebuilt onto the cheaper extractor.

**The index is SQLite and lives beside the store**, not in `data/interim/`: it
is derived from these documents and only these, and it has to move and be
deleted with them. FTS5 ships with Python and `bm25()` is built into it, so the
index adds no dependency.

## The plausibility window

An impossible date is a hole in the time filter, not untidy data, and it is
worth being exact about why. The filter admits a document when
`published_at <= :as_of`. A document stamped `1970-01-01` therefore satisfies it
for **every claim ever made**. Its real publication date is unknown — that is
what the stamp means, since the Unix epoch is what a failed timestamp conversion
produces — and it may well be *after* the claim the document is now being served
as evidence for. So the leak the store exists to prevent comes back, wearing the
opposite sign: instead of a document that is visibly too new, one that is
invisibly undated.

`make_document` is the single place this is enforced, because all three sources
funnel through it. A date is refused when it is:

- **outside 1900 … today.** The floor is far below anything real here — the
  store's earliest genuine document is from 1970 and its bulk begins in 2016 —
  and the ceiling is today, because nothing can have been published tomorrow.
- **a sentinel**: `1970-01-01` and its two neighbouring days, which are a
  timestamp of zero read in any UTC offset, and `1901-12-13/14`, which is the
  same failure in 32-bit signed seconds. These sit *inside* the window, which is
  exactly why the window alone is not enough.

A refused date is not deleted. It is kept on the record as `raw_date_rejected`,
with the reason in `provenance.date_rejected_because`, so a later parser or
calendar fix has something to work from. The document moves to `undated/` and is
counted like any other quarantine.

### Two things the window deliberately does not touch

**Jalali and Buddhist-era dates are real dates.** Solar Hijri runs about 621
years behind the Gregorian year and the Buddhist era 543 ahead, so both fall
outside the window at opposite ends — and both are tested before it and kept.
Converting them is a separate job, and a wrong conversion is worse than a
flagged one. This is recorded as a known issue rather than fixed, and the cost
is stated rather than hidden: until they are converted, a Jalali-dated document
carries the same admit-everywhere property described above. There are 65 of
them, so the exposure is small and bounded, but it is not zero.

**An isolated old date is not a broken one.** Every pre-1996 date in this store
that is not a sentinel was checked by hand, and each is a genuine archival
article whose URL carries the same date: New York Times pieces from 1970 and
1971, a Christian Science Monitor story from 1982, a Washington Post archive
page from 1984, the Congressional Record for 1995. An earlier draft of the data
card called them typos. That was an assertion, not a measurement, and it was
wrong. Do not tighten the floor to make the per-year table look tidy; it would
delete correct data to improve the appearance of a summary.

## Reporting on a store larger than the machine

The report reads four fields per document — a date, a source, a confidence and a
domain — and getting them meant opening all 8.6 GB of the store. On a 16 GB
machine that fills the file cache until the OS reports a few hundred megabytes
free, and the report was killed for memory three times while using 217 MB
itself. The process was never the problem; the walk was.

Free memory returns the moment a process exits, so the fix is to touch less of
the store per process rather than to use less memory. `bible.py --scan` reads
the store in resumable slices and appends about 90 bytes per document to
`data/interim/bible_scan_docs.jsonl`. Each invocation exits cleanly and the next
resumes from the file's own line count; a run killed mid-write leaves a partial
final line, which the next truncates before appending. Afterwards the report
reads 120 MB instead of 8.6 GB.

The scan is an optimisation and never a source of truth. `--report` uses it only
when its line count matches the folder's file count, and otherwise walks the
store as before. That check is deliberately weak — it catches a truncated or
stale scan, not an edited document — and the scan lives in `data/interim/`
precisely so that the answer to any doubt is to delete it. A test asserts the
report produces identical numbers scanned or walked, because an optimisation
that changes a number is not an optimisation.

## The red-test

The dating invariant is not taken on trust. `scripts/redtest.py` plants three
documents into the **real** index — one dated the day before the as-of date, one
dated the day after, and one absent from the temporal table entirely — and
demands the first comes back and the other two never do. It writes
`data/reports/redtest.json`.

It runs against the store as built, not a fixture, because a filter that holds
on four rows says nothing about one that has to hold on 1.4 million: SQLite
picks different plans at different table sizes, and a `k` that exceeds a toy
corpus is dwarfed by a real one. The plants go in inside a `BEGIN IMMEDIATE`
transaction that is rolled back in a `finally`, so the index is restored whether
the test passes or fails, and nothing is ever committed. Restoration is checked
three ways: both row counts, a `quick_check`, and a ranking fingerprint of a
fixed query taken before planting and compared after — the last is what would
catch a rollback that restored the count but disturbed the FTS segments.

Measured on 2026-09-11 against the 1,368,613-document store, `as_of`
2020-06-15. Three probes, because each closes a hole the others leave open:

| Probe | k | Returned | Future plant | Null plant | Day-earlier plant |
| --- | --- | --- | --- | --- | --- |
| nonce, k = 1 … 2,000,000 | up to 2,000,000 | 1 | never | never | always |
| bounded (`hospital`) | 880,160 | 62,380 of 88,016 matched | no | no | yes |
| whole corpus (`a`) | 2,000,000 | 703,362 of 973,757 matched | no | no | yes |

The **nonce** probe matches only the three plants, so nothing but the date
predicate can be excluding anything. The **bounded** probe sets k to ten times
its term's whole match count, so the LIMIT provably is not what excludes them.
The **whole corpus** probe is the literal "k large enough to return everything
else": 703,362 documents came back in one result set and not one of them was
dated after `as_of`. The newest was 2020-06-15 itself, which is the inclusive
boundary behaving correctly.

The run took 7m17s, 112s of it the whole-corpus probe. `--skip-corpus` drops
that probe for a quicker check; the full run is the one to make after any change
to `retrieve.SEARCH_SQL` or to the index build. Afterwards both row counts were
1,368,613, `quick_check` returned `ok`, and the ranking fingerprint was
identical. As a one-off check on the rollback itself, both databases were also
copied before the run and compared byte-for-byte after it: `cmp` reported no
difference in either. The script does not do that copy — 13.8 GB is too much to
spend on every run — but it is what establishes that the three cheap checks it
does do are measuring the right thing.

The synthetic versions of the same test in `tests/test_bible.py` are kept. They
run in milliseconds with no store on disk, which is what a contract test needs
to be; the red-test is what makes them worth believing.

## Open questions

- Passage-level retrieval. Documents are indexed whole; long documents are
  therefore ranked as a unit.
- Whether `wayback_capture` dates earn their place. They are sound as an upper
  bound but postdate their own claim 95.4% of the time, so they admit documents
  that can rarely be used for the claim that fetched them.
