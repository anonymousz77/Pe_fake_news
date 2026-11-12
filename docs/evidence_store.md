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

Since 2026-10-28 the code says exactly that: `published_at < :before`,
strictly. Until then it was `<=`, and the store holds a fact-check dated on its
own fact-check date for two M4FC claims: the inclusive bound returned each as
evidence for its own claim.

### Which date bounds a claim's evidence

The claim's own date, wherever its dataset carries one. The fact-check's
publication date is only a fallback: a fact-check appears after its claim, so
bounding by it admits evidence the claimant could not have seen (the person's
rule, 2026-10-28). A claim with neither date gets no evidence; retrieval
refuses rather than guess one. The rule per dataset is `configs/as_of.yaml`,
read by `retrieve.evidence_bound`:

| dataset | bound | field |
| --- | --- | --- |
| averitec | the claim's date | `claim_date`; 105 train rows have none and get no evidence |
| averimatec | the claim's date | `date` |
| fakeddit | the post's own time, UTC | `created_utc` |
| isot | the article's own date | `date`; 10 of 44,898 values are URLs or text and refuse |
| m4fc | the fact-check's date, as M4FC's authors use it | `fc_pub_date` |
| dgm4, factify2, fakenewsnet, liar, mocheg, verite, welfake | none | retrieval refused |

M4FC carries no date of the claim's post. Its other date fields, `claimed_date`
and `date`, are context items: the claimed and the actual date of the photo.

## Relationship to the register

The evidence store is *derived*: it is built from corpora declared in
`data/sources.yaml`, but it is not itself a register entry and gets no
`raw_dir()`. `forbidden_patterns` therefore constrain it transitively — a
document that could not be ingested into `data/raw/` cannot reach the store.

The averitec knowledge store is the case that matters most. Its `test/` shard
is forbidden, which means test-claim retrieval evidence never enters
`data/raw/averitec/`, and therefore never enters the evidence store, and
therefore cannot be retrieved for any claim.

## Fact-checker pages

A fact-checker's page is not evidence: it states the verdict the model is
asked for. The store holds such pages. On 2026-10-28 it held the claim's own
fact-check for 3 M4FC claims, 53 AVeriTeC train and dev claims and 4
AVerImaTeC claims, and 10 of them (3 M4FC, 7 AVeriTeC) were reachable as
evidence for their own claim. So retrieval never returns a page from a listed
fact-checker, by a predicate inside `retrieve.SEARCH_SQL`, beside the date.

**The list** is `configs/factcheckers.yaml`, built by `scripts/factcheckers.py
build` from the fact-checkers behind our datasets: M4FC's 22 organisations, the
fact-check URLs of AVeriTeC, AVerImaTeC, MOCHEG and VERITE, Factify2's
fact-checker image hosts, and PolitiFact and GossipCop by construction for LIAR
and FakeNewsNet. It reads **every split, test included**, by the person's
decision of 2026-10-28: a domain list carries no label, and a fact-checker we
miss leaks answers into test evidence, which is worse than tailoring the list
to test. It is a stated exception to "test data does not enter the pipeline".
3 of its 107 hosts come only from test files, and 5 were added by hand (below).

**Scope, per host** (the 2026-10-29 build):

| scope | hosts | what is blocked |
| --- | ---: | --- |
| domain | 79 | the host and its subdomains: dedicated fact-checkers, hosts named for checking, hosts with 50 store pages or fewer, and the 5 added by hand |
| path | 19 | only the fact-check desk of a news publisher (indiatoday.in `/fact-check`, usatoday.com `/story/news/factcheck`, the TEGNA stations' `/article/news/verify`) |
| docs | 9 | only the store pages that are the publisher's cited fact-checks, by doc id (reuters.com, bbc.co.uk, washingtonpost.com, …): their fact-checks share no section |

The host rules block 439 store pages: 398 by host, 40 by section, 1 by doc id.
Blocking every cited host whole would have blocked 17,115, most of them news
(reuters.com alone has 9,384 pages in the store). The nine `docs` hosts stay
docs-scoped, by the person's decision of 2026-10-28: only their cited
fact-checks are blocked, not the hosts whole. The list records the decision
where it had the `review` flag, and a docs-scoped host the builder finds later
is flagged again.

**Path words, on every host** (the person's decision, 2026-10-28). A page whose
path contains any of 19 fact-check desk words (`fact-check`, `factcheck`,
`hoax-alert`, `verificador`, `webqoof`, …: `path_words` in the list) is
blocked, whatever its host, listed or not. The words are matched as substrings
of the lower-cased path; a Wayback capture's path is the archived URL's. The
host name, the query string and the fragment are not read. This blocks 1,038
store pages on 588 hosts that the host rules miss. Among them are fact-checks
our datasets never cite:

- Reuters `/article/...factcheck...` pages (3 of its 12 blocked pages);
- CNN `fact-check` and Reality Check pieces;
- the VERIFY segments of TEGNA stations (wusa9.com);
- Western Journal's `fact-check-` pages;
- @PIBFactCheck tweets.

Others are news, the price the person accepted:

- stories about fake news (284 of the pages carry `fake-news`, among them
  Reuters' other 9);
- aktiencheck.de stock notes titled `faktencheck`;
- Wikinews articles about hoaxes;
- `verification` stories, which `verifica` takes.

How the 1,038 split between the two was not counted. The list counts what each
word takes (`store_blocked_by_word`). The rules block 1,477 store pages in
all.

**Added by hand** (the person's decision, 2026-10-29). `python
scripts/factcheckers.py residual` reads host names only. On 2026-10-28 it found
13 store pages on 6 unlisted hosts whose name carries a path word, and 7,329 on
31 more whose name carries a weaker checker word such as `fact`, `check` or
`truth`, 7,114 of them from aktiencheck.de. Read by an agent, 10 of these pages,
on 5 hosts, were fact-checks. The person added the 5 hosts by hand, each blocked
whole (`ADDED_BY_HAND`):

- factcheck.aap.com.au (1), AAP's desk on a subdomain of a docs-scoped host;
- smhoaxslayer.com (2) and turnbackhoax.id (2);
- hoax.sk (4), Slovak scam and hoax warnings;
- myanmar.factcrescendo.com (1).

They hold exactly those 10 store pages, and no path word had blocked any of
them. No dataset of ours cites them, so none counts as coming only from test.

**What nothing blocks** (the same command, 2026-10-29): 4 store pages on 2
unlisted hosts whose name carries a path word, and 7,328 on 30 whose name
carries a checker word. mediabiasfactcheck.com (1) rates sources rather than
claims; doubleverify.com (3) and factba.se (2) are not fact-checkers; the rest
are almost all news (aktiencheck.de 7,114, redlandsdailyfacts.com 47,
truthout.org 37).

**How it is applied.** `retrieve.connect()` loads the list into TEMP tables and
defines the TEMP views the statement reads URLs through, so a connection opened
any other way fails with "no such table" rather than returning a fact-check. A
Wayback capture is judged by the URL it archived: the store files 3,056
captures under `web.archive.org`, among them africacheck.org and politifact.com
pages. The SQL that computes a page's host and path has a Python twin
(`factcheckers.page`); the two agree on all 1,368,613 store rows. Nine captures
of captures are judged by the inner capture. Every rule has a twin too
(`factcheckers.blocker`), and the red-test checks it against the SQL on every
store row, rule by rule. `retrieve.load_blocklist` refuses a list that blocks
no host or no path word, and any path word shorter than 4 characters or not
lower-case ASCII.

**Measured on the store, own fact-checks reachable for their own claim:**

| | current (`<=`, no list) | strict only | list only | both |
| --- | ---: | ---: | ---: | ---: |
| M4FC (3 in store) | 3 | 1 | 0 | 0 |
| AVeriTeC (48 with a claim date) | 7 | 2 | 0 | 0 |
| AVerImaTeC (4) | 0 | 0 | 0 | 0 |

The strict bound alone misses the pages dated before their bound, such as La
República's fact-check stored a day before the date M4FC gives it. The list
catches them. (Measured before the path words, which can only block more.)

## Speed

`retrieve.search` averaged 93.5 s per query over the M4FC evidence run's 200
claims. `scripts/retrieval_bench.py` profiled three of them, at the 10th, 50th
and 90th percentile of that run's search time:

| run (s) | terms | matches | content fetch | whole query | `ORDER BY rank` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 83.1 | 4 | 854,081 | 116.4 s | 123.5 s | 3.1 s |
| 94.1 | 12 | 1,077,624 | 127.1 s | 143.3 s | 7.3 s |
| 101.0 | 21 | 1,031,911 | 131.2 s | 138.6 s | 16.3 s |

A claim's words are ORed, so almost the whole corpus matches. Ordered by
`bm25(docs_fts)`, SQLite read every matching document's content row, title and
up to 20,000 characters of text, to get the join key: that fetch was 87–95% of
the query. Ordered by FTS5's own `rank`, FTS5 sorts, rows stream out best first,
and the join stops at k (`EXPLAIN` shows `INDEX 32` and no temp b-tree). It is
the same bm25 under another name, so the results are unchanged. On all 200
queries the new statement returned exactly the evidence run's hits: the same
domains and dates in the same order, and the same top-1 score. On the three
profiled queries the doc ids and scores matched as well. The index is
unchanged.

**Separate passes over the 200 queries, k = 20** (2026-10-28):

| option | median s | p90 s | overlap@20 with E1 | top-1 same |
| --- | ---: | ---: | ---: | ---: |
| E1, `ORDER BY rank` | 2.59 | 5.25 | 1.000 | 200 |
| E1 + 256 MiB cache, 16 GiB mmap | 3.07 | 7.55 | 1.000 | 200 |
| L, the leak rules of 2026-10-28 (host rules only) | 4.14 | 9.96 | 0.997 | 200 |

All three ran in the person's terminal on one evening, in the order E1, L, E1+.
The larger cache did not help. Passes run hours apart are not strictly
comparable: the OS file cache decides much of a query's time, and E1's median
was 2.58 s there against 9.6 s in a three-query gauge in another session. No
exact option reached the 2 s target, so the person put pruning in play.

**Pruning changes which documents come back.** The options are three ways of
choosing a claim's terms:

- P2: only the terms in at most 1% of documents;
- P3: the claim's named entities and its 5 rarest other terms;
- P4: its 8 rarest terms;
- P4b: P4 less the terms in half the store or more, which FTS5 already scores
  at idf 1e-6 (the person's request, 2026-10-29, after P4's pass).

P4 keeps such a term when a claim has fewer than 8 distinct rarer words: 44 of
the 200 queries. Those 44 took a median 3.85 s under P4, against 0.71 s for the
other 156, and 16 of P4's 20 slowest queries are among them. P4b drops 1 term
from 30 of them and 2 from 14, and leaves every query at least one term; on the
other 156 it is P4.

**The production candidates, timed together** (2026-10-29,
`retrieval_bench.py session`, in the person's terminal). One process ran L and
the four L+P options on each of the 200 queries in a Williams-design order:
each option ran in each position for 40 queries, and directly after each other
option 40 times. Every option is measured against the session's own L. "≥1"
and "≥2" count the queries whose top 5 contains at least one or two of the
claim's named entities; India is the 48 queries from Indian fact-checkers.

| option | median s | mean s | p90 s | overlap@20 | overlap@5 | top-1 same | ≥1 | ≥2 | India ≥1 | India ≥2 | < 20 hits | fell back |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L | 6.11 | 6.85 | 11.98 | 1.000 | 1.000 | 200 | 160 | 95 | 44 | 27 | 0 | 0 |
| L+P2 | 0.08 | 0.09 | 0.16 | 0.338 | 0.334 | 62 | 150 | 79 | 42 | 21 | 1 | 5 |
| L+P3 | 1.09 | 1.74 | 4.15 | 0.786 | 0.779 | 154 | 165 | 92 | 43 | 27 | 0 | 0 |
| L+P4 | 0.85 | 1.51 | 3.95 | 0.798 | 0.782 | 159 | 163 | 94 | 43 | 28 | 0 | 0 |
| L+P4b | 0.68 | 1.02 | 2.59 | 0.798 | 0.782 | 159 | 163 | 94 | 43 | 28 | 0 | 0 |

Entity matches against L, as queries gained / lost:

| option | ≥1 | ≥2 | India ≥1 | India ≥2 |
| --- | ---: | ---: | ---: | ---: |
| L+P2 | +8 / −18 | +4 / −20 | +1 / −3 | +1 / −7 |
| L+P3 | +6 / −1 | +3 / −6 | +0 / −1 | +0 / −0 |
| L+P4 | +5 / −2 | +4 / −5 | +0 / −1 | +1 / −0 |
| L+P4b | +5 / −2 | +4 / −5 | +0 / −1 | +1 / −0 |

- L+P4b returns L+P4's top 20 on all 200 queries, in the same order on 199,
  and moves no score by more than 4.27e-6.
- L's 6.11 s is not its 4.14 s of 2026-10-28. That was a pass of its own on
  another evening; here five options share one process and its cache.
- L+P3 also needs the claim's entities: NER costs 24 ms a claim, after an
  11.7 s model load.
- Since L's own pass, the rules added (5 hosts, 19 path words) changed 5 of its
  200 queries: 6 hits left, each a page today's rules block.

### Decision, 2026-10-29: L+P4b is the default

The rule was set before the session: pick the option closest to L with a
median under 2 s, overlap@5 at least 0.75 against L, no net loss in entity
matches, and at most 10 queries with fewer than 20 hits. That option is L+P4b:
it ties L+P4 on every result measure and is faster. It met three bars (median
0.68 s, overlap@5 0.782, no query under 20 hits) and failed the fourth as
written. Against L it nets +3 queries on ≥1 and +1 on India ≥2, but −1 on ≥2
(94 against 95) and −1 on India ≥1 (43 against 44). No option met the rule:
L+P3 nets −3 on ≥2 and −1 on India ≥1, and L+P2 fails overlap and all four
entity measures.

**The person ruled the "no net loss" bar too strict and adopted L+P4b**
(2026-10-29): −1 on 200 queries (+4 / −5 on ≥2) is noise. The exception is to
that bar alone; the other three stand as set.

Of L+P4b's 6 entity losses against L:
- in 3, the cut to 8 dropped part of the lost entity's name: "Alan" and
  "París" in a Spanish claim, "Public Service" of "Public Service Commission"
  (the India loss), and "D.C." of "Washington, D.C.";
- in the other 3, the entity's words were kept and other pages ranked higher.

**As wired** (`retrieve.query_terms`):
- **The default** searches a claim by its 8 rarest distinct words, none of them
  in half the store or more. When every word is, it falls back to the one
  rarest; none of the 200 claims does.
- **The counts:** each distinct word's document frequency is counted per
  query, as bm25 counts it. The store's size is the temporal table's row
  count, which is bm25's own N (1,368,613).
- **Exact mode:** `exact=True` (`--exact`) searches every word, as L did.
- **Both modes** run the one statement, so both leak rules hold in either. The
  red-test's probes are one word each, and a one-word query searches the same
  way in both.

**Timed with the counts** (`retrieval_bench.py default`, 2026-10-29). The agent
ran it in its own session, not in the person's terminal: one process,
`retrieve.search`'s default over the 200 queries.

| | median s | mean s | p90 s | max s |
| --- | ---: | ---: | ---: | ---: |
| whole query | 1.33 | 1.62 | 3.12 | 5.44 |
| counting the words | 0.37 | 0.40 | 0.69 | 1.26 |
| the statement | 0.91 | 1.22 | 2.75 | 4.46 |

- **The 2 s target:** the median is under it.
- **The same results:** on all 200 queries the default searched by the
  session's L+P4b terms and returned its top 20, with the same ids, order and
  scores.
- **Counting** is about a quarter of a query.
- **The statement** alone took a median 0.91 s, against L+P4b's 0.68 s in the
  session. This was another process at another time, and what the OS has
  cached decides much of a query's time.

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

Neither retrieval rule is taken on trust. `scripts/redtest.py` plants documents
into the **real** index and demands that the right ones come back and the rest
never do. It writes `data/reports/redtest.json`.

It runs against the store as built, not a fixture, because a filter that holds
on four rows says nothing about one that has to hold on 1.4 million: SQLite
picks different plans at different table sizes, and a `k` that exceeds a toy
corpus is dwarfed by a real one. The plants go in inside a `BEGIN IMMEDIATE`
transaction that is rolled back in a `finally`, so the index is restored whether
the test passes or fails, and nothing is ever committed. Restoration is checked
four ways: both row counts, a `quick_check`, a ranking fingerprint of a fixed
query taken before planting and compared after, and the sha256 of both index
files before and after. The fingerprint catches a rollback that restored the
count but disturbed the FTS segments; the hashes catch anything else.

**The plants** (since 2026-10-28; every one carries a nonce found nowhere else):

| plant | must |
| --- | --- |
| dated the day before `before` | come back |
| dated on `before`, the day after, or absent from the temporal table | never come back |
| a blocked host, its subdomain, the host in upper case, a Wayback capture of it | never come back |
| a publisher's fact-check desk, in upper case, and as a Wayback capture | never come back |
| a document the list names by id | never come back |
| a path word on an unlisted host: as a segment, in upper case, inside a slug, in a Wayback capture | never come back |
| the same publisher outside its desk; `not<host>`; `<host>.redtest.invalid`; the unlisted host with no path word | come back |

The last row keeps the list honest the other way: a rule that blocked those
would be matching substrings of hosts, or blocking a host for one page's path.

Every fact-check desk names a path word, so with the words on, the desk plants
are blocked twice and prove nothing about the desk rule. The nonce probe
therefore runs a second time with the words deleted, inside the same
transaction. The desk plants must still stay out (the desk rule alone), and
the word plants must come back (nothing but the words blocked them). A
**parity** probe then checks each rule's SQL verdict against its Python twin on
every store row, because the whole-corpus probe judges real rows by that twin.
After the rollback, the connection's TEMP rules are counted again too.

**Measured with the path words and the hosts added by hand** (the person's
terminal; report stamped 2026-10-28 21:47 UTC), against the 1,368,613-document
store, `before` 2020-06-15, with the list's 98 host rows, 1 document id and 19
path words. It passed:

| probe | k | returned | leaked | wrongly excluded |
| --- | ---: | ---: | ---: | ---: |
| nonce, k = 1 … 2,000,000 | up to 2,000,000 | the 5 that must come back (1 at k = 1) | 0 at every k | 0 from k = 10 |
| nonce, path words deleted | up to 2,000,000 | those 5 and the 4 word plants (1 at k = 1) | 0 at every k | 0 from k = 10 |
| bounded (`hospital`) | 880,330 | 62,287 of 88,033 matches | 0 | 0 |
| whole corpus (`a`) | 2,000,000 | 702,481 of 973,774 matches | 0 | 0 |

The **nonce** probe matches only the plants, so nothing but the rules can be
excluding anything. With the words deleted the desk plants still stayed out,
which is the desk rule on its own, and the word plants came back. The
**bounded** probe sets k to ten times its term's whole match count, so the
LIMIT provably is not what excludes them. The **whole corpus** probe is the
literal "k large enough to return everything else", and every one of its
702,481 rows was also checked against the list in Python
(`scripts/factcheckers.py`, the SQL's twin). Not one came from a blocked
source, and the newest was 2020-06-14, the day before the bound. The
**parity** probe compared every rule's SQL verdict with its twin's on all
1,368,632 rows (the store and its 19 plants): 0 differ. The SQL blocked 445 by
source, 2 by document id and 1,168 by path word; a page can be blocked by more
than one rule.

The whole-corpus probe took 287 s and the parity probe 295 s: both rules are
evaluated on every row they read (at k = 20 they cost milliseconds).
`--skip-corpus` drops the whole-corpus probe for a quicker check. The full run
is the one to make after any change to `retrieve.SEARCH_SQL`, the blocklist or
the index build. Afterwards both row counts were 1,368,613, `quick_check`
returned `ok`, the ranking fingerprint was identical, and both files hashed to
the sha256 taken before the plants, which is the one taken on 2026-10-28
before any of the retrieval work (`bm25.sqlite` 8b3ef94f…, `temporal.sqlite`
0f8b85c3…).

The first run with the list (2026-10-28, its 93 host rows, no path words)
passed the same way in 22m23s, 612 s of it the whole-corpus probe.

The first red-test (2026-09-11, the inclusive bound and no list) planted three
documents: a day earlier, a day later and undated. It returned 703,362 of
973,757 matches for `a` in 7m17s, and a one-off byte-for-byte `cmp` of both
databases against copies taken before it established that the rollback leaves
them unchanged; the sha256 check now makes that comparison on every run.

The synthetic versions of the same test in `tests/test_bible.py` are kept. They
run in milliseconds with no store on disk, which is what a contract test needs
to be; the red-test is what makes them worth believing.

## Open questions

- Passage-level retrieval. Documents are indexed whole; long documents are
  therefore ranked as a unit.
- Whether `wayback_capture` dates earn their place. They are sound as an upper
  bound but postdate their own claim 95.4% of the time, so they admit documents
  that can rarely be used for the claim that fetched them.
- Whether the filter inside the query changes results relative to
  post-filtered live retrieval, which is how prior datasets do it
  ([related_work.md](related_work.md)). To be measured, not assumed.
