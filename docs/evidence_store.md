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

To be settled when the first documents are ingested (nothing exists as of
2026-09-01). Whatever is chosen, it must make the publication date visible
without opening the document — a date in the path or in a sidecar index — so
that the time filter is cheap and auditable.

## Open questions

- Deduplication policy across source corpora (same article, several origins).
- Whether to store full text or normalised passages.
- Index format, and where the index lives (`data/interim/` or alongside).
