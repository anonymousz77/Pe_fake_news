# docs/

Hand-written, git-tracked prose. Everything in this directory was typed by a
person and is meant to be read by a person.

This is **not** `data/bible/`. That directory is the evidence store — the
time-filtered corpus of dated documents claims are checked against. It is data,
it is gitignored, and it holds no writing. The two are easy to confuse by name
and must never be confused in practice.

| File | What it holds |
| --- | --- |
| [data_card.md](data_card.md) | What each dataset is, why it is in the register, what it may and may not be used for |
| [provenance.md](provenance.md) | Where each corpus came from, licence terms, and what was verified by hand |
| [counts_ledger.md](counts_ledger.md) | Row and file counts at each pipeline stage — the arithmetic that has to reconcile |
| [evidence_store.md](evidence_store.md) | Design and contents of `data/bible/` |

## Conventions

- Record **absolute dates**, never "last week" or "recently".
- When a number changes, update the ledger in the same commit that changes it.
- Mark anything unverified as unverified, explicitly. An honest gap is worth
  more than a plausible guess, and a guess that later reads as fact is the
  specific failure this project cannot afford.
