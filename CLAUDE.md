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

## Image-to-text: standing rules

`scripts/image_text.py` writes one row per distinct image: OCR, an
appearance-only description, and five entity lines (PERSON / ORG / GPE /
EVENT / DATE).

- **The entity answer is held to its five lines while it is generated.**
  Unconstrained, the model ended 63 of the first pilot's 199 answers right
  after the PERSON line. Where it would break the form, the skeleton puts in
  the newline or the header, records the replacement, and marks every line
  from there on `forced`, on the category and on each of its items. An answer
  that loses lines anyway stops the run; it is never written.
- **A runaway or an empty answer is an error, never `ok`.** An entity answer
  cut at `max_new_tokens` (`truncated`; the first pilot had 2) or with every
  line forced (`no_free_line`: the model began outside the form, like the
  first pilot's 13 bare "NONE" answers) is written with that as the entity
  error: the row is `partial`, no entity leaves it, and neither is in the
  probe's headline. A description cut at `max_new_tokens` is an error too,
  `truncated`, with the flag kept: half a description is not a description.
  `row_problems` refuses every one of these as `ok`.
- **A forced line is weaker evidence than a free one.** Entity statistics are
  reported split by forced and free, and **forced-line entities never become
  supervision without that distinction visible**:
  `image_text.entities_of(row, forced=...)` is the only sanctioned way to read
  entities out of a row, and it has no default. Splits and evaluation-only
  corpora are enforced by the training loader, as for every other supervision.
- **A VLM name is a candidate, never a verification.** On the first pilot's
  recognition probe most names produced from appearance alone were not the
  figure the claim names, and confidence did not separate them. Identity is
  decided by an external gallery. The pilot report
  (`data/reports/image_text_pilot.json`, `recognition_probe.thresholds`) names
  both bars: the precision a name needs to replace a gallery (0.90), and what
  it needs as a candidate the gallery verifies -- which depends on how often
  the gallery accepts a wrong name (FMR / TAR), measured on the VLM's own
  wrong names.

## Entity gallery: standing rules

`scripts/gallery.py` verifies identity against a gallery of known entities:
Wikidata items with reference images from Wikimedia Commons. It is how the
"a VLM name is a candidate" rule above is honoured.

- **1:1 only.** `Gallery.verify` answers whether an image shows the entity a
  CLAIMED name resolves to, and returns the reference it matched (Commons page,
  licence, attribution, face box) so a person can check it. It never searches
  the gallery for an unnamed face, and it refuses a name several people hold
  (`ambiguous_name`, unscored): the best of several candidates is a search
  among them.
- **Built from the names our corpora use**, never from Wikidata at large
  (`count` reports them first). The gallery is not a register dataset: it lives
  in `data/gallery/` (ignored), outside `sources.yaml`, as the bible does.
- **Scope: VERITE, AVerImaTeC and Factify2** (`BUILD_SCOPE`, the November
  path). DGM4 is left out: it is a synthetic corpus not chosen for training,
  and its captions alone name 38,165 people. Scope is per corpus on both sides
  (VLM names and claim names). Widening it (`--scope`) adds and never rebuilds:
  `link --all` and `fetch --all` skip what is done and checkpoint every chunk,
  and `embed --pin-consistency` pins the consistency threshold, so existing
  entries do not change. Re-run `calibrate` after widening: more entities can
  give a name more candidates.
- **Clean licences, kept that way.** Every reference's licence and attribution
  are recorded exactly as Commons states them. Media, faces, embeddings and the
  per-entity tables never enter git; only aggregates
  (`data/reports/gallery_*.json`, no name, no record id) are published.
- **Fetching follows the data layer's discipline, at Wikimedia's published
  pace.** One worker, one request in flight (`gallery.WikimediaWorker`):
  - On `www.wikidata.org`, `commons.wikimedia.org`, `upload.wikimedia.org` and
    `thumb.wikimedia.org`, at most one request starts per second.
  - After an API answer that took over a second, the next waits 5 s.
  - A 429's Retry-After, a maxlag refusal, and a 502, 503 or 504 (15 min) hold
    the whole host. Every other host keeps 5 s.
  - A lookup or download that gets no answer, or a "not now", after its
    retries leaves the entity for the next run. It is never recorded as a
    missing file or as "no type match".

  The quotes behind this, and the policy points the gallery does not meet (no
  on-wiki authentication, the robots.txt `/w/` line, no dumps), are in
  `docs/licences.md`. `recover.Worker` itself is unchanged for recovery. API
  calls send `maxlag` and ask for gzip; media requests do not. The User-Agent
  names the anonymised repository URL; `PE_FAKE_NEWS_UA_CONTACT` overrides it
  from the environment, never from tracked code. Thumbnails come at a standard
  width (960). Every reference passes the three gates, then two more: the bytes
  are the rendition the API described, and they are not an image of any corpus.
  Rejected images that decode are quarantined in `data/gallery/rejected/`.
- **A name links only to an EXACT label or alias of the right type**: P31 Q5 for
  a person, an instance of a subclass of Q43229 for an organisation. A name
  whose candidates don't dominate each other is kept, linked to the most-linked
  candidate, and marked `ambiguous`.
  - **Initials are also searched dotted and spaced** (`dotted_initials`,
    2026-10-23). Wikidata's prefix search finds "P. Chidambaram" only in that
    form, so a person's name with initials is searched as written and dotted,
    and the answers are pooled. This covers "P Chidambaram", "SS Ahluwalia"
    and "K.S. Eshwarappa". An exact match is still `same_name`: the same
    letters, never a respelling. A name that linked nothing and was never
    searched dotted is searched again (`link_due`).
  - **Exact initials carry a namesake risk.** Of the 309 names searched
    again, 136 linked (566 mentions). Some matched someone else entirely:
    "K Sasikala" matched only a botanist, "MK Sinha" a mycologist, and "R
    Shankar" an American physicist.
  - **Held links (`HELD`).** 17 of those links are held by the person's
    decision. By an agent's reading of the items' descriptions, 7 are
    namesakes and 10 are unclear.
    - The link row keeps the match, the reason and who held it, so a later
      pass revisits it rather than rediscovering it.
    - Nothing resolves the name to that person, not even their own label.
    - The linker never searches the name again. `link --release` restores
      the link once a person has checked it.
  - Of the 119 links kept, 8 (17 mentions) point to someone whose
    description reads academic, and none to an athlete or a historical
    figure. This is a count, not a review. It was small enough to fetch, and
    the same rule flags 10 of the 17 held.
  - **Near-matches are reported, never linked** (decided 2026-10-23). A
    wrong link puts the wrong face in the gallery, where nothing downstream
    can see it, and is worse than none. Linking the alias gaps would need a
    sheet a person reviews, as faces have, and at 139 names (499 mentions)
    that is not worth it now.
- **How a name resolves** (`Gallery.resolve_key`; decided 2026-10-23). Two
  rules beyond an exact key; `resolution_rules` counts what each changes,
  every `calibrate` run.
  - **The cost, by design: 407 names that resolved to one person, 10.3% of
    mentions, now refuse rather than guess** (`cost_by_design`). The largest
    is "Trump" (1,181 mentions), ambiguous among ten Trumps.
    - Defend it at review as a choice; do not hide it as a regression.
    - A guess checks one person's face, and nothing downstream can tell a
      wrong guess from a right one: "Modi" resolved to Amedeo Modigliani for
      653 mentions.
  - **Spacing:** a name is looked up by its letters, as linking's `same_name`
    matches it, so "MK Stalin" is "M. K. Stalin". 14 more names resolve (49
    mentions; 20 and 155 before the initials re-link linked most of them
    directly). 7 that resolved to one person are ambiguous instead (19
    mentions), because two Wikidata items differ only in spacing:
    "Chandrashekhar Singh" (an Indian politician) and "Chandra Shekhar Singh"
    (the prime minister's alias).
  - **One word:** a one-word person name resolves only to a gallery person
    who holds it, when that word is also their label, alias or linked name,
    and no one else is a reading of it.
    - Holding it means: it is their one-word label, a word of one of their
      longer names, or a longer name of theirs run together.
    - Two readings or more make it ambiguous, never the likeliest.
    - A nickname alias never resolves a word, but its owner is still a
      reading: "Modi" is ambiguous among six Modis and Amedeo Modigliani
      ("Modì"), and "Bapu" between Gandhi and Asaram Bapu.
    - "Trump" is ambiguous among ten Trumps. "Mueller", which no item lists as
      an alias, resolves to no one: a surname alone is never a guess.
  - Of those 407 names (3,377 mentions), judged by the corpora's full names
    (the 90% rule below):
    - 50 had resolved to someone else (850 mentions, "Modi" 653);
    - 7 had resolved to the person meant (76 mentions);
    - 343 cannot be told either way (2,432 mentions; "Trump" alone is 1,181);
    - 7 are the spacing collisions.
- **Names collapse through Wikidata's labels and aliases, and the gaps are
  counted** (`alias_gaps`, every `calibrate` run).
  - Of the build scope's 6,734 person names, 601 collapse into the 270 people
    who have two names or more. "Priyanka Gandhi", "Priyanka Vadra" and
    "Priyanka Gandhi Vadra" are one QID.
  - 230 names (5.4% of mentions) plausibly name a gallery person they do not
    resolve to: 139 resolve to no one, 91 to several, and none to someone
    else. "Plausibly" is by explicit rules that count and never link:
    `name_variant` for spacing, initials, a name left out or one letter; for
    one word, the person holding 90% of the corpora's full names with that
    surname.
  - No sample label turns on a gap: the probe's patterns name the figures the
    gaps touch most. The handles NER misses are the review sheet's part 3.
- **Names that resolve to no one are counted by why, never merged**
  (`no_one_names`). There are 2,713 of them, 17.1% of mentions (2,826 and
  18.5% before the initials re-link).
  - **Held for a person's check:** 17 (25 mentions).
  - **A linking failure, the person in the gallery** (an alias gap): 139 (499
    mentions). 38 of them are one word, which the one-word rule leaves
    unresolved by design.
  - **One-word names the resolver refuses:** 36 (89 mentions).
  - **What Wikidata's search answered at linking:**
    - no item by that form: 1,537 (2,652 mentions);
    - items beginning with it, none of them it: 264 (503);
    - exactly a surname or first-name item: 76 (148);
    - exactly something that is no person, such as a party, a place or a
      deity: 435 (1,200);
    - both: 209 (484).
  - **"No item" is not "no entry".** A seeded sample of 100 of the multi-word
    ones was checked on 2026-10-23, with Wikidata's full-text search
    restricted to humans:

    | | named twice or more | named once |
    | --- | ---: | ---: |
    | a person entry, by the counting rules | 8 of 50 | 2 of 50 |
    | read by an agent: notable, in a form the linker missed | 20 of 50 | 13 of 50 |
    | read by an agent: people with no entry | 17 of 50 | 22 of 50 |
    | read by an agent: not people | 6 of 50 | 10 of 50 |

    The agent's reading is not a person's decision. The forms the linker
    missed were initials without their dots ("P Chidambaram"), initials run
    together ("SS Ahluwalia"), misspellings ("Shahi Tharoor") and part of a
    name ("Lal Khattar"). The sample predates the initials re-link, which has
    since searched the first two forms again (see linking above). The last
    two are not searched again: matching them is fuzzy, a different risk.
- **References must agree.** A person's reference faces are checked against
  their medoid. The threshold is the 0.999 quantile of similarities between
  different entities' medoids, taken from gallery references only. When two
  references disagree, Wikidata's own image (P18) is kept.
- **Calibration is the deliverable, and it reads Factify2 TRAIN only**: only
  images every one of whose records is train. VERITE is never read for any
  threshold.
  - Trials are the VLM's own names produced from appearance alone (no word of
    the name in the OCR text).
  - A trial is right when its name resolves, by QID, to **anyone the image's
    claims name**: the probe's listed figures and claim NER's people. A claim
    name that could be several people decides nothing. Otherwise it is wrong.
  - The figure (the probe's list first, otherwise the claim's single NER
    person) is the image's stratum for per-figure reporting, not the reference.
  - Until 2026-10-22 a name was scored against that one figure alone, and
    every other person the claims named counted as wrong. That was a defect in
    our measurement code, and it understated both earlier figures: the pilot's
    47.8% and the sample's 0.5455. Corrected on the sample, the figure is
    0.7045; the report keeps the defective one beside it
    (`single_figure_rule`).
  - The threshold is the lowest one at which the one-sided 95% lower bound of
    accepted-name precision reaches 0.90. A point-estimate threshold is
    reported beside it, never presented as the bound, and **never adopted**.
  - **Only a calibration over the complete Factify2 run sets the threshold**
    (`factify2_ledger.complete`: the server's, once the run finishes), and
    only by the bound. `threshold.certified` says whether it is set.
    - On a partial ledger, `calibrate` reports the candidate and every figure
      at it, and leaves `threshold.chosen` null.
    - **Short of the bound, the threshold stays null on the complete run too.**
      The report then gives three things: the curve; the best candidate (the
      threshold with the highest lower bound); and why it falls short
      (`bound_shortfall`). There are two possible reasons:
      - **Too few trials:** its precision clears 0.90 as a point estimate.
        `accepted_names_needed` says how many accepted names would certify
        it.
      - **Overlap:** it is not above 0.90 even as a point estimate. Wrong
        names score among the right ones, and more trials cannot fix that.

      A fallback to a point estimate would hide which. A gallery that says it
      cannot decide at this confidence is a usable result; one that accepts at
      an uncertified threshold is not.
    - On the sample's corrected labels before review, nothing reaches the
      bound. The closest is 0.5531, where 23 of 24 accepted names are right
      (lower bound 0.817). The reason is too few trials: at that precision,
      about 103 accepted names would certify it.
    - The sample's adjudicated candidate, 0.4383, was not adopted, for three
      reasons. Its bound clears 0.90 with no margin (31 of 31 accepted names
      right). The bound rule finds no threshold in any cross-validation fold.
      And it rests on 44 trials from 6.3% of the ledger. With every sheet row
      decided (37, 2026-10-22) it stays at 0.4383, and TAR at it falls from
      0.9394 to 0.8857: the second sheet's two relabels are faces the gallery
      fails to recognise.
  - **`verify` decides only at a certified threshold** (`certified_threshold`).
    - Without one it refuses: `match` stays null, and it returns the score and
      the matched reference for a person to check. It never defaults to
      another threshold.
    - A report of another face configuration, a candidate, or an older report
      without the `certified` flag counts as no threshold.
    - `--threshold` adds a labelled comparison, never a decision.
  - The cost in true accepts is always reported. Random-pair FMR is a labelled
    contrast, never an input.
  - **Why: the VLM's wrong names are not random strangers.** There are two
    measurements on the sample (44 trials, scored against the full gallery).
    - **As first measured:** the right names' faces reach at most 0.281
      against random strangers, yet 12 of the 18 scorable wrong VLM names
      scored above every stranger. A threshold set on random pairs (0.263)
      would accept 14 of 20 wrong names: precision 0.63.
    - **Adjudicated** (all 37 decisions): 1 of the 8 scorable wrong names
      scores above every stranger (at most 0.346). A random-pair threshold
      (0.181) would accept 6 of 9 wrong names: precision 0.85. On the first
      sheet's 23 decisions it was 4 of 10, and 8 of 11 at 0.194: precision 0.80.
      The strangers are drawn per right name from the whole gallery, so the
      maximum moves as the gallery grows and the floor drops faces (0.291 and
      0.179 before the 97 people of 2026-10-23 and the floor).
    - **Why they differ:** most of the first "wrong" names were right, because
      the labels scored a name against one figure per image. Of the 8
      highest-scoring wrong names, 7 were the person the VLM named. Five of
      them are named in the claim itself, which the corrected rule now finds.
      A person found the other two on the review sheet.
    - The second measurement uses labels against everyone the claims name,
      with the review applied. Corrected labels without the review give 3 of
      12 above every stranger, and precision 0.74. The strangers are drawn
      per right name, so their maximum moves with the labels.
    - The point survives: random-pair calibration still fails 0.90, so the
      threshold is calibrated on the VLM's own names. `calibrate` measures
      this every run (`look_alike_finding`), raw and adjudicated.
  - **Coverage ceiling, stated beside every calibration.** 1,166 of the
    gallery's 3,556 people have no usable reference face; 1,049 of them have
    no Commons reference at all. Verify can only answer "no usable reference"
    for them, whatever the threshold. `calibrate` reports it (`coverage`), and
    so does `embed`. A right name the gallery cannot score counts against TAR
    and is never a true accept. That covers no usable reference, a name
    several people hold, and a face under the size floor.
    `cost_in_true_accepts.tar_scorable` is TAR without them, the recognizer's
    own rate: 0.9394 on the adjudicated sample.
  - **A face under 20 px from eyes to mouth is never scored** (`MIN_EYE_MOUTH`,
    decided 2026-10-23).
    - The floor is measured: it is the smallest face the gallery recognised
      among the names a person confirmed.
    - When the face that best matches is under it, verify answers
      `no_usable_face` with no score. Calibration counts such a trial as
      coverage, and the look-alike contrast and the review sheet do not score
      it either.
    - A score from an 18 px face is noise presented as a measurement.
  - **Recognition failures are reported apart from label noise**
    (`review_findings`). A name a person confirmed is one of three things:
    - recognised: at or above the gallery's 0.381 floor;
    - a recognition failure: below the floor, where different people's faces
      score;
    - coverage: it cannot be scored.

    On the sample, 31, 2 and 2 of the 35 confirmed names. The failures are the
    recognizer's limit on our images, not label errors, and they lower TAR.
    Their two causes are counted apart (`face_geometry`), because the remedies
    differ:
    - **Pose:** side-on, meaning the nose is at least half an eye span from
      the eyes' midpoint, past the far eye. References in more poses can help:
      the gallery recognised two side-on faces, each over three times larger.
      On the sample: 2, both Modi profiles (eyes 5-6 px apart, 22 px from eyes
      to mouth).
      - **Commons has the side views** (`side-views`, 2026-10-23; a
        measurement that stored nothing). Of the 50 most-mentioned people, 38
        have one the gallery would keep: 21 among their references, the rest
        on Commons, side-on and agreeing with their medoid. That is enough to
        fetch for by the half rule (`SIDE_VIEW_ENOUGH`), though fetching is
        the person's decision and has not been done.
      - **Both sample failures would reach the floor.** Modi has none of his
        5 references side-on, while Commons has 5 agreeing side views of him.
        Against them, both sample failures reach the floor (0.302 → 0.431,
        0.288 → 0.448).
      - **The consistency check is half the obstacle.** For those 50 people,
        15 of the 24 references it refused are side-on, against 27 of the 176
        it kept. Of the 89 side-on Commons faces, 22 disagree with a mostly
        face-on medoid, and only a person can say whether they show the
        person turned away.
    - **Size:** from eyes to mouth, smaller than every face the gallery
      recognised. No reference lifts it. Under the 20 px floor a face is never
      scored, so it counts as coverage rather than as a failure. On the
      sample, the face-on face 37 px wide (18 px from eyes to mouth) was a size
      failure before the floor existed. It is now coverage, beside the person
      who has no usable reference face.

    A decision that changed a label is counted by where its name falls
    (`label_noise`): the gallery had matched two of the four wrong-to-right
    relabels, and the other two are the profile faces.
  - **Per figure, always.** Every run reports:
    - per-figure counts;
    - the concentration: the top-two share and the effective number of
      figures;
    - a figure-stratified estimate, where every figure weighs the same and the
      intervals come from resampling figures;
    - the pooled numbers without the top two figures.

    When the top two carry more than half the trials, the report says the
    pooled number is about those two people, not the method. The full
    population is 510 train-only images. About a fifth show one person, the
    most-named figure.
  - **Adjudication is a person's**, on the `review` sheet, and partial is
    usable. The decisions file has a null slot per trial, in sheet order, and a
    null keeps the label. Decisions are keyed by image and name, so they carry
    over to the full calibration.
  - **The sheet holds only undecided trials, in three parts:**
    1. every trial the gallery scores at or above its consistency threshold,
       whatever its label;
    2. names labelled right that score below that floor, or cannot be scored;
    3. names labelled wrong that a claim's text on the image carries, such as
       handles like @narendramodi, which neither NER nor the probe's patterns
       resolve.

    An accepted name nobody looked at is where an unseen false accept would
    sit, and a bound that holds only because no one looked is not a bound.
    `calibrate` reports how many decisions changed a label, and in which
    direction (`relabelled`).
  - The published `data/reports/gallery_calibration.json` gives figures by rank
    only, never a name or QID, and drops every `local` part (the trials and
    names a person reads). `public_report` refuses otherwise.
- **The pilot's 47.8% is approximate and understated.**
  - Approximate: its reference is a loose token pattern over the claim.
  - Understated: it scored a name against one figure per image, so a person
    the claim names beside that figure, correctly named, counted as wrong. The
    sample's first QID-based figure, 0.5455, carried the same defect.
  - Say so wherever either figure appears.
  - Corrected on the sample, the QID-based precision is 0.7045 (0.7955 with
    the person's review of all 37 sheet rows). The full calibration replaces
    both.
- **Organisations get entries and references only.** No ORG answer claims a
  visual match: logo matching is a different model and a different problem.
- **Models.** YuNet (MIT) finds faces, SFace (Apache-2.0) embeds them; both
  are pinned by sha256 beside their licence files. AuraFace (Apache-2.0, training
  data described only as "a commercial dataset") is a bake-off contrast. Adopting
  it is a licence decision for a person to make, with the data in hand. InsightFace's
  packs are excluded: their models are "for non-commercial research only".

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
scripts/image_text.py      image-to-text: description, entities, OCR per distinct image; resumable
scripts/gallery.py         entity gallery: names, Wikidata links, Commons references, faces, 1:1 verify, calibration
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
data/gallery/              the entity gallery: links, Commons references + licences, faces (ignored)
data/interim/gallery/      name table, claim NER cache, calibration faces and trials (ignored)
data/interim/image_text/<config_id>/<dataset>.jsonl  one row per distinct image (ignored)
models/                    model weights, configs.paths.MODELS (ignored; refused by prepush_check)
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

python scripts/image_text.py fetch-models                 # once: pinned Qwen2.5-VL + EasyOCR weights (network)
python scripts/image_text.py run --pilot 200              # measure (VERITE + Factify2 recognition probe), then stop
python scripts/image_text.py run                          # VERITE: 607 distinct images, resumable
python scripts/image_text.py run --through factify2       # + AVerImaTeC 1,345, Factify2 45,186
python scripts/image_text.py run --through fakeddit --fakeddit-n N   # + nested stratified sample of 140,839
python scripts/image_text.py recheck --n 3                # fresh process; finished rows must be byte-identical
python scripts/image_text.py status                       # offline counts per ledger
PE_FAKE_NEWS_GPU_TESTS=1 pytest -q tests/test_image_text.py   # the model-backed tests

python scripts/gallery.py fetch-models                    # pinned NER, YuNet, SFace (network)
python scripts/gallery.py count                           # names in our corpora, before any fetch
python scripts/gallery.py link --dry-run                  # size the sample offline
python scripts/gallery.py link && python scripts/gallery.py fetch   # the sample: Wikidata, then Commons (network)
python scripts/gallery.py link --all && python scripts/gallery.py fetch --all --budget-gb 8   # the build; resumable
python scripts/gallery.py link --hold KEYS --reasons R.json --by WHO   # links a person must check first; offline
python scripts/gallery.py link --release KEYS --by WHO    # restore held links a person has checked; offline
python scripts/gallery.py embed --backend sface --pin-consistency   # faces; CPU, offline; pin once, after the build
python scripts/gallery.py embed --backend sface           # faces; CPU, offline
python scripts/gallery.py calibrate --backend sface       # Factify2 train only
python scripts/gallery.py review --backend sface          # a local sheet for a person to check
python scripts/gallery.py sample-report                   # data/reports/gallery_sample.json
python scripts/gallery.py verify --image P --name "N"     # one 1:1 verification; no certified threshold, no decision
python scripts/gallery.py side-views --backend sface      # side views on Commons, top 50: a measurement (network)
PE_FAKE_NEWS_GALLERY_MODELS=1 pytest -q tests/test_gallery.py  # the pinned-weight tests
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
