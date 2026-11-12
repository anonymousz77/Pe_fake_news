# The consistency block

Does a claim agree with what is known about its image? For each claim and
image, and each item (date, location, people, event), the block gives a
verdict: consistent, contradicted or unsupported. Each verdict carries a
score, the evidence it used and a reason code. v0 is zero-shot and trains
nothing. The code is `scripts/consistency.py`; COVE is the closest prior
(`docs/related_work.md`).

The stages, each gated by the person before the next starts:

- **A, the comparator on gold pairs.** The person's sheet (110 rows of the
  four items) and M4FC train `true` rows, consistent by construction.
  `data/reports/consistency_a.json`.
- **B, the context predictor.** What the image's evidence says of its date,
  place and event, scored against M4FC's annotated context.
  `data/reports/consistency_b.json`.
- **C, end to end** on M4FC val and VERITE, CLIP beside every number.
- **D, a demonstration** for the supervisor.

This file grows with the block. The final numbers and limits come at its end.

## Dataset findings

### M4FC: many images are screenshots of the claim (2026-11-10)

Found in report B; recorded as a property of the dataset, by the person's
decision.

- Take the out-of-context train rows whose claimed place contradicts the
  actual one: 472.
- In 291 of them, the image model's own GPE line agrees with the claim and
  contradicts the actual place. That line never sees the claim.
- In 161 of those 291, its value is written in the image's OCR text. In 215
  it is written in the claim's text.
- Many M4FC images are screenshots of the post that made the claim. The
  image then carries the claim, and evidence read from its text echoes the
  claim instead of telling where the photograph was taken.
- C answers this with the `image_carries_claim` rule (Decisions, below).
  Any image-side evidence that reads text in the image is exposed to it.

## Decisions

Recorded as they are taken. A decision is the person's unless it says otherwise.

### Date is decided by the rule alone (2026-10-30)

- **The decision.** In C, a date verdict is the date rule's and nothing
  else's. The rule reads each part of a date at its own granularity and calls
  a pair consistent when the intervals overlap, contradicted when none do.
  Where the rule cannot read a part, the verdict is unsupported
  (`comparator_unclear`). It is never sent to the judge.
- **Why.** On the server, the judge's GPU test asked whether "the 2010s" and
  "2014-02" can both be true. It answered no, at 0.554 (yes 0.336, unclear
  0.109, label mass 0.988), and the answer was deterministic. A judge that
  contradicts a decade by a month inside it cannot be the fallback for the
  dates the rule cannot read.
- **What it changed.** The fixed configuration for date in Stage A was "the
  rule, then the judge where the rule says unclear". On the gold pairs the
  rule decided every one, so the date figures are unchanged:
  - sheet recall 16/16;
  - true-row false contradictions 11/57.

  `ROUTES` and `date_verdict` hold the routing, and a test asserts it.
- **Stage B too.** B scores a predicted date against M4FC's by the same
  rule; the rule's unclear is "undecided".

### Gates A and B, and what C's score counts (2026-11-10)

The person's decisions, taken on report A after their review and on report B.

- **Gate A: event fails.** After the review, 8 of 78 true-row event pairs are
  false contradictions (0.1026, against at most 0.10). The bar was set
  before the run, so event stays unsupported in the primary score
  (`gate_a_not_cleared`). An "event included" score is reported beside the
  primary one, labelled as a variant (`with_event`).
- **Dates leave the primary score (Gate B).** B's dates agreed with M4FC's
  actual date 8% of the time on train and 5% on val. A date verdict is kept
  on the record but counts as unsupported (`gate_b_left_out`). A second
  variant (`with_grounded_date`) counts a date only where its year is
  written in the evidence it cites, a document's publication date excluded
  (`date_not_grounded` otherwise).
- **The primary score is location plus people.**
- **Gate B is accepted as measured, with one rule added before C ran:
  `image_carries_claim`.** A verdict is unsupported where both of these hold:
  - the predicted value rests only on text read in the image: everything it
    cites is the OCR line, or an entity line the image model says is
    "written in the photograph" (basis `text`), forced lines included;
  - that text contains the claimed value, by the same `contains` report B
    used (case-folded; a date by its years, anything else by a part of three
    characters or more).

  A value that also cites the description, a face, a document, or an entity
  recognised by how it looks does not rest on text alone. The rule applies
  to every item it can reach: location, and the variants' event and date.
  - The image is a screenshot of the claim, not independent evidence.
  - Fixed before any C verdict existed. It is measured on M4FC train (report
    C gives each score with and without it) and applied unchanged to val and
    VERITE, where only what it changed is counted.
- **The predictor's 19 answers short of their final brace are not
  repaired.** They stay errors and are reported.
- **Decades and spans are dates in the extraction** (parser 2, no rerun).
  An extracted "1980s" or "1951-1967" is read as the interval the date rule
  gives it, when the claim writes every year in it.
  - C reads C.1's kept answers again by the current parser.
  - That changed 9 items and nothing else: 6 on M4FC (five decades, one
    span) and 3 on VERITE (two spans, one decade). Partial rows fell from
    48 to 42 and from 9 to 6.
  - The run's prompt configuration is unchanged, so its config id stands.

### The predictor's first pilot (2026-10-30)

The decisions in this section are the assistant's, taken on the pilot slice
of M4FC train (the only rows the predictor is developed on). The person may
overrule them.

- **What ran.** 30 M4FC train images, prompt config 0798333af501.
  - 26 of 30 answers were not JSON. Each wrote its evidence ids bare
    (`"evidence": [E3]`), as the prompt shows them.
  - Read as ids, the answers would have been 28 ok and 2 partial: two
    locations cited no evidence.
- **Dates echoed the documents.** All 30 answers gave a date. 22 of them
  cited documents, and 20 were exactly a cited document's publication year
  or month. The documents are found through the claim's words, published
  before its fact-check, so such a date is the claim's own time, not the
  photograph's.
- **The fixes.**
  - The prompt now says that a document's date is when it was published,
    not when the photograph was taken.
  - It asks for each id as a quoted string, and for null in place of a value
    that rests on no evidence.
  - The parser reads an unquoted id inside an evidence array as that id, and
    flags it (`bare_ids`). Nothing else is repaired.
- **Repeated documents.** The predictor's documents are the top five
  distinct by title and excerpt. The store holds one story under several doc
  ids: 380 of 3,021 M4FC claims had a repeat in their top five.

### The predictor's second pilot (2026-10-30)

The same 30 images, prompt config fac8cc437a8d.

- **The result.** 23 ok and 7 partial, against the person's bar of at most
  one. Every answer was JSON. Every accepted value cited only ids its
  prompt gave, and every accepted date was in one of the three forms.
- **The partials.**
  - Six were locations naming a country ("India", "United States") with no
    evidence cited. The prompt's "with its country" asked for one.
  - One event cited "E1" in a prompt without entity lines. The prompt's
    example id "E1" had put it there.
  - One more event gave a value with no evidence.
- **Dates.** 26 answers gave one. 12 were still a cited document's
  publication date (20 in the first pilot), and 13 came from the image
  model's DATE line.
- **The fix, the last before the bar would stop the run.**
  - One invented worked example, shown as a prior turn
    (`configs/consistency_fewshot.yaml`, `predict`). It quotes its ids, rests
    every value on cited evidence, leaves an unrelated document unused,
    names no country the evidence lacks, and gives a null date although
    both its documents are dated.
  - Each item now lists its evidence before its value.
  - A country is named only where the evidence gives one, and the
    instructions carry no example id.
- **What was not done.** An uncited value is not turned into a null by the
  parser. That would pass the bar without changing what the model does.

### The predictor's third pilot, and the full run (2026-10-30)

- **The bar passes.** The same 30 images, prompt config 660d8161fe03, came
  back 30 ok, at 2.3 s each.
  - Every value cites only ids its prompt gave.
  - Every date is in one of the three forms.
  - The model is more cautious now: 18 dates, 11 locations and 23 events,
    where the first fix gave 26, 23 and 28.
- **Answers that look wrong, read by the assistant against their evidence.**
  - 9 of the 18 dates add a year to a day and a month the image model read
    ("September 12", "23. Mai").
  - 2 still echo a document's publication date.
  - 2 more cite a description or OCR text that holds no date.
  - One event adds what no cited document says. One location adds a
    country the evidence lacks.
- **The format bar does not measure this.** Report B scores dates twice:
  as predicted, and only where the predicted year is written in the
  evidence the date cites, a document's publication date excluded
  (`date_grounded`).
- **The full run.** 4,106 prompts under the pilot's configuration:
  - 3,021 M4FC train+val with retrieval;
  - 497 M4FC val without;
  - 588 VERITE images.

### Stage B: the full run and report B (2026-11-10)

Gate B is the person's reading of `data/reports/consistency_b.json`. This
records what ran and what the report measures; it decides nothing.

- **The predictor's full run.** Config 44010bda9178 (code 39b0b64, prompt
  config 660d8161fe03): 4,106 rows in 2.41 GPU-h. A recheck of 3 rows in a
  fresh process was identical.

  | variant | rows | ok | partial | error |
  | --- | ---: | ---: | ---: | ---: |
  | M4FC train+val, with retrieval | 3,021 | 2,906 | 97 | 18 |
  | M4FC val, without retrieval | 497 | 463 | 32 | 2 |
  | VERITE, image only | 588 | 577 | 11 | 0 |

  - Every error is an answer that is not JSON. 19 of the 20 lack only their
    final closing brace. None is repaired; whether to is the person's call.
  - Item errors with retrieval: a date in none of the three forms 53, an
    event with no evidence 30, a date citing an id its prompt did not give
    12, and 4 more.
- **The scoring judge.** 11,099 pairs (event 6,733, location 4,366), stage
  A's prompt config dbff86db903a, config 007d326ccf33 (code 2c34857), at
  0.83 s each: 2.56 GPU-h. Every answer was in format (label mass at least
  0.928). A recheck of 3 in a fresh process was identical.
- **What the report measures.** Counts by the assistant, not a review.
  - **Dates.** Of the predicted dates the rule compares, 8% agree with
    M4FC's actual date on train (106 of 1,281) and 5% on val. Of the 1,174
    train disagreements, 994 are a different year, and the predicted year
    is the later one in 714 of them, the earlier in 279. Only dates whose
    year their cited evidence writes: 15% agree on train, 10% on val.
  - **The image often carries the claim.** On out-of-context train rows
    whose claimed place contradicts the actual one, the image model's own
    GPE line, which never sees the claim, agrees with the claim and
    contradicts the actual place in 291 of 472. In 161 of those 291 its
    value is written in the image's OCR text. Many M4FC images are
    screenshots of the post that made the claim.
  - **Retrieval finds the claim's context, not the image's.** M4FC's actual
    place is written in the top five documents for 14% of train rows, and
    its actual event for 2%. For the claimed place and event the figures
    are 41% and 6%.

### Stage C, prepared on the laptop (2026-11-08)

Built and tested while the server was out of reach. Nothing in it has run on
a GPU. The assistant designed it; the person may overrule any of it.

- **C.1, the extractor.**
  - Text-only Qwen reads the claim alone, with five invented exemplars as
    prior turns (`configs/consistency_fewshot.yaml`, `extract`), checked
    against M4FC and VERITE.
  - Its answer is read as strictly as the predictor's. In addition, an
    extracted date whose year the claim does not write is an error
    (`date_not_in_claim`): the predictor's third pilot added years, and here
    the claim is the only evidence.
  - Inputs: a 30-claim M4FC train pilot, and 3,636 distinct claims in full
    (3,015 M4FC train+val, 621 VERITE). Prompt config 8db87f51b671, about
    1,060 tokens.
- **C.2, verdicts** (`item_verdict`).
  - The fixed comparator's outcome is kept on every record. An item that
    Gate A has not cleared is unsupported (`gate_a_not_cleared`);
    `gate_cleared` reads the reviewed gate once the person has decided any
    contested pair.
  - A missing prediction is `no_image_evidence`; a refused one
    `prediction_refused`; the comparator's unclear `comparator_unclear`.
- **The people rule** (`people_verdict`).
  - A claimed name the gallery accepts is consistent. A reject is
    unsupported (`gallery_reject`).
  - A different VLM-named person accepted on the image's only face is
    unsupported until the person's decision. Only their "contradicts" makes
    it contradicted.
- **C.4, the score.** The largest decided P(contradicts); a record with no
  decided item abstains at 0. The threshold is fitted on M4FC train alone,
  by balanced accuracy of true against out-of-context, and any other record
  refuses.
- **C.6, metrics.** AUROC, and a paired bootstrap against CLIP that
  resamples whole clusters. VERITE's triplets are joined by shared image or
  caption; an M4FC record is its own cluster.
- **C.5, CLIP.** ViT-B/32 at revision 3d74acf9, on the CPU: each record's
  image-claim cosine.
  - Run 2026-11-08: all 3,898 records (3,021 M4FC train+val, 877 VERITE),
    no errors, 482 s.
  - AUROC with 95% intervals from 2,000 cluster resamples:

    | | AUROC |
    | --- | --- |
    | VERITE, true against out-of-context | 0.753 [0.718, 0.785] (0.764 on all VERITE before) |
    | VERITE, true against miscaptioned | 0.575 [0.543, 0.608] (0.587 before) |
    | M4FC val, true against out-of-context | 0.512 [0.399, 0.635] |
    | M4FC train, true against out-of-context | 0.599 [0.552, 0.647] |

  - Its threshold, fitted on M4FC train, is a similarity of 0.296 or less
    (balanced accuracy 0.581).
  - At that threshold it flags 13 of 31 true and 147 of 342 out-of-context
    records on M4FC val. On VERITE it flags 17% of true, 25% of
    miscaptioned and 52% of out-of-context records.
- **D, the demonstration.**
  - Every sentence is a template filled from the verdict record (`EXPLAIN`),
    and a reason with no template refuses.
  - The page is static and local, and refused if it shows an id form.
  - `demo --rehearse` builds pages from the third pilot's real outputs for
    its 30 train images, against M4FC's own claimed values. Today every item
    on them is unsupported: Gate A clears none, the judge has not compared
    places or events for C, and the gallery accepts none of their claimed
    names.
  - The live path (a new image and claim run through every stage) is not
    built yet.

### C.1, the extraction run (2026-11-10)

What ran and what its output holds. Places and events are compared by the
judge in C, which waits for the person's Gates A and B.

- **The pilot.** 30 M4FC train claims, config d1d7aa2f42d4 (code c489cd9,
  prompt config 8db87f51b671), at 1.08 s each: 29 ok and 1 partial, at the
  bar's limit of one. The partial added 2016 to a claim's "November 14"
  (`date_not_in_claim`).
- **The full run.** 3,636 claims under the same config, 1.11 GPU-h (1.10 s
  each). Every answer ended on its own, the longest at 70 of 120 tokens. A
  recheck of 3 in a fresh process was identical.

  | | claims | ok | partial | error |
  | --- | ---: | ---: | ---: | ---: |
  | M4FC train+val | 3,015 | 2,967 | 48 | 0 |
  | VERITE | 621 | 612 | 9 | 0 |

  - Every partial is a date, refused and never repaired.
    - 41 add a year the claim does not write. 17 of them add 2016, the
      year of the first exemplar's date. 31 of the 41 claims write a month
      name.
    - 16 are in none of the three forms. 6 are decades and 4 are spans; the
      claim writes the years of 9 of those 10, and the date rule could read
      them. Of the rest, 2 give two dates, 2 are relative words ("today",
      "last weekend") and 2 are a bare "79".
  - Checked apart from the parser: every accepted date is in one of the
    three forms, with its year written in the claim.
- **Against M4FC's claimed values**, on its 3,015 claims. Dates are compared
  by the rule; places and events by their text alone.
  - **Date.** 312 extracted: 295 agree with M4FC's claimed date, 3
    contradict it and 14 have none to compare.
    - The 3: a claim that writes two years, a day and month read in the
      other order, and a claimed date whose year the claim does not write.
    - 557 claims have a claimed date in M4FC but no extracted one. 545 of
      them write no year at all: M4FC's claimed dates often come from beyond
      the claim's words, and the extractor reads only those words.
  - **Location.**
    - same text 709, one holds the other 537, different 374;
    - M4FC's only 496, the extraction's only 40, neither 859.
  - **Event.**
    - different 1,686, one holds the other 490, same 131;
    - the extraction's only 560, M4FC's only 32, neither 116.
- **Read by the assistant, not a review.** String counts, and a seeded
  sample of 40 M4FC train rows.
  - **Locations.** Of 1,357 accepted on M4FC train:
    - 1,336 are written whole in the claim;
    - 12 more join places it writes apart ("Texas and Arkansas");
    - 9 are written in no form the text matches ("Switzerland", "a street
      in London").

    Not written whole: 9 of 303 on val, and 12 of 352 on VERITE.
  - **Events run long.** The median is 8 words on M4FC train, and 313 of
    2,396 run over 12. Many restate the whole claim; M4FC's claimed events
    are short phrases.
  - **In the sample:**
    - a distance read as a place ("6 million kilometres away");
    - events that name a subject or a picture, not something happening
      ("current tanks", "PM Modi's pictures with …");
    - an event the caption does not state (a protest, for a caption that
      describes a sign);
    - a date left null where the claim's only year sits in a quoted title,
      which M4FC reads as the date.

### Stage C: the run and report C (2026-11-10)

Built under the person's gate decisions (Decisions, above). Gate C is the
person's reading of `data/reports/consistency_c.json`; `consistency_c.md`
beside it is local. This records what ran and what the report measures; it
decides nothing.

- **The C judge.** 9,547 pairs: event 6,884, location 2,663.
  - Run: stage A's prompt config dbff86db903a, config 007d326ccf33 (code
    c489cd9), at 0.83 s each, 2.21 GPU-h.
  - Every answer was in format; the lowest label mass was 0.933.
  - A recheck of 3 in a fresh process was identical.
  - 2,357 of the pairs had been asked word for word in B's run under the
    same config. They were asked again, so that C has one clean run.
- **The verdicts.** 6,919 records: M4FC train+val under the extracted and
  the gold claims, and VERITE's 877 under the extracted ones.
- **Against CLIP: the headline does not beat it.** The primary score on
  the claims as extracted, with paired cluster bootstrap intervals:

  | | primary | CLIP | Δ [95%] | decided |
  | --- | ---: | ---: | --- | ---: |
  | M4FC val, out-of-context vs true | 0.633 | 0.512 | +0.121 [-0.015, +0.248] | 161/373 |
  | VERITE, out-of-context vs true | 0.497 | 0.753 | -0.256 [-0.296, -0.215] | 92/588 |
  | VERITE, miscaptioned vs true | 0.489 | 0.575 | -0.086 [-0.120, -0.051] | 87/578 |

  - **On the records the system decided**, CLIP on the same records:
    - M4FC val ties at 0.597 against 0.597.
    - VERITE scores 0.631 against 0.702 (out-of-context) and 0.531 against
      0.619 (miscaptioned). Neither interval excludes 0.
  - **The variants.**
    - With event: 0.615 on M4FC val. On the records it decided it scores
      0.371 against CLIP's 0.569, Δ -0.198 [-0.387, -0.003].
    - With grounded dates: 0.638 on M4FC val. VERITE moves by under 0.01.
  - **The gold-claim diagnostic** (M4FC's own claimed values): 0.626.
    With event, 0.683, Δ +0.171 [+0.021, +0.314]: the only interval that
    excludes 0 in the system's favour. It is a variant on annotated claims,
    not the system.
- **What the report measures.** Counts by the assistant, not a review.
  - **Coverage is low.** The primary score decides 43% of the M4FC val
    pair, and 15-16% of VERITE's. Most claimed places have no image-side
    value: on VERITE, 397 of 503 places are `no_image_evidence`.
  - **Part of M4FC val's 0.633 is abstention.** True records abstain more
    (24 of 31) than out-of-context ones (188 of 342), and an abstention
    scores 0. On the decided records it ties CLIP.
  - **Decided places do not separate the classes on M4FC train.**
    - Out-of-context records: 115 contradicted of 572 decided, 0.20.
    - True records: 7 of 35, 0.20.
    - The image side's place mostly agrees with the claim: report B's echo.
  - **The train threshold is low:** 0.026 on the primary score (train
    balanced accuracy 0.552).
  - **Flags at that threshold.**
    - M4FC val: 0.40 of out-of-context records and 0.13 of true ones.
      CLIP's threshold flags 0.43 and 0.42.
    - VERITE: 0.08 to 0.10 of every class. CLIP flags 0.17 of true
      records, 0.25 of miscaptioned and 0.52 of out-of-context ones.
  - **`image_carries_claim` barely bites.**
    - On M4FC train it changed 6 place verdicts and 2 grounded dates, all
      on out-of-context records that had been "consistent".
    - It moved the primary score's train AUROC from 0.545 to 0.544.
    - Applied unchanged, it changed 2 places and 2 dates on M4FC val, and
      2 dates on VERITE.
    - **Why so little.** Of the 1,057 train places with a predicted value,
      21 rest on text alone. 727 rest on one place line the image model
      marked as recognised "from how it looks" (`seen`).
      - In 391 of those, the cited name is written in the image's OCR text.
      - In 133, that text also holds the claimed place: 118 out-of-context,
        12 manipulated, 2 true and 1 mixture record.
      - These are the screenshot cases the rule was meant for. It misses
        them because the image model labels the line `seen`, not "written in
        the photograph".
      - Whether to widen the rule (to an entity line whose name is in the
        OCR) is the person's call. It is not applied.
- **The people check waits for the person.** 7 other_person
  contradictions are on `c/review/other_person.html`, and its export goes to
  `other_person_decisions.json`. Until then the figures before and after
  the check are the same.
- **The live path.** `scripts/live_demo.sh`, from WSL, takes a new image and
  claim through every stage to a local page.
  - The image model and the text model run on the server, as GPU jobs
    through bgrun, each refusing any configuration but the evaluated one.
    The gallery, retrieval and the verdicts run on the laptop.
  - The driver refuses unless the server runs the laptop's committed HEAD.
  - **Tested on 4 requests:** 3 VERITE records (one per class, three
    images, a seeded draw) and one M4FC train record with its own evidence
    bound as the date, so that retrieval runs.
  - Every stage was identical to the batch run: the image model's
    computation, the faces, the claim's people, the predictor's prompt and
    answer, the extraction's answer, the judge's answers and the verdicts.
    `live check` compares them.
  - The server stages took 2.9 to 4.6 s per image and 3.4 to 4.9 s per
    request, plus model loads.
  - All 4 abstain on the primary score, as most VERITE records do.

### Gate C: the negative result (2026-11-10)

The person's decision on report C: **accepted as measured.**

- **Zero-shot context-then-compare does not beat CLIP.** On M4FC val the
  primary score's interval includes 0 (+0.121 [-0.015, +0.248], as first
  measured).
- **It is clearly worse on VERITE:** -0.256 [-0.296, -0.215] for
  out-of-context against true, and -0.086 [-0.120, -0.051] for miscaptioned
  against true. Both intervals exclude 0, against the system.
- **Coverage is low.** The primary score decides 43% of the M4FC val pair
  and 15-16% of VERITE's. Most claimed places have nothing on the image side
  to compare with.
- **Abstention inflates M4FC val.** An abstaining record scores 0, and true
  records abstain more often (24 of 31) than out-of-context ones (188 of
  342). Part of the 0.633 is which records abstain, not what was decided. On
  the decided records it ties CLIP, 0.597 against 0.597.

What followed, each the person's decision of the same day: the rule widened,
the people checks applied, and a fused score.

### image_carries_claim, widened (2026-11-10)

- **The defect.** On M4FC train, most screenshot cases cite a place line the
  image model marked as recognised "by how it looks" (`seen`), although the
  name is written in the image's text. The narrow rule counted only lines
  marked "written in the photograph", so it missed them.
- **The widened rule.** For a place, a place line recognised by how it looks
  also counts as text read in the image, when its name is in the OCR text.
  The text it rests on is then that OCR text. Everything else is unchanged:
  - the value must cite nothing else;
  - the text must contain the claimed place, by `contains`.
- **Found on train, applied unchanged** to val and VERITE.
  `CARRY_RULE = "widened"`; report C keeps the narrow form and no rule beside
  it.
- **The 133 counted at report C was a looser reading.** That count took entity
  lines of any kind, not place lines only. The rule as worded changes the
  numbers below.

**Before and after**, extracted claims, primary score:

| | without the rule | narrow | widened |
| --- | ---: | ---: | ---: |
| M4FC train AUROC | 0.546 | 0.544 | 0.523 |
| M4FC train balanced accuracy | 0.554 | 0.552 | 0.531 |
| M4FC train abstains | 0.68 | 0.68 | 0.73 |
| M4FC val, out-of-context vs true | 0.634 | 0.633 | 0.604 |
| VERITE, out-of-context vs true | 0.500 | 0.500 | 0.500 |
| VERITE, miscaptioned vs true | 0.489 | 0.489 | 0.487 |

- **On train, widening changed 129 place verdicts** beyond the narrow form.
  All of them became unsupported:
  - out-of-context: 107 had been consistent, 7 contradicted;
  - 12 manipulated, 1 mixture and 1 true record had been consistent;
  - 1 true record had been contradicted.
- **Applied unchanged:** 22 more places on M4FC val (21 out-of-context, 1
  AI-generated) and 1 on VERITE (miscaptioned).
- **Why the score falls when the rule is right.** The verdicts it removes are
  mostly "consistent" on out-of-context records. As consistent verdicts they
  scored their small P(contradiction), which ranked them above true records
  that abstain at 0. Now unsupported, they score 0 as well. That is the
  abstention artefact of Gate C's last point, working in reverse.

### The people checks applied (2026-11-10)

The person decided all 7 `other_person` contradictions
(`c/review/other_person_decisions.json`):

- **4 contradicts:** 1 on M4FC train, 3 on VERITE.
- **3 not a contradiction:** 2 on M4FC train, 1 on M4FC val.

After the check:
- VERITE's primary score flags 76 of 877 records, against 74 before.
- VERITE's out-of-context AUROC moves from 0.497 to 0.500.
- M4FC val is unchanged.

One of the "not a contradiction" cases is a Facebook screenshot. There the
gallery matched the poster's avatar, not the photograph's subject. The person's
check set it aside, which is how it stays (the face-rule decision below).

### The system score: CLIP and the consistency score, fused (2026-11-10)

- **The score.** One logistic regression on [CLIP similarity, the primary
  consistency score, decided], fitted on M4FC train only: 164 true records
  against 1,836 out-of-context ones. Fixed before fitting:
  - claims as extracted;
  - classes balanced;
  - L2 1 on standardised features;
  - Newton's method in numpy (`fit_logistic`; there is no sklearn here).
- **Where the block abstains,** the consistency score and the flag are 0, so
  the fused score is a function of CLIP's similarity alone.
- **Explanations come only from the verdicts.** Where the block abstains, the
  demo says that the flag rests on image–text similarity alone (`FUSED_BASIS`).
- **The fit.** Coefficients per raw unit:
  - similarity -7.875;
  - consistency score +0.853;
  - decided -0.196.

  Train threshold 0.512 (balanced accuracy 0.582). Train AUROC in sample is
  0.5987 for the fused score, against CLIP's 0.5992.
- **It does not beat CLIP.** No interval excludes 0:

  | | fused | CLIP | Δ [95%] | block decided |
  | --- | ---: | ---: | --- | ---: |
  | M4FC val, out-of-context vs true | 0.507 | 0.512 | -0.005 [-0.058, +0.037] | 140/373 |
  | VERITE, out-of-context vs true | 0.758 | 0.753 | +0.005 [-0.006, +0.018] | 94/588 |
  | VERITE, miscaptioned vs true | 0.577 | 0.575 | +0.001 [-0.008, +0.010] | 86/578 |

  - On the records the block decided, where the two rankings can differ:
    - M4FC val: 0.582 against 0.606;
    - VERITE: 0.748 against 0.714 (out-of-context), 0.622 against 0.625
      (miscaptioned).

    Every interval includes 0.
- **Per class, at each score's train threshold.**

  | | true | out-of-context | miscaptioned / manipulated |
  | --- | ---: | ---: | ---: |
  | VERITE, fused | 0.19 | 0.56 | 0.25 (miscaptioned) |
  | VERITE, CLIP | 0.17 | 0.52 | 0.25 (miscaptioned) |
  | M4FC val, fused | 0.48 | 0.41 | 0.42 (manipulated) |
  | M4FC val, CLIP | 0.42 | 0.43 | 0.42 (manipulated) |

  65-91% of the fused score's flags fall on records where the block
  abstained, so they rest on similarity alone.

### No face rule for other_person: the person's review is the filter (2026-11-10)

- **The defect.** The other_person rule fired on a screenshot's profile
  picture (the poster's avatar), not on the photograph's subject.
- **A size rule was built, measured and reverted.** The person's design: a
  lone face counts only at 5% or more of the image's shorter side and, in a
  screenshot layout, at 10% or more of its width.
  - On M4FC train it set aside 2 faces:
    - the Facebook post's avatar (5.1% of the width), which the person had
      marked "not a contradiction";
    - the face in the Bigg Boss tweet's photograph (4.8% of the width), which
      the person had confirmed.
  - Size cannot separate them: the photograph's subject is smaller than the
    avatar, relative to the image. On val and VERITE it set nothing aside.
  - Position would separate them: the avatar's centre sits at 0.11 of the
    image's height, the Bigg Boss face's at 0.40. Those are two points, the
    person's own checks.
- **The decision** (the person's, 2026-11-10): revert the size rule, and do
  not add the position cue. Both would be fitted to the person's own two
  checks.
  - Every other_person contradiction already needs the person's review
    before it counts, and that review is the filter. An avatar is the
    person's to set aside there, as they did.
  - The figures above are without any face rule.
- **The claimed-person side,** measured while the rule existed (counts by
  the assistant): the same test would have set aside 23 of 125
  claimed-person accepts on M4FC train (none on a true record), 5 of 31 on
  val and 2 of 54 on VERITE. Nothing is applied there.
- **The review page** showed "[object Object]" where the reference's licence
  and attribution should be: the gallery keeps them as Commons states them,
  as objects. The verdict's evidence now carries their words: the licence's
  short name, and the artist and credit. The page holds all 7 cases.
