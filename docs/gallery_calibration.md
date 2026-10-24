# Entity gallery: calibration report

**2026-10-24. The full calibration, on the complete Factify2 ledger** (run
`93cd9d679720`, 45,186 images). Every figure below is read from
`data/reports/gallery_calibration.json`, written by `calibrate` on the server on
2026-10-23 (face configuration `b6abc4a9f987`: YuNet finds faces, SFace embeds
them; consistency threshold pinned at 0.3811). A second run on another machine
reproduced that report in every value (section 6).

Figures marked *derived* are computed from the local trials
(`data/interim/gallery/calibration/sface_trials.jsonl`, ignored because it
carries names and record ids), by the method given beside them. Figures are
reported by rank, never by name, as the published per-figure rule requires.

The question the gallery answers is 1:1: does this image show the person a
name, proposed by the VLM from the face alone, resolves to? A trial is one such
name on one Factify2 TRAIN claim image. It is labelled right when the name
resolves, by QID, to anyone the image's claims name.

## 1. The result: the threshold stays unset

No threshold reaches the bar, so the gallery does not decide at this confidence.
That is the result, stated as one: it is not a failure to report one. The bar is
accepted-name precision whose one-sided 95% lower bound (Clopper-Pearson) is at
least 0.90. A point estimate is never adopted.

| At the candidate threshold, 0.5214 | Raw labels | Adjudicated |
| --- | ---: | ---: |
| VLM precision (QID rule), over 282 trials | 0.6348 | 0.6489 |
| Accepted names, right of all | 129 of 141 | 131 of 141 |
| Accepted-name precision (one-sided 95% lower bound) | 0.9149 (0.8658) | 0.9291 (0.8827) |
| TAR / FMR | 0.7207 / 0.1165 | 0.7158 / 0.1010 |
| Accepted names needed to certify at that precision | about 1,183 | about 333 |
| Figure-stratified precision (one-sided 95% lower bound) | 0.9126 (0.821) | 0.9506 (0.8828) |

- **Why it falls short: too few trials, not overlap.** The point estimate
  clears 0.90 and the bound does not. With 5-fold cross-validation, the bound
  rule finds no threshold in any fold.
- **Adjudication is partial.** The adjudicated column applies the 37 decisions a
  person made on the sample's sheets (2026-10-22): 35 confirmed the VLM's name,
  2 found someone else, and 4 of them turned a wrong label right.
  - Only 28 of the 141 accepted names are among them. The other 113 rest on
    the claim-based labels alone.
  - An accepted name nobody looked at is where an unseen false accept would
    sit, so these labels are not yet a bound.
- **How close it is.** On 141 accepted names the bound reaches 0.90 at 134
  right (133 right gives 0.89996). The count is 131 now.
  - A review that turns 3 of the 10 wrong labels right, and finds none of the
    131 right labels wrong, certifies 0.5214 on these trials.
  - A review that finds right labels wrong moves certification further away.
  - This is arithmetic, not a forecast.

*Derived* (`accepted_names_needed`): how many accepted names a precision, held,
needs before its bound reaches 0.90.

| Accepted-name precision | 0.92 | 0.9291 | 0.94 | 0.95 | 0.96 | 0.97 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted names needed | 671 | 333 | 191 | 129 | 89 | 61 |

## 2. The VLM's wrong names are look-alikes, not random strangers

Face verification is usually calibrated on random pairs: a threshold is set from
the false-match rate between different people chosen at random. This gallery is
asked about names a VLM proposed from a face, and the VLM's wrong names are not
random people. They are the people it mistook the face for.

**Method.**
- Each right name the gallery can score has its query face scored against a
  random other gallery person (seeded). This is what a random-pair calibration
  measures.
- A wrong name is scored as verify scores it: the face against the references
  of the person the VLM named.
- The random-pair threshold is the lowest at which
  p·TAR / (p·TAR + (1−p)·FMR) ≥ 0.90, with FMR measured on the strangers and p
  the VLM's precision.

**Result, on the run's own draw of strangers:**

| | Raw labels | Adjudicated |
| --- | ---: | ---: |
| Wrong names (of them scorable) | 103 (70) | 99 (66) |
| Median score: right names / wrong names / strangers | 0.627 / 0.308 / 0.148 | 0.625 / 0.297 / 0.151 |
| Threshold a random-pair calibration would set | 0.233 | 0.239 |
| Wrong names that threshold accepts | 49 of 103 | 44 of 99 |
| Accepted-name precision there (one-sided 95% lower bound) | 0.75 (0.69) | 0.77 (0.72) |
| Scorable wrong names above every stranger | 33 of 70 | 16 of 66 |

**Robustness** (*derived*: the pipeline's own procedure, `random_pair_scores`
then `look_alike_finding`, over 200 seeds, with nothing but the seed varied):
- **Random-pair calibration gives accepted-name precision of 0.73 to 0.77 on
  raw labels (median 0.754) and 0.74 to 0.79 adjudicated (median 0.774).** Not
  one of the 200 draws reaches 0.90.
- **"Above every stranger" is the maximum of one draw, and it moves with the
  draw.** It runs from 3 to 35 (median 26) on raw labels and from 8 to 31
  (median 22) adjudicated. That is why it halved under adjudication: 4
  relabels changed which strangers were drawn.
  - The stable form: 30 of 70 wrong names (raw) and 27 of 66 (adjudicated)
    score above the 99th percentile of the stranger scores pooled over the 200
    draws (28,863 and 29,597 scores; percentiles 0.356 and 0.353).
- **A wrong VLM name outscores a random stranger with probability 0.87** (0.86
  adjudicated). A right name outscores a wrong one with probability 0.85 (0.86).

**What it means.**
- A false-match rate measured on random pairs understates the false accepts of
  names a VLM proposes, because the VLM proposes look-alikes. Its wrong names
  score about twice as high as random strangers (medians 0.31 and 0.15).
- Calibrated on random pairs, the gallery would accept roughly one wrong name in
  four: precision about 0.75, against the 0.90 required.
- Calibrated on the VLM's own names, the candidate is 0.52, more than twice the
  random-pair threshold. A verifier must be calibrated on the names it will
  actually be asked to verify.

**History.** The 6% sample pointed the same way on 44 trials: random-pair
precision 0.63 as first labelled, 0.85 adjudicated. The full run measures it on
282 trials, and its raw and adjudicated figures agree.

**Limits.**
- One recognizer (SFace) and one population (Factify2 train under the claim
  rule, section 4).
- The strangers come from the 3,184 gallery people with a usable reference face.
- Labels are partly adjudicated: 37 of 282 trials.
- 33 of the 103 raw wrong names cannot be scored and are outside the score
  comparison.

## 3. Accuracy depends on who is being verified

The gallery's precision and recall differ with the person being verified, and
the pooled figures are mostly about two people.
- 61 figures (each image's stratum, the one figure its claims name) carry the
  282 trials.
- The two most-tried carry 47.9% of them, and the first alone 36.9%. The
  effective number of figures is 6.2, and 36 figures have a single trial.

*Derived*: adjudicated labels, names accepted at the candidate (0.5214), and
two-sided 95% Clopper-Pearson intervals.

| Figure | Trials | Accepted, right of all | Precision [95% CI] | Right names accepted (TAR) [95% CI] | Accepted names a person has checked |
| --- | ---: | ---: | --- | --- | ---: |
| rank 1 | 104 | 64 of 66 | 0.97 [0.89, 1.00] | 64/85 = 0.75 [0.65, 0.84] | 8 of 66 |
| rank 2 | 31 | 6 of 10 | 0.60 [0.26, 0.88] | 6/19 = 0.32 [0.13, 0.57] | 2 of 10 |
| rank 3 | 23 | 10 of 12 (raw 9) | 0.83 [0.52, 0.98] | 10/15 = 0.67 [0.38, 0.88] | 5 of 12 |
| rank 4 | 10 | 3 of 3 | 1.00 [0.29, 1.00] | 3/4 = 0.75 [0.19, 0.99] | 1 of 3 |
| rank 5 | 8 | 2 of 3 | 0.67 [0.09, 0.99] | 2/4 = 0.50 [0.07, 0.93] | 0 of 3 |
| rank 6 | 7 | 5 of 5 | 1.00 [0.48, 1.00] | 5/5 = 1.00 [0.48, 1.00] | 1 of 5 |
| rank 7 | 7 | 5 of 5 | 1.00 [0.48, 1.00] | 5/5 = 1.00 [0.48, 1.00] | 2 of 5 |
| rank 8 | 7 | 3 of 3 | 1.00 [0.29, 1.00] | 3/4 = 0.75 [0.19, 0.99] | 2 of 3 |
| rank 9 | 6 | none | — | no right name | — |
| the other 52 (fewer than 5 trials each) | 79 | 33 of 34 | 0.97 [0.85, 1.00] | 33/42 = 0.79 [0.63, 0.90] | 7 of 34 |

**Reading it.**
- **Rank 1:** 64 of its 66 accepted names are right (0.97).
- **Rank 2:** 6 of 10 (0.60). Even the top of its interval is under 0.90.
  - Its TAR is 0.32. Of its 19 right names, 7 cannot be scored (5 faces under
    the 20 px floor, 2 images with no face found), and 6 score below the
    candidate.
- **Weighting every figure the same** (figure-stratified, adjudicated):
  precision 0.95 (one-sided lower bound 0.88) and TAR 0.81.
  - Without the two most-tried figures, the pooled count is 61 of 65 (0.94,
    lower bound 0.86).
  - The threshold at which figure-weighted precision's bound reaches 0.90
    would be 0.766, against the pooled candidate 0.521: the pooled threshold
    leans on the most-tried figures.

**The caveat that decides how large the spread is.**
- Rank 2's precision rests on 4 accepted names that the claim rule labels
  wrong and no person has checked.
- They score 0.61 to 0.74, well above the candidate. They name other people in
  images whose claims name rank 2's figure and not them.
- A person has turned that kind of label from wrong to right 4 times before:
  the claims named the figure, and not the bystander the VLM named correctly.
- Until they are checked, rank 2's 0.60 is a measurement on unverified labels.
  Its TAR of 0.32 does not depend on them: it is coverage and recognition.

**Stated plainly.** The method's accuracy depends on who is being verified:
- through the faces their images carry (size, pose, crowding);
- through the look-alikes the VLM proposes for them.

So every result is reported per figure, with the figure-stratified estimate
beside the pooled one.

## 4. Reaching 333 accepted names: what widening would cost

At the adjudicated precision (0.9291), certifying needs about 333 accepted
names; at the raw precision, about 1,183. The population gives 141.

**First, the review the method requires anyway.**
- 113 of the 141 accepted names are unchecked. A local sheet holds them, 157
  rows in all (`review --backend sface`).
- If 3 wrong labels turn right and no right label turns wrong, 0.5214
  certifies at 134 of 141, with no widening. Section 1 has the arithmetic, and
  either outcome is open.

**Then widening.** *Derived* counts, from the ledger: the VLM's appearance-only
person names that resolve to one gallery person with a usable reference face.
Projected accepted names apply today's yield (141 of 278 such names, 51%), and
are projections, not measurements.

| Population | Images | Scorable VLM names | Projected accepted | What it costs |
| --- | ---: | ---: | ---: | --- |
| Factify2 train, the claim rule (now) | 510 | 278 | 141, measured | — |
| … a "pictured" word, any number of people named | 698 | 387 | about 196 | still short |
| … one figure named, no "pictured" word | 4,994 | 2,860 | about 1,450 | weaker labels (below) |
| … anyone named, no "pictured" word | 6,641 | 3,958 | about 2,000 | weaker labels, and several people per claim |
| … every train image, labelled by a person alone | 19,546 | 6,590 | about 3,340 | no claim label at all: every accepted name is a person's call |
| Factify2 val, the claim rule | 63 | 31 | about 16 | spends the split model selection uses |
| AVerImaTeC, the claim rule | 147 | 29 | about 15 | evaluation-only in this project: spends it |
| AVerImaTeC, every image | 1,345 | 200 | about 100 | evaluation-only; still short |
| VERITE | — | — | — | off-limits: never read for any threshold |

**What the looser rules cost in label quality.**
- Under the claim rule, a claim names one figure and uses a word for a picture
  ("photo", "image", "video"), and a name is right when the claims name that
  person. Even so, a person turned 4 of 37 checked labels from wrong to right:
  bystanders the claims did not name.
- Without the "pictured" word, a claim need not be about the image at all. A
  right label then says only that the claim mentions the person, and a wrong
  one that it does not.
- The share of scorable names labelled right falls from 66% to 59%.
- How much of that is label noise rather than VLM error is unmeasured
  (UNVERIFIED). Only a person checking a sample can measure it.

**The honest route.**
- Any widening that reaches 333 needs a person to check its accepted names. A
  bound that holds only because no one looked is not a bound.
- At 0.93 precision that is about 333 checked names, and at 0.92 about 671. The
  constraint is a person's time, not data: Factify2 train without the
  "pictured" word, with every accepted name reviewed on the sheet, holds
  enough.
- AVerImaTeC and val would spend held-out data for too few names, and VERITE
  is off-limits.
- If that review is not done, the threshold stays unset as a stated result.
  The gallery then returns a score and the matched reference for a person to
  check, and never decides.

## 5. Known limits, stated beside the calibration

- **Coverage ceiling.** 1,592 of the gallery's 4,776 people have no usable
  reference face, and 1,430 of them have no Commons reference at all.
  - Of the 183 right names (adjudicated), 29 cannot be scored: 17 faces under
    the 20 px floor, 11 images with no face found, and 1 person with no
    usable reference.
  - They count against TAR, never as accepts.
- **Lone P18 references.**
  - 685 people's only single-face reference is a lone Wikidata (P18) image,
    which no consistency check can test.
  - By title, at least 3 of those are not photographs: two postage stamps and a
    bust.
  - Title keywords mark 30 usable P18 references in all, some of them wrongly.
    That is an agent's reading of titles, not a person's check.
  - Whether this is load-bearing waits on the review above.
- **How far apart SFace puts one face.** Two official photographs a person
  confirmed show the same man score 0.167 against each other.
  - In all, 10 pairs a person confirmed as one face score under the 0.3811
    threshold, at 0.167 to 0.362.
  - Genuine pairs across pose, lighting and headwear can score this low,
    which bounds the TAR this recognizer can reach.
- **Names that refuse rather than guess.** 459 names that would have resolved
  to one person refuse instead (7.3% of mentions), because the word is several
  people's or only a nickname. In this run, 2 trials are unscorable as a name
  several people hold.
- **One corpus, one rule, one recognizer:** Factify2 train under the claim rule,
  concentrated on two figures, and SFace.

## 6. Reproducibility

- A second run on another machine produced the same report in every value,
  apart from the time it was written. It used the same ledger, gallery,
  decisions and code (snapshot `e579766`).
- Two decisions of 2026-10-23 are why it can: similarities are computed in
  float64, and the medoid of two references is broken by rule (the P18, else
  the first). Before them, float rounding could break a near-tie differently
  on each machine.
- To reproduce:
  `python scripts/gallery.py calibrate --backend sface --ledger-dir data/interim/image_text/93cd9d679720`.
  Its prerequisites are in CLAUDE.md under "Entity gallery: standing rules".
