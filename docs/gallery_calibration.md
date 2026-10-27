# Entity gallery: calibration report

**2026-10-27. The gallery's threshold is certified: 0.5214, on names a person
has checked, every one it accepts.** It was calibrated on the complete Factify2
ledger (run `93cd9d679720`, 45,186 images).

Every figure below is read from `data/reports/gallery_calibration.json`.
- The server wrote the first version on 2026-10-23: face configuration
  `b6abc4a9f987`, where YuNet finds faces and SFace embeds them, with the
  consistency threshold pinned at 0.3811.
- A second machine reproduced that report in every value, and wrote the final
  version on 2026-10-27 (section 6).

Figures marked *derived* come from the local trials
(`data/interim/gallery/calibration/sface_trials.jsonl`, ignored because it
carries names and record ids), by the method given beside them. Figures are
reported by rank, never by name, as the published per-figure rule requires.

The question the gallery answers is 1:1: does this image show the person that a
name, proposed by the VLM from the face alone, resolves to?
- A trial is one such name on one Factify2 TRAIN claim image, keyed by image
  and name. Two names on one image are two trials and two decisions; 7 images
  carry two or three.
- A trial is labelled right when the name resolves, by QID, to anyone the
  image's claims name. A person's decision, where there is one, replaces that
  label.

## 1. The result: certified at 0.5214

| At 0.5214 | Claim labels | Checked by a person |
| --- | ---: | ---: |
| Accepted names, right of all | 129 of 141 | 141 of 141 |
| Accepted-name precision (one-sided 95% lower bound) | 0.915 (0.866) | 1.000 (0.979) |
| True accepts (TAR) | 0.721 | 0.731 (141 of 193) |
| False accepts (FMR) | 12 of 103 | 0 of 89 (upper bound 0.033) |
| VLM precision, over 282 trials | 0.635 | 0.684 |
| Threshold | unset | **certified** |

**The rule.** A threshold certifies only where the one-sided 95% lower bound of
accepted-name precision (Clopper-Pearson) reaches 0.90, and only where a person
has checked every name it accepts.
- An accepted name nobody looked at is where an unseen false accept would sit.
  A bound that holds only because no one looked is not a bound.
- The code enforces this since 2026-10-27: `choose_threshold(..., checked=True)`,
  and `verify` decides at the threshold of the checked labels.
- A point estimate is never adopted, and the threshold is set only on the
  complete ledger.

**The review.** A person made 150 decisions: 37 on the sample's sheets, and 113
on every accepted name not yet checked.
- 148 found the face to be the person the VLM named, and 2 found someone else.
- 14 decisions changed a label, all of them from wrong to right. Each was a
  bystander the claims did not name, whom the VLM named correctly.
- No right label turned wrong.

**The bound alone reaches lower, and is not taken.**
- At 0.3959 it accepts 162 names, 153 of them right: 0.944, lower bound 0.905,
  TAR 0.79.
- But a person has checked only 146 of those 162; 16 are unchecked.
- Checking those 16 could lower the certified threshold and raise TAR from
  0.73 to about 0.79. This is open work, not a result.

**Two caveats, stated beside it.**
- **The precision rests on labels a person has checked for 141 of 141 accepted
  names.** The rule requires it, and the figure is only as good as that review.
- **A low score is weak evidence that the person is absent.**
  - TAR is 0.73: 52 of 193 right names are rejected. 29 of them cannot be
    scored at all, and 23 score below the threshold.
  - Two official photographs a person confirmed as one man score 0.167
    against each other.
  - So a rejection says the gallery could not confirm the name. It does not
    say the face is someone else. The threshold certifies accepts; it does
    not certify "this is not X".

**Validation.**
- Bootstrap over trials on the checked labels: TAR 0.66 to 0.80, precision
  1.00 to 1.00.
- 5-fold cross-validation over images, with the bound rule alone: fold
  thresholds 0.39 to 0.42, and held-out precision 0.944. That rule does not
  require checked names.

## 2. The VLM's wrong names are look-alikes, not random strangers

Face verification is usually calibrated on random pairs: a threshold is set from
the false-match rate between different people chosen at random. This gallery is
asked about names a VLM proposed from a face, and the VLM's wrong names are not
random people. They are the people it mistook the face for.

**Method.**
- Each right name the gallery can score has its query face scored against a
  random other gallery person (seeded). That is what a random-pair calibration
  measures.
- A wrong name is scored as verify scores it: the face against the references
  of the person the VLM named.
- The random-pair threshold is the lowest at which
  p·TAR / (p·TAR + (1−p)·FMR) ≥ 0.90, with FMR measured on the strangers and p
  the VLM's precision.
- The contrast is measured over 200 seeded draws of strangers
  (`look_alike_finding.over_draws` in the report). Each draw gives every right
  name one stranger.

| | Claim labels | Checked labels |
| --- | ---: | ---: |
| Wrong names (of them scorable) | 103 (70) | 89 (56) |
| Median score: right names / wrong names / strangers | 0.627 / 0.308 / 0.148 | 0.630 / 0.279 / 0.169 |
| Threshold a random-pair calibration would set: median (range over draws) | 0.242 (0.216 to 0.269) | 0.230 (0.204 to 0.260) |
| Accepted-name precision there, on the VLM's own names: median (range) | 0.754 (0.728 to 0.767) | 0.817 (0.785 to 0.838) |
| Draws in which that precision reaches 0.90 | 0 of 200 | 0 of 200 |
| Scorable wrong names above the 99th percentile of every stranger score | 30 of 70 | 17 of 56 |
| How often a wrong name outscores a random stranger | 0.869 | 0.832 |
| How often a right name outscores a wrong one | 0.849 | 0.959 |

On the run's own first draw, a random-pair calibration sets 0.233, accepts 49 of
103 wrong names, and gives precision 0.75 (lower bound 0.69). On the checked
labels it sets 0.229, accepts 37 of 89, and gives 0.81 (0.76).

**Why these statistics, and not the count above every stranger.**
- The first version of this finding counted the wrong names that score above
  *every* stranger: 33 of 70. That count is the maximum of one random draw, and
  a maximum moves with whichever stranger happens to score highest.
- Across the 200 draws it ran from 9 to 37 on claim labels and from 0 to 21 on
  the checked labels. An earlier, independent set of 200 draws gave 3 to 35.
- On the run's own draw it read 33, then 16 after the first 37 decisions, then
  7 after all 150. The labels changed which strangers were drawn, not how the
  VLM's names score.
- A statistic that one draw can move tenfold is not a measurement. The published
  statistics are the ones that hold whichever strangers are drawn:
  - the accepted-name precision a random-pair threshold gives, with its range
    over draws;
  - the share of wrong names above a percentile of every stranger score
    pooled;
  - how often a wrong name outscores a stranger.
- The count above every stranger stays in the report only with its range, to
  show why it is not used.

**What it means.**
- A false-match rate measured on random pairs understates the false accepts of
  names a VLM proposes, because the VLM proposes look-alikes. Its wrong names
  score about twice as high as random strangers.
- Calibrated on random pairs, the gallery would accept roughly one wrong name in
  four or five: precision 0.75 on claim labels, 0.82 on checked ones, against
  the 0.90 required.
- The certified threshold, 0.52, is more than twice the random-pair one. A
  verifier must be calibrated on the names it will actually be asked to verify.
- The review sharpened the separation between right and wrong names (0.85 to
  0.96): part of the apparent overlap was label noise. It did not remove the
  look-alike effect.

**History.** The 6% sample pointed the same way on 44 trials: random-pair
precision 0.63 as first labelled, 0.85 adjudicated.

**Limits.**
- One recognizer (SFace), and one population (Factify2 train under the claim
  rule).
- The strangers come from the 3,184 gallery people with a usable reference face.
- 33 of the 103 wrong names on claim labels cannot be scored, and are outside
  the score comparison.

## 3. Accuracy depends on who is being verified: in recall, not in precision

**Precision does not depend on the person.**
- Before the review, precision seemed to: the second most-tried figure had 6 of
  10 accepted names right.
- After it, every accepted name of every figure is right. The difference was
  noise in the claim labels: 14 wrong labels were correctly named bystanders,
  4 of them in that figure's images.

**Recall does depend on the person.**
- 61 figures carry the 282 trials, and the two most-tried carry 47.9% of them.
- *Derived* table: checked labels, names accepted at the certified threshold,
  two-sided 95% Clopper-Pearson intervals.

| Figure | Trials | Accepted, right of all | Precision [95% CI] | Right names accepted (TAR) [95% CI] | Right names it cannot score |
| --- | ---: | ---: | --- | --- | ---: |
| rank 1 | 104 | 66 of 66 (2 labelled wrong before review) | 1.00 [0.95, 1.00] | 66/87 = 0.76 [0.65, 0.84] | 16 |
| rank 2 | 31 | 10 of 10 (4 labelled wrong before review) | 1.00 [0.69, 1.00] | 10/23 = 0.43 [0.23, 0.66] | 7 |
| rank 3 | 23 | 12 of 12 (3 labelled wrong before review) | 1.00 [0.74, 1.00] | 12/17 = 0.71 [0.44, 0.90] | 1 |
| rank 4 | 10 | 3 of 3 | 1.00 [0.29, 1.00] | 3/4 = 0.75 [0.19, 0.99] | 1 |
| rank 5 | 8 | 3 of 3 (1 labelled wrong before review) | 1.00 [0.29, 1.00] | 3/5 = 0.60 [0.15, 0.95] | 0 |
| rank 6 | 7 | 5 of 5 | 1.00 [0.48, 1.00] | 5/5 = 1.00 [0.48, 1.00] | 0 |
| rank 7 | 7 | 5 of 5 | 1.00 [0.48, 1.00] | 5/5 = 1.00 [0.48, 1.00] | 0 |
| rank 8 | 7 | 3 of 3 | 1.00 [0.29, 1.00] | 3/4 = 0.75 [0.19, 0.99] | 0 |
| rank 9 | 6 | none | — | no right name | — |
| the other 52 (fewer than 5 trials each) | 79 | 34 of 34 (2 labelled wrong before review) | 1.00 [0.90, 1.00] | 34/43 = 0.79 [0.64, 0.90] | 4 |

**Reading it.**
- Recall ranges from 0.43 to 0.76 among the three most-tried figures.
- The second figure's right names are lost to coverage (7 cannot be scored) and
  to recognition (6 score below the threshold).
- What its images carry decides much of it (section 5):
  - crowding: TAR 0.60 with several faces in the image, 0.93 with one;
  - masks: 0.29 masked, 0.88 not;
  - faces under the 20 px floor.
- Weighting every figure the same (figure-stratified, checked labels):
  precision 1.00 and TAR 0.82. Without the two most-tried figures: 65 of 65
  accepted names right (lower bound 0.955), and TAR 0.78.

**Stated plainly.** The method's accuracy depends on who is being verified. On
these trials, an accepted name is right whoever it names. How often a right name
is accepted at all depends on the person, through the faces their images carry.
Every result is reported per figure, with the figure-stratified estimate beside
the pooled one.

## 4. Widening the population: not needed

At the precision measured before this review (0.929, with the sample's 37
decisions), certifying would have needed about 333 accepted names (1,183 on
claim labels alone), and the population gives 141. The review the method
requires anyway, of the 113 unchecked accepted names, certified 0.5214 on those
141 with no widening.

For the record, here is what widening would have cost (*derived* counts from the
ledger; projected accepted names apply the yield of the time, 51%):

| Population | Images | Scorable VLM names | Projected accepted | Cost |
| --- | ---: | ---: | ---: | --- |
| Factify2 train, the claim rule (used) | 510 | 278 | 141, measured | — |
| … no "pictured" word, one figure named | 4,994 | 2,860 | about 1,450 | weaker labels: a person checks every accepted name |
| … every train image, labelled by a person alone | 19,546 | 6,590 | about 3,340 | every accepted name a person's call |
| Factify2 val / AVerImaTeC, the claim rule | 63 / 147 | 31 / 29 | about 16 / 15 | spends held-out data |
| VERITE | — | — | — | off-limits: never read for any threshold |

Raising TAR is open work. Checking the 16 unchecked names at 0.3959 is the
cheapest step.

## 5. Known limits, stated beside the calibration

- **Coverage ceiling.** 1,592 of the gallery's 4,776 people have no usable
  reference face, and 1,430 of them have no Commons reference at all.
  - Of the 193 right names, 29 cannot be scored: 17 faces under the 20 px
    floor, 11 images with no face found, and 1 person with no usable
    reference.
  - They count against TAR, never as accepts.
- **Masked faces** (measured 2026-10-27; measured, never refused).
  - 9 of the 220 faces the gallery scores are masked over nose and mouth: an
    agent's reading of the crops, not a person's check.
  - Masks cost recall, not precision. 2 of 7 masked right names are accepted
    (TAR 0.29, against 0.88 unmasked); both accepted names are right, and
    both masked wrong names are rejected.
  - Detection is cheap. On the aligned crop the recognizer embeds, the mouth
    region's skin share falling 0.6 or more below the eyes' band marked
    exactly those 9, and no other face. A face's detection score did not
    separate them.
  - The calibration now reports masked and unmasked faces as strata. A hand
    over the mouth, being skin, is not read. 5 faces are too grey or dark to
    read at all.
- **Crowded images.** TAR is 0.60 when the image holds several faces, against
  0.93 with one, at no cost in precision.
- **Lone P18 references.**
  - 685 people's only single-face reference is a lone Wikidata (P18) image,
    which no consistency check can test.
  - By title, at least 3 of those are not photographs: two postage stamps and a
    bust.
  - Title keywords mark 30 usable P18 references in all, some of them wrongly.
    That is an agent's reading of titles, not a person's check.
- **How far apart SFace puts one face.** Two official photographs a person
  confirmed show the same man score 0.167 against each other. In all, 10 pairs
  a person confirmed as one face score under the 0.3811 consistency threshold,
  at 0.167 to 0.362.
- **Names that refuse rather than guess.** 459 names that would have resolved
  to one person refuse instead (7.3% of mentions), because the word is several
  people's or only a nickname. In this run, 2 trials are unscorable as a name
  several people hold.
- **One corpus, one rule, one recognizer:** Factify2 train under the claim rule,
  concentrated on two figures, and SFace.

## 6. Reproducibility

- A second machine, with the same ledger, gallery, decisions and code
  (snapshot `e579766`), produced the server's report in every value, apart
  from the time it was written.
- The final report was written there, after the person's 113 decisions. It adds
  the look-alike contrast over 200 draws, the rule that certifies only checked
  names, and the masked-face strata. The claim-label figures are the server's.
- Two decisions of 2026-10-23 are why runs agree across machines:
  similarities are computed in float64, and the medoid of two references is
  broken by rule (the P18, else the first). Before them, float rounding could
  break a near-tie differently on each machine.
- To reproduce:
  `python scripts/gallery.py calibrate --backend sface --ledger-dir data/interim/image_text/93cd9d679720`.
  Its prerequisites are in CLAUDE.md under "Entity gallery: standing rules".
