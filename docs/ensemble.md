# The ensemble (step 8)

## Main result: what M4FC teaches does not transfer to photographs

A model fitted on M4FC learns M4FC's screenshots, and that knowledge points
the wrong way on real photographs.

- **The cause.** Many M4FC out-of-context images are screenshots of the post
  that made the claim, so the image carries the claim's own text. On M4FC,
  image-text agreement and OCR text therefore mark a false record. This was
  first seen in report B as the image side echoing the claim (`docs/consistency.md`,
  Dataset findings).
- **Four times, the same result.** Every model fitted on M4FC train does
  better on M4FC val and worse on VERITE than unfitted image-text similarity.
  On VERITE out-of-context vs true, CLIP ViT-L/14 alone scores 0.851:

  | fitted on M4FC train | M4FC val, true vs all false | VERITE, out-of-context vs true | VERITE, miscaptioned vs true |
  | --- | ---: | ---: | ---: |
  | the stacked ensemble (gate E1) | 0.746 | 0.611 | 0.496 |
  | the same, signs constrained by meaning (post hoc) | 0.720 | 0.671 | 0.562 |
  | a linear probe on OpenCLIP embeddings, photographs only | 0.812 | 0.528 | 0.524 |
  | (unfitted) the equal-weight image-text average | 0.513 | 0.831 | 0.597 |
  | (unfitted) CLIP L/14 similarity | 0.580 | 0.851 | 0.566 |

  Beside every M4FC number: Track B's text-only claim classifier scores 0.784 [0.741, 0.826] (true vs all false) and 0.819 (out-of-context vs true) on M4FC's official test. It reports no M4FC val AUROC. No system here beats it on M4FC.

- **It is in the coefficients.** The ensemble weighs BLIP-2's image-text
  similarity +0.62 *toward false*, and the absence of OCR text -0.44.
- **It survives the sign constraints.** Forcing every similarity to argue for
  the claim recovers part of VERITE (0.611 to 0.671), not most of it.
- **It survives restricting to photographs.** The probe was fitted only on the
  547 M4FC train images routed as photographs. It gains 0.21 AUROC in M4FC
  train cross-validation, and loses 0.30 on VERITE against the average it
  would replace.
- **The consequence.** On these data, M4FC train cannot select or weight a
  system for photographs. Every number below that is fitted on M4FC carries
  this caveat.

Pretrained scorers, stacked by one small model. The code is
`scripts/ensemble.py`, the pinned models are in `configs/ensemble.yaml`, and the
tests are in `tests/test_ensemble.py`. It builds on the consistency block
(`docs/consistency.md`) and changes nothing in it: the block's verdicts are one
of its inputs. Explanations still come from those verdicts alone; the ensemble
decides the flag.

## Decisions (the person's, 2026-11-10)

- **Scorers.** OpenCLIP ViT-H/14 and SigLIP so400m for image-text similarity.
  BLIP and BLIP-2 matching heads. A DeBERTa-v3 NLI cross-encoder, comparing the
  claim with the image's description and OCR, and with the retrieved
  documents. Our consistency verdicts. Track B's text-side stance.
  - Each is verified for licence, weights and pinned revision before use, and
    nothing non-commercial is used.
  - **OpenAI's CLIP is out of the ensemble.** Its model card says "Any
    deployed use case of the model - whether commercial or not - is currently
    out of scope." Track B may use CLIP ViT-L/14 only to reproduce the MUSE
    baseline, as research evaluation. The same card covers the ViT-B/32 that
    C uses as its baseline, which stays a baseline only.
- **The combiner fits on M4FC train.**
  - Factify2's source-disjoint train enters a labelled variant only.
  - VERITE is evaluation only.
  - M4FC's authors' test split is used once, in the final run, through a
    one-time `--final-test` path that records its opening and refuses a
    second.
- **scikit-learn** is used at the release Track B pins in
  `requirements-trackb.txt`, and no second pin is written. The combiner reads
  that pin at run time, and uses gradient boosting only where the installed
  release is that one.
- **Track B's stance** comes from `data/interim/textcheck/verdicts.jsonl`,
  joined on `consistency.claim_key`, with the keys claim_key, judge, verdict,
  reason_code, p_supported, p_refuted, p_nei, doc_ids and status. It is read,
  never written, and a row missing a key refuses.
- **LoRA only if the linear probe gains.** The probe comes after gate E1.
- **The server GPU is shared with Track B,** which needs about 5 GPU-hours
  over Nov 12-13. Every GPU job starts only when under 1 GB of GPU memory is
  in use.

## The scorers, pinned

Licences were read from the Hub's model-card metadata on 2026-11-10.
`fetch-models` re-reads that metadata, refuses if the revision or the licence
has changed, and checks every file's size and sha256 after download.

| scorer | repo @ revision | licence | what it gives a pair |
| --- | --- | --- | --- |
| `openclip_h14` | laion/CLIP-ViT-H-14-laion2B-s32B-b79K @ 1c2b8495 | MIT | cosine of image and claim |
| `siglip_so400m` | google/siglip-so400m-patch14-384 @ 9fdffc58 | Apache-2.0 | cosine of image and claim |
| `blip_itm_large` | Salesforce/blip-itm-large-coco @ 19502f1e | BSD-3-Clause | P(match) from the matching head; the contrastive similarity |
| `blip2_itm_g` | Salesforce/blip2-itm-vit-g @ 520bf73f | MIT | P(match); the contrastive similarity |
| `nli_deberta_large` | cross-encoder/nli-deberta-v3-large @ bab4bc71 | Apache-2.0 (trained on SNLI, CC BY-SA 4.0; MultiNLI) | P(entailment, neutral, contradiction) of the claim, given a premise |

**Excluded:**
- `openai/clip-vit-large-patch14`, for its card (above).
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, because it was
  trained on ANLI, which is CC BY-NC 4.0.

BLIP ITM large ships only a `.bin`. It is pinned by sha256, and transformers
loads it with torch's `weights_only` unpickler.

## Inputs, and the guards on them

- **What is scored.** Every record C scores: 2,524 M4FC train, 497 val, and
  VERITE's 877. The variant also gets Factify2's source-disjoint train:
  23,315 rows. 502 more had no usable claim image.
  - That makes 27,213 image-claim pairs over 20,744 distinct images.
  - NLI pairs number 62,519: the claim against the image's description
    (26,840), its OCR (20,605) and M4FC's retrieved documents (15,074).
  - VERITE has no claim date, so no retrieval. Factify2 was never retrieved.
- **What a model sees.**
  - The scorer inputs carry neutral keys and text only: the claim, the OCR
    and description as the predictor saw them, and each retrieved title and
    excerpt.
  - Each text is refused if it holds its record's id in a form its own
    content does not explain, as B's and C's prompts were. Any id shape is
    refused outright.
  - Images are found by sha256 through a separate locator, and the bytes are
    checked against that sha256 before a model sees them.
  - The labels never leave the laptop.

## The combiner

- **Target.** True (0) against false (1). On M4FC, false is out-of-context,
  manipulated and AI-generated, as in M4FC's verdict prediction; mixture and
  unverified are not scored. VERITE's out-of-context and miscaptioned are
  false.
- **Form.** Chosen by grouped 5-fold cross-validation within M4FC train.
  Records sharing an image or a claim stay in one fold.
  - The candidates are an L2 logistic regression (L2 0.1, 1 or 10, classes
    balanced) and, at Track B's scikit-learn pin, histogram gradient
    boosting (depth 2 or 3, 100 or 300 iterations).
  - The threshold maximises balanced accuracy on the out-of-fold scores.
- **Missing values.** A feature group some records lack (OCR, documents,
  consistency, stance) carries a missing flag. The regression reads a
  missing value as 0 beside its flag; boosting reads NaN.
- **The ablation.** The chosen form is refitted without each group. Each
  single score is also reported in a direction fixed by what it measures,
  with nothing fitted.
- **The Factify2 variant.** The same form, fitted on M4FC train plus
  Factify2's train. Its label is reason code `none` against the rest:
  whether a document supports the claim, never whether the image fits it.

## Published numbers to compare with (fetched 2026-11-10)

None of these is like for like with ours:
- They train on NewsCLIPpings or on generated data, which we do not hold.
  NewsCLIPpings' images are VisualNews's, behind a request form.
- They score all of VERITE. We score its 877 evaluation records.
- Their evidence, where they use any, is not leak-filtered.
- They report accuracy; our primary figure is AUROC, with accuracy at the
  train threshold beside it.

| benchmark | task | best published | method, training data | source |
| --- | --- | ---: | --- | --- |
| VERITE | true vs out-of-context, accuracy | 76.3 | LAMAR (E2E, Gate), NewsCLIPpings, CLIP ViT-L/14 | [arXiv 2504.06010](https://arxiv.org/html/2504.06010v3) |
| VERITE | true vs miscaptioned, accuracy | 66.0 | LAMAR (E2E, Gate), generated data (LLaVA-𝒟3) | same |
| VERITE | multiclass, accuracy | 54.9 | LAMAR (E2E, Gate) | same |
| VERITE | true vs out-of-context, accuracy | 76.9 | RED-DOT-SSL, NewsCLIPings+, with Google-retrieved evidence | [arXiv 2311.09939](https://arxiv.org/html/2311.09939v2) |
| VERITE | true vs out-of-context / miscaptioned / multiclass | 74.1 / 63.2 / 52.0 | AITR, as LAMAR's table reports it | [arXiv 2504.06010](https://arxiv.org/html/2504.06010v3) |
| VERITE | true vs out-of-context | 72.0 | MUSE, NewsCLIPpings, as LAMAR's table reports it | same; MUSE is from [arXiv 2407.13488](https://arxiv.org/pdf/2407.13488) |
| M4FC | verdict prediction (true vs false), macro F1, test (1,478) | 89.16 | GPT-4o-mini, with intermediate task outputs and reverse-image-search evidence | [arXiv 2510.23508](https://arxiv.org/html/2510.23508) |
| M4FC | verdict prediction, macro F1 | 86.63 / 84.14 | DEFAME / Qwen2.5-VL-7B | same |

- M4FC's verdict-prediction figures are English-only and imbalanced. The
  paper says no model exceeds 80% F1 in its balanced settings.
- M4FC has no separate out-of-context task.

## The scorer runs (2026-11-10)

- **Weights.** `fetch-models` ran on the server: 5 models, 14.61 GiB, every
  file's size and sha256 checked against the Hub's metadata.
- **A missing dependency.** SigLIP's tokenizer needs SentencePiece, which
  neither machine had, and transformers 5.17 has no fast SigLIP tokenizer.
  `sentencepiece==0.2.1` (Apache-2.0, released 2025-08-12) is pinned in
  `requirements.txt` and installed on both.
- **A pilot of 64 rows per scorer.** It found the SentencePiece gap, and
  BLIP-2's processor quirk: blip2-itm-vit-g's processor sets no
  `num_query_tokens`, so its text and images are prepared apart.
- **The first full run was set aside.** OpenCLIP's recheck in a fresh
  process differed by 1.15e-3 in cosine, beyond the 1e-3 bar fixed before
  the run. The cause was TF32, on by default for cuDNN's convolutions.
  - The bar was not moved. Scoring now runs in full float32: TF32 is off for
    matmul and cuDNN, and cuDNN is deterministic with no benchmarking.
  - That run's ledgers are kept aside on the server
    (`scores_tf32_run/`), not used.
- **The full run** on the server's RTX 4060 Ti. Each scorer started only
  while under 1 GB of GPU memory was in use, the GPU being shared with
  Track B. Each was rechecked on 3 rows in a fresh process.

  | scorer | config | rows | errors | seconds per row | recheck: largest difference |
  | --- | --- | ---: | ---: | ---: | ---: |
  | `openclip_h14` | 12e4bb8c9900 | 27,213 pairs (20,744 images) | 0 | 0.039 per image | 0 |
  | `siglip_so400m` | f3f8930e9ab2 | 27,213 pairs | 0 | 0.081 per image | 0 |
  | `blip_itm_large` | 8a91747ff75a | 27,213 pairs | 0 | 0.104 | 0 |
  | `blip2_itm_g` | ee44e77f36bc | 27,213 pairs | 0 | 0.111 | 1.0e-6 |
  | `nli_deberta_large` | 80d3b455b6b9 | 62,519 pairs | 0 | 0.025 | 1.0e-6 |

## Gate E1 (2026-11-10)

The record is `data/reports/ensemble_e1.json`, with a local `.md` beside it.

- **What the fit had.** Track B's `requirements-trackb.txt` did not exist
  yet, so the candidates were the three logistic regressions only (no
  gradient boosting). Track B's stance file did not exist either, so no
  stance group.
- **The bar.** Track B found that CLIP ViT-L/14 similarity alone scores
  0.852 AUROC on VERITE out-of-context vs true.
  - Its cosines are recomputed here from Track B's embeddings, read and never
    written. They join 3,894 of 3,898 records and give 0.851.
  - Every comparison with it is paired, on the same records.

**The chosen form.** An L2 logistic regression (L2 10). Its out-of-fold AUROC
on M4FC train, true against false, is 0.701; the three candidates were within
0.0004. The threshold is 0.509 (out-of-fold balanced accuracy 0.662).

**It beats CLIP L/14 on M4FC val and loses to it on VERITE.**

| | ensemble | CLIP L/14 | Δ [95%] | CLIP B/32 | fused (Gate C) |
| --- | ---: | ---: | --- | ---: | ---: |
| M4FC val, out-of-context vs true | 0.763 | 0.597 | +0.166 [+0.041, +0.289] | 0.512 | 0.507 |
| M4FC val, true vs all false | 0.746 | 0.580 | +0.166 [+0.039, +0.299] | 0.505 | 0.495 |
| VERITE, out-of-context vs true | 0.611 | **0.851** | -0.240 [-0.285, -0.192] | 0.753 | 0.758 |
| VERITE, miscaptioned vs true | 0.496 | 0.566 | -0.070 [-0.109, -0.033] | 0.575 | 0.577 |

Beside every M4FC number: Track B's text-only claim classifier scores 0.784 [0.741, 0.826] (true vs all false) and 0.819 (out-of-context vs true) on M4FC's official test. It reports no M4FC val AUROC. No system here beats it on M4FC.

- The transfer gap (M4FC val minus VERITE, out-of-context vs true) is 0.152.
- **At the train threshold:**
  - VERITE accuracy is 0.57 for out-of-context vs true and 0.49 for
    miscaptioned vs true. The best published figures are 76.3 and 66.0, with
    the caveats above.
  - M4FC val macro F1, true vs false, is 0.543.
- **The Factify2 variant** (labelled; not the system) scores 0.705 on M4FC
  val, and 0.692 and 0.565 on VERITE. It is still below CLIP L/14.
- **Why it does not transfer.** The regression learns M4FC's screenshots.
  Many M4FC out-of-context images are screenshots of the post that made the
  claim, so they carry the claim's own text (Dataset findings in
  `docs/consistency.md`). On M4FC, then, image-text agreement and OCR text
  mark a false record. The largest coefficients:
  - BLIP-2's contrastive similarity, +0.618 toward false;
  - its P(match), -0.534;
  - having no OCR text, -0.442;
  - a consistent event, +0.360.

  Single scores on M4FC val agree: BLIP's P(match) scores 0.423, below chance
  in its own direction. On VERITE's photographs the same signs mean the
  opposite.
- **The ablation.** No single group's removal rescues VERITE. The best
  refits score 0.624 (without BLIP) and 0.623 (without consistency), against
  0.611. The weighting as a whole is M4FC's.
- **Single scores** (direction fixed, nothing fitted; each paired against
  CLIP L/14). Reported, not chosen: they are read off VERITE.
  - **VERITE out-of-context vs true:** none beats L/14.
    - OpenCLIP H/14 0.829, Δ -0.022 [-0.045, +0.001];
    - SigLIP 0.825, Δ -0.027 [-0.052, -0.001];
    - BLIP-2 P(match) 0.780;
    - BLIP P(match) 0.733.
  - **VERITE miscaptioned vs true:** two beat L/14.
    - SigLIP 0.606, Δ +0.040 [+0.007, +0.072];
    - BLIP-2's contrastive similarity 0.607, Δ +0.041 [+0.001, +0.080].
  - **M4FC val:** none beats L/14. The closest are document NLI (entailment)
    at 0.646 and the consistency score at 0.604.
- **What gate E1 decides is the person's.** Any change to the combiner now
  (a sign constraint by meaning, a fixed average of the similarity scores,
  another training corpus) would be chosen after VERITE was seen, and must be
  reported as such.

## The routed system (post hoc, 2026-11-10)

The person's E1 decision, taken after VERITE was seen at gate E1, is labelled
**post hoc** wherever it is reported. The route rule and every fitted number
come from M4FC train alone, and VERITE fits nothing. The record is
`data/reports/ensemble_routed.json`, with a local `.md` beside it.

- **The rule.** A screenshot layout (the image model's description calls the
  image a screenshot in its first sentence), or OCR text covering at least
  *t* of the image, goes to the M4FC branch: the ensemble of gate E1. Every
  other photograph goes to the photo branch: an unfitted, equal-weight
  average of OpenCLIP H/14, SigLIP, BLIP ITM's P(match) and BLIP-2's
  P(match), each rank-normalised on M4FC train.
- **Where the scores meet.** Each branch's score is a percentile on its own
  M4FC-train scale, and each branch has its own train threshold (balanced
  accuracy).
- **Fitting *t*.**
  - The first objective tried was the routed system's AUROC on M4FC train.
    It chose *t* = 0: every image to the M4FC branch, since on M4FC train
    that ensemble beats the average on photographs too. That leaves no
    photograph to route, so it was not used, and it is reported beside.
  - *t* is instead the OCR share that best tells a screenshot layout from
    the rest on M4FC train: *t* = 0.041 (balanced accuracy 0.727).
  - This change was made on what train alone showed. It is still part of a
    post-hoc system.
- **The routes.**
  - M4FC train: 1,962 to the M4FC branch, 547 to the photo branch.
  - M4FC val: 416 and 81.
  - VERITE: 96 and 781.
- **Misroutes on M4FC train: 9.8%.** A misroute is a record whose branch
  flags it wrongly while the other branch flags it rightly. That is 8.4% of
  the M4FC branch's records and 15.0% of the photo branch's.
- **The linear probe**, on the photo branch only (the person's rule: kept
  only if it gains in M4FC train cross-validation).
  - It is an L2 logistic regression on the elementwise product of OpenCLIP
    H/14's image and claim embeddings, over the 547 photo-routed train
    records.
  - Grouped 5-fold out-of-fold AUROC: 0.781, against the average's 0.570 on
    the same records. Δ +0.211 [+0.128, +0.287], so by the rule it is
    **adopted**.
  - It is the main result's third row: it does not transfer.
- **Track B's stance** did not exist when the system was frozen, so the
  M4FC branch has no stance feature, and no "with and without" comparison
  could be run.
  - The final test runs the system exactly as frozen (the person's decision,
    2026-11-11).
  - Stance is added afterwards as a separate variant, reported on M4FC val
    only and never on the test set.
- **Results.** CLIP L/14 is beside every number: `clip_l14` from Track B's
  embeddings, paired. This pipeline's own run of the same weights scores
  0.852 on VERITE out-of-context.

  | | M4FC val, out-of-context vs true | M4FC val, true vs all false | VERITE, out-of-context vs true | VERITE, miscaptioned vs true |
  | --- | ---: | ---: | ---: | ---: |
  | **routed (chosen: with the probe)** | 0.798 | 0.782 | 0.541 | 0.525 |
  | routed without the probe | 0.662 | 0.634 | 0.809 | 0.594 |
  | the M4FC branch alone | 0.763 | 0.746 | 0.611 | 0.496 |
  | the photo average alone | 0.526 | 0.513 | 0.831 | 0.597 |
  | the probe alone | 0.824 | 0.812 | 0.528 | 0.524 |
  | the sign-constrained refit alone (ablation) | 0.730 | 0.720 | 0.671 | 0.562 |
  | **CLIP L/14** | 0.597 | 0.580 | **0.851** | 0.566 |

  Beside every M4FC number: Track B's text-only claim classifier scores 0.784 [0.741, 0.826] (true vs all false) and 0.819 (out-of-context vs true) on M4FC's official test. It reports no M4FC val AUROC. No system here beats it on M4FC.

  Paired against L/14, the chosen system:
  - M4FC val, true vs all false: +0.203 [+0.090, +0.329];
  - VERITE out-of-context: -0.310 [-0.350, -0.269];
  - VERITE miscaptioned: -0.041 [-0.082, +0.001].

  Nothing beats L/14 on VERITE out-of-context. The photo average alone comes
  closest: -0.020 [-0.043, +0.003]. On miscaptioned it beats L/14: +0.031
  [+0.002, +0.059].
- **Frozen.** `configs/ensemble_system.json`, system sha256
  `5eb6f48b78f0b5db86b1a101d5593195b84dc1c8af503c1e95fa8f62c30ea237` (frozen from b062acf).
  - Before the test opened, it declares its ablations: routed without the
    probe, the M4FC branch alone, the photo average alone and the probe
    alone.
  - Re-scored from that file alone, VERITE gives the report's figures
    exactly, on the same records.

## The final test: M4FC's official test, scored once (2026-11-11)

The frozen system (`5eb6f48b`) was run exactly as frozen. Its declared
ablations are reported beside it. Stance was not in it; it enters later, as a
val-only variant.

- **Opened** at 09:19 UTC, under the code commit bcb8e1f.
  - 1,479 records had a usable image; 10 images are dead (404s and verdict
    placeholders).
  - Scored: 1,468. Verdict prediction leaves out mixture and unverified.
    The classes are out-of-context 833, manipulated 379, AI-generated 158 and
    true 98.
  - **Scored once**, at 19:03 UTC. `final_test/scored.json` refuses a second
    run, and the report's sha256 is recorded there.
- **How it ran.** Every stage ran through the live path, under the evaluated
  configurations: the image model, the gallery, retrieval bounded by each
  record's own evidence date, extraction, prediction, the judge, the
  verdicts, then the five scorers and CLIP L/14.
  - Every scorer's recheck in a fresh process was within 1e-6.
  - The image model returned 1,454 rows ok and 25 partial.
  - One retrieved excerpt held a web page's asset path, an id shape. That
    premise was left out and counted.
  - A Windows update disabled WSL mid-run; the jobs on the server continued.
- **The routes.** 1,153 records went to the M4FC branch and 326 to the photo
  branch. Of the 98 true records, 65 went to photos.

| | true vs all false | out-of-context vs true |
| --- | ---: | ---: |
| **routed (the frozen system)** | **0.669** | **0.694** |
| routed without the probe (declared ablation) | 0.598 | 0.627 |
| the M4FC branch alone (declared) | 0.739 | 0.754 |
| the photo average alone (declared) | 0.499 | 0.555 |
| the probe alone (declared) | 0.708 | 0.773 |
| CLIP L/14 (this pipeline's run) | 0.589 | 0.635 |
| CLIP B/32 | 0.534 | 0.577 |
| **Track B's text-only claim classifier** | **0.784** | **0.819** |
| Track B's MUSE (image, text and evidence; MLP) | 0.670 | 0.704 |

- **Paired against CLIP L/14, the frozen system:**
  - true vs all false: **+0.080 [+0.006, +0.152]**, which beats it;
  - out-of-context vs true: +0.059 [-0.015, +0.136], which does not.
- **Against CLIP B/32:** +0.135 [+0.051, +0.220] and +0.117 [+0.035, +0.207].
- **At its branch thresholds:**
  - macro F1 (true vs false) is 0.494, and accuracy 0.674;
  - flag rates by class are true 0.42, out-of-context 0.72, manipulated 0.65
    and AI-generated 0.55.
- **Against the published figures,** M4FC's verdict prediction scores macro
  F1 84.14 to 89.16. Those use web reverse-image-search evidence and an
  imbalanced English-only setting, so they are not like for like; this
  system's evidence excludes fact-checkers' pages.
- **The test agrees with the main result,** from the other side. On M4FC's
  own test, the models fitted on M4FC lead: the M4FC branch alone scores
  0.739, and the probe alone 0.708. The unfitted photo average is at chance
  (0.499), the reverse of VERITE, where it is the only system near L/14.
  M4FC and VERITE reward opposite things.

## Demo system v1.1 (post hoc, 2026-11-12)

- **What it is.** The frozen system without its photo-branch probe, so the
  photo branch is the unfitted average. It is
  `configs/ensemble_system_v1_1.json`, with its own sha256 (`066e5554`), and
  it records the frozen system it comes from (`5eb6f48b`).
- **Post hoc.** v1.1 was decided after VERITE was seen at gate E1, and after
  the frozen system's test was scored.
- **v1.1 has no test number.** The test report's "routed without the probe"
  row was an ablation of the frozen system, declared before the test opened.
  v1.1 was chosen after that row was seen, so it is not a test result for
  v1.1.
- **On M4FC val and VERITE** (`data/reports/ensemble_v1_1.json`), CLIP L/14
  beside it, paired:

  | | M4FC val, out-of-context vs true | M4FC val, true vs all false | VERITE, out-of-context vs true | VERITE, miscaptioned vs true |
  | --- | ---: | ---: | ---: | ---: |
  | **v1.1** | 0.662 | 0.634 | 0.809 | 0.594 |
  | the frozen system | 0.798 | 0.782 | 0.541 | 0.525 |
  | CLIP L/14 | 0.597 | 0.580 | 0.851 | 0.566 |

  Beside every M4FC number: Track B's text-only claim classifier scores 0.784 [0.741, 0.826] (true vs all false) and 0.819 (out-of-context vs true) on M4FC's official test. It reports no M4FC val AUROC. No system here beats it on M4FC.

  - **v1.1 against L/14:**
    - VERITE out-of-context: -0.043 [-0.070, -0.016];
    - VERITE miscaptioned: +0.028 [-0.000, +0.058];
    - M4FC val: +0.064 [-0.027, +0.147].
  - **v1.1 against the frozen system:**
    - VERITE: +0.267 [+0.227, +0.306] and +0.069 [+0.031, +0.105];
    - M4FC val: -0.137 [-0.230, -0.043].
- **The demo shows v1.1's flag** (`demo-v11`, and the live path). The page
  says which branch decided it, as a template: the M4FC branch for an image
  that carries text, the photo branch's similarity alone for a photograph.
  - The reasons still come from the consistency verdicts.
  - The Gate C fused flag no longer shows: it read OpenAI's CLIP B/32.
- **The text side on the page.** Where Track B's Qwen verdict cites
  documents, the page adds one template sentence: its verdict, and each
  cited document's site and date. It is shown as evidence and not counted in
  the flag. NLI-B never speaks on a page.

## Text-side stance: a val-only variant (2026-11-12)

- **The variant.** Track B's `verdicts.jsonl` holds two judges per claim.
  - **NLI-B** (the DeBERTa cross-encoder over the retrieved documents) is a
    feature only: its three probabilities, with a missing flag.
  - **Qwen's cited verdicts** serve explanations only.
- **The fit.** The M4FC branch is refitted in the frozen form (L2 10) with
  the stance group, on M4FC train. It is reported on M4FC val only, never on
  VERITE or the test (`data/reports/ensemble_stance_val.json`).
- **It adds nothing on M4FC val.** Coverage is 497 of 497.
  - Beside every M4FC number: Track B's text-only claim classifier scores 0.784 [0.741, 0.826] (true vs all false) and 0.819 (out-of-context vs true) on M4FC's official test. It reports no M4FC val AUROC. No system here beats it on M4FC.
  - v1.1 with stance scores 0.660 against 0.662: Δ -0.001 [-0.017, +0.013]
    (out-of-context vs true), and -0.003 [-0.019, +0.012] (true vs all
    false).
  - The M4FC branch alone moves by -0.002 and -0.004.
  - Its standardised coefficients are small: supported +0.16, refuted
    -0.13, not enough evidence +0.04.

## The live path under v1.1, tested end to end (2026-11-12)

- **The run.** `scripts/live_demo.sh` ran the 4 test requests from scratch:
  3 VERITE and 1 M4FC train. Their server-side outputs were cleared first.
  - Every stage ran: the image model and the text model on the server, the
    gallery, retrieval and the verdicts on the laptop, then the five scorers
    on the server.
  - v1.1's flag and page were written on the laptop.
- **Identical to the batch.**
  - `consistency.py live check`: every stage identical (CLIP within 1e-4).
  - `ensemble.py live check`: the features within 1e-6, and v1.1's score,
    branch and flag identical, for all 4. That is 3 requests on the photo
    branch and 1 on the M4FC branch.
- **The driver.** It refuses unless the server runs the laptop's committed
  HEAD, and runs each scorer as one GPU job, refused while the GPU is in use.

## The test a second time, for comparison only (2026-11-12)

The person approved this on 2026-11-12. It was M4FC's official test labels'
**second read**, for paired comparison only; nothing was tuned or changed.

- **The replay.** The frozen system's per-record test outputs were replayed
  from the stored inputs by the scoring path's own code, with no label read
  (`final-test replay`).
  - The first run had stored aggregates only.
  - Before any label was read, the replay matched the sealed report's
    label-free totals: 1,145 and 323 records by branch, and 973 flagged.
- **The read.** It was logged before it happened (`final_test/second_read.json`),
  and it refuses a third.
  - The replay reproduced the sealed report to four decimals: every AUROC
    and Δ, the record count and the macro F1. Only then were the comparisons
    computed.
  - The record is `data/reports/ensemble_test_comparison.json`.
- **The frozen system, paired against Track B's baselines** on the same
  1,468 records:

  | | frozen | text-only claim classifier | Δ [95%] | MUSE (best) | Δ [95%] | CLIP L/14 | Δ [95%] |
  | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
  | true vs all false | 0.669 | 0.784 | **-0.115 [-0.180, -0.053]** | 0.670 | -0.002 [-0.071, +0.069] | 0.588 | +0.080 [+0.007, +0.153] |
  | out-of-context vs true | 0.694 | 0.819 | **-0.125 [-0.184, -0.060]** | 0.704 | -0.010 [-0.074, +0.059] | 0.635 | +0.059 [-0.016, +0.135] |

- **Our system does not beat the text-only claim classifier**, which reads
  the claim alone. It is clearly worse, on both pairs. It ties MUSE. It
  beats CLIP L/14 on true vs all false only.

## Demo pages under v1.1 (2026-11-12)

The pages are local; none is published.

- **The approved picks**, rebuilt under v1.1. Five keep their categories.
  - The former "miss" (VERITE, the Trump children) is flagged by v1.1, so it
    is a miss no longer.
  - It was replaced by the person's approved seeded draw: a VERITE
    out-of-context record (an Imran Khan rally in Lahore). v1.1 does not
    flag it: 0.45 against the photo branch's 0.50.
- **Indian sports cases** (Prof. Maitra's example), each run through the live
  path. No record we hold mentions the Asiad. These are the records where the
  gallery accepts a face the claim names:

  | record | dataset label | gallery | v1.1 |
  | --- | --- | --- | --- |
  | Deepa Malik, Paralympian, joining the BJP | Factify2 train, Insufficient_Multimodal | Deepa Malik 0.58 | flagged, 0.62, photo branch |
  | Virat Kohli on Bangladesh | Factify2 train, Insufficient_Text | Virat Kohli 0.73 | flagged, 0.48, M4FC branch |
  | Virat Kohli on India's jersey | Factify2 train, Insufficient_Multimodal | Virat Kohli 0.78 | flagged, 0.83, M4FC branch |
  | Rohit Sharma's 26th ODI century | Factify2 val, Support_Text | Rohit Sharma 0.77 | flagged, 0.39, M4FC branch |
  | MS Dhoni "converted to Buddhism" | M4FC train, out-of-context: **in-sample** | Dhoni 0.53 | not flagged, 0.20, M4FC branch |

  - Factify2's labels say whether a document supports the claim, not whether
    the image fits it, so these flags are not scored right or wrong.
  - Factify2 carries no date, so no documents were retrieved for those four
    (the as-of rule).
  - The Dhoni page says it is in-sample: the M4FC branch was fitted on M4FC
    train.
