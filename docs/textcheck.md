# Track B: the text-side check and the paper's baselines

Track B gives the 30 November system its text side: whether what was published
before a claim's date supports the claim, refutes it, or says nothing. It is
the system's source of the reason codes `content_refuted` and
`content_unverified`. Beside it sit the baselines the paper needs, and one
collector that prints every required number from every block.

- Code: `scripts/textcheck.py`, `scripts/baselines.py`, `scripts/eval_all.py`;
  tests in `tests/test_textcheck.py`, `tests/test_baselines.py`,
  `tests/test_eval_all.py`.
- Built on the local branch `track-b`, in a worktree beside the consistency
  block's session, so neither moves the other's HEAD or trips its
  `live_demo.sh` check.
- Ledgers live in `data/interim/textcheck/` (ignored). Reports are
  `data/reports/textcheck.json`, `baselines.json` and `eval_all.json`:
  aggregates only, refused if they hold an id (`consistency.public_report`).

This file grows with the work. A decision is the person's unless it says
otherwise.

## Decisions (2026-11-10)

- **Where retrieval and the judges are scored:** AVeriTeC dev (500), M4FC val
  (500) and AVerImaTeC (945, evaluation-only).
  - MOCHEG and LIAR carry no claim date. `configs/as_of.yaml` gives them no
    retrieval, and LIAR is also unmapped. They get the text-only classifier only.
- **NLI: the system's model is B**, `cross-encoder/nli-deberta-v3-large`
  (Apache-2.0).
  - Model A, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, is
    trained partly on ANLI (CC BY-NC 4.0). It runs and is reported as a
    contrast only, and never enters the system or the ensemble.
- **scikit-learn** (1.8.0, with joblib 1.5.2 and threadpoolctl 3.6.0) is
  installed in the laptop venv only, recorded in `requirements-trackb.txt`.
  - It serves MUSE's tree and forest, so they match MUSE's code.
  - The text-only classifier is the repo's own (`m4fc_audit.Tfidf`, `fit_logreg`).
- **The ensemble's input** is `data/interim/textcheck/verdicts.jsonl`, the
  contract below.

## The ensemble's file: `verdicts.jsonl`

The consistency block's ensemble reads it, joined on `consistency.claim_key`.

- **Keys.** Each row has exactly these keys, in this order, and no other:
  `claim_key, judge, verdict, reason_code, p_supported, p_refuted, p_nei, doc_ids, status`.
- **Uniqueness.** There is one row per (claim_key, judge). `judge` is `qwen`
  or `nli`, and `nli` is model B. Model A, the no-documents probe and the
  no-enrich ablation never appear.
- **verdict and reason_code:**

  | verdict | reason_code |
  | --- | --- |
  | `supported` | `none` (adjudicated true) |
  | `refuted` | `content_refuted` |
  | `not_enough_evidence` | `content_unverified` |
  | null | null |

- **doc_ids** are the evidence store's ids:
  - for supported or refuted, the documents the verdict rests on: Qwen's
    citations, or NLI's decisive document;
  - for not enough evidence, every document the judge was shown.
- **status:**

  | status | what it means |
  | --- | --- |
  | `ok` | a verdict in format |
  | `partial` | the verdict stands; a citation is not one of the prompt's documents, or the documents line is missing |
  | `uncited` | supported or refuted, citing nothing |
  | `off_format` | no verdict: label mass under 0.5, or an answer other than the three |
  | `no_documents` | nothing in the store before the bound; not enough evidence, no judge called, no probabilities |
  | `no_bound` | the claim has no date (VERITE): retrieval never ran, and every value is null |

- **Rows:** M4FC train and val (per claim, the unit with the earliest bound),
  and VERITE's 877 evaluation rows (621 distinct claims) as `no_bound`.
- **NLI probabilities** are the decisive document's three. Its verdict is
  not enough evidence when its larger of entailment and contradiction is
  under the threshold fitted on M4FC train.
- **When it is written:** once, as soon as the Qwen and NLI-B verdicts exist,
  with a sidecar `verdicts.meta.json` (counts, config ids, model B's revision,
  sha256). A later rewrite is announced to the person and the other session.

## The text-side check

- **Query units.** A query unit is one (claim, bound).
  - Records sharing a claim's text and bound share its evidence, and every
    record's evidence is strictly before its own bound.
  - **M4FC train+val:** 3,021 units, 3,015 claims; 5 claims carry two bounds
    a few days apart.
  - **AVeriTeC train+dev:** 3,499 units, 3,485 claims; 105 train units have no
    date and get no evidence.
  - **AVerImaTeC:** 944 units.
- **Retrieval** is the consistency block's recipe, imported unchanged:
  `retrieve.search_with_terms` (both leak rules in SQL), k=10, the densest
  600-character excerpt, and the top five distinct by title and excerpt.
  - The imported functions' source is hashed into the ledger's config.
  - Each hit also records its store source: ccnews, wikinews or enrich.
  - `--check` runs the consistency block's strict twin of the statement, then
    counts id shapes and sources. On M4FC it also compares against the
    consistency block's own ledger.
- **What a model sees** (`ClaimView`): the claim, and its documents' titles,
  dates, hosts and excerpts, numbered [1] to [5]. Never an id, a path, a URL, a
  label, a split or the claim's bound.
- **The Qwen judge.** It runs on the pinned Qwen2.5-VL-7B-Instruct (NF4),
  text-only. The exemplars are six invented ones, two of each answer
  (`configs/textcheck_fewshot.yaml`).
  - **One greedy generate call** produces the answer:
    - the first step's logits give P(answer) over the answers' first tokens
      (supported / ref / not);
    - that step is steered to the most probable answer, and recorded as forced
      when the free argmax was another token;
    - every later token is the checked argmax.
  - **The answer is read strictly:** line one an answer as written, line two
    "Documents:" with numbers the prompt gave.
  - `nodocs` asks the claim alone. It probes what the model already knows, and
    its training likely postdates these fact-checks.
- **NLI.** Each pair is a document's title and excerpt (premise) against the
  claim (hypothesis), in fp32.
  - Labels are read by name: model A's logits are ordered entailment, neutral,
    contradiction, and model B's contradiction, entailment, neutral.
  - A unit's verdict rests on its decisive document, at a threshold fitted by
    macro-F1 on the corpus's train units. AVerImaTeC takes AVeriTeC's.

## Baselines

- **Hostname-only, Factify2's source-disjoint test.**
  - **Reproduced** from the data and asserted equal to
    `factify2_domain_only_final.json`, which no script produced before:
    - source-disjoint: 0.1266, majority 0.5686, unseen 0.5119, 13,045 pairs;
    - random 70/15/15: 0.4027.
  - **Per class, with intervals that resample records:**

    | Unit | Accuracy | Comparator |
    | --- | --- | --- |
    | by (record, host) pair | 0.1266 [0.1206, 0.1328] | pair majority 0.5686 (Refute) |
    | by record (claim image's host) | 0.1358 [0.1294, 0.1426] | record majority 0.5443 |

- **Text-only claim classifier** (m4fc_audit's TF-IDF + logistic regression,
  class-balanced, C=1, nothing tuned).
  - It reads the claim text only: MOCHEG's `evidence_text` is the
    fact-checker's own text, and AVeriTeC's is the justification.
  - Results, 2026-11-10:

    | Scored on | Rows | Accuracy | Macro-F1 | Applicable majority |
    | --- | ---: | --- | --- | --- |
    | AVeriTeC dev, three-way | 462 | 0.645 [0.600, 0.691] | 0.450 [0.393, 0.500] | 0.660 |
    | AVeriTeC dev, four-way | 500 | 0.564 [0.518, 0.608] | 0.356 | 0.610 |
    | AVerImaTeC (AVeriTeC's model) | 937 | 0.882 | 0.345 | 0.956 |
    | M4FC val | 497 | 0.912 [0.885, 0.936] | 0.445 | 0.936 |
    | MOCHEG test | 2,278 | 0.466 [0.445, 0.487] | 0.432 | 0.395 |
    | LIAR test, six-way | 1,283 | 0.249 [0.225, 0.271] | 0.247 | 0.208 |
    | Factify2 source-disjoint test | 10,325 | 0.193, per class in the report | — | 0.544 |

  - **Factify2:** the split's train holds 15 usable Refute records, so Refute's
    F1 is near 0. Its per-class F1 is in `baselines.json`, beside the
    hostname baseline (CLAUDE.md's Factify2 rules).
  - **M4FC:** the M4FC audit's published figure (0.9279 on 2,558/499 rows) was
    computed on the raw verdict-task rows, with true and false only. This one
    is on the processed usable rows (2,524/497), with NEI. The populations
    differ, so neither reproduces the other; both stand.
- **MUSE, adapted** (github.com/stevejpapad/outcontext-misinfo-progress,
  Apache-2.0; WACV 2025). Reimplemented from its description; no code is copied.
  - **What differs from MUSE, stated as an adaptation:**
    - it trains on M4FC train, true (164) against out-of-context (1,836),
      because NewsCLIPpings is not held;
    - VERITE has no date, so it gets the image-caption similarity only;
    - M4FC's evidence is the text side's dated retrieval, its top five titles,
      so two of MUSE's five evidence similarities exist;
    - there is no evidence image;
    - class weights are balanced.
  - **Image-only results** (2026-11-10, CLIP ViT-L/14, AUROC with
    cluster-bootstrap intervals):

    | Score | M4FC val, OOC vs true | VERITE, OOC vs true | VERITE, miscaptioned vs true |
    | --- | --- | --- | --- |
    | CLIP L/14 cosine alone | 0.597 [0.499, 0.704] | 0.852 [0.824, 0.878] | 0.566 [0.528, 0.603] |
    | MUSE-DT | 0.551 | 0.719 | 0.552 |
    | MUSE-RF | 0.420 | 0.592 | 0.530 |
    | MUSE-MLP | 0.597 | 0.852 | 0.566 |

  - On one feature the trees lose the ranking, and the MLP is a monotone
    function of the cosine.
  - CLIP L/14 alone is far stronger on VERITE's out-of-context pairs than
    report C's CLIP B/32 (0.753).
  - The evidence variant waits for M4FC's retrieval.

## Models and licences (read at source, 2026-11-10)

`docs/licences.md` belongs to the data layer. These entries are recorded here
for it to take.

- **cross-encoder/nli-deberta-v3-large @ bab4bc7178836f731dcfd18c06ca9def0a137712**
  - Hub `cardData.license`: `apache-2.0`.
  - model.safetensors is 1,740,312,740 bytes, sha256 01cb91b0…063c.
  - Trained on SNLI and MultiNLI. SNLI's dataset card: "licensed under a
    Creative Commons Attribution-ShareAlike 4.0 International License".
    MultiNLI's card lists `cc-by-3.0, cc-by-sa-3.0, mit, other` ("Open Portion
    of the American National Corpus"); its prose licence section was not read
    verbatim.
- **MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli @
  b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7**: a contrast only.
  - Hub `cardData.license`: `mit`. model.safetensors is 870,182,606 bytes,
    sha256 c03cd208…524b.
  - Trained on MultiNLI, FEVER-NLI, ANLI, LingNLI and WANLI.
  - **ANLI:** "ANLI is licensed under Creative Commons-Non Commercial 4.0."
    Its LICENSE is "Attribution-NonCommercial 4.0 International".
  - **WANLI:** "License: cc-by-4.0".
  - **FEVER:** "under the license terms described on the applicable Wikipedia
    article pages, or, where Wikipedia license terms are unavailable, under the
    Creative Commons Attribution-ShareAlike License (version 3.0)".
  - **LingNLI:** no licence stated.
- **openai/clip-vit-large-patch14 @ 32bd64288804d66eefd0ccbe215aa642df71cc41**
  - The card states no licence. The code's LICENSE (openai/CLIP) is "MIT
    License", "Copyright (c) 2021 OpenAI".
  - The card: "Any deployed use case of the model - whether commercial or not
    - is currently out of scope." A baseline is a research use.
  - model.safetensors is 1,710,540,580 bytes, sha256 a2bf730a…6dcb.
- **scikit-learn 1.8.0:** `BSD-3-Clause` (installed metadata; COPYING: "BSD
  3-Clause License").
- **joblib 1.5.2:** "BSD 3-Clause".
- **threadpoolctl 3.6.0:** `BSD-3-Clause`.

## Runs

- **Retrieval sample** (2026-11-10): 30 M4FC units spread over the corpus.
  - 1.63 s per query on a cold cache, projecting 81 min for M4FC, so the full
    run goes to the person's terminal.
  - `--check`: every hit is strictly before its bound, and no rule blocks one.
  - All 30 units are identical to the consistency block's ledger.
- **The NLI probe** (server): both models load from their pins and run at
  about 58 pairs per second.
  - **Model B calls an unrelated document a contradiction.** "A bridge
    collapsed in Pellow" against a storm-season outlook scores contradiction
    0.999, where model A says neutral 0.94.
  - That is SNLI's convention, in which different scenes contradict. The pilot
    and the train figures measure what it costs.
- **The NLI pilot** (server, 2026-11-10): 30 seeded train units each of M4FC
  and AVeriTeC, five documents each.
  - **Speed:** 0.10 to 0.11 s per unit. A recheck of 3 in a fresh process was
    identical for both models.
  - **How often a pair scores contradiction ≥ 0.9:**

    | | M4FC | AVeriTeC |
    | --- | --- | --- |
    | model B | 63 of 150 | 26 of 150 |
    | model A | 23 of 150 | 8 of 150 |

    Entailment ≥ 0.9 is 1 of 150 for both models on both corpora.
  - **At an uncalibrated 0.5:**
    - model B calls 24 of M4FC's 30 claims refuted, all 3 true ones among them;
    - on AVeriTeC's 7 supported claims, B says refuted 3, not enough evidence
      3, supported 1;
    - model A says not enough evidence far more often.
  - These are 30 units each, read by the assistant, and decide nothing. The
    threshold is fitted on the full train splits, and the train figures go to
    the person.
- **Full retrieval** (the person's terminal, 2026-11-10/11).
  - **M4FC:** 3,021 units, 30,205 hits. The check passed, and all 3,021 units
    are identical to the consistency block's ledger. Sources: ccnews 26,229,
    wikinews 3,004, enrich 972.
  - **AVeriTeC:** 3,499 units. 105 have no date (train), and 9 dated units
    found nothing. The check passed.
  - Two hits hold an id shape in their own text. Both are page image URLs
    under `images/`, which is content and not an identifier. They are counted,
    not failed.

## The relevance filter (the person's rule, 2026-11-11)

- **Why.** Model B scores unrelated pages as contradictions. So a document
  counts for NLI only if it shares at least one of the claim's entities, or if
  its BM25 score (FTS5's rank: lower is more relevant) is at or under a floor.
- **Entities.** They come from the gallery's pinned NER
  (`dslim/bert-base-NER`), with every label it gives: PER, ORG, LOC, MISC. The
  gallery keeps people and organisations, but relevance needs places and
  events too.
  - They are matched whole word and case-folded in what NLI reads: the title
    and the excerpt.
  - Claims with at least one entity: M4FC 2,773 of 3,015 (92.0%), AVeriTeC
    3,136 of 3,485 (90.0%), AVerImaTeC 875 of 944 (92.7%).
- **The fit.** The floor and the threshold are fitted jointly, by macro-F1, on
  train only.
  - The candidate floors run from strictest to no filter: entities alone,
    then the 0th to 100th percentile (in steps of 5) of the train documents'
    scores, then every document.
  - Ties go to the stricter floor.
  - AVerImaTeC takes AVeriTeC's fit.
- **A unit none of whose documents counts** is `no_documents`: there is no
  usable document, as when the store holds none.
- **Reporting.** Model B is reported with and without the filter. The
  adoption is the person's, on 5-fold cross-validated train figures
  (`textcheck.py nli-adoption`), before any evaluation figure is read.

## The no-enrich ablation

- **What enrich is.** The store's enrich pages are the evidence URLs that
  AVeriTeC's and AVerImaTeC's annotators cited.
- **How much depends on them.** On AVeriTeC dev, 272 of 491 units have a
  different top five without them.
  - Those units are asked again, by Qwen and by NLI, with the enrich pages
    removed.
  - `score` reports the evaluation both ways.

## MUSE with dated evidence (2026-11-11)

- **What it adds.** On M4FC, the text side's retrieval gives each record its
  top five titles, so two of MUSE's evidence similarities exist:
  image–evidence text and claim–evidence text, each the top one, 0 with none.
- **The encoding run.** CLIP L/14 encoded 3,609 images and 15,097 texts on
  the laptop GPU, in 5 min.
- **M4FC val, out-of-context against true** (373 records, 31 of them true, so
  the intervals are wide):

  | Score | AUROC [95%] |
  | --- | --- |
  | CLIP L/14 cosine alone (no evidence) | 0.597 [0.499, 0.704] |
  | MUSE-DT, with evidence | 0.671 [0.559, 0.783] |
  | MUSE-RF, with evidence | 0.644 [0.545, 0.739] |
  | MUSE-MLP, with evidence | **0.762** [0.663, 0.855] |
  | report C's primary score, same records | 0.604 |
  | report C's CLIP B/32, same records | 0.512 |

- **What the evidence is.** It is the dated retrieval, under both leak rules:
  strictly before the fact-check date, and never a fact-checker's page.
- **VERITE** has no date, so it gets none of it.

## NLI: the full runs and the adoption figures (2026-11-11)

- **The runs** (server, one job taken when the GPU freed, every input).
  - Model B (config 0791d1c4e43e) and model A (config 58464e81ce58).
  - Inputs: M4FC 3,021 units, AVeriTeC 3,385, AVerImaTeC 944, AVeriTeC
    no-enrich 272, AVerImaTeC no-enrich 327.
  - A recheck of 3 in a fresh process was identical for every input and model.
- **The figures the adoption is decided on** are train only, 5-fold
  cross-validated within train. Macro-F1 with 95% intervals:

  | | M4FC train (2,524 units) | AVeriTeC train (2,702 units) |
  | --- | --- | --- |
  | B, no filter | 0.324 [0.312, 0.336] | 0.393 [0.374, 0.412] |
  | B, relevance filter | 0.322 [0.310, 0.334] | 0.386 [0.369, 0.405] |
  | A (contrast) | 0.342 [0.329, 0.356] | 0.398 [0.379, 0.417] |
  | majority | 0.321 (accuracy 0.929) | 0.254 (accuracy 0.614) |

- **Train class counts.** M4FC train is 2,345 refuted, 164 supported and 15
  not enough evidence. AVeriTeC train is 1,659, 783 and 260.
- **The filter barely filters.**
  - On M4FC its fitted floor (-6.56) keeps 12,596 of 12,596 documents.
  - On AVeriTeC (-19.61) it keeps 13,047 of 13,474, and 20 units are left with
    none. The dropped documents score contradiction ≥ 0.9 more often (34%)
    than the kept (23%).
  - Retrieval searches by the claim's rarest words, which are mostly its
    entities, so nearly every retrieved document shares one. Pages about the
    same entity and a different event still read as contradictions.
- **The fitted threshold is near 0** (0.004 to 0.006): macro-F1 is best when
  B almost never says not enough evidence.
  - On M4FC, B's macro-F1 is the majority's (0.32), at lower accuracy (0.78
    against 0.93). It carries little beyond the prior there.
  - On AVeriTeC it is above the majority's macro-F1, 0.39 against 0.25.
- No evaluation figure has been read. The adoption is the person's.

## The Qwen pilot (2026-11-11)

- **The run.** 30 seeded train claims each of M4FC and AVeriTeC, judge and
  no-documents probe. Judge config c023fb80c10a (prompt 99126884a924), probe
  config 061d4da17c16 (prompt 2d830791049f).
  - The judge took 1.04 s a claim and the probe 0.28 s.
  - A recheck of 3 in a fresh process was identical for both.
- **Format.**
  - All 120 answers were in format: label mass at least 0.986, and no first
    token forced.
  - 8 of the 60 judge answers were `uncited`.
  - No answer copies an exemplar's wording, names or numbers.
- **The judge** (true label → predicted):
  - **M4FC:** refuted → refuted 7, → not enough evidence 19, → supported 1;
    supported → not enough evidence 2, → refuted 1.
  - **AVeriTeC:** refuted → refuted 6, → not enough evidence 8; supported →
    supported 2, → not enough evidence 5; not enough evidence → itself 3,
    refuted 1, supported 1; conflicting → refuted 4.
- **The person's stop rule fired on the probe, not the judge.** It stops on
  any label above 70% of a pilot's predictions.
  - Not enough evidence was 23 of 30 on AVeriTeC (77%) and 43 of 60 pooled.
    The judge reached exactly 70% on M4FC.
  - The person decided: run everything as built. Abstaining without documents
    is what the probe should do: it measures what the model answers from
    memory.

## The evaluation (2026-11-11)

- **The runs.** Every Qwen run finished on the server. Each recheck of 3 in a
  fresh process was identical.
  - Judge: M4FC 3,021, AVeriTeC 3,385, AVerImaTeC 944.
  - No-enrich: 272 and 327.
  - No-documents: 497, 491 and 944.
- **`verdicts.jsonl`** has 7,270 rows (sha256 4df07d29b126…).
  - M4FC: 3,014 claims × {qwen, nli}. Qwen: 2,500 ok, 514 uncited, none off
    format. NLI-B is at threshold 0.003971.
  - VERITE: 621 claims × 2, all `no_bound`.
  - One caption is in both M4FC train and VERITE. It keeps VERITE's row (the
    person confirmed the decision).
- **Macro-F1 on the evaluation sets**, 95% intervals. Δ is against the
  text-only classifier, paired. Conflicting rows are kept apart (AVeriTeC 38,
  AVerImaTeC 7).

  | | M4FC val (497) | AVeriTeC dev (462) | AVerImaTeC (937) |
  | --- | --- | --- | --- |
  | majority | 0.322 | 0.265 | 0.326 |
  | text-only classifier | 0.445 | 0.450 | 0.345 |
  | Qwen judge | 0.186 [0.164, 0.307], Δ -0.258 [-0.461, -0.199] | 0.392 [0.351, 0.431], Δ -0.058 [-0.128, 0.014] | 0.240 [0.219, 0.261], Δ -0.105 [-0.154, -0.067] |
  | Qwen, no enrich pages | — | 0.318 [0.277, 0.359] | 0.249 [0.223, 0.278] |
  | Qwen, no documents (probe) | 0.148 | 0.375 [0.331, 0.418] | 0.154 |
  | NLI-B (the system's) | 0.345 [0.315, 0.555], Δ -0.099 [-0.209, -0.043] | 0.401 [0.354, 0.450], Δ -0.049 [-0.119, 0.022] | 0.319 [0.296, 0.350], Δ -0.027 [-0.077, 0.018] |
  | NLI-B, relevance filter | 0.345 (identical: it kept every document) | 0.412 [0.361, 0.462] | 0.338 [0.304, 0.379] |

- **How often each judge abstains** (the person's request), as the share
  that answers not enough evidence:

  | | overall | true supported | true refuted | true NEI | base rate |
  | --- | --- | --- | --- | --- | --- |
  | **Qwen**, M4FC | 54.5% | 74.2% | 53.1% | 100% (n=1) | 93.6% refuted |
  | **Qwen**, AVeriTeC | 34.0% | 37.7% | 31.8% | 40.0% | 66.0% refuted |
  | **Qwen**, AVerImaTeC | 41.4% | 47.1% | 41.3% | 41.7% | 95.6% refuted |
  | **NLI-B**, M4FC | 3.4% | 3.2% | 3.4% | 0% | 93.6% refuted |
  | **NLI-B**, AVeriTeC | 8.0% | 8.2% | 7.5% | 11.4% | 66.0% refuted |
  | **NLI-B**, AVerImaTeC | 2.8% | 0% | 2.8% | 4.2% | 95.6% refuted |

- **Coverage:**
  - Every evaluation unit has documents.
  - The top documents hold one of the claim's entities for 85% (M4FC), 82%
    (AVeriTeC) and 83% (AVerImaTeC) of units, and two for 45%, 48% and 48%.
  - Of the top documents, 3.0%, 19.4% and 9.6% come from enrich.
- **What the numbers say:**
  - **No text-side judge beats the claim-only text classifier on any
    evaluation set.** Qwen trails it everywhere, and clearly on M4FC and
    AVerImaTeC. NLI-B trails on M4FC and ties within the interval on AVeriTeC
    and AVerImaTeC.
  - **Qwen abstains on half of M4FC**, where 94% of claims are false, and on 53%
    of the false ones: retrieval finds the claim's context, rarely evidence
    that settles it.
  - **On AVeriTeC the no-documents probe (0.375) comes close to the judge with
    documents (0.392).** Much of the judge's AVeriTeC figure may come from what
    the model already knows, not from the evidence. That is the hindsight risk
    the probe exists to show.
  - **Without the annotators' own cited pages, the AVeriTeC figure falls**
    from 0.392 to 0.318.
  - **NLI-B almost never abstains.** At its fitted threshold it calls nearly
    everything refuted or supported.

## M4FC's official test: the baselines' one read (2026-11-12)

- **The read.** It is the person's go, and logged in
  `data/interim/textcheck/test_baselines.read.json`. A second read is refused.
  - It came after the frozen system's own read (`ensemble.py final-test`,
    scored 2026-11-11T19:03), which stays that system's only one.
  - Same 1,468 records: 833 out-of-context, 379 manipulated, 158 AI-generated,
    98 true.
  - The inputs are the other session's live requests: claim, image, and the
    titles of the retrieval its live path ran. They were read only.
- **MUSE** was refitted on M4FC train and checked to reproduce every published
  M4FC-val AUROC before the read.
- **The per-record file** is `data/interim/textcheck/test_baselines.jsonl`,
  keyed by `claim_key` (all distinct). It holds scores only, no label.

  | AUROC [95%] | false vs true | out-of-context vs true |
  | --- | --- | --- |
  | the frozen system, routed (its own report) | 0.669 | 0.694 |
  | CLIP L/14 cosine | 0.588 [0.533, 0.641] | 0.635 [0.578, 0.690] |
  | MUSE image-only, DT / RF / MLP | 0.551 / 0.521 / 0.588 | 0.572 / 0.527 / 0.635 |
  | MUSE with dated evidence, DT / RF | 0.617 / 0.619 | 0.641 / 0.649 |
  | MUSE with dated evidence, MLP | 0.670 [0.604, 0.728] | 0.704 [0.644, 0.763] |
  | text-only claim classifier, P(false) | **0.784** [0.741, 0.826] | **0.819** [0.777, 0.857] |

- **What it shows.** On M4FC's official test the claim text alone separates
  false from true better than the frozen system and every image baseline. M4FC's
  claims carry a strong textual prior. A paper table should put the system
  beside this baseline.

## NLI without the enrich pages (2026-11-11/12)

- **Model B** (the system's, at its train threshold): AVeriTeC 0.401 with the
  enrich pages, 0.391 [0.341, 0.443] without them. AVerImaTeC 0.3185 both ways.
- **Model A** (contrast): AVeriTeC 0.412 with, 0.355 without; AVerImaTeC 0.352
  with, 0.355 without.
- **Against Qwen.** Qwen falls further without them on AVeriTeC (0.392 to
  0.318): it leans on the annotators' pages more than NLI does.

## How much of the text-only M4FC score is the fact-checker? (2026-11-12)

- **Scope.** The person's audit, on M4FC train and val only, with no test
  read. The claim language is kept only in raw `M4FC.json`; it was filtered to
  the train and val records' ids on load, before any field was used.
- **How each record is scored, out of sample:** train by 5-fold
  cross-validation within train; val from fits on all of train. False against
  true, with NEI left out: train 2,345 against 164, val 465 against 31.

  | AUROC [95%] | train (CV) | val | train + val |
  | --- | --- | --- | --- |
  | fact-checker only (P(false) by org) | **0.918** [0.895, 0.938] | **0.954** [0.895, 0.986] | **0.923** [0.902, 0.941] |
  | text-only classifier | 0.760 [0.721, 0.799] | 0.807 [0.710, 0.889] | 0.769 [0.733, 0.801] |
  | fact-check language only | 0.711 | 0.757 | 0.717 |
  | claim language only | 0.678 | 0.757 | 0.689 |
  | text-only, within the same fact-checker | 0.583 [0.527, 0.640] | 0.678 [0.493, 0.842] | **0.595** [0.540, 0.650] |

- **The text-only classifier within each large org**, over train + val. A
  large org has at least 10 true and 10 false records. AUROC:
  - correctiv.org (10 true, 139 false): 0.607 [0.376, 0.827];
  - poligrafo.sapo.pt (60, 142): 0.639 [0.552, 0.722];
  - snopes.com (100, 103): 0.540 [0.458, 0.623].
- **Where the true claims are.** Snopes is 90 of 191 true on train (47%) and
  10 of 12 on val. Polígrafo is 45 of 172 (26%) and 15 of 30. Together they
  hold 135 of train's 164 true claims and 25 of val's 31. Most other orgs have
  next to none: pesacheck 0 of 271, fatabyyano 1 of 411, factly 2 of 347,
  leadstories 0 of 169.
- **What it means.** M4FC's true class is mostly Snopes and Polígrafo.
  - Knowing only the fact-checker separates false from true better than the
    claim text does.
  - Within one fact-checker, the text-only classifier is close to chance (0.60
    pooled).
  - So most of its M4FC score is the source, read through wording and
    language, not the claim's content. Its official-test AUROC (0.784) should
    be read that way, and so should any M4FC figure from a system that can see
    the claim's text.
