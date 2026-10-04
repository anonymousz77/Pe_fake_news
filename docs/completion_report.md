# Data layer completion report

**2026-10-03. The data layer is closed.** Every figure below is read from a
report written by the stage that measured it (`data/reports/completion.json`,
`processed.json`, `dead_images.json`, `recovery_closeout.json`,
`verdict_images.json`, `factify2_domain_only_final.json`). None is an estimate
unless it is marked as one.

Pass id `closeout-2026-10` (2026-10-01 to 2026-10-03). Standing rules held
throughout: `data/raw` read and never edited in place, rejected files
quarantined and never deleted, every fetch through three gates, one worker at
one request per 5 s per host, every attempt in the provenance ledger.

## 1. Per dataset, per class

Rows are processed records (`data/processed/records/<dataset>.jsonl.gz`).
**Usable** means label present, not conflicting, text present, and every
*required* image usable (MOCHEG's evidence images are an optional set: a
removed member is dropped from the set, not counted missing). **Removed by
registry** counts referenced files that decode but are not evidence (status
`verdict`, `furniture`, `placeholder`, `blank`, or `wrong_image` from a
fingerprint rejection). **Dead** counts images with no usable copy from any
route, grouped by what the dataset's own URL returned; the full terminal reason
(origin outcome + archive outcome) is per image in `data/dead/<dataset>.csv`; for VERITE,
whose image names are its labels, per record (row index).

| Dataset | Class | Rows | Usable rows | Images expected | Images usable | Coverage | Removed by registry | Dead (origin outcome, top 3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| averimatec | Conflicting Evidence/Cherrypicking | 7 | 7 | 16 | 16 | 100.0% | — | — |
| averimatec | Not Enough Evidence | 24 | 24 | 28 | 28 | 100.0% | — | — |
| averimatec | Refuted | 897 | 896 | 1,329 | 1,327 | 99.8% | placeholder 2 | — |
| averimatec | Supported | 17 | 17 | 20 | 20 | 100.0% | — | — |
| averitec | Conflicting Evidence/Cherrypicking | 233 | 233 | 0 | 0 | — | — | — |
| averitec | unlabelled | 2,215 | 0 | 0 | 0 | — | — | — |
| averitec | Not Enough Evidence | 317 | 317 | 0 | 0 | — | — | — |
| averitec | Refuted | 2,047 | 2,047 | 0 | 0 | — | — | — |
| averitec | Supported | 971 | 970 | 0 | 0 | — | — | — |
| dgm4 | face_attribute | 32,512 | 32,512 | 32,512 | 32,512 | 100.0% | — | — |
| dgm4 | face_attribute&text_attribute | 3,595 | 3,595 | 3,595 | 3,595 | 100.0% | — | — |
| dgm4 | face_attribute&text_swap | 7,968 | 7,968 | 7,968 | 7,968 | 100.0% | — | — |
| dgm4 | face_swap | 38,492 | 38,492 | 38,492 | 38,492 | 100.0% | — | — |
| dgm4 | face_swap&text_attribute | 4,182 | 4,182 | 4,182 | 4,182 | 100.0% | — | — |
| dgm4 | face_swap&text_swap | 9,480 | 9,480 | 9,480 | 9,480 | 100.0% | — | — |
| dgm4 | orig | 60,550 | 60,550 | 60,550 | 60,550 | 100.0% | — | — |
| dgm4 | text_attribute | 6,834 | 6,834 | 6,834 | 6,834 | 100.0% | — | — |
| dgm4 | text_swap | 15,682 | 15,682 | 15,682 | 15,682 | 100.0% | — | — |
| factify2 | Insufficient_Multimodal | 8,500 | 8,454 | 17,000 | 16,954 | 99.7% | — | 46: host_serves_placeholder 46 |
| factify2 | Insufficient_Text | 8,500 | 8,039 | 17,000 | 16,533 | 97.3% | — | 467: host_serves_placeholder 445, gone_404 9, blocked_403 7 |
| factify2 | Refute | 8,500 | 5,635 | 17,000 | 13,877 | 81.6% | — | 3,123: blocked_403 1,522, host_serves_placeholder 1,315, fetched_furniture 152 |
| factify2 | Support_Multimodal | 8,500 | 8,448 | 17,000 | 16,944 | 99.7% | — | 56: blocked_403 42, host_serves_placeholder 6, gone_404 4 |
| factify2 | Support_Text | 8,500 | 7,231 | 17,000 | 15,698 | 92.3% | — | 1,302: host_serves_placeholder 988, blocked_403 283, gone_404 15 |
| fakeddit | 0 | 245,401 | 242,060 | 58,925 | 55,584 | 94.3% | — | 3,341: gone_404 3,126, fetched_furniture 196, fetched_placeholder 11 |
| fakeddit | 1 | 37,002 | 36,564 | 8,909 | 8,471 | 95.1% | — | 438: gone_404 427, fetched_furniture 7, fetched_placeholder 4 |
| fakeddit | 2 | 118,498 | 116,862 | 28,503 | 26,867 | 94.3% | — | 1,636: gone_404 1,618, fetched_blank 10, fetched_furniture 5 |
| fakeddit | 3 | 13,022 | 12,888 | 3,140 | 3,006 | 95.7% | — | 134: gone_404 122, fetched_furniture 12 |
| fakeddit | 4 | 185,667 | 183,786 | 44,801 | 42,920 | 95.8% | — | 1,881: removed_by_host 1,669, dns_failure 65, host_serves_non_image 59 |
| fakeddit | 5 | 23,752 | 23,662 | 5,722 | 5,632 | 98.4% | — | 90: gone_404 74, fetched_furniture 9, fetched_placeholder 6 |
| fakenewsnet | conflicting (label null) | 2 | 0 | 0 | 0 | — | — | — |
| fakenewsnet | fake | 5,753 | 5,753 | 0 | 0 | — | — | — |
| fakenewsnet | real | 17,439 | 17,439 | 0 | 0 | — | — | — |
| isot | fake | 23,481 | 23,481 | 0 | 0 | — | — | — |
| isot | real | 21,417 | 21,417 | 0 | 0 | — | — | — |
| liar | barely-true | 2,108 | 2,108 | 0 | 0 | — | — | — |
| liar | false | 2,511 | 2,511 | 0 | 0 | — | — | — |
| liar | half-true | 2,638 | 2,638 | 0 | 0 | — | — | — |
| liar | mostly-true | 2,466 | 2,466 | 0 | 0 | — | — | — |
| liar | pants-fire | 1,050 | 1,050 | 0 | 0 | — | — | — |
| liar | true | 2,063 | 2,063 | 0 | 0 | — | — | — |
| m4fc | false | 3,282 | 3,230 | 3,282 | 3,230 | 98.4% | — | 52: gone_404 40, fetched_verdict 6, blocked_403 4 |
| m4fc | not enough information | 16 | 16 | 16 | 16 | 100.0% | — | — |
| m4fc | true | 195 | 195 | 195 | 195 | 100.0% | — | — |
| mocheg | NEI | 5,377 | 5,377 | 3,724 | 3,215 | 86.3% | verdict 489, furniture 20 | — |
| mocheg | refuted | 8,004 | 8,004 | 7,546 | 6,959 | 92.2% | verdict 520, furniture 36, placeholder 31 | — |
| mocheg | supported | 7,803 | 7,803 | 5,021 | 4,616 | 91.9% | verdict 385, furniture 20 | — |
| verite | miscaptioned | 338 | 307 | 338 | 307 | 90.8% | — | 31: blocked_403 23, gone_404 4, dns_failure 2 |
| verite | out-of-context | 325 | 300 | 325 | 300 | 92.3% | — | 25: blocked_403 10, gone_404 5, host_serves_non_image 4 |
| verite | true | 338 | 307 | 338 | 307 | 90.8% | — | 31: blocked_403 23, gone_404 4, dns_failure 2 |
| welfake | fake | 37,106 | 37,106 | 0 | 0 | — | — | — |
| welfake | real | 35,028 | 35,028 | 0 | 0 | — | — | — |

The `m4fc` test split (forbidden in the register) is excluded: 3,493 of 4,982
image records are in scope. One M4FC image has a malformed URL in the dataset
itself (`Nhttps://...`) and is dead as `malformed_url_in_dataset`.

## 2. Dead images: final, and a permanent limitation

Every route was tried for every missing image: the dataset's URL (first pass),
a retry with fresh headers and a different user agent, the Wayback Machine (CDX
prefix lookup, up to three snapshots), every alternate URL the dataset's own
metadata declares (VERITE's mediaproxy inner URL, M4FC's authors' Wayback URL),
and an offline match against files already on disk (URL identity across
corpora; CLIP fingerprint for VERITE). **No row is left open.**

| Dataset | Dead images | Not closed | Origin outcome | Archive outcome |
| --- | ---: | ---: | --- | --- |
| verite | 56 | 0 | blocked_403 33, gone_404 9, host_serves_non_image 5, dns_failure 3, server_error_5xx 2, unreachable_ssl_error 1, gone_410 1, refused_http_202 1, fetched_blank 1 | no_snapshot 54, snapshot_wrong_image 2 |
| factify2 | 4,994 | 0 | host_serves_placeholder 2,800, blocked_403 1,854, fetched_furniture 152, gone_404 84, dns_failure 25, host_serves_non_image 17, refused_http_400 11, fetched_verdict 11, refused_http_406 10, unreachable_ssl_error 10, server_error_5xx 6, unreachable_connection_error 6, refused_http_429 4, unreachable_timeout 1, refused_http_415 1, fetched_blank 1, refused_http_412 1 | no_snapshot 2,913, snapshots_are_placeholders 1,242, snapshot_verdict 515, snapshot_furniture 321, archive_refuses_host_404 2, snapshots_failed 1 |
| fakeddit | 7,520 | 0 | gone_404 5,415, removed_by_host 1,669, fetched_furniture 229, dns_failure 65, host_serves_non_image 59, fetched_placeholder 23, server_error_5xx 20, fetched_blank 19, blocked_403 6, unreachable_ssl_error 4, refused_http_400 3, host_serves_placeholder 3, gone_410 2, unreachable_timeout 1, refused_http_429 1, refused_http_409 1 | no_snapshot 6,175, snapshot_placeholder 892, snapshot_furniture 419, snapshot_blank 28, snapshots_not_images 3, snapshots_failed 2, snapshots_are_placeholders 1 |
| m4fc | 61 | 0 | gone_404 49, fetched_verdict 7, blocked_403 4, fetched_blank 1 | archive_refuses_host_404 47, snapshot_verdict 6, no_snapshot 5, snapshots_are_placeholders 3 |

`archive_refuses_host_404`: the archive's index answered HTTP 404 for one object
host (47 M4FC images) on every pass over three days; a refusal, not an outage.
`snapshot_verdict` / `snapshot_furniture`: the archive's only copy is itself a
verdict graphic or a site logo. Published manifests carry the record key and the
reason only, never the URL or the label.

**Recovery this pass** (provenance ledgers, `data/reports/recovery_closeout.json`):
68,464 requests, 3.98 GB in. Items now usable from it: VERITE 122, Factify2
4,178, Fakeddit 4,069, M4FC 4,920 (M4FC's first fetch). A further 3,647 files
came back with status 200 and were then refused by the content gates.
**A re-crawl of refused items returns the same refused content, re-encoded:** of
the 453 files the second follow-up crawl recovered, 13 were genuine images.

## 3. The placeholder registry: version 13

`data/placeholders.yaml` holds **4,135** content hashes: verdict 2,665,
furniture 1,038, generic_stock 236 (recorded, not counted missing),
placeholder 113, blank 83. File-level matches on disk: MOCHEG 8,846, Fakeddit
6,332, Factify2 4,643, M4FC 20, VERITE 5, AVerImaTeC 5, DGM4 0.
`data/fingerprint_rejections.yaml` (version 1) lists 2 VERITE files that decode
and are real images but are provably not the authors' image (CLIP cosine to the
shipped fingerprint 0.591 and 0.671).

New this pass: the `verdict` status. Fact-check articles illustrate themselves
with their own ruling, and MOCHEG and Factify2 take their images from them.
Image-only, a CLIP nearest-neighbour vote scores **91.7%** against a 36.0%
majority on MOCHEG records that carry one; on Factify2, **1,486 of 8,500
Refute** records carry one against 15 of the other 34,000. Found by duplicate
review, CLIP expansion, a CLIP probe (`scripts/content_probe.py`) and OCR; each
entry confirmed by eye; 19 thumbnail-only calls removed after full-size review.
Residual estimated, not assumed (`data/reports/verdict_images.json`).

## 4. Licences

Read at source on 2026-10-01 and quoted verbatim in `docs/licences.md`.
`redistributable: true` only where the text grants it; split files publish
`record_id,split` only.

| Dataset | Licence | Redistributable |
| --- | --- | --- |
| mocheg | CC-BY-4.0 | yes |
| welfake | CC-BY-4.0 | yes |
| averitec | CC-BY-NC-4.0 | yes (non-commercial) |
| dgm4 | Apache-2.0 (dataset card); code S-Lab 1.0, non-commercial | yes |
| m4fc | CC-BY-SA-4.0 | yes |
| verite | Apache-2.0 repository; dataset research-only, images not redistributable | no |
| liar | research use only, no licence named | no |
| factify2, averimatec, fakeddit, isot, fakenewsnet | none stated | no |
| visualnews, newsclippings (deferred) | none stated | no |

## 5. New datasets

Fourteen candidates surveyed 2026-10-01 (`docs/data_card.md`, "New corpora").
Nothing was signed, no one was emailed, no terms were accepted.

**Acquired:** **DGM4** (179,295 distinct train+val records, 206,362 images; test
forbidden) and **M4FC** (4,982 images, CC-BY-SA-4.0). DGM4's text-swap class
**failed its shortcut audit** and is unmapped (section 6).

**Skipped:** NewsCLIPpings and VisualNews (no licence); COSMOS (Google Form);
MMFakeBench (gated terms); X-POSE (institutional e-mail); RW-Post (terms accepted
by access); XFacta and MiRAGeNews (no data licence); Factify3M and MultiCaption
(no download located); Weibo (no licensed distribution); MediaEval VMU (no gap:
the image alone fixes the label); VERITE's own image release (institutional
e-mail and terms; see section 10).

## 6. Reason codes

Derived from labels by `scripts/reason_codes.py`, enforced by
`scripts/processed_loader.py`. Usable rows that may supervise the head:

| Use | content_refuted | content_unverified | media_mismatch | wrong_image | none | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trainable (Factify2 37,807, MOCHEG 21,184, M4FC 3,441, AVeriTeC 3,334) | 18,916 | 22,203 | 0 | 0 | 24,647 | 65,766 |
| evaluation-only (VERITE 914, AVerImaTeC 937) | 896 | 24 | 300 | 307 | 324 | 1,851 |

**`media_mismatch` and `wrong_image` have no training source.** Fakeddit's False
Connection is unmapped (its labels are its subreddits); DGM4's text swaps are
unmapped (half cross publishers; the within-publisher subset fails the
pre-fixed purity test on base rate). The head can be scored on both and trained
on neither. Never train on DGM4 and evaluate on VERITE without reporting the
transfer gap.

## 7. Splits and leakage

Assigned by `scripts/split_all.py`, enforced by `assert_atomic()`. Every image
with five or more copies and every image in more than one corpus keeps all its
records in one split; DGM4 captions are grouped; Factify2 is source-disjoint on
label-predictive domains (purity >= 0.90; 780 domains, 0 violations); VERITE and
AVerImaTeC are test-only.

**Cross-corpus duplicates** (`data/cross_corpus_duplicates.yaml`, regenerated
from the final layer): 547 hashes; Factify2+M4FC 409, Factify2+MOCHEG 112, and
**37 VERITE records whose image also sits in Factify2, M4FC or MOCHEG**. Those 37
must be dropped from any VERITE evaluation of a model trained on those corpora;
`placeholders.assert_no_cross_corpus_leak` raises otherwise.

**Factify2 hostname baseline, re-measured on usable images:** corpus-wide
held-out **40.4%** (NMI 0.326); random 70/15/15 split **40.3%**; the
source-disjoint split **12.7%**, but its test set is **54.4% Refute** (5,620 of
10,325), so the comparator there is the **56.9% majority**, not the hostname
figure. **Factify2 `val` holds 0 Refute records.**

## 8. Budgets

| | Cap | This pass | Source |
| --- | ---: | --- | --- |
| Stored | 60 GB | **58.18 GB**: raw 55.79 (MOCHEG images 41.28, DGM4 10.96, Factify2 1.40, M4FC 1.21, Fakeddit 0.75, ISOT 0.16, VERITE 0.02) + derived 2.39 | sizes of files written since 2026-10-01 |
| Transferred | 120 GB | **about 102.6 GB**: 88.44 GB logged in `FETCH_LOG.jsonl` for completed fetches; 13.41 GB of MOCHEG streams that crashed or were reaped before logging (in-session tally, not re-measurable); 0.71 GB of recovery runs killed before logging (provenance ledgers) | as stated |

`data/raw` now holds 96.75 GB; the register's `est_gb` (GiB) sums to 89.86
against the 120 budget.

## 9. Corrected values

Every figure the docs quoted before this pass that the pass changed:

| Figure | Was | Now |
| --- | --- | --- |
| LIAR rows | 12,791 | **12,836** (CSV parsed with `QUOTE_NONE`; default quoting merged 45 lines) |
| ISOT | not fetched | **44,898** (real 21,417, fake 23,481, exact) |
| VERITE rows with a usable image | 721 (72.0%), later 71.3% | **914 (91.3%)**: true 307, miscaptioned 307, out-of-context 300 |
| Factify2 Refute image coverage | 70.0% (then 67.0%) | **81.6%** |
| Factify2 other classes, image coverage | 97.0-99.9% | **92.3-99.7%** (registry v7-v13 removed placeholders and logos) |
| Factify2 Refute row completeness | 58.6% | **66.3%** (5,635 of 8,500) |
| Factify2 other classes, row completeness | 95.1-99.9% | **85.1-99.4%** |
| Factify2 usable rows | 38,425 complete | **37,807** |
| Factify2 `val` Refute records | 4 | **0** |
| Factify2 domain-only, corpus-wide held out | 38.2% (projected 41.7% if recovery completed) | **40.4%** |
| Factify2 domain-only, random split | 37.5% | **40.3%** |
| Factify2 domain-only, source-disjoint split | 22.2% | **12.7%**, against a 56.9% test majority |
| Fakeddit images | 142,434 (94.96%), per class 92.3-99.3% | **142,480 usable (95.0%)**, per class 94.3-98.4% |
| MOCHEG images | "contains no images" | image release acquired: 137,619 files; 14,790 of 16,291 evidence references usable (90.8%) |
| Evidence-free images | 5,057 files | registry v13: 4,135 hashes; 19,851 files on disk count as missing (generic_stock is recorded but stays usable) |
| Datasets in the register | twelve | **fourteen** (twelve enabled) |
| Enabled measured size | 38.06 GB | **96.75 GB** in `data/raw` |
| M4FC `est_gb` | 1.0 (estimate) | **1.125 GiB**, measured |
| FakeNewsNet records | 23,196 | **23,194**: two ids sit in both label files and are one record each, label null |
| ISOT and FakeNewsNet record ids | `fake_12`, `fake:gossipcop-1` | label-free: a content hash (ISOT), the news id (FakeNewsNet) |
| VERITE file names in tracked files | `true_N` / `false_N` | row-index aliases (`rows-108-109.jpg`); real paths in a gitignored map |

## 10. What is not closed

**Blocking (only a person can close these):**
- **VERITE images.** 87 of 1,001 rows have no usable image. The authors host the
  full set on Hugging Face behind an institutional-email gate; requesting it is a
  human decision this pass did not make.
- **`media_mismatch` / `wrong_image` training data.** None exists in the layer.
  Closing it needs a new licensed corpus, or a decision to accept DGM4's
  within-publisher subset (`reason_codes.DGM4_AUDIT_PASSED`).

**Noted (permanent limitations):**
- M4FC claims: 6,789 measured against 6,980 stated (-191), unresolved.
- DGM4 totals cannot be checked against the paper: its test split is forbidden.
- MOCHEG: the Zenodo release matches Zenodo's counts, not the SIGIR paper's;
  tweet evidence is not hydrated (the release carries no tweet ids).
- The verdict sweep is not exhaustive; the residual is estimated in
  `verdict_images.json`.
- Factify2: `val` has no Refute; the source-disjoint test is 54% Refute.
- Fakeddit: labels are subreddits (all 22 pure); False Connection is unmapped;
  fakehistoryporn is excluded from reason supervision.
- Seven of twelve enabled corpora state no licence; nothing from them is
  redistributed.
- `FETCH_LOG.jsonl` is written when a fetch completes, so killed runs leave no
  line; the provenance ledgers are complete.
