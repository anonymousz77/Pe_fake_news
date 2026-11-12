# Related work

Prior work behind a design decision in this project: what it shares with the
project, and where the project differs. The citations were checked against the
papers on 2026-10-28.

## Time-filtered evidence

Several real-world multimodal datasets already exclude evidence from
fact-checkers and evidence published after the claim (M4FC §2.2 names
Fauxtography, 5Pils and AVerImaTeC; M4FC does the same), filtering web-search
results after retrieval. Ours is a closed, dated snapshot with the filter inside
the query. Whether that changes results relative to post-filtered live retrieval
is to be measured, not assumed.

- M4FC: Geng, Tonglet & Gurevych, arXiv 2510.23508 (v2), §2.2 and Appendix B.
  M4FC and AVerImaTeC are in the register (`m4fc`, `averimatec`); Fauxtography
  and 5Pils are not.
- The filter inside the query is `retrieve.SEARCH_SQL`, red-tested against the
  real index by `scripts/redtest.py` ([evidence_store.md](evidence_store.md)).

## COVE: the closest prior to the consistency block

Tonglet, Thiem & Gurevych, NAACL 2025, arXiv 2502.01194. Code:
[github.com/UKPLab/naacl2025-cove](https://github.com/UKPLab/naacl2025-cove).

COVE first predicts seven context items for the image (source, date, location,
motivation, people, things, event) from reverse-image-search captions, Wikipedia
entities and automated captions, then predicts the caption's veracity by
comparing it against that predicted context. The consistency block asks the same
question of a caption ("the caption claims X; the image is not X"), which makes
COVE its closest prior.
