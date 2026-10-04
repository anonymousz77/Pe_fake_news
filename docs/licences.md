# Licences — what each distribution actually says

Retrieved 2026-10-01 from the distribution point of each dataset. Every quote
below is verbatim from the URL beside it; the raw copies (READMEs, API
responses, the ISOT PDF and its extracted text) are kept in
`data/interim/licences/` so the wording can be re-checked without trusting this
page. Where a distribution says nothing, this page says **none stated** rather
than inferring a licence from the paper, the venue or a mirror.

"Redistributable" below means: does the stated licence grant permission to
redistribute the dataset's own content? It is not a judgement about third-party
material inside it (news photographs, articles, tweets), which stays under its
publishers' copyright whatever the dataset licence says.

## Summary

| Dataset | Licence as stated | Source | Redistributable |
| --- | --- | --- | --- |
| mocheg | CC BY 4.0 | Zenodo record metadata; repo README | **yes**, with attribution |
| welfake | CC BY 4.0 | Zenodo record metadata | **yes**, with attribution |
| averitec | CC BY-NC 4.0 | Hugging Face repo card | **yes**, non-commercial, with attribution |
| verite | Apache-2.0 (repository); "intended solely for research purposes" (dataset) | GitHub repo + README | **no** for images (gated, "No Redistribution"); CSVs ship in an Apache-2.0 repo but the README restricts the dataset to research |
| liar | research use only; no licence named | README inside the zip | **no** — no grant to redistribute |
| factify2 | **none stated** | README in the Drive folder; shared-task page | **no** — registration-gated, no grant |
| fakeddit | **none stated** | GitHub repo (no licence file); project website | **no** — no grant |
| averimatec | **none stated** | Hugging Face dataset card (no licence field) | **no** — no grant |
| isot | **none stated**; citation requested | lab page + ReadMe PDF | **no** — no grant |
| fakenewsnet | **none stated**; README says the complete dataset "cannot be distributed" | GitHub repo (no licence file) + README | **no** |
| dgm4 | Apache-2.0 (dataset card); code repo S-Lab License 1.0 (non-commercial) | Hugging Face dataset card; GitHub LICENSE | **yes** — non-commercially satisfies both |
| m4fc | CC BY-SA 4.0 (dataset); Apache-2.0 (code) | GitHub README, data/README.md, data/license | **yes**, with attribution and share-alike |

Counted over the ten datasets held before this pass: **3 redistributable**
(mocheg, welfake, averitec — the last non-commercially), **7 not**, of which
**5 state no licence at all** (factify2, fakeddit, averimatec, isot,
fakenewsnet) and 2 state terms without a grant (liar, verite). The two added
on 2026-10-01, dgm4 and m4fc, were admitted *because* each states a licence
that permits research use; see the section-10 table in `docs/data_card.md`.

## Verbatim, per dataset

### mocheg
- <https://zenodo.org/records/6653772> — record metadata (`/api/records/6653772`):
  `"license": {"id": "cc-by-4.0"}`, `"access_right": "open"`.
- <https://github.com/PLUM-Lab/Mocheg> — README:
  > Our dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The associated codes are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- The image-bearing release on the authors' server
  (<http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/>)
  carries no separate licence file; the README shipped inside the Zenodo
  tarball points to it: "The complete dataset can be accessed from
  http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/."
  The GitHub README instead links a Google Form; it was not filled in.

### welfake
- <https://zenodo.org/records/4561253> — record metadata (`/api/records/4561253`):
  `"license": {"id": "cc-by-4.0"}`, `"access_right": "open"`. WELFake merges
  four earlier corpora; the CC BY grant is the uploader's and cannot extend to
  rights the uploader does not hold.

### averitec
- <https://huggingface.co/chenxwh/AVeriTeC> — repository card front matter:
  ```
  license: cc-by-nc-4.0
  ```
  The Hub API reports the tag `license:cc-by-nc-4.0` and `gated: false`.

### verite
- <https://github.com/stevejpapad/image-text-verification> — repository licence
  Apache-2.0 (GitHub API `spdx_id: Apache-2.0`). README:
  > Please note that this dataset is intended solely for research purposes.

  > The images are sourced from within the articles of Snopes and Reuters, as well as Google Images. We do not provide the images, only their URLs.

  > 🤗 **UPDATE (July 2026):** Because many of the original image URLs are no longer accessible, the raw images are provided on [Hugging Face](https://huggingface.co/datasets/stefpapad/VERITE). Access requires an official academic or research institution email address for verification.
- <https://huggingface.co/datasets/stefpapad/VERITE> — `gated: manual`, card
  `license: apache-2.0`, and the access prompt:
  > By requesting access to this dataset, you agree to the following terms:
  > * **Research Use Only:** This dataset is strictly for non-commercial, scientific, or academic research purposes.
  > * **No Redistribution:** You agree not to share, distribute, or re-publish this dataset (or any portion of it) with third parties.
  > * **Full Liability:** You assume full responsibility for your use of the data, including ensuring compliance with copyright laws and protecting the rights of third parties. The dataset providers hold no liability for infringements arising from misuse.

  **Not requested.** It needs an institutional login and acceptance of terms,
  which this pass does not do on anyone's behalf.
- <https://github.com/stevejpapad/relevant-evidence-detection> (the copy we hold)
  — Apache-2.0: "This project is licensed under the Apache License 2.0".

### liar
- <https://www.cs.ucsb.edu/~william/data/liar_dataset.zip> — `README` inside the zip:
  > The original sources retain the copyright of the data.
  >
  > Note that there are absolutely no guarantees with this data,
  > and we provide this dataset "as is",
  > but you are welcome to report the issues of the preliminary version
  > of this data.
  >
  > You are allowed to use this dataset for research purposes only.

### factify2
- Google Drive folder `13JwnIBzDfe8a5E1anPkt7J90r4NBIYES`, `README.md`: describes
  the task and per-class counts; **no licence or terms**.
- <https://aiisc.ai/defactify2/>: **no licence, terms or copyright statement**;
  access is through a registration form (<https://forms.gle/L43vLWdYX3gGMTnV>)
  and the archives are password-protected.

### fakeddit
- <https://github.com/entitize/Fakeddit>: no licence file (GitHub API
  `license: null`); README states none.
- <https://fakeddit.netlify.app/>: **no licence**; challenge guidelines only, e.g.
  "Do not attempt to extract ground truth labels for our samples on the Internet."

### averimatec
- <https://huggingface.co/datasets/Rui4416/AVerImaTeC>: card has no `license`
  field (Hub API `cardData.license: null`), README states none. The register's
  previous `licence: research` had no source and has been corrected.

### isot
- <https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/>:
  no licence or terms on the page.
- `ISOT_Fake_News_Dataset_ReadMe.pdf` (same site): no licence; a citation
  request only — "To cite this dataset use: 1. Ahmed H, Traore I, Saad S.
  “Detecting opinion spams and fake news using text classification”, Journal of
  Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018. 2. …
  ISDDC 2017. Lecture Notes in Computer Science, vol 10618."

### fakenewsnet
- <https://github.com/KaiDMML/FakeNewsNet>: no licence file (GitHub API
  `license: null`). README:
  > Complete dataset cannot be distributed because of Twitter privacy policies and news publisher copy rights.

### dgm4 (added 2026-10-01)
- <https://huggingface.co/datasets/rshaojimmy/DGM4> — dataset card front matter
  `license: apache-2.0`; Hub API `gated: false`. No access request.
- <https://github.com/rshaojimmy/MultiModal-DeepFake> — `LICENSE`:
  > S-Lab License 1.0
  >
  > Copyright 2023 S-Lab
  > Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  > …
  > 4. In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.

  The two disagree on commercial use; both permit research use. The images
  are VisualNews news photographs and remain their publishers' copyright.

### m4fc (added 2026-10-01)
- <https://github.com/UKPLab/M4FC> — README:
  > The code is released under an **Apache 2.0** license, while the dataset is released under a **CC-BY-SA-4.0** license.
- `data/README.md`: "M4FC is made available under a **CC-BY-SA-4.0** license."
  `data/license` carries the full "Attribution-ShareAlike 4.0 International"
  legal code. No access request. The README also states: "Given the graphic
  nature of some images, we do not release them directly. Instead, we do
  publicly release the URLs of the FC articles and the images."
