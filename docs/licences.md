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

## The entity gallery: its sources and its models (added 2026-10-21)

The entity gallery (`scripts/gallery.py`) is not a dataset in the register. Its
two sources and the models it runs are recorded here on the same terms: the
quotes are verbatim (wiki markup removed), and the raw copies are in
`data/interim/licences/gallery_*`. All were retrieved on 2026-10-21.

| Source or model | Licence as stated | Where it was read | Use here |
| --- | --- | --- | --- |
| Wikidata items | CC0 (structured data) | Wikidata:Licensing | labels, aliases, classes, image claims |
| Commons reference images | **per file**, as each file page states | imageinfo `extmetadata`, recorded per image | references; never published |
| dslim/bert-base-NER @ `d1a3e8f` | MIT | model card front matter | claim-text NER |
| YuNet `face_detection_yunet_2023mar.onnx` @ `3cc26e7` | MIT | LICENSE and README | face detection |
| SFace `face_recognition_sface_2021dec.onnx` @ `3d70824` | Apache-2.0 | LICENSE and README | face embedding |
| AuraFace `glintr100.onnx` @ `af6d057` | Apache-2.0 | model card, LICENSE.md | bake-off contrast only, not adopted |
| InsightFace model packs (`buffalo_l`, ...) | "for non-commercial research only" | python-package README | **excluded** |

Weights are fetched to `models/` (ignored) by `gallery.py fetch-models`, which
checks each file's sha256, and each licence file's sha256, against the values
read on 2026-10-21. A licence text that changes is read again before anything
is fetched.

### Wikidata
- <https://www.wikidata.org/wiki/Wikidata:Licensing> (raw wikitext):
  > All structured data (i.e. the main, Property, Lexeme, and EntitySchema namespaces) is released into the public domain under Creative Commons Zero. Text in other namespaces is copyrighted and available under the Creative Commons Attribution-ShareAlike 4.0 License.

### Wikimedia Commons
- <https://commons.wikimedia.org/wiki/Commons:Licensing> (raw wikitext):
  > Wikimedia Commons only accepts free content, that is, images and other media files that are not subject to copyright restrictions which would prevent them being used by anyone, anytime, for any purpose. The use may however be restricted by issues not related to copyright […]

  > Media licensed exclusively under non-commercial only licenses (like CC BY-NC-SA) are not accepted either.

  > The license that applies to an image or media file must be indicated clearly on the file description page using a copyright tag. All information required by that license must be provided on the description page.

  So the licence is per file. For every reference the gallery records Commons's
  `LicenseShortName`, `LicenseUrl`, `UsageTerms`, `AttributionRequired`,
  `Copyrighted`, `Restrictions`, `Artist` and `Credit`, exactly as the API
  returns them. These live in `data/gallery/entities.jsonl`, which is local.
  `Restrictions` carries the non-copyright limits Commons flags, such as
  `personality` (rights of the person pictured) and `trademarked`. On the sample,
  the usable person references were mostly GODL-India (103), public domain
  (83) and CC BY / BY-SA variants. 24 carry `personality`; 230 require
  attribution.

### Request rate: Wikimedia's robot policy (read 2026-10-21)
Not a licence, but the terms on which these sources are read. Raw copies are
`data/interim/licences/gallery_robot_policy.wikitext` and
`gallery_api_etiquette.wikitext`.

- <https://wikitech.wikimedia.org/wiki/Robot_policy> (last edited 2026-03-16), Action API rules:
  > If unauthenticated, keep the concurrency of your requests to 1 at a time, and below 5 requests per second overall.

  > Avoid using expensive API endpoints: if your request takes more than 1 second to serve, please wait 5 seconds before making another request.

  Media API rules (`upload.wikimedia.org`):
  > Always keep a total concurrency of at most 2, and limit your total download speed to 25 Mbps (as measured over 10 second intervals).

  > Only use originals or one of our standard thumbnail sizes, which you can see in mediawiki.org.

  Generally applicable rules:
  > When we reply with a 429 Too Many Requests status code respect the delay specified by the Retry-After header sent with the response.

  > Always request content with an `Accept-Encoding: gzip` HTTP header to reduce bandwidth usage. This is not necessary when you are directly requesting media files (i.e. images or video) which are already in a compressed format.

  Rules for other resources ("Gerrit, GitLab, Phabricator, and other wikimedia.org services"), applied here too:
  > Pause crawling for at least 15 minutes if you receive a 5xx status code.
- <https://www.mediawiki.org/wiki/API:Etiquette>:
  > Making your requests in series rather than in parallel, by waiting for one request to finish before sending a new request, should result in a safe request rate.

`gallery.WikimediaWorker` keeps one request in flight in all. It starts at most
one request per second on each of `www.wikidata.org`, `commons.wikimedia.org`,
`upload.wikimedia.org` and `thumb.wikimedia.org`, so at most two a second reach
the Action API, and thumbnails arrive at about 1.5 Mbit/s. After an API answer
that took over a second it waits five seconds. A 429's Retry-After (a minute
when none is given), a maxlag refusal, and a 502, 503 or 504 (fifteen
minutes) hold the whole host. A 500 on one file is that file's answer, a
thumbnail that cannot be rendered, and holds only its own retries. A request
that still gets no answer, or a "not now", after its retries leaves its entity
for the next run; it is never recorded as a missing file. API calls send
`maxlag` and ask for gzip; media requests do not.
References are the standard 960-pixel thumbnail. Every other host keeps the
data layer's five seconds.

Three of the policy's rules are not met, and that is said here rather than
glossed over:
- **No authentication.** The policy asks a client that makes "a significant
  number of requests" to give a URL listing its address ranges, or to
  "Authenticate your requests using an on-wiki account when you are making API
  requests". The gallery does neither. Authenticating would mean creating an
  account and accepting its terms on the project's behalf, which is a
  person's decision. So the gallery stays within the unauthenticated limits
  quoted above.
- **robots.txt.** The policy says "Honor every directive in our robots.txt
  file". The wikis' robots.txt disallows `/w/` to unnamed agents, and that
  covers `/w/api.php`. The same policy gives the Action API rules quoted above
  for bots. The gallery takes those as the rule for an API client, and
  robots.txt as the rule for crawlers.
- **No dumps.** The policy says to "Consider if dumps are more efficient than
  live requests". The build needs the records of about 7,000 entities and
  their images, and a Wikidata or Commons dump has to be downloaded and parsed
  whole, so live requests were used.

### Models
- dslim/bert-base-NER, <https://huggingface.co/dslim/bert-base-NER>, model card front matter: `license: mit`.
- YuNet, <https://huggingface.co/opencv/face_detection_yunet> (OpenCV Zoo):
  > All files in this directory are licensed under [MIT License](./LICENSE).

  LICENSE: "MIT License", "Copyright (c) 2020 Shiqi Yu". The README does not
  state the training data; it points to `libfacedetection.train`.
- SFace, <https://huggingface.co/opencv/face_recognition_sface> (OpenCV Zoo):
  > All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

  > Model files encode MobileFaceNet instances trained on the SFace loss function, see the SFace paper for reference.

  LICENSE: "Apache License Version 2.0, January 2004". The training data is
  **not stated**. The SFace paper's experiments use CASIA-WebFace, VGGFace2
  and MS-Celeb-1M; which of them trained this file is not said.
- AuraFace, <https://huggingface.co/fal/AuraFace-v1>: model card front matter
  `license: apache-2.0`; LICENSE.md is the Apache License 2.0 text. README:
  > AuraFace was trained on a commercial dataset comprising face images from various sources.

  Used only as the bake-off contrast, and only `glintr100.onnx`, behind YuNet.
  The same repository ships `scrfd_10g_bnkps.onnx`, `1k3d68.onnx`,
  `2d106det.onnx` and `genderage.onnx`, the file names of InsightFace's
  model packs. They are never fetched. Adopting AuraFace is a licence
  decision for a person, with data in hand.
- InsightFace, <https://github.com/deepinsight/insightface/tree/master/python-package>, README:
  > The library code is released under the **MIT License**, for academic and commercial use. **The pretrained models provided with this library are for non-commercial research only**, whether downloaded automatically or manually.

  Excluded: the brief rules out a model that needs a commercial licence.
