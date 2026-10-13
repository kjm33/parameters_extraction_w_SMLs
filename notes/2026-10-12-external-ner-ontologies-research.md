# 2026-10-12 — External NER systems, taxonomies/ontologies & other NLP techniques for slot filling

Context: after the round-3 wikineural prompt-hint failure, evaluate what EXTERNAL resources
(dedicated NER systems, gazetteers/ontologies, other NLP techniques) could add to the slot
filling stack. Question from user: which external NER systems, taxonomies/ontologies (e.g.
DBpedia), and other NLP techniques are useful for extracting information from navigation
user prompts. Companion TODO: `plans/2026-10-12_external-ner-ontology-todo.md`.

Online research done 2026-10-12 via direct URL fetches (the web-search backend returned
noise all session; every fact below was verified on the project page / HF model card /
GitHub README itself).

## Prior evidence that constrains any proposal

- Round-3: NER as PROMPT HINTS failed (baseline×hinted −3.4 F1, FP 128→198 — model copies
  hint spans into wrong slots; ner-trained −1.5, hint-dependent). → External NER is allowed
  POST-HOC only (rescoring, canonicalization, arbitration, detector features).
- Round-9: pointer/span outputs rejected — gold rewards CANONICAL forms (pl "Torunia" vs
  gold "Toruń"; gold-side ", CA" appendages absent from the utterance). A surface→canonical
  lookup (gazetteer) is exactly the right shape to close this gap; canonicalize v1 already
  proves the mechanism (train-derived alias map, pl suffix fallback) but is ≈neutral on the
  JSON winner because coverage is tiny.
- Remaining measured error mass (round-7 winner, lang-free): q boundary (37 FN + 29 FP),
  canonical-form fidelity (pl −6.9, en −6.4 in the spans round), enum/null judgment
  (avoid_features 29 FP; 128 hallucinated slots are enum/null judgment, not free-text
  fabrication).
- Round-5/9: BERT BIO tagger (bert-tagger-fixd-v2+canon 0.7813 F1) is our in-house span
  machinery; external NER competes with / complements it.

## 1. Dedicated external NER systems

| System | License | Fit for us (5 langs: en/de/pl/fr/es) |
|---|---|---|
| **GLiNER** (github.com/urchade/GLiNER, NAACL 2024, arXiv 2311.08526) | Apache-2.0 | BEST schema-conditioned option: zero-shot NER with arbitrary label strings (can use slot-oriented labels 'POI','address','city','category query'); encoder-only, CPU/ONNX/INT8 viable, fine-tunable. Multilingual checkpoints: `urchade/gliner_multi`, `urchade/gliner_multi-v2.1` (+ ONNX export by onnx-community); v2.5 series under gliner-community (small…xxl). Ecosystem: GLiClass (zero-shot clf), GUILayouter (entity linking), GLiNER.cpp (edge C++ inference). |
| **Stanza** (stanfordnlp.github.io/stanza) | Apache-2.0 | Strongest off-the-shelf PER-LANGUAGE quality incl. Polish: NKJP F1 88.73 (6 tags incl. placeName/geogName); GermEval 85.2 (de), WikiNER 92.9 (fr), AnCora 88.6 (es), CoNLL03 92.1 (en). Per-language lemmatization/morphology = canonicalization asset (pl inflections). |
| **Flair** (huggingface.co/flair) | **all NONCOMMERCIAL** | LICENSE CORRECTION (2026-10-12, checked on all six model cards during the item-4 benchmark): BOTH classic 1.0 models AND the new 2.0 'entity-*' models (entity-{english,german,multilingual}-{4,8}class) are **Flukes NC 1.0 / noncommercial** — the earlier "2.0 commercial-OK" reading was wrong. Flair is commercially unusable for us on every language. Benchmark (eval/results/compare_ner.json, exact gold-value coverage on core_test): flair beats GLiNER only on de (87.5 vs 86.4), ties en/fr, loses pl/es → DROP. PyPI flair max = 0.15.1 (no 2.x lib); 2.0-style checkpoints load via a shim in eval/flair_spans.py. |
| **spaCy** | MIT | No proper Polish NER model; en/de/fr/es decent but adds nothing over Stanza/GLiNER. Weak fit. |
| **mordecai3** (github.com/ahalterman/mordecai3) | MIT | Full geoparser (spaCy NER + GeoNames in local Elasticsearch + neural ranker, calibrated probs; 92.5% exact match / 96.7% @161km given span found; prebuilt 2.2GB index). ENGLISH-ONLY pipeline (en_core_web_trf) + ES server = not edge, not multilingual. Concept to copy: gazetteer-backed candidate ranking with calibrated confidence. |
| Edinburgh Geoparser (ltg.ed.ac.uk) | GPL, academic | English-centric, GPL → skip. |

Also spotted on HF (unvetted): knowledgator/gliner-multitask-large-v0.5,
modern-gliner-bi-base/large-v1.0, numind/NuNER_Zero(-span/-4k),
E3-JSI/gliner-multi-pii-domains-v1.

## 2. Taxonomies / ontologies / gazetteers

| Resource | License | Use for us |
|---|---|---|
| **GeoNames** (geonames.org) | CC-BY 4.0 | KEY ASSET: 11M+ places dump; `alternateNames` table = per-language names + exonyms (Warszawa/Warsaw/Varsovie) + ASCII/short variants → canonicalize v2 lookup (surface→canonical, incl. inflected pl forms via fuzzy/stemmed match); admin codes in dump → recover gold-side ", CA" appendages. Filtered per-country offline index = CPU-trivial at inference. |
| **Wikidata** | **CC0** | Safest license (automotive-friendly). Multilingual labels + aliases, P31 place types. Alias coverage partial for case inflections — complement, not replacement, for GeoNames. |
| **DBpedia** | data CC-BY-SA (share-alike) | EXCLUDED by decision (share-alike friction for automotive use, despite Apache-2.0 Spotlight code). |
| **OSM / Nominatim / Photon** | ODbL / Apache-2.0 (Photon) | Self-hosted geocoders = powerful offline place-resolution VALIDATORS (Photon typo-tolerant), but server-weight (ES/postgres) → build/eval-time only, not edge runtime. |
| HERE category taxonomy | proprietary | q values are free-text category queries ('parking', 'charging station') → a phrase→category-id mapping improves downstream search quality, but changes the task target (out of F1 scope) → training-time/data-build experiment only. |

## 3. Other NLP techniques mapped to our error classes

- **Gazetteer canonicalization v2** (GeoNames alternateNames + Morfeusz2 pl lemmatization +
  rapidfuzz fuzzy match, offline index): attacks canonical-form fidelity (pl −6.9, en −6.4)
  and ", CA" appendages. Extends `eval/canonicalize.py`; rescored via
  `eval/score_no_lang.py --canon`.
- **Embedding enum-matcher** (multilingual-e5-small / LaBSE, CPU, offline): precompute
  embeddings of enum values (+ multilingual paraphrases); map emitted enum-ish strings to
  nearest VALID enum value above a cosine threshold. Attacks the 29 avoid_features FPs +
  enum/null judgment (128 hallucinated slots). Complements grammar-constrained decoding
  (round-4 G: guarantees valid enums but cannot choose them).
- **GLiNER span arbitration (post-hoc)**: verify model free-text values overlap a GLiNER
  span; truncate/repair boundary errors, flag disagreements as detector features.
  Attacks q boundary (37 FN + 29 FP). NEVER as prompt hints (round-3).
- **Morphological analysis (Stanza/Morfeusz2 lemmas)**: pl surface→lemma matching for both
  canonicalization and eval lenient-compare (A2 headroom 2.5–3.5 F1 exists).
- **ASR-confusion fuzzy matching** (future, voice input): phonetic confusions on foreign
  place names — same rapidfuzz/gazetteer machinery.

## Licensing summary (automotive constraint)

Wikidata CC0 > GeoNames CC-BY (attribution) > DBpedia CC-BY-SA (excluded). Model licenses:
GLiNER/Stanza/spaCy/mordecai3 permissive; Flair mixed (1.0 noncommercial; 2.0 en/de/multi
commercial); Edinburgh GPL.

## User decisions (binding, 2026-10-12)

1. **Strictly offline at inference** — local dumps/models only (matches edge goal);
   build-time downloads fine.
2. Priorities: **gazetteer canonicalize v2 + GLiNER span arbitration + embedding
   enum-matcher + benchmark Flair** (separate per-language pipelines allowed).
3. **DBpedia excluded; Wikidata CC0** as the ontology complement.
