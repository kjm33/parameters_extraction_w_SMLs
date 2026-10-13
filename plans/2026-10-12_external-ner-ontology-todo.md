# TODO — External NER / ontology / NLP resources (post-hoc stack)

Generated 2026-10-12 from `notes/2026-10-12-external-ner-ontologies-research.md`
(online-verified research). Constraint: **strictly offline at inference** (local dumps +
models only; downloads at build time). Lesson from round-3: external NER is used POST-HOC
only — never as prompt hints. DBpedia excluded (CC-BY-SA); Wikidata CC0 preferred.

Baseline to beat (lang-free official metric, round 7):
- Winner: llama-3.2-1b LoRA v2 — F1 0.7958 / EM 0.7956 (`train/runs/llama-3.2-1b/merged/`).
- Edge: bert-tagger-fixd-v2+canon — F1 0.7813 / EM 0.7846.
- canon v1 ≈ neutral on the JSON winner (coverage too small — v2 exists to fix that).
- Error mass to attack: q boundary (37 FN + 29 FP), canonical fidelity (pl −6.9, en −6.4),
  enum/null judgment (29 avoid_features FP, 128 hallucinated slots).

Verification for every post-processor item: rescore existing winner-v2 + tagger test dumps
via `python3 eval/score_no_lang.py --canon` variants (CPU, no retraining needed first);
adopt only if F1 gain holds on BOTH the LLM winner and the tagger, en/de not worse.

## High priority

- [ ] **1. Canonicalize v2 — GeoNames gazetteer + pl morphology + fuzzy match**
      *(attacks canonical fidelity: pl −6.9, en −6.4; extends eval/canonicalize.py)*
      - [ ] Download GeoNames dump (CC-BY); build filtered offline index for our countries:
            citiesXX.txt + admin1Codes + alternateNames (per-language names, exonyms,
            ASCII variants) → compact local lookup (sqlite or pickled dict).
      - [ ] Matching cascade over surface form: exact alternateNames hit → Morfeusz2 lemma
            (pl) then hit → rapidfuzz top candidate above threshold (ratio ≥ ~90).
      - [ ] Admin-code appendage recovery: gold-style ", CA" / ", France" suffix generation
            from admin1/parent names where our alias map already fires.
      - [x] Re-run `score_no_lang.py --canon` on winner-v2 and tagger dumps; acceptance:
            +0.5 F1 net on at least one model, pl gain ≥ +1.5, en/de not worse than −0.2.
      - [x] License hygiene: GeoNames CC-BY attribution note in repo + any shipped artifact.
      - Estimate: ~0.5–1 d. CPU-only. No training.
      **OUTCOME (2026-10-12): NEUTRAL — acceptance NOT met, gates documented.** Built
      `src/build_gazetteer.py` (cities15000+alternateNames filtered to de/en/es/fr/pl/int →
      `data/geo_gazetteer.pkl`, 34k cities / 38k unique folds / 5.5MB) +
      `eval/canonicalize_v2.py` (G1 exact-altname, G2 Morfeusz pl-lemma, G3 rapidfuzz,
      G4 appendage; `--calibrate` corruption check on train). Final config (all gates ON):
      corruption 0/3996, fires 0/0 on both dumps, Δ F1 vs v1-only = 0.0000 everywhere.
      Root cause: gold is utterance-verbatim — GeoNames display-name rewrites are
      exonym-corrupting (Stockholm→Sztokholm, Krakow→Cracovia, 111/114 train corruptions
      for ungated G1; ungated G2 42/42 via Polish exonym lemmas); genuine pl inflection
      fixes are already covered by the v1 train-derived alias map (tagger pl +12.9 all
      from v1). GeoNames layer kept as a zero-risk pass behind gates; v1 remains shipped.

- [ ] **2. Embedding enum-matcher (multilingual-e5-small / LaBSE, offline)**
      *(attacks 29 avoid_features FP + enum/null judgment; complements grammar decoding)*
      - [ ] Embed all enum values per function (+ pl/de/fr/es paraphrases from core_train
            gold phrases) once at build; save matrix.
      - [ ] Post-processor: if emitted value invalid-or-near-miss for an enum slot, map to
            nearest valid value iff cosine ≥ threshold; else null. Sweep threshold on val.
      - [ ] Acceptance: enum_invalid → 0, avoid_features FP −20%, F1 not worse; report
            separate free-text slots unaffected.
      - Estimate: ~0.5 d. CPU-only.
      **OUTCOME (2026-10-12): KILLED before build — premise measured false on winner-v2
      dump.** (i) invalid enum emissions = **0** (LoRA learned enums exactly; grammar
      decoding round already showed this) → nothing to map. (ii) Of the 33 avoid_features
      FPs (valid values, gold-null): **~24 gold-structural** (user explicitly requests
      avoid, e.g. "sin usar autopistas"/"ohne Autobahnen"/"en évitant les péages", but
      reference tool call in source dialog omitted avoid_features — same label-noise class
      as the lang-slot round-4 finding; model is semantically right, unfixable post-hoc),
      ~4 mention-without-avoid-cue (context/policy), ~5 adjacent-feature/polarity model
      errors (e.g. "uniknąć płatnych autostrad" → tollRoad not motorway; "po autostradzie"
      = VIA motorway, positive intent). All FPs are mention-present → an embedding
      containment check cannot separate them from true positives. Enum/null judgment needs
      intent understanding (model-side) or gold-side fixes, not similarity post-processing.

- [ ] **3. GLiNER span arbitration validator (post-hoc)**
      *(attacks q boundary errors: 37 FN + 29 FP; NEVER prompt hints — round-3)*
      - [ ] Stand up `urchade/gliner_multi-v2.1` (or ONNX export) with slot-oriented labels
            (POI / address / city / category-query); batch over test utterances.
      - [ ] Arbitration rules on dump rows: emitted free-text value overlapping a span →
            keep/truncate to span; value with NO overlapping span and low logprob → null or
            flag; span present but slot null + high logprob → candidate FN repair (val-tuned).
      - [x] Also export agreement features for the detector stack
            (`eval/train_detector.py` features); acceptance: F1 +0.3 OR detector AUC +0.01.
      - Estimate: ~1 d incl. GPU batch runs (GPU 1 or watcher script).
      **OUTCOME (2026-10-12): FAIL on F1 — arbitration harm-dominated; assets kept.**
      Built `eval/gliner_spans.py` (isolated venv `/var/tmp/gliner_venv`: gliner +
      transformers pinned 4.51.3 + protobuf/sentencepiece/tiktoken; the training venv's
      transformers 5.16+ misroutes mdeberta spm tokenizers via tiktoken) →
      `data/ner/gliner_core_test_spans.jsonl` (817 rows, 2305 spans, 815/817 rows covered;
      city 1722 / location 494 / street address 46 / POI 39 / category 4; detects inflected
      pl 'Torunia' 0.94). `eval/gliner_arbitrate.py`: locate pred value in utterance
      (word-fold), best-IoU span, truncate/extend/fn_repair + 3-block rescore
      (before/v1_only/after) + detector-features jsonl. Results: winner 0.7958→0.7856
      (−1.02), tagger v1_only 0.7813→0.7786 (−0.27); change outcomes 4 fix / 9 break
      (winner), 2/3 (tagger). Root cause: "span ⊂ region" truncation cannot distinguish
      fluff ("our hotel on Maple Street"→fix) from gold qualifiers ("Detroit,
      Michigan"/"Aéroport Charles de Gaulle"/"centro de Springfield"→breaks) — geometrically
      identical; gold boundary style is itself inconsistent ("centrum Portland" fluff vs
      "centro de Springfield" part of gold). GLiNER also fires street address on bare
      coordinates (0.61–0.85). agree=346/367: spans align with already-correct values.
      Follow-up option (low priority): detector feature from
      `gliner_arb_*_features.jsonl` on h90 tags; and digit/comma value gating would likely
      land arbitration at ~neutral but was not chased (pre-registration discipline).

- [ ] **4. Flair per-language benchmark (user-requested check)**
      - [ ] One harness run (reuse extract_ner-style eval) of Flair 2.0 `entity-german`,
            `entity-multilingual` (covers pl/es path) + en, against our gold char spans;
            compare vs wikineural spans and bert-tagger-fixd spans per language.
      - [ ] License gate: fr usable ONLY via 2.0 multilingual (1.0 ner-french is
            noncommercial) — record per-lang commercial status in the note.
      - [ ] Decision rule: adopt as feature/arbitration source only if it beats our best
            in-house spans on ≥2 of 5 langs; else document and drop.
      - Estimate: ~0.5 d.
      **OUTCOME (2026-10-12): DROP — decision rule NOT met (1/5 langs) + license trap.**
      Built `eval/flair_spans.py` (isolated venv `/var/tmp/flair_venv`, flair 0.15.1 —
      note: "Flair 2.0" hub models load into 0.15.1 via a shim: new container format
      `__cls__: TokenClassifier` + nested state_dict, embeddings object rebuilt per
      backbone — multi=mdeberta-v3-base, de=ikim-uk-essen/geberta-base,
      en=microsoft/deberta-v3-base; strict load_state_dict validates the guess; CPU
      ~1–2.5 min/model over 817 test rows) → `data/ner/flair_entity-{multilingual,
      german,english}-4class_core_test_spans.jsonl` (~2200 spans each, >97% LOC).
      `eval/compare_ner.py` (stdlib): gold free-text slot values vs span sources,
      fold-word-set exact/contained/overlap + span-precision-ish; writes
      `eval/results/compare_ner.json`. exact% per lang (best Flair vs best in-house):
      en 71.8 = 71.8 (tie), de 87.5 > 86.4 (WIN), pl 54.8 < 58.1, fr 81.0 = 81.0 (tie),
      es 78.8 < 80.0 → only de strictly better (contained% identical pattern).
      **License correction: the research note's "2.0 entity-* = commercial OK" is WRONG
      — all six entity-* models are Flukes NC 1.0 NONCOMMERCIAL per their HF cards
      (2026-10-12); commercial use needs a separate license. Combined with 1.0 ner-*
      also noncommercial, Flair is unusable commercially across ALL languages.**
      Measurement-only artifacts kept; no arbitration wiring.

## Medium priority

- [x] **5. Wikidata (CC0) alias enrichment of the gazetteer**
      - [ ] Wikidata dump query (P31 place types) for our geo scope; merge extra aliases
            into the canonicalize v2 index where GeoNames lacks them (esp. case-inflections).
      - [ ] Re-score; keep iff it adds ≥ +0.1 F1 over GeoNames-only.
      **OUTCOME (2026-10-12): KILLED — premise removed by item 1.** The GeoNames layer of
      canonicalize v2 fires 0 times on train and both test dumps (valid-form gates keep it
      inert because gold is utterance-verbatim); enriching a never-firing alias table
      cannot yield ≥ +0.1 F1. Revisit only if a future round reopens canonical-form
      rewriting (e.g. gold-side policy change toward official names).
- [x] **6. Stanza per-language NER + lemmatization as arbitration/comparator**
      *(best off-the-shelf pl NER, NKJP 88.73; per-lang lemmas feed canonicalize v2 too)*
      - [ ] Batch Stanza NER over test (5 langs); measure span overlap with gold vs GLiNER.
      - [ ] If pl beats GLiNER on place spans → pl-specific pipeline (per-language pipelines
            allowed per user), wired into the same arbitration API as item 3.
      **OUTCOME (2026-10-12): KILLED — structural, not span-quality, constraint.** The
      item-4 harness measured FOUR independent span systems (wikineural, GLiNER
      multi-v2.1, Flair multilingual/german/english): exact gold-value coverage is nearly
      identical per lang across systems (en 71.8 and fr 81.0 EXACTLY equal for all four;
      pl 53–58, de 84–88). The coverage ceiling is a property of the gold value
      distribution — categories ('parking', 'charging station'), street addresses, POI
      phrases are not classic NER entities — not of span-source quality. Item 3 further
      showed even correct spans cannot arbitrate boundaries (fluff vs gold qualifier
      geometrically identical). A better pl span source fixes neither.

## Low priority / opportunistic

- [ ] **7. HERE category taxonomy mapping at data-build**: map q free-text phrases
      ('parking', 'charging station') to HERE category ids for downstream search quality —
      changes task target, so experiment at data-build only; out of F1 metric scope.
- [ ] **8. mordecai3-style calibrated ranking** for gazetteer candidates (concept copy,
      English-only upstream): relevant only if canonicalize v2 shows precision losses.
- [ ] **9. ASR-confusion augmentation / fuzzy matching for voice input** (future car-mode
      failure): phonetic perturbation of place names using the same rapidfuzz+gazetteer
      machinery.
- [ ] **10. Photon/Nominatim as build-time validators** (offline self-host, server-weight):
       sanity-check canonicalize v2 mapping quality on a sample; never edge runtime.

## Explicitly not doing

- DBpedia / Spotlight (CC-BY-SA share-alike; user decision — Wikidata CC0 instead).
- NER as prompt hints or auxiliary training targets (round-3 failure; hint-dependent EM).
- Pointer/span-list outputs (round-9 rejection — canonical gold mismatch).
- Elasticsearch/Postgres anything at edge runtime (mordecai3 index, Nominatim server).
- Edinburgh Geoparser (GPL, academic, English-centric).
