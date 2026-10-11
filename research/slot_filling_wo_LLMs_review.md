# Review: slot_filling_wo_LLMs.md — LLM-free slot filling for Polish

Assessment of the research note, cross-referenced with this repo's measured results
(`reports/RESULTS.md`).

## As a survey/architecture — solid

The taxonomy is correct and the Polish-specific instincts are right:

- BILOU/CRF sequence labeling as the classical workhorse.
- Dependency parsing for free word order ("z"+gen → origin, "do"+gen → destination);
  DependencyMatcher rules are permutation-invariant, so one rule covers
  "Do Krakowa z Warszawy wyznacz trasę" and its word-order variants.
- Rasa roles/groups separating entity *type* (location) from *semantic function*
  (origin/destination) — the right conceptual framing.
- Deterministic → syntactic → statistical cascade is the textbook reference architecture.
- The normalization section (Morfeusz 2 lemmatization + fuzzy matching against
  TERYT/SIMC gazetteer, Duckling/dateparser for times and numbers) is the strongest part.

## As "research results" — there are none

It is a literature overview with zero experiments, datasets, or numbers. The central
claim — accuracy comparable to neural models at a fraction of resources — is asserted,
not demonstrated. This repo's own evidence complicates it:

- Zeroshot small LLMs are indeed terrible (F1 0.05–0.12) — superficially supporting the
  doc. But one LoRA fine-tune gets F1 0.75 / null-acc 0.96 / EM 0.67 across
  **5 languages with schema-conditioning** — vs the doc's stack needing per-language
  models, rules, and lexicons that multiply with every function and schema change.
- The doc's core bet (external NER spans as the extractor backbone) was empirically
  tested here — the NER-hint experiment (wikineural) and `eval/snap_spans.py` ablation
  were **run and dropped** for not adding value.
- The <10ms / <20MB regime is real only for MCU-class targets; Rasa lands at
  100–500MB / 10–30ms per its own table — not far from a Q4 0.5B with greedy decoding,
  which is deterministic in practice.
- Snips NLU is archived/unmaintained, and its Polish inflection support is rated
  "medium" in the doc's own comparison table.
- The doc ignores the problem this repo found central: **honest nulls and abstention**
  (when *not* to fill a slot), plus ASR noise.

## Where it is genuinely useful for this project

Polish is the lowest per-language F1 for most models (0.74 for the winner,
Llama-3.2-1B LoRA) — inflection is the real bottleneck, as the doc argues. Its
normalization layer is extractor-agnostic and would slot in cleanly as a
post-processor:

- Morfeusz 2 lemmatization + fuzzy gazetteer matching (TERYT/SIMC) for geo/enum slots.
- dateparser / Duckling for time and numeric slots.

This would attack exactly the remaining errors of the winner without touching the
model.

## Verdict

Good map of the classical landscape, but treat "comparable accuracy" as unverified;
the empirically supported play here remains fine-tuned small LMs + classical
*normalization* as a hybrid post-processor.

## Possible follow-up

Small ablation: Morfeusz/fuzzy-gazetteer normalization applied to the winner's test
predictions (`*_ft_preds.jsonl` + lenient re-scoring, same harness as
`eval/snap_spans.py`) to quantify the hybrid claim.
