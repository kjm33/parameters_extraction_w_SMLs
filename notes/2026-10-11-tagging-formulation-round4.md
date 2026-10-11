# 2026-10-11 — Tagging formulation (round 4): param-name tags + seq2seq/BIO, encoder tagger

Context: two ideas discussed — (1) inject parameter names into model input so attention
detects which input parts relate to parameters; (2) recast slot filling as seq2seq
tagging: per input word output empty token or parameter name, instead of generating JSON.

## What the repo already tells us

- **Idea 1 (inject param names) — weak form already exists**: `user_content()` in
  src/render.py puts the full schema (all param names, enums, required) into the prompt
  before the utterance. In a decoder LLM the "attention finds the related input parts"
  mechanism is already implicit: when generating key `"q"`, the model attends back to both
  the schema key and the utterance span.
- **Idea 2 (seq2seq tagging) — feasibility measurable from data** (core_val, 307 non-null
  non-lang values):
  - 67% verbatim normalized substrings of the utterance
  - 30% enums (`avoid_features`, `routingMode`, ...): canonical value ("toll") never
    appears in fr/pl/es utterances ("péages", "płatne") — pure span-tagger needs the
    `ENUM_STEMS` lexicon as reverse mapping at decode
  - 3% canonical appendage ("Seattle, WA" when user said "Seattle") — irrecoverable from
    the utterance, strict-EM loss for any tagger
  - values avg 1.17 words; `lang` is *inferred*, not a span (dominates current errors:
    146 of ~321 slot errors per RESULTS.md)
- **Snap-span ablation** (eval/results/snap_spans_report.json): forcing FT predictions
  onto utterance spans changed F1 by ~0.0 (≤5 values snapped / 817 rows) — FT models
  already emit near-verbatim spans, so tagging's main wins = eliminating value
  fabrication (128 hallucinated slots) + shorter/faster decoding, not normalization.

## Verified feasibility numbers (all splits, 9,498 non-null values)

- Value→span alignment: 59.3% verbatim substring, 27.2% enum-via-lexicon, 3.4%
  token-aligned, 0.8% unaligned (mostly `48.2082, 16.3738` comma-space artifacts —
  fixable with v2 normalization; residual ~0.5% → nulled in tag targets, logged).
- Effective label set: 23 params cover 99.9% of non-null values → BIO ≈ 46 labels +
  ~20 enum-value labels ≈ **70 labels** (154-param union is mostly never-non-null noise).
- Trainer needs zero changes: `train_sft.py` takes `--train/--val` file lists;
  `render.py` already branches on row fields (`ner_spans` precedent).
- wikineural (bert-base-multilingual-cased) already in HF cache for track B.

## Plan (approved)

Track A — LLM BIO tagging (llama-3.2-1b LoRA), Track B — encoder tagger
(JointBERT-style). Enums: canonical value in tag (`B-avoid_features:toll`).
Baseline: llama-3.2-1b LoRA F1 0.7515 / EM 0.667 on core_test (n=817).

### Track A
1. `src/build_tags.py` → `data/{core,aug}_{train,val,test}*_tags.jsonl` (same rows +
   `bio_tags` field): alignment unit = whitespace/hyphen-split word; lookup order:
   substring → enum lexicon (canonical into tag) → tok_match align (contiguous or
   discontinuous = multiple `B-` segments) → null (logged to `data/tags_build_stats.json`).
   `lang` = leading pseudo-tag `lang:fr`; decode contract: tag i ↔ word i, segments
   joined ", " (reproduces gold like "Św. Kierka, Chicago"), O-only params → null.
2. `src/render.py`: branch on `bio_tags` presence → tag-mode system prompt; assistant =
   tag string. Chat-template machinery untouched.
3. Train: `train_sft.py --candidate llama-3.2-1b --train data/core_train_tags.jsonl,data/aug_train_tags.jsonl ...` (~2h, GPU watcher if busy).
4. `eval/eval_tags.py`: generate → validate (word-count match; 1 regen @ t=0.7;
   left-align fallback) → decode to JSON → reuse eval_extract scoring verbatim →
   `eval/results/llama-3.2-1b-tags-ft.json` + `_preds.jsonl` (analyze_errors-compatible).
   Log output tokens + tok/s vs baseline.

### Track B
1. `train/train_tagger.py`: bert-base-multilingual-cased (wikineural, cached), input =
   `[CLS] param names [SEP] utterance [SEP]`; token-classification head over ~70-label
   set (first-subword labeling) + aux 5-way `lang` head on [CLS]. ~15 min train.
2. `eval/eval_tagger.py`: argmax → same tag→JSON contract → same scorer →
   `eval/results/bert-tagger-ft.json`; log ms/row.

### Decision rules (reports/RESULTS.md round-4 section)
- A: adopt if F1 ≥ ~0.735 (parity −1.5pt) AND shorter decode (anti-fabrication +
  guaranteed parse).
- B: adopt as edge candidate if F1 within ~2 pts of 0.7515 (5-10× faster than 1B).

Known structural losses, documented honestly: canonical appendage gold (~3% of values —
unspannable; lenient tok_match scoring shows the fair view), `lang` errors unchanged.

Effort ≈ 1.5 days. Optional if time permits: qwen2.5-0.5b tags variant, GBNF tag grammar
via existing `eval_constrained.py`.
