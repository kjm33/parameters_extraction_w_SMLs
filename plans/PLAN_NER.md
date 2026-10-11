# PLAN_NER — Classical-NLP assistance for the tiny slot-filler (approved)

Question: does classic NLP (NER / span alignment / lemmatization) help small models with
slot filling? Baseline to beat: llama-3.2-1b LoRA F1 0.7515 / row EM 0.667 on core_test
(n=817). Error motivation: ~66 q-span-boundary errors (classic NER territory), ~128
hallucinated values (constrained-decoding territory), 14% of misses are normalization
near-misses.

## TODO — Experiment A: span-snapping post-processor (CPU, minutes)

- [x] **A1** `eval/snap_spans.py`: for each of the 7 FT preds dumps
      (`eval/results/{llama-3.2-1b,llama-3.2-3b,qwen2.5-0.5b,qwen3-0.6b,gemma-3-1b}-ft_preds.jsonl`
      + 2 fullft; utterance joined from `data/core_test.jsonl` by id):
      for each non-null free-text predicted value (skip `lang`, enum-typed params, numbers;
      lists item-wise): if it passes the verbatim check → keep; else snap to the
      best-aligned utterance span (token-window scoring, fold-normalized).
- [x] **A2** free ablation: re-score ORIGINAL preds with inflection-tolerant compare
      (tok_match) → quantifies how much of the measured error is eval-normalization artifact
      (gold builder is inflection-tolerant, eval norm_val is fold-strict — eval is stricter
      than the data builder; snapping toward utterance forms can HURT vs gold, e.g.
      Torunia/Toruń; gold may keep 'Los Angeles, CA' because <3-char tokens skip checks).
- [x] **A3** report F1/EM before/after per error class (q-boundary, canonical-appendage
      'Lyon, France', inflection) → `eval/results/snap_spans_report.json`.

## TODO — Experiment B: NER hints in the prompt

- [x] **B1** `src/extract_ner.py`: `transformers.pipeline('ner')` with
      `Davlan/xlm-roberta-base-ner-hrl` (~670MB → HF cache /var/tmp/fcft_hf_cache; no venv
      change), keep LOC/GPE/PER spans; run over core_train + aug_train + core_test →
      `data/ner_spans.json` cache; spot-check 20 rows × 5 langs manually.
      *(executed with `Babelscape/wikineural-multilingual-ner` — Davlan repo needs
      sentencepiece, absent from the venv; same XLM-R-base class, LOC/PER kept.)*
- [x] **B2** `src/render.py`: if row carries `ner_spans`, append a
      `CANDIDATE_SPANS: ["Toruń", ...]` line to user content (cap 12 spans, LOC first);
      rows without the field must render byte-identically (backward compat).
- [x] **B3** inject spans into the 3 datasets, retrain llama-3.2-1b LoRA — identical
      recipe, FULL core_train+aug (apples-to-apples vs 0.7515 baseline) →
      `train/runs/llama-3.2-1b-ner` (~1.75h GPU behind free-GPU watcher; fcft siblings
      never killed, exit-120 benign).
- [x] **B4** 2×2 eval on core_test (dump preds): baseline×plain (ref 0.7515),
      baseline×hinted (isolates info value of hints), ner-trained×hinted (MAIN),
      ner-trained×plain (robustness without hints); compare F1/EM + q error-class deltas.

## TODO — wrap-up

- [x] update `reports/RESULTS.md` (+ new section), `reports/OVERVIEW.md` (§7 plain
      language), `AGENTS.md` (new scripts/commands).
- [x] verdict: keep/drop each intervention; final expectation if kept: +2–3 F1 → ~0.78.

## Known risks

- Snapping can hurt on inflected utterances (measured per class in A3 before any adoption).
- Davlan NER is weaker on Polish — baseline×hinted cell isolates this.
- NER hint adds prompt tokens (~latency/edge cost) — only adopt if B4 shows real gains.

## OUTCOME (round 3, closed)

**Both dropped.** Full write-up: `reports/RESULTS.md` §"classical-NLP assistance".

- **A (snapping): no-op.** ≤5 values snapped per model, F1 delta ≤ +0.0013 (1 row fixed
  net across 7 models); all snapped values neutral vs strict gold. FT models already copy
  utterance spans near-verbatim. Free A2 finding: lenient (tok_match) re-scoring of the
  SAME preds is +2.4–3.7 F1 / +2–3 EM everywhere — the "q-boundary error" mass is mostly
  eval-normalization artifact (eval fold-strict vs gold-builder inflection-tolerant).
  Follow-up worth doing: make eval lenient; do NOT ship the post-processor.
- **B (NER hints): negative.** 2×2 (F1/EM): baseline-plain 0.7515/0.6671 >
  ner-hinted 0.7366/0.6524 > baseline-hinted 0.7176/0.6071 > ner-plain 0.7124/0.5655.
  Hints alone make the baseline hallucinate (FP 128→198 — copies CANDIDATE_SPANS into
  unset slots); NER training cancels the copying (FP 128) but FN rises 192→207, still
  −1.5 F1 vs baseline; without hints the retrained model collapses (EM −10 pts).
  Polish weakest as predicted (ner-hinted pl F1 0.7106). Prompt format and winner model
  unchanged; artifacts kept (`data/*_ner.jsonl`, `train/runs/llama-3.2-1b-ner`,
  `scripts/run_ner_experiment.sh`).
- Next real lever for the 128 hallucinations: constrained decoding over schema enums —
  not classic NLP.
