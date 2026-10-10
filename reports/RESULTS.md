# Slot Filling / Parameter Extraction — Final Results

Task: (function schema + user utterance) → compact JSON, one key per schema parameter, value
verbatim-normalized from the utterance or `null` when not stated. 5 languages (en/de/pl/fr/es).
Eval on `core_test.jsonl` (817 execution-verified gold rows, balanced langs).

## Setup

- Data: core 9,186 train rows from fcft verified dialogs + 5,818 heuristic-augmented rows from
  instruction_extraction (shingle-deduped vs core, lang-balanced). Targets honest-null
  (unspoken params → null). 14,415 train rows after seq-len 2048 filtering; val 759.
- Training: LoRA r=32 α=64 all-linear, lr 1e-4, 3 epochs, bf16, early stop (patience 3),
  completion-only loss (string-level prompt/completion split). Full-FT ablation: lr 2e-5.
- Eval: greedy generate, max 220 new tokens, n=817. Metrics: parse rate; slot P/R/F1 over
  non-null gold slots (normalized compare, comma-spacing collapsed); null-slot accuracy;
  row exact-match. Result files: `eval/results/*.json`, per-row dumps `*_preds.jsonl`.

## Main table (test, n=817)

| Model | Mode | Parse | P | R | F1 | Null acc | Row EM | Train time |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | zeroshot | 0.980 | 0.033 | 0.095 | 0.049 | 0.452 | 0.100 | |
| Qwen2.5-0.5B | LoRA | 1.000 | 0.728 | 0.710 | 0.719 | 0.964 | 0.639 | 1h50m |
| Qwen2.5-0.5B | full-FT | 1.000 | 0.723 | 0.727 | 0.725 | 0.963 | 0.644 | 1h30m |
| Qwen3-0.6B | zeroshot | 1.000 | 0.073 | 0.298 | 0.118 | 0.302 | 0.006 | |
| Qwen3-0.6B | LoRA | 1.000 | 0.728 | 0.735 | 0.732 | 0.959 | 0.639 | 2h30m |
| Qwen3-0.6B | full-FT | 1.000 | 0.742 | 0.727 | 0.734 | 0.964 | 0.647 | 1h30m |
| Llama-3.2-1B | zeroshot | 0.947 | 0.080 | 0.197 | 0.114 | 0.589 | 0.048 | |
| **Llama-3.2-1B** | **LoRA** | **1.000** | **0.755** | **0.748** | **0.7515** | **0.963** | **0.667** | 1h45m |
| gemma-3-1b | zeroshot | 0.999 | 0.028 | 0.083 | 0.042 | 0.486 | 0.011 | |
| gemma-3-1b | LoRA | 1.000 | 0.738 | 0.711 | 0.726 | 0.962 | 0.639 | 2h40m |
| Llama-3.2-3B | zeroshot | 0.984 | 0.070 | 0.189 | 0.102 | 0.571 | 0.050 | |
| Llama-3.2-3B | LoRA | 1.000 | 0.725 | 0.766 | 0.745 | 0.953 | 0.649 | 3h35m |

(FT rows re-scored with v2 normalization: comma-spacing collapsed; deltas ≤ +0.2 pt.
Full-FT timings on free GPU; zeroshot requires no training.)

## Conclusions

1. **Winner: Llama-3.2-1B-Instruct + LoRA** — best F1 (0.7515), row EM (0.667), parse 1.0.
   Merged export: `train/runs/llama-3.2-1b/merged/` (~2.5 GB bf16; ~0.8 GB as Q4 GGUF).
2. **Scale doesn't help here**: 3B < 1B (0.745 vs 0.7515) despite lower val loss (0.0486 vs
   0.0367-family) — the 3B over-extracts (R 0.766 but P 0.725, null acc 0.953). Task is
   extraction-fidelity-bound, not capacity-bound. 1B is the sweet spot for edge deployment.
3. **LoRA ≈ full-FT**: +0.3/+0.6 F1 for 2–6× the checkpoint size — LoRA is the right
   default; merge for deployment.
4. **Fine-tuning is decisive**: F1 0.04–0.12 → 0.72–0.75, parse → 1.0 everywhere. Zeroshot
   models hallucinate values for unspoken slots (null acc 0.30–0.59 vs ≥0.95 FT) — the
   honest-null behavior must be taught.
5. **Language gap**: pl/es/fr trail en/de by ~4–8 pts for every model (llama-1b: en/de 0.785
   vs fr 0.702). Likely causes: en-boosted augmentation quotas + inflection-heavy
   morphologies (pl) + place-name normalization (exonyms like Varsovie/Warszawa).

## Per-language F1 (FT LoRA)

| Model | en | de | pl | fr | es |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.742 | 0.757 | 0.662 | 0.722 | 0.705 |
| Qwen3-0.6B | 0.764 | 0.756 | 0.709 | 0.719 | 0.709 |
| **Llama-3.2-1B** | **0.785** | **0.786** | 0.740 | 0.702 | 0.741 |
| gemma-3-1b | 0.745 | 0.751 | 0.684 | 0.703 | 0.739 |
| Llama-3.2-3B | 0.771 | 0.764 | 0.699 | **0.749** | 0.738 |

(3B is notably better on fr — only place it beats 1B.)

## Per-function F1 (FT LoRA)

| Function | n | Q2.5-0.5B | Q3-0.6B | L-3.2-1B | gemma | L-3.2-3B |
|---|---|---|---|---|---|---|
| here_geocode | 301 | 0.720 | 0.723 | **0.752** | 0.730 | 0.741 |
| here_directions | 264 | 0.692 | 0.723 | 0.726 | 0.727 | **0.727** |
| nominatim_geocode | 83 | 0.771 | 0.779 | 0.792 | 0.774 | **0.802** |
| tomtom-geocode | 44 | 0.805 | **0.850** | 0.805 | 0.805 | 0.815 |
| here_search_places | 43 | 0.652 | 0.596 | **0.711** | 0.133 | 0.667 |
| osrm_route | 44 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | N/A¹ |
| open_meteo_forecast | 35 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | N/A¹ |
| tomtom-traffic | 2 | N/A¹ | N/A¹ | N/A¹ | N/A¹ | N/A¹ |

¹ These functions have **zero non-null gold slots** in test (params not speakable: lat/lon
coordinates etc.) → F1 undefined; correct behavior is all-null output (covered by null acc).
gemma-3-1b additionally fails here_search_places (0.133) — an outlier specific to gemma.

## Error analysis (winner, llama-3.2-1b-ft)

128 hallucinated-value slots (gold null, pred non-null) + 193 missed/wrong (of ~1,700
non-null gold slots); only 14% of misses are near-miss normalization gaps.

- **`lang` dominates both directions** (100 FN + 46 FP): the "lang = dialog language" rule
  confuses when the utterance discusses foreign places (French text about Varsovie→Lisbonne).
  Hardest single behavior taught.
- **`q` boundary ambiguity** (37 FN + 29 FP): "our hotel on Maple Street" (gold `Maple
  Street`), "Los Angeles, CA" vs `Los Angeles` — gold takes minimal referent; model sometimes
  over/under-spans. Inherent task ambiguity, not model capacity.
- **`avoid_features`** (8 FN + 29 FP): inflected forms ("autobahn vermeiden" without explicit
  motorway word) and compound constraints ("Wohnmobil" trip → truck?) — lexicon gaps in
  training data, and some gold nulls where the model infers a plausible avoid.
- Coordinate gold values ("aux coordonnées 45.764, 4.8357") are correctly extracted when
  spoken — honest-null policy only nulls *unspoken* params.

## Failure detection (when is the model wrong?)

`eval/detect_uncertainty.py` scores three signal families per test row against actual row
errors: token logprobs (greedy), self-consistency (3 samples, temp 0.7), and deterministic
validators (schema keys, value-groundedness in the utterance, enum membership, langid check).
Reports: `eval/results/<tag>-uncert…/{,*_rows}.jsonl` → `eval/results/{llama-1b-merged,
qwen3-0.6b-ft,qwen2.5-0.5b-ft}.json` (+ per-row `_rows.jsonl` for threshold tuning).

| Model | errors (rows) | AUC min-logprob | AUC self-cons | AUC validators | AUC best combo |
|---|---|---|---|---|---|
| Llama-3.2-1B | 274/817 | **0.755** | 0.655 | 0.569 | 0.735 |
| Qwen3-0.6B | 295/817 | 0.757 | 0.648 | 0.55 | 0.736 |
| Qwen2.5-0.5B | 295/817 | **0.765** | 0.669 | 0.57 | 0.752 |

Findings:

1. **Min token logprob of the generated JSON is the best single detector** (AUC 0.75–0.77,
   free — no extra inference). Combining signals does NOT beat it (rank combos ≤ 0.75).
2. **Practical cascade rule** (all three models behave the same): escalate the ~30%
   lowest-logprob rows → catch ~50% of all errors (~100 false alarms); escalate 50% →
   catch ~73%. At 10% escalation: catch ~20% of errors with only 18–28 false alarms.
3. **Validators catch hallucinations only**: `ungrounded` (value not in utterance) fires
   ~130 times with precision ~0.52 and error-recall ~0.25; keys/enum/langid flags ≈ never
   fire. Miss-type errors (confident wrong `null`) are invisible to them — that's why
   logprob + self-consistency are needed.
4. The tiny built-in 5-lang detector agrees with gold `lang` on 94.4% of test rows.
5. Implication for a 1B→LLM cascade: a logprob threshold calibrated to escalate ~30% of
   traffic halves the error rate of the auto-accepted answers at ~3× cost on the escalated
   share only.

## Improving failure detection (learned detectors, probes, ensemble, conformal)

Setup: to get *labeled* (natural) errors without touching core_test, the three LoRA models
were retrained on 90% of core_train (`data/core_train_h90.jsonl` + aug; `src/split_holdout.py`,
dialog-id-grouped, lang×fn stratified; the held-out 10% = 915 rows is the detector-training
set). `eval/detect_uncertainty.py` was extended: k=5 samples, per-slot logprobs (teacher-forced
re-forward), per-slot semantic entropy (norm_val clustering), and last-token hidden states
every 4th layer → `<tag>_features.jsonl` + `<tag>_hidden.npz`. Detectors fit on holdout
(`eval/train_detector.py`, numpy logistic + 5-fold CV) and evaluated once on core_test.

### Detector AUC on core_test (n=817)

| Detector | llama-h90 | llama-merged | qwen2.5-h90 | qwen3-h90 |
|---|---|---|---|---|
| min-logprob (prev best) | 0.756 | 0.750 | 0.745 | 0.727 |
| **Logistic combiner (15 features)** | **0.770** | **0.763** | **0.772** | 0.725 |
| Hidden-state probe (layer 8, CV-chosen) | 0.641 | 0.602 | 0.647 | 0.683 |
| Stack (probe + scalars) | 0.654 | 0.610 | 0.642 | 0.691 |
| Slot semantic entropy (se_max) | 0.676 | 0.670 | 0.683 | 0.662 |

1. **Learned combiner wins on 3/4 models** (+1.2–2.7 pts over min-logprob; 0.763–0.772) and
   transfers to the full-data winner model (fit on llama-h90 holdout → 0.763 on merged).
   Qwen3 is the exception (tie). CV AUC on holdout 0.81 → some overfit to the 915-row set.
2. **Hidden-state probes disappoint** (0.60–0.68 vs 0.80+ in the BFCL literature on larger
   non-FT models): consistent with the known caveats — fine-tuned small models + 915 training
   rows for 2048-dim states. Mid layers (h8) CV-best (0.726) but generalize worst.
3. **Ensemble (majority vote of the h90 trio)** is an *accuracy* win, not a detection win:
   voted row EM 0.6634 > all members (0.6561 / 0.6414 / 0.6450) despite −10% training data —
   but slot-disagreement AUC is only 0.652 and cross-member min-logprob 0.739, below the
   single-model combiner. Not stacked (combos never beat the combiner in any test).
4. **Conformal abstention** (combiner score, threshold on holdout): targeting ≤5% error
   accepts ~27–29% of rows at 92–93% accuracy; ≤10% → ~44% coverage at ~85.5%; ≤15% →
   ~63% at ~79%. Slightly optimistic (realized error ~1.4× target) — calibrate τ on a
   larger error set for production.

Files: `eval/results/detector_<c>-h90-holdout.json`, `eval/results/ensemble_trio{,_rows}.jsonl`,
`eval/results/<c>-h90-{holdout,val,test}{,_features,_hidden,_rows}.*`.

## Artifacts

- Adapters: `train/runs/<candidate>/best/`; full-FT: `train/runs/<candidate>-full/best/`.
- Winner merged: `train/runs/llama-3.2-1b/merged/` (base + LoRA merged, bf16 safetensors).
- Metrics: `eval/results/<candidate>-(zeroshot|ft|fullft).json`; per-row predictions:
  `eval/results/<tag>_preds.jsonl`; error tooling: `eval/analyze_errors.py`.
