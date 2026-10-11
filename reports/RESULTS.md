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

## Addendum: Jev decision-model probes (OpenRouter, `typesafe/jev-1.13`)

Jev (TypeSafe "System One") answers typed questions (`choice`/`noul`/`score`) against a shared
state — it cannot generate text. Probes used the OpenRouter alpha decisions endpoint via
`eval/jev_client.py` (retry/backoff, 8 workers, mock mode; key from `../.env`).

### Probe 1 — Jev as direct enum-slot extractor (`eval/jev_enum_probe.py`, core_val, n=227
enum-bearing rows, $0.013): one call/row, `noul` presence + `choice` value (single) or
per-value `noul` (multi) per enum param, threshold 0.5.

| System (enum slots only) | slot F1 | null acc | row EM |
|---|---|---|---|
| Jev zeroshot | 0.696 | 0.949 | 0.762 |
| Llama-3.2-1B FT (same rows) | **0.895** | **0.976** | **0.916** |

By language (F1): fr 0.88 / en 0.75 / es 0.73 / de 0.70 / **pl 0.36** — the reported
multilingual weakness is real and concentrated in Polish (precision 0.28). By param:
routingMode 0.82, avoid_features 0.76 (presence over-fires, null acc 0.79), transportMode
0.36. Decent zeroshot, clearly behind the FT 1B model; viable only as an enums-only fallback
where no local model exists.

### Probe 2 — Jev as verification gate (`eval/jev_verify.py`, llama-merged test rows, n=817,
$0.024, 39 s): one call/row, per slot `noul` "value grounded in utterance" (risk=1−p) or
"utterance specifies value" for predicted nulls; row risk = max slot risk.

| Detector (row-level AUC) | AUC |
|---|---|
| Jev grounding risk | 0.577 |
| min-logprob baseline | **0.750** |
| rank-avg(Jev, min-logprob) | 0.715 |

Jev ≈ the free validator flags (any-flag AUC 0.572) and adds nothing on top of logprobs.
Slot-level AUC 0.657 is strong exactly where errors are rare (routingMode 0.92, destination
0.75, via/at 1.00) and weak on the dominant errors — `lang` 0.54 (26% err) and `q` 0.56
(22% err): span-boundary and grounded-but-wrong mistakes pass a grounding check by
construction. The cookbook's Jev-verified cascade pattern targets hallucination-style
failures, which this FT model barely makes.

**Verdict**: decision models are cheap ($0.04/M in, free out), calibrated, and native on
enums/nulls, but (a) can't produce text spans, (b) Polish collapses, (c) grounding ≠
correctness — the exact failure modes that dominate here. Logprob combiner stays the
escalation gate. Untried: Jev per-slot risk as an extra feature in `train_detector.py`'s
combiner (plausible small gain on enum slots), selection-over-candidates for `q`.

Files: `eval/results/jev_enum_val{,_rows}.jsonl`, `eval/results/jev_verify_llama-1b-merged-test{,_rows}.jsonl`.

## Addendum: classical-NLP assistance (span snapping + NER hints, `plans/PLAN_NER.md`)

Question: do classic NLP tricks (span alignment, prompt-injected NER spans) help the tiny
slot-filler? Error analysis motivated both: ~66 q-span-boundary errors, 128 hallucinated
values, 14% of misses normalization near-misses. **Answer: no — both dropped.**

### A — Span-snapping post-processor (`eval/snap_spans.py`, 7 FT pred dumps)

For each non-null free-text predicted value failing the verbatim check, snap to the
best-aligned utterance span (token-window scoring, fold-normalized; enums/numbers/lang and
list items skipped item-wise).

| Model | strict F1 | snapped F1 | Δ | values snapped |
|---|---|---|---|---|
| llama-3.2-1b-ft | 0.7515 | 0.7515 | 0.0000 | 3 |
| llama-3.2-3b-ft | 0.7449 | 0.7462 | +0.0013 | 5 |
| qwen2.5-0.5b-ft | 0.7189 | 0.7189 | 0.0000 | 5 |
| qwen3-0.6b-ft | 0.7315 | 0.7315 | 0.0000 | 2 |
| gemma-3-1b-ft | 0.7259 | 0.7259 | 0.0000 | 5 |
| qwen2.5-0.5b-fullft | 0.7251 | 0.7251 | 0.0000 | 1 |
| qwen3-0.6b-fullft | 0.7343 | 0.7343 | 0.0000 | 1 |

**A2 free ablation** (re-score ORIGINAL preds with the gold-builder's inflection-tolerant
`tok_match`): F1 +2.4–3.7 pts everywhere (llama-1b 0.7515→0.7779, EM 0.6671→0.6903;
qwen3-fullft +0.0371). The FT models already copy utterance spans nearly verbatim — the
"boundary error" mass is mostly **eval-normalization artifact**: `norm_val` is fold-strict
while the gold builder tolerates inflection/clipping (Torunia↔Toruń), and gold may keep
canonical forms the user never said (`Los Angeles, CA`, <3-char tokens skip checks).
Snapping toward utterance forms therefore can't help against such gold (all snapped values
scored neutral; 1 row fixed net across all 7 models). **Verdict: drop the post-processor;
the actionable finding is a ~2.5–3.5 pt lenient-compare headroom in the eval itself.**

### B — NER hints in the prompt (`src/extract_ner.py`, `src/render.py`)

NER spans (`Babelscape/wikineural-multilingual-ner` — Davlan repo substituted: it needs
sentencepiece, absent from the venv; wikineural is the same XLM-R-base class) extracted for
all 15.5k train/aug/test utterances → `data/ner_spans.json`; LOC-first, ≤12 spans appended
as `CANDIDATE_SPANS: [...]` (rows without the field render byte-identically). llama-3.2-1b
LoRA retrained on hinted core_train+aug (identical recipe) → `train/runs/llama-3.2-1b-ner`.

2×2 on core_test (n=817; error classes = hallucinated FP slots / missed-or-wrong FN slots):

| Cell | slot F1 | row EM | FP | FN |
|---|---|---|---|---|
| baseline × plain (reference) | **0.7515** | **0.6671** | 128 | 192 |
| baseline × hinted | 0.7176 | 0.6071 | 198 | 180 |
| ner-trained × hinted | 0.7366 | 0.6524 | 128 | 207 |
| ner-trained × plain | 0.7124 | 0.5655 | 258 | 155 |

- Hints alone **hurt** the baseline (−3.4 F1): the model copies CANDIDATE_SPANS into slots
  (hallucinations 128→198).
- NER training neutralizes the copying (FP back to 128) but FN rises 192→207 — still −1.5
  F1 below baseline. The +2–3 pt expectation inverted.
- Strong hint-dependence: ner-trained without hints drops EM −10 pts (0.5655) — unusable
  robustness profile for edge deployment.
- Polish remains weakest under hints (ner-hinted pl F1 0.7106 vs de 0.7697) — the known
  multilingual-NER weakness shows up as predicted.

**Verdict: drop.** Classic-NLP assistance — post-hoc snapping or prompt NER — does not help
this model family; the 0.7515/0.667 winner and its prompt format stand. What WOULD be worth
a next round: fix the eval to lenient compare (A2), and attack the 128 true hallucinations
with constrained decoding over schema enums (not utterance spans).

Files: `eval/results/snap_spans_report.json`, `eval/results/{baseline-hinted,ner-hinted,ner-plain}{,_preds.jsonl}`,
`data/{core_train,aug_train,core_test}_ner.jsonl`, `data/ner_spans.json`,
`scripts/run_ner_experiment.sh`.

## Round 4 (2026-10-11) — teacher distillation, synthetic data, constrained decoding, negative training

Motivation: `plans/2026-10-10_improvement-todo.md` (top-3 research-ranked improvements) + the
128 true hallucinated FP slots from Round 3 error analysis. All runs use the identical
llama-3.2-1b LoRA recipe as the 0.7515 reference; core_test n=817.

### D — teacher distillation (`src/build_distill.py`)

gpt-5.4-mini (OpenAI free tier, KEY_1) annotates unused instruction_extraction texts
(pl/es/fr) using the EXACT training prompt; outputs pass the gold-builder grounding
(`param_target`, Data-Turnstile-style: keys⊆schema, verbatim/enum/number grounding,
origin/destination/via force-nulled). Teacher usage logged per-request
(logs/distill_usage.jsonl); rejects logged with reasons.

- Run 1 (seed 42, 2 fns): `data/distill_train.jsonl` 6933 rows — 1157 directions (teacher)
  + 5776 geocode-local (deterministic regex q=origin; no API call). 1.2M-token key budget.
- Run 2 (seed 43, directions-only, dedup vs all existing data): `data/distill2_train.jsonl`
  1100 rows (avoid_features-heavy).

Five retrains (train mix varies, core+aug always included):

| run | extra data | slot F1 | row EM | de | en | es | fr | pl |
|---|---|---|---|---|---|---|---|---|
| base | — | **0.7515** | **0.6671** | .7859 | .7849 | .7412 | .7077 | **.7399** |
| ds | distill + syn | **0.7548** | .6634 | .7811 | **.8178** | **.7548** | .7147 | .7085 |
| d2 | distill_dir + syn | .7500 | .6597 | **.7884** | .7970 | .7308 | .7222 | .7090 |
| d5 | distill_dir + distill2 + syn_nopl | .7431 | .6585 | .7638 | .7794 | .7395 | .7183 | .7138 |
| d4 | distill_dir + distill2 | .7459 | .6524 | .7543 | .7849 | .7358 | **.7234** | .7361 |

Findings:

- **here_directions distill works**: fn F1 .7257 → .7558 (ds, +3.0); survives across mixes.
- **geocode-local distill is poison for pl**: surface-form q (inflected "Torunia") and
  q-hallucination on route-intent rows conflict with core gold style → pl −3.1 (ds). Note it
  *helped* es (+1.4 ds) — the policy conflict is lang-specific morphology, not universal.
- **No mix dominates** — each gain in some langs is paid in others. Identical-recipe LoRA
  reruns vary ±1.5–2 pts per lang; effect sizes here are inside that noise band. Squeezing
  more from data scaling needs seed-averaged evals (compute-bound) — out of scope this round.
- **Structural ceiling found — `lang` slot**: core gold is context-dependent (same utterance
  appears with lang=dialog-lang AND lang=null; the resolver decides). Single-turn input
  cannot resolve it: base FNs on gold-lang rows = en 7 / de 18 / pl 23 / es 28 / fr 26.
  ~2–3 pts F1 headroom locked behind multi-turn/context features, not data volume.
- Acceptance (pl/es/fr each +3) **FAIL**; ds kept as the best-F1 variant but the shipped
  winner stays `train/runs/llama-3.2-1b/best` (round-to-round consistency).

### S — synthetic closed loop (`src/build_synthetic.py`)

Templated generator seeded from core vocab (places harvested from targets/sources), phrasing
tables built so every constraint phrase contains its ENUM_STEMS grounding stem; validators =
keys⊆schema + grounding + coverage(fn×lang×slot, 98 enum combos); shingle-dedup vs
core+aug+distill. 5700 rows (1140/lang) → syn_train.jsonl. Effect: en +3.3 in ds (syn is the
only new en data), but pl-only analysis pointed at syn-pl templates for geocode.q multi-word
nulling; a syn_nopl ablation (d5) did not recover pl. Verdict: useful generator, do not ship
the current templates wholesale; regenerate per-lang after template audit.

### G — grammar-constrained decoding (`src/target_schema.py`, `src/export_grammar.py`, `eval/eval_constrained.py`)

- Per-fn JSON schemas → xgrammar (compile_json_schema any_order) + llama.cpp GBNF exports
  (`grammars/<fn>.gbnf`, 25 fns; enums incl. lang ∈ {de,en,es,fr,pl}, all params required +
  nullable).
- Full-test A/B on the FT adapter: F1 0.7515→0.7510, EM 0.6671→0.6659, parse 1.0 both,
  null_acc 0.9626 both — **metrics-neutral with a hard schema-validity guarantee** (FT model
  already parses perfectly on-distribution; the guarantee pays off OOD/edge). Fits in VRAM
  alongside training (~3.4GB for 1B bf16, bs 4).

### N — Hammer-style negative training (`src/build_negatives.py`)

Targets the 128 true hallucinated slots (geocode.q 29, avoid_features 29, lang 46
[structural, off-limits], routingMode 6, transportMode 5, nominatim q 6, tomtom query 5,
search q 2). Families: B1 avoid-use-mentions ("I use toll roads" → avoid null), B2 chatter
place-mentions (no search intent), B3 vehicle-incidental clauses, A constraint-phrase under
geocode-family schema. Grounding CANNOT validate mention-vs-extraction (enum stems present)
→ lexical intent guards reject any route/search phrasing. 1067 rows + 33 val
(data/neg_train.jsonl). Run `llama-3.2-1b-neg` (core+aug+neg, 15,484 rows):

`llama-3.2-1b-neg` (core+aug+neg, 15,484 rows): slot F1 0.7368 (−1.5), EM 0.6548 (−1.2),
null_acc 0.9632 (≥0.96 held), parse 1.0. Hallucination counters vs base: true-halluc FP
slots 128→126 (noise; lang 46→41, q 37→41, avoid_features 29→31), total FP 185→193.
**Acceptance FAIL** — trap-template negatives at ~7% of mix do not generalize to real-dialog
hallucinations; the shift between slots is within run variance. Consistent with the round's
lesson: at 1B+LoRA the remaining error mass is structural (lang label context, canonical-form
gold) or needs inference-time mechanisms, not more data of this kind.

Files: `src/build_{distill,synthetic,negatives}.py`, `data/{distill,distill2,syn,neg}_{train,val}.jsonl`,
`scripts/run_{distill_retrain,d2_retrain,d5_d4_chain,neg_retrain}.sh`,
`eval/results/llama-3.2-1b-{ds,d2,d4,d5}-ft{,-g}.json`, `grammars/`.

## Round 5 (2026-10-11) — BIO-tagging formulation (param names as output vocabulary)

Plan + full analysis: `notes/2026-10-11-tagging-formulation-round4.md`. Two user ideas
combined: (1) parameter names injected so the model attends to schema-relevant input parts,
(2) recast generation as per-word tagging (utterance → one tag per whitespace word) instead of
JSON synthesis. Decision rules fixed before running: Track A (llama LoRA emitting tags) adopts
if F1 ≥ ~0.735 AND shorter decode; Track B (BERT encoder tagger) adopts as edge candidate if
within ~2 pts of the 0.7515 JSON winner.

### Data — `src/build_tags.py`

`data/*_tags.jsonl` add `bio_tags`: O / B-<param>[:<enumval>] / I-<param>, one per
whitespace word, optional leading `lang:<v>` pseudo-tag. Enum canonical values are embedded
in the tag (`B-avoid_features:tollRoad`) because fr/pl/es utterances never contain the
canonical form (péage → tollRoad). Alignment cascade: substring → enum-stem word →
monotonic subsequence → appendage-prefix (≤2 leftover chars); rest nulled honestly.

- Gold roundtrip (decode(tags)==target, norm): core 92.1/91.4/91.8% (train/val/test),
  aug 99.7/100% → **structural ceiling on core_test: 91.1% of non-null slots / 91.8% of
  rows** recoverable from tags alone.
- Ceiling losses: Polish inflection slice-vs-canonical (utterance "Krakowa", gold
  "Kraków", ~1%), US-state appendage ("Seattle, WA" → q="Seattle", ~2%), 2-char codes
  that pass the old substring check but have no span ("DE", "es", ~0.5%).

### Track B — JointBERT-style encoder (`train/train_tagger.py`, `eval/eval_tagger.py`)

bert-base-multilingual (wikineural init) + schema string prepended (`name=v1|v2` for enums)
+ tag head + lang head (CLS). 5 epochs, val tag-acc 98.34%. core_test: slot F1 0.7152
(P .7088 / R .7218), row EM 0.6279, null-acc .9585, parse 1.0; 5.0 ms/row GPU (bs 64),
96.6 ms/row CPU — full comparison table in Track A below.

by-lang F1: tagger de .7642 en .7306 es .6978 fr .7327 pl .6403 vs JSON
de .7859 en .7849 es .7412 fr .7077 pl .7399 — tagger wins fr (+2.5) only; pl collapses
(−9.9, morphology slice losses + lang head). Error profile mirrors the generative models:
fn q 95, fn+fp lang 150 (84+66), fp q 34, fp avoid_features 29 — tagging removes value
fabrication but keeps the same dominant confusion classes.

**Verdict: REJECT as edge candidate** (0.7152 is 3.6 pts below winner; rule allowed −2).
8.6× faster on GPU and CPU-viable at ~97 ms/row, but the structural ceiling (91.1% slots)
plus encoder capacity cap it below the adoption bar.

### Track A — llama-3.2-1b LoRA emitting tags (`src/render.py` TAG_SYSTEM_PROMPT, `eval/eval_tags.py`)

Trained on core+aug tags (14,432 rows). Greedy decode with one sampled regen on
tag-count mismatch, then `decode_tags` → JSON, scored identically to eval_extract.

| system (core_test, n=817) | slot F1 | P | R | row EM | null acc | parse | ms/row |
|---|---|---|---|---|---|---|---|
| JSON winner (llama-3.2-1b LoRA) | **0.7515** | .7550 | .7480 | **0.6671** | .9626 | 1.0 | 43.1 gpu |
| llama-3.2-1b LoRA + tags (Track A) | 0.4745 | .5246 | .4331 | 0.4529 | .9658 | 1.0 | 126.3 gpu |
| BERT tagger (Track B) | 0.7152 | .7088 | .7218 | 0.6279 | .9585 | 1.0 | **5.0 gpu / 96.6 cpu** |

Track A detail: tag_count_ok 67.3%, 343/817 rows needed sampled regen, 267 left-align
fallback; avg 36.4 new tokens; by-lang F1 de .5993 en .4141 es .4685 fr .4415 pl .4280.
Emit-count delta vs required (tags−words): 550 exact, 131 short, 136 long — errors
concentrate at ±1. **Diagnosis: small-LM word-counting failure** — the contract "emit
exactly one tag per whitespace word" demands implicit length control; JSON generation
has natural closure (braces) and no counting requirement. Both decision-rule criteria
FAIL: F1 0.4745 ≪ 0.735, and decode is 2.9× slower (126.3 vs 43.1 ms/row; regen pass
doubles work despite fewer avg tokens).

**Round verdict: both tracks REJECT; the JSON-generative winner stands.** Tagging
formulation is cap-able for encoder models (91.1% structural ceiling) and length-hostile
for 1B decoders. Artifacts kept: `data/*_tags.jsonl`, `src/build_tags.py`,
`train/train_tagger.py`, `eval/eval_tags.py`, `eval/eval_tagger.py`,
`eval/results/{bert-tagger-ft,llama-3.2-1b-tags-ft}*.json[l]`, adapters
`train/runs/{bert-tagger,llama-3.2-1b-tags}/best`.

## Artifacts

- Adapters: `train/runs/<candidate>/best/`; full-FT: `train/runs/<candidate>-full/best/`.
- Winner merged: `train/runs/llama-3.2-1b/merged/` (base + LoRA merged, bf16 safetensors).
- Metrics: `eval/results/<candidate>-(zeroshot|ft|fullft).json`; per-row predictions:
  `eval/results/<tag>_preds.jsonl`; error tooling: `eval/analyze_errors.py`.
