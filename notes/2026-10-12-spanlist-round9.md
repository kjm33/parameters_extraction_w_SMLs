# 2026-10-12 — Span-list formulation (round 9): pointers instead of strings

Context: user proposal — train the SAME winner model (llama-3.2-1b LoRA recipe) to emit
span lists instead of JSON string values. Free-text params become `[first,last]` 1-based
whitespace-word index pairs into the utterance; enum params stay canonical literals
(canonical values are never spoken: "płatne drogi" → tollRoad); nulls stay null (keys
preserved). Motivation: the winner's 2nd-largest error class is free-text fabrication /
boundary errors on `q` (29 hallucinated + 37 FN/FP q slots); spans make free-text values
structurally grounded (cannot be fabricated) and shorten decode.

Prior art that frames the risk:

- Round-5 Track A (BIO tags, same model): F1 0.4745 — REJECTED for the per-word counting
  contract. Span lists drop per-word alignment but keep absolute index arithmetic → the
  numbered-utterance variant (`1|word 2|word …` in the prompt, indices copyable) is the
  mitigation under test.
- Span ceiling on v2 core_test (measured at plan time): 97.9% of non-null slot VALUES
  recoverable via the build_tags alignment cascade (up from 91.1% once lang dropped);
  row-level gold roundtrip of `span_target`: core_test 0.9388, core_train 0.9376,
  aug_train 0.9962 (data/spans_build_stats.json). Losses: canonical appendages
  ("Los Angeles, CA"), inflected canonicals (gold "Kraków" vs surface "Krakowa") —
  decode yields surface forms; the canonicalize post-processor (+ score_no_lang --canon)
  is part of the headline metric for exactly this reason.
- Slot mix (v2 core_test, n=817): 559 non-null = 406 free (q 356, query 37) + 153 enum
  (avoid_features 114); 3024 nulls.

## Artifacts

- `src/build_spans.py` → `data/{core_train,core_val,core_test,aug_train,aug_val}_spans.jsonl`
  (+ `_spansn.jsonl` numbered variant, `span_numbering: true`), `data/spans_build_stats.json`.
  Reuses build_tags alignment (find_spans/enum_item_span); unaligned free values (~1.2%
  core_train) fall back to literal; `enum_spans` side-field records enum phrase locations
  for a possible later pure-pointer round.
- `src/render.py`: SPAN_SYSTEM_PROMPT + span_target/span_numbering branches.
- `eval/eval_spans.py`: generate → parse (fence-tolerant) → repair (clamp/swap/coerce bare
  ints, drop invalid) → decode_spans (imported from build_spans — single decode contract)
  → identical norm_val scoring → dump → span_diag index-error histogram. Compatible with
  `score_no_lang.py` (official lang-free metric, `--canon`).
- Trains (winner recipe verbatim: LoRA r=32 α=64, lr 1e-4, 3 ep, bf16, early stop):
  `train/runs/llama-3.2-1b-spans` (plain prompt) + `train/runs/llama-3.2-1b-spansn`
  (numbered prompt), both on core+aug spans v2 (14.4k rows).
- Orchestration: `scripts/run_span_round9_when_free.sh` (GPU-poll watcher, exit-120
  tolerant, yields to the round-7 v2 SFT on GPU 0).

## Decision rules (pre-registered before any run)

Reference R = winner-v2 lang-free canon F1 (round-7 retrain in flight; fallback
0.7993 = v1-winner converted).

1. Best span variant F1 > R → **replace** the JSON formulation (re-run trio/detector
   implications noted before shipping).
2. Best variant F1 ≥ R − 0.005 AND free-text hallucinated FP slots reduced ≥30% vs winner
   → **adopt** as grounding-guaranteed alternative (cf. round-4 G constrained decoding:
   parity + hard guarantee can still ship).
3. F1 ∈ [R − 0.02, R − 0.005) → document as parity-band, no adoption.
4. F1 < R − 0.02 → **reject**: with round-5 Track A this closes pointer-style outputs for
   1B decoders (both per-word and index-pair contracts).
5. Numbered vs plain gap ≥ 0.02 F1 → index arithmetic was the binding constraint;
   report span_diag histograms either way (exact/±1/±2/form-mismatch rates).

## Expectations (honest prior)

- Plain variant may Track-A-style fail on index arithmetic; numbered variant is the real
  candidate. Decode should be FASTER (≈40-70 new tokens vs ~90 for JSON strings).
- Enums (literals) unaffected by spans — avoid_features hallucinations (29 FP) persist.
- Null slots (84% of slot occurrences) unchanged in form.
- Canonical-form golds (appendage/inflection) lose vs the generative winner unless
  canonicalize recovers them — watch the before/after canon delta closely.

## Deferred (round 10 candidate, NO work this round)

Pure-pointer LLM (ALL params spans) + cross-lingual embedding matcher for enums
("płatne drogi" → nearest canonical). Gate: offline matcher eval first (e5/LaBSE via
plain-transformers mean pooling — venv has no sentence_transformers/sklearn) vs
ENUM_STEMS lexicon and vs the FT-literal incumbent (0.895 enum-slot F1, from the Jev
probe). Only if the matcher is competitive does error composition (LLM span errors ×
matcher errors) have a chance vs hybrid. `enum_spans` side-fields are already recorded.

## Execution addendum (2026-10-12 16:24)

- **R fixed: winner-v2 lang-free canon F1/EM = 0.7958 / 0.7956** (`no_lang_llama-3.2-1b-v2-ft.json`,
  canon applied; slightly below the v1-converted 0.7993/0.8005 — round-7's replacement
  bar note). Decision rules above use R = 0.7958.
- Parallel lanes: spans (plain) on GPU 1 via `scripts/run_span_round9_when_free.sh`
  (started 15:21); spansn (numbered) on GPU 0 via `scripts/run_spansn_gpu0.sh` manual lane
  (started 16:24, placeholder `llama-3.2-1b-spansn-ft.json` prevents watcher double-launch,
  real result overwrites it). Rationale: GPU compute is the constraint (93–100% util at
  micro-bs 2), not VRAM — bigger micro-bs would gain ≤ ~15%; parallelizing the two
  variants saves ~1h40m with zero recipe risk.
- spansn numbered prompts push 29 more rows past seq-len 2048 (628 vs 599 dropped;
  14,376 vs 14,405 train rows) — expected from the `1|word` prefixes, deemed harmless.
- **From round 10 on**: default `--micro-bs 4 --grad-accum 16` (effective batch 64
  unchanged → identical optimization; ~10% wall-clock; avoid micro-bs 8 — padding waste
  without group_by_length). Round-9 runs stay micro-bs 2/accum 32 for verbatim-recipe
  comparability.
