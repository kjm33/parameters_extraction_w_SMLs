# Baseline pipeline — tiny-LLM slot filling for car navigation

Status: done
Start: 2026-10-07
End: 2026-10-10

> Retrofitted 2026-10-19 from `tasks.md` (was `plans/PLAN.md`, approved 2026-10-07) + `reports/RESULTS.md` main table.

## Goal

Train and compare small (≤1B) local LLMs for slot filling: given (user utterance, selected
function schema) → compact JSON, one key per parameter, value from the utterance or `null`
(honest-null policy). 5 languages (en/de/pl/fr/es). Motivation: decompose fcft's full
function-calling task so a dedicated ≤1B extractor beats the 3–4B joint arg_acc (~0.37–0.41).

## Expected results

- Core dataset from fcft verified dialogs (`src/build_extraction.py`, honest-null targets) +
  heuristic augmentation from `instruction_extraction` (`src/build_augment.py`).
- 4 candidates (qwen2.5-0.5b, qwen3-0.6b, llama-3.2-1b, gemma-3-1b) × {zeroshot, LoRA,
  full-FT} on the same held-out test; metrics: parse rate, slot P/R/F1, null acc, row EM,
  per-lang × per-fn breakdowns.
- Winner recommendation + merged-adapter export in `reports/RESULTS.md`.

## Non-goals

- No teacher/API calls for data (heuristic augmentation only).
- No function selection (chosen upstream), no multi-turn, no packing.
- No constrained decoding at baseline (later round).

## Acceptance criteria (as was)

- [x] Datasets build reproducibly; aug ≤40% of mix, deduped, 50-row spot-check documented.
- [x] All candidates trained (LoRA; full-FT ablation) and evaluated on core_test (n=817).
- [x] Winner exported merged (`train/runs/llama-3.2-1b/merged/`).

## References

- `tasks.md` (original approved plan), `reports/RESULTS.md` §Setup/Main table, AGENTS.md.
