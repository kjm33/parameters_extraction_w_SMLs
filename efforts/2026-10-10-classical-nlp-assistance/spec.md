# Classical-NLP assistance — span snapping + NER hints

Status: done
Start: 2026-10-10
End: 2026-10-11

> Retrofitted 2026-10-19 from `tasks.md` (was `plans/PLAN_NER.md`, incl. its OUTCOME block)
> + `reports/RESULTS.md` §"classical-NLP assistance".

## Goal

Test whether classic NLP helps the tiny slot-filler: (A) a span-snapping post-processor that
re-aligns predicted values to utterance spans, and (B) NER spans injected into the prompt as
candidate hints. Attack ~66 q-span-boundary errors and 128 hallucinated values.

## Expected results

- A: `eval/snap_spans.py` + before/after report per error class; free lenient (tok_match)
  re-scoring ablation quantifying eval-normalization artifact.
- B: `src/extract_ner.py` (wikineural spans), `src/render.py` CANDIDATE_SPANS injection,
  retrain `train/runs/llama-3.2-1b-ner`, 2×2 eval (train×prompt) vs 0.7515 baseline.

## Non-goals

- No constrained decoding, no gold changes, no new languages.

## Acceptance criteria (as was)

- [x] A: per-model F1/EM before/after snapping + per-class deltas reported.
- [x] B: 2×2 isolation of hint value vs NER training; keep only if ≥ +2–3 F1 (~0.78).

## References

- `tasks.md` OUTCOME block, `reports/RESULTS.md` §classical-NLP addendum, artifacts
  `data/*_ner.jsonl`, `scripts/run_ner_experiment.sh`.
