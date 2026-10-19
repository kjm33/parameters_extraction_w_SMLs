# Tasks — Failure detection

> Retrofitted 2026-10-19 (this effort had no separate plan file; work driven from round-2
> discussions, reconstructed from RESULTS.md and git history).

## Plan

Signals first (free), then labeled-error infra (h90), then learned detectors, then ensemble
and conformal layers.

## Tasks

- [x] 2026-10-09: `eval/detect_uncertainty.py` — min-logprob / self-consistency(3) /
      validators vs actual errors; AUC tables for llama-1b, qwen3, qwen2.5
- [x] 2026-10-09: h90 split (`src/split_holdout.py`) + trio retrains
      (`train/runs/{llama-3.2-1b,qwen2.5-0.5b,qwen3-0.6b}-h90/`)
- [x] 2026-10-09: extended features — k=5 samples, per-slot logprobs (teacher-forced),
      per-slot semantic entropy, hidden states every 4th layer
- [x] 2026-10-10: `eval/train_detector.py` — logistic combiner (15 feats), hidden probe
      (layer+λ by CV), stack, conformal abstention; holdout→test protocol
- [x] 2026-10-10: `eval/ensemble_vote.py` — majority vote of h90 trio + disagreement AUC

## Blockers / open questions

- none (effort closed; follow-ups live in BACKLOG.md — e.g. BERT-tagger agreement feature,
  Jev per-slot risk feature)
