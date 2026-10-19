# Failure detection — knowing when the model is wrong

Status: done
Start: 2026-10-09
End: 2026-10-12

> Retrofitted 2026-10-19 from git history (e8d6037/a0a97fe/cdd9e79) + `reports/RESULTS.md`
> §"Failure detection" and §"Improving failure detection".

## Goal

Per-row failure detection for the slot-filler: score how likely a prediction is wrong using
signals that are free or cheap at inference (token logprobs, self-consistency, deterministic
validators), then learned detectors on naturally-occurring labeled errors — to power an
escalation cascade (auto-accept confident rows, escalate the rest).

## Expected results

- `eval/detect_uncertainty.py`: per-row signals (min logprob, k-sample self-consistency,
  validators) vs actual errors → AUC/coverage tables.
- Labeled-error source without touching core_test: h90 split (`src/split_holdout.py`,
  dialog-grouped, lang×fn stratified) + LoRA retrains (`train/runs/*-h90/`).
- `eval/train_detector.py`: logistic combiner, hidden-state probe, stack, conformal
  abstention, evaluated once on test.
- `eval/ensemble_vote.py`: majority vote of the h90 trio.

## Non-goals

- No changes to the extraction model itself or the data.
- No production cascade wiring (that stays a prototype idea).

## Acceptance criteria (as was)

- [x] Detector AUC reported for ≥3 models on core_test; best single signal identified.
- [x] Learned combiner vs min-logprob comparison on holdout-trained, test-evaluated setup.
- [x] Practical escalation rule quantified (coverage vs error-recall).

## References

- `reports/RESULTS.md` §Failure detection, §Learned detectors; AGENTS.md eval commands.
