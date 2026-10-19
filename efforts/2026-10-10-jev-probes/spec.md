# Jev decision-model probes

Status: done
Start: 2026-10-10
End: 2026-10-10

> Retrofitted 2026-10-19 from git (ae3ae2d) + `reports/RESULTS.md` §"Jev decision-model
> probes" + `2026-10-10_jev_probes_explained.md` (kept in this dir).

## Goal

Probe whether the Jev decision model (typed choice/noul/score questions, OpenRouter alpha
endpoint) is useful for slot filling: (1) as a direct enum-slot extractor, (2) as a
verification gate on the local model's predictions.

## Expected results

- `eval/jev_client.py` (retry/backoff/thread-pool client, mock mode) +
  `eval/jev_enum_probe.py` + `eval/jev_verify.py` with AUC/F1 comparisons vs local FT model
  and vs min-logprob detector.

## Non-goals

- No text-span generation from Jev (it can't), no cascade wiring.

## Acceptance criteria (as was)

- [x] Probe 1: enum-slot F1 vs llama-1B FT on same 227 val rows.
- [x] Probe 2: row/slot detection AUC vs min-logprob on 817 test rows.

## References

- `2026-10-10_jev_probes_explained.md` (this dir), `reports/RESULTS.md` addendum,
  `eval/results/jev_*`.
