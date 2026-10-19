# BERT tagger improvements + benchmark v2 (rounds 6–7)

Status: done
Start: 2026-10-11
End: 2026-10-12

> Retrofitted 2026-10-19 from `2026-10-11-bert-tagger-improvements-round6.md` (in this dir)
> + `reports/RESULTS.md` Rounds 6–7.

## Goal

Lift the round-5 BERT tagger (0.7152) toward the JSON winner via post-processing and data
ablations (round 6), and fix the metric itself — `lang` removed from targets, lenient
normalization — then re-baseline everything on benchmark v2 (round 7).

## Expected results

- B1 appendage audit; B2 `eval/canonicalize.py` (train-derived alias/city2code/pl-fallback
  post-processor); B3 data/capacity ablations (+distill, XLM-R); B4 conditional CRF (only
  if B3 ≥ 0.7415). Pre-registered: edge adopt ≥ 0.735, replace winner ≥ 0.7515.
- Round 7: lang-free official scorer (`eval/score_no_lang.py`), v2 retrains of winner and
  tagger, new decision bars on the v2 metric.

## Non-goals

- No changes to task definition beyond the `lang` metric removal; no new languages.

## Acceptance criteria (as was)

- [x] B2 ships if ≥ +0.5 F1 on BERT (actual: +1.69).
- [x] Edge candidate adopted at ≥ 0.735 (bert-tagger-fixd+canon 0.7366).
- [x] v2 champion baselines frozen: winner-v2 0.7958/0.7956, edge-v2 0.7813/0.7846.

## References

- `2026-10-11-bert-tagger-improvements-round6.md`, `reports/RESULTS.md` Rounds 6–7,
  `eval/canonicalize.py`, `eval/score_no_lang.py`.
