# State — where we are now (2026-10-19)

> One screen. Read this at session start. Update at session end (30-second anti-drift rule).

## Active efforts

| Effort | Since | State | Next |
|---|---|---|---|
| [efforts/2026-10-13-gold-v3/](efforts/2026-10-13-gold-v3/) | 2026-10-13 | Phase A done (v3 frozen); **adoption gate pending** | User 50-row spot-check ≥95% → then Phase B retrains |

## Live numbers

- Benchmark v2 winner (cloud lane): llama-3.2-1b LoRA, `train/runs/llama-3.2-1b/merged/` — F1 0.7958 / EM 0.7956
- Edge lane: bert-tagger-fixd-v2+canon — F1 0.7813 / EM 0.7846, 4.3 ms GPU / 107 ms CPU
- Detection: combiner AUC 0.763–0.772; cascade escalates 30% → catches ~50% of errors
- Gold v3 (frozen 2026-10-13): 14,877 grounded + 675 glm-recovered + 1,064 v2_fallback (6.4% overall, 3.6% on test); ~$4.66 total cost — see `data/v3_stats.json`
- Full tables: `reports/RESULTS.md`

## Blocked

- gold-v3 adoption gate: waiting on user 50-row spot-check (`efforts/2026-10-13-gold-v3/2026-10-13-gold-v3-spotcheck.md`), threshold ≥95% agreement

## Next

1. Spot-check v3 (50 rows, ≥95%) → flip pre-registered nulls (spoken here_directions o/d/tM ~28% of rows) if gate passes
2. gold-v3 Phase B: retrain winner + tagger on v3, compare against v2 baselines
3. gold-v3 Phase C: cascade/committee prototype on v3-trained models
