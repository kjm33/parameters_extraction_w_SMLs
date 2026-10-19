# State — where we are now (2026-10-19)

> One screen. Read this at session start. Update at session end (30-second anti-drift rule).

## Active efforts

| Effort | Since | State | Next |
|---|---|---|---|
| [efforts/2026-10-13-gold-v3/](efforts/2026-10-13-gold-v3/) | 2026-10-13 | Phase A done; adoption gate **PASSED** 2026-10-19 (spot-check 97.9% teacher-sourced) — v3 = headline gold | Phase B retrains (1B winner on v3 → 3B; optional edge-tagger lane) — awaiting free GPU, user signals |

## Live numbers

- Benchmark v2 winner (cloud lane): llama-3.2-1b LoRA, `train/runs/llama-3.2-1b/merged/` — F1 0.7958 / EM 0.7956
- Edge lane: bert-tagger-fixd-v2+canon — F1 0.7813 / EM 0.7846, 4.3 ms GPU / 107 ms CPU
- Detection: combiner AUC 0.763–0.772; cascade escalates 30% → catches ~50% of errors
- Gold v3 (frozen 2026-10-13, **adopted as headline gold 2026-10-19** — gate passed): 14,877 grounded + 675 glm-recovered + 1,064 v2_fallback (6.4% overall, 3.6% on test); ~$4.66 total cost — see `data/v3_stats.json`
- Full tables: `reports/RESULTS.md`

## Blocked

- gold-v3 Phase B retrains: both GPUs busy; user will signal when free (plan: 1B winner recipe
  on v3 first, then 3B; optional ~30 min edge-tagger lane).

## Next

1. gold-v3 Phase B: retrain winner + tagger on v3, compare against v2 baselines (0.7958/0.7956 llama-1B; 0.7813/0.7846 edge)
2. gold-v3 Phase C: cascade/committee prototype on v3-trained models
