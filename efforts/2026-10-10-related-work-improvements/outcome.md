# Outcome — Related-work improvements (plain language)

## What we did

Ran the four highest-ranked ideas from related-work research, each with a pre-registered
acceptance bar: teacher distillation (gpt-5.4-mini annotations for pl/es/fr), a synthetic
data generator with validator loop, grammar-constrained decoding (GBNF/xgrammar), and
adversarial negative training.

## What worked

- **Grammar-constrained decoding SHIPS**: F1 0.7515→0.7510, EM 0.6671→0.6659, parse 1.0 —
  statistically neutral, but every output is now guaranteed schema-valid. Free hard
  guarantee for edge deployment (`grammars/*.gbnf` for 25 fns, `eval/eval_constrained.py`).
- Directions-only distillation is genuinely useful (+2–3 F1 on here_directions).
- Synthetic data: en +3.3 F1 — a real lever for coverage gaps, kept in the toolbox.
- Diagnostics: the pl gap root cause is `lang`-label inconsistency in core (structural
  ceiling ~2–3 pts in the single-turn formulation) — not fixable by data scaling.

## What didn't

- **Distillation FAILED its bar**: best mix 0.7548 overall but pl regressed −3.1 (teacher's
  geocode-q policy conflicts with core gold style); LoRA run variance (±1.5–2) exceeds
  effect sizes at this data scale.
- **Negatives FAILED**: F1 −1.5, hallucinated slots 128→126 (noise). Trap templates at ~7%
  of mix don't generalize to real-dialog hallucinations.

## Bottom line

One clean ship (constrained decoding), two honest fails that saved us from scaling dead
ends, one lever parked (synthetic data). The recurring lesson: label-noise and eval
strictness — not model capacity — are the binding constraints, which motivates gold-v3.
