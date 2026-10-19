# Outcome — Jev decision-model probes (plain language)

## What we did

Asked whether a cheap typed-decision API ("pick one of these options / is this value
grounded?") could either fill enum slots directly or fact-check our local model's answers.

## What worked

- Cheap ($0.04/M tokens), calibrated, comfortable with enums and nulls; zeroshot enum F1
  0.696 is respectable for no training. Client infrastructure (`jev_client.py`) is reusable.

## What didn't

- **Direct extraction: clearly behind** — 0.696 vs 0.895 F1 for the fine-tuned 1B, with
  Polish collapsing to 0.36 F1 (precision 0.28).
- **Verification gate: useless here** — AUC 0.577 vs 0.750 for free logprobs. Grounded ≠
  correct: the dominant errors (lang judgment, q boundaries) pass a grounding check by
  construction.

## Bottom line

Decision models answer the questions our model doesn't get wrong. The logprob combiner
stays the escalation gate. Untried idea parked in BACKLOG: Jev per-slot risk as one more
feature in the learned combiner.
