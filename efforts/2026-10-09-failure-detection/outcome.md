# Outcome — Failure detection (plain language)

## What we did

Taught the system to smell its own mistakes. For every test row we computed cheap suspicion
signals (lowest token probability in the answer, disagreement across 5 sampled retries,
rule-based validators), checked them against real errors, then trained a small logistic
"combiner" on 915 naturally-occurring errors from a held-out slice of training data.

## What worked

- **Min token logprob is the best single lie-detector** (AUC 0.75–0.77) and it's free.
- **Learned combiner (15 features) wins on 3/4 models** (AUC 0.763–0.772) and transfers to
  the full-data winner.
- Practical cascade rule: escalate the ~30% lowest-confidence rows → catch ~50% of all
  errors; escalate 50% → catch ~73%.
- Conformal abstention gives tunable guarantees: ~63% of rows auto-accepted at ~79% accuracy,
  or ~27% at ~93% (targets run ~1.4× optimistic — recalibrate on more errors for production).
- Trio majority vote is an accuracy win (EM 0.6634 > all members) but not a detection win.

## What didn't

- Hidden-state probes flopped (AUC 0.60–0.68): 915 labels is too few for 2048-dim states on
  a fine-tuned small model.
- Validators only catch hallucinations; confident-wrong-nulls are invisible to them.
- Stacking never beat the plain combiner.

## Bottom line

A free logprob threshold already halves the error rate of auto-accepted answers; the learned
combiner adds a little on top. This is the gate the future cascade (gold-v3 Phase C) plugs into.
