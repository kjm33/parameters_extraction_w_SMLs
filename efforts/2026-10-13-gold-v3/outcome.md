# Outcome — Gold-v3 (plain language)

> Drafted 2026-10-19 at SDD adoption while the effort is mid-flight; finalize at close-out.

## What we did (so far — Phase A complete)

Diagnosed that our remaining "errors" are largely the answer key's fault, wrote a
pre-registered grading policy, ran a 6-teacher bake-off (claude-haiku-5.5 won: EM|grounded
0.377, $0.142/500 rows), and regenerated all 16,616 targets (~$4.66 total: haiku full run +
glm-5.3-flash on rejects + v2 fallback for 6.4%).

## What worked

- Bake-off before commit: teachers differ wildly (gpt-5.4-mini parses 100% but agrees with
  v2 gold only 18.6% — fluent ≠ right policy).
- The disagreement report caught a real builder bug class: v2 nulls spoken
  origin/destination/transportMode on ALL here_directions rows (~28% of rows).
- Grounding validators + reject recovery keep quality auditable; usage logged per request.

## What didn't (yet / open)

- **Adoption gate not passed**: the 50-row spot-check sheet awaits user verdicts (≥95%
  agreement required). v3 is frozen but NOT yet the headline metric.

## Bottom line (so far)

For ~$5 and one day, the label-noise ceiling looks liftable; everything now hinges on the
user's spot-check, then Phase B retrains answer whether cleaner gold moves the models.
