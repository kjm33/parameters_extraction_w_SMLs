# Outcome — Classical-NLP assistance (plain language)

## What we did

Two classic-NLP tricks, pre-registered and measured: snapping predicted values onto NER-like
utterance spans (a post-processor), and showing the model candidate place-name spans in the
prompt (plus retraining on hinted data).

## What worked

- The free ablation finding: re-scoring the SAME predictions with a lenient, inflection-
  tolerant compare gives +2.4–3.7 F1 everywhere — the "span boundary error" mass is mostly
  an artifact of the strict evaluator, not the model. (This insight later drives the
  lang-free benchmark v2 in rounds 6–7.)

## What didn't

- **Snapping: a no-op.** ≤5 values snapped per model, F1 ≤ +0.0013. Fine-tuned models already
  copy utterance spans near-verbatim.
- **NER hints: actively harmful.** baseline 0.7515 > ner-trained+hinted 0.7366; hints alone
  make the model hallucinate (FP 128→198 — it copies CANDIDATE_SPANS into unset slots);
  retraining cancels the copying but loses recall; the retrained model without hints
  collapses (EM −10 pts). Dependency on hints = fragility.

## Bottom line

Both dropped. Classic NER is not the lever for this task; the next real lever for the
hallucinations is grammar-constrained decoding (done in round 4) — and the eval itself was
stricter than the data builder, which rounds 6–7 fixed.
