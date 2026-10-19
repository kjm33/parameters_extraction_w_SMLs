# Outcome — mmBERT backbone swap (plain language)

## What we did

Swapped the encoder under the edge tagger to mmBERT (ModernBERT's multilingual successor)
with everything else identical, then also built flash-attention from source to chase the
hypothesized speed win.

## What worked

- Clean pre-registered protocol; nominal F1 win 0.7838 vs 0.7813 (+0.25) with best-of-any-
  tagger pl (.6292); artifacts and regression checks for mBERT/XLM-R intact.

## What didn't

- **+0.25 is inside the pre-registered ±0.5 noise band → statistical tie**, and the
  hypothesized latency win inverted: CPU regresses ~16% (no flash-attn/unpadding in venv).
- **Flash-attn is ~16% SLOWER than sdpa here** (5.0 vs 4.3 ms/row): sequences are short
  (~50–80 tokens) — flash's packing overhead dominates; its regime is long context.

## Bottom line

Backbone-swap lane closed: mBERT ≈ XLM-R ≈ mmBERT. Architecture is not decisive in the
tagger lane — the 91.1% tag ceiling and Polish morphology are. Edge pick stays mBERT where
CPU latency matters; mmBERT documented as tie-alternative.
