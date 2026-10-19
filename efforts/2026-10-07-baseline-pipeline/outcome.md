# Outcome — Baseline pipeline (plain language)

## What we did

Built the whole vertical slice: extraction dataset from fcft verified dialogs with the
honest-null policy ("if the value isn't spoken, say null"), heuristic augmentation from
instruction_extraction, chat rendering with prompt-masked loss, LoRA/full-FT training for 4
small candidates, and a full eval harness (parse rate, slot F1, null accuracy, row EM).

## What worked

- **Llama-3.2-1B + LoRA is the winner**: slot F1 0.7515 / row EM 0.667 on core_test (n=817),
  exported merged (base+LoRA). en/de ~0.785, pl/es/fr 0.70–0.74.
- Fine-tuned tiny models crush their zeroshot selves; JSON parse rate 1.0 after FT.
- Decomposition thesis holds: a 1B dedicated extractor far exceeds the 3–4B joint model's
  parameter accuracy.
- Full-FT ≈ LoRA at this scale → LoRA recipe is the default going forward.

## What didn't

- gemma-3-1b is weak at JSON (as predicted; kept as comparison).
- The `lang` slot is the single hardest behavior (foreign-place discussions confuse it) —
  later removed from the metric in round 7.
- Polish lags (inflection + canonical-form mismatch) — recurring theme of later rounds.

## Bottom line

A 1B LoRA slot-filler at F1 0.7515 with honest nulls, merged export ready; remaining error
mass (q boundaries, 128 hallucinated slots, lang) defined the agenda for rounds 2–12.
