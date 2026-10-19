# Outcome — BIO-tagging formulation (plain language)

## What we did

Taught the task as "label each word" (like classic NER) instead of "write a JSON answer",
two ways: the 1B decoder writing tag sequences, and a BERT encoder with a tag head.

## What worked

- The BERT tagger (Track B) is real: F1 0.7152, 8.6× faster on GPU (5.0 vs 43.1 ms/row),
  CPU-viable at 96.6 ms/row — it spawned the whole edge-tagger lane (rounds 6–8).
- Gold roundtrip analysis quantified the formulation's structural ceiling: only 91.1% of
  non-null slots are recoverable from tags (Polish inflection, US-state appendages).

## What didn't

- **Track A (decoder emitting tags) collapsed**: F1 0.4745. Small LMs can't count words —
  the "one tag per word" contract fails at ±1 tag, and the regen pass makes it slower, not
  faster (126 vs 43 ms/row).
- **Track B rejected as winner substitute** (0.7152 is 3.6 pts below the bar), with pl
  collapsing −9.9 (morphology).

## Bottom line

Both tracks rejected; the JSON-generative winner stands. Tagging is capped by what words
alone can express (91.1%), and decoders hate length contracts. Round 9 later closed the
pointer variant too — plain JSON "copy + canonicalize" is what gold style rewards.
