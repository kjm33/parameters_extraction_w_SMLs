# mmBERT backbone swap for the edge tagger

Status: done
Start: 2026-10-12
End: 2026-10-12

> Retrofitted 2026-10-19 from `2026-10-12-mmbert-tagger-round8.md` (in this dir; rules
> pre-registered) + `reports/RESULTS.md` Round 8 + flash-attn addendum.

## Goal

Answer "would a modern encoder (ModernBERT-class) be better?" for the edge tagger lane by
swapping mBERT → mmBERT (jhu-clsp, ModernBERT arch, 1,833 langs) under the identical
fixd-v2 recipe, plus a follow-up flash-attention test.

## Expected results

- Compat-only code changes (tokenizer-agnostic special tokens, attention fallback);
  mmBERT vs mBERT on benchmark v2 with pre-registered tie rules (adopt only if > +0.5).

## Non-goals

- JSON-winner lane untouched; no recipe tuning beyond backbone.

## Acceptance criteria (as was)

- [x] v2 comparison F1/EM + latency GPU/CPU; flash-attn re-check on user follow-up.

## References

- `2026-10-12-mmbert-tagger-round8.md`, `reports/RESULTS.md` Round 8,
  `train/runs/mmbert-tagger-fixd-v2/`.
