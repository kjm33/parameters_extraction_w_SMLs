# Outcome — Span-list formulation (plain language)

## What we did

Made the model point at answers ("words 3–4") instead of typing them, in two flavors —
plain, and with the utterance numbered so the indices are copyable.

## What worked

- **Numbering fixes pointing itself**: index diagnostics 95% exact (vs 27% plain), parse
  1.0, decode 3.4× shorter, null acc 0.9791. The round-5 word-counting diagnosis was
  correct and addressable.

## What didn't

- **Still loses 2.1 F1 (0.7748 < reject bar 0.7758)**: pointers structurally emit surface
  forms, but gold rewards canonical style — pl −6.9, en −6.4 where gold keeps forms absent
  from the utterance ("Toruń" vs "Torunia", ", CA").
- **Grounding didn't buy anti-hallucination**: free-text FP only −10% (bar was −30%);
  avoid_features unchanged — the hallucination mass is enum/null judgment, not fabrication.
  q FN rose 68→90.

## Bottom line

REJECT (pre-registered, honestly inside noise). With round-5 this closes pointer-style
outputs at 1B scale: plain JSON "copy + canonicalize" is what the gold rewards. Also seeded
the round-10/12 insight that canonical fidelity and intent judgment are gold-side problems.
