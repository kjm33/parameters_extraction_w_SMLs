# Span-list formulation — pointers instead of strings

Status: done
Start: 2026-10-12
End: 2026-10-12

> Retrofitted 2026-10-19 from `2026-10-12-spanlist-round9.md` (in this dir; pre-registered
> rules) + `reports/RESULTS.md` Round 9.

## Goal

Test the user's proposal: the winner model emits `[first,last]` word-index pairs instead of
free-text values — structurally grounded (can't fabricate), shorter decode. Plain and
numbered-utterance variants (numbering = mitigation for round-5's word-counting failure).

## Expected results

- `src/build_spans.py` hybrid targets (roundtrip 0.9388; 97.9% of non-null values pointable);
  two trained variants vs reference winner-v2 0.7958; pre-registered adopt/reject rules
  incl. anti-hallucination bar (free-text FP −30%).

## Non-goals

- No changes to the JSON lane or gold.

## Acceptance criteria (as was)

- [x] Both variants trained + scored lang-free canon; index-exactness diagnostics; verdict
      by pre-registered rule.

## References

- `2026-10-12-spanlist-round9.md`, `reports/RESULTS.md` Round 9,
  `train/runs/llama-3.2-1b-spans{,n}/`.
