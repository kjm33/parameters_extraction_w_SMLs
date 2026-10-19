# BIO-tagging formulation — param names as output vocabulary

Status: done
Start: 2026-10-11
End: 2026-10-11

> Retrofitted 2026-10-19 from `2026-10-11-tagging-formulation-round4.md` (in this dir) +
> `reports/RESULTS.md` Round 5.

## Goal

Recast slot filling as per-word tagging instead of JSON synthesis: (Track A) llama-3.2-1b
LoRA emitting BIO tags, (Track B) JointBERT-style multilingual encoder tagger — testing two
user ideas: schema-relevant attention via param-name tags, and cheaper/edgier inference.

## Expected results

- `src/build_tags.py` BIO-tags dataset (enum values embedded in tags, alignment cascade).
- Track A adopts if F1 ≥ ~0.735 AND shorter decode; Track B adopts as edge candidate if
  within ~2 pts of the 0.7515 JSON winner (pre-registered rules).

## Non-goals

- No changes to the JSON lane's data or training recipe.

## Acceptance criteria (as was)

- [x] Gold roundtrip measured (structural ceiling quantified).
- [x] Both tracks trained + evaluated on core_test vs pre-registered bars.

## References

- `2026-10-11-tagging-formulation-round4.md`, `reports/RESULTS.md` Round 5,
  `notes/2026-10-11-bio-tagging-bert-tagger-explained.md` (educational).
