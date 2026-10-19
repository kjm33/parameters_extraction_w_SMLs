# Related-work improvements — distillation, synthetic data, constrained decoding, negatives

Status: done   <!-- items 5–10 not executed → BACKLOG.md -->
Start: 2026-10-10
End: 2026-10-11

> Retrofitted 2026-10-19 from `tasks.md` (was `plans/2026-10-10_improvement-todo.md`,
> OUTCOME blocks inside) + `reports/RESULTS.md` Round 4 + `2026-10-10_related-work-slot-filling.md`.

## Goal

Execute the top research-ranked improvements to the F1 0.7515 / EM 0.667 baseline:
(1) teacher distillation for pl/es/fr, (2) synthetic-data closed loop, (3) grammar-constrained
decoding for deployment, (4) hammer-style negative training against the 128 hallucinated slots.

## Expected results

Per item, acceptance pre-registered in `tasks.md`: distill +3 F1 on pl/es/fr with en/de
≤−0.5; synthetic ≤40% capped mix with validator loop; constrained decoding with enum_invalid
→ 0 and F1 unchanged; negatives with FP −20% and null acc ≥0.96.

## Non-goals

- AgentWeave-style pre-routing; hidden-state probes round 2 (failed at 915 rows/2048 dims).

## Acceptance criteria (as was)

- [x] 1 executed — FAILED acceptance (see outcome).
- [x] 2 executed — kept as future lever, not shipped alone.
- [x] 3 executed and SHIPPED.
- [x] 4 executed — FAILED acceptance.
- [ ] 5–10 (format ablation remnant, tagger-agreement feature, Q4 quantization, cascade,
      ASR augmentation, committee) → moved to BACKLOG.md (cascade superseded by gold-v3 Phase C).

## References

- `tasks.md` (original TODO with OUTCOME blocks), `2026-10-10_related-work-slot-filling.md`
  (source research), `reports/RESULTS.md` Round 4.
