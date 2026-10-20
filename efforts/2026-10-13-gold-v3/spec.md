# Gold-v3 regeneration + 3B decoder + cascade (round 12)

Status: active
Start: 2026-10-13
End: —

> Retrofitted 2026-10-19 from `tasks.md` (was `plans/2026-10-13_gold-v3-round12-todo.md`),
> the pre-registered `2026-10-13-gold-v3-policy.md` (in this dir), bake-off results and
> spot-check sheet (also in this dir). This is the ACTIVE effort — read `tasks.md` for
> live state.

## Goal

Fix the measured gold-label noise (blank-receipts avoid_features, appendage style
inconsistency, spoken-but-nulled here_directions slots) by regenerating ALL 16,616 targets
with a teacher LLM under a pre-registered policy → gold-v3 as new headline metric
(dual-gold: v2 stays continuity benchmark), then retrain (incl. first 3B v2-family run)
and prototype the detection cascade.

## Expected results

- Phase A: policy doc + teacher pipeline + 6-teacher bake-off + disagreement report +
  frozen v3 splits — `data/*_v3.jsonl`, 5 files (core train/val/test + aug train/val;
  no aug_test) — **done, adoption gate pending**.
- Phase B: retrains on v3 with pre-registered replacement bars (3B replaces 1B winner only
  if > +0.5; 4B only if 3B ≥ +1.5).
- Phase C: cascade prototype (existing combiner at ~15% escalation + trio vote).
- Phase D (conditional): contrastive intent data iff a measured avoid/q gap remains.

## Non-goals

- Touching v2 files (frozen; dual-gold scoring only).
- Teacher calls beyond approved scope (bake-off + one full pass).
- New NER/ontology/gazetteer work (round-11 lane closed); 4B unless 3B clears ≥ +1.5.

## Acceptance criteria

- [x] A0–A3 executed (v3 frozen: 14,877 teacher-grounded + 675 glm-recovered + 1,064
      v2_fallback = 6.4% overall / 3.6% on test; 297 dup ids positionally aligned).
- [ ] **Adoption gate**: user 50-row spot-check agreement ≥95% + measured avoid
      blank-receipts → ~0. Only then is v3 usable as headline metric.
- [ ] Phase B bars met or documented misses; Phase C end-system F1/EM vs latency reported.

## References

- `tasks.md` (live checkboxes + binding user decisions), `2026-10-13-gold-v3-policy.md`,
  `2026-10-13-gold-v3-bakeoff-results.md`, `2026-10-13-gold-v3-spotcheck.md` (the pending
  spot-check), `data/v3_stats.json`, `logs/goldv3_usage.jsonl`.
- Raw teacher dumps feeding the freeze (naming pattern
  `eval/results/goldv3/<teacher>_<tag>_preds.jsonl`, see `src/build_gold_v3.py` docstring):
  `eval/results/goldv3/t1_haiku55_v3full_preds.jsonl` (primary teacher, 16,616 rows) and
  `eval/results/goldv3/t4_glm53flash_v3full_preds.jsonl` (rejects pass). Bake-off dumps use
  tag `r12a` (one per teacher). Paths also hardcoded in `src/freeze_gold_v3.py`
  (`T1_PREDS`/`T4_PREDS_DEFAULT`).
