# TODO — Round 12: Gold-v3 regeneration + 3B decoder + cascade

Generated 2026-10-13. Approved plan (user decisions recorded below). Companion diagnosis:
round-11 gold-noise census (`notes/2026-10-13-round11-plain-language-summary.md` + fresh
measurements in the round-12 planning discussion).

Baseline to beat (v2 frozen): winner llama-3.2-1B LoRA v2 lang-free F1/EM 0.7958/0.7956;
edge bert-tagger-fixd-v2+canon 0.7813/0.7846.

## Diagnosis (measured, round-12 planning)

- avoid_features "blank receipts" (explicit user avoid-request, gold null): **49 train /
  20 test**; gold-set-without-cue 765 train / 47 test (cue lexicon undercounts — needs LLM
  adjudication to classify properly).
- Appendage/style inconsistency: `, XX` 2-letter codes 101 train / 23 test; word-comma
  tails 59 / 15; q ≥3-word boundary rows 5.4% train / 7.9% test.
- llama-3.2-3B never retrained on v2-family data (v1 run: 0.7899 lang-free-converted,
  over-extraction P .725 — plausibly correct-intent-penalized-by-noise).

## Binding user decisions (2026-10-13)

1. **Dual-gold**: v2 stays the continuity benchmark; v3 becomes headline after validation;
   old dumps re-scoreable on both (targets embedded in dumps).
2. **Teacher**: bake-off first (500-row stratified sample: gpt-5.4-mini reference vs
   claude-haiku-5.5 vs gemini-2.5-flash-lite, ~$0.10, judged by 50-row manual spot-check
   with pl rows scored separately — Jev pl-collapse precedent) → winner runs full corpus.
3. **Scope**: ALL core+aug rows = **16,616** (verified: core 9,186+489+817, aug 5,818+306).
4. **Mode**: Paid Batch API for an OpenAI winner (−50%, 24h turnaround, capability-check
   at runtime — needs payment-enabled org); OpenRouter models sync (already cheap).
5. **Decoder**: 3B retrain on v3, then decide (4B only if 3B clears its bar).

Full-run cost envelope (5.9M in + 0.75M out, llm-prices.com 2026-10-07): gpt-5.4-mini
$7.81 sync/$3.90 batch; claude-haiku-5.5 ~$0.97; gemini-2.5-flash-lite ~$0.89.

## Step 0 — housekeeping

- [x] Commit round-11 artifacts (done as `037049a`; working tree clean).
- [x] Pre-flight: row counts + key inventory (OPENAI_API_KEY_{1,2,3}, OPENROUTER_API_KEY,
      MISTRAL_API_KEY_{1,2}, GROQ_API_KEY_{1,2} in `../.env`; KEY_2 historically dead).

## Phase A — Gold-v3

- [x] **A0. Pre-registered policy doc** `notes/2026-10-13-gold-v3-policy.md`: verbatim-first;
      appendage rule = only-if-spoken DEFAULT (data-driven flip to always-for-US allowed
      from A2 evidence BEFORE freezing v3; decision documented in the note); avoid_features
      from utterance INTENT not reference call (mention ≠ extraction; polarity
      "po autostradzie" = via not avoid; adjacent-feature precision "płatnych autostrad"
      → tollRoad not motorway); pl canonical = v1 alias-map style; enums canonical literals;
      lang stays null (v2 rule); honest-null preserved; adversarial-cases appendix from the
      round-11 census. §7 flip #2 pre-registered: here_directions origin/destination/
      transportMode (v2 nulls by builder rule even when spoken).
- [x] **A1. Teacher pipeline** `src/build_gold_v3.py`: exact training prompt + schema +
      utterance; grounding validators (keys ⊆ schema, verbatim/enum/number — reuse
      build_extraction helpers); `--bakeoff N` stratified sampling
      (lang × fn stratified, seed fixed); OpenRouter client (jev_client.py
      pattern: retry/backoff, workers, 400-extras-drop); rejects + usage logged to
      `logs/goldv3_usage.jsonl`; per-model max_tokens (reasoning models need 4096).
- [x] **A1b. Bake-off** (500 rows, EXTENDED to 6 teachers incl. non-Big-3 challengers
      glm-5.3-flash / deepseek-v4.1-flash / ling-3.1-flash, ~$1.6 total): per-teacher
      grounded-parse rate + teacher-vs-v2-gold disagreement profile; 50-row stratified
      manual spot-check sheet written → **awaiting user verdicts**.
- [x] **A2. Disagreement report** `eval/gold_v3_report.py`: teacher-vs-v2-gold per noise
      class (blank-receipts, appendages, word-commas, q-boundary, pl canonical); pairwise
      inter-teacher EM; KEY finding: v2 nulls spoken origin/destination/transportMode on
      ALL here_directions rows (~28%) — flip #2.
- [x] **A3. Freeze v3**: `data/{core,aug}_{train,val,test}_v3.jsonl` (targets only; prompts
      byte-identical); v2 frozen alongside. DONE: t1 haiku full run ($4.44, grounded 14,877)
      + t4 glm on 1,739 rejects ($0.22, recovered 675) + v2_fallback 1,064 (6.4%; test only
      29/817=3.6%); `data/v3_stats.json`; positional alignment (297 dup ids!). **Adoption
      gate**: spot-check agreement ≥95%, measured avoid blank-receipts → ~0 — PENDING user
      50-row spot-check. Only then is v3 usable as headline metric.

## Phase B — Retrains on v3 (bars pre-registered BEFORE any run)

- [ ] 1B winner recipe on v3 → new reference R_v3 (llama-3.2-1b LoRA, core+aug v3).
- [ ] llama-3.2-3B on v3 (first 3B v2-family run): **replace winner if > 1B-v3 +0.5**;
      within ±0.5 keep 1B (edge cost); gemma-3-4b only if 3B > 1B by ≥+1.5.
- [ ] Optional cheap lane: rebuild `*_tags_v3` + retrain edge tagger (~30 min GPU).
- [ ] Detector/h90 infra only if the winner model changes.

## Phase C — Cascade prototype (independent of B)

- [ ] Wire existing detector combiner (AUC 0.77) at ~15% escalation + h90 trio vote →
      escalate judgment-call rows to the biggest available model; report end-system
      F1/EM vs latency on core_test. No new research.

## Phase D — conditional

- [ ] Contrastive intent data (same place phrase, different intent → different q) only if
      v3 + 3B leaves a measured avoid/q gap.

## Explicitly not doing

- Touching v2 files (frozen; dual-gold scoring instead).
- Teacher calls beyond the approved scope (bake-off + one full pass; no iterative
  re-annotation loops without a new decision).
- New NER/ontology/gazetteer work (round-11 lane closed).
- 4B decoder unless 3B clears ≥+1.5.

Effort: A ≈1-2 d hands-on + API wall-clock; B ≈3-5 h GPU; C ≈0.5 d.
