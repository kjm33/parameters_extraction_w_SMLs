# 2026-10-11 — BERT tagger improvements (round 6): research note + plan

Context: round-5 BERT tagger landed at F1 0.7152 vs JSON-winner 0.7515 (core_test, n=817)
— 3.6 pts off the ±2pt edge-candidate bar, but 8.6× faster on GPU (5.0 vs 43.1 ms/row)
and CPU-viable (96.6 ms/row). User ask: research what could improve this model; inflected
predictions ("Krakowa" instead of gold "Kraków") are acceptable if normalizable via NLP
post-processing. Baseline for all simulations below: exact repo scorer reproduced on the
existing `bert-tagger-ft_preds.jsonl` dump (base reproduces 0.7152/0.6279 exactly).

## 1. Error census (where the 3.6 pts actually are)

Non-lang errors total just **128 slots**; lang is **216** (150 FN + 66 FP):

| class | slots | examples | fixable by |
|---|---|---|---|
| lang wrong/missed | 216 | context-dependent gold (dialog lang vs null) | nothing single-turn (see §2) |
| span boundary `, XX` | 52 | pred "Los Angeles" vs gold "Los Angeles, CA" | B1 data rule fix |
| pred_null (recall) | 44 | coords "45.764,4.8357", street "Schottenring 1" | B3 capacity/data |
| other-wrong | 24 | wrong city, fast vs short | B3 capacity |
| array diff | 8 | avoid_features multi-value selection | B3 data |
| inflection (non-`,`) | ~29 | "Krakowa" vs "Kraków" | B2 canonicalizer |

## 2. Key findings (simulations, exact scorer)

1. **Appendage-rule bug is the single biggest recoverable block.** `build_tags.py`'s
   appendage rule rejoins leftover parts only ≤2 chars; `", WA"` is 4 chars → the tags
   data taught the tagger to DROP state codes. Deterministic fix (join trailing
   comma+2-letter-uppercase) simulates to **F1 0.7451 (+3.0, 23 fixes, all gold-longer
   direction)** — closes 3 of the 3.6 pts alone. Same rule applied to the JSON winner's
   own predictions → 0.7752 (winner also benefits from state suffixes; lang unchanged).
2. **lang is NOT the BERT gap.** Swapping the JSON winner's lang predictions into BERT
   gives 0.7102 (slightly WORSE) — the CLS lang head is already llama-par. Oracle-lang
   jumps BOTH models ~+10 (BERT 0.8149, winner 0.8544) — structural, context-dependent
   gold, single-turn input cannot resolve → **out of scope for round 6**.
3. **Train-target lookup canonicalizer fixes ZERO slots** — inflected surfaces never
   appear as gold targets, so a `surface→canonical` map keyed on targets is empty for
   exactly the cases we need. The gazetteer must be built from *utterance→target pairs*
   (train rows where gold q="Kraków" while the utterance says "Krakowa"), plus pl
   suffix-stripping fallback.

## 3. Literature check (web research, 2026-10-11)

- **XLM-RoBERTa vs mBERT** (Conneau et al. 2019; XNLI +14.6 avg over mBERT, 100 langs):
  the standard capacity swap for multilingual tagging; `xlm-roberta-base` already in
  HF cache (`models--xlm-roberta-base`, plus `Davlan/xlm-roberta-base-ner-hrl`).
  Targets pl collapse (−9.9 by-lang vs winner) + the 68 non-lang recall/confusion slots.
- **CRF layer on BERT** (hf threads; ceur-ws 2025 comparison; SciBERT linear+CRF 2026):
  mixed evidence, typically +0.1–0.5 F1; enforces legal BIO transitions (no `O→I-x`,
  no holey spans) — matches our 52-slot boundary class. Cheap decode (Viterbi over
  ~70 labels). Gated: only if B3 lands within 1 pt of the winner.
- **Semi-supervised data augmentation / pseudo-labeling for NER** (experimental studies
  2024–2026, LLM-generated augmentation): more grounded rows help recall-heavy errors.
  We already own 6,933 teacher-grounded rows (`data/distill_train.jsonl`, round-4) the
  tagger has never seen — no new API calls needed (repo is teacher-call-free).
- **JointBERT lineage** (arXiv 1902.10909 + follow-ups: co-interactive transformer,
  syntax-encoding, pre-trained joint models): schema-string prefix conditioning we
  already have; deeper slot-attention variants are research-y, low expected ROI here.
- **Span-based (start/end pointer) heads**: alternative to BIO+CRF; more invasive
  rewrite, skipped in favor of CRF.

## 4. Plan (approved scope: B1+B2+B3 separate ablations; B4 conditional)

- **Phase 0 (CPU only, ~30 min)** — B2 canonicalizer prototype: alias gazetteer from
  train utterance→target pairs + pl suffix rules; measure on existing BERT dump
  (ship if ≥+0.5 F1). Bonus: same post-proc + `, XX` rule on the JSON winner dump
  (measures JSON-track gain; simulated 0.7752 for the boundary part alone).
- **Phase 1 (CPU ~10 min)** — B1: relax `build_tags.py` appendage rule to join trailing
  `,<space><2-letter uppercase>` into the span; first audit gold keep-vs-drop consistency
  (how often gold itself drops the state — rule must match gold majority behavior);
  rebuild `data/*_tags.jsonl`; verify roundtrip stats + new ceiling.
- **Phase 2 (GPU, 4 separate ablations ~30–45 min each; watcher scripts — parallel
  session occupies both GPUs; mBERT bs16 ≈5.7 GB)**:
  1. mBERT+fix (isolates B1 gain alone)
  2. mBERT+fix+distill (+6,933 rows)
  3. XLM-R-base+fix (capacity swap)
  4. XLM-R+fix+distill (combined)
  Eval each with existing `eval/eval_tagger.py` (same scorer, latency incl. --cpu).
- **Phase 3 (conditional)** — B4 CRF layer on the best B3 config, only if it lands
  within 1 pt of 0.7515.
- **Phase 4** — RESULTS.md round-6 section + this note updated with outcomes.

Decision rules (from round 5, user-agreed):
- adopt BERT as edge candidate if F1 ≥ 0.735 with latency intact;
- BERT **replaces** the JSON winner if F1 ≥ 0.7515;
- B2 canonicalizer ships if ≥+0.5 F1;
- expected landing: 0.75–0.77 (B1 +3.0, B2 +0.5–1, B3 +0.5–1.5).

## 5. Artifacts touched (planned)

- `src/build_tags.py` (appendage rule), data rebuild `data/*_tags.jsonl`
- `eval/canonicalize.py` (NEW: gazetteer builder + post-proc, CPU, dump-based eval)
- `train/train_tagger.py` (`--base`, `--train-files` flags already default core+aug;
  add distill list), `eval/eval_tagger.py` (unchanged)
- runs: `train/runs/bert-tagger-{fix,fixd,xlmr-fix,xlmr-fixd}/`, optional CRF variant
- results: `eval/results/bert-tagger-*.json` + RESULTS.md round-6
