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

### TODO list

Phase 0 — simulations (CPU only):

- [ ] B2a: canonicalizer prototype (`eval/canonicalize.py`) — alias gazetteer from train
      utterance→target pairs + pl suffix rules; measure on existing BERT dump
      (ship criterion ≥+0.5 F1)
- [ ] B2b: same post-proc + `, XX` append-state rule on the JSON winner dump
      (expect ~0.7752)

Phase 1 — data fix (CPU):

- [ ] B1a: audit gold keep-vs-drop of `, XX` state suffixes (rule must match gold
      majority behavior)
- [ ] B1b: relax `build_tags.py` appendage rule (join trailing comma + 2-letter
      uppercase into span), rebuild `data/*_tags.jsonl`, verify roundtrip + ceiling

Phase 2 — ablations (GPU, separate runs; watcher scripts if GPUs busy):

- [ ] B3.1: mBERT+fix retrain + eval (isolates data-fix gain alone)
- [ ] B3.2: mBERT+fix+distill (+6,933 rows) retrain + eval
- [ ] B3.3: XLM-R-base+fix retrain + eval (capacity swap)
- [ ] B3.4: XLM-R+fix+distill combined retrain + eval
      (eval each with existing `eval/eval_tagger.py`, same scorer, latency incl. --cpu)

Phase 3 — conditional:

- [ ] B4: CRF layer on best B3 config, only if it lands within 1 pt of 0.7515

Phase 4 — write-up:

- [ ] RESULTS.md round-6 section + update this note with outcomes

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

## 6. Outcomes (2026-10-12)

- **B1a audit falsified §2.1**: the `, XX` codes are gold-side (23/23 test, 99/101 train
  absent from utterances) — the +3.0 simulation was oracle-tinted (it appended gold's own
  suffix). No span rule possible; mechanism moved to a city→code gazetteer in post-proc.
  B1b: tags data unchanged; `data/distill_train_tags.jsonl` added (6,933 rows) so B3
  could train on the full corpus; `tags_build_stats.json` rebuilt over all 6 files.
- **B2 ships**: `eval/canonicalize.py` v3 — BERT dump 0.7152→**0.7321** (+1.69 ≥ +0.5);
  winner dump 0.7515→0.7502 (−0.13 ≈ neutral, winner pipeline untouched; the expected
  0.7752 was the oracle artifact); ceiling 0.9180→0.9364. Keeps 24 aliases / 44
  city2code (evidence gate kills London-style bare-gold entries; Canterbury→UK only
  survivor at apply time).
- **B3** (same post-proc on all): mBERT+distill **0.7366** (best F1) > XLM-R+distill
  0.7340 (best EM 0.6475, fastest: 3.0 ms/row gpu / 91.7 cpu) > mBERT 0.7321 >
  XLM-R 0.7186. Distill: +0.5 (mBERT) / +1.6 (XLM-R) raw. XLM-R does not beat mBERT
  on F1 (en drops, pl recovers slightly). pl stays the tagger weak spot (.61–.62).
- **Decisions**: adopt bert-tagger-fixd+canonicalize as edge candidate (0.7366 ≥ 0.735;
  4.5 ms/row gpu, 107.3 cpu); B4 CRF skipped (0.7366 < 0.7415); JSON winner stands
  (0.7366 < 0.7515). Expected landing 0.75–0.77 was wrong because +3.0 of it was the
  oracle B1. Full section: reports/RESULTS.md round 6.

Addendum (round 7, 2026-10-12): the round-6 error census showed `lang` = 46% of edge-model
error events and unextractable by construction (216 dual-labeled test utterances) → round 7
dropped `lang` from targets (benchmark v2) and switched the headline metric to lang-free
F1/EM (winner 0.7515 → 0.7993; edge 0.7366 → 0.7792). See RESULTS.md round 7.
