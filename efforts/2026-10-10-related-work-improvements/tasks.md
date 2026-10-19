# TODO — Improvements from Related-Work Research

> Status: done (2026-10-10 → 2026-10-11) for items 1–4; items 5–10 not executed → BACKLOG.md. Retrofitted into SDD 2026-10-19 (original `efforts/2026-10-10-related-work-improvements/tasks.md`).

Generated 2026-10-10 from `efforts/2026-10-10-related-work-improvements/2026-10-10_related-work-slot-filling.md`.
Updated 2026-10-10: teacher = OpenAI free tier (keys verified working, see below).
Updated 2026-10-11: items 1–3 executed (see `reports/RESULTS.md` Round 4); item 4 in flight.
Baseline to beat: Llama-3.2-1B+LoRA, slot F1 0.7515, row EM 0.667 (en/de ~0.785,
pl/es/fr 0.70–0.74); detector combiner AUC 0.77.

## Teacher resources (verified 2026-10-10)

OpenAI keys in parent `../.env` — free daily usage per account: 250k tokens across gpt-5.4/
5.2/5.1/5/4.1/4o/o1/o3; 2.5M tokens across gpt-5.4-mini/-nano, gpt-5-mini/-nano, gpt-4.1-mini/
-nano, gpt-4o-mini, o3-mini, o4-mini.

| Key | Status | Notes |
|---|---|---|
| `OPENAI_API_KEY_1` | **works** | account A (org user-wsaqb…); separate daily pool |
| `OPENAI_API_KEY_2` | DEAD | `credit_balance_exhausted` — do not use |
| `OPENAI_API_KEY_3` | **works** | account B (org user-b13ox…); separate daily pool |

- Default teacher: **gpt-5.4-mini** via KEY_1, overflow/parallel to KEY_3 (≈5M free tokens/day
  combined; ~300–500 tokens/row → ~5–8k annotations/day/key conservatively).
- Exact daily usage not readable via API (keys lack `api.usage.read` scope) — budget
  conservatively (~60% of cap) and log per-request token usage ourselves.
- Keys are secrets: load from `../.env` at runtime; never hardcode, never print in full.

## High priority

- [x] **1. Teacher distillation for pl/es/fr** *(research rank 1; attacks the language gap —
      biggest measured weakness, literature effect size up to +26 F1)*
      - [x] Teacher = gpt-5.4-mini via OpenAI free tier (KEY_1/KEY_3, see resources above);
            local fcft 3B/4B only as offline fallback. Cost: free, ~5–8k rows/day.
      - [x] Distillation prompt: reuse our SYSTEM_PROMPT + function schema + candidate
            utterance; ask for the same compact-JSON target format; temperature 0.
      - [x] Generate slot annotations for unannotated instruction_extraction texts
            (dialogues.jsonl driver turns, v3 leftovers) in pl/es/fr; validate with existing
            verbatim/enum/number grounding checks (reuse `src/build_extraction.py` helpers)
            — drop rows where teacher output fails grounding (Data-Turnstile-style loop).
      - [ ] Optionally add code-switched variants (foreign place names in native utterances).
      - [x] Retrain llama-3.2-1b LoRA on core + new data; acceptance: pl/es/fr F1 each +3 pts,
            en/de not worse than −0.5 pt.
      **OUTCOME (2026-10-11): FAIL on acceptance.** Runs: `ds` (+distill 6933 +syn),
      `d2` (+distill directions-only +syn), `d5`, `d4`. Best overall F1 0.7548 (ds) vs base
      0.7515, but pl regressed −3.1 under ds (distill geocode-local q policy conflicts with
      core gold style: inflected surface forms vs canonical, q-hallucination on route-intent
      rows); no mix beats base in every lang (±1.5–2pt LoRA run variance > effect size).
      directions distill is genuinely useful (+2–3 F1 on here_directions). Root cause of the
      residual pl gap: `lang` slot label inconsistency in core (same utterance gold-null and
      gold-lang; resolver context) — structural ceiling ~2–3 pts, not fixable by data scaling
      in the single-turn formulation.

- [x] **2. Synthetic data closed loop (Data-Turnstile style)** *(research rank 2; SLMs are
      data-bound)*
      - [x] Templated utterance generator per function schema (5 langs), seeded from core
            vocab (places, constraints, vehicle types).
      - [x] Deterministic validators = detector validators (ungrounded / enum_invalid /
            keys_mismatch) run in generation loop until pass; coverage check per
            function×lang×slot.
      - [x] Cap synthetic ≤40% of mix (same policy as current aug); dedup vs core shingles.
      **OUTCOME (2026-10-11):** `src/build_synthetic.py` → data/syn_train 5700 rows, all
      langs×fns×slots covered, 98 enum combos. en gained +3.3 F1 in ds (mostly syn), but
      syn-pl correlated with pl geocode.q regressions (multi-word q nulling) → syn_nopl
      variant tested in d5; net verdict: keep as future lever, don't ship alone.

- [x] **3. Grammar-constrained decoding for deployment** *(research rank 3; free at
      inference, zero training)*
      - [x] Export GBNF grammar (llama.cpp) or xgrammar spec per function: exact key set,
            enum vocabularies (routingMode, travel modes, avoid_features values).
      - [x] Measure: enum_invalid → 0, F1 unchanged, latency delta on 3090/CPU.
      - [x] Optional: re-run eval_extract with outlines/xgrammar to confirm no accuracy loss.
      **OUTCOME (2026-10-11):** `src/target_schema.py` + `src/export_grammar.py`
      (grammars/*.gbnf for 25 fns) + `eval/eval_constrained.py` (xgrammar LogitsProcessor).
      Full-test A/B on FT adapter: F1 0.7515→0.7510, EM 0.6671→0.6659, parse 1.0 both —
      metrics-neutral, hard schema-validity guarantee at inference. Ship-worthy for edge.

## Medium priority

- [x] **4. Hammer-style negative training** *(research rank 4; targets the 128 hallucinated
      slots)*
      - [x] Build adversarial rows: near-miss schemas (similar param names/descriptions),
            utterances that mention plausible-but-wrong values, all-null gold targets.
      - [x] Mix ~5–10% negatives into training data; retrain; acceptance: FP slots −20%,
            null acc ≥ 0.96 maintained.
      **EXECUTED 2026-10-11:** `src/build_negatives.py` → data/neg_train 1067 rows (B1
      avoid-use-mentions 228 / B2 chatter place-mentions 372 / B3 vehicle-incidental 240 /
      A constraint-phrase-under-geocode 260; lang slot off-limits — structural label noise).
      Lexical guards (no route/search intent) since enum grounding cannot validate
      mention-vs-extraction traps. Run `llama-3.2-1b-neg` = core+aug+neg.
      **OUTCOME (2026-10-11): FAIL.** F1 0.7368 (−1.5), true-halluc FP slots 128→126
      (noise; lang −5, q +4, avoid +2), total FP 185→193, null_acc 0.9632 held. Trap-template
      negatives at ~7% of the mix do not generalize to real-dialog hallucinations; slot
      shifts are within ±run variance. Details: Round 4 §N in reports/RESULTS.md.

- [ ] **5. Output-format ablation** *(research rank 5; cheap experiment — ACL 2026 says
      format skews IE accuracy)*
      - [ ] Render targets 3 ways: compact JSON (current) / slot-per-line / span list.
      - [ ] Train llama-1b LoRA per format (3 short runs); eval same test set; keep winner.

- [ ] **6. BERT-tagger agreement signal for the detector** *(research rank 6; complements
      logprob combiner, AUC 0.77 today)*
      - [ ] Train tiny per-function span tagger (multilingual BERT mini or DistilBERT) on
            core_train spans (we have char spans from grounding).
      - [ ] Agreement feature = fraction of slots where tagger span == model value;
            add to combiner features; acceptance: AUC +0.01–0.02 on test.

## Low priority / opportunistic

- [ ] **7. Q4 quantization + validation of merged winner** (prerequisite for edge claims;
      IFFC result suggests the decomposed design survives aggressive quantization).
      - [ ] llama-quantize to Q4_K_M GGUF; re-run eval via llama.cpp on test; acceptance:
            F1 drop < 1.5 pts.
- [ ] **8. End-to-end cascade prototype** (detector → escalation): wire combiner threshold
      at ~30% escalation (catches ~50% errors today) to gpt-5.4-mini (free tier; KEY_1/KEY_3
      ≈ 250k big-model + 2.5M mini tokens/day fallback budget) on escalated rows;
      report final system F1/EM vs compute budget.
- [ ] **9. ASR-noise augmentation** for future voice input: perturb training utterances with
      phonetic confusions on place names (foreign-name ASR errors are the expected car-mode
      failure).
- [ ] **10. Committee deployment option**: majority vote of h90 trio already beats members
       (EM 0.6634); package vote path if 3×1B inference fits the target hardware budget.

## Explicitly not doing

- AgentWeave-style pre-routing (function already selected upstream).
- Hidden-state probes round 2 (failed at 915 rows/2048 dims — revisit only with ≥5k
  labeled errors or a bigger model).
