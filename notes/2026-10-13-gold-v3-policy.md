# Gold-v3 Policy (pre-registered) — Round 12 Phase A

Date: 2026-10-13. Written BEFORE any teacher output is seen (pre-registration). Binding
decisions come from `plans/2026-10-13_gold-v3-round12-todo.md` (user-approved 2026-10-13).
v2 stays frozen; v3 targets replace v2 targets only after the adoption gate passes.

## 1. Scope

All core+aug rows (16,616): core_train 9,186 / core_val 489 / core_test 817 /
aug_train 5,818 / aug_val 306. Prompts (system+user bytes) stay IDENTICAL to v2 rows;
only `target` is regenerated. `lang` field in data unchanged; `lang` target key stays null.

## 2. Teacher slate (bake-off, 500 stratified rows each)

| slot | model | why |
|---|---|---|
| T1 | `anthropic/claude-haiku-5.5` | highest AA intelligence in price class (43.4), $0.10/$0.50 |
| T2 | `openai/gpt-5.4-mini` | plan's reference teacher; supports reasoning effort none |
| T3 | `openai/gpt-6-luna` | substitute for gemini-2.5-flash-lite (AA 38.1 at $0.10/$0.50) |

`google/gemini-2.5-flash-lite` was DROPPED from the plan's slate: OpenRouter lists
expiration 2026-10-20 (mid-run), and it is the weakest of the trio (AA 10.4). Decision
made 2026-10-13 before any teacher call; user approved the bake-off with this slate.

**Slate extension (post-Big-3, user-approved 2026-10-13):** three non-Big-3 challengers
added on the SAME 500-row sample (same seed), same tag `r12a`:
T4 `z-ai/glm-5.3-flash` (AA 41.8, reasoning effort low — mandatory reasoning),
T5 `deepseek/deepseek-v4.1-flash` (AA 39.5, reasoning effort low — default high),
T6 `inclusionai/ling-3.1-flash` (AA 41.1, FREE, no reasoning extras; rate-limit risk
accepted). Rationale: decorrelated error families (not Anthropic/OpenAI) + historically
strong Polish. All three kept for the full run only if they beat the Big-3 on the same
acceptance criteria (parse/grounded rate, no systematic annex-violation class, spot-check).

**Teacher decision (user, 2026-10-13, after r12a):** T1 haiku-5.5 = primary;
T4 glm-5.3-flash = second opinion on T1 rejects/audit. Results:
`notes/2026-10-13-gold-v3-bakeoff-results.md`.
Sampling: (lang × function-name) stratified, seed 20261013, N=500 over the full 16,616
frame (all five files pooled). Budget guard: hard cap $2/teacher for the bake-off.

## 3. Target policy rules (what "correct" means in v3)

1. **Verbatim-first.** Free-text values are copied from the utterance, in the original
   language/script. Two named exceptions: (a) Polish inflected place forms → base form
   (v1 alias-map style: "Torunia"→"Toruń", "Krakowa"→"Kraków", "z Białegostoku"→"Białystok");
   (b) canonical enum literals (exact schema enum value).
2. **Appendages** (", DE", ", France", " in Spain" style tails): include ONLY if the user
   actually spoke them. DEFAULT = only-if-spoken. Data-driven flip to always-for-US-place-names
   is allowed ONLY from A2 bake-off evidence, BEFORE v3 freeze; decision + evidence recorded
   in this note (§7).
3. **avoid-type params from utterance INTENT**, never from the reference tool call.
   Mention ≠ extraction. Polarity: "po autostradzie" / "via X" / "around X" = NOT avoid.
   Adjacent-feature precision: "płatnych autostrad" (toll motorways) → tollRoad, not motorway;
   "unpaid roads" / "bez autostrad" → motorway.
4. **Boundaries**: value = only the words that name the queried place. Drop discourse fluff
   ("our hotel on ..." → "Maple Street") but KEEP meaningful qualifiers the user said
   ("Detroit, Michigan"). No word-comma tails ("ulica, Marszałkowska" → "Marszałkowska").
5. **Honest null**: a key is non-null ONLY if grounded in the utterance (incl. pl base-form
   alias). Numeric params (limit/radius/coords) non-null only when the user states a number
   explicitly. Never copy reference `args` verbatim.
6. **lang**: always null (v2 rule, unchanged).

## 4. Teacher prompt

- system = `render.SYSTEM_PROMPT` (byte-identical to training) + TEACHER_ANNEX (below).
- user = `render.user_content(row)` byte-identical (no NER hints, no span numbering).
- temperature 0; max_tokens 1024; gpt-5.4-mini `reasoning.effort=none`, claude-haiku-5.5
  `reasoning.effort=low`, gpt-6-luna default.

**TEACHER_ANNEX** (verbatim, embedded in `src/build_gold_v3.py`):

```
ADJUDICATION POLICY (follow strictly):
- One key per schema parameter, null when the message does not state it. lang is always null.
- Free-text values: copy the user's words verbatim, except Polish place names, where you
  output the dictionary base form (e.g. "z Torunia" -> "Toruń", "do Białegostoku" ->
  "Białystok", "Krakowa" -> "Kraków").
- Include a trailing country/region qualifier (like ", France" or ", DE") ONLY if the user
  actually said it. Do not add qualifiers the user did not say.
- The value contains only the words naming the place: drop phrases like "our hotel on",
  but keep meaningful qualifiers the user said, like "Detroit, Michigan".
- Enum parameters: output the exact enum literal from the schema, chosen from the user's
  intent. "via X" / "around X" / "po autostradzie" is NOT an avoid request. "płatnych
  autostrad" (toll motorways) means tollRoad, not motorway. An avoid parameter is non-null
  only when the user asks to AVOID that feature.
- Numeric parameters: non-null only when the user explicitly states the number.
- Output ONLY the JSON object.
```

## 5. Grounding validators (reject, not auto-fix)

Reuse `src/build_extraction.py` helpers (`fold`, `tokens`, `tok_match`, `verbatim`,
`number_found`, `enum_supported`). A teacher row is GROUNDED iff: parses as JSON object;
key set ⊆ schema params (missing keys → filled null, flagged); every non-null value passes
its type check (string: `verbatim` or `tok_match`, where the pl base-form alias counts as
verbatim; number: `number_found`; enum: exact canonical literal after fold match).
Ungrounded rows are logged with reason and EXCLUDED from teacher-quality metrics (reported
as grounded-parse rate).

## 6. Bake-off procedure + adoption gate

1. 500 rows × 3 teachers (~$0.30 total; usage logged to `logs/goldv3_usage.jsonl`).
2. Per teacher: grounded-parse rate; disagreement profile vs v2 gold (slot-level
   null↔value flips, value≠value; tagged by noise class: blank-receipt, appendage-code,
   word-comma, q-boundary≥3 words, pl-canonical).
3. 50-row stratified manual spot-check WITH user (pl rows ≥12 and scored separately).
4. **Adoption gate for the winning teacher → full run**: spot-check agreement ≥95% AND
   measured avoid blank-receipts ≈ 0. v3 becomes headline only after gate passes.
5. Full run (winner only, one pass): mode sync (OpenRouter) or `:batch` variant; NO
   iterative re-annotation loops without a new user decision.

## 7. Pre-registered flip decisions (to fill from A2 evidence BEFORE freeze)

- Appendage flip: [pending A2 — flip to always-for-US iff bake-off evidence shows teachers
  systematically under-include spoken-or-not US qualifiers AND user spot-check prefers it]
- **here_directions coords-class params (NEW, discovered in bake-off r12a, 2026-10-13)**:
  v2 nulls `origin`/`destination` (and `transportMode`) on ALL here_directions rows by
  builder rule ("objects: always null", build_extraction.py:12) even when the user states
  the places ("de Lyon a Marsella"). All three teachers independently fill them with the
  verbatim place names (destination n2v = 139/139/138 — identical across teachers =
  systematic). Proposed v3 rule: object/coords-typed params take the SPOKEN place name
  (verbatim-grounded); only truly unstated/computed values stay null; `transportMode`
  follows user intent ("drive me" → car) per annex enum rule. PENDING user spot-check
  verdicts on the directions rows.
- Teacher winner: [pending bake-off + spot-check]
- (anything else that changes §3 must be recorded here with evidence before freeze)

## Appendix — adversarial classes (from round-11 census, teacher must survive these)

1. Blank receipts: explicit avoid-request in utterance, v2 gold null (49 train / 20 test);
   cue-lexicon undercounts (765/47 without cue) — teacher adjudicates by intent.
2. Unspoken appendages: ", XX" 2-letter codes (101 train / 23 test); word-comma tails
   (59/15).
3. q-boundary: ≥3-word q rows 5.4% train / 7.9% test — fluff vs qualifier distinction
   (GLiNER failed exactly here, −1.0 F1: "our hotel on Maple Street" vs "Detroit, Michigan").
4. Verbatim beats "correction": never normalize Stockholm→Sztokholm etc. (GeoNames rewrite
   was 111/114 harmful); the ONLY rewriting allowed is the pl base-form alias (§3.1).
5. Enum adjacency: toll/near-toll feature confusions ("płatnych autostrad").
