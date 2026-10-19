# Gold-v3 teacher bake-off — results (r12a, 2026-10-13)

500-row stratified sample (seed 20261013, identical rows for all teachers; lang×fn over the
full 16,616 frame). Slate extended mid-round from Big-3 to 6 teachers (user-approved; see
policy §2). Costs are live totals incl. one infra-broken first attempt of t5.

## Final table (t6 = first-attempt data, re-run still in flight)

| teacher | model | parse | grounded | EM vs v2 | EM\|g | cost/500 | est. full sync |
|---|---|---|---|---|---|---|---|
| t1 | anthropic/claude-haiku-5.5 | 0.966 | 0.880 | 0.332 | **0.377** | $0.142 | ~$4.2 |
| t2 | openai/gpt-5.4-mini | 1.000 | 0.916 | 0.170 | 0.186 | $0.365 | ~$12.1 |
| t3 | openai/gpt-6-luna | 1.000 | 0.904 | 0.174 | 0.193 | $0.108 | ~$3.6 |
| t4 | z-ai/glm-5.3-flash | 1.000 | 0.896 | 0.206 | 0.230 | **$0.030** | **~$1.0** |
| t5 | deepseek/deepseek-v4.1-flash | 0.910 | 0.838 | 0.158 | 0.189 | $0.622† | ~$20† |
| t6 | inclusionai/ling-3.1-flash | 0.368‡ | 0.188‡ | 0.036 | 0.192 | $0.000 | infeasible§ |

† completion-heavy (reasoning tokens; ~1850 out/row even at effort low). Batch tier cuts
  the dominant completion side ~3x → ~$8 full.
‡ first attempt: 223×HTTP 429 (free-tier) + truncation at max_tokens 1024; slow re-run
  (workers 2, max_tokens 4096) still in flight — report auto-refreshes on completion.
§ ~16 s/row sustained even when healthy → ~74 h for 16,616 rows. Free ≠ usable at scale.

## Failure-mode taxonomy (systematic vs retryable)

- **t2 q_boundary 68**: over-trims valid q phrasings (annex-violating, systematic).
- **t3 word_comma_tail 89**: re-adds ", France"-style tails user never said (systematic).
- **t1 blank_receipt 17 + 17 parse fails**: rejects/parse-fails are *re-askable*, not silent
  corruption; smallest systematic classes overall (word_comma 15, q_boundary 14).
- **t4** balanced small classes (q_boundary 19, word_comma 17, blank_receipt 17); v2n=91
  (most aggressive nulling after t2) — check in spot-check.
- **t5** 45 parse fails = reasoning + pretty-printed JSON overflowing even 4096 tokens
  (usage log shows completion_tokens == 4096 exactly on fails); would need re-ask loop.

## Pairwise inter-teacher EM (n = both grounded)

Highest pairs: t1~t4 **0.533**, t4~t5 0.540, t1~t3 0.510. Lowest: t1~t2 0.365, t2~t4 0.367.
t2 and t4 agree least with everyone else despite high grounded rates → their extra "grounded"
rows carry systematic policy drift, not shared truth.

## Polish slice (n=108)

| t | grounded | EM\|g | n2v |
|---|---|---|---|
| t1 | 0.741 | **0.300** | 75 |
| t4 | 0.787 | 0.212 | 77 |
| t5 | 0.676 | 0.205 | 54 |
| t2 | 0.796 | 0.151 | 85 |
| t3 | 0.722 | 0.154 | 72 |

t1 clearly strongest on the risk language; challengers t4/t5 beat t2/t3 but trail t1.

## Structural v2 defect (flip #2, see policy §7)

All here_directions rows (~28%): v2 nulls origin/destination/transportMode unconditionally
(builder rule "objects: always null", src/build_extraction.py:12) even when spoken. All
teachers fill verbatim spoken place names (destination n2v 139/139/138/… identical across
families). EM vs v2 is depressed *by construction* on these rows for every teacher.

## Freeze (v3full, 2026-10-13)

t1 full: 16,616 rows, $4.44, parse 0.971, grounded 14,877 (0.895). t4 pass on the 1,739
rejects: $0.22, recovered 675. Final `data/*_v3.jsonl` (positional alignment — 297 dup ids
with different content make id-joins unsafe; `src/freeze_gold_v3.py`):

| file | rows | t1 | t4 | v2_fallback | EM vs v2 |
|---|---|---|---|---|---|
| core_train | 9,186 | 7,781 | 538 | 867 | 0.463 |
| core_val | 489 | 414 | 24 | 51 | 0.483 |
| core_test | 817 | 756 | 32 | **29 (3.6%)** | 0.406 |
| aug_train | 5,818 | 5,628 | 76 | 114 | 0.259 |
| aug_val | 306 | 298 | 5 | 3 | 0.261 |
| overall | 16,616 | 14,877 | 675 | 1,064 | 0.386 |

Quirks logged in `data/v3_stats.json`: 594 rows share 297 duplicate ids; 46 legacy-key rows
(v2 target keys outside current schema — preserved via merge); t1-reject profile: 489
parse_fail, ~935 not-verbatim (free-text slots), 159 extra_keys. Total teacher spend $5.10.

## Recommendation

**DECIDED (user, 2026-10-13): t1 haiku-5.5 is the primary teacher.** Full run launched
(tag `v3full`, ~$3.2 projected, ~2.7 h). Spot-check adoption gate still applies before v3
replaces v2 as headline.

1. **Primary teacher: t1 haiku-5.5** — best EM|g overall and on pl, smallest systematic
   violation classes, failures are retryable rejects. Full run ~$4.2 sync (batch $2.1).
2. **Second opinion: t4 glm-5.3-flash** — different model family (decorrelated errors),
   highest agreement *with t1* (0.533), no dominant violation class, ~$1 full run.
   Proposed full-run config: t1 primary + t4 on t1-rejects (and optionally a 5% consensus
   audit sample where t1~t4 disagree → route to spot-check, not auto-resolve).
3. t2/t3 rejected (systematic annex violations = silent corruption on 15–18% of rows).
4. t5 usable only with re-ask loop + batch tier; not worth it vs t1+t4 pricing.
5. t6 infra-infeasible at full scale; keep first-attempt data as evidence.

Pending: 50-row manual spot-check (efforts/2026-10-13-gold-v3/2026-10-13-gold-v3-spotcheck.md) — user verdicts
decide the adoption gate (≥95% agreement) and confirm flip #2 before freezing v3.
