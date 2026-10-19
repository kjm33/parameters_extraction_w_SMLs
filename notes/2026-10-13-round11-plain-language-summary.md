# Round 11 Results — Plain-Language Summary (External NER, Ontologies, Post-Processing)

Date: 2026-10-13
Scope: what was checked, what worked, what didn't in the external NER / ontology / post-processing round
(fully documented in `reports/RESULTS.md` § Round 11; technical details in `efforts/2026-10-12-external-ner-ontologies/2026-10-12-external-ner-ontologies-research.md`
and outcome blocks in `efforts/2026-10-12-external-ner-ontologies/tasks.md`).

## What we checked

We tested whether **external NER systems** (GLiNER, Flair — neural networks trained specifically to find
places/addresses) and **external knowledge bases** (GeoNames gazetteer, Wikidata, DBpedia) could fix the
remaining errors of our models.

## What worked

- **Detecting entities works great.** GLiNER correctly finds cities, streets, POIs in all 5 languages — even
  inflected forms like "Torunia" (Polish for "in Toruń"). Flair is similar. Entity *detection* is a solved
  problem here.
- **Our existing simple fix (v1 alias map)** — a small lookup table learned from training data
  (e.g. "Krakowa"→"Kraków") — turns out to already capture **all** the recoverable value. It gives the small
  model +2.5 F1, mostly Polish (+12.9).
- **GeoNames gazetteer as a safety net**: after adding safety gates it never fires wrongly — but also never
  helps. A harmless no-op.

## What didn't work

1. **Rewriting names via GeoNames** — the "correct" answers quote the user's words **verbatim**. So
   "correcting" `Stockholm` to the official Polish name `Sztokholm`, or `Krakow`→`Cracovia`, *breaks* matches
   instead of fixing them. 111 of 114 rewrites on training data were harmful → we gated everything off.
2. **GLiNER arbitration** (using its spans to trim/fix predicted values) — score went **down** (−1.0 F1).
   Reason: GLiNER can't distinguish *fluff* ("our hotel **on Maple Street**" → trimming is right) from
   *meaningful qualifiers* ("**Detroit, Michigan**" → trimming is wrong). Both look identical to a span
   detector; telling them apart needs understanding *intent*.
3. **Embedding enum-matcher** — killed before building it: our models **never** emit invalid enum values
   anymore. The remaining "avoid features" errors are mostly cases where the gold answer itself looks wrong
   (user explicitly says "without highways" but gold says `null`).
4. **Flair** — only wins in German (87.5 vs 86.4), ties or loses elsewhere. And we discovered its license is
   **non-commercial only** — unusable for an automotive product anyway.
5. **DBpedia** — excluded (share-alike license); **Stanza / Wikidata / mordecai3** — skipped with evidence:
   four independent NER systems hit *exactly* the same per-language ceiling, meaning better entity finders
   won't move the needle.

## Bottom line

**The bottleneck was never entity detection — it's that gold answers quote users verbatim, and the leftover
errors need intent understanding, not better dictionaries.** External NER is now conclusively closed in every
formulation (hints, targets, pointers, post-hoc arbitration — all tested, all neutral or negative). Best
models unchanged: Llama-3.2-1B (0.796 F1), BERT tagger + alias map (0.781 F1).
