# External NER systems, ontologies & post-hoc post-processing

Status: done
Start: 2026-10-12
End: 2026-10-13

> Retrofitted 2026-10-19 from `tasks.md` (was `plans/2026-10-12_external-ner-ontology-todo.md`,
> OUTCOME blocks inside) + `2026-10-12-external-ner-ontologies-research.md` (in this dir)
> + `reports/RESULTS.md` Round 11.

## Goal

Evaluate external NER/ontology/NLP resources as strictly-offline post-hoc additions to the
stack: GeoNames gazetteer canonicalization, embedding enum-matcher, GLiNER span arbitration,
Flair benchmark, Wikidata aliases, Stanza NER — attacking canonical fidelity (pl −6.9,
en −6.4), q boundaries (37 FN + 29 FP), enum/null judgment (29 avoid_features FP).

## Expected results

- Per-item pre-registered acceptance (see `tasks.md`): adopt post-processors only if F1
  gain holds on BOTH the LLM winner and the edge tagger, en/de not worse; strictly offline
  at inference; license hygiene (CC0/CC-BY only, DBpedia excluded).

## Non-goals

- NER as prompt hints or training targets (round-3 failure); pointer outputs (round-9
  rejection); Elasticsearch/Postgres at edge runtime; anything needing intent understanding
  built post-hoc.

## Acceptance criteria (as was)

- [x] Items 1–6 executed/closed with measured verdicts (NEUTRAL / KILLED / FAIL / DROP).
- [x] Items 7–10 left unchecked → BACKLOG.md (low-priority opportunistic).

## References

- `tasks.md`, `2026-10-12-external-ner-ontologies-research.md`,
  `notes/2026-10-13-round11-plain-language-summary.md`,
  `notes/2026-10-13-ner-never-the-bottleneck-verbatim-gold-and-intent-errors.md`.
