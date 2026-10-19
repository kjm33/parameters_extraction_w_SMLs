# Outcome — External NER / ontologies (plain language)

## What we did

Six pre-registered experiments trying to bolt external knowledge (gazetteers, NER systems,
embeddings) onto the finished stack as post-processors. Analogy: we tried six smarter
spell-checkers for a student who actually mishears the question.

## What worked

- Discipline: every item got a measured verdict; two were killed before wasting build time
  by measuring the premise first (0 invalid enums to fix; 0-fire layer to enrich).
- The round's real yield is the cross-cutting diagnosis (below) — it redirected all later
  work to the gold data itself (round 12).

## What didn't

- **GeoNames canonicalize v2: NEUTRAL** (fires 0/0 after safety gates; ungated it would
  corrupt — Stockholm→Sztokholm). Gold is utterance-verbatim; the train-derived v1 alias
  map already carries the recoverable value.
- **GLiNER arbitration: FAIL** (winner −1.02 F1): "our hotel on Maple Street" (fluff) vs
  "Detroit, Michigan" (gold qualifier) are geometrically identical — spans can't arbitrate.
- **Flair: DROP** (wins only de, 1/5; and ALL entity-* models are noncommercial — the
  research note's license claim was wrong).
- **Wikidata, Stanza, embedding-matcher: KILLED** — coverage ceiling is the gold value
  distribution (parking/POI phrases aren't NER entities), not span quality.

## Bottom line

The external-NER lane is closed in every formulation (hints r3, targets r5, pointers r9,
arbitration r11). NER is never the bottleneck; the bottlenecks are gold noise and intent
judgment → that's exactly what gold-v3 (round 12) attacks. See the two recap notes in
`notes/` for the plain-language version.
