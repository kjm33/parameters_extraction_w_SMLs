# Efforts index

What has been done. Current state: `../STATE.md`. Not yet done: `../BACKLOG.md`.
Per-effort dirs: `YYYY-MM-DD-<slug>/` (spec.md + tasks.md + outcome.md). Template: `_TEMPLATE/`.

| Started | Effort | Status | Outcome (one line) |
|---|---|---|---|
| 2026-10-07 | [baseline-pipeline](2026-10-07-baseline-pipeline/) | done | llama-3.2-1b LoRA winner F1 0.7515 / EM 0.667 (n=817); 3B < 1B, LoRA ≈ full-FT, pl/es/fr trail en/de by 4–8 pts |
| 2026-10-09 | [failure-detection](2026-10-09-failure-detection/) | done | min-logprob AUC 0.755–0.765 (best single), combiner 0.763–0.772; cascade escalates 30% → catches ~50% of errors |
| 2026-10-10 | [classical-nlp-assistance](2026-10-10-classical-nlp-assistance/) | done | lenient-rescoring gains (+2.4–3.7 F1) were eval artifacts; both interventions dropped |
| 2026-10-10 | [jev-probes](2026-10-10-jev-probes/) | done | Jev-probe addendum shipped (explanation + report); per-slot Jev risk parked as future combiner feature |
| 2026-10-10 | [related-work-improvements](2026-10-10-related-work-improvements/) | done (1–4) | grammar-constrained decoding SHIPS (parse 1.0); distill & negatives FAIL; synthetic kept as lever; items 5–10 → BACKLOG |
| 2026-10-11 | [bio-tagging-formulation](2026-10-11-bio-tagging-formulation/) | done (rejected) | decoder tagger 0.4745 (word-counting failure); BERT tagger 0.7152 — the edge lane is born; 91.1% structural ceiling |
| 2026-10-11 | [bert-tagger-rounds6-7](2026-10-11-bert-tagger-rounds6-7/) | done | canonicalize.py ships (+1.69 F1 tagger); benchmark v2 (lang-free) adopted; edge model 0.7813/0.7846 @ 4.3 ms GPU |
| 2026-10-12 | [mmbert-backbone](2026-10-12-mmbert-backbone/) | done | mmBERT tie 0.7838 (inside ±0.5 band); flash-attn ~16% slower; lane closed |
| 2026-10-12 | [spanlist-formulation](2026-10-12-spanlist-formulation/) | done (rejected) | spansn 0.7748 but canonical fidelity collapses (pl −6.9 / en −6.4); grounding ≠ anti-hallucination |
| 2026-10-12 | [external-ner-ontologies](2026-10-12-external-ner-ontologies/) | done | all 6 external NER lanes closed (GeoNames neutral, matcher/GLiNER/Wikidata/Stanza killed, Flair NC-license); NER never the bottleneck |
| 2026-10-13 | [gold-v3](2026-10-13-gold-v3/) | **active** | v3 frozen: 14,877 grounded + 675 glm-recovered + 1,064 v2_fallback (~$4.66); adoption gate pending 50-row spot-check ≥95%; Phases B–D remain |
