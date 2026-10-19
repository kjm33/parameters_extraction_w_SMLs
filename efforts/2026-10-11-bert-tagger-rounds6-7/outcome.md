# Outcome — BERT tagger improvements + benchmark v2 (plain language)

## What we did

Tried to close the tagger's 3.6-pt gap with smarter post-processing and more data, then —
the bigger move — fixed the ruler: dropped the noisy `lang` slot from the metric (round 3
had shown the evaluator was stricter than the data builder) and re-ran everything as
benchmark v2.

## What worked

- **`eval/canonicalize.py` ships**: train-derived alias map ("krakowa→Kraków") +1.69 F1 on
  the tagger, ≈neutral on the JSON winner. Simple dictionaries beat fancy gazetteers
  (confirmed hard in round 11).
- **Distillation data helps the tagger** (+0.5–1.6 F1) even though it was noise for the
  JSON models — encoders soak up grounded rows.
- **Honest correction**: the B1 audit falsified our own round-5 census — the "appendage
  bug" was gold-side knowledge (", CA" never spoken), unfixable at data level.
- **Benchmark v2 re-baselined everything**: winner llama-3.2-1b LoRA v2 F1/EM 0.7958/0.7956;
  edge bert-tagger-fixd-v2+canon 0.7813/0.7846 (4.3 ms/row GPU). XLM-R documented as
  latency/EM alternative.

## What didn't

- B4 CRF skipped (bar not reached); replacing the JSON winner with a tagger: no.
- pl stays the structural weak spot for all taggers (.61–.62 vs winner .74).

## Bottom line

A legitimate v2 leaderboard with an adopted edge candidate and a shipping canonicalizer;
the lang-free metric becomes the official currency for rounds 8–12.
