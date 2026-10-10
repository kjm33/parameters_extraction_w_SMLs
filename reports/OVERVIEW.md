# Slot-Filling Project — Plain-Language Report

Big-picture summary of what was done, how good the models are, and why.
(Full numbers and tables: `reports/RESULTS.md`.)

## 1. The goal

Build a small AI model for an **in-car navigation assistant**. When the driver says something
like *"Take me from Berlin to Munich, avoiding tolls"* and the system has already chosen a
function (e.g. `here_directions`), the model's only job is to read the sentence and fill in
that function's parameters as JSON:

```
{"origin": "Berlin", "destination": "Munich", "avoid_features": ["tollRoad"], "lang": "en", ...}
```

Rules of the task:
- A value is only filled in **if the driver actually said it**. Anything not spoken → `null`
  (no guessing, no inventing).
- Works in **5 languages**: English, German, Polish, French, Spanish.
- The model must be small enough to run **in the car** (on-edge hardware) after 4-bit
  quantization — no cloud needed.

This "decomposition" approach (one tiny model that just extracts parameters) is much more
reliable than asking a small model to do full function-calling in one shot (the sibling fcft
project showed 3–4B models only reach ~40% argument accuracy that way).

## 2. What was done — step by step

**Step 1: Built the training data (~15,000 examples).**
- **Core (9,186 rows)**: taken from the sibling repo's *verified* dialogs — real conversations
  where the tool calls were actually executed successfully. The correct answers therefore come
  from real, working API calls. Only values that literally appear in the sentence were kept as
  targets; everything else was set to `null`. Hard multilingual details were handled: Polish
  inflections ("do Warszawy" → `Warszawa`), "Lyon, France" → `Lyon`, keyword dictionaries for
  enums (tolls/ferries/motorways) in all 5 languages.
- **Augmentation (5,818 rows)**: generated from the instruction-extraction repo's texts
  ("drive from X to Y avoiding highways") using per-language keyword rules, with duplicate
  detection so nothing overlaps the core set. English was boosted on purpose.
- **Test set (817 rows)**: separate, execution-verified, language-balanced, never seen during
  training. This is what all scores below come from.

**Step 2: Trained 5 models** — Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B, gemma-3-1b,
Llama-3.2-3B.
- Method: **LoRA** = cheap fine-tuning that trains a tiny adapter (~1–2% of the weights)
  instead of the whole model. 3 epochs, ~1.5–3.5 h each on one RTX 3090.
- Also ran **zeroshot baselines** (no training) and **full fine-tuning** (all weights) for the
  two smallest models, to know whether the extra cost is worth it.

**Step 3: Evaluated everything** with strict metrics: slot F1 (did it extract the right
values?), null accuracy (did it stay quiet when nothing was said?), row exact-match (whole
JSON perfect?), parse rate (valid JSON at all?). Plus per-language and per-function breakdowns
and a manual error analysis.

**Step 4: Shipped the winner** — merged into a single standalone model folder,
sanity-checked, documented in `reports/RESULTS.md`.

## 3. How good are the models?

| Model | Zeroshot F1 | After fine-tuning F1 | Rows 100% correct |
|---|---|---|---|
| Qwen2.5-0.5B | 0.05 | 0.719 | 64% |
| Qwen3-0.6B | 0.12 | 0.732 | 64% |
| **Llama-3.2-1B (winner)** | 0.11 | **0.7515** | **67%** |
| gemma-3-1b | 0.04 | 0.726 | 64% |
| Llama-3.2-3B | 0.10 | 0.745 | 65% |

### How to read these metrics

Imagine the driver says: *"Take me from Berlin to Munich, avoiding tolls"* — the perfect
answer is `{"origin":"Berlin","destination":"Munich","avoid_features":["tollRoad"],"lang":"en","departure_time":null}`.

| Metric | Simple meaning | Example of a failure it catches |
|---|---|---|
| **Parse** | Did the model produce valid JSON at all? | Model rambles or breaks the format |
| **P (precision)** | Of the values the model **did** fill in, what fraction were correct? | Model invents `"avoid_features":["ferry"]` nobody mentioned → precision drops |
| **R (recall)** | Of the values that **should** be filled (actually spoken), what fraction did the model find? | Model outputs `null` for Munich → recall drops |
| **F1** | The balance of P and R (one number for both). 0.75 ≈ "3 out of 4 values handled perfectly" | |
| **Null acc** | When the right answer was "not said → stay quiet", how often did the model actually output `null`? This is the *honesty* score. | Driver never mentioned time; model makes one up |
| **Row EM** | % of answers where the **entire** JSON was 100% perfect — every value AND every null. The strictest, most "production-like" number. | One wrong key anywhere ruins the whole row |

Rule of thumb: **P + null acc = "doesn't invent things", R = "doesn't miss things", EM =
"whole answer trustworthy"**.

In plain terms:

- **Fine-tuning is the whole story.** Out of the box these models are useless for the task
  (F1 4–12%) — they hallucinate values for things the driver never said (null accuracy as low
  as 30%). After training: F1 ~0.72–0.75, **100% valid JSON**, ~96% correct "stay quiet"
  behavior.
- **The winner is Llama-3.2-1B + LoRA**: best F1 (0.7515) and best exact-match (66.7%).
  Exported at `train/runs/llama-3.2-1b/merged/` — ~2.5 GB as-is, **~0.8 GB after 4-bit
  quantization** → genuinely edge-deployable.
- **Bigger is NOT better.** The 3B scored *worse* than the 1B despite fitting the training
  data better — it over-extracts (recalls more, but invents more). The task is limited by
  extraction discipline, not model capacity. This saves a lot of memory and compute.
- **Full fine-tuning ≈ LoRA** (within 0.6 pts) but produces 2–6× larger checkpoints — LoRA is
  the right choice.
- **Even the tiniest 0.5B is only ~3 points behind** the winner — a real option for very
  constrained hardware.

## 4. Where the remaining errors are

Of ~1,700 real values in the test set, the winner missed or got wrong ~193 and invented ~128.
Three causes dominate:

1. **The `lang` field (biggest issue)**: the rule "lang = language of the conversation" breaks
   when, say, a French sentence discusses Warsaw and Lisbon — the model gets tempted by the
   foreign place names. ~150 errors alone.
2. **Where does the query start/end**: "our hotel on Maple Street" — is the query
   "Maple Street" or "our hotel on Maple Street"? Genuinely ambiguous; humans would disagree
   too.
3. **Constraint keywords in inflected languages** (German compounds, Polish endings): lexicon
   gaps in training data.

There is also a consistent **language gap**: Polish/French/Spanish score 4–8 points below
English/German for every model — likely fixable with more balanced augmentation data.

(Note: a few functions show "F1 = N/A" — that's not a failure; their test rows simply contain
no spoken values, so the correct answer is all-null, which is measured by null accuracy
instead.)

## 5. Bottom line

~15k verified examples + a few GPU-hours of cheap LoRA training turned a generic 1B chat
model into a **reliable multilingual parameter extractor**: perfect JSON format, 96% honest
about unspoken values, 75% of individual values exactly right, 67% of whole answers flawless —
in 5 languages, in a package that quantizes to under 1 GB for the car. The clearest next
wins: fix or drop the `lang` field rule, rebalance augmentation toward pl/fr/es, and extend
the constraint keyword lexicons.

## Artifacts

- Winner model (merged, standalone): `train/runs/llama-3.2-1b/merged/`
- All metrics: `eval/results/*.json`; per-row predictions: `*_preds.jsonl`
- Full technical report: `reports/RESULTS.md`
- Pipeline code: `src/` (data), `train/` (SFT), `eval/` (metrics + error analysis)

## 6. Update: teaching the system to know when it's wrong (round 2)

The first detection round (chapter above / "Failure detection" in RESULTS.md) found one
simple, free signal: the model's weakest-word confidence (min logprob), quality 0.75–0.77
(AUC). We then tried to beat it with five fancier ideas. To get honest training labels, three
models were retrained on 10% less data; their mistakes on the unseen 10% became the
detector's textbook.

What we tried and what happened, in plain words:

| Idea | What it is | Result |
|---|---|---|
| **Learned combiner** | A small statistical recipe mixing ~15 signals (word confidences, agreement between 5 attempts, sanity-check flags…) | **Wins: 0.763–0.772** on 3 of 4 models (+1–3 pts). Modest but real, and it transfers to the final winner model. |
| Brain probe | Look inside the model's activations at the moment it answers | Flop (0.60–0.68). Works in papers on big cloud models, not on our tiny fine-tuned one with few examples to learn from. |
| Committee disagreement | 3 models vote; disagreement = suspicion | Surprise: voting **fixes** errors (67% perfect answers vs 64–66% for each model alone) — but disagreement itself detects poorly (0.65). Use the committee for accuracy, not detection. |
| Calibrated auto-accept | "Accept only answers where I guarantee ≤X% chance of error" | Works reasonably: promising ≤5% error auto-accepts ~28% of requests at ~93% real accuracy. |
| Per-value entropy | How scattered are the 5 sampled attempts for each individual value? | Middle of the pack (0.66–0.68) — no gain over simpler signals. |

**Bottom line of round 2**: the simple free signal was already near the ceiling — a learned
combiner squeezes out a few more points (0.77), and a 3-model committee is better used to
*correct* answers than to *doubt* them. For the car: run the winner, auto-accept the ~45%
most confident answers (≈85%+ accurate), and escalate the rest to the bigger model or ask
the driver to confirm.
