# Jev Decision-Model Probes — Results Explained in Plain Language

*Plain-language companion to the formal write-up in `reports/RESULTS.md` (§ Addendum: Jev
decision-model probes). Raw numbers: `eval/results/jev_enum_val.json`,
`eval/results/jev_verify_llama-1b-merged-test.json`. Date: 2026-10-10.*

## What we tested, in plain terms

**What Jev is:** a special kind of AI model that can't write text at all. It can only answer
pre-made question types: "pick one option from this list" (`choice`) and "yes or no, with a
probability" (`noul`). Think of it as a multiple-choice examiner rather than a writer.

We ran two experiments asking: *"Could this thing help with our task — filling in navigation
parameters like destination, language, avoid-tolls from what the user said?"*

---

## Experiment 1: Can Jev fill in the blanks by itself?

Our enum slots (parameters with a fixed menu of options, like `routingMode: fast/short` or
`avoid_features: tolls/ferry/motorway/tunnel`) are perfect for Jev's question style. So for
each row we asked it:

- "Does the user actually mention this parameter, yes or no?" (the yes/no question)
- "If yes, which option do they want?" (the multiple-choice question)

**Result on 227 validation rows (cost: ~1 cent):**

| | Jev | Our fine-tuned 1B model |
|---|---|---|
| Slot accuracy (F1) | 0.70 | **0.90** |
| Correctly saying "not mentioned" | 95% | **98%** |
| All parameters in a row correct | 76% | **92%** |

**Translation:** Jev is decent for a model that was never trained on our data — but our small
local model, which we spent the whole project training, clearly wins. Jev is like a smart
stranger answering a quiz; our model has done the homework.

**The language problem:** this was the big discovery. Jev's accuracy by language:

- French: 0.88 (great)
- English: 0.75, Spanish: 0.73, German: 0.70 (okay)
- **Polish: 0.36 (barely better than guessing)**

So the warning we found during research — "Jev is weak outside English" — turned out to be
true, and it's concentrated in Polish. For a 5-language in-car assistant, that's close to a
dealbreaker for using Jev as the main extractor.

---

## Experiment 2: Can Jev be the proofreader?

The idea (from OpenRouter's own cookbook): let our cheap local model do the extracting, then
have Jev double-check each answer — *"Is this value actually supported by what the user
said?"* Only rows Jev flags as suspicious get escalated to something expensive. We measured:
**how well does Jev's suspicion predict real mistakes?**

**Result on 817 test rows (cost: ~2 cents):**

| Failure detector | AUC (0.5 = coin flip, 1.0 = perfect) |
|---|---|
| Jev's suspicion score | 0.58 |
| Our existing method (model's own confidence / logprobs) | **0.75** |
| Both combined | 0.72 (worse than ours alone) |

**Translation:** Jev is nearly useless at spotting our model's mistakes — about as good as
the free rule-based checks we already have (0.57). It actually made our detector *worse*
when combined.

**Why it failed — this is the interesting part.** Jev checks *"is this answer grounded in the
user's words?"* But look at our model's two most common errors:

1. **The `lang` slot** (26% of its errors): the model outputs "de" when the answer should be
   "en". Both values appear plausible from the utterance — nothing is hallucinated. Jev sees
   a grounded answer and says "looks fine."
2. **The `q` slot** (22% of errors): the user says *"Lyon, France"* and the model extracts
   *"Lyon, France"* when gold is just *"Lyon"*. Both are literally in the utterance.
   Grounded? Yes. Correct? No.

Jev only catches **hallucinations** — answers made up out of thin air. But our fine-tuned
model barely hallucinates (that was never its weakness). Its mistakes are *subtle judgment
calls*, and a grounding check is structurally blind to those. Where Jev DID score well
(routingMode 0.92, destination 0.75) are slots where errors almost never happen in the
first place.

---

## Bottom line

- **Jev as extractor:** works okay, but our trained model beats it, and it fails badly in
  Polish.
- **Jev as quality control:** sounds great in theory, fails in practice — it catches
  fabricated answers, but our model's real mistakes are "true but subtly wrong," which Jev
  cannot see.
- **Our existing confidence-based escalation stays.** Nothing we tested beats it.
- For ~$0.04 total, both questions are now answered with data instead of speculation — and
  the results are documented in `reports/RESULTS.md`.

The one untested idea worth keeping in a drawer: feeding Jev's per-slot suspicion into our
learned detector (`train_detector.py`) as one extra signal for enum slots only. Given how
weak the row-level signal was, expected gains are small.
