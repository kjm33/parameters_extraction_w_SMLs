# Related Work — Methods for Slot Filling / Parameter Extraction

Research notes (Oct 2026): what other methods work for extracting function parameters
from user input, and how they map onto this project. Companion to `reports/RESULTS.md`.

## A. Data-centric methods (strongest lever, per literature)

**1. Teacher distillation + auto-annotation** — Rana et al., COLING 2025 (arXiv 2411.18980),
"Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems".
Enterprise conversational slot filling: a big LLM auto-annotates dialogues (with *slot
induction* = discovering which slots matter from the data), then knowledge is distilled into
a small model. +26 points absolute F1 over vanilla LLM prompting; near-real-time inference.
Takeaway: annotation quality, not student size, sets the small model's ceiling.

**2. Synthetic data with validation loops** — Data Turnstile (arXiv 2607.29250);
PluginEval (2608.08700); APIGen / ToolACE (verified-data generators, known from earlier
planning round). Pattern: stepwise constrained generation + deterministic validators +
execution feedback in a closed loop until coverage converges. Data Turnstile: Qwen3-0.6B
trained on their synthetic data reaches 75.9% BFCL (base: 67.4% with thinking), closing on
models 3–7× larger; on τ²-bench a Turnstile-trained 1.7B beats Qwen2.5-32B (19× larger).
For SLMs, data quality is THE bottleneck.

**3. Function masking / negative training** — Hammer (arXiv 2410.04587), on-device
function-calling family. Training data augmented with irrelevant-function examples so the
model learns "none of these apply" instead of being misled by plausible function names;
Hammer-7B was 2nd only to GPT-4-class models at the time. Analogue here: adversarial
near-miss schemas + all-null rows to sharpen null discipline (our ~128 hallucinated slots).

**4. Cross-lingual transfer** — Cambridge 2025 (Persian SF: translated training data,
cross-lingual embeddings, multilingual encoders); Amazon Science multilingual voice agents;
code-switching augmentation (mixing languages inside an utterance) for zero-shot transfer.
Cheapest effective trick for our pl/es/fr gap: code-switched + translated augmentation.

## B. Inference-time / representation methods

**5. Constrained decoding (grammar masking)** — "Flexible and Efficient Grammar-Constrained
Decoding" (ICML 2025); parser-stack GCD (arXiv 2026); tools: llama.cpp GBNF grammars,
xgrammar, outlines. Token masks enforce the JSON schema → output guaranteed schema-valid,
no validate-retry loop; ICML paper shows how to keep the mask construction cheap.
For us: parse rate is already 1.0, but forcing enum vocabularies (routingMode, travel modes)
and key names could remove residual enum-invalid errors at zero training cost.

**6. Output-format choice matters** — Ravikumar et al., ACL 2026, "How Output Formats Skew
LLM Performance on Information Extraction". Fine-tuned LLMs perform differently depending
on output format (JSON vs line-per-slot vs span lists). Our compact-JSON choice was never
ablated against alternatives — cheap experiment.

**7. Schema representation beats scale** — "From Fixed Keys to Readable Schemas: Small
Language Models for Vehicle Agent Function Calls" (arXiv 2609.09476, NeurIPS 2026
SLM-Agents workshop; in-vehicle assistants, 270M–1.7B). Functional Tokens (fixed per-function
vocab) vs Schema-in-Prompt (SIP): on SEEN functions a 270M matches a 1.7B; only SIP
generalizes to unseen functions and refuses out-of-scope requests properly. Independently
replicates two of our findings: scale doesn't help on seen functions; schema-in-prompt is
the right representation.

**8. Decomposition into an instruction-follower** — IFFC, "Small Reasoning Models are
Instruction Followers in Function Calling" (arXiv 2608.22472). Decoupling function-calling
logic into a small model operating in plain instruction-following mode beats native
tool-call format (NFC) and prompt-based FC; robust under aggressive quantization. This is
essentially our architecture — independent validation of the paradigm for edge use.

**9. Edge-cloud cascade with learned router** — carbon-aware routing (arXiv 2609.13559).
A lightweight k-NN predictor in a semantic-lexical embedding space estimates per-query
accuracy/latency/power per tier and routes edge-vs-cloud; matches cloud accuracy at 4×
lower carbon. Same shape as our detector→cascade plan; supports per-row routing.

**10. Pre-inference candidate routing** — AgentWeave (arXiv 2608.23078): shrink the visible
tool set before the LLM sees it (+12.5 pts on BFCL V4 hard subset, 70% fewer tools exposed).
Not needed here (function is pre-selected upstream) but relevant if the scope grows to
multi-tool selection.

**11. Compact specialist models** — FunctionGemma-270M + MobileActionsExtended (arXiv
2609.25373, EMNLP 2026): 270M model, TRL SFT completion-only loss, schema-validated
synthetic data → end-to-end accuracy 29.3% → 76.5% on mobile actions. Evidence that
sub-1B extraction models are viable when the domain is bounded.

## C. Classical companions (still competitive)

**12. BERT+CRF span taggers** (joint intent+slot filling literature — Louvan & Binder 2020
survey; "CRF vs BERT for NER" 2025). Tiny, fast, strong on fixed closed schema sets. A
per-function specialist ensemble is a legitimate edge-cheap alternative, or an *auxiliary
signal*: LLM-vs-tagger agreement = free confidence feature to stack onto our detector.

**13. Retrieval-based filling** (DPR/RAG zero-shot slot filling, arXiv 2104.08610; LEONA
2021). Retrieve candidate values from a KB, then select. Standard fix for canonicalizing
place names — on our side that's the geocoder's job downstream, worth remembering if
canonical forms (exonyms: Varsovie/Warszawa) keep hurting multilingual F1.

**14. Speech-aware slot filling** — Whisper+LLM pipeline (OpenReview 2024); Spoken Function
Calling for large audio models (arXiv 2608.05126, ACM MM 2026). Car input is voice: ASR
errors on foreign place names are a real future failure mode. Mitigations: ASR-noise
augmentation of training texts, or joint audio post-training.

**15. SLOT benchmark** (Salesforce 2025, from earlier research round): LLMs can do
schema-grounded slot filling from parametric knowledge without fine-tuning — useful
zero-shot baseline framing; our zeroshot F1 ~0.1 vs FT 0.75 shows how much the honest-null
+ exact-span discipline must be taught.

## D. Evaluation methodology

**16. Fine-grained error attribution** — PluginEval (2608.08700): classify failures as
missed calls / spurious calls / parameter errors via an execution-anchored judge. Mirrors
our FP/FN breakdown + honest-null analysis; their closed loop (generation → validation →
coverage check → regenerate) is also a data-generation recipe.

## Mapping onto this project (ranked by expected value)

| Priority | Method | Why |
|---|---|---|
| 1 | Teacher distillation targeting pl/es/fr (local big model or sibling fcft models as teacher) | Attacks our biggest gap (language) with the biggest literature effect size (+26 F1) |
| 2 | Schema-constrained synthetic data + validators, closed loop (Data-Turnstile-style) | SLMs are data-bound; we already have validator code from the detector work |
| 3 | Enum/key constrained decoding at deployment (llama.cpp GBNF / xgrammar) | Free, removes residual enum errors, edge-friendly |
| 4 | Hammer-style negative / all-null adversarial rows | Directly targets our 128 hallucinated slots |
| 5 | Output-format ablation (compact JSON vs slot-per-line vs spans) | Cheap experiment; ACL 2026 says format skews IE performance |
| 6 | BERT-tagger agreement signal | Independent view to complement the logprob combiner (AUC 0.77) |

Cross-cutting validation of our existing results: the vehicle-domain NeurIPS paper
(2609.09476) independently found scale doesn't help on seen functions and SIP is the right
representation; IFFC (2608.22472) confirms the decomposed instruction-following design and
its robustness under quantization.
