# PLAN — slot_filling: Tiny-LLM Parameter Extraction for Car Navigation

Status: approved (2026-10-07). Decisions confirmed with user:
1. Non-extractable slots → **null them out** (honest extraction; downstream fills derived slots).
2. Models: user gave Qwen2.5-0.5B / Llama-3.2-1B as *examples*, asked for other small-model
   suggestions → 4-candidate comparison table below.
3. Training: **LoRA first + full-FT ablation**.
4. Augmentation from `../instruction_extraction`: **heuristic mapping** (no teacher API calls).

## Goal

Train and compare small (≤1B) local LLMs for **slot filling**: given a pair of
(user input = command to car navigation, selected function to execute + its expected
parameters), parse the input and extract the values of required parameters.

Motivation: sibling `function_calling_edge_fine_tuning` (fcft) trains 3–4B models for full
function calling (arg_acc ~0.37–0.41 on 643-row gold). Decomposing the task — external
selector picks the function, a dedicated ≤1B model fills slots — should push parameter
accuracy much higher at lower cost.

Online research summary:
- Community: fine-tuned Qwen-0.5B-class models reliably handle focused tool/JSON extraction
  (e.g. Reddit LocalLLaMA: 11 tools, 5 epochs, Qwen 1.5 0.5B).
- SLOT (arXiv 2025): structured-JSON output transformer line of work; confirms schema-in-
  prompt + strict JSON out is the right shape for extraction tasks.
- Verified-data lesson (APIGen/ToolACE, already embodied in fcft): quality over quantity —
  our core data is already dual-judge + execution verified.
- Llama-3.2-1B / Qwen2.5-0.5B fine-tuning for FC is well-trodden (Kaitchup, Unsloth,
  Databricks TRL guides); LoRA r=32 is standard for this scale; full-FT of 0.5B fits a 3090.

## Task definition

- Input: system prompt (short, English) + selected function's JSON schema (name,
  description, parameters w/ types+enums, required[]) + delimiter + user utterance.
- Output: compact JSON — one key per schema param (canonical schema key order):
  - value if stated in utterance (case/diacritic-insensitive normalized substring match),
  - `lang` param: inferred from utterance language (learnable signal, kept by design),
  - `null` otherwise (coords / limit / radius-from-context etc. → null; downstream fills).
- 5 languages (en/de/pl/fr/es), single-turn extraction rows.

## Data sources

### Core — fcft verified dialogs
`../function_calling_edge_fine_tuning/data/final/{train,val}.jsonl`
(5,102 / 270 rows; 15,017 tool calls in train; top fns: here_geocode 6.2k, here_directions
2.9k, nominatim_geocode 1.6k, tomtom-geocode 1.0k, here_search_places 683, tomtom-routing
571, osrm_route 497, …).

- `src/build_extraction.py`: walk messages; per assistant `tool_call` emit one row:
  (utterance = last user message before the call, function schema from `tools[]` by name,
  args = parsed `tool_call.arguments` JSON string).
- Extractability rule per param: normalized-substring match against utterance → keep value;
  `lang` == dialog lang → keep (inference); everything else (limit, lat/lng, radius, …) → null.
- Multi-call turns → one row per call (same utterance, different function).
- Skip `no_tool` rows (no function); `error_recovery` rows treated same as tool_use.
- Dedup (utterance, function); stratified split by lang × function-family.
- Yield ≈ 12–15k train / ~700 val.
- Held-out test: derive from fcft `data/test_gold.jsonl` (execution-verified args).

### Augmentation — instruction_extraction (heuristic, free)
`../instruction_extraction/data/`:

- `instructions_v3.jsonl` (21,959): rows already have `origin/destination/constraints/
  language` fields → synthesize rows for a `here_directions`-style schema:
  - origin/destination as text slots,
  - `avoid_features` via per-language keyword lexicon (toll: płatn/maut/péage/peaje…,
    ferry: prom/fähre/ferry/transbordador…, motorway: autopista/autobahn/Autobahn/motorway…),
  - geocode rows (`q` = single location) for texts with only one location field.
- `dialogues.jsonl` (2,029): driver turns via sibling's `FROMTO_RE` (src/elicitation.py) OD
  extraction → route/geocode rows.
- Cap augmentation to ≤40% of final mix; dedup vs fcft utterances (shingle Jaccard);
  manual spot-check of 50 rows (documented in reports).

## Rendering (`src/render.py`)

- Chat format per model via `tokenizer.apply_chat_template`; deterministic key order,
  `ensure_ascii=False` (Polish chars stay literal).
- System (English, all langs): extract-params-for-function, output-only-JSON, null-when-
  not-stated.
- Completion-only loss: tokenize prompt/completion separately, labels masked on prompt
  (fcft's pre-masked-labels + custom-collator pattern — avoids TRL assistant_only_loss
  template issues on Llama).

## Training (`train/train_sft.py` adapted from fcft)

Reuse fcft `train/venv` (torch + transformers + trl + peft) and its verified gotchas:
single-GPU pin via CUDA_VISIBLE_DEVICES, `processing_class=tokenizer`, NO packing,
pre-masked labels + custom collator, auto-resume, bf16.

- Seq-len 2048 (rows short), LoRA r=32 / α=64 / dropout 0.05, lr 1e-4, 3 epochs,
  early-stop on val loss (patience 3), warmup 3%, cosine.
- Effective batch ~64 rows (micro-bs 4–8 + grad-accum).
- Shared HF cache `/var/tmp/fcft_hf_cache`.
- GPU 1 free (GPU 0 busy with sibling eval server) — schedule accordingly.

### Candidates

| key | model_id | note |
|---|---|---|
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B-Instruct | open, proven FT tool-calling |
| qwen3-0.6b   | Qwen/Qwen3-0.6B            | newer, stronger; disable thinking |
| llama-3.2-1b | unsloth/Llama-3.2-1B-Instruct | multilingual; meta gated → unsloth mirror (fcft precedent) |
| gemma-3-1b   | unsloth/gemma-3-1b-it      | weak JSON prior; included for completeness |

- Full-FT ablation for 0.5B/0.6B winners (~8GB VRAM fits a 3090); 1B full-FT only if LoRA
  disappoints, via `adamw_bnb_8bit`.

## Eval (`eval/eval_extract.py`)

- transformers batch-generate, greedy/temp-0 (no vLLM needed at this scale/n).
- Zero-shot baselines (same prompt) + FT adapters, same held-out test for all.
- Metrics:
  - JSON parse rate
  - per-slot exact match (normalized: case/diacritics/number format)
  - slot-level P/R/F1 over non-null gold
  - null-slot accuracy (does the model correctly emit null)
  - full-row exact match (all slots right)
  - breakdowns per lang × function
  - tokens/sec + output length (edge-deployment relevance)
- Optional deployment bonus: export JSON-schema → GBNF grammar / Outlines for constrained
  decoding.

## Reports

`reports/RESULTS.md` — candidates × {zeroshot, lora, full-ft} metrics tables, winner
recommendation, merged-adapter export (+ GGUF note).

## Estimated effort

Data build ~0.5d; 4 trainings ≈ 30–60 min each on a 3090; eval fast. ~1–2 days total runs.

## Risks

- gemma-3-1b may fail JSON (acceptable — it's a comparison).
- Heuristic augmentation noise (keyword lexicon misses) — capped share + 50-row spot-check.
- `lang` slot is inferred rather than extracted (by design; noted in metrics).
- Unsloth mirrors are public (fcft already uses unsloth/Llama-3.2-3B-Instruct) — 1B fine.
