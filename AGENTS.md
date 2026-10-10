# AGENTS.md — Guide for AI agents working in this repo

Train small local LLMs (edge-viable after Q4 quantization) for **slot filling / parameter extraction** for an in-car
navigation assistant: given (user utterance, selected function schema) → JSON with one key
per parameter, value from the utterance or `null`. 5 languages (en/de/pl/fr/es).
Plan + decisions: `plans/PLAN.md`. Final comparison: `reports/RESULTS.md`.

## Layout

- `src/build_extraction.py` — core dataset from fcft verified dialogs + gold
  (`data/core_{train,val,test}.jsonl`); honest-null target policy, enum keyword lexicon,
  comma-segment rewrite ("Lyon, France" → "Lyon"), prefix-inflection token matching
- `src/build_augment.py` — heuristic augmentation from `../instruction_extraction`
  (`data/aug_{train,val}.jsonl`); shingle dedup vs core; lang-balanced (en boosted)
- `src/render.py` — chat rendering + prompt-masked labels (string-level split; matches
  inference tokenization). SYSTEM_PROMPT lives here.
- `train/train_sft.py` — transformers+peft LoRA/full SFT, single pinned GPU
- `train/configs/<candidate>.json` — model_id + template_kwargs
- `eval/eval_extract.py` — zeroshot/FT eval: parse rate, slot P/R/F1, null accuracy, row EM
  (`--dump` writes per-row preds; `--model-dir` evaluates full-FT checkpoints)
- `eval/analyze_errors.py` — FP/FN slot breakdown + samples from `*_preds.jsonl` dumps
- `eval/detect_uncertainty.py` — per-row failure detection: logprobs + self-consistency +
  validators vs actual errors (AUC/coverage tables → `eval/results/<tag>*.json` + `_rows.jsonl`);
  extended: k=5 samples, per-slot logprobs (teacher-forced re-forward), per-slot semantic
  entropy, hidden states every 4th layer → `<tag>_features.jsonl` + `<tag>_hidden.npz`
- `eval/train_detector.py` — fits detectors on a labeled error set (`<tag>` from h90 holdout):
  logistic combiner over scalar features, hidden-state probe (layer+λ by CV), stack, conformal
  abstention; evaluates once on test tags → `eval/results/detector_<train-tag>.json`
- `eval/ensemble_vote.py` — majority vote of the h90 trio on test (voted EM vs members,
  disagreement AUC) → `eval/results/ensemble_trio{,_rows}.jsonl`
- `eval/jev_client.py` — OpenRouter Jev decision-model client (alpha decisions endpoint,
  retry/backoff, thread pool, mock mode; OPENROUTER_API_KEY from env or `../.env`)
- `eval/jev_enum_probe.py` — Jev as direct enum-slot extractor (noul presence + choice /
  per-value noul) vs local FT on same rows → `eval/results/jev_enum_*`
- `eval/jev_verify.py` — Jev grounding verification of local preds as failure detector
  (row/slot AUC + coverage vs logprob baseline) → `eval/results/jev_verify_*`
- `src/split_holdout.py` — core_train → `core_train_h90.jsonl` + `core_train_holdout.jsonl`
  (dialog-id-grouped, lang×fn stratified; holdout = labeled-error source for detectors)
- `train/runs/<c>-h90/` — LoRA retrains on h90+aug (detector ensemble members)
- `train/runs/llama-3.2-1b/merged/` — WINNER export (base+LoRA merged, see reports/RESULTS.md)
- `data/` — datasets + `*_stats.json`; `logs/` — training logs (tee'd)

## Environments

- Interpreter: `../function_calling_edge_fine_tuning/train/venv/bin/python`
  (torch 2.14 + transformers 5.18 + peft 0.21 + trl 1.14, datasets available)
- No API keys needed (no teacher calls anywhere).
- HF cache: `/var/tmp/fcft_hf_cache` (set `HF_HOME`); adapters live under `train/runs/`.

## Key commands

```bash
# rebuild data
python3 src/build_extraction.py
python3 src/build_augment.py        # needs core_train.jsonl first

# train (GPU 1 free by default; GPU 0 often runs fcft eval servers)
../function_calling_edge_fine_tuning/train/venv/bin/python train/train_sft.py \
  --candidate qwen2.5-0.5b --gpu 1            # add --full for full-FT ablation

# eval (zeroshot baseline / fine-tuned adapter)
../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_extract.py \
  --candidate qwen2.5-0.5b [--adapter train/runs/qwen2.5-0.5b/best] --gpu 1

# uncertainty features (logprobs/self-consistency/validators/hidden states)
../function_calling_edge_fine_tuning/train/venv/bin/python eval/detect_uncertainty.py \
  --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-h90/best \
  --files data/core_test.jsonl --gpu 1          # or --model-dir <merged dir>

# learned detectors (combiner/probe/stack/conformal) — train on labeled-error tag
python3 eval/train_detector.py --train-tag llama-3.2-1b-h90-holdout \
  --test-tags llama-3.2-1b-h90-test,llama-1b-merged-test

# ensemble majority vote of the h90 trio
python3 eval/ensemble_vote.py
```

## Gotchas

- `apply_chat_template` prefix property FAILS at token level (BPE merges across the
  assistant-marker boundary) — render.py splits at STRING level on purpose; this matches
  generate()-time prompt tokenization. Don't "fix" it back.
- Gold rows (`test_gold_merged.jsonl`) use `reference.tool_calls[].args` (not `arguments`).
- Trainer: `dtype=` (not `torch_dtype=`) in transformers 5.x; `processing_class=tokenizer`.
- Targets must stay honest: coords/limit/radius → null. Don't copy teacher args verbatim.
- v3 instruction rows whose text starts with `{`/`[` are structured artifacts — filtered.
- train_sft.py exits with code 120 (stdout Tee flush fails at interpreter shutdown) AFTER
  a successful train+save — benign; orchestration scripts must not `set -e` on it.
- Sibling fcft repo jobs may occupy either GPU; use scripts/run_llama3b_when_free.sh-style
  watchers (poll nvidia-smi, launch when <6GB used) instead of killing their processes.
