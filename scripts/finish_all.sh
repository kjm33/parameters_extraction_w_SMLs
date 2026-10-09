#!/usr/bin/env zsh
# Finish remaining pipeline after the llama-3.2-3b watcher completes:
#   1. re-run FT evals with --dump (per-row predictions for error analysis)
#   2. full-FT ablation for qwen3-0.6b and qwen2.5-0.5b (lr 2e-5)
#   3. eval full-FT checkpoints
# No set -e: train_sft.py exits 120 (benign stdout-Tee flush at shutdown).
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
LOG=logs/run_all.log

log() { builtin print -r -- "$(date '+%a, %d %b %Y, %H:%M:%S %Z') $*" >> $LOG }

# --- wait for llama-3.2-3b watcher to finish (its last artifact) ---
log "finish_all: waiting for eval/results/llama-3.2-3b-ft.json"
while [[ ! -f eval/results/llama-3.2-3b-ft.json ]]; do sleep 120; done
log "finish_all: llama-3.2-3b results present"

# --- helper: wait for a GPU with <6GB used, echo its index ---
wait_free() {
  while true; do
    for g in 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      (( used < 6000 )) && { print -r -- $g; return }
    done
    sleep 60
  done
}

# --- 1. dump evals for all FT candidates ---
G=$(wait_free)
log "finish_all: dump evals on GPU $G"
for c in qwen2.5-0.5b qwen3-0.6b llama-3.2-1b gemma-3-1b llama-3.2-3b; do
  $PY eval/eval_extract.py --candidate $c --adapter train/runs/$c/best --gpu $G --dump
  log "DUMP_EXIT[$c]=$?"
done

# --- 2. full-FT ablation ---
for c in qwen3-0.6b qwen2.5-0.5b; do
  if [[ ! -f train/runs/${c}-full/best/config.json ]]; then
    G=$(wait_free)
    log "finish_all: full-FT $c on GPU $G"
    $PY train/train_sft.py --candidate $c --full --lr 2e-5 --gpu $G
    print "FULLFT_EXIT[$c]=$?"
  else
    log "finish_all: full-FT $c already done, skip"
  fi
  # --- 3. eval full-FT ---
  G=$(wait_free)
  $PY eval/eval_extract.py --candidate $c --model-dir train/runs/${c}-full/best \
      --tag ${c}-fullft --gpu $G --dump
  log "FULLFT_EVAL_EXIT[$c]=$?"
done

log "finish_all: ALL DONE"
