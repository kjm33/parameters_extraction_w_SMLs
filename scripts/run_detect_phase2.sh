#!/bin/zsh
# Phase 2 of detector pipeline: extract uncertainty features with the NEW
# detect_uncertainty.py for the h90 trio (holdout + val + test) and the merged
# winner (test). Waits for all three h90 adapters, then a free GPU per run;
# retries runs whose output JSON is missing (sibling GPU collisions).
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

log() { builtin print -r -- "$*"; }

wait_gpu() {
  while true; do
    for g in 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if (( used < 8000 )); then print -r -- $g; return; fi
    done
    sleep 60
  done
}

wait_trio() {
  while true; do
    ok=1
    for c in llama-3.2-1b qwen2.5-0.5b qwen3-0.6b; do
      [[ -f train/runs/$c-h90/best/adapter_model.safetensors ]] || ok=0
    done
    (( ok )) && return 0
    sleep 120
  done
}

run_detect() { # args: outfile tag extra-args...
  local out=$1 tag=$2; shift 2
  local attempt=0
  while [[ ! -f $out ]]; do
    attempt=$((attempt + 1))
    if (( attempt > 8 )); then log "=== GIVE UP $tag after $attempt attempts"; return 1; fi
    GPU=$(wait_gpu)
    log "=== GPU $GPU -> DETECT $tag (attempt $attempt) $(date)"
    $PY eval/detect_uncertainty.py "$@" --gpu $GPU --tag $tag
    log "DETECT_EXIT_${tag}=$? $(date)"
    [[ -f $out ]] || sleep 60
  done
  log "=== DONE $tag $(date)"
}

log "=== phase2: waiting for trio adapters $(date)"
wait_trio
log "=== phase2: trio ready $(date)"

for c in llama-3.2-1b qwen2.5-0.5b qwen3-0.6b; do
  run_detect eval/results/$c-h90-holdout.json $c-h90-holdout \
    --candidate $c --adapter train/runs/$c-h90/best --files data/core_train_holdout.jsonl --bs 8
  run_detect eval/results/$c-h90-val.json $c-h90-val \
    --candidate $c --adapter train/runs/$c-h90/best --files data/core_val.jsonl --bs 8
  run_detect eval/results/$c-h90-test.json $c-h90-test \
    --candidate $c --adapter train/runs/$c-h90/best --files data/core_test.jsonl --bs 8
done

run_detect eval/results/llama-1b-merged-test.json llama-1b-merged-test \
  --model-dir train/runs/llama-3.2-1b/merged --files data/core_test.jsonl --bs 8

log "=== PHASE 2 ALL DONE $(date)"
