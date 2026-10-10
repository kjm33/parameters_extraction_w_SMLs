#!/bin/zsh
# Retrain the detector-ensemble trio on the h90 split (core_train_h90 + aug).
# Waits for a free GPU (<6GB used) before EACH attempt; RETRIES until
# train/runs/<c>-h90/best/adapter_model.safetensors exists (sibling fcft jobs can
# balloon mid-run and OOM us — a failed attempt just re-queues).
# exit code 120 from train_sft.py (Tee flush at shutdown) is benign — no set -e.
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

log() { builtin print -r -- "$*"; }

wait_gpu() {
  while true; do
    for g in 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if (( used < 6000 )); then print -r -- $g; return; fi
    done
    sleep 60
  done
}

for c in llama-3.2-1b qwen2.5-0.5b qwen3-0.6b; do
  attempt=0
  while [[ ! -f train/runs/$c-h90/best/adapter_model.safetensors ]]; do
    attempt=$((attempt + 1))
    if (( attempt > 12 )); then
      log "=== GIVE UP $c-h90 after $attempt attempts $(date)"
      break
    fi
    GPU=$(wait_gpu)
    log "=== GPU $GPU free -> TRAIN $c-h90 (attempt $attempt) $(date)"
    $PY train/train_sft.py --candidate $c \
      --train data/core_train_h90.jsonl,data/aug_train.jsonl \
      --out-dir train/runs/$c-h90 --gpu $GPU
    log "TRAIN_EXIT_${c}=$? (attempt $attempt) $(date)"
    if [[ ! -f train/runs/$c-h90/best/adapter_model.safetensors ]]; then
      log "=== no best yet for $c-h90 -> will retry after GPU wait $(date)"
      sleep 120
    fi
  done
  if [[ -f train/runs/$c-h90/best/adapter_model.safetensors ]]; then
    log "=== DONE $c-h90 $(date)"
  fi
done
log "=== TRIO H90 ALL DONE $(date)"
