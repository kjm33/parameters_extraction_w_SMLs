#!/usr/bin/env bash
# Wait for llama-3.2-1b-tags training (pid arg) to exit, then run eval_tags on core_test.
set -u
PID="$1"
VENV=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache

while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[watch] trainer pid $PID exited at $(date)"

if [ ! -f train/runs/llama-3.2-1b-tags/best/adapter_model.safetensors ]; then
  echo "[watch] ERROR: best adapter missing"; exit 1
fi

cd /home/kamil/projects/here/slot_filling
CUDA_VISIBLE_DEVICES=0 "$VENV" eval/eval_tags.py \
  --candidate llama-3.2-1b \
  --adapter train/runs/llama-3.2-1b-tags/best \
  --tag llama-3.2-1b-tags-ft \
  --dump --gpu 0
echo "[watch] eval done rc=$?"
