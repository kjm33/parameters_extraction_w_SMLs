#!/bin/zsh
# Hammer-style negative-training retrain (plan task 4): core+aug+neg (~6% negatives),
# then eval with dump (FP-hallucination acceptance measurement). No distill/syn:
# isolate the negatives effect vs the base run (core+aug).
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

TRAIN_FILES="data/core_train.jsonl,data/aug_train.jsonl,data/neg_train.jsonl"
VAL_FILES="data/core_val.jsonl,data/aug_val.jsonl,data/neg_val.jsonl"
OUT=train/runs/llama-3.2-1b-neg

# wait for a free GPU (prefer GPU1), like run_llama3b_when_free.sh
GPU=""
while true; do
  for g in 1 0; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [[ $used -lt 6000 ]]; then GPU=$g; break 2; fi
  done
  echo "=== waiting for free GPU $(date)"
  sleep 60
done

echo "=== TRAIN llama-3.2-1b-neg on GPU $GPU $(date)"
if [[ ! -f $OUT/best/adapter_model.safetensors ]]; then
  $PY train/train_sft.py --candidate llama-3.2-1b --gpu $GPU \
      --train $TRAIN_FILES --val $VAL_FILES --out-dir $OUT
  echo "TRAIN_EXIT=$? (120 benign) $(date)"
else
  echo "=== SKIP TRAIN (best exists) $(date)"
fi

$PY eval/eval_extract.py --candidate llama-3.2-1b --adapter $OUT/best \
    --gpu $GPU --tag llama-3.2-1b-neg-ft --dump
echo "FT_EXIT=$? $(date)"
echo "=== LLAMA-1B-NEG DONE $(date)"
