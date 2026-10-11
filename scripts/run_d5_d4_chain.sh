#!/bin/zsh
# Chain: wait for a free GPU, then d5 (syn without pl) then d4 (no syn) train+eval.
# d5: core+aug+distill_dir+distill2+syn_nopl  (is syn-pl the pl-geocode culprit?)
# d4: core+aug+distill_dir+distill2           (no syn at all — diagnostic)
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

pick_gpu() {
  local free_mb
  for g in 1 0; do
    free_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if (( free_mb < 6000 )); then echo $g; return 0; fi
  done
  return 1
}

echo "=== waiting for free GPU $(date)"
GPU=""
while true; do
  GPU=$(pick_gpu) && break
  sleep 60
done
echo "=== picked GPU $GPU $(date)"

run_variant() {
  local NAME=$1 TAG=$2 TRAIN_FILES=$3 VAL_FILES=$4 OUT=$5
  if [[ ! -f $OUT/best/adapter_model.safetensors ]]; then
    $PY train/train_sft.py --candidate llama-3.2-1b --gpu $GPU \
        --train $TRAIN_FILES --val $VAL_FILES --out-dir $OUT
    echo "${NAME}_TRAIN_EXIT=$? $(date)"
  else
    echo "=== SKIP $NAME TRAIN (best exists) $(date)"
  fi
  $PY eval/eval_extract.py --candidate llama-3.2-1b --adapter $OUT/best \
      --gpu $GPU --tag ${TAG}-ft --dump
  echo "${NAME}_FT_EXIT=$? $(date)"
  $PY eval/eval_constrained.py --candidate llama-3.2-1b --adapter $OUT/best \
      --gpu $GPU --tag ${TAG}-ft-g --dump
  echo "${NAME}_CONSTRAINED_EXIT=$? $(date)"
}

run_variant d5 llama-3.2-1b-d5 \
  "data/core_train.jsonl,data/aug_train.jsonl,data/distill_dir_train.jsonl,data/distill2_train.jsonl,data/syn_train_nopl.jsonl" \
  "data/core_val.jsonl,data/aug_val.jsonl,data/distill_dir_val.jsonl,data/distill2_val.jsonl,data/syn_val_nopl.jsonl" \
  train/runs/llama-3.2-1b-d5

run_variant d4 llama-3.2-1b-d4 \
  "data/core_train.jsonl,data/aug_train.jsonl,data/distill_dir_train.jsonl,data/distill2_train.jsonl" \
  "data/core_val.jsonl,data/aug_val.jsonl,data/distill_dir_val.jsonl,data/distill2_val.jsonl" \
  train/runs/llama-3.2-1b-d4

echo "=== D5+D4 CHAIN DONE $(date)"
