#!/bin/zsh
# Retrain llama-3.2-1b LoRA on core+aug+distill(directions only)+syn, then eval
# (unconstrained + grammar). Variant d2 after geocode-distill regression analysis.
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

TRAIN_FILES="data/core_train.jsonl,data/aug_train.jsonl,data/distill_dir_train.jsonl,data/syn_train.jsonl"
VAL_FILES="data/core_val.jsonl,data/aug_val.jsonl,data/distill_dir_val.jsonl,data/syn_val.jsonl"
OUT=train/runs/llama-3.2-1b-d2
GPU=${1:-1}

echo "=== TRAIN llama-3.2-1b-d2 on GPU $GPU $(date)"
if [[ ! -f $OUT/best/adapter_model.safetensors ]]; then
  $PY train/train_sft.py --candidate llama-3.2-1b --gpu $GPU \
      --train $TRAIN_FILES --val $VAL_FILES --out-dir $OUT
  echo "TRAIN_EXIT=$? $(date)"
else
  echo "=== SKIP TRAIN (best exists) $(date)"
fi

$PY eval/eval_extract.py --candidate llama-3.2-1b --adapter $OUT/best \
    --gpu $GPU --tag llama-3.2-1b-d2-ft --dump
echo "FT_EXIT=$? $(date)"
$PY eval/eval_constrained.py --candidate llama-3.2-1b --adapter $OUT/best \
    --gpu $GPU --tag llama-3.2-1b-d2-ft-g --dump
echo "CONSTRAINED_EXIT=$? $(date)"
echo "=== LLAMA-1B-D2 DONE $(date)"
