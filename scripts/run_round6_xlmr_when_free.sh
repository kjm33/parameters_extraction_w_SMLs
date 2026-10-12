#!/bin/zsh
# Round 6 B3.3/B3.4: wait for a GPU with >=10GB free, then train+eval xlmr-fix and
# xlmr-fixd sequentially (sibling OCR jobs currently occupy ~21GB/GPU).
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

GPU=""
while true; do
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if (( used <= 14000 )); then GPU=$g; break; fi
  done
  [[ -n $GPU ]] && break
  sleep 60
done
echo "=== GPU $GPU has room (used<=14GB) at $(date) -> XLM-R runs"

COREAUG="data/core_train_tags.jsonl,data/aug_train_tags.jsonl"
DIST="data/distill_train_tags.jsonl"
VAL="data/core_val_tags.jsonl,data/aug_val_tags.jsonl"

for spec in "xlmr-fix:$COREAUG" "xlmr-fixd:$COREAUG,$DIST"; do
  name=${spec%%:*}; files=${spec#*:}
  if [[ ! -f train/runs/$name/best.pt ]]; then
    echo "=== TRAIN $name $(date)"
    $PY train/train_tagger.py --base xlm-roberta-base --train "$files" --val $VAL \
        --out train/runs/$name --gpu $GPU 2>&1 | tee logs/$name.log
    echo "TRAIN_EXIT_$name=$? $(date)"
  else
    echo "=== SKIP TRAIN $name (best.pt exists) $(date)"
  fi
  echo "=== EVAL $name $(date)"
  $PY eval/eval_tagger.py --ckpt train/runs/$name/best.pt --gpu $GPU --dump \
      --tag $name-ft 2>&1 | tee logs/$name-eval.log
  echo "EVAL_EXIT_$name=$? $(date)"
done
echo "=== XLMR DONE $(date)"
