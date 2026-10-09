#!/bin/zsh
# Sequential: train all 4 candidates, then zeroshot + FT eval on core_test. Logs -> logs/.
set -e
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

for c in qwen2.5-0.5b qwen3-0.6b llama-3.2-1b gemma-3-1b; do
  echo "=== TRAIN $c $(date)"
  $PY train/train_sft.py --candidate $c --gpu 0
  echo "=== EVAL zeroshot $c $(date)"
  $PY eval/eval_extract.py --candidate $c --gpu 0
  echo "=== EVAL ft $c $(date)"
  $PY eval/eval_extract.py --candidate $c --adapter train/runs/$c/best --gpu 0
done
echo "=== ALL DONE $(date)"
