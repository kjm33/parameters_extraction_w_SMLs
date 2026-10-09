#!/bin/zsh
# Resume run_all: skip training if best adapter exists; always continue on error.
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

for c in qwen2.5-0.5b qwen3-0.6b llama-3.2-1b gemma-3-1b; do
  if [[ ! -f train/runs/$c/best/adapter_model.safetensors ]]; then
    echo "=== TRAIN $c $(date)"
    $PY train/train_sft.py --candidate $c --gpu 0
    echo "TRAIN_EXIT=$? $(date)"
  else
    echo "=== SKIP TRAIN $c (best exists) $(date)"
  fi
  echo "=== EVAL zeroshot $c $(date)"
  $PY eval/eval_extract.py --candidate $c --gpu 0
  echo "ZS_EXIT=$? $(date)"
  echo "=== EVAL ft $c $(date)"
  $PY eval/eval_extract.py --candidate $c --adapter train/runs/$c/best --gpu 0
  echo "FT_EXIT=$? $(date)"
done
echo "=== ALL DONE $(date)"
