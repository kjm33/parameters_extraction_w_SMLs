#!/bin/zsh
# Wait for a GPU with <6GB used, then train llama-3.2-3b LoRA + evals on it.
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

GPU=""
while true; do
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if (( used < 6000 )); then GPU=$g; break; fi
  done
  [[ -n $GPU ]] && break
  sleep 60
done
echo "=== GPU $GPU free (waited since $(date)) -> TRAIN llama-3.2-3b $(date)"

if [[ ! -f train/runs/llama-3.2-3b/best/adapter_model.safetensors ]]; then
  $PY train/train_sft.py --candidate llama-3.2-3b --gpu $GPU
  echo "TRAIN_EXIT=$? $(date)"
else
  echo "=== SKIP TRAIN llama-3.2-3b (best exists) $(date)"
fi
$PY eval/eval_extract.py --candidate llama-3.2-3b --gpu $GPU
echo "ZS_EXIT=$? $(date)"
$PY eval/eval_extract.py --candidate llama-3.2-3b --adapter train/runs/llama-3.2-3b/best --gpu $GPU
echo "FT_EXIT=$? $(date)"
echo "=== LLAMA3B DONE $(date)"
