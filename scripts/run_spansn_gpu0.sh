#!/bin/zsh
# Round 9 manual lane: spansn (numbered variant) on GPU 0, parallel to the watcher's
# spans run on GPU 1. The placeholder eval/results/llama-3.2-1b-spansn-ft.json keeps
# the round-9 watcher from double-launching this variant (its DONE2 gate); the real
# eval result overwrites the placeholder at the end.
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
mkdir -p logs

echo '{"placeholder": true}' > eval/results/llama-3.2-1b-spansn-ft.json

if [[ ! -d train/runs/llama-3.2-1b-spansn/best ]]; then
  echo "=== train spansn gpu=0 $(date)"
  $PY train/train_sft.py --candidate llama-3.2-1b \
    --train data/core_train_spansn.jsonl,data/aug_train_spansn.jsonl \
    --val data/core_val_spansn.jsonl,data/aug_val_spansn.jsonl \
    --out-dir train/runs/llama-3.2-1b-spansn --gpu 0 2>&1 | tee logs/llama-3.2-1b-spansn.log
  echo "TRAIN_EXIT=$? (120 = benign tee flush) $(date)"
fi
if [[ -d train/runs/llama-3.2-1b-spansn/best ]]; then
  echo "=== eval spansn gpu=0 $(date)"
  $PY eval/eval_spans.py --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-spansn/best \
    --files data/core_test_spansn.jsonl --gpu 0 --dump 2>&1 | tee logs/llama-3.2-1b-spansn-eval.log
  echo "EVAL_EXIT=$? $(date)"
  python3 eval/score_no_lang.py --dump eval/results/llama-3.2-1b-spansn-ft_preds.jsonl --canon \
    2>&1 | tee logs/llama-3.2-1b-spansn-nolang.log
fi
echo "=== spansn manual lane DONE $(date)"
