#!/bin/zsh
# Round 8: mmBERT tagger — mmbert-tagger-fixd-v2 train + eval + lang-free canon score.
# Backbone jhu-clsp/mmBERT-base; identical recipe/data/eval as bert-tagger-fixd-v2
# (core+aug+distill tags v2, 5 epochs, bs 16, lr 5e-5, seed 42). Decision rules in
# notes/2026-10-12-mmbert-tagger-round8.md (pre-registered).
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
GPU=${1:-1}
mkdir -p logs
echo "=== round8 mmbert start $(date) gpu=$GPU"

NAME=mmbert-tagger-fixd-v2
if [[ ! -f train/runs/$NAME/best.pt ]]; then
  $PY train/train_tagger.py --base jhu-clsp/mmBERT-base \
    --train data/core_train_tags.jsonl,data/aug_train_tags.jsonl,data/distill_train_tags.jsonl \
    --val data/core_val_tags.jsonl,data/aug_val_tags.jsonl \
    --out train/runs/$NAME --gpu $GPU 2>&1 | tee logs/$NAME.log
fi
echo "TRAIN_EXIT=$NAME=$? $(date)"

$PY eval/eval_tagger.py --ckpt train/runs/$NAME/best.pt --gpu $GPU --dump \
  --tag $NAME-ft 2>&1 | tee logs/$NAME-eval.log
echo "EVAL_EXIT=$NAME=$? $(date)"

$PY eval/eval_tagger.py --ckpt train/runs/$NAME/best.pt --cpu \
  --tag $NAME-ft 2>&1 | tee logs/$NAME-eval-cpu.log
echo "EVAL_CPU_EXIT=$NAME=$? $(date)"

python3 eval/score_no_lang.py --dump eval/results/$NAME-ft_preds.jsonl --canon \
  --out-tag no_lang_$NAME-ft 2>&1 | tee logs/$NAME-nolang.log
echo "NOLANG_EXIT=$NAME=$? $(date)"
echo "=== round8 mmbert DONE $(date)"
