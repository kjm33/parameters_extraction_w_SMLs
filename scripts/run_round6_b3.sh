#!/bin/zsh
# Round 6 phase 2 (B3): tagger retrains + evals, sequential on one GPU.
# bert-fixd = mBERT + distill rows; xlmr-fix = XLM-R base; xlmr-fixd = XLM-R + distill.
# Tags data unchanged from round 5 (B1 audit: code suffixes absent from utterances ->
# no span-level rule possible; recovery lives in eval/canonicalize.py post-proc).
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
GPU=${1:-1}
mkdir -p logs
echo "=== round6 B3 start $(date) gpu=$GPU"

COREAUG="data/core_train_tags.jsonl,data/aug_train_tags.jsonl"
DIST="data/distill_train_tags.jsonl"
VAL="data/core_val_tags.jsonl,data/aug_val_tags.jsonl"

run_train () {  # name base trainfiles
  local name=$1 base=$2 trainfiles=$3
  if [[ ! -f train/runs/$name/best.pt ]]; then
    echo "=== TRAIN $name $(date)"
    $PY train/train_tagger.py --base "$base" --train "$trainfiles" --val $VAL \
        --out train/runs/$name --gpu $GPU 2>&1 | tee logs/$name.log
    echo "TRAIN_EXIT_$name=$? $(date)"
  else
    echo "=== SKIP TRAIN $name (best.pt exists) $(date)"
  fi
  echo "=== EVAL $name $(date)"
  $PY eval/eval_tagger.py --ckpt train/runs/$name/best.pt --gpu $GPU --dump \
      --tag $name-ft 2>&1 | tee logs/$name-eval.log
  echo "EVAL_EXIT_$name=$? $(date)"
}

run_train bert-tagger-fixd Babelscape/wikineural-multilingual-ner "$COREAUG,$DIST"
run_train xlmr-fix xlm-roberta-base "$COREAUG"
run_train xlmr-fixd xlm-roberta-base "$COREAUG,$DIST"
echo "=== round6 B3 DONE $(date)"
