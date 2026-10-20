#!/bin/zsh
# Round 12 tagger lane: retrain edge tagger on v3 tags (two variants, sequential, one GPU).
#   bert-tagger-fixd-v3      = champion mix (core+aug v3 tags + distill v2 tags = 21,937 rows)
#   bert-tagger-fixd-v3-pure = core+aug v3 tags only (15,004 rows, policy-pure)
# v3 tags built by: python3 src/build_tags.py core_train_v3,core_val_v3,core_test_v3,aug_train_v3,aug_val_v3
# NOTE v3 join_sep=',' (data/tags_build_stats_v3.json, razor-thin 11547 vs 11540);
# evals must pass --stats data/tags_build_stats_v3.json (patched into eval_tagger.py).
# Ceiling context: v3 tags roundtrip rate 0.6949 (v2 was 0.9622) — intent-derived enum
# values (enum-unaligned class, 4,387) are not surface-recoverable by the tag decoder.
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
GPU=${1:-1}
mkdir -p logs
echo "=== round12 tagger-v3 start $(date) gpu=$GPU"

V3AUG="data/core_train_v3_tags.jsonl,data/aug_train_v3_tags.jsonl"
DIST="data/distill_train_tags.jsonl"
VAL3="data/core_val_v3_tags.jsonl,data/aug_val_v3_tags.jsonl"
TEST3="data/core_test_v3_tags.jsonl"

run_one () {  # name trainfiles
  local name=$1 trainfiles=$2
  if [[ ! -f train/runs/$name/best.pt ]]; then
    echo "=== TRAIN $name $(date)"
    $PY train/train_tagger.py --base Babelscape/wikineural-multilingual-ner \
      --train "$trainfiles" --val $VAL3 --out train/runs/$name --gpu $GPU 2>&1 | tee logs/$name.log
    echo "TRAIN_EXIT_$name=$? $(date)"
  fi
  echo "=== EVAL $name $(date)"
  $PY eval/eval_tagger.py --ckpt train/runs/$name/best.pt --files $TEST3 \
    --stats data/tags_build_stats_v3.json --gpu $GPU --dump --tag $name-ft 2>&1 | tee logs/$name-eval.log
  echo "EVAL_EXIT_$name=$? $(date)"
  python3 eval/score_no_lang.py --dump eval/results/$name-ft_preds.jsonl --canon \
    --train data/core_train_v3.jsonl,data/aug_train_v3.jsonl \
    --files data/core_test_v3.jsonl --out-tag no_lang_$name-ft 2>&1 | tee logs/$name-nolang.log
  echo "NOLANG_EXIT_$name=$? $(date)"
}

run_one bert-tagger-fixd-v3 "$V3AUG,$DIST"
run_one bert-tagger-fixd-v3-pure "$V3AUG"
echo "=== round12 tagger-v3 DONE $(date)"
