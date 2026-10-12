#!/bin/zsh
# Round 7 retrains on the lang-dropped v2 benchmark (core+aug / +distill-norm).
# Watcher: sibling ocr-judges-yiddish jobs may occupy both GPUs; poll and launch
# when headroom appears (tagger needs ~5GB -> <=18000MiB used; SFT needs ~12GB
# -> <=10000MiB used). Exits when both runs have their eval result JSON.
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
mkdir -p logs
echo "=== round7 watcher start $(date)"

used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

TAG_DONE=eval/results/bert-tagger-fixd-v2-ft.json
SFT_DONE=eval/results/llama-3.2-1b-v2-ft.json

while [[ ! -f $TAG_DONE || ! -f $SFT_DONE ]]; do
  for g in 0 1; do
    u=$(used $g)
    if [[ ! -f $TAG_DONE && $u -le 18000 ]]; then
      echo "=== tagger launch gpu=$g used=${u}MiB $(date)"
      if [[ ! -f train/runs/bert-tagger-fixd-v2/best.pt ]]; then
        $PY train/train_tagger.py --base Babelscape/wikineural-multilingual-ner \
          --train data/core_train_tags.jsonl,data/aug_train_tags.jsonl,data/distill_train_tags.jsonl \
          --val data/core_val_tags.jsonl,data/aug_val_tags.jsonl \
          --out train/runs/bert-tagger-fixd-v2 --gpu $g 2>&1 | tee logs/bert-tagger-fixd-v2.log
      fi
      echo "TAGGER_TRAIN_EXIT=$? $(date)"
      $PY eval/eval_tagger.py --ckpt train/runs/bert-tagger-fixd-v2/best.pt --gpu $g --dump \
        --tag bert-tagger-fixd-v2-ft 2>&1 | tee logs/bert-tagger-fixd-v2-eval.log
      echo "TAGGER_EVAL_EXIT=$? $(date)"
    fi
    if [[ ! -f $SFT_DONE && $u -le 10000 ]]; then
      echo "=== sft launch gpu=$g used=${u}MiB $(date)"
      $PY train/train_sft.py --candidate llama-3.2-1b \
        --out-dir train/runs/llama-3.2-1b-v2 --gpu $g 2>&1 | tee logs/llama-3.2-1b-v2.log
      echo "SFT_TRAIN_EXIT=$? (120 = benign tee flush) $(date)"
      $PY eval/eval_extract.py --candidate llama-3.2-1b \
        --adapter train/runs/llama-3.2-1b-v2/best --gpu $g --dump \
        --tag llama-3.2-1b-v2-ft 2>&1 | tee logs/llama-3.2-1b-v2-eval.log
      echo "SFT_EVAL_EXIT=$? $(date)"
    fi
  done
  [[ -f $TAG_DONE && -f $SFT_DONE ]] && break
  sleep 60
done
echo "=== round7 watcher DONE $(date)"
