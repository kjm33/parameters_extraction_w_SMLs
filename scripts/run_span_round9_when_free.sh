#!/bin/zsh
# Round 9: span-list formulation — 2 retrains (plain + numbered) + evals + official metric.
# Watcher: GPU 0 runs the round-7 v2 SFT; poll both, launch when <=10000MiB used
# (SFT needs ~12GB). Sequential: spans first, then spansn. Exit code 120 from
# train_sft after a successful save is benign (tee flush) — gate on artifacts, not codes.
set -u
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export HF_HOME=/var/tmp/fcft_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
mkdir -p logs
echo "=== round9 watcher start $(date)"

used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }

DONE1=eval/results/llama-3.2-1b-spans-ft.json
DONE2=eval/results/llama-3.2-1b-spansn-ft.json

if [[ ! -f data/spans_build_stats.json ]]; then
  python3 src/build_spans.py 2>&1 | tee logs/build_spans.log
fi

run_variant () {
  local v=$1 g=$2
  local outdir=train/runs/llama-3.2-1b-${v}
  if [[ ! -d ${outdir}/best ]]; then
    echo "=== train ${v} gpu=${g} $(date)"
    $PY train/train_sft.py --candidate llama-3.2-1b \
      --train data/core_train_${v}.jsonl,data/aug_train_${v}.jsonl \
      --val data/core_val_${v}.jsonl,data/aug_val_${v}.jsonl \
      --out-dir ${outdir} --gpu ${g} 2>&1 | tee logs/llama-3.2-1b-${v}.log
    echo "TRAIN_EXIT=$? (120 = benign tee flush) $(date)"
  fi
  if [[ -d ${outdir}/best && ! -f eval/results/llama-3.2-1b-${v}-ft.json ]]; then
    echo "=== eval ${v} gpu=${g} $(date)"
    $PY eval/eval_spans.py --candidate llama-3.2-1b --adapter ${outdir}/best \
      --files data/core_test_${v}.jsonl --gpu ${g} --dump 2>&1 | tee logs/llama-3.2-1b-${v}-eval.log
    echo "EVAL_EXIT=$? $(date)"
  fi
  if [[ -f eval/results/llama-3.2-1b-${v}-ft_preds.jsonl && ! -f eval/results/no_lang_llama-3.2-1b-${v}-ft.json ]]; then
    python3 eval/score_no_lang.py --dump eval/results/llama-3.2-1b-${v}-ft_preds.jsonl --canon \
      2>&1 | tee logs/llama-3.2-1b-${v}-nolang.log
  fi
}

while [[ ! -f $DONE1 || ! -f $DONE2 ]]; do
  launched=0
  for g in 0 1; do
    u=$(used $g)
    [[ $u -le 10000 ]] || continue
    if [[ ! -f $DONE1 ]]; then
      run_variant spans $g
      launched=1
      break
    elif [[ ! -f $DONE2 ]]; then
      run_variant spansn $g
      launched=1
      break
    fi
  done
  [[ -f $DONE1 && -f $DONE2 ]] && break
  sleep 60
done
echo "=== round9 watcher DONE $(date)"
