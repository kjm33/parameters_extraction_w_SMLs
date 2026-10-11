#!/bin/zsh
# Experiment B: retrain llama-3.2-1b LoRA on NER-hinted data, then 2x2 eval.
# Waits for free GPU (<6GB) before each step; retries training until best exists.
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/var/tmp/fcft_hf_cache
cd "$(dirname "$0")/.."

log() { builtin print -r -- "$*"; }

wait_gpu() {
  while true; do
    for g in 0 1; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if (( used < 6000 )); then print -r -- $g; return; fi
    done
    sleep 60
  done
}

attempt=0
while [[ ! -f train/runs/llama-3.2-1b-ner/best/adapter_model.safetensors ]]; do
  attempt=$((attempt + 1))
  if (( attempt > 12 )); then log "=== GIVE UP llama-3.2-1b-ner"; break; fi
  GPU=$(wait_gpu)
  log "=== GPU $GPU -> TRAIN llama-3.2-1b-ner (attempt $attempt) $(date)"
  $PY train/train_sft.py --candidate llama-3.2-1b \
    --train data/core_train_ner.jsonl,data/aug_train_ner.jsonl \
    --out-dir train/runs/llama-3.2-1b-ner --gpu $GPU
  log "TRAIN_EXIT=$? (attempt $attempt) $(date)"
  [[ -f train/runs/llama-3.2-1b-ner/best/adapter_model.safetensors ]] || sleep 120
done

# 2x2 evals (baseline x plain exists: F1 0.7515)
run_eval() { # tag outfile args...
  local tag=$1 out=$2; shift 2
  local a=0
  while [[ ! -f $out ]]; do
    a=$((a + 1)); (( a > 8 )) && { log "=== GIVE UP eval $tag"; return 1; }
    GPU=$(wait_gpu)
    log "=== GPU $GPU -> EVAL $tag (attempt $a) $(date)"
    $PY eval/eval_extract.py "$@" --gpu $GPU --tag $tag --dump
    log "EVAL_EXIT_${tag}=$? $(date)"
    [[ -f $out ]] || sleep 60
  done
}

run_eval baseline-hinted eval/results/baseline-hinted.json \
  --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b/best --files data/core_test_ner.jsonl
run_eval ner-hinted eval/results/ner-hinted.json \
  --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-ner/best --files data/core_test_ner.jsonl
run_eval ner-plain eval/results/ner-plain.json \
  --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-ner/best --files data/core_test.jsonl

log "=== EXPERIMENT B ALL DONE $(date)"
