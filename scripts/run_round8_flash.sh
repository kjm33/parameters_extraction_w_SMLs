#!/usr/bin/env bash
# Round 8 addendum: flash-attn + unpadding latency check for mmbert-tagger-fixd-v2.
# Requires the C++20-patched flash-attn 2.8.3.post1 wheel built from /tmp/opencode/fa-src
# (FLASH_ATTN_CUDA_ARCHS=80, sm80 cubin runs on the sm86 GPU). Run AFTER installing:
#   /home/kamil/.local/bin/uv pip install --python ../function_calling_edge_fine_tuning/train/venv/bin/python /tmp/opencode/fa-src/dist/flash_attn-*.whl
# Comparisons (GPU 1, core_test_tags n=817):
#   sdpa fp32  : existing mmbert-tagger-fixd-v2-ft.json  (4.3 ms/row baseline)
#   sdpa bf16  : dtype control (is bf16 alone faster?)
#   fa   bf16  : treatment (flash kernels + ModernBERT unpadding)
# F1 drift checked via the fa dump through score_no_lang --canon (official metric).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../function_calling_edge_fine_tuning/train/venv/bin/python
CKPT=train/runs/mmbert-tagger-fixd-v2/best.pt
LOG=logs/mmbert-tagger-fixd-v2-flash.log
exec > >(tee -a "$LOG") 2>&1
echo "=== round8 flash-attn check $(date -Is)"

echo "--- import smoke (GPU 1)"
CUDA_VISIBLE_DEVICES=1 "$PY" - <<'EOF'
import torch, transformers
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
q = torch.randn(2, 8, 16, 64, dtype=torch.bfloat16, device="cuda")
o = flash_attn_func(q, q, q, causal=False)
assert o.shape == q.shape and o.dtype == torch.bfloat16
cu = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
qf = q.reshape(-1, 8, 64)
ov = flash_attn_varlen_func(qf, qf, qf, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=8, max_seqlen_k=8)
assert ov.shape == qf.shape
print(f"flash_attn {flash_attn.__version__} OK, torch {torch.__version__}, transformers {transformers.__version__}, dev {torch.cuda.get_device_name(0)}")
EOF
[ $? -ne 0 ] && { echo "FLASH-ATTN IMPORT SMOKE FAILED — aborting"; exit 1; }

echo "--- eval: flash_attention_2 + bf16 (dump)"
TAGGER_ATTN_IMPL=flash_attention_2,sdpa "$PY" eval/eval_tagger.py \
  --ckpt "$CKPT" --gpu 1 --bf16 --dump --tag mmbert-tagger-fixd-v2-fa

echo "--- eval: sdpa + bf16 control (no dump)"
TAGGER_ATTN_IMPL=sdpa "$PY" eval/eval_tagger.py \
  --ckpt "$CKPT" --gpu 1 --bf16 --tag mmbert-tagger-fixd-v2-sdpa-bf16

echo "--- eval: flash_attention_2 + bf16, bs 128 (unpadding scaling probe)"
TAGGER_ATTN_IMPL=flash_attention_2,sdpa "$PY" eval/eval_tagger.py \
  --ckpt "$CKPT" --gpu 1 --bf16 --bs 128 --tag mmbert-tagger-fixd-v2-fa-bs128

echo "--- official lang-free + canon score of the fa dump"
python3 eval/score_no_lang.py --dump eval/results/mmbert-tagger-fixd-v2-fa_preds.jsonl \
  --canon --out-tag no_lang_mmbert-tagger-fixd-v2-fa

echo "--- done $(date -Is); compare ms/row:"
for t in mmbert-tagger-fixd-v2-ft mmbert-tagger-fixd-v2-sdpa-bf16 mmbert-tagger-fixd-v2-fa mmbert-tagger-fixd-v2-fa-bs128; do
  f=eval/results/$t.json
  [ -f "$f" ] && python3 -c "import json;r=json.load(open('$f'));print(f\"{r['device']:4s} bs-default/see-log {r['ms_per_row']:6.1f} ms/row  F1={r['overall']['slot_f1']}\", '$t')"
done
