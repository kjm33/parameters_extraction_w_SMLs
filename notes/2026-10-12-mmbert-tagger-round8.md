# 2026-10-12 — mmBERT tagger (round 8): modern-encoder swap for the edge lane

Context: user asked about different models ("e.g. ModernBERT"). Vanilla ModernBERT is
English-only (pretraining + tokenizer) → wrong tool for en/de/pl/fr/es (pl already the
weak spot). The multilingual successor **mmBERT** (`jhu-clsp/mmBERT-base`, Sep 2025,
JHU-CLSP): ModernBERT architecture (22L, RoPE, unpadding), Gemma-2 tokenizer, 3T tokens,
1,833 languages, beats XLM-R on XTREME NLU/QA, 2–4× faster than XLM-R/mBERT. Caveats
from their own blog: on token-tagging/NER it is only ≈ previous generation (tokenizer
word-boundary differences); flash-attn is NOT in the venv → sdpa/eager fallback, so the
speed win will be partial.

Round-6 precedent: encoder capacity did not decide F1 (XLM-R ≤ mBERT, 0.7340 vs 0.7366
+ canon); distill data helped encoders (+0.5–1.6). Tag-formulation structural ceiling
(91.1% of non-null slots) stands regardless of backbone.

## Plan (single pre-registered run, mirrors bert-tagger-fixd-v2 exactly)

- Run `mmbert-tagger-fixd-v2`: backbone `jhu-clsp/mmBERT-base`, train =
  core+aug+distill tags (v2), val = core_val+aug_val, 5 epochs, bs 16, lr 5e-5,
  lang-weight 0.5, seed 42, MAX_LEN 256. GPU 1.
- Code change (compat only): tokenizer-agnostic special tokens in `train_tagger.py`
  (cls→bos / sep→eos / unk / pad fallbacks), attn-implementation fallback
  (sdpa→eager) in `Joint`; mBERT/XLM-R behavior unchanged (regression-checked).
- Eval: `eval_tagger.py --dump` → `score_no_lang.py --canon` (official lang-free
  metric, same canonicalize post-processor). Plus a `--cpu` latency pass.
- Reference: `bert-tagger-fixd-v2 + canon` = **0.7813 F1 / 0.7846 EM** lang-free
  (raw 0.7566/0.7674; 4.3 ms/row GPU, ~107 ms/row CPU).

## Decision rules (pre-registered, before any run)

1. **Adopt** mmBERT as the edge candidate iff lang-free canon F1 > 0.7813.
2. If F1 ∈ [0.7763, 0.7813] (±0.5 noise band) AND CPU ms/row improves ≥1.5× →
   "latency-competitive alternative": document, no change to the edge pick.
3. If F1 < 0.7763 → **reject**; third base confirms backbone swaps are non-decisive in
   this lane; edge lane closed for encoder swaps.
4. JSON-winner lane (Llama-3.2-1B) untouched regardless.

## Expectations (honest prior)

- F1 gain unlikely (blog tagging caveat + 91.1% ceiling + XLM-R precedent); the
  plausible win is latency (RoPE/unpadded arch) — hence rule 2.
- Watch: Gemma-2 SP word-tokenization alignment (first-subtoken labels are per
  whitespace word, so unchanged contract); lang head on token 0 (BOS) for Gemma.

## Outcome (2026-10-12, run complete)

Run: `scripts/run_round8_mmbert.sh` → `train/runs/mmbert-tagger-fixd-v2/` (best val
tag-acc 0.9837 @ ep 2 vs mBERT 0.9832, ~21 min).

| metric (core_test n=817, v2) | mmBERT | mBERT fixd-v2 |
|---|---|---|
| raw F1 / EM | 0.7586 / 0.7699 | 0.7566 / 0.7674 |
| lang-free + canon F1 / EM | **0.7838 / 0.7870** | 0.7813 / 0.7846 |
| GPU / CPU ms/row | 4.3 / 123.8 | 4.3 / ~107 |

By-lang F1 (mmBERT vs mBERT): de .8320/.8346, en .7236/.7463 (en drop, XLM-R pattern),
es .7913/.7848, fr .7747/.7656, pl .6292/.6129 (best pl of any tagger, still ≪ winner
.7399).

Decision per rules: rule 1 nominally triggers (0.7838 > 0.7813) but +0.25 < the ±0.5
noise band → **statistical tie, not a win**. Rule 2's latency condition failed (CPU
~16% worse, GPU tie — no flash-attn/unpadding in venv). Outcome: mmBERT documented as
tie-alternative; edge pick stays mBERT where CPU latency matters; backbone-swap lane
closed (3rd backbone, capacity/arch non-decisive); JSON-winner lane untouched.
Reported in RESULTS.md "Round 8".

## Flash-attn follow-up (same day, tested → negative)

Built flash-attn 2.8.3.post1 from source against torch 2.14 (needs `-std=c++20` patch —
stock sdist uses c++17 and fails on `std::strong_ordering`; prebuilt wheels are
torch-2.9-ABI and fail to import; `FLASH_ATTN_CUDA_ARCHS=80` since setup.py hardcodes
80;90;100;120 and sm80 cubin runs on our sm86 GPU; PATH must include venv/bin or the
build silently degrades to serial distutils). Smoke: `flash_attn_func` +
`flash_attn_varlen_func` (kwargs `cu_seqlens_q/k`, NOT `cu_seqlens`) pass on GPU 1.
Code: `TAGGER_ATTN_IMPL` env override in train_tagger.py attn chain; `--bf16` in
eval_tagger.py. `scripts/run_round8_flash.sh`, logs/mmbert-tagger-fixd-v2-flash.log.

| pass (GPU 1, n=817) | ms/row | raw F1 |
|---|---|---|
| sdpa fp32 (baseline) | 4.3 | 0.7586 |
| sdpa bf16 | 4.3 | 0.7563 |
| flash bf16 | 5.0 | 0.7577 |
| flash bf16 bs128 | 4.7 | 0.7577 |

**Negative**: flash ~16% slower than sdpa in this regime — short sequences (≈50–80
tok) + small batches mean varlen pack/unpack overhead dominates; flash's advantage is
long-context only. bf16 alone: no speed change, F1 noise. Lang-free+canon flash dump
0.783/0.7858 ≈ baseline 0.7838/0.787 (kernel noise). Round-8 decision stands, now
with the flash-attn branch explicitly closed too.
