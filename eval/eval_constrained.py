"""Evaluate extraction with grammar-constrained decoding (xgrammar).

Same scoring as eval_extract.py; generation is masked token-by-token with an
xgrammar GrammarMatcher compiled from each row's function schema (all params
required, nullable, enum-constrained — see src/target_schema.py). Use
--no-grammar to A/B the unconstrained baseline with identical batching.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_constrained.py \
      --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b/best --gpu 1 --tag llama-3.2-1b-ft-g
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from eval_extract import extract_json, norm_val  # noqa: E402
from render import render_prompt  # noqa: E402
from target_schema import target_json_schema  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--files", default="data/core_test.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--no-grammar", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--dump", action="store_true")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    return a


def main():
    args = parse_args()
    import torch
    import xgrammar as xgr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((ROOT / "train" / "configs" / f"{args.candidate}.json").read_text())
    tkw = cfg.get("template_kwargs") or {}
    rows = []
    for part in args.files.split(","):
        with open(part) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"candidate={args.candidate} adapter={args.adapter} rows={len(rows)} "
          f"grammar={not args.no_grammar}")

    tok = AutoTokenizer.from_pretrained(args.model_dir or cfg["model_id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir or cfg["model_id"], dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"merged adapter {args.adapter}")

    eos_ids = []
    for name in ("<|im_end|>", "<|eot_id|>", "<end_of_turn|>", "<eos>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and 0 < i < tok.vocab_size and i not in eos_ids:
            eos_ids.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)

    compiler = None
    compiled_by_fn: dict[str, object] = {}
    if not args.no_grammar:
        tok_info = xgr.TokenizerInfo.from_huggingface(
            tok, vocab_size=model.config.vocab_size, stop_token_ids=eos_ids)
        compiler = xgr.GrammarCompiler(tok_info)

        def compiled_for(row):
            fname = row["function"]["name"]
            if fname not in compiled_by_fn:
                compiled_by_fn[fname] = compiler.compile_json_schema(
                    json.dumps(target_json_schema(row["function"])),
                    any_order=True)
            return compiled_by_fn[fname]

    prompts = [render_prompt(tok, r, tkw) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))
    preds: list[str] = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), args.bs):
        idx = order[s:s + args.bs]
        enc = tok.pad({"input_ids": [prompts[i] for i in idx]}, return_tensors="pt",
                      padding=True).to("cuda")
        logits_processor = None
        if not args.no_grammar:
            from xgrammar.contrib.hf import LogitsProcessor as XgrLP
            logits_processor = [XgrLP([compiled_for(rows[i]) for i in idx])]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None,
                logits_processor=logits_processor)
        for j, i in enumerate(idx):
            preds[i] = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if (s // args.bs) % 10 == 0:
            print(f"  batch {s // args.bs + 1}/{(len(order) + args.bs - 1) // args.bs} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- scoring (same as eval_extract) ----
    def agg():
        return {"n": 0, "parse_ok": 0, "tp": 0, "fp": 0, "fn": 0,
                "null_gold": 0, "null_correct": 0, "row_em": 0}

    overall, by_lang, by_fn = agg(), {}, {}

    def add(a: dict, row: dict, pred: dict | None):
        a["n"] += 1
        a["parse_ok"] += int(pred is not None)
        ok = True
        gold = row["target"]
        for k, gv in gold.items():
            pv = (pred or {}).get(k, None)
            gvn, pvn = norm_val(gv), norm_val(pv)
            if gvn is None:
                a["null_gold"] += 1
                a["null_correct"] += int(pvn is None)
                if pvn is not None:
                    a["fp"] += 1
                    ok = False
            elif pvn == gvn and pvn is not None:
                a["tp"] += 1
            else:
                a["fn"] += 1
                if pvn is not None:
                    a["fp"] += 1
                ok = False
        a["row_em"] += int(ok)

    examples = []
    for row, p in zip(rows, preds):
        pred = extract_json(p)
        add(overall, row, pred)
        by_lang.setdefault(row["lang"], agg()); add(by_lang[row["lang"]], row, pred)
        fn = row["function"]["name"]
        by_fn.setdefault(fn, agg()); add(by_fn[fn], row, pred)
        if pred is None and len(examples) < 8:
            examples.append({"utterance": row["utterance"][:120], "raw": (p or "")[:200]})

    def metrics(a: dict) -> dict:
        prec = a["tp"] / max(1, a["tp"] + a["fp"])
        rec = a["tp"] / max(1, a["tp"] + a["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"n": a["n"], "parse_rate": round(a["parse_ok"] / max(1, a["n"]), 4),
                "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
                "slot_f1": round(f1, 4),
                "null_accuracy": round(a["null_correct"] / max(1, a["null_gold"]), 4),
                "row_em": round(a["row_em"] / max(1, a["n"]), 4)}

    result = {
        "candidate": args.candidate, "adapter": args.adapter, "files": args.files,
        "grammar": not args.no_grammar,
        "gen_seconds": round(time.time() - t0, 1), "overall": metrics(overall),
        "by_lang": {k: metrics(v) for k, v in sorted(by_lang.items())},
        "by_function": {k: metrics(v) for k, v in sorted(by_fn.items(), key=lambda kv: -kv[1]["n"])},
        "parse_failures": examples,
    }
    tag = args.tag or (args.candidate + ("-ft" if args.adapter else "-zeroshot")
                       + ("-g" if not args.no_grammar else ""))
    if args.dump:
        dump_path = ROOT / "eval" / "results" / f"{tag}_preds.jsonl"
        with open(dump_path, "w") as f:
            for row, p in zip(rows, preds):
                f.write(json.dumps({
                    "id": row["id"], "lang": row["lang"],
                    "function": row["function"]["name"], "target": row["target"],
                    "pred": extract_json(p), "raw": (p or "")[:300],
                }, ensure_ascii=False) + "\n")
        print(f"-> {dump_path}")
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print("by_lang:", {k: v["slot_f1"] for k, v in result["by_lang"].items()})
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
