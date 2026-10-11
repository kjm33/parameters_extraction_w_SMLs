"""Evaluate the tagging formulation (round 4): generate BIO tags, decode to JSON,
score with the exact eval_extract scoring.

Pipeline: greedy generate tag string -> parse/validate (word-count match after the
optional leading lang: tag; unknown tokens dropped) -> one sampled regen (t=0.7) for
mismatched rows -> left-align/truncate fallback -> decode_tags (src/build_tags) ->
norm_val scoring (eval/eval_extract) -> eval/results/<tag>.json + _preds.jsonl.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_tags.py \
      --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-tags/best --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from eval_extract import norm_val  # noqa: E402
from build_tags import decode_tags, props_of  # noqa: E402

VALID_PREFIXES = ("O", "B-", "I-", "lang:")


def parse_tags(raw: str, n_words: int):
    """-> (word_tags, lang_pred_or_ABSENT, count_ok, n_junk)."""
    toks = raw.split()
    tags, junk = [], 0
    for t in toks:
        if t == "O" or t.startswith(("B-", "I-", "lang:")):
            tags.append(t)
        else:
            junk += 1
    lang = "ABSENT"
    if tags and tags[0].startswith("lang:"):
        lang = tags[0][5:]
        tags = tags[1:]
    ok = len(tags) == n_words
    if len(tags) > n_words:
        tags = tags[:n_words]
    tags = tags + ["O"] * (n_words - len(tags))
    return tags, lang, ok, junk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--files", default="data/core_test_tags.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--tag", default=None)
    p.add_argument("--dump", action="store_true")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from render import render_prompt

    cfg = json.loads((ROOT / "train" / "configs" / f"{a.candidate}.json").read_text())
    tkw = cfg.get("template_kwargs") or {}
    rows = []
    for part in a.files.split(","):
        with open(part) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if a.limit:
        rows = rows[:a.limit]
    print(f"candidate={a.candidate} adapter={a.adapter} rows={len(rows)}")

    join_stats = json.loads((ROOT / "data" / "tags_build_stats.json").read_text())
    join_sep = join_stats.get("join_sep", ", ")

    tok = AutoTokenizer.from_pretrained(a.model_dir or cfg["model_id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir or cfg["model_id"], dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        model = model.merge_and_unload()
        print(f"merged adapter {a.adapter}")
    eos_ids = []
    for name in ("<|im_end|>", "<|eot_id|>", "<end_of_turn|>", "<eos>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and 0 < i < tok.vocab_size and i not in eos_ids:
            eos_ids.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)

    prompts = [render_prompt(tok, r, tkw) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))

    def gen(idx_batch, sample=False):
        enc = tok.pad({"input_ids": [prompts[i] for i in idx_batch]},
                      return_tensors="pt", padding=True).to("cuda")
        kw = dict(max_new_tokens=a.max_new_tokens,
                  pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None)
        if sample:
            kw.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        return [tok.decode(out[j][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True) for j in range(len(idx_batch))]

    raws, ntok = [None] * len(rows), 0
    t0 = time.time()
    for s in range(0, len(order), a.bs):
        idx = order[s:s + a.bs]
        outs = gen(idx)
        for i, o in zip(idx, outs):
            raws[i] = o
            ntok += len(tok(o)["input_ids"])
        if (s // a.bs) % 10 == 0:
            print(f"  batch {s // a.bs + 1}/{(len(order) + a.bs - 1) // a.bs} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    parsed = [parse_tags(r, len(row["utterance"].split()))
              for r, row in zip(raws, rows)]
    bad = [i for i, (_, _, ok, _) in enumerate(parsed) if not ok]
    if bad:
        print(f"regen {len(bad)} rows with tag-count mismatch ...")
        for s in range(0, len(bad), a.bs):
            idx = bad[s:s + a.bs]
            outs = gen(idx, sample=True)
            for i, o in zip(idx, outs):
                t2 = parse_tags(o, len(rows[i]["utterance"].split()))
                if t2[2]:
                    parsed[i] = t2
                    raws[i] = o
    gen_s = time.time() - t0

    # ---- scoring (identical to eval_extract) ----
    def agg():
        return {"n": 0, "parse_ok": 0, "tp": 0, "fp": 0, "fn": 0,
                "null_gold": 0, "null_correct": 0, "row_em": 0}

    overall, by_lang, by_fn = agg(), {}, {}

    def add(acc: dict, row: dict, pred: dict | None):
        acc["n"] += 1
        acc["parse_ok"] += int(pred is not None)
        ok = True
        for k, gv in row["target"].items():
            pv = (pred or {}).get(k, None)
            gvn, pvn = norm_val(gv), norm_val(pv)
            if gvn is None:
                acc["null_gold"] += 1
                acc["null_correct"] += int(pvn is None)
                if pvn is not None:
                    acc["fp"] += 1
                    ok = False
            elif pvn == gvn and pvn is not None:
                acc["tp"] += 1
            else:
                acc["fn"] += 1
                if pvn is not None:
                    acc["fp"] += 1
                ok = False
        acc["row_em"] += int(ok)

    n_fallback = sum(1 for _, _, ok, _ in parsed if not ok)
    n_junk = sum(j for _, _, _, j in parsed)
    for row, (tags, lang, _, _) in zip(rows, parsed):
        props = props_of(row)
        words = row["utterance"].split()
        pred = decode_tags(tags, words, props, join_sep=join_sep)
        if lang != "ABSENT":
            pred["lang"] = None if lang == "null" else lang
        add(overall, row, pred)
        by_lang.setdefault(row["lang"], agg())
        add(by_lang[row["lang"]], row, pred)
        fn = row["function"]["name"]
        by_fn.setdefault(fn, agg())
        add(by_fn[fn], row, pred)

    def metrics(acc: dict) -> dict:
        prec = acc["tp"] / max(1, acc["tp"] + acc["fp"])
        rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"n": acc["n"], "parse_rate": round(acc["parse_ok"] / max(1, acc["n"]), 4),
                "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
                "slot_f1": round(f1, 4),
                "null_accuracy": round(acc["null_correct"] / max(1, acc["null_gold"]), 4),
                "row_em": round(acc["row_em"] / max(1, acc["n"]), 4)}

    result = {
        "candidate": a.candidate, "adapter": a.adapter, "files": a.files,
        "formulation": "bio-tags", "join_sep": join_sep,
        "gen_seconds": round(gen_s, 1),
        "avg_new_tokens": round(ntok / max(1, len(rows)), 1),
        "tag_count_ok_rate": round(1 - n_fallback / max(1, len(rows)), 4),
        "regen_rows": len(bad), "align_fallback_rows": n_fallback, "junk_tokens": n_junk,
        "overall": metrics(overall),
        "by_lang": {k: metrics(v) for k, v in sorted(by_lang.items())},
        "by_function": {k: metrics(v) for k, v in
                        sorted(by_fn.items(), key=lambda kv: -kv[1]["n"])},
    }
    tag = a.tag or (a.candidate + "-tags" + ("-ft" if a.adapter else "-zeroshot"))
    if a.dump:
        dump_path = ROOT / "eval" / "results" / f"{tag}_preds.jsonl"
        with open(dump_path, "w") as f:
            for row, raw, (tags, lang, ok, _) in zip(rows, raws, parsed):
                props = props_of(row)
                pred = decode_tags(tags, row["utterance"].split(), props, join_sep=join_sep)
                if lang != "ABSENT":
                    pred["lang"] = None if lang == "null" else lang
                f.write(json.dumps({
                    "id": row["id"], "lang": row["lang"],
                    "function": row["function"]["name"], "target": row["target"],
                    "pred": pred, "raw": (raw or "")[:300], "tag_count_ok": ok,
                }, ensure_ascii=False) + "\n")
        print(f"-> {dump_path}")
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print("by_lang:", {k: v["slot_f1"] for k, v in result["by_lang"].items()})
    print(f"tag_count_ok={result['tag_count_ok_rate']} avg_new_tokens={result['avg_new_tokens']}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
