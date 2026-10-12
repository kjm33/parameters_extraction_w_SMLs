"""Evaluate the span-list formulation (round 9): model emits compact JSON where
free-text param values are 1-based inclusive [first, last] whitespace-word index
pairs, enums are literals, nulls are null; decode to a normal prediction dict and
score with the exact eval_extract scoring.

Pipeline: greedy generate JSON -> parse_pred (markdown-fence tolerant) -> one
sampled regen (t=0.7) for unparseable rows -> repair_spans (clamp/swap/drop into
valid 1-based word indexes) -> decode_spans (src/build_spans) -> norm_val scoring
(eval/eval_extract) -> eval/results/<tag>.json + _preds.jsonl.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_spans.py \
      --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b-spans/best --gpu 1
  python3 eval/eval_spans.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from eval_extract import norm_val  # noqa: E402
from build_tags import props_of  # noqa: E402
from build_spans import decode_spans  # noqa: E402

FENCE_OPEN = re.compile(r"^```(?:json)?[ \t]*\n?")
FENCE_CLOSE = re.compile(r"```[ \t]*$")


def parse_pred(raw: str | None):
    """-> dict | None: json.loads, one markdown-fence retry, must be a dict."""
    if not isinstance(raw, str):
        return None
    try:
        v = json.loads(raw.strip())
        return v if isinstance(v, dict) else None
    except ValueError:
        pass
    s = raw.strip()
    if s.startswith("```"):
        s = FENCE_OPEN.sub("", s, count=1)
        s = FENCE_CLOSE.sub("", s.strip()).strip()
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else None
        except ValueError:
            return None
    return None


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pairish(x) -> bool:
    return (isinstance(x, (list, tuple)) and len(x) == 2
            and all(_is_num(i) for i in x))


def _scalarish(x) -> bool:
    return isinstance(x, str) or _is_num(x)


def _to_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and math.isfinite(x):
        return int(round(x))
    return None


def repair_spans(pred: dict, n_words: int):
    """Clamp/swap/drop predicted span values into valid 1-based [i, j] pairs.

    -> (repaired dict, Counter of clamped/swapped/dropped/form_broken).
    """
    rep: dict = {}
    counts: Counter = Counter()

    def fix(pairs: list) -> list:
        out = []
        for p in pairs:
            a, b = _to_int(p[0]), _to_int(p[1])
            if a is None or b is None or n_words < 1:
                counts["dropped"] += 1
                continue
            ca, cb = min(max(a, 1), n_words), min(max(b, 1), n_words)
            counts["clamped"] += int(ca != a) + int(cb != b)
            if ca > cb:
                ca, cb = cb, ca
                counts["swapped"] += 1
            out.append([ca, cb])
        return out

    for param, v in pred.items():
        if v is None or _scalarish(v):
            rep[param] = v
            continue
        if not isinstance(v, list):
            rep[param] = None
            counts["form_broken"] += 1
            continue
        scalar_pairs: list | None = None
        array_items: list | None = None
        if all(_is_num(x) for x in v):              # bare ints -> [i, i]
            scalar_pairs = [[i, i] for i in v]
        elif all(_pairish(x) for x in v):           # list of pairs -> scalar form
            scalar_pairs = [list(x) for x in v]
        elif all(isinstance(x, (list, tuple)) and x and all(_pairish(p) for p in x)
                 for x in v):                       # list of pair-lists -> array form
            array_items = [[list(p) for p in x] for x in v]
        elif all(_scalarish(x) for x in v):         # literal enum array
            rep[param] = v
            continue
        else:
            rep[param] = None
            counts["form_broken"] += 1
            continue
        if scalar_pairs is not None:
            fixed = fix(scalar_pairs)
            rep[param] = fixed if fixed else None
        else:
            assert array_items is not None
            items = [it for it in (fix(x) for x in array_items) if it]
            rep[param] = items if items else None
    return rep, counts


def _gold_pair_list(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(_pairish(p) for p in x)


def update_span_diag(span_diag: Counter, span_target: dict, repaired: dict):
    """Bucket |di_pred-i_gold| + |dj_pred-j_gold| per aligned segment; gold
    pair-form vs pred not pair-form -> form_mismatch."""
    for param, g in span_target.items():
        if not _gold_pair_list(g):
            continue
        pv = (repaired or {}).get(param)
        if not _gold_pair_list(pv):
            span_diag["form_mismatch"] += 1
            continue
        assert isinstance(pv, list)
        for gp, pp in zip(g, pv):
            d = abs(pp[0] - gp[0]) + abs(pp[1] - gp[1])
            span_diag["exact" if d == 0 else "pm1" if d <= 2
                      else "pm2" if d <= 4 else "gt2"] += 1


def _selftest() -> int:
    words = list("find a gas station, near me avoiding tolls".split())
    row = {"function": {"name": "t", "parameters": {"properties": {
        "q": {"type": "string"},
        "avoid_features": {"type": "array", "items": {
            "type": "string", "enum": ["tollRoad", "ferry"]}},
    }}}}
    p = parse_pred('{"q": [[3,4]], "avoid_features": ["tollRoad"], "lang": null}')
    assert isinstance(p, dict), p
    rep, _ = repair_spans(p, len(words))
    dec = decode_spans(rep, words, props_of(row), ", ")
    assert " ".join(words[2:4]) == "gas station,"  # raw slice has edge punct
    assert dec["q"] == "gas station", dec          # decode_spans strips it
    assert dec["avoid_features"] == ["tollRoad"], dec
    assert dec["lang"] is None, dec

    assert parse_pred('```json\n{"q": [[3, 4]]}\n```') == {"q": [[3, 4]]}
    assert parse_pred('```\n{"q": [5]}\n```') == {"q": [5]}
    assert parse_pred('{"q": [') is None
    assert parse_pred('[[1, 2]]') is None

    rep, cnt = repair_spans({"q": [[99, 100]]}, len(words))
    assert rep["q"] == [[len(words), len(words)]] and cnt["clamped"] == 2, (rep, cnt)
    rep, cnt = repair_spans({"q": [[5, 3]]}, len(words))
    assert rep["q"] == [[3, 5]] and cnt["swapped"] == 1, (rep, cnt)
    rep, _ = repair_spans({"q": [5]}, len(words))
    assert rep["q"] == [[5, 5]], rep

    rep, cnt = repair_spans({"q": [[1, 2], 3]}, len(words))
    assert rep["q"] is None and cnt["form_broken"] == 1, (rep, cnt)
    rep, _ = repair_spans({"q": [[[3, 4]], [[6, 8]]]}, len(words))
    dec = decode_spans(rep, words, props_of(row), ", ")
    assert dec["q"] == ["gas station", "me avoiding tolls"], dec
    print("SELFTEST OK")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--files", default="data/core_test_spans.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--tag", default=None)
    p.add_argument("--dump", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return _selftest()
    if not a.candidate:
        p.error("--candidate is required (unless --selftest)")
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

    stats_path = ROOT / "data" / "spans_build_stats.json"
    join_sep = ", "
    if stats_path.exists():
        join_sep = json.loads(stats_path.read_text()).get("join_sep") or ", "

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

    raws: list[str | None] = [None] * len(rows)
    ntok = 0
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

    parsed = [parse_pred(r) for r in raws]
    bad = [i for i, p in enumerate(parsed) if p is None]
    if bad:
        print(f"regen {len(bad)} rows with unparseable JSON ...")
        for s in range(0, len(bad), a.bs):
            idx = bad[s:s + a.bs]
            outs = gen(idx, sample=True)
            for i, o in zip(idx, outs):
                p2 = parse_pred(o)
                if p2 is not None:
                    parsed[i] = p2
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

    repairs: list = [None] * len(rows)
    preds: list = [None] * len(rows)
    repair_counters: Counter = Counter()
    span_diag: Counter = Counter()
    for i, (row, p) in enumerate(zip(rows, parsed)):
        words = row["utterance"].split()
        props = props_of(row)
        if p is not None:
            rep, cnt = repair_spans(p, len(words))
            repair_counters.update(cnt)
            pred = decode_spans(rep, words, props, join_sep)
            update_span_diag(span_diag, row.get("span_target") or {}, rep)
            repairs[i], preds[i] = rep, pred
        add(overall, row, preds[i])
        by_lang.setdefault(row["lang"], agg())
        add(by_lang[row["lang"]], row, preds[i])
        fn = row["function"]["name"]
        by_fn.setdefault(fn, agg())
        add(by_fn[fn], row, preds[i])

    def metrics(acc: dict) -> dict:
        prec = acc["tp"] / max(1, acc["tp"] + acc["fp"])
        rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"n": acc["n"], "parse_rate": round(acc["parse_ok"] / max(1, acc["n"]), 4),
                "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
                "slot_f1": round(f1, 4),
                "null_accuracy": round(acc["null_correct"] / max(1, acc["null_gold"]), 4),
                "row_em": round(acc["row_em"] / max(1, acc["n"]), 4)}

    n_parse_fail = sum(1 for p in parsed if p is None)
    result = {
        "candidate": a.candidate, "adapter": a.adapter, "files": a.files,
        "formulation": "span-lists",
        "prompt_variant": "numbered" if "spansn" in a.files else "plain",
        "join_sep": join_sep,
        "gen_seconds": round(gen_s, 1),
        "ms_per_row": round(gen_s * 1000 / max(1, len(rows)), 1),
        "avg_new_tokens": round(ntok / max(1, len(rows)), 1),
        "parse_rate": round(overall["parse_ok"] / max(1, overall["n"]), 4),
        "regen_rows": len(bad), "parse_fail_rows": n_parse_fail,
        "repair_counters": dict(sorted(repair_counters.items())),
        "span_diag": dict(sorted(span_diag.items())),
        "overall": metrics(overall),
        "by_lang": {k: metrics(v) for k, v in sorted(by_lang.items())},
        "by_function": {k: metrics(v) for k, v in
                        sorted(by_fn.items(), key=lambda kv: -kv[1]["n"])},
    }
    tag = a.tag or (a.candidate + ("-spansn" if "spansn" in a.files else "-spans")
                    + ("-ft" if a.adapter else "-zeroshot"))
    if a.dump:
        dump_path = ROOT / "eval" / "results" / f"{tag}_preds.jsonl"
        with open(dump_path, "w") as f:
            for row, raw, pred, rep in zip(rows, raws, preds, repairs):
                f.write(json.dumps({
                    "id": row["id"], "lang": row["lang"],
                    "function": row["function"]["name"], "target": row["target"],
                    "pred": pred, "raw": (raw or "")[:300],
                    "span_gold": row.get("span_target"), "span_pred": rep,
                }, ensure_ascii=False) + "\n")
        print(f"-> {dump_path}")
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print("by_lang:", {k: v["slot_f1"] for k, v in result["by_lang"].items()})
    print(f"parse_rate={result['parse_rate']} regen={result['regen_rows']} "
          f"avg_new_tokens={result['avg_new_tokens']} "
          f"ms_per_row={result['ms_per_row']}")
    print(f"span_diag={result['span_diag']} repair={result['repair_counters']}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
