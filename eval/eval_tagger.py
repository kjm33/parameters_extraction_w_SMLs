"""Evaluate the Track-B encoder tagger (round 4) on core_test: first-subtoken argmax
per utterance word -> BIO tag -> decode_tags -> JSON -> eval_extract scoring.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/eval_tagger.py \
      --ckpt train/runs/bert-tagger/best.pt --gpu 0 --dump
  add --cpu for edge-latency measurement on CPU
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from eval_extract import norm_val  # noqa: E402
from build_tags import decode_tags, props_of  # noqa: E402
from train_tagger import Joint, encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/runs/bert-tagger/best.pt")
    ap.add_argument("--files", default="data/core_test_tags.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tag", default="bert-tagger-ft")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--stats", default="data/tags_build_stats.json",
                    help="tags build stats (join_sep); use tags_build_stats_v3.json for *_v3_tags files")
    a = ap.parse_args()
    import os
    if not a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from transformers import AutoTokenizer

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    labels, langs = ck["labels"], ck["langs"]
    label2id = {t: i for i, t in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(ck["base"])
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model = Joint(ck["base"], len(labels)).to(
        device=device, dtype=torch.bfloat16 if a.bf16 else torch.float32
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    rows = []
    for part in a.files.split(","):
        with open(ROOT / part) as f:
            rows.extend(json.loads(ln) for ln in f if ln.strip())
    if a.limit:
        rows = rows[:a.limit]
    print(f"bert-tagger rows={len(rows)} device={device}")

    join_stats = json.loads((ROOT / a.stats).read_text())
    join_sep = join_stats.get("join_sep", ", ")

    preds = []
    t0 = time.time()
    for s in range(0, len(rows), a.bs):
        chunk = rows[s:s + a.bs]
        enc = [encode(tok, r, label2id) for r in chunk]
        mx = max(len(e["input_ids"]) for e in enc)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(enc), mx), pad, dtype=torch.long)
        att = torch.zeros((len(enc), mx), dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :len(e["input_ids"])] = torch.tensor(e["input_ids"])
            att[j, :len(e["input_ids"])] = 1
        with torch.no_grad():
            tag_logits, lang_logits = model(ids.to(device), att.to(device))
        lab = tag_logits.argmax(-1).cpu()
        langp = lang_logits.argmax(-1).cpu()
        for j, (r, e) in enumerate(zip(chunk, enc)):
            words = r["utterance"].split()
            tags = []
            for wi in range(len(words)):
                pos = e["first_pos"].get(wi)
                if pos is None or pos >= e["n_utt"]:
                    tags.append("O")
                    continue
                p = e["base"] + pos
                if p >= lab.shape[1]:
                    tags.append("O")
                    continue
                tags.append(labels[lab[j, p].item()])
            lang = "ABSENT"
            if "lang" in r["target"]:
                lang = langs[langp[j].item()]
            preds.append((tags, lang))
    elapsed = time.time() - t0

    def agg():
        return {"n": 0, "parse_ok": 0, "tp": 0, "fp": 0, "fn": 0,
                "null_gold": 0, "null_correct": 0, "row_em": 0}

    overall, by_lang, by_fn = agg(), {}, {}

    def add(acc, row, pred):
        acc["n"] += 1
        acc["parse_ok"] += 1
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

    for row, (tags, lang) in zip(rows, preds):
        pred = decode_tags(tags, row["utterance"].split(), props_of(row), join_sep=join_sep)
        if lang != "ABSENT":
            pred["lang"] = None if lang == "null" else lang
        add(overall, row, pred)
        by_lang.setdefault(row["lang"], agg())
        add(by_lang[row["lang"]], row, pred)
        fn = row["function"]["name"]
        by_fn.setdefault(fn, agg())
        add(by_fn[fn], row, pred)

    def metrics(acc):
        prec = acc["tp"] / max(1, acc["tp"] + acc["fp"])
        rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"n": acc["n"], "parse_rate": 1.0,
                "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
                "slot_f1": round(f1, 4),
                "null_accuracy": round(acc["null_correct"] / max(1, acc["null_gold"]), 4),
                "row_em": round(acc["row_em"] / max(1, acc["n"]), 4)}

    result = {
        "formulation": "bert-tagger", "ckpt": a.ckpt, "files": a.files,
        "device": device, "seconds": round(elapsed, 1),
        "ms_per_row": round(1000 * elapsed / max(1, len(rows)), 1),
        "overall": metrics(overall),
        "by_lang": {k: metrics(v) for k, v in sorted(by_lang.items())},
        "by_function": {k: metrics(v) for k, v in
                        sorted(by_fn.items(), key=lambda kv: -kv[1]["n"])},
    }
    tag = a.tag + ("-cpu" if a.cpu else "")
    if a.dump:
        dump_path = ROOT / "eval" / "results" / f"{tag}_preds.jsonl"
        with open(dump_path, "w") as f:
            for row, (tags, lang) in zip(rows, preds):
                pred = decode_tags(tags, row["utterance"].split(), props_of(row),
                                   join_sep=join_sep)
                if lang != "ABSENT":
                    pred["lang"] = None if lang == "null" else lang
                f.write(json.dumps({
                    "id": row["id"], "lang": row["lang"],
                    "function": row["function"]["name"], "target": row["target"],
                    "pred": pred, "raw": " ".join(tags)[:300],
                }, ensure_ascii=False) + "\n")
        print(f"-> {dump_path}")
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print("by_lang:", {k: v["slot_f1"] for k, v in result["by_lang"].items()})
    print(f"ms/row={result['ms_per_row']} ({device})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
