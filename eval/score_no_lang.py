"""Re-score a prediction dump on the lang-free metric (round 7 official metric).

Drops the "lang"/"language" keys from both target and pred before scoring with
the exact eval_extract metrics (via canonicalize.agg/add/metrics), optionally
applying the canonicalize gazetteer post-processor in memory first. Reports
four blocks: all_slots and lang_free, each before vs after canonicalization.

Usage:
  python3 eval/score_no_lang.py --dump eval/results/llama-3.2-1b-ft_preds.jsonl
  python3 eval/score_no_lang.py --dump eval/results/bert-tagger-fixd-ft_preds.jsonl --canon
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from canonicalize import (agg, add, build_gazetteer, gaz_summary, metrics,  # noqa: E402
                          postprocess, read_jsonl, rpath)
from build_tags import props_of  # noqa: E402

NO_LANG_KEYS = ("lang", "language")


def strip_lang(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in NO_LANG_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="eval/results/<tag>_preds.jsonl to re-score lang-free")
    ap.add_argument("--canon", action="store_true",
                    help="apply the canonicalize post-processor before scoring")
    ap.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl",
                    help="comma-separated files to build the gazetteer from (with --canon)")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="dataset files joined by id for props_of (enum detection)")
    ap.add_argument("--out-tag", default=None,
                    help="result tag (default no_lang_<dump tag>)")
    args = ap.parse_args()
    gaz = build_gazetteer(args.train.split(",")) if args.canon else None
    rows = read_jsonl(rpath(args.dump))
    ds: dict = {}
    for part in args.files.split(","):
        for r in read_jsonl(rpath(part)):
            ds[r["id"]] = r
    all_b, all_a = agg(), agg()
    lang_b, lang_a = agg(), agg()
    counts, examples = Counter(), []
    for d in rows:
        dsrow = ds.get(d.get("id"))
        props = props_of(dsrow) if dsrow else {}
        target = d.get("target") or {}
        pred = d.get("pred")
        if gaz is not None and pred is not None:
            pred2 = postprocess(pred, d.get("lang"), props, gaz, counts, examples,
                                d.get("id"), target)
        else:
            pred2 = pred
        add(all_b, target, pred)
        add(all_a, target, pred2)
        pred_f = strip_lang(pred) if pred is not None else None
        pred2_f = strip_lang(pred2) if pred2 is not None else None
        add(lang_b, strip_lang(target), pred_f)
        add(lang_a, strip_lang(target), pred2_f)
    name = pathlib.Path(args.dump).name
    tag = name[:-len("_preds.jsonl")] if name.endswith("_preds.jsonl") else name
    args.out_tag = args.out_tag or f"no_lang_{tag}"
    result = {"preds": str(args.dump), "canon": gaz is not None,
              "gazetteer": gaz_summary(gaz) if gaz is not None else None,
              "all_slots": {"before": metrics(all_b), "after": metrics(all_a)},
              "lang_free": {"before": metrics(lang_b), "after": metrics(lang_a)},
              "applied": dict(counts), "changed_examples": examples}
    out = ROOT / "eval" / "results" / f"{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    a, l = result["all_slots"], result["lang_free"]
    print(f"{name}: all_slots F1={a['before']['slot_f1']} EM={a['before']['row_em']} "
          f"-> F1={a['after']['slot_f1']} EM={a['after']['row_em']} | "
          f"lang_free F1={l['before']['slot_f1']} EM={l['before']['row_em']} "
          f"-> F1={l['after']['slot_f1']} EM={l['after']['row_em']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
