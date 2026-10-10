"""Probe: Jev decision model (OpenRouter) as verification gate / failure detector.

Takes a detect_uncertainty-style *_rows.jsonl dump (pred + correctness of the
fine-tuned local model), joins the raw dataset rows by id, and asks Jev ONE noul
question per slot in a single per-row call:
  pred[k] is null  -> "utterance specifies a value for k"        (risk = p)
  pred[k] = v      -> "value v for k is stated/implied by utterance" (risk = 1-p)
Row risk = max slot risk (parse failures -> 1.0). Reports row-level AUC +
escalation coverage vs the local model's actual errors, slot-level AUC, and a
rank-average combination with the logprob baseline from *_features.jsonl —
directly comparable to reports in eval/results/llama-1b-merged-test.json.

Outputs: eval/results/<tag>.json, eval/results/<tag>_rows.jsonl

Usage:
  python3 eval/jev_verify.py \
      --rows eval/results/llama-1b-merged-test_rows.jsonl \
      --files data/core_test.jsonl --tag jev_verify_llama-1b-merged-test
  python3 eval/jev_verify.py --limit 20 --mock     # offline plumbing test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from detect_uncertainty import auc, coverage_table  # noqa: E402
from eval_extract import norm_val  # noqa: E402
from jev_client import JevClient  # noqa: E402


def props_of(row: dict) -> dict:
    params = row["function"].get("parameters") or {}
    return params.get("properties") or params or {}


def render_val(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def build_task(core: dict, pred: dict | None):
    keys = list(core["target"].keys())
    if not keys:
        return None
    props = props_of(core)
    questions = {}
    for k in keys:
        desc = (props.get(k) or {}).get("description") or k
        pv = (pred or {}).get(k)
        if pv is None:
            questions[f"{k}__specified"] = {
                "type": "noul",
                "instructions": (
                    f"The user's utterance explicitly specifies a value for the "
                    f"parameter '{k}' ({desc}). Only explicit user intent counts, "
                    f"not values the system could pick on its own.")}
        else:
            questions[f"{k}__grounded"] = {
                "type": "noul",
                "instructions": (
                    f"The value '{render_val(pv)}' for the parameter '{k}' ({desc}) "
                    f"is stated in, or directly implied by, the user's utterance.")}
    state = {"function": core["function"]["name"],
             "function_description": core["function"].get("description", ""),
             "extracted_parameters": pred or {},
             "user_utterance": core["utterance"]}
    return keys, state, questions


def slot_risks(keys: list, pred: dict | None, answers: dict) -> dict:
    out = {}
    for k in keys:
        pv = (pred or {}).get(k)
        if pv is None:
            p = (answers.get(f"{k}__specified") or {}).get("noul")
            out[k] = None if p is None else float(p)
        else:
            p = (answers.get(f"{k}__grounded") or {}).get("noul")
            out[k] = None if p is None else 1.0 - float(p)
    return out


def rank_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    return ra + rb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="eval/results/llama-1b-merged-test_rows.jsonl")
    p.add_argument("--files", default="data/core_test.jsonl",
                   help="dataset with utterance+target, joined by id")
    p.add_argument("--features", default=None,
                   help="*_features.jsonl for logprob baseline (default: sibling of --rows)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default="typesafe/jev-1.13")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--mock", action="store_true", help="offline plumbing test")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows_path = ROOT / args.rows
    recs = [json.loads(l) for l in open(rows_path) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    core = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / args.files)
                                 if l.strip())}
    tasks, kept = [], []
    for rec in recs:
        c = core.get(rec["id"])
        bt = build_task(c, rec.get("pred")) if c else None
        if bt:
            keys, state, questions = bt
            tasks.append((state, questions))
            kept.append((rec, c, keys))
    print(f"rows={len(recs)} usable={len(kept)} model={args.model} mock={args.mock}")

    cli = JevClient(model=args.model, mock=args.mock, workers=args.workers)
    t0 = time.time()
    results = cli.run_many(tasks)
    wall = time.time() - t0

    out_rows, wrong, risks, n_fail = [], [], [], 0
    slot_scores, slot_labels, slot_names = [], [], []
    for (rec, c, keys), res in zip(kept, results):
        answers = (res or {}).get("answers") or {}
        if not res:
            n_fail += 1
        pred = rec.get("pred")
        sr = slot_risks(keys, pred, answers)
        if pred is None:
            risk = 1.0
        else:
            vals = [v for v in sr.values() if v is not None]
            risk = max(vals) if vals else 0.0
        wrong.append(int(not rec["correct"]))
        risks.append(risk)
        gold = c["target"]
        for k in keys:
            if sr[k] is None:
                continue
            slot_scores.append(sr[k])
            slot_labels.append(int(norm_val((pred or {}).get(k)) != norm_val(gold.get(k))))
            slot_names.append(k)
        out_rows.append({"id": rec["id"], "lang": rec["lang"], "correct": rec["correct"],
                         "pred": pred, "slot_risk": sr, "row_risk": round(risk, 4)})

    risks, wrong = np.array(risks), np.array(wrong)
    report = {
        "model": args.model, "rows_file": str(rows_path), "files": args.files,
        "n_rows": len(kept), "n_wrong": int(wrong.sum()),
        "n_calls_failed": n_fail, "mock": cli.mock,
        "wall_seconds": round(wall, 1), "usage": cli.usage,
        "row_auc_jev": round(auc(risks, wrong), 4),
        "row_coverage_jev": coverage_table(risks, wrong),
        "slot_auc_jev": round(auc(slot_scores, slot_labels), 4),
        "n_slot_scored": len(slot_scores),
    }

    by_param = {}
    for s, y, k in zip(slot_scores, slot_labels, slot_names):
        by_param.setdefault(k, [[], []])
        by_param[k][0].append(s)
        by_param[k][1].append(y)
    report["slot_auc_by_param"] = {
        k: {"n": len(v[0]), "err_rate": round(float(np.mean(v[1])), 4),
            "auc": round(auc(v[0], v[1]), 4)}
        for k, v in sorted(by_param.items(), key=lambda kv: -len(kv[1][0]))}

    feats_path = (ROOT / args.features) if args.features else \
        rows_path.parent / (rows_path.stem.replace("_rows", "") + "_features.jsonl")
    if feats_path.exists():
        fl = {f["id"]: f for f in (json.loads(x) for x in open(feats_path) if x.strip())}
        ml = np.array([fl[r["id"]].get("min_logprob")
                       if r["id"] in fl and fl[r["id"]].get("min_logprob") is not None
                       else -99.0 for r, _, _ in kept])
        report["row_auc_min_logprob"] = round(auc(-ml, wrong), 4)
        report["row_auc_jev_plus_min_logprob_rankavg"] = round(
            auc(rank_avg(risks, -ml), wrong), 4)
        report["features_file"] = str(feats_path)

    base_path = rows_path.parent / (rows_path.stem.replace("_rows", "") + ".json")
    if base_path.exists():
        base = json.loads(base_path.read_text())
        report["baseline_signals_detect_uncertainty"] = {
            k: v.get("auc") for k, v in (base.get("signals") or {}).items()}

    tag = args.tag or ("jev_verify_" + rows_path.stem.replace("_rows", ""))
    (ROOT / "eval" / "results" / f"{tag}_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n_rows", "n_wrong", "row_auc_jev", "slot_auc_jev",
                       "row_auc_min_logprob", "row_auc_jev_plus_min_logprob_rankavg",
                       "baseline_signals_detect_uncertainty", "usage")
                      if k in report}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
