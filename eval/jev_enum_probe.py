"""Probe: Jev decision model (OpenRouter) as a DIRECT extractor for enum slots.

For every row, one Jev call fans out typed questions per enum parameter:
  <k>__present      noul   "utterance explicitly specifies <k>"
  <k>__value        choice which enum value (single-select params)
  <k>__has__<v>     noul   per-value membership (multi-select/array params)
Prediction: null if present<0.5, else choice pick / values with p>=0.5 (argmax if
none pass). Scores enum slots only (slot P/R/F1, null accuracy, row EM), per
language (multilingual risk check) and per parameter, and compares against the
fine-tuned local model's preds on the same rows (joined by id from a
detect_uncertainty-style *_rows.jsonl dump).

Outputs: eval/results/<tag>.json, eval/results/<tag>_rows.jsonl

Usage:
  python3 eval/jev_enum_probe.py --files data/core_val.jsonl --tag jev_enum_val
  python3 eval/jev_enum_probe.py --limit 20 --mock     # offline plumbing test
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from eval_extract import norm_val  # noqa: E402
from jev_client import JevClient  # noqa: E402


def enum_of(ps: dict):
    ps = ps or {}
    return ps.get("enum") or (ps.get("items") or {}).get("enum")


def is_multi(ps: dict) -> bool:
    ps = ps or {}
    return bool((ps.get("items") or {}).get("enum")) or ps.get("type") == "array"


def props_of(row: dict) -> dict:
    params = row["function"].get("parameters") or {}
    return params.get("properties") or params or {}


def build_task(row: dict):
    enum_slots = {k: ps for k, ps in props_of(row).items() if enum_of(ps)}
    if not enum_slots:
        return None
    params = {}
    questions = {}
    for k, ps in enum_slots.items():
        vals = enum_of(ps)
        multi = is_multi(ps)
        desc = (ps or {}).get("description") or k
        params[k] = {"description": desc,
                     "selection": "list of values" if multi else "one value",
                     "allowed_values": vals}
        questions[f"{k}__present"] = {
            "type": "noul",
            "instructions": (
                f"The user's utterance explicitly specifies a value for the parameter "
                f"'{k}' ({desc}). Only explicit user intent counts, not values the "
                f"system could pick on its own. Allowed values: {vals}.")}
        if multi:
            for v in vals:
                questions[f"{k}__has__{v}"] = {
                    "type": "noul",
                    "instructions": (
                        f"The user's utterance indicates that the parameter '{k}' "
                        f"({desc}) should include the value '{v}'.")}
        else:
            questions[f"{k}__value"] = {
                "type": "choice",
                "instructions": (
                    f"Which value of the parameter '{k}' ({desc}) does the user's "
                    f"utterance specify? If unclear, pick the most likely."),
                "criteria": {str(v): str(v) for v in vals}}
    state = {"function": row["function"]["name"],
             "function_description": row["function"].get("description", ""),
             "parameters": params,
             "user_utterance": row["utterance"]}
    return enum_slots, state, questions


def jev_predict(enum_slots: dict, answers: dict) -> dict:
    pred = {}
    for k, ps in enum_slots.items():
        vals = enum_of(ps)
        multi = is_multi(ps)
        pres = (answers.get(f"{k}__present") or {}).get("noul")
        if pres is None:
            pred[k] = None
            continue
        if pres < 0.5:
            pred[k] = None
            continue
        if multi:
            scored = []
            for v in vals:
                p = (answers.get(f"{k}__has__{v}") or {}).get("noul")
                if p is not None:
                    scored.append((p, v))
            pred[k] = [v for p, v in scored if p >= 0.5] or \
                [max(scored)[1]] if scored else None
        else:
            pred[k] = (answers.get(f"{k}__value") or {}).get("choice")
    return pred


def agg():
    return {"n": 0, "tp": 0, "fp": 0, "fn": 0, "null_gold": 0, "null_correct": 0,
            "row_em": 0}


def add(a: dict, target: dict, pred: dict):
    a["n"] += 1
    ok = True
    for k, gv in target.items():
        pv = (pred or {}).get(k)
        gvn, pvn = norm_val(gv), norm_val(pv)
        if gvn is None:
            a["null_gold"] += 1
            a["null_correct"] += int(pvn is None)
            if pvn is not None:
                a["fp"] += 1
                ok = False
        elif pvn is not None and pvn == gvn:
            a["tp"] += 1
        else:
            a["fn"] += 1
            if pvn is not None:
                a["fp"] += 1
            ok = False
    a["row_em"] += int(ok)


def metrics(a: dict) -> dict:
    prec = a["tp"] / max(1, a["tp"] + a["fp"])
    rec = a["tp"] / max(1, a["tp"] + a["fn"])
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n": a["n"], "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
            "slot_f1": round(f1, 4),
            "null_accuracy": round(a["null_correct"] / max(1, a["null_gold"]), 4),
            "row_em": round(a["row_em"] / max(1, a["n"]), 4)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", default="data/core_val.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default="typesafe/jev-1.13")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--mock", action="store_true", help="offline plumbing test")
    p.add_argument("--compare-rows", default=None,
                   help="*_rows.jsonl with local model preds (auto-detected if omitted)")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(ROOT / args.files) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tasks, kept = [], []
    for row in rows:
        bt = build_task(row)
        if bt:
            enum_slots, state, questions = bt
            tasks.append((state, questions))
            kept.append((row, enum_slots))
    print(f"rows={len(rows)} with_enum_slots={len(kept)} model={args.model} "
          f"mock={args.mock}")

    cli = JevClient(model=args.model, mock=args.mock, workers=args.workers)
    t0 = time.time()
    results = cli.run_many(tasks)
    wall = time.time() - t0

    local = {}
    cmp_src = None
    if args.compare_rows:
        cand = [ROOT / args.compare_rows]
    else:
        stem = pathlib.Path(args.files).stem
        suffix = {"core_val": "val", "core_test": "test"}.get(stem, stem)
        cand = [ROOT / "eval" / "results" / f"{c}-{suffix}_rows.jsonl"
                for c in ("llama-3.2-1b-h90", "llama-1b-merged")]
    for path in cand:
        if path.exists():
            dump = [json.loads(l) for l in open(path) if l.strip()]
            ids = {d["id"] for d in dump} & {r["id"] for r, _ in kept}
            if ids and len(ids) >= 0.8 * len(kept):
                local = {d["id"]: d.get("pred") for d in dump}
                cmp_src = str(path)
                break
    if cmp_src:
        print(f"comparing against local preds from {cmp_src} ({len(local)} rows)")

    jeu, by_lang, by_param, loc = agg(), {}, {}, agg()
    out_rows = []
    n_fail = 0
    for (row, enum_slots), res in zip(kept, results):
        answers = (res or {}).get("answers") or {}
        if not res:
            n_fail += 1
        target = {k: row["target"][k] for k in enum_slots if k in row["target"]}
        pred = jev_predict(enum_slots, answers)
        add(jeu, target, pred)
        by_lang.setdefault(row["lang"], agg())
        add(by_lang[row["lang"]], target, pred)
        for k in target:
            key = (k, "multi" if is_multi(enum_slots[k]) else "single")
            by_param.setdefault(key, agg())
            add(by_param[key], {k: target[k]}, {k: pred.get(k)})
        lp = local.get(row["id"])
        if lp is not None:
            add(loc, target, lp)
        out_rows.append({
            "id": row["id"], "lang": row["lang"], "function": row["function"]["name"],
            "enum_target": target, "jev_pred": pred,
            "local_pred": {k: lp.get(k) for k in target} if lp is not None else None,
            "jev_probs": {k: {
                "present": (answers.get(f"{k}__present") or {}).get("noul"),
                "value": (answers.get(f"{k}__value") or {})}
                for k in enum_slots},
        })

    report = {
        "model": args.model, "files": args.files, "n_rows": len(kept),
        "n_calls_failed": n_fail, "mock": cli.mock,
        "wall_seconds": round(wall, 1),
        "usage": cli.usage, "compare_src": cmp_src,
        "jev_enum_overall": metrics(jeu),
        "jev_enum_by_lang": {k: metrics(v) for k, v in sorted(by_lang.items())},
        "jev_enum_by_param": {f"{k}[{sel}]": metrics(v) for (k, sel), v in
                              sorted(by_param.items(), key=lambda kv: -kv[1]["n"])},
        "local_enum_on_same_rows": metrics(loc) if cmp_src else None,
    }
    tag = args.tag or f"jev_enum_{pathlib.Path(args.files).stem}"
    (ROOT / "eval" / "results" / f"{tag}_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n_rows", "jev_enum_overall", "jev_enum_by_lang",
                       "local_enum_on_same_rows", "usage")}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
