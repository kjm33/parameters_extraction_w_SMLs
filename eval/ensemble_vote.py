"""Ensemble of the h90 trio on the test set: majority vote per slot + disagreement
as a failure signal.

Loads <c>-h90-test_rows.jsonl for llama-3.2-1b / qwen2.5-0.5b / qwen3-0.6b, joins by
id (targets come from data/core_test.jsonl). Per slot: cluster the three predicted
values by norm_val equality (null = own value), take the majority cluster; ties ->
null. Reports: voted row EM / error rate vs each member, AUC of disagreement
(fraction of slots lacking unanimity, fallback pairwise row disagreement) against
voted-row errors, and the same for combined disagreement + min logprob rank.

Usage: python eval/ensemble_vote.py  ->  eval/results/ensemble_trio.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_extract import norm_val  # noqa: E402

RES = ROOT / "eval" / "results"
MEMBERS = ["llama-3.2-1b", "qwen2.5-0.5b", "qwen3-0.6b"]


def auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def norm(v):
    if v is None:
        return "<null>"
    if isinstance(v, list):  # norm_val keeps lists -> make hashable string
        return "|".join(sorted(str(norm_val(x)) if x is not None else "<null>" for x in v))
    return str(norm_val(v))


def main():
    targets = {}
    with open(ROOT / "data" / "core_test.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                targets[d["id"]] = d["target"]

    preds, minlp = {}, {}
    for c in MEMBERS:
        with open(RES / f"{c}-h90-test_rows.jsonl") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    preds.setdefault(d["id"], {})[c] = d.get("pred") or {}
                    minlp.setdefault(d["id"], {})[c] = d.get("min_logprob")

    rows = []
    for rid, per in preds.items():
        if rid not in targets or len(per) != len(MEMBERS):
            continue
        keys = set()
        for p in per.values():
            keys |= set(p.keys())
        voted, dis_slots, n_slots = {}, 0, 0
        for k in keys:
            vals = [norm(per[c].get(k)) for c in MEMBERS]
            uniq = sorted(set(vals))
            counts = [vals.count(u) for u in uniq]
            top = max(counts)
            if top >= 2:
                voted[k] = None if uniq[counts.index(top)] == "<null>" else \
                    per[MEMBERS[vals.index(uniq[counts.index(top)])]].get(k)
            else:
                voted[k] = None  # 3-way tie -> abstain to null
            n_slots += 1
            if top != 3:
                dis_slots += 1
        tgt = targets[rid]
        ok = set(voted.keys()) == set(tgt.keys()) and all(
            norm(voted.get(k)) == norm(v) for k, v in tgt.items())
        rows.append({"id": rid, "em": ok, "dis_frac": dis_slots / max(1, n_slots),
                     "min_minlp": min(minlp[rid].values()),
                     "mean_minlp": float(np.mean(list(minlp[rid].values())))})

    y = np.array([0 if r["em"] else 1 for r in rows])
    em = float(np.mean([r["em"] for r in rows]))
    dis = np.array([r["dis_frac"] for r in rows])
    mlp = -np.array([r["mean_minlp"] for r in rows])
    mmin = -np.array([r["min_minlp"] for r in rows])
    rank = (dis.argsort().argsort() + mlp.argsort().argsort()) / 2.0

    member_em = {}
    for c in MEMBERS:
        with open(RES / f"{c}-h90-test_rows.jsonl") as f:
            member_em[c] = round(float(np.mean([
                bool(json.loads(l).get("correct")) for l in f if l.strip()])), 4)

    out = {"n": len(rows), "voted_row_em": round(em, 4),
           "member_row_em": member_em,
           "auc": {"slot_disagreement": round(auc(y, dis), 4),
                   "mean_min_logprob": round(auc(y, mlp), 4),
                   "min_min_logprob": round(auc(y, mmin), 4),
                   "rank_combo_dis+logprob": round(auc(y, rank), 4)}}
    p = RES / "ensemble_trio.json"
    p.write_text(json.dumps(out, indent=1))
    (RES / "ensemble_trio_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"-> {p}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
