"""Error analysis over eval/results/<tag>_preds.jsonl dumps.

Reports: per-slot FP (hallucinated values) and FN (missed extractions) counts,
near-miss rate (pred is substring/superstring of gold or shares >=60% tokens),
and sample utterances per category.

Usage: python3 eval/analyze_errors.py qwen2.5-0.5b-ft [more tags...]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import unicodedata
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def norm(v):
    if v is None:
        return None
    if isinstance(v, list):
        return sorted(re.sub(r"\s*,\s*", ",", fold(str(x))) for x in v)
    return re.sub(r"\s*,\s*", ",", fold(v))


def near(gv, pv):
    g, p = norm(gv), norm(pv)
    if not isinstance(g, str) or not isinstance(p, str):
        return norm(gv) == norm(pv)
    if g == p:
        return True
    if len(g) >= 3 and (g in p or p in g):
        return True
    gt, pt = set(g.split()), set(p.split())
    return bool(gt and pt and len(gt & pt) / max(len(gt), len(pt)) >= 0.6)


def main():
    tags = sys.argv[1:]
    test = {}
    with open(ROOT / "data" / "core_test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            test[r["id"]] = r
    for tag in tags:
        path = ROOT / "eval" / "results" / f"{tag}_preds.jsonl"
        rows = [dict(json.loads(l), utterance=test[json.loads(l)["id"]]["utterance"])
                for l in open(path)]
        fp_slots, fn_slots, near_fn, fn_samples, fp_samples = Counter(), Counter(), 0, [], []
        for r in rows:
            pred = r["pred"] or {}
            for k, gv in r["target"].items():
                pv = pred.get(k)
                if gv is None:
                    if pv is not None:
                        fp_slots[f"{r['function']}.{k}"] += 1
                        if len(fp_samples) < 12:
                            fp_samples.append((r["function"], k, r["lang"], pv, r["utterance"][:90]))
                elif norm(pv) == norm(gv):
                    pass
                else:
                    fn_slots[f"{r['function']}.{k}"] += 1
                    if near(gv, pv):
                        near_fn += 1
                    if len(fn_samples) < 16:
                        fn_samples.append((r["function"], k, r["lang"], gv, pv, r["utterance"][:90]))
        n_fn = sum(fn_slots.values())
        print(f"\n===== {tag} (n={len(rows)}) =====")
        print(f"hallucinated-value slots (gold null, pred non-null): {sum(fp_slots.values())}")
        for s, c in fp_slots.most_common(10):
            print(f"  {c:4d}  {s}")
        print(f"missed/wrong-value slots: {n_fn}  (near-miss normalization gaps: {near_fn} "
              f"= {near_fn / max(1, n_fn) * 100:.0f}%)")
        for s, c in fn_slots.most_common(12):
            print(f"  {c:4d}  {s}")
        print("--- FN samples (fn, slot, lang, gold, pred, utterance) ---")
        for s in fn_samples[:10]:
            print("  ", s)
        print("--- FP samples (fn, slot, lang, pred, utterance) ---")
        for s in fp_samples[:8]:
            print("  ", s)


if __name__ == "__main__":
    main()
