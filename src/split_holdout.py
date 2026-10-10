"""Split core_train into 90% (h90) + 10% detector-labeled holdout.

Holdout rows are NEVER trained on by the -h90 ensemble members; natural errors on them
provide detector training labels (row EM + per-slot). Stratified by lang x function,
fixed seed. Test set untouched.

Usage: python3 src/split_holdout.py [--frac 0.1] [--seed 1234]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    src = ROOT / "data" / "core_train.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    # group rows by dialog id (one id can yield several rows) — never split a dialog
    rng = random.Random(a.seed)
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    seen: dict[str, tuple] = {}
    for r in rows:
        key = (r["lang"], r["function"]["name"])
        if r["id"] not in seen:
            seen[r["id"]] = key
            groups[key].append(r["id"])

    hold_ids = set()
    for key, ids in sorted(groups.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        k = round(len(ids) * a.frac) if len(ids) >= 2 else (1 if len(ids) > 8 else 0)
        hold_ids.update(ids[:k])

    h90 = [r for r in rows if r["id"] not in hold_ids]
    hold = [r for r in rows if r["id"] in hold_ids]

    # sanity: disjoint ids
    ids90 = {r["id"] for r in h90}
    idsH = {r["id"] for r in hold}
    assert not (ids90 & idsH)
    test_ids = {json.loads(l)["id"] for l in (ROOT / "data" / "core_test.jsonl").read_text().splitlines() if l.strip()}
    assert not (idsH & test_ids), "holdout/test id overlap"

    (ROOT / "data" / "core_train_h90.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in h90))
    (ROOT / "data" / "core_train_holdout.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hold))

    langs = collections.Counter(r["lang"] for r in hold)
    fns = collections.Counter(r["function"]["name"] for r in hold)
    print(f"total={len(rows)} h90={len(h90)} holdout={len(hold)}")
    print(f"holdout langs: {dict(sorted(langs.items()))}")
    print(f"holdout top fns: {dict(fns.most_common(8))}")


if __name__ == "__main__":
    main()
