"""Gold-v3 bake-off report: teacher quality vs v2 gold + manual spot-check sheet.

Reads eval/results/goldv3/*_<tag>_preds.jsonl, joins v2 gold by id from the
source files, and reports per teacher: grounded/parse rate, row-EM vs v2,
slot-level diff counts (null->value / value->null / value!=value) tagged by
noise class (appendage_code, word_comma_tail, q_boundary, blank_receipt,
pl_canonical), plus pairwise inter-teacher row EM. With --spotcheck N writes a
lang-stratified manual review sheet (pl rows >= 12, diff-prioritized) to
efforts/2026-10-13-gold-v3/2026-10-13-gold-v3-spotcheck.md.

Usage:
  python3 eval/gold_v3_report.py --tag r12a --spotcheck 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import fold, tok_match  # noqa: E402
from build_gold_v3 import SEED, allocate  # noqa: E402

SRC_FILES = ["core_train", "core_val", "core_test", "aug_train", "aug_val"]
CODE_RE = re.compile(r",\s*[A-Za-z]{2}$")
WORDCOMMA_RE = re.compile(r",\s+\S+$")


def load_teachers(tag: str) -> dict:
    d = ROOT / "eval" / "results" / "goldv3"
    teachers = {}
    for p in sorted(d.glob(f"*_{tag}_preds.jsonl")):
        short = p.name[: -len(f"_{tag}_preds.jsonl")]
        teachers[short] = [json.loads(l) for l in p.open() if l.strip()]
    if not teachers:
        raise SystemExit(f"no eval/results/goldv3/*_{tag}_preds.jsonl found")
    return teachers


def load_gold() -> dict:
    gold = {}
    for f in SRC_FILES:
        for line in (ROOT / "data" / f"{f}.jsonl").open():
            if line.strip():
                r = json.loads(line)
                gold[r["id"]] = r
    return gold


def diff_classes(slot: str, gv, tv, lang) -> list:
    classes = []
    texts = [str(x) for x in (gv, tv) if x is not None]
    if any(CODE_RE.search(t) for t in texts):
        classes.append("appendage_code")
    if any(WORDCOMMA_RE.search(t) and not CODE_RE.search(t) for t in texts):
        classes.append("word_comma_tail")
    if slot == "q" and any(len(t.split()) >= 3 for t in texts):
        classes.append("q_boundary")
    if "avoid" in slot and gv is None and tv is not None:
        classes.append("blank_receipt")
    if lang == "pl" and gv is not None and tv is not None and str(gv) != str(tv) \
            and tok_match(fold(str(gv)), fold(str(tv))):
        classes.append("pl_canonical")
    return classes


def teacher_stats(short: str, rows: list, gold: dict) -> dict:
    n = len(rows)
    n_err = sum(1 for r in rows if any(str(x).startswith("error:") for x in r["fail_reasons"]))
    n_skip = sum(1 for r in rows if "skipped_budget" in r["fail_reasons"])
    called = max(1, n - n_err - n_skip)
    n_parse_fail = sum(1 for r in rows if r["fail_reasons"][:1] == ["parse_fail"])
    n_grounded = sum(1 for r in rows if r["grounded"])
    em = ngr = 0
    diffs = {"null_to_value": 0, "value_to_null": 0, "value_neq_value": 0}
    classes = {}
    for r in rows:
        g = gold.get(r["id"])
        if not g:
            continue
        gv = g["target"]
        tv = r.get("teacher_target") if r["grounded"] else None
        if r["grounded"]:
            ngr += 1
            if tv == gv:
                em += 1
                continue
        elif tv is None:
            continue
        for k, gvv in gv.items():
            tvv = (tv or {}).get(k)
            if gvv is None and tvv is None:
                continue
            if gvv is None and tvv is not None:
                diffs["null_to_value"] += 1
            elif gvv is not None and tvv is None:
                diffs["value_to_null"] += 1
            elif tvv != gvv:
                diffs["value_neq_value"] += 1
            else:
                continue
            for c in diff_classes(k, gvv, tvv, r.get("lang")):
                classes[c] = classes.get(c, 0) + 1
    return {"n": n, "called": n - n_err - n_skip,
            "parse_rate": round(1 - n_parse_fail / called, 4),
            "grounded_rate": round(n_grounded / max(1, n), 4),
            "em_rate": round(em / max(1, n), 4),
            "em_rate_grounded": round(em / max(1, ngr), 4),
            "diffs": diffs,
            "diff_total": sum(diffs.values()),
            "diff_classes": dict(sorted(classes.items(), key=lambda kv: -kv[1]))}


def pairwise(teachers: dict) -> dict:
    out = {}
    for a, b in itertools.combinations(sorted(teachers), 2):
        ra = {r["id"]: r for r in teachers[a]}
        rb = {r["id"]: r for r in teachers[b]}
        common = [i for i in ra if i in rb and ra[i]["grounded"] and rb[i]["grounded"]]
        same = sum(1 for i in common if ra[i]["teacher_target"] == rb[i]["teacher_target"])
        out[f"{a}~{b}"] = {"n": len(common),
                           "em": round(same / len(common), 4) if common else None}
    return out


def alloc_langs(sizes: dict, n: int, min_pl: int = 12) -> dict:
    alloc = allocate(sizes, n)
    need = min(min_pl, sizes.get("pl", 0)) - alloc.get("pl", 0)
    if need > 0:
        alloc["pl"] = alloc.get("pl", 0) + need
        while need > 0:
            donors = sorted([l for l in sizes if l != "pl" and alloc[l] > 1],
                            key=lambda l: (-alloc[l], l))
            if not donors:
                break
            alloc[donors[0]] -= 1
            need -= 1
    return alloc


def spotcheck(teachers: dict, gold: dict, n: int, tag: str) -> tuple:
    shorts = sorted(teachers)
    per_id = {}
    for s in shorts:
        for r in teachers[s]:
            per_id.setdefault(r["id"], {})[s] = r
    cands = []
    for i, per in per_id.items():
        if len(per) < len(shorts) or i not in gold:
            continue
        g = gold[i]
        gv = g["target"]
        diff_v2 = any(per[s]["grounded"] and per[s]["teacher_target"] != gv for s in shorts)
        tvals = {json.dumps(per[s].get("teacher_target"), sort_keys=True, ensure_ascii=False)
                 for s in shorts}
        cands.append({"id": i, "lang": g.get("lang") or "?",
                      "file": next(iter(per.values())).get("file", "?"),
                      "gold": g, "per": per,
                      "prio": 0 if diff_v2 else (1 if len(tvals) > 1 else 2)})
    by_lang = {}
    for c in cands:
        by_lang.setdefault(c["lang"], []).append(c)
    langs = sorted(by_lang)
    alloc = alloc_langs({l: len(by_lang[l]) for l in langs}, n)
    rng = random.Random(SEED)
    picked = []
    for l in langs:
        groups = {0: [], 1: [], 2: []}
        for c in by_lang[l]:
            groups[c["prio"]].append(c)
        for grp in groups.values():
            rng.shuffle(grp)
        picked.extend((groups[0] + groups[1] + groups[2])[:alloc[l]])
    picked.sort(key=lambda c: (c["lang"], c["id"]))

    labels = ["t1", "t2", "t3"][:len(shorts)] if len(shorts) <= 3 else shorts
    out = [f"# Gold-v3 spot-check ({tag}) — {len(picked)} rows",
           "",
           "Pick N rows stratified by lang (pl >= 12), prioritized: (a) any teacher "
           "differs from v2, (b) teachers disagree, (c) rest. Seed 20261013.",
           f"teachers: " + " ".join(f"{lab}={s}" for lab, s in zip(labels, shorts)),
           "",
           "verdict per column: ok (agree with v2 / correct fix) | bad (wrong) | ?"]
    for k, c in enumerate(picked, 1):
        g = c["gold"]
        props = ((g["function"].get("parameters") or {}).get("properties")) or {}
        params = ", ".join(
            f"{p}:{(props[p] or {}).get('type', '?')}(enum={len((props[p] or {}).get('enum') or [])})"
            for p in props)
        cells = [json.dumps(g["target"], ensure_ascii=False, separators=(",", ":"))]
        cells += [json.dumps(c["per"][s].get("teacher_target"), ensure_ascii=False,
                             separators=(",", ":")) for s in shorts]
        out.append(f"## [{k}] {c['id']} ({c['lang']}, {g['function']['name']}, {c['file']})")
        out.append("")
        out.append(f"utt: {g['utterance']}")
        out.append("")
        out.append(f"params: {params}")
        out.append("")
        out.append("| v2 | " + " | ".join(labels) + " |")
        out.append("|" + "---|" * (len(shorts) + 1))
        out.append("| " + " | ".join(cells) + " |")
        gr = " ".join(f"{lab}={'yes' if c['per'][s]['grounded'] else 'NO'}"
                      for lab, s in zip(labels, shorts))
        if "NO" in gr:
            out.append("")
            out.append(f"grounded: {gr}")
        out.append("")
        out.append("verdict: v2=__ " + " ".join(f"{lab}=__" for lab in labels))
        out.append("")
    path = ROOT / "notes" / "2026-10-13-gold-v3-spotcheck.md"
    path.write_text("\n".join(out))
    return [{"id": c["id"], "lang": c["lang"], "prio": c["prio"]} for c in picked], str(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--spotcheck", type=int, default=0,
                   help="write a manual review sheet with N rows")
    return p.parse_args()


def main():
    args = parse_args()
    teachers = load_teachers(args.tag)
    gold = load_gold()
    report = {"tag": args.tag, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "teachers": {s: teacher_stats(s, rows, gold)
                           for s, rows in sorted(teachers.items())},
              "pairwise": pairwise(teachers)}
    print(f"{'teacher':<16}{'n':>6}{'parse':>8}{'ground':>8}{'EM':>8}{'EM|g':>8}"
          f"{'n2v':>6}{'v2n':>6}{'v!=v':>6}  top diff classes")
    for s, st in report["teachers"].items():
        print(f"{s:<16}{st['n']:>6}{st['parse_rate']:>8.3f}{st['grounded_rate']:>8.3f}"
              f"{st['em_rate']:>8.3f}{st['em_rate_grounded']:>8.3f}"
              f"{st['diffs']['null_to_value']:>6}{st['diffs']['value_to_null']:>6}"
              f"{st['diffs']['value_neq_value']:>6}  {st['diff_classes']}")
    for pair, st in report["pairwise"].items():
        print(f"pairwise {pair}: em={st['em']} (n={st['n']})")
    if args.spotcheck:
        rows, path = spotcheck(teachers, gold, args.spotcheck, args.tag)
        report["spotcheck"] = {"n": len(rows), "path": path,
                               "n_pl": sum(1 for r in rows if r["lang"] == "pl")}
        print(f"spotcheck: {len(rows)} rows "
              f"(pl={report['spotcheck']['n_pl']}) -> {path}")
    out = ROOT / "eval" / "results" / f"gold_v3_{args.tag}_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
