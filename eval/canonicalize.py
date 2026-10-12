"""Canonicalize free-form slot values: gazetteer post-processor + dump evaluator.

Builds a param-aware gazetteer from --train targets: an alias majority vote
(utterance surface variant -> canonical value), a city->country-code majority
vote, and a per-param canonical value set used by a Polish inflection fallback.
Aliases are guarded: targets with sentence punctuation or < 3 votes are dropped,
and an alias fires only on rows whose lang votes for it (>= 1 vote, strict
plurality membership). Code appends fire only with >= 3 with-code votes forming
>= 0.75 of the value's with-code + bare uses.
Applies the post-processor to a prediction dump (eval/results/<tag>_preds.jsonl)
and re-scores with the exact eval_extract/eval_tagger metrics, before vs after.
Alternative --ceiling mode round-trips decoded BIO tags through the same
post-processor to measure how much of the decode gap it can close.

Usage:
  python3 eval/canonicalize.py --dump eval/results/bert-tagger-ft_preds.jsonl
  python3 eval/canonicalize.py --ceiling data/core_test_tags.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from eval_extract import norm_val  # noqa: E402
from build_tags import (EDGE_PUNCT, _contiguous, decode_tags, norm_eq,  # noqa: E402
                        props_of, vparts, wfold)
from build_extraction import fold  # noqa: E402

CITY_RE = re.compile(r",\s*([A-Za-z]{2})$")
SENT_PUNCT_TAIL = re.compile(r"[.,;:!?…]+$")
SENT_PUNCT_HEAD = re.compile(r"^[.,;:!?…]+")
ALIAS_MIN_VOTES = 3
CODE_MIN_VOTES = 3
CODE_MIN_SHARE = 0.75
PL_SUFFIXES = ("owie", "owa", "ego", "iem", "em", "ie", "ę", "ą", "u", "y", "a", "e", "o")


def is_enum(props: dict, k: str) -> bool:
    ps = props.get(k) or {}
    return "enum" in ps or "enum" in (ps.get("items") or {})


def dirty_punct(s: str) -> bool:
    s = s.strip()
    return bool(SENT_PUNCT_TAIL.search(s) or SENT_PUNCT_HEAD.match(s))


def rpath(part: str) -> pathlib.Path:
    p = pathlib.Path(part)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def resolve(counts: dict) -> tuple[dict, int]:
    out, dropped = {}, 0
    for key, c in counts.items():
        top = c.most_common(2)
        if not top or (len(top) > 1 and top[0][1] == top[1][1]):
            dropped += 1
            continue
        out[key] = top[0][0]
    return out, dropped


def gaz_summary(gaz: dict) -> dict:
    return {"aliases": len(gaz["aliases"]), "city2code": len(gaz["city2code"]),
            "alias_dropped_ambiguous": gaz["alias_dropped"],
            "alias_dropped_punct": gaz["alias_dropped_punct"],
            "alias_dropped_rare": gaz["alias_dropped_rare"],
            "city2code_dropped_weak": gaz["code_dropped_weak"]}


def build_gazetteer(files: list[str]) -> dict:
    alias_counts, code_counts = defaultdict(Counter), defaultdict(Counter)
    alias_langs: dict = defaultdict(Counter)
    bare_counts: dict = defaultdict(int)
    param_values: dict[str, dict] = {}
    for part in files:
        for row in read_jsonl(rpath(part)):
            words = row["utterance"].split()
            wfolds = [wfold(w) for w in words]
            props = props_of(row)
            lang = row.get("lang")
            for k, v in row["target"].items():
                if v is None or not isinstance(v, str) or k == "lang" or is_enum(props, k):
                    continue
                param_values.setdefault(k, {})[v] = True
                m = CITY_RE.search(v)
                if m:
                    city = v[:m.start()].strip()
                    code_counts[(k, fold(city))][m.group(1).upper()] += 1
                elif fold(v):
                    bare_counts[(k, fold(v))] += 1
                hit = _contiguous(vparts(v), wfolds)
                if hit:
                    idxs, _cls = hit
                    surface = EDGE_PUNCT.sub("", " ".join(words[i] for i in idxs))
                    if surface and fold(surface) != fold(v):
                        alias_counts[(k, fold(surface))][v] += 1
                        alias_langs[(k, fold(surface))][lang] += 1
    aliases, alias_dropped = resolve(alias_counts)
    kept_aliases: dict = {}
    dropped_punct = dropped_rare = 0
    for key, target in aliases.items():
        if dirty_punct(target):
            dropped_punct += 1
        elif sum(alias_counts[key].values()) < ALIAS_MIN_VOTES:
            dropped_rare += 1
        else:
            kept_aliases[key] = target
    city2code, code_dropped = resolve(code_counts)
    code_votes = {key: sum(c.values()) for key, c in code_counts.items() if key in city2code}
    code_dropped_weak = sum(1 for key, wc in code_votes.items()
                            if wc < CODE_MIN_VOTES
                            or wc < CODE_MIN_SHARE * (wc + bare_counts.get(key, 0)))
    return {"aliases": kept_aliases, "city2code": city2code, "param_values": param_values,
            "alias_langs": alias_langs, "code_votes": code_votes, "bare_votes": bare_counts,
            "alias_dropped": alias_dropped, "alias_dropped_punct": dropped_punct,
            "alias_dropped_rare": dropped_rare, "code_dropped": code_dropped,
            "code_dropped_weak": code_dropped_weak}


def post_value(k, v, lang, props, gaz, counts, examples, row_id, target):
    if is_enum(props, k):
        return v
    if isinstance(v, list):
        return [post_value(k, x, lang, props, gaz, counts, examples, row_id, target)
                if isinstance(x, str) else x for x in v]
    if not isinstance(v, str):
        return v

    def record(rule, before, after):
        counts[rule] += 1
        if len(examples) < 12:
            examples.append({"id": row_id, "param": k, "rule": rule,
                             "before": before, "after": after,
                             "gold": (target or {}).get(k)})

    akey = (k, fold(v))
    lv = gaz["alias_langs"].get(akey)
    if akey in gaz["aliases"] and lv is not None:
        mine = lv.get(lang, 0)
        if mine >= 1 and all(mine >= nv for nv in lv.values()):
            before = v
            v = gaz["aliases"][akey]
            record("alias", before, v)
    elif lang == "pl":
        words = v.split()
        if words and len(words[-1]) >= 5:
            stem = wfold(words[-1])
            for suf in PL_SUFFIXES:
                if stem.endswith(suf) and len(stem) - len(suf) >= 4:
                    core = stem[:-len(suf)]
                    seen: dict[str, str] = {}
                    for c in gaz["param_values"].get(k, {}):
                        lw = wfold(c.split()[-1])
                        if lw.startswith(core) and lw not in seen:
                            seen[lw] = c
                    if len(seen) == 1 and next(iter(seen)) != stem:
                        before = v
                        canon = next(iter(seen.values()))
                        v = " ".join(words[:-1] + [canon.split()[-1]])
                        record("pl_fallback", before, v)
                    break
    ckey = (k, fold(v))
    if ckey in gaz["city2code"] and not CITY_RE.search(v):
        wc = gaz["code_votes"].get(ckey, 0)
        if wc >= CODE_MIN_VOTES and wc >= CODE_MIN_SHARE * (wc + gaz["bare_votes"].get(ckey, 0)):
            before = v
            v = f"{v}, {gaz['city2code'][ckey]}"
            record("city2code", before, v)
    return v


def postprocess(pred, lang, props, gaz, counts, examples, row_id, target):
    return {k: post_value(k, v, lang, props, gaz, counts, examples, row_id, target)
            for k, v in (pred or {}).items()}


def agg():
    return {"n": 0, "parse_ok": 0, "tp": 0, "fp": 0, "fn": 0,
            "null_gold": 0, "null_correct": 0, "row_em": 0}


def add(acc: dict, target: dict, pred):
    acc["n"] += 1
    acc["parse_ok"] += int(pred is not None)
    ok = True
    for k, gv in target.items():
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


def metrics(acc: dict) -> dict:
    prec = acc["tp"] / max(1, acc["tp"] + acc["fp"])
    rec = acc["tp"] / max(1, acc["tp"] + acc["fn"])
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n": acc["n"], "parse_rate": round(acc["parse_ok"] / max(1, acc["n"]), 4),
            "slot_precision": round(prec, 4), "slot_recall": round(rec, 4),
            "slot_f1": round(f1, 4),
            "null_accuracy": round(acc["null_correct"] / max(1, acc["null_gold"]), 4),
            "row_em": round(acc["row_em"] / max(1, acc["n"]), 4),
            "tp": acc["tp"], "fp": acc["fp"], "fn": acc["fn"]}


def run_dump(args, gaz: dict):
    rows = read_jsonl(rpath(args.dump))
    ds: dict = {}
    for part in args.files.split(","):
        for r in read_jsonl(rpath(part)):
            ds[r["id"]] = r
    before, after = agg(), agg()
    counts, examples = Counter(), []
    for d in rows:
        dsrow = ds.get(d.get("id"))
        props = props_of(dsrow) if dsrow else {}
        target = d.get("target") or {}
        pred = d.get("pred")
        add(before, target, pred)
        pred2 = postprocess(pred, d.get("lang"), props, gaz, counts, examples,
                            d.get("id"), target) if pred is not None else None
        add(after, target, pred2)
    result = {"preds": str(args.dump), "train": args.train.split(","),
              "gazetteer": gaz_summary(gaz),
              "before": metrics(before), "after": metrics(after),
              "applied": dict(counts), "changed_examples": examples}
    out = ROOT / "eval" / "results" / f"{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{args.dump}: before F1={result['before']['slot_f1']} EM={result['before']['row_em']} "
          f"-> after F1={result['after']['slot_f1']} EM={result['after']['row_em']} "
          f"applied={dict(counts)}")
    print(f"gazetteer: {result['gazetteer']}")
    print(f"-> {out}")


def run_ceiling(args, gaz: dict):
    join_sep = json.loads((ROOT / "data" / "tags_build_stats.json").read_text()) \
        .get("join_sep", ", ")
    files, counts, examples = [], Counter(), []
    for part in args.ceiling.split(","):
        rows = read_jsonl(rpath(part))
        rt_before = rt_after = 0
        for row in rows:
            words = row["utterance"].split()
            props = props_of(row)
            target = row["target"]
            dec = decode_tags(row["bio_tags"], words, props, join_sep=join_sep)
            out = {k: dec.get(k, None) for k in target}
            if all(norm_eq(out[k], v) for k, v in target.items()):
                rt_before += 1
            out2 = postprocess(out, row.get("lang"), props, gaz, counts, examples,
                               row.get("id"), target)
            if all(norm_eq(out2[k], v) for k, v in target.items()):
                rt_after += 1
        n = max(1, len(rows))
        files.append({"file": part, "rows": len(rows),
                      "roundtrip_before": round(rt_before / n, 4),
                      "roundtrip_after": round(rt_after / n, 4)})
    result = {"ceiling": args.ceiling.split(","), "join_sep": join_sep,
              "gazetteer": gaz_summary(gaz), "files": files,
              "applied": dict(counts), "changed_examples": examples}
    out = ROOT / "eval" / "results" / "canonicalize_ceiling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for f in files:
        print(f"{f['file']}: rows={f['rows']} roundtrip {f['roundtrip_before']} "
              f"-> {f['roundtrip_after']}")
    print(f"gazetteer: {result['gazetteer']} applied={dict(counts)}")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None,
                    help="eval/results/<tag>_preds.jsonl to post-process and re-score")
    ap.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl",
                    help="comma-separated files to build the gazetteer from")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="dataset files joined by id for props_of (enum detection)")
    ap.add_argument("--out-tag", default=None,
                    help="result tag (default canonicalize_<dump tag>)")
    ap.add_argument("--ceiling", default=None,
                    help="comma-separated <name>_tags.jsonl files for roundtrip ceiling mode")
    args = ap.parse_args()
    if not args.dump and not args.ceiling:
        ap.error("--dump or --ceiling is required")
    gaz = build_gazetteer(args.train.split(","))
    if args.ceiling:
        run_ceiling(args, gaz)
    else:
        name = pathlib.Path(args.dump).name
        tag = name[:-len("_preds.jsonl")] if name.endswith("_preds.jsonl") else name
        args.out_tag = args.out_tag or f"canonicalize_{tag}"
        run_dump(args, gaz)


if __name__ == "__main__":
    main()
