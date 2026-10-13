"""Canonicalize v2: v1 gazetteer post-processor + GeoNames cascade + dump evaluator.

Runs the v1 canonicalize post-processor (eval/canonicalize.py) unchanged, then
applies a GeoNames-based cascade on non-enum string slot values (first fire
wins per value; GeoNames data © GeoNames, CC-BY 4.0,
https://download.geonames.org/export/dump/):

  G1_exact_altname  fold(value) in GeoNames key2canon -> per-row-lang display
                    form (else the name column); skipped on ", XX"-style tails.
                    GATED: fires only when fold(value) is NOT itself a key in
                    alt2gids (the pred is not already a valid GeoNames name-form
                    in any language); protected rewrites are counted in
                    g1_skipped_valid
  G2_pl_lemma       pl rows only: Morfeusz2 lemma of the last word (single-token
                    lemmas only) -> the one GeoNames entry matching it exactly.
                    GATED like G1: skipped when fold(value) is already a valid
                    GeoNames name-form (counted in g2_skipped_valid); no-ops
                    (replacement == value) do not count as fires
  G3_fuzzy          conservative rapidfuzz token_sort fallback (second-best must
                    stay < --g3-second-max, length >= 5, no digits); DISABLED by
                    default (default --g3-cutoff 0), re-enable with --g3-cutoff
  G4_appendage      after G1/G2: append the city's 2-letter admin1 country code
                    only if the train comma-code votes (v1 code_counts) contain
                    >= 1 vote for that city with that code

v1's city2code rule is kept exactly as-is (train-evidence-gated). Enum params,
nulls, list values and search_places category params are never touched.
Degrades gracefully: a missing morfeusz2/rapidfuzz or --gazetteer pickle only
disables the affected rules (stderr warning); v1 always runs.

Usage:
  <venv python> eval/canonicalize_v2.py --dump eval/results/<tag>_preds.jsonl
  <venv python> eval/canonicalize_v2.py --calibrate data/core_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from canonicalize import (CITY_RE, agg, add, build_gazetteer, gaz_summary,  # noqa: E402
                          is_enum, metrics, postprocess, read_jsonl, rpath)
from eval_extract import norm_val  # noqa: E402
from score_no_lang import strip_lang  # noqa: E402
from build_tags import props_of, wfold  # noqa: E402
from build_extraction import fold  # noqa: E402

try:
    import morfeusz2
    _MORF_ERR = None
except Exception as _e:  # pragma: no cover
    morfeusz2 = None
    _MORF_ERR = _e
try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process
    _RF_ERR = None
except Exception as _e:  # pragma: no cover
    rf_fuzz = rf_process = None
    _RF_ERR = _e

DISPLAY_LANGS = ("en", "de", "pl", "fr", "es")

G3_CUTOFF = 0.0  # 0 = G3 disabled
G3_SECOND_MAX = 88.0

_MORF = None


def _morfeusz():
    global _MORF
    if _MORF is None:
        _MORF = morfeusz2.Morfeusz()
    return _MORF


def morfeusz_lemmas(word: str) -> set[str]:
    """Single-token lemma forms only ('z:q:im'-style and spaced lemmas drop out)."""
    out = set()
    for _s, _e, interp in _morfeusz().analyse(word):
        lemma = interp[0]
        if lemma and " " not in lemma and ":" not in lemma:
            out.add(lemma)
    return out


def load_geo(path: pathlib.Path) -> dict | None:
    if not path.exists():
        print(f"WARNING: --gazetteer {path} missing -> GeoNames rules disabled", file=sys.stderr)
        return None
    with open(path, "rb") as f:
        geo = pickle.load(f)
    geo["keys_list"] = list(geo.get("key2canon") or {})
    return geo


def geo_summary(geo: dict | None) -> dict | None:
    if geo is None:
        return None
    m = geo.get("meta") or {}
    kept = m.get("kept") or {}
    return {"cities": kept.get("cities"), "altnames_kept": kept.get("altnames"),
            "unique_keys": len(geo["key2canon"]),
            "dropped_ambiguous": m.get("dropped_ambiguous_folds"),
            "attribution": m.get("attribution")}


def fname_of(row: dict | None) -> str:
    fn = (row or {}).get("function")
    if isinstance(fn, dict):
        return fn.get("name") or ""
    return str(fn or "")


def build_code_counts(files: list[str]) -> dict:
    """Raw v1 code_counts: (param, fold(city)) -> Counter(code) from train targets."""
    counts: dict = defaultdict(Counter)
    for part in files:
        for row in read_jsonl(rpath(part)):
            props = props_of(row)
            for k, v in (row.get("target") or {}).items():
                if v is None or not isinstance(v, str) or k == "lang" or is_enum(props, k):
                    continue
                m = CITY_RE.search(v)
                if m:
                    counts[(k, fold(v[:m.start()].strip()))][m.group(1).upper()] += 1
    return counts


def geo_post_value(k, v, lang, props, fname, geo, code_counts,
                   counts, examples, row_id, target, ex_cap=12, skips=None):
    """Returns (new_value, rule_or_None, matched_key_or_None). G1 -> G2 -> G3 (+ G4)."""
    if geo is None or isinstance(v, list) or not isinstance(v, str):
        return v, None, None
    if is_enum(props, k) or (fname and "search_places" in fname):
        return v, None, None

    def record(rule, before, after):
        counts[rule] += 1
        if len(examples) < ex_cap:
            examples.append({"id": row_id, "param": k, "rule": rule,
                             "before": before, "after": after,
                             "gold": (target or {}).get(k)})

    key2canon = geo["key2canon"]
    fkey = fold(v)
    fired = None
    rule = None

    # G1 exact-altname (gated: skip pred forms that are already a valid
    # GeoNames name-form in some language -> counted as g1_skipped_valid)
    if fkey in key2canon and not CITY_RE.search(v):
        entry = key2canon[fkey]
        disp = entry.get(lang) or entry["default"]
        if fold(disp) != fkey:
            if fkey in geo["alt2gids"]:
                if skips is not None:
                    skips["g1_skipped_valid"] += 1
            else:
                before = v
                v = disp
                record("G1_exact_altname", before, v)
                fired = fkey
                rule = "G1_exact_altname"

    # G2 pl-lemma (gated: never rewrite a valid GeoNames name-form; no-ops
    # (replacement == value) are not counted as fires)
    if fired is None and lang == "pl" and morfeusz2 is not None and "," not in v:
        words = v.split()
        if words and len(words[-1]) >= 5:
            if fkey in geo["alt2gids"]:
                if skips is not None:
                    skips["g2_skipped_valid"] += 1
            else:
                lem_folds = {fold(l) for l in morfeusz_lemmas(wfold(words[-1]))}
                cands = sorted(lf for lf in lem_folds if lf in key2canon)
                if len(cands) == 1:
                    entry = key2canon[cands[0]]
                    disp = entry.get("pl") or entry["default"]
                    newv = " ".join(words[:-1] + [disp])
                    if fold(disp) != fold(v) and newv != v:
                        before = v
                        v = newv
                        record("G2_pl_lemma", before, v)
                        fired = cands[0]
                        rule = "G2_pl_lemma"

    # G3 fuzzy (fallback, conservative; disabled when G3_CUTOFF == 0)
    if (fired is None and rf_process is not None and G3_CUTOFF > 0
            and len(fkey) >= 5 and not any(ch.isdigit() for ch in v)
            and not CITY_RE.search(v)):
        hits = rf_process.extract(fkey, geo["keys_list"], scorer=rf_fuzz.token_sort_ratio,
                                  score_cutoff=G3_CUTOFF, limit=2)
        if hits and (len(hits) == 1 or hits[1][1] < G3_SECOND_MAX):
            entry = key2canon[hits[0][0]]
            disp = entry.get(lang) or entry["default"]
            if fold(disp) != fold(v):
                before = v
                v = disp
                record("G3_fuzzy", before, v)
                fired = hits[0][0]
                rule = "G3_fuzzy"

    # G4 appendage (admin1 country code, train-evidence-gated)
    if fired is not None and not CITY_RE.search(v):
        admin = None
        for gid in geo["alt2gids"].get(fired, ()):
            admin = geo["admin1_of"].get(gid)
            if admin:
                break
        info = geo["admin1_codes"].get(admin) if admin else None
        if info and code_counts.get((k, fold(v)), {}).get(info["code2"], 0) >= 1:
            before = v
            v = f"{v}, {info['code2']}"
            record("G4_appendage", before, v)
            rule = "G4_appendage"

    return v, rule, fired


def geo_postprocess(pred, lang, props, fname, geo, code_counts,
                    counts, examples, row_id, target, ex_cap=12, skips=None):
    if pred is None or geo is None:
        return pred
    out = {}
    for k, v in pred.items():
        out[k], _, _ = geo_post_value(k, v, lang, props, fname, geo, code_counts,
                                      counts, examples, row_id, target, ex_cap,
                                      skips)
    return out


def rules_config() -> dict:
    return {"g1_gated": True,
            "g1_gate": "G1 fires only when fold(v) not in alt2gids "
                       "(valid GeoNames name-forms skipped -> g1_skipped_valid)",
            "g2_gated": True,
            "g2_gate": "G2 skipped when fold(v) in alt2gids "
                       "(valid GeoNames name-forms skipped -> g2_skipped_valid); "
                       "replacement == value is not counted as a fire",
            "g3_cutoff": G3_CUTOFF,
            "g3_second_max": G3_SECOND_MAX,
            "g4_mode": "after_g1_g2_append_admin1_code2_if_ge_1_train_code_vote"}


def run_dump(args, gaz: dict, geo: dict | None, code_counts: dict):
    rows = read_jsonl(rpath(args.dump))
    ds: dict = {}
    for part in args.files.split(","):
        for r in read_jsonl(rpath(part)):
            ds[r["id"]] = r
    all_b, all_a = agg(), agg()
    lang_b, lang_a = agg(), agg()
    v1_b, v1_a = agg(), agg()
    vl_b, vl_a = agg(), agg()
    counts, examples = Counter(), []
    skips = Counter()
    per_lang: dict[str, list] = defaultdict(lambda: [agg(), agg(), agg()])
    for d in rows:
        dsrow = ds.get(d.get("id"))
        props = props_of(dsrow) if dsrow else {}
        fname = fname_of(dsrow) or fname_of(d)
        lang = d.get("lang")
        target = d.get("target") or {}
        pred = d.get("pred")
        pred_v1 = postprocess(pred, lang, props, gaz, counts, examples,
                              d.get("id"), target) if pred is not None else None
        pred2 = geo_postprocess(pred_v1, lang, props, fname, geo, code_counts,
                                counts, examples, d.get("id"), target, skips=skips)
        add(all_b, target, pred)
        add(all_a, target, pred2)
        add(v1_b, target, pred)
        add(v1_a, target, pred_v1)
        tf, pf, pf1 = strip_lang(target), None, None
        if pred is not None:
            pf = strip_lang(pred)
        if pred_v1 is not None:
            pf1 = strip_lang(pred_v1)
        add(lang_b, tf, pf)
        add(lang_a, tf, strip_lang(pred2) if pred2 is not None else None)
        add(vl_b, tf, pf)
        add(vl_a, tf, pf1)
        accs = per_lang[lang if lang in DISPLAY_LANGS else "other"]
        add(accs[0], tf, pf)
        add(accs[1], tf, pf1)
        add(accs[2], tf, strip_lang(pred2) if pred2 is not None else None)
    result = {"preds": str(args.dump), "train": args.train.split(","),
              "gazetteer": gaz_summary(gaz),
              "rules_config": rules_config(),
              "geo": {"pickle": str(args.gazetteer), "available": geo is not None,
                      "morfeusz2": morfeusz2 is not None, "rapidfuzz": rf_process is not None,
                      "g3_cutoff": G3_CUTOFF, "g3_second_max": G3_SECOND_MAX,
                      "summary": geo_summary(geo)},
              "all_slots": {"before": metrics(all_b), "after": metrics(all_a)},
              "lang_free": {"before": metrics(lang_b), "after": metrics(lang_a)},
              "v1_only": {"all_slots": {"before": metrics(v1_b), "after": metrics(v1_a)},
                          "lang_free": {"before": metrics(vl_b), "after": metrics(vl_a)}},
              "per_lang_f1": {lg: {"before": metrics(accs[0])["slot_f1"],
                                   "v1_only": metrics(accs[1])["slot_f1"],
                                   "after": metrics(accs[2])["slot_f1"]}
                              for lg, accs in sorted(per_lang.items())},
              "g1_skipped_valid": skips["g1_skipped_valid"],
              "g2_skipped_valid": skips["g2_skipped_valid"],
              "applied": dict(counts), "changed_examples": examples}
    out = ROOT / "eval" / "results" / f"{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    a, l = result["all_slots"], result["lang_free"]
    print(f"{pathlib.Path(args.dump).name}: "
          f"all_slots F1={a['before']['slot_f1']} EM={a['before']['row_em']} "
          f"-> F1={a['after']['slot_f1']} EM={a['after']['row_em']} | "
          f"lang_free F1={l['before']['slot_f1']} EM={l['before']['row_em']} "
          f"-> F1={l['after']['slot_f1']} EM={l['after']['row_em']}")
    print(f"applied={dict(counts)} g1_skipped_valid={skips['g1_skipped_valid']} "
          f"g2_skipped_valid={skips['g2_skipped_valid']}")
    print("per_lang lang_free F1 (v1_only->after, delta vs v1_only) " +
          " | ".join(f"{lg} {v['v1_only']}->{v['after']} ({round(v['after'] - v['v1_only'], 4)})"
                     for lg, v in result["per_lang_f1"].items() if lg in DISPLAY_LANGS))
    print(f"-> {out}")


def run_calibrate(args, geo: dict | None, code_counts: dict):
    fires, fires_exact, fires_norm, skips = Counter(), Counter(), Counter(), Counter()
    examples = []
    total = corrupt_exact = corrupt_norm = 0
    corrupt_examples = []
    for part in args.calibrate.split(","):
        for row in read_jsonl(rpath(part)):
            props = props_of(row)
            fname = fname_of(row)
            lang = row.get("lang")
            target = row.get("target") or {}
            for k, v in target.items():
                if v is None or not isinstance(v, str) or k == "lang" or is_enum(props, k):
                    continue
                total += 1
                nv, rule, _key = geo_post_value(k, v, lang, props, fname, geo, code_counts,
                                                fires, examples, row.get("id"), target,
                                                ex_cap=0, skips=skips)
                if nv != v:
                    exact = nv != target.get(k)
                    norm = norm_val(nv) != norm_val(target.get(k))
                    if exact:
                        fires_exact[rule] += 1
                    if norm:
                        fires_norm[rule] += 1
                    if exact:
                        corrupt_exact += 1
                        if len(corrupt_examples) < 20:
                            corrupt_examples.append({"id": row.get("id"), "param": k,
                                                     "rule": rule, "before": v,
                                                     "after": nv, "gold": target.get(k)})
                    if norm:
                        corrupt_norm += 1
    rate, rate_norm = corrupt_exact / max(1, total), corrupt_norm / max(1, total)
    result = {"calibrate": args.calibrate.split(","),
              "rules_config": rules_config(),
              "geo": {"pickle": str(args.gazetteer), "available": geo is not None,
                      "morfeusz2": morfeusz2 is not None, "rapidfuzz": rf_process is not None,
                      "g3_cutoff": G3_CUTOFF, "g3_second_max": G3_SECOND_MAX,
                      "summary": geo_summary(geo)},
              "g1_skipped_valid": skips["g1_skipped_valid"],
              "g2_skipped_valid": skips["g2_skipped_valid"],
              "values_examined": total, "fires": dict(fires),
              "fire_total": sum(fires.values()),
              "corruption_count": corrupt_exact, "corruption_rate": round(rate, 4),
              "corruption_definition": "fired rewrite != gold (exact string)",
              "corruption_norm_count": corrupt_norm,
              "corruption_norm_rate": round(rate_norm, 4),
              "corruption_norm_definition":
                  "fired rewrite != gold under norm_val (fold+comma; the metric's equality)",
              "fires_exact": dict(fires_exact), "fires_norm": dict(fires_norm),
              "corruption_examples": corrupt_examples}
    out = ROOT / "eval" / "results" / "canonicalize_v2_calibrate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"calibrate: values={total} fires={dict(fires)} "
          f"g1_skipped_valid={skips['g1_skipped_valid']} "
          f"g2_skipped_valid={skips['g2_skipped_valid']} "
          f"corruption_exact={corrupt_exact}/{total} ({rate:.4f}) "
          f"corruption_norm={corrupt_norm}/{total} ({rate_norm:.4f})")
    print(f"fires_exact={dict(fires_exact)} fires_norm={dict(fires_norm)}")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="v1 canonicalize + GeoNames cascade post-processor/re-scorer")
    ap.add_argument("--dump", default=None,
                    help="eval/results/<tag>_preds.jsonl to post-process and re-score")
    ap.add_argument("--calibrate", default=None,
                    help="comma-separated dataset files: fire-count + corruption safety check")
    ap.add_argument("--gazetteer", default="data/geo_gazetteer.pkl",
                    help="GeoNames pickle from src/build_gazetteer.py")
    ap.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl",
                    help="comma-separated files to build the v1 gazetteer from (--dump)")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="dataset files joined by id for props_of (enum detection)")
    ap.add_argument("--out-tag", default=None,
                    help="result tag (default canonicalize_v2_<dump tag>)")
    ap.add_argument("--g3-cutoff", type=float, default=0.0,
                    help="G3 fuzzy best-score cutoff; 0 disables G3 (default 0 = off)")
    ap.add_argument("--g3-second-max", type=float, default=88.0,
                    help="G3 fuzzy second-best must stay below this (default 88)")
    args = ap.parse_args()
    if not args.dump and not args.calibrate:
        ap.error("--dump or --calibrate is required")
    global G3_CUTOFF, G3_SECOND_MAX
    G3_CUTOFF, G3_SECOND_MAX = args.g3_cutoff, args.g3_second_max
    if morfeusz2 is None:
        print(f"WARNING: morfeusz2 unavailable ({_MORF_ERR}) -> G2 disabled", file=sys.stderr)
    if rf_process is None:
        print(f"WARNING: rapidfuzz unavailable ({_RF_ERR}) -> G3 disabled", file=sys.stderr)
    geo = load_geo(rpath(args.gazetteer))
    if args.calibrate:
        run_calibrate(args, geo, build_code_counts(args.calibrate.split(",")))
    else:
        name = pathlib.Path(args.dump).name
        tag = name[:-len("_preds.jsonl")] if name.endswith("_preds.jsonl") else name
        args.out_tag = args.out_tag or f"canonicalize_v2_{tag}"
        run_dump(args, build_gazetteer(args.train.split(",")), geo,
                 build_code_counts(args.train.split(",")))


if __name__ == "__main__":
    main()
