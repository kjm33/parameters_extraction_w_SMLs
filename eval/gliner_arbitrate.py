"""GLiNER post-hoc span arbitration over a prediction dump + re-scorer.

Consumes a GLiNER span dump (eval/gliner_spans.py output) and a prediction dump
(eval/results/<tag>_preds.jsonl). For every non-null STRING pred value at a
non-enum param the value is located in the utterance by word-fold contiguous
match; the best-overlapping GLiNER span (IoU on word-index sets) then decides:

  agree           IoU == 1.0                          -> no change
  truncate        span words proper-subset of region  -> value := span text
                  (>=1 word, >=3 chars, else truncate_skipped)
  extend          region proper-subset of span words  -> value := span text,
                  ONLY for geocode q/query params (here_geocode, nominatim_geocode,
                  tomtom-geocode); others -> extend_skipped
  partial_skip    0 < IoU < 1                         -> count only
  no_span         no overlapping span at all          -> count only
  not_locatable   value not found in the utterance    -> count only

  fn_repair       geocode fns only: pred[q/query] null AND target non-null AND
                  exactly ONE span with score >= repair-threshold among labels
                  {street address, point of interest, city, location} AND that
                  span not consumed by another param value -> q := span text

Arbitrated preds then go through the v1 canonicalize post-processor and are
re-scored lang-free. Reports before / v1_only / after + per-lang F1 +
geocode q-boundary FP/FN + per-row features jsonl for the detector stack.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python \
    eval/gliner_arbitrate.py --dump eval/results/llama-3.2-1b-v2-ft_preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from canonicalize import (agg, add, build_gazetteer, gaz_summary,  # noqa: E402
                          is_enum, metrics, postprocess, read_jsonl, rpath)
from canonicalize_v2 import fname_of  # noqa: E402
from eval_extract import norm_val  # noqa: E402
from build_tags import EDGE_PUNCT, _contiguous, props_of, vparts, wfold  # noqa: E402
from build_extraction import fold  # noqa: E402
from score_no_lang import strip_lang  # noqa: E402

GEO_FNS = {"here_geocode", "nominatim_geocode", "tomtom-geocode"}
Q_PARAMS = ("q", "query")
REPAIR_LABELS = {"street address", "point of interest", "city", "location"}
DISPLAY_LANGS = ("en", "de", "pl", "fr", "es")

ARBITRATION_RULES = ("agree", "truncate", "extend", "fn_repair")


def word_offsets(utt: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for w in utt.split():
        i = utt.index(w, pos)
        out.append((i, i + len(w)))
        pos = i + len(w)
    return out


def span_words(span: dict, offs: list[tuple[int, int]]) -> frozenset:
    return frozenset(i for i, (ws, we) in enumerate(offs)
                     if ws < span["end"] and we > span["start"])


def qagg() -> dict:
    a = agg()
    a["q_fp"] = 0
    a["q_fn"] = 0
    return a


def add_q(acc: dict, target: dict, pred, fname: str):
    """q-boundary slot accounting (geocode q/query params), mirroring add()."""
    if fname not in GEO_FNS:
        return
    for k in Q_PARAMS:
        if k not in target:
            continue
        gvn, pvn = norm_val(target.get(k)), norm_val((pred or {}).get(k))
        if gvn is None:
            acc["q_fp"] += int(pvn is not None)
        elif pvn == gvn and pvn is not None:
            pass
        else:
            acc["q_fn"] += 1
            acc["q_fp"] += int(pvn is not None)


def arbitrate(pred, dsrow, row_spans, target, props, fname, args,
              counts, examples):
    """Returns arbitrated pred dict. Counts every rule; per-rule examples cap 12."""
    def record(rule, param, before, after, utt, row_id):
        ex = examples[rule]
        if len(ex) < 12:
            ex.append({"id": row_id, "param": param, "rule": rule,
                       "before": before, "after": after,
                       "gold": (target or {}).get(param), "utt": utt[:120]})

    out = dict(pred)
    utt = dsrow["utterance"]
    row_id = dsrow.get("id")
    words = utt.split()
    wfolds = [wfold(w) for w in words]
    offs = word_offsets(utt)
    cand = [s for s in row_spans if s["score"] >= args.span_threshold]
    cwords = [span_words(s, offs) for s in cand]
    consumed: set = set()
    is_geo = fname in GEO_FNS
    n_locatable = n_agree = n_trunc = n_ext = 0

    for k, v in pred.items():
        if k in ("lang", "language") or is_enum(props, k):
            continue
        if isinstance(v, list) or not isinstance(v, str) or not v.strip():
            continue
        hit = _contiguous(vparts(v), wfolds)
        if not hit:
            counts["not_locatable"] += 1
            record("not_locatable", k, v, v, utt, row_id)
            continue
        region = set(hit[0])
        n_locatable += 1
        best_i, best_iou = None, 0.0
        for i, sws in enumerate(cwords):
            inter = len(region & sws)
            if not inter:
                continue
            iou = inter / len(region | sws)
            if iou > best_iou + 1e-9 or (abs(iou - best_iou) <= 1e-9 and best_i is not None
                                         and cand[i]["score"] > cand[best_i]["score"]):
                best_i, best_iou = i, iou
        if best_i is None:
            counts["no_span"] += 1
            record("no_span", k, v, v, utt, row_id)
            continue
        consumed.add((cand[best_i]["start"], cand[best_i]["end"]))
        span = cand[best_i]
        sws = cwords[best_i]
        if best_iou >= 1.0 - 1e-9:
            counts["agree"] += 1
            n_agree += 1
            continue
        text = EDGE_PUNCT.sub("", utt[span["start"]:span["end"]].strip())
        usable = bool(text) and len(text) >= 3 and fold(text) != fold(v)
        if len(sws) < len(region):  # span proper-subset of region -> truncate
            if usable:
                counts["truncate"] += 1
                n_trunc += 1
                record("truncate", k, v, text, utt, row_id)
                out[k] = text
            else:
                counts["truncate_skipped"] += 1
                record("truncate_skipped", k, v, v, utt, row_id)
        elif len(region) < len(sws):  # region proper-subset of span -> extend
            if is_geo and k in Q_PARAMS and usable:
                counts["extend"] += 1
                n_ext += 1
                record("extend", k, v, text, utt, row_id)
                out[k] = text
            else:
                counts["extend_skipped"] += 1
                record("extend_skipped", k, v, v, utt, row_id)
        else:
            counts["partial_skip"] += 1
            record("partial_skip", k, v, v, utt, row_id)

    if args.fn_repair and is_geo:
        for qk in Q_PARAMS:
            if qk not in target or target.get(qk) is None or out.get(qk) is not None:
                continue
            rc = [s for s in row_spans if s["label"] in REPAIR_LABELS
                  and s["score"] >= args.repair_threshold]
            if len(rc) != 1:
                counts[f"fn_repair_skip_{len(rc)}_cands"] += 1
                continue
            key = (rc[0]["start"], rc[0]["end"])
            if key in consumed:
                counts["fn_repair_skip_consumed"] += 1
                continue
            text = EDGE_PUNCT.sub("", utt[key[0]:key[1]].strip())
            if not text or len(text) < 3:
                counts["fn_repair_skip_text"] += 1
                continue
            counts["fn_repair"] += 1
            record("fn_repair", qk, None, text, utt, row_id)
            out[qk] = text
            consumed.add(key)
            break

    feats = {"id": row_id, "n_spans": len(row_spans), "n_locatable": n_locatable,
             "agree": n_agree, "truncate": n_trunc, "extend": n_ext,
             "fn_repair": 0, "any_change": out != pred}
    return out, feats


def change_outcomes(pred, pred_arb, target, outcomes: Counter):
    """Gold-based REPORTING tally of value changes (fix/break/neutral); decisions
    above are gold-free, this only quantifies arbitration quality post-hoc."""
    for k, old in (pred or {}).items():
        if k in ("lang", "language"):
            continue
        new = (pred_arb or {}).get(k)
        if norm_val(old) == norm_val(new):
            continue
        g = norm_val((target or {}).get(k))
        old_ok = norm_val(old) == g
        new_ok = new is not None and norm_val(new) == g
        if new_ok and not old_ok:
            outcomes["fix"] += 1
        elif old_ok:
            outcomes["break"] += 1
        else:
            outcomes["neutral"] += 1


def main():
    ap = argparse.ArgumentParser(
        description="GLiNER span arbitration post-processor + lang-free re-scorer")
    ap.add_argument("--dump", required=True,
                    help="eval/results/<tag>_preds.jsonl to post-process and re-score")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="dataset files joined by id (utterance + props_of)")
    ap.add_argument("--spans", default=None,
                    help="gliner_spans.py jsonl (default data/ner/gliner_<stem>_spans.jsonl)")
    ap.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl",
                    help="comma-separated files to build the v1 gazetteer from")
    ap.add_argument("--span-threshold", type=float, default=0.5,
                    help="min GLiNER score for truncate/extend candidates")
    ap.add_argument("--repair-threshold", type=float, default=0.6,
                    help="min GLiNER score for the fn_repair single span")
    ap.add_argument("--fn-repair", dest="fn_repair", action="store_true", default=True,
                    help="null-q repair from a single confident span (geocode fns only)")
    ap.add_argument("--no-fn-repair", dest="fn_repair", action="store_false")
    ap.add_argument("--out-tag", default=None,
                    help="result tag (default gliner_arb_<dump tag>)")
    args = ap.parse_args()

    parts = args.files.split(",")
    if args.spans:
        spans_path = rpath(args.spans)
    else:
        stems = "_".join(pathlib.Path(p).stem for p in parts)
        spans_path = ROOT / "data" / "ner" / f"gliner_{stems}_spans.jsonl"
    spans_by_id: dict = {}
    if spans_path.exists():
        for r in read_jsonl(spans_path):
            spans_by_id[r["id"]] = r.get("spans") or []
    else:
        print(f"WARNING: spans file {spans_path} missing -> run eval/gliner_spans.py first",
              file=sys.stderr)

    gaz = build_gazetteer(args.train.split(","))
    rows = read_jsonl(rpath(args.dump))
    ds: dict = {}
    for part in parts:
        for r in read_jsonl(rpath(part)):
            ds[r["id"]] = r

    b_all, a_all = qagg(), qagg()
    b_lf, a_lf = qagg(), qagg()
    v1_all, v1_lf = qagg(), qagg()
    arb_counts: Counter = Counter()
    outcomes: Counter = Counter()
    canon_v1: Counter = Counter()
    canon_after: Counter = Counter()
    examples: dict = defaultdict(list)
    per_lang: dict = defaultdict(lambda: [agg(), agg(), agg()])
    feats_rows: list = []
    for d in rows:
        dsrow = ds.get(d.get("id"))
        props = props_of(dsrow) if dsrow else {}
        fname = fname_of(dsrow) or fname_of(d)
        lang = d.get("lang")
        target = d.get("target") or {}
        pred = d.get("pred")
        row_spans = spans_by_id.get(d.get("id")) or []
        if pred is not None and dsrow is not None:
            before_fn = arb_counts["fn_repair"]
            pred_arb, feats = arbitrate(pred, dsrow, row_spans, target, props,
                                        fname, args, arb_counts, examples)
            feats["fn_repair"] = arb_counts["fn_repair"] - before_fn
        else:
            pred_arb, feats = pred, {"id": d.get("id"), "n_spans": len(row_spans),
                                     "n_locatable": 0, "agree": 0, "truncate": 0,
                                     "extend": 0, "fn_repair": 0, "any_change": False}
        feats_rows.append(feats)
        if pred is not None and pred_arb is not None:
            change_outcomes(pred, pred_arb, target, outcomes)
        pred_v1 = postprocess(pred, lang, props, gaz, canon_v1, examples["canon_v1"],
                              d.get("id"), target) if pred is not None else None
        pred_after = postprocess(pred_arb, lang, props, gaz, canon_after,
                                 examples["canon_after"], d.get("id"),
                                 target) if pred_arb is not None else None
        add(b_all, target, pred)
        add(a_all, target, pred_after)
        add(v1_all, target, pred_v1)
        add_q(b_all, target, pred, fname)
        add_q(a_all, target, pred_after, fname)
        add_q(v1_all, target, pred_v1, fname)
        tf = strip_lang(target)
        pf = strip_lang(pred) if pred is not None else None
        pf1 = strip_lang(pred_v1) if pred_v1 is not None else None
        pf2 = strip_lang(pred_after) if pred_after is not None else None
        add(b_lf, tf, pf)
        add(a_lf, tf, pf2)
        add(v1_lf, tf, pf1)
        accs = per_lang[lang if lang in DISPLAY_LANGS else "other"]
        add(accs[0], tf, pf)
        add(accs[1], tf, pf1)
        add(accs[2], tf, pf2)

    name = pathlib.Path(args.dump).name
    tag = name[:-len("_preds.jsonl")] if name.endswith("_preds.jsonl") else name
    args.out_tag = args.out_tag or f"gliner_arb_{tag}"
    result = {"preds": str(args.dump), "spans": str(spans_path),
              "spans_available": bool(spans_by_id), "train": args.train.split(","),
              "gazetteer": gaz_summary(gaz),
              "rules_config": {
                  "span_threshold": args.span_threshold,
                  "repair_threshold": args.repair_threshold,
                  "fn_repair": args.fn_repair,
                  "geo_fns": sorted(GEO_FNS), "q_params": list(Q_PARAMS),
                  "repair_labels": sorted(REPAIR_LABELS),
                  "extend_gate": "extend only for geocode q/query params",
                  "locate": "word-fold contiguous match (_contiguous), no partial"},
              "all_slots": {"before": metrics(b_all), "after": metrics(a_all)},
              "lang_free": {"before": metrics(b_lf), "after": metrics(a_lf)},
              "v1_only": {"all_slots": {"before": metrics(b_all), "after": metrics(v1_all)},
                          "lang_free": {"before": metrics(b_lf), "after": metrics(v1_lf)}},
              "per_lang_f1": {lg: {"before": metrics(a[0])["slot_f1"],
                                   "v1_only": metrics(a[1])["slot_f1"],
                                   "after": metrics(a[2])["slot_f1"]}
                              for lg, a in sorted(per_lang.items())},
              "q_boundary": {"before": {"fp": b_all["q_fp"], "fn": b_all["q_fn"]},
                             "v1_only": {"fp": v1_all["q_fp"], "fn": v1_all["q_fn"]},
                             "after": {"fp": a_all["q_fp"], "fn": a_all["q_fn"]}},
              "applied": {k: v for k, v in sorted(arb_counts.items())},
              "change_outcomes": dict(outcomes),
              "canon_applied_v1_only": dict(canon_v1),
              "canon_applied_after": dict(canon_after),
              "changed_examples": {k: v for k, v in sorted(examples.items()) if v}}
    out = ROOT / "eval" / "results" / f"{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    feats_path = ROOT / "eval" / "results" / f"{args.out_tag}_features.jsonl"
    with open(feats_path, "w") as f:
        for fr in feats_rows:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")
    b, a, v1 = result["lang_free"]["before"], result["lang_free"]["after"], \
        result["v1_only"]["lang_free"]["after"]
    print(f"{name}: lang_free F1 before={b['slot_f1']} EM={b['row_em']} "
          f"v1_only={v1['slot_f1']} after={a['slot_f1']} EM={a['row_em']} "
          f"(d_vs_before={round(a['slot_f1'] - b['slot_f1'], 4)}, "
          f"d_vs_v1={round(a['slot_f1'] - v1['slot_f1'], 4)})")
    print(f"applied={result['applied']}")
    print(f"change_outcomes={result['change_outcomes']}")
    print(f"q_boundary(before fp/fn={result['q_boundary']['before']['fp']}/"
          f"{result['q_boundary']['before']['fn']} -> "
          f"after {result['q_boundary']['after']['fp']}/"
          f"{result['q_boundary']['after']['fn']})")
    print("per_lang lang_free F1 (before->v1_only->after) " +
          " | ".join(f"{lg} {v['before']}->{v['v1_only']}->{v['after']}"
                     for lg, v in result["per_lang_f1"].items()
                     if lg in DISPLAY_LANGS))
    print(f"-> {out}")
    print(f"-> {feats_path}")


if __name__ == "__main__":
    main()
