"""Experiment A: span-snapping post-processor + eval-normalization ablation (CPU-only).

For each of the 7 FT prediction dumps:
  1. baseline re-score (sanity vs eval_extract numbers)
  2. A2 ablation: re-score ORIGINAL preds with inflection-tolerant comparison
     (tok_match token-set equality) -> quantifies eval-normalization artifact share
  3. A1 snap: non-verbatim free-text pred values (skip lang/enum/number params,
     lists item-wise) snapped to the best-aligned utterance span; re-score
  4. per-change classification: canonical-appendage | inflection | boundary | other,
     with improved/worsened/neutral row deltas

Usage: ../function_calling_edge_fine_tuning/train/venv/bin/python eval/snap_spans.py
  -> eval/results/snap_spans_report.json
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from build_extraction import enum_supported, fold, number_found, tokens, tok_match, verbatim  # noqa: E402
from eval_extract import norm_val  # noqa: E402

RES = ROOT / "eval" / "results"
DUMPS = ["llama-3.2-1b-ft", "llama-3.2-3b-ft", "qwen2.5-0.5b-ft", "qwen3-0.6b-ft",
         "gemma-3-1b-ft", "qwen2.5-0.5b-fullft", "qwen3-0.6b-fullft"]
LANG_PARAMS = {"lang", "language"}
MIN_SNAP_TOKEN_MATCH = 0.5


def word_spans(utterance: str):
    """[(folded_token, start, end)] over word tokens of the original utterance."""
    out = []
    for m in re.finditer(r"\w+", utterance):
        out.append((fold(m.group(0)), m.start(), m.end()))
    return out


def snap_span(value: str, ws, utt_f: str, utt_t: set[str]):
    """(start, end) char span of the best-aligned utterance window; None if hopeless."""
    vt = sorted(tokens(value))
    if not vt or not ws:
        return None
    n = len(vt)
    best, best_score = None, 0.0
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, len(ws) - size + 1):
            window = ws[i:i + size]
            wtoks = [t for t, _, _ in window]
            matched = sum(1 for v in vt if any(tok_match(v, u) for u in wtoks))
            score = matched / n
            if score > best_score:
                best, best_score = window, score
    if best is None or best_score < MIN_SNAP_TOKEN_MATCH:
        return None
    return best[0][1], best[-1][2]


def slot_lenient_eq(gv, pv) -> bool:
    """Strict equality OR inflection-tolerant string match (A2 ablation)."""
    gn, pn = norm_val(gv), norm_val(pv)
    if gn == pn:
        return True
    if isinstance(gv, str) and isinstance(pv, str) and gn is not None and pn is not None:
        gt, pt = tokens(gv), tokens(pv)
        if gt and pt and len(gt) == len(pt):
            return all(any(tok_match(a, b) for b in pt) for a in gt)
    return False


def score_rows(rows, strict=True):
    """Returns (f1, em, per-row correctness list) using strict or lenient slot compare."""
    tp = fp = fn = em = 0
    per_row = []
    for r in rows:
        gold, pred = r["target"], (r["pred"] or {})
        ok = True
        for k, gv in gold.items():
            pv = pred.get(k)
            gn, pn = norm_val(gv), norm_val(pv)
            if gn is None:
                if pn is not None:
                    fp += 1
                    ok = False
            elif pn is not None and (gn == pn if strict else slot_lenient_eq(gv, pv)):
                tp += 1
            else:
                fn += 1
                if pn is not None:
                    fp += 1
                ok = False
        per_row.append(ok)
        em += int(ok)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return round(f1, 4), round(em / max(1, len(rows)), 4), per_row


def classify_change(gold_v, old_v, new_v):
    if new_v is None or norm_val(new_v) == norm_val(old_v):
        return None
    old_ok = norm_val(old_v) == norm_val(gold_v)
    new_ok = norm_val(new_v) == norm_val(gold_v)
    if "," in str(old_v) and "," not in str(gold_v):
        kind = "canonical-appendage"
    elif slot_lenient_eq(gold_v, new_v) and not slot_lenient_eq(gold_v, old_v):
        kind = "inflection"
    else:
        kind = "boundary"
    delta = "improved" if (new_ok and not old_ok) else ("worsened" if (old_ok and not new_ok) else "neutral")
    return kind, delta


def main():
    test = {}
    schemas = {}
    for line in open(ROOT / "data" / "core_test.jsonl"):
        if line.strip():
            d = json.loads(line)
            test[d["id"]] = d
            schemas.setdefault(d["function"]["name"], d["function"].get("parameters") or {})

    report = {}
    for tag in DUMPS:
        rows = []
        for line in open(RES / f"{tag}_preds.jsonl"):
            if not line.strip():
                continue
            d = json.loads(line)
            d["utterance"] = test[d["id"]]["utterance"]
            rows.append(d)

        # baseline (strict) + A2 lenient re-score of original preds
        f1_s, em_s, ok_strict = score_rows(rows, strict=True)
        f1_l, em_l, _ = score_rows(rows, strict=False)

        # A1 snap
        snap_rows = [dict(r, pred=json.loads(json.dumps(r["pred"])) if r["pred"] else r["pred"])
                     for r in rows]
        changes = {"canonical-appendage": {"improved": 0, "worsened": 0, "neutral": 0},
                   "inflection": {"improved": 0, "worsened": 0, "neutral": 0},
                   "boundary": {"improved": 0, "worsened": 0, "neutral": 0}}
        n_snapped = 0
        for r in snap_rows:
            pred = r["pred"]
            if not pred:
                continue
            ws = word_spans(r["utterance"])
            utt_f = fold(r["utterance"])
            utt_t = tokens(r["utterance"])
            params = schemas.get(r["function"], {})
            props = params.get("properties") or params
            for k, v in list(pred.items()):
                if k in LANG_PARAMS or v is None:
                    continue
                ps = props.get(k) or {}
                is_enum = "enum" in ps or "enum" in (ps.get("items") or {})
                def snap_one(x):
                    nonlocal n_snapped
                    if isinstance(x, str) and x and not is_enum:
                        if verbatim(x, utt_f, utt_t):
                            return x
                        span = snap_span(x, ws, utt_f, utt_t)
                        if span is not None:
                            a, b = span
                            cand = r["utterance"][a:b]
                            if not verbatim(cand, utt_f, utt_t):
                                return x  # safety: snap must itself be grounded
                            n_snapped += 1
                            return cand
                        return x
                    if isinstance(x, (int, float)):
                        return x if number_found(x, utt_f) else x
                    return x
                if isinstance(v, list):
                    pred[k] = [snap_one(x) for x in v]
                elif isinstance(v, str):
                    new = snap_one(v)
                    c = classify_change(r["target"].get(k), v, new)
                    if c:
                        changes[c[0]][c[1]] += 1
                    pred[k] = new
        f1_snap, em_snap, ok_snap = score_rows(snap_rows, strict=True)

        flipped = {"fixed": sum(1 for a, b in zip(ok_strict, ok_snap) if not a and b),
                   "broken": sum(1 for a, b in zip(ok_strict, ok_snap) if a and not b)}
        report[tag] = {
            "n": len(rows),
            "baseline_strict": {"f1": f1_s, "em": em_s},
            "a2_lenient_rescore_original": {"f1": f1_l, "em": em_l,
                                            "f1_gain_from_normalization": round(f1_l - f1_s, 4)},
            "a1_snapped_strict": {"f1": f1_snap, "em": em_snap,
                                  "f1_delta": round(f1_snap - f1_s, 4)},
            "n_values_snapped": n_snapped,
            "row_flips": flipped,
            "change_classes": changes,
        }
        print(f"{tag}: F1 {f1_s} -> lenient {f1_l} (+{f1_l - f1_s:.4f}) | snap {f1_snap} "
              f"({f1_snap - f1_s:+.4f}), snapped {n_snapped}, fixes {flipped['fixed']}, "
              f"breaks {flipped['broken']}")

    out = RES / "snap_spans_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
