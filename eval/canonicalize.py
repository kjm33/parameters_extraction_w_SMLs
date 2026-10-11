"""B2 canonicalizer: alias gazetteer from train utterance→target pairs + pl suffix rules.

Measures post-processing recovery on an existing *_preds.jsonl dump (no GPU):
- builds an alias map  folded_surface -> canonical_value  from train rows where the
  gold value does not literally appear in the utterance but a close inflected token
  does (shared prefix ≥ min_len*0.7, or shorter is a prefix of the longer with a
  ≤3-char leftover);
- applies it to predicted values (ambiguous aliases skipped);
- optional --append-state: if pred is a strict prefix of the utterance span
  "<pred>, XX" (2-letter uppercase), re-append ", XX" (utterance-grounded, no gold);
- reports exact-scorer F1/EM before/after + fixed/broken counts, writes JSON report.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_extract import norm_val  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_tags import fold, wfold  # noqa: E402

STATE_RE = re.compile(r"^[A-ZÀ-Ý]{2}$")


def build_alias(rows, min_prefix=4, max_leftover=3):
    """alias[(fn, param, lang)][folded_token] -> Counter(canonical values); token != value."""
    alias = defaultdict(Counter)
    for r in rows:
        utt_tokens = {wfold(w) for w in r["utterance"].split()}
        fn = r["function"]["name"]
        lang = r.get("lang")
        for k, v in r["target"].items():
            if not isinstance(v, str) or not v:
                continue
            vf = fold(v)
            if vf in utt_tokens:
                continue  # canonical already present; no alias needed
            for wf in utt_tokens:
                if not wf or abs(len(wf) - len(vf)) > max_leftover:
                    continue
                n = min(len(wf), len(vf))
                shared = 0
                while shared < n and wf[shared] == vf[shared]:
                    shared += 1
                ok = shared >= min_prefix and (
                    shared >= 0.7 * min(len(wf), len(vf))
                )
                leftover = abs(len(wf) - len(vf))
                if ok and leftover <= max_leftover:
                    alias[(fn, k, lang)][wf][v] += 1
    return alias


def pl_stem(w: str) -> str:
    """Crude Polish case-ending stripper (fallback only)."""
    for suf in ("ie", "owie", "ach", "owi", "em", "ie", "ą", "ę", "y", "i", "e", "a", "u", "o", "w"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: len(w) - len(suf)]
    return w


def canonicalize(val, alias, rev_index):
    """Return (new_val, rule) or (val, None)."""
    if not isinstance(val, str) or not val:
        return val, None
    toks = [wfold(w) for w in val.split()]
    if len(toks) != 1:
        return val, None  # single-token values only (safe subset)
    wf = toks[0]
    cands = alias.get(wf)
    if cands and len(cands) == 1:
        return next(iter(cands)), "gazetteer"
    stem = pl_stem(wf)
    if stem != wf and stem in rev_index and len(rev_index[stem]) == 1:
        return next(iter(rev_index[stem])), "pl-stem"
    return val, None


def append_state(pred_val, utterance):
    """If utterance contains '<pred>, XX' with 2-letter uppercase XX -> re-append."""
    if not isinstance(pred_val, str) or not pred_val or ", " in pred_val:
        return pred_val, None
    m = re.search(re.escape(pred_val) + r",\s+([A-ZÀ-Ý]{2})(?![\w])", utterance)
    if m:
        return f"{pred_val}, {m.group(1)}", "append-state"
    return pred_val, None


def score(rows, preds_by_id, mutate=None):
    tp = fn = fp = 0
    em = 0
    fixes = Counter()
    for g in rows:
        pred = dict(preds_by_id[g["id"]]["pred"])
        ok = True
        if mutate:
            for k in list(pred):
                nv, rule = mutate(pred[k], g["utterance"])
                if rule:
                    pred[k] = nv
                    fixes[rule] += 1
        for k, gv in g["target"].items():
            gvn, pvn = norm_val(gv), norm_val(pred.get(k))
            if gvn is None:
                if pvn is not None:
                    fp += 1
                    ok = False
            elif pvn is not None and pvn == gvn:
                tp += 1
            else:
                fn += 1
                if pvn is not None:
                    fp += 1
                ok = False
        em += int(ok)
    return {
        "slot_f1": round(2 * tp / max(1, 2 * tp + fn + fp), 4),
        "row_em": round(em / len(rows), 4),
        "tp": tp, "fn": fn, "fp": fp,
        "applied": dict(fixes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="eval/results/<tag>_preds.jsonl")
    ap.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl")
    ap.add_argument("--test", default="data/core_test.jsonl")
    ap.add_argument("--append-state", action="store_true")
    ap.add_argument("--out", default=None, help="write patched preds jsonl here")
    ap.add_argument("--report", default=None, help="write JSON report here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test)]
    preds = [json.loads(l) for l in open(args.preds)]
    preds_by_id = {p["id"]: p for p in preds}

    train_rows = []
    for f in args.train.split(","):
        train_rows += [json.loads(l) for l in open(f)]
    alias = build_alias(train_rows)
    rev_index = defaultdict(set)
    for wf, cands in alias.items():
        rev_index[pl_stem(wf)] |= set(cands)
    for r in train_rows:
        for k, v in r["target"].items():
            if isinstance(v, str) and v:
                toks = [wfold(w) for w in v.split()]
                if len(toks) == 1:
                    rev_index.setdefault(pl_stem(toks[0]), set()).add(v)
    n_alias = sum(1 for c in alias.values() if len(c) == 1)

    def mutate(val, utt):
        nv, rule = canonicalize(val, alias, rev_index)
        if rule:
            return nv, rule
        if args.append_state:
            return append_state(val, utt)
        return val, None

    before = score(rows, preds_by_id)
    after = score(rows, preds_by_id, mutate=mutate)

    report = {
        "preds": args.preds,
        "gazetteer_entries": len(alias),
        "unambiguous": n_alias,
        "before": before,
        "after": after,
        "append_state": bool(args.append_state),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        with open(args.out, "w") as f:
            for p in preds:
                q = dict(p)
                for k in list(q["pred"]):
                    nv, rule = mutate(q["pred"][k], "")
                    if rule and rule != "append-state":
                        q["pred"][k] = nv
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
    if args.report:
        json.dump(report, open(args.report, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
