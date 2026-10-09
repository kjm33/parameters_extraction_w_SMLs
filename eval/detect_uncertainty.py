"""Per-row failure detection / uncertainty signals for a fine-tuned extractor.

For each test row, runs the model greedily (collecting token logprobs) plus k sampled
generations (self-consistency), applies deterministic validators (schema keys, value
groundedness in the utterance, enum membership, lang sanity), and measures how well each
signal predicts actual row errors (row EM vs gold target). No fallback logic — this only
answers "when is the small model wrong, and how would we know?"

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/detect_uncertainty.py \
      --model-dir train/runs/llama-3.2-1b/merged --gpu 0 [--tag llama-1b-merged]
  # or: --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b/best
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from build_extraction import enum_supported, fold, number_found, tokens, verbatim  # noqa: E402
from eval_extract import extract_json, norm_val  # noqa: E402
from render import render_prompt  # noqa: E402

# ---------------------------------------------------------------- language id
LANG_CHARS = {
    "pl": "ąęółśżźćń",
    "de": "äöüß",
    "fr": "éèêàçùâîôû",
    "es": "ñáéíóúü¿¡",
}
LANG_STOPWORDS = {
    "en": ["i", "you", "want", "need", "please", "take", "drive", "me", "my", "the",
           "from", "without", "avoid", "route", "directions"],
    "de": ["ich", "bitte", "nach", "von", "zu", "ohne", "uber", "fahr", "mich", "mein",
           "will", "karte", "straße", "strasse"],
    "pl": ["do", "na", "bez", "przez", "prosze", "chce", "jedz", "mnie", "moj",
           "unikaj", "trasy", "autostrady"],
    "fr": ["je", "vais", "sans", "vers", "moi", "mon", "veux", "silk", "plait",
           "itineraire", "eviter"],
    "es": ["quiero", "ir", "sin", "hacia", "favor", "mi", "coche", "evitar",
           "ruta", "calle"],
}


def detect_lang(utterance: str) -> str | None:
    """Tiny marker-based 5-lang detector (chars weighted 3x, stopwords 1x)."""
    f = " " + fold(utterance) + " "
    words = set(re.findall(r"\w+", f))
    scores = {}
    for lang, chars in LANG_CHARS.items():
        scores[lang] = 3 * sum(f.count(c) for c in chars)
    for lang, sws in LANG_STOPWORDS.items():
        scores[lang] = scores.get(lang, 0) + sum(1 for w in sws if w in words)
    best = max(scores, key=scores.get)
    vals = sorted(scores.values(), reverse=True)
    if vals[0] == 0 or vals[0] == vals[1]:
        return None  # no signal / tie -> abstain
    return best


# ---------------------------------------------------------------- validators
def schema_props(row: dict) -> dict:
    params = row["function"].get("parameters") or {}
    return params.get("properties") or params


def grounding_violations(pred: dict, row: dict) -> list[str]:
    """Non-null predicted values that cannot be found in the utterance (= hallucinated)."""
    utt_f = fold(row["utterance"])
    utt_t = tokens(row["utterance"])
    props = schema_props(row)
    viol = []

    def check(v, ps):
        if isinstance(v, bool) or v is None:
            return True
        if isinstance(v, list):
            return all(check(x, ps) for x in v)
        if isinstance(v, (int, float)):
            return number_found(v, utt_f)
        if isinstance(v, str):
            if "enum" in (ps or {}):
                return enum_supported(v, utt_f)
            return verbatim(v, utt_f, utt_t)
        return False

    for k, v in pred.items():
        if v is None or k == "lang":
            continue
        if not check(v, props.get(k)):
            viol.append(k)
    return viol


def enum_violations(pred: dict, row: dict) -> list[str]:
    props = schema_props(row)
    viol = []
    for k, v in pred.items():
        ps = props.get(k) or {}
        allowed = ps.get("enum") or (ps.get("items") or {}).get("enum")
        if not allowed or v is None:
            continue
        vals = v if isinstance(v, list) else [v]
        allowed_n = {norm_val(a) for a in allowed}
        if any(isinstance(x, str) and norm_val(x) not in allowed_n for x in vals):
            viol.append(k)
    return viol


def validator_flags(pred: dict | None, row: dict) -> dict:
    keys_ref = set(row["target"].keys())
    flags = {
        "parse_fail": pred is None,
        "keys_mismatch": pred is not None and set(pred.keys()) != keys_ref,
        "ungrounded": pred is not None and bool(grounding_violations(pred, row)),
        "enum_invalid": pred is not None and bool(enum_violations(pred, row)),
        "lang_mismatch": False,
    }
    if pred and "lang" in pred and pred.get("lang"):
        det = detect_lang(row["utterance"])
        flags["lang_mismatch"] = det is not None and fold(str(pred["lang"])) != det
    return flags


# ---------------------------------------------------------------- comparison
def row_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    for k in set(a) | set(b):
        if norm_val(a.get(k)) != norm_val(b.get(k)):
            return False
    return True


# ---------------------------------------------------------------- AUC / cov
def auc(scores, labels) -> float:
    """Rank AUC (tie-aware). Labels: 1 = positive (wrong row). Higher score = more positive."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    sorted_s = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def coverage_table(risk, wrong, fracs=(0.1, 0.2, 0.3, 0.4, 0.5)) -> list[dict]:
    """If we escalate the riskiest X% rows, how many errors do we catch?"""
    r = np.asarray(risk, dtype=float)
    w = np.asarray(wrong, dtype=int)
    order = np.argsort(-r, kind="mergesort")
    out = []
    for frac in fracs:
        k = max(1, int(round(frac * len(r))))
        top = order[:k]
        caught = int(w[top].sum())
        out.append({
            "escalate_frac": round(frac, 2),
            "n_rows": k,
            "errors_total": int(w.sum()),
            "errors_caught": caught,
            "error_recall": round(caught / max(1, w.sum()), 4),
            "false_alarms": int(k - caught),
        })
    return out


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=None, help="merged/full model dir")
    p.add_argument("--candidate", default=None, help="config key (alt. to --model-dir)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--files", default="data/core_test.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--k", type=int, default=3, help="self-consistency samples")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if not a.model_dir and not a.candidate:
        p.error("need --model-dir or --candidate")
    return a


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    for part in args.files.split(","):
        with open(part) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows={len(rows)} model_dir={args.model_dir} candidate={args.candidate}")

    tkw = {}
    if args.candidate:
        cfg = json.loads((ROOT / "train" / "configs" / f"{args.candidate}.json").read_text())
        tkw = cfg.get("template_kwargs") or {}
    src = args.model_dir or json.loads(
        (ROOT / "train" / "configs" / f"{args.candidate}.json").read_text())["model_id"]

    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"merged adapter {args.adapter}")

    eos_ids = []
    for name in ("<|im_end|>", "<|eot_id|>", "<end_of_turn|>", "<eos>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and 0 < i < tok.vocab_size and i not in eos_ids:
            eos_ids.append(i)
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)
    eos_set = set(eos_ids)

    prompts = [render_prompt(tok, r, tkw) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))

    # ---- pass 1: greedy + logprobs
    mean_lp = [None] * len(rows)
    min_lp = [None] * len(rows)
    greedy_raw: list[str | None] = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), args.bs):
        idx = order[s:s + args.bs]
        enc = tok.pad({"input_ids": [prompts[i] for i in idx]}, return_tensors="pt",
                      padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None)
        ts = model.compute_transition_scores(out.sequences, out.scores,
                                             normalize_logits=True)  # [bs, gen]
        for j, i in enumerate(idx):
            gen = out.sequences[j][enc["input_ids"].shape[1]:].tolist()
            L = next((t for t, t_id in enumerate(gen) if t_id in eos_set), len(gen)) or 1
            lp = ts[j, :L].float().cpu().numpy()
            mean_lp[i] = float(lp.mean())
            min_lp[i] = float(lp.min())
            greedy_raw[i] = tok.decode(gen[:L], skip_special_tokens=True)
        if (s // args.bs) % 10 == 0:
            print(f"  greedy batch {s // args.bs + 1}/{(len(order) + args.bs - 1) // args.bs}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: self-consistency samples
    agree = [None] * len(rows)
    torch.manual_seed(0)
    for s in range(0, len(order), max(1, args.bs // args.k)):
        idx = order[s:s + max(1, args.bs // args.k)]
        enc = tok.pad({"input_ids": [prompts[i] for i in idx]}, return_tensors="pt",
                      padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.9,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None)
        for j, i in enumerate(idx):
            g = extract_json(greedy_raw[i])
            same = 0
            for kk in range(args.k):
                raw = tok.decode(out[j * args.k + kk][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                if row_equal(extract_json(raw), g):
                    same += 1
            agree[i] = same / args.k
        if (s // args.bs) % 10 == 0:
            print(f"  sample batch {s // args.bs + 1} ({time.time() - t0:.0f}s)", flush=True)

    # ---- assemble per-row records
    recs = []
    for row, raw, ml, mn, ag in zip(rows, greedy_raw, mean_lp, min_lp, agree):
        pred = extract_json(raw)
        flags = validator_flags(pred, row)
        correct = row_equal(pred, row["target"])
        recs.append({
            "id": row["id"], "lang": row["lang"], "correct": correct,
            "mean_logprob": None if ml is None else round(ml, 4),
            "min_logprob": None if mn is None else round(mn, 4),
            "agree_frac": ag,
            "flags": flags,
            "ungrounded_slots": grounding_violations(pred, row) if pred else [],
            "pred": pred,
        })

    wrong = np.array([0 if r["correct"] else 1 for r in recs])
    ml_arr = np.array([r["mean_logprob"] if r["mean_logprob"] is not None else -99
                       for r in recs])
    mn_arr = np.array([r["min_logprob"] if r["min_logprob"] is not None else -99
                       for r in recs])
    ag_arr = np.array([r["agree_frac"] if r["agree_frac"] is not None else 0.0
                       for r in recs])
    any_flag = np.array([int(any(r["flags"].values())) for r in recs])

    signals = {
        "1-minus_mean_logprob": {"auc": auc(-ml_arr, wrong)},
        "minus_min_logprob": {"auc": auc(-mn_arr, wrong)},
        "1-minus_agree_frac": {"auc": auc(1 - ag_arr, wrong)},
        "any_validator_flag": {"auc": auc(any_flag, wrong)},
    }
    thr_ml = float(np.percentile(ml_arr[wrong == 0], 5))
    thr_ag = 1.0
    combo = np.maximum(any_flag,
                       np.maximum((ml_arr < thr_ml).astype(float),
                                  (ag_arr < thr_ag).astype(float)))
    signals["combined (any flag | logprob<p5-of-correct | agree<1.0)"] = {
        "auc": auc(combo, wrong), "mean_logprob_p5_of_correct": round(thr_ml, 4)}

    validators = {}
    for fname in ("parse_fail", "keys_mismatch", "ungrounded", "enum_invalid", "lang_mismatch"):
        f = np.array([int(r["flags"][fname]) for r in recs])
        fires, fires_wrong = int(f.sum()), int((f & wrong).sum())
        validators[fname] = {
            "fires": fires,
            "precision_as_error": round(fires_wrong / max(1, fires), 4),
            "error_recall": round(fires_wrong / max(1, wrong.sum()), 4),
        }

    risks = {
        "combined": combo.astype(float),
        "validators_only": any_flag.astype(float),
        "selfconsistency_only": 1 - ag_arr,
        "logprob_only": -ml_arr,
    }
    coverage = {name: coverage_table(risk, wrong) for name, risk in risks.items()}

    report = {
        "model_dir": args.model_dir, "candidate": args.candidate, "adapter": args.adapter,
        "files": args.files, "n": len(recs), "n_wrong": int(wrong.sum()),
        "row_em": round(1 - wrong.mean(), 4),
        "gen_seconds": round(time.time() - t0, 1),
        "signals": signals, "validators": validators, "coverage": coverage,
        "detect_lang_agreement_with_gold": round(np.mean([
            detect_lang(r["utterance"]) == r["lang"] for r in rows]), 4),
    }

    tag = args.tag or (args.candidate or pathlib.Path(args.model_dir).name) + "-uncert"
    rows_path = ROOT / "eval" / "results" / f"{tag}_rows.jsonl"
    with open(rows_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n", "n_wrong", "row_em", "signals", "validators",
                       "detect_lang_agreement_with_gold")}, indent=2))
    print("coverage(combined):", json.dumps(report["coverage"]["combined"], indent=2))
    print(f"-> {out}\n-> {rows_path}")


if __name__ == "__main__":
    sys.exit(main())
