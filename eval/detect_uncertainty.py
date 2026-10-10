"""Per-row failure detection / uncertainty signals for a fine-tuned extractor.

For each row, runs the model greedily (collecting token logprobs), a teacher-forced
re-forward (per-slot token confidence via char-span -> token mapping, plus hidden
states at every 4th layer), k sampled generations (self-consistency + per-slot
semantic entropy over normalized-value clusters), and deterministic validators
(schema keys, value groundedness, enum membership, lang sanity). Measures how well
each signal predicts actual row errors (row EM vs gold target). No fallback logic.

Outputs (tag defaults to candidate/model-dir name + "-uncert"):
  eval/results/<tag>.json          summary: AUCs, validator stats, coverage tables
  eval/results/<tag>_rows.jsonl    per-row: pred, flags, correctness, signals
  eval/results/<tag>_features.jsonl per-row scalar features (joins _rows by id)
  eval/results/<tag>_hidden.npz    last-completion-token hidden states, fp16 [n, H]
                                   arrays "h<i>" for layer index i, plus "ids"

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python eval/detect_uncertainty.py \
      --model-dir train/runs/llama-3.2-1b/merged --gpu 0 --tag llama-1b-merged
  # or: --candidate llama-3.2-1b --adapter train/runs/llama-3.2-1b/best
  # multiple files: --files data/core_train_holdout.jsonl,data/core_val.jsonl
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


def slot_norm(v) -> str:
    v = norm_val(v)
    return "<null>" if v is None else str(v)


def slot_entropy(sample_vals: list[str]) -> float:
    """Entropy of the empirical cluster distribution for one slot across samples."""
    n = len(sample_vals)
    counts: dict[str, int] = {}
    for v in sample_vals:
        counts[v] = counts.get(v, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * np.log(p)
    return float(h)


def value_spans(text: str, keys) -> dict[str, tuple[int, int]]:
    """Char spans of each slot's value inside the completion JSON text."""
    dec = json.JSONDecoder()
    spans = {}
    for k in keys:
        m = re.search(r'"%s"\s*:' % re.escape(k), text)
        if not m:
            continue
        pos = m.end()
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        try:
            _, end = dec.raw_decode(text, pos)
        except ValueError:
            continue
        spans[k] = (pos, end)
    return spans


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
    p.add_argument("--k", type=int, default=5, help="self-consistency samples")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--cpu", action="store_true", help="run on CPU (debug/smoke only)")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    if not a.cpu:
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
    dev = torch.device("cpu" if args.cpu else "cuda")
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float32 if args.cpu else torch.bfloat16,
        attn_implementation="sdpa").to(dev).eval()
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

    prompt_ids = [render_prompt(tok, r, tkw) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompt_ids[i]))

    # ---- pass 1: greedy generation (left-padded) + per-token logprobs
    greedy_raw: list[str | None] = [None] * len(rows)
    gen_mean_lp = [None] * len(rows)
    gen_min_lp = [None] * len(rows)
    t0 = time.time()
    tok.padding_side = "left"
    for s in range(0, len(order), args.bs):
        idx = order[s:s + args.bs]
        enc = tok.pad({"input_ids": [prompt_ids[i] for i in idx]}, return_tensors="pt",
                      padding=True).to(dev)
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
            L = next((t for t, t_id in enumerate(gen) if t_id in eos_set), len(gen))
            lp = ts[j, :L].float().cpu().numpy()
            gen_mean_lp[i] = float(lp.mean()) if len(lp) else None
            gen_min_lp[i] = float(lp.min()) if len(lp) else None
            greedy_raw[i] = tok.decode(gen[:L], skip_special_tokens=True)
        if (s // args.bs) % 10 == 0:
            print(f"  greedy batch {s // args.bs + 1}/{(len(order) + args.bs - 1) // args.bs}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: teacher-forced re-forward (right-padded) for per-slot logprobs
    #      + last-token hidden states at every 4th layer
    preds = [extract_json(r) for r in greedy_raw]
    full_ids, offs, comp_lens = [], [], []
    for i in range(len(rows)):
        comp = greedy_raw[i] or ""
        enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
        ids = prompt_ids[i] + enc["input_ids"]
        full_ids.append(ids)
        offs.append(enc["offset_mapping"])
        comp_lens.append(len(enc["input_ids"]))
    hidden_pick = sorted(set(list(range(0, model.config.num_hidden_layers + 1))[::4]))
    hidden = {li: np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
              for li in hidden_pick}
    slot_lp: list[dict] = [{} for _ in rows]      # key -> [lp of covering tokens]
    first_val_lp: list[float | None] = [None] * len(rows)
    tok.padding_side = "right"
    order2 = sorted(range(len(rows)), key=lambda i: -len(full_ids[i]))
    bs2 = max(4, args.bs // 2)
    for s in range(0, len(order2), bs2):
        idx = order2[s:s + bs2]
        maxlen = max(len(full_ids[i]) for i in idx)
        input_ids = torch.full((len(idx), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(idx), maxlen), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(full_ids[i])
            input_ids[j, :n] = torch.tensor(full_ids[i])
            attn[j, :n] = 1
        with torch.no_grad():
            fout = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                         output_hidden_states=True)
        for j, i in enumerate(idx):
            n = len(full_ids[i])
            # causal logprob of token t = logit[t-1, tok_t] - logsumexp(logits[t-1])
            row = fout.logits[j, :n]
            tgt = torch.tensor(full_ids[i], device=row.device)
            # causal logprob of token t+1 = logits[t, tok_{t+1}] - logsumexp(logits[t])
            g = row[:-1].gather(-1, tgt[1:].unsqueeze(-1)).squeeze(-1).float()
            lse = torch.logsumexp(row[:-1].float(), dim=-1)
            lp_t = (g - lse).cpu().numpy()  # lp_t[t] = logprob of token t+1
            tok_lp = np.concatenate([[0.0], lp_t])    # tok_lp[t] = logprob of token t
            cl = comp_lens[i]
            comp_lp = tok_lp[n - cl:]
            # hidden: last completion token
            for li in hidden_pick:
                hidden[li][i] = fout.hidden_states[li][j, n - 1].detach().to(torch.float16).cpu().numpy()
            # per-slot spans
            spans = value_spans(greedy_raw[i] or "", preds[i].keys()) if preds[i] else {}
            for k, (a, b) in spans.items():
                lps = [float(comp_lp[t]) for t, (ta, tb) in enumerate(offs[i])
                       if tb > a and ta < b and t < cl]
                if lps:
                    slot_lp[i][k] = lps
            if spans:
                first_key = next(iter(spans))
                a, _ = spans[first_key]
                for t, (ta, tb) in enumerate(offs[i]):
                    if tb > a and t < cl:
                        first_val_lp[i] = float(comp_lp[t])
                        break
        if (s // bs2) % 20 == 0:
            print(f"  forward batch {s // bs2 + 1}/{(len(order2) + bs2 - 1) // bs2}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 3: self-consistency samples (k, temp) -> row agreement + per-slot entropy
    agree = [None] * len(rows)
    sample_slots: list[list[dict]] = [[] for _ in rows]  # per sample: {key: cluster str}
    torch.manual_seed(0)
    tok.padding_side = "left"
    rows_per_batch = max(1, args.bs // args.k)
    for s in range(0, len(order), rows_per_batch):
        idx = order[s:s + rows_per_batch]
        enc = tok.pad({"input_ids": [prompt_ids[i] for i in idx]}, return_tensors="pt",
                      padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.9,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None)
        for j, i in enumerate(idx):
            g = preds[i]
            same = 0
            keys = set(g.keys()) if g else set()
            per_k: dict[str, list[str]] = {}
            for kk in range(args.k):
                raw = tok.decode(out[j * args.k + kk][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                sp = extract_json(raw)
                if row_equal(sp, g):
                    same += 1
                sk = sp if sp else {}
                keys |= set(sk.keys())
                for k in keys:
                    per_k.setdefault(k, []).append(slot_norm(sk.get(k)))
            agree[i] = same / args.k
            sample_slots[i] = per_k
        if (s // rows_per_batch) % 20 == 0:
            print(f"  sample batch {s // rows_per_batch + 1}"
                  f"/{(len(order) + rows_per_batch - 1) // rows_per_batch}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    # ---- assemble per-row records
    recs = []
    feats = []
    for i, row in enumerate(rows):
        pred = preds[i]
        flags = validator_flags(pred, row)
        correct = row_equal(pred, row["target"])
        recs.append({
            "id": row["id"], "lang": row["lang"], "correct": correct,
            "mean_logprob": None if gen_mean_lp[i] is None else round(gen_mean_lp[i], 4),
            "min_logprob": None if gen_min_lp[i] is None else round(gen_min_lp[i], 4),
            "agree_frac": agree[i],
            "flags": flags,
            "ungrounded_slots": grounding_violations(pred, row) if pred else [],
            "pred": pred,
        })
        slp = slot_lp[i]
        mins = [min(v) for v in slp.values()]
        means = [float(np.mean(v)) for v in slp.values()]
        nulls = [k for k in (pred or {}) if (pred or {}).get(k) is None and k in slp]
        null_mins = [min(slp[k]) for k in nulls]
        val_mins = [min(slp[k]) for k in slp if k not in nulls]
        slot_se = {k: round(slot_entropy(v), 4) for k, v in sample_slots[i].items()}
        se_vals = list(slot_se.values())
        feats.append({
            "id": row["id"], "lang": row["lang"], "function": row["function"]["name"],
            "utt_len": len(row["utterance"]),
            "n_slots": len((row["target"] or {})),
            "n_pred_null": sum(1 for v in (pred or {}).values() if v is None),
            "mean_logprob": recs[-1]["mean_logprob"], "min_logprob": recs[-1]["min_logprob"],
            "first_val_logprob": None if first_val_lp[i] is None else round(first_val_lp[i], 4),
            "slot_lp_min_overall": round(float(min(mins)), 4) if mins else None,
            "slot_lp_min_mean": round(float(np.mean(mins)), 4) if mins else None,
            "slot_lp_mean_min": round(float(min(means)), 4) if means else None,
            "null_lp_min": round(float(min(null_mins)), 4) if null_mins else None,
            "value_lp_min": round(float(min(val_mins)), 4) if val_mins else None,
            "se_max": round(float(max(se_vals)), 4) if se_vals else None,
            "se_mean": round(float(np.mean(se_vals)), 4) if se_vals else None,
            "agree_frac": agree[i],
            "flag_parse_fail": int(flags["parse_fail"]),
            "flag_keys_mismatch": int(flags["keys_mismatch"]),
            "flag_ungrounded": int(flags["ungrounded"]),
            "flag_enum_invalid": int(flags["enum_invalid"]),
            "flag_lang_mismatch": int(flags["lang_mismatch"]),
            "n_ungrounded": len(recs[-1]["ungrounded_slots"]),
            "slot_se": slot_se,
            "slot_lp_min": {k: round(float(min(v)), 4) for k, v in slp.items()},
        })

    wrong = np.array([0 if r["correct"] else 1 for r in recs])
    ml_arr = np.array([r["mean_logprob"] if r["mean_logprob"] is not None else -99
                       for r in recs])
    mn_arr = np.array([r["min_logprob"] if r["min_logprob"] is not None else -99
                       for r in recs])
    ag_arr = np.array([r["agree_frac"] if r["agree_frac"] is not None else 0.0
                       for r in recs])
    any_flag = np.array([int(any(r["flags"].values())) for r in recs])
    semax = np.array([f["se_max"] if f["se_max"] is not None else 0.0 for f in feats])

    signals = {
        "1-minus_mean_logprob": {"auc": auc(-ml_arr, wrong)},
        "minus_min_logprob": {"auc": auc(-mn_arr, wrong)},
        "1-minus_agree_frac": {"auc": auc(1 - ag_arr, wrong)},
        "any_validator_flag": {"auc": auc(any_flag, wrong)},
        "slot_se_max": {"auc": auc(semax, wrong)},
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
        "slot_se_max": semax,
    }
    coverage = {name: coverage_table(risk, wrong) for name, risk in risks.items()}

    report = {
        "model_dir": args.model_dir, "candidate": args.candidate, "adapter": args.adapter,
        "files": args.files, "n": len(recs), "n_wrong": int(wrong.sum()),
        "row_em": round(1 - wrong.mean(), 4),
        "gen_seconds": round(time.time() - t0, 1),
        "signals": signals, "validators": validators, "coverage": coverage,
        "hidden_layers": hidden_pick,
        "detect_lang_agreement_with_gold": round(np.mean([
            detect_lang(r["utterance"]) == r["lang"] for r in rows]), 4),
    }

    tag = args.tag or (args.candidate or pathlib.Path(args.model_dir).name) + "-uncert"
    rows_path = ROOT / "eval" / "results" / f"{tag}_rows.jsonl"
    with open(rows_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    feats_path = ROOT / "eval" / "results" / f"{tag}_features.jsonl"
    with open(feats_path, "w") as f:
        for r in feats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hid_path = ROOT / "eval" / "results" / f"{tag}_hidden.npz"
    np.savez_compressed(hid_path, ids=np.array([r["id"] for r in recs]),
                        **{f"h{li}": hidden[li] for li in hidden_pick})
    out = ROOT / "eval" / "results" / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n", "n_wrong", "row_em", "signals", "validators",
                       "detect_lang_agreement_with_gold")}, indent=2))
    print("coverage(logprob_only):", json.dumps(report["coverage"]["logprob_only"], indent=2))
    print(f"-> {out}\n-> {rows_path}\n-> {feats_path}\n-> {hid_path}")


if __name__ == "__main__":
    sys.exit(main())
