"""Gold-v3 teacher annotation pipeline (OpenRouter bake-off, pre-registered policy).

Samples a (lang x function) stratified bake-off frame from the five pooled v2
files, asks each teacher model to re-annotate the byte-identical v2 prompt
(system = render.SYSTEM_PROMPT + TEACHER_ANNEX, user = render.user_content),
then grounds teacher output with build_extraction validators (reject, not fix).

Outputs:
  eval/results/goldv3/<short>_<tag>_preds.jsonl   one line per sampled row
  logs/goldv3_usage.jsonl                          one line per call (append)

Usage:
  GOLDV3_MOCK=1 python3 src/build_gold_v3.py --bakeoff 12 --tag mocktest
  python3 src/build_gold_v3.py --bakeoff 500 --tag r12a --workers 8 --budget 2.0
  python3 src/build_gold_v3.py --full --tag r12full
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import sys
import threading
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import (  # noqa: E402
    fold, number_found, read_jsonl, tok_match, tokens, verbatim)
from render import SYSTEM_PROMPT, user_content  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
SEED = 20261013
FILES = ["data/core_train.jsonl", "data/core_val.jsonl", "data/core_test.jsonl",
         "data/aug_train.jsonl", "data/aug_val.jsonl"]
TEACHER_ANNEX = """ADJUDICATION POLICY (follow strictly):
- One key per schema parameter, null when the message does not state it. lang is always null.
- Free-text values: copy the user's words verbatim, except Polish place names, where you
  output the dictionary base form (e.g. "z Torunia" -> "Toruń", "do Białegostoku" ->
  "Białystok", "Krakowa" -> "Kraków").
- Include a trailing country/region qualifier (like ", France" or ", DE") ONLY if the user
  actually said it. Do not add qualifiers the user did not say.
- The value contains only the words naming the place: drop phrases like "our hotel on",
  but keep meaningful qualifiers the user said, like "Detroit, Michigan".
- Enum parameters: output the exact enum literal from the schema, chosen from the user's
  intent. "via X" / "around X" / "po autostradzie" is NOT an avoid request. "płatnych
  autostrad" (toll motorways) means tollRoad, not motorway. An avoid parameter is non-null
  only when the user asks to AVOID that feature.
- Numeric parameters: non-null only when the user explicitly states the number.
- Output ONLY the JSON object."""

MODELS = [
    ("anthropic/claude-haiku-5.5", "t1_haiku55", {"reasoning": {"effort": "low"}}, 1024),
    ("openai/gpt-5.4-mini", "t2_gpt54mini", {"reasoning": {"effort": "none"}}, 1024),
    ("openai/gpt-6-luna", "t3_gpt6luna", {}, 1024),
    ("z-ai/glm-5.3-flash", "t4_glm53flash", {"reasoning": {"effort": "low"}}, 1024),
    ("deepseek/deepseek-v4.1-flash", "t5_dsv41flash", {"reasoning": {"effort": "low"}}, 4096),
    ("inclusionai/ling-3.1-flash", "t6_ling31flash", {}, 4096),
]


def load_key() -> str:
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found (env or .env in repo root/parent)")


def pool_all() -> list:
    rows = []
    for f in FILES:
        for r in read_jsonl(ROOT / f):
            r["file"] = pathlib.Path(f).stem
            rows.append(r)
    return rows


def allocate(sizes: dict, n: int) -> dict:
    keys = list(sizes)
    if n <= 0:
        return {k: 0 for k in keys}
    if n < len(keys):
        return {k: (1 if i < n else 0) for i, k in enumerate(keys)}
    total = sum(sizes.values())
    base = {k: max(1, n * sizes[k] // total) for k in keys}
    rem = n - sum(base.values())
    if rem < 0:
        for k in sorted(keys, key=lambda k: (-sizes[k], k)):
            while rem < 0 and base[k] > 1:
                base[k] -= 1
                rem += 1
    else:
        order = sorted(keys, key=lambda k: (-(n * sizes[k] / total - n * sizes[k] // total), k))
        i = 0
        while rem > 0:
            base[order[i % len(order)]] += 1
            rem -= 1
            i += 1
    return base


def build_sample(n: int) -> list:
    rows = pool_all()
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r.get("lang") or "?", r["function"]["name"]), []).append(i)
    keys = sorted(strata)
    alloc = allocate({k: len(strata[k]) for k in keys}, n)
    rng = random.Random(SEED)
    picked = []
    for k in keys:
        ks = strata[k]
        picked.extend(rng.sample(ks, min(alloc[k], len(ks))))
    picked.sort()
    return [rows[i] for i in picked]


def extract_json_obj(text):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except Exception:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def check_value(pschema: dict, v, utt_f: str, utt_t: set):
    ps = pschema or {}
    if isinstance(v, bool):
        return False, "bool"
    if isinstance(v, dict):
        return False, "object"
    if isinstance(v, list):
        if not v:
            return False, "empty-array"
        for x in v:
            ok, why = check_value(ps.get("items") or ps, x, utt_f, utt_t)
            if not ok:
                return False, f"array-item:{why}"
        return True, ""
    if "enum" in ps:
        enums = {fold(str(e)) for e in ps["enum"]}
        if fold(str(v)) in enums:
            return True, ""
        return False, "enum-not-canonical"
    if ps.get("type") in ("integer", "number"):
        if not isinstance(v, (int, float, str)) or isinstance(v, bool):
            return False, "number-type"
        return (True, "") if number_found(v, utt_f) else (False, "number-not-stated")
    sv = str(v)
    if verbatim(sv, utt_f, utt_t):
        return True, ""
    if tok_match(fold(sv), utt_f):
        return True, ""
    return False, "not-verbatim"


def ground(obj: dict, row: dict):
    props = ((row["function"].get("parameters") or {}).get("properties")) or {}
    extra = sorted(k for k in obj if k not in props)
    if extra:
        return None, [f"extra_keys:{','.join(extra)}"]
    utt_f, utt_t = fold(row["utterance"]), tokens(row["utterance"])
    out, reasons = {}, []
    for k, ps in props.items():
        v = obj.get(k)
        if v is None:
            out[k] = None
            continue
        ok, why = check_value(ps, v, utt_f, utt_t)
        if ok:
            out[k] = v
        else:
            reasons.append(f"{k}:{why}")
    if reasons:
        return None, reasons
    missing = [k for k in props if k not in obj]
    if missing:
        reasons.append("filled_missing_keys:" + ",".join(missing))
    return out, reasons


def post_once(key: str, body: dict, timeout: int):
    return requests.post(ENDPOINT,
                         headers={"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"},
                         json=body, timeout=timeout)


def teacher_call(key: str, model: str, system: str, row: dict, extras: dict,
                 timeout: int = 120, max_retries: int = 6, max_tokens: int = 1024) -> dict:
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_content(row)}],
            "max_tokens": max_tokens}
    body.update(extras)
    delay, last = 0.5, None
    for attempt in range(max_retries):
        try:
            r = post_once(key, body, timeout)
        except requests.RequestException as e:
            last = f"network: {e}"
        else:
            if r.status_code == 200:
                out = r.json()
                content = ((out.get("choices") or [{}])[0].get("message") or {}).get("content")
                return {"status": "ok", "content": content, "usage": out.get("usage") or {}}
            try:
                msg = r.json().get("error", {}).get("message", r.text[:300])
            except Exception:
                msg = r.text[:300]
            last = f"HTTP {r.status_code}: {msg}"
            if r.status_code == 400 and extras and any(
                    w in msg.lower() for w in ("unsupported", "unknown", "not supported",
                                               "unexpected", "unrecognized")):
                body2 = {k: v for k, v in body.items() if k not in extras}
                try:
                    r2 = post_once(key, body2, timeout)
                except requests.RequestException as e2:
                    return {"status": "error", "error": f"network after extras-drop: {e2}",
                            "usage": {}}
                if r2.status_code == 200:
                    out = r2.json()
                    content = ((out.get("choices") or [{}])[0].get("message") or {}).get("content")
                    return {"status": "ok", "content": content, "usage": out.get("usage") or {}}
                return {"status": "error", "error": f"HTTP {r2.status_code} after extras-drop",
                        "usage": {}}
            if r.status_code not in RETRY_STATUS:
                return {"status": "error", "error": last, "usage": {}}
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                delay = max(delay, float(ra))
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return {"status": "error", "error": last, "usage": {}}


def run_model(entry: tuple, rows: list, args, key, system: str, usage_path: pathlib.Path):
    model, short, extras, max_tokens = entry
    mock = os.environ.get("GOLDV3_MOCK") == "1"
    out_dir = ROOT / "eval" / "results" / "goldv3"
    out_path = out_dir / f"{short}_{args.tag}_preds.jsonl"
    spent, lock = [0.0], threading.Lock()

    def call(row):
        with lock:
            if spent[0] >= args.budget:
                return {"status": "skipped_budget", "content": None, "usage": {}, "error": None}
        if mock:
            content = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
            return {"status": "ok", "content": content,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                    "error": None}
        res = teacher_call(key, model, system, row, dict(extras), max_tokens=max_tokens)
        with lock:
            spent[0] += (res.get("usage") or {}).get("cost") or 0.0
        return res

    t0 = time.time()
    results: list = [None] * len(rows)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call, r): i for i, r in enumerate(rows)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"status": "error", "error": str(e), "usage": {}}
            done += 1
            if done % 50 == 0:
                print(f"  [{short}] {done}/{len(rows)} ({time.time() - t0:.0f}s, "
                      f"cost ${spent[0]:.4f})", flush=True)

    n_called = n_parse = n_grounded = pt = ct = 0
    lines = []
    with usage_path.open("a") as uf:
        for row, res in zip(rows, results):
            usage = res.get("usage") or {}
            pt += usage.get("prompt_tokens") or 0
            ct += usage.get("completion_tokens") or 0
            reasons, target, grounded = [], None, False
            if res["status"] == "ok":
                n_called += 1
                obj = extract_json_obj(res.get("content"))
                if obj is None:
                    status, reasons = "parse_fail", ["parse_fail"]
                else:
                    n_parse += 1
                    target, reasons = ground(obj, row)
                    grounded = target is not None
                    status = "ok" if grounded else "reject"
                    n_grounded += int(grounded)
            elif res["status"] == "skipped_budget":
                status, reasons = "skipped_budget", ["skipped_budget"]
            else:
                status, reasons = "error", [f"error:{res.get('error')}"]
            lines.append({"id": row["id"], "file": row["file"], "lang": row.get("lang"),
                          "fn": row["function"]["name"], "teacher_target": target,
                          "raw": (res.get("content") or "")[:2000], "grounded": grounded,
                          "fail_reasons": reasons})
            uf.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                                 "id": row["id"], "status": status,
                                 "prompt_tokens": usage.get("prompt_tokens") or 0,
                                 "completion_tokens": usage.get("completion_tokens") or 0,
                                 "cost": usage.get("cost") or 0.0},
                                ensure_ascii=False) + "\n")
    out_path.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines))
    print(f"[{short}] {model}: n={len(rows)} called={n_called} "
          f"parse={n_parse / max(1, n_called):.3f} "
          f"grounded={n_grounded / max(1, len(rows)):.3f} ({n_grounded}/{len(rows)}) "
          f"tokens={pt}+{ct} cost=${spent[0]:.4f} elapsed={time.time() - t0:.0f}s -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bakeoff", type=int, default=0,
                   help="stratified sample size over the pooled 16,616-row frame")
    p.add_argument("--tag", required=True)
    p.add_argument("--models", default=None, help="comma-separated model ids (default: slate)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget", type=float, default=2.0, help="per-model cost cap (USD)")
    p.add_argument("--full", action="store_true", help="all rows, no sampling")
    p.add_argument("--ids-from", default=None, metavar="PREDS",
                   help="run only rows that are NOT grounded in this preds jsonl (by id)")
    return p.parse_args()


def load_ids_from(path: str) -> set:
    ids = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("grounded"):
                ids.add(r["id"])
    return ids


def main():
    args = parse_args()
    entries = MODELS
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        known = {m[0]: m for m in MODELS}
        entries = [known.get(m, (m, m.split("/")[-1], {}, 1024)) for m in want]
    n_modes = sum(bool(x) for x in (args.full, args.bakeoff, args.ids_from))
    if n_modes != 1:
        raise SystemExit("need exactly one of --bakeoff N / --full / --ids-from PREDS")
    if args.full:
        rows = pool_all()
    elif args.ids_from:
        want_ids = load_ids_from(args.ids_from)
        rows = [r for r in pool_all() if r["id"] in want_ids]
        missing = len(want_ids) - len({r["id"] for r in rows})
        if missing:
            raise SystemExit(f"--ids-from: {missing} ids not found in pooled frame")
        if len(rows) != len(want_ids):
            print(f"note: {len(rows) - len(want_ids)} duplicate-id rows re-run "
                  f"(ids not unique across/within files)", flush=True)
    else:
        rows = build_sample(args.bakeoff)
    mock = os.environ.get("GOLDV3_MOCK") == "1"
    print(f"rows={len(rows)} tag={args.tag} models={[e[1] for e in entries]} "
          f"mock={mock} budget=${args.budget:.2f} workers={args.workers}")
    (ROOT / "eval" / "results" / "goldv3").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    key = None if mock else load_key()
    system = SYSTEM_PROMPT + "\n\n" + TEACHER_ANNEX
    usage_path = ROOT / "logs" / "goldv3_usage.jsonl"
    for entry in entries:
        run_model(entry, rows, args, key, system, usage_path)


if __name__ == "__main__":
    sys.exit(main())
