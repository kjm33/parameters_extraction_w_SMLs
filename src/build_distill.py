"""Teacher distillation: annotate leftover instruction_extraction texts with gpt-5.4-mini.

Pipeline (plans/2026-10-10_improvement-todo.md task 1):
  1. Candidate pool = instruction_extraction texts (v3 + dialogues driver turns) NOT
     consumed by build_augment (fold-prefix exclusion vs aug files), shingle-deduped vs
     core_train, self-deduped. Focus langs: pl/es/fr (the measured weakness).
  2. here_directions jobs -> one teacher call per text: exact training prompt format
     (render.SYSTEM_PROMPT + "FUNCTION: <schema>\\n\\nUSER_MESSAGE: <utt>"), chat
     completions, JSON mode. here_geocode jobs are deterministic (q = regex-proven
     location, no teacher call — the teacher dumps whole phrases into q).
  3. Data-Turnstile validation: parse -> keys must be schema params -> run param_target
     grounding (build_extraction helpers) with origin/destination/via forced null
     (lat,lng slots, resolver's job — matches core supervision) -> honest target;
     drop rows where the teacher produced no grounded non-lang value.
  4. Keys/budget: OPENAI_API_KEY_1 primary, OPENAI_API_KEY_3 overflow, loaded from ../.env
     (never hardcoded/printed). Conservative per-run token budget (default 1.2M/key,
     ~50% of the 2.5M mini pool); every request's usage appended to logs/distill_usage.jsonl.

Outputs: data/distill_train.jsonl (+ small distill_val.jsonl) + data/distill_train_stats.json.
Retrain: train_sft.py --train data/core_train.jsonl,data/aug_train.jsonl,data/distill_train.jsonl
        (append distill_val.jsonl to --val likewise).

Usage:
  python3 src/build_distill.py --dry-run          # pool stats only, no API calls
  python3 src/build_distill.py --max-rows 2000    # live run
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import threading
import time
from collections import Counter

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_extraction import fold, param_target, schema_param_order, tokens  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTR = ROOT.parent / "instruction_extraction"
OUT = ROOT / "data"
LOGS = ROOT / "logs"
ENV = ROOT.parent / ".env"

API_URL = "https://api.openai.com/v1/chat/completions"

# (env var, label) — KEY_2 is dead (credit_balance_exhausted), do not add
KEYS = [("OPENAI_API_KEY_1", "key1"), ("OPENAI_API_KEY_3", "key3")]


def load_keys() -> list[tuple[str, str]]:
    vals = {}
    for line in ENV.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    out = [(vals[name], label) for name, label in KEYS if vals.get(name)]
    if not out:
        sys.exit("no working keys found in ../.env (need OPENAI_API_KEY_1/3)")
    return out


# ---------------------------------------------------------------- candidates

def text_ok(t: str) -> bool:  # same policy as build_augment
    if not t or len(t) < 30 or len(t) > 600:
        return False
    if t.lstrip()[:1] in "{[<#|-":
        return False
    letters = re.findall(r"[A-Za-zÀ-ž]", t)
    return len(letters) >= 0.6 * max(1, len(t.replace(" ", "")))


FROMTO_RE = {
    "de": re.compile(r"\bvon\s+(?P<origin>[\w .'-]{1,30}?)\s+nach\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:und|oder|über|ohne|via|mit)\b|$)", re.I),
    "fr": re.compile(r"\bde\s+(?P<origin>[\w .'-]{1,30}?)\s+à\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:en|par|sans|puis|et|avec|pour)\b|$)", re.I),
    "es": re.compile(r"\bde\s+(?P<origin>[\w .'-]{1,30}?)\s+a\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:sin|evitando|por|y|con|para)\b|$)", re.I),
    "pl": re.compile(r"\bz\s+(?P<origin>[\w .'-]{1,30}?)\s+do\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:i|bez|przez|unikaj|z)\b|$)", re.I),
    "en": re.compile(r"\bfrom\s+(?P<origin>[\w .'-]{1,30}?)\s+to\s+(?P<dest>[\w .'-]{1,30}?)(?=[,.;:!?\n]|\s(?:via|without|avoiding|and|with|for)\b|$)", re.I),
}


def shingles(text: str, n: int = 5) -> set[int]:
    ws = re.findall(r"\w+", fold(text))
    return {hash(" ".join(ws[i:i + n])) for i in range(max(0, len(ws) - n + 1))}


def iter_texts(sources: set[str]):
    if "v3" in sources:
        for line in (INSTR / "data/instructions_v3.jsonl").open():
            r = json.loads(line)
            t = (r.get("text") or "").strip()
            if text_ok(t):
                yield "distill_i", r.get("language") or "en", t, r.get("origin"), r.get("destination")
    if "dialogues" in sources:
        for line in (INSTR / "data/dialogues.jsonl").open():
            d = json.loads(line)
            lang = d.get("language") or "en"
            rx = FROMTO_RE.get(lang)
            for turn in d.get("turns") or []:
                if turn.get("speaker") != "driver":
                    continue
                t = (turn.get("text") or "").strip()
                if not text_ok(t):
                    continue
                origin = dest = None
                if rx:
                    m = rx.search(t)
                    if m:
                        origin, dest = (m.group("origin") or "").strip(), (m.group("dest") or "").strip()
                yield "distill_d", lang, t, origin, dest


def load_canonical_schemas() -> dict[str, dict]:
    schemas = {}
    for line in (OUT / "core_train.jsonl").open():
        r = json.loads(line)
        fn = r["function"]["name"]
        if fn not in schemas:
            schemas[fn] = r["function"]
        if {"here_geocode", "here_directions"} <= set(schemas):
            break
    assert {"here_geocode", "here_directions"} <= set(schemas)
    return schemas


def build_jobs(langs: set[str], sources: set[str], limit: int,
               fns: set[str] | None = None, seed: int = 42,
               skip_pairs: set | None = None) -> tuple[list[dict], dict]:
    """Jobs = (source, lang, fn_schema, utterance, teacher_args_hint)."""
    schemas = load_canonical_schemas()
    geo_s, dir_s = schemas["here_geocode"], schemas["here_directions"]
    fns = fns or {"here_geocode", "here_directions"}
    skip_pairs = skip_pairs or set()  # {(fold_prefix, fn)} already in training data

    core_sh: set[int] = set()
    used_prefix: set[str] = set()
    for fname in ("core_train.jsonl", "aug_train.jsonl", "aug_val.jsonl"):
        for line in (OUT / fname).open():
            r = json.loads(line)
            core_sh |= shingles(r["utterance"])
            used_prefix.add(fold(r["utterance"])[:160])

    seen: set[str] = set()
    jobs: list[dict] = []
    info = Counter()
    for source, lang, t, origin, dest in iter_texts(sources):
        if lang not in langs:
            continue
        info["texts_seen"] += 1
        key = fold(t)[:160]
        if key in seen or key in used_prefix:
            info["dup"] += 1
            continue
        seen.add(key)
        sh = shingles(t)
        if sh and len(sh & core_sh) / len(sh) > 0.55:
            info["dup_core"] += 1
            continue
        # directions: teacher annotates constraint/mode slots (lat,lng slots forced null)
        if "here_directions" in fns and (key, "here_directions") not in skip_pairs:
            jobs.append({"source": source, "lang": lang, "fn": "here_directions",
                         "schema": dir_s, "utterance": t, "teacher": True})
        # geocode: deterministic (q = regex-proven location) — no teacher needed,
        # the teacher tends to dump whole phrases into q
        if "here_geocode" in fns and (key, "here_geocode") not in skip_pairs:
            for loc in (origin, dest):
                if loc and 2 <= len(loc) <= 60:
                    jobs.append({"source": source, "lang": lang, "fn": "here_geocode",
                                 "schema": geo_s, "utterance": t, "teacher": False,
                                 "geo_q": loc})
                    break
    random.Random(seed).shuffle(jobs)
    # interleave langs round-robin so budget isn't eaten by one language
    by_lang: dict[str, list[dict]] = {}
    for j in jobs:
        by_lang.setdefault(j["lang"], []).append(j)
    ordered: list[dict] = []
    i = 0
    while True:
        added = False
        for lang in sorted(by_lang):
            if i < len(by_lang[lang]):
                ordered.append(by_lang[lang][i])
                added = True
        if not added:
            break
        i += 1
    return ordered[:limit], info


# ---------------------------------------------------------------- teacher client

class Teacher:
    def __init__(self, keys, model: str, budget: int, max_completion: int):
        self.keys = keys
        self.model = model
        self.budget = budget  # per key
        self.usage: dict[str, dict] = {label: {"requests": 0, "prompt": 0, "completion": 0}
                                       for _, label in keys}
        self.lock = threading.Lock()
        self.ki = 0
        self.max_completion = max_completion
        self.ledger_path = LOGS / "distill_usage.jsonl"
        self.stopped = False

    def _spend(self, label: str, u: dict):
        with self.lock:
            d = self.usage[label]
            d["requests"] += 1
            d["prompt"] += u.get("prompt_tokens", 0)
            d["completion"] += u.get("completion_tokens", 0)
            if d["prompt"] + d["completion"] >= self.budget:
                self.stopped = True
        with self.lock:
            with self.ledger_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "key": label, "model": self.model, **u}) + "\n")

    def budget_left(self) -> bool:
        with self.lock:
            return not self.stopped

    def call(self, user_msg: str) -> tuple[str | None, str]:
        """Return (content|None, error_reason)."""
        last_err = "no-attempt"
        for _ in range(len(self.keys) * 3):  # rotate keys across retries
            if not self.budget_left():
                return None, "budget"
            with self.lock:
                key, label = self.keys[self.ki]
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": self.max_completion,
            }
            try:
                resp = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"},
                                     json=body, timeout=120)
                if resp.status_code == 200:
                    r = resp.json()
                    self._spend(label, r.get("usage") or {})
                    content = r["choices"][0]["message"].get("content") or ""
                    return content, ""
                err = f"http{resp.status_code}"
                if resp.status_code in (429, 500, 502, 503, 529):
                    # quota/rate: rotate key and back off
                    with self.lock:
                        self.ki = (self.ki + 1) % len(self.keys)
                    if resp.status_code == 429 and "insufficient" in resp.text.lower():
                        with self.lock:
                            # this key's pool is done for today; try others, else stop
                            if all(v["requests"] > 0 for v in self.usage.values()):
                                self.stopped = True
                    time.sleep(min(30, 2 ** (_ % 5)))
                    last_err = err
                    continue
                return None, err + ":" + resp.text[:200]
            except requests.RequestException as e:
                last_err = f"net:{type(e).__name__}"
                time.sleep(min(30, 2 ** (_ % 5)))
        return None, last_err


# ---------------------------------------------------------------- validation

def parse_teacher_json(content: str) -> tuple[dict | None, str]:
    t = content.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t, flags=re.S)
    try:
        obj = json.loads(t)
    except Exception:
        return None, "parse"
    if not isinstance(obj, dict):
        return None, "not-object"
    return obj, ""


# lat,lng slots are the resolver's job — supervision keeps them null (core: 12/2582)
FORCE_NULL_PARAMS = {"origin", "destination", "via"}


def validate_row(job: dict, teacher_args: dict) -> tuple[dict | None, str]:
    """Ground teacher args -> honest target; drop if nothing real survives."""
    schema = job["schema"]
    props = (schema.get("parameters") or {}).get("properties") or {}
    allowed = set(props)
    extras = [k for k in teacher_args if k not in allowed]
    if extras:
        return None, "keys_mismatch:" + ",".join(sorted(extras)[:3])
    utt_f, utt_t = fold(job["utterance"]), tokens(job["utterance"])
    target = {}
    for p in schema_param_order(schema.get("parameters") or {}, teacher_args):
        if job["fn"] == "here_directions" and p in FORCE_NULL_PARAMS:
            target[p] = None
            continue
        target[p] = param_target(p, props.get(p, {}), teacher_args.get(p), utt_f, utt_t, job["lang"])[0]
    grounded = [k for k, v in target.items() if v is not None and k != "lang"]
    if not grounded:
        return None, "no_grounded_value"
    import hashlib
    row = {
        "id": "dst-" + hashlib.sha1(f"{job['utterance']}|{job['fn']}".encode()).hexdigest()[:12],
        "source": job["source"], "lang": job["lang"], "function": schema,
        "utterance": job["utterance"], "args": teacher_args, "target": target,
    }
    return row, ""


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="pl,es,fr")
    ap.add_argument("--sources", default="v3,dialogues")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--max-rows", type=int, default=4000, help="candidate jobs to annotate")
    ap.add_argument("--budget-tokens", type=int, default=1_200_000, help="per-key token cap for this run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-completion", type=int, default=2400)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out-prefix", default="distill")
    ap.add_argument("--fns", default="here_geocode,here_directions",
                    help="comma list of functions to emit jobs for")
    ap.add_argument("--seed", type=int, default=42, help="job-order shuffle seed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    global SYSTEM_PROMPT
    from render import SYSTEM_PROMPT as SP  # noqa: F401
    SYSTEM_PROMPT = SP

    langs = {x.strip() for x in a.langs.split(",") if x.strip()}
    sources = {x.strip() for x in a.sources.split(",") if x.strip()}
    fns = {x.strip() for x in a.fns.split(",") if x.strip()}
    # exclude (utterance, fn) pairs already emitted by existing dataset files
    # other than this run's own prefix -> new-seed reruns stay disjoint
    skip_pairs: set[tuple[str, str]] = set()
    skip_names: list[str] = []
    for pat in ("*_train.jsonl", "*_val.jsonl"):
        for p in sorted(OUT.glob(pat)):
            if p.name.startswith(a.out_prefix):
                continue
            skip_names.append(p.name)
            for line in p.open():
                r = json.loads(line)
                fn = r["function"]["name"] if isinstance(r.get("function"), dict) else None
                skip_pairs.add((fold(r["utterance"])[:160], fn))
    jobs, info = build_jobs(langs, sources, a.max_rows, fns, a.seed, skip_pairs)
    print(f"[pool] {dict(info)} -> jobs={len(jobs)} "
          f"skip_files={skip_names} "
          f"langs={dict(Counter(j['lang'] for j in jobs))} "
          f"fns={dict(Counter(j['fn'] for j in jobs))}")
    if a.dry_run or not jobs:
        return 0

    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    teacher = Teacher(load_keys(), a.model, a.budget_tokens, a.max_completion)
    rejects_path = LOGS / "distill_rejects.jsonl"
    rows: list[dict] = []
    stats = Counter()

    def work(job: dict):
        if job.get("teacher") is False:
            row, verr = validate_row(job, {"q": job["geo_q"], "lang": job["lang"]})
            return ("ok" if row else "reject", verr if verr else "", job, row or None)
        if not teacher.budget_left():
            return None
        schema_json = json.dumps(job["schema"], ensure_ascii=False, separators=(",", ":"))
        user_msg = f"FUNCTION: {schema_json}\n\nUSER_MESSAGE: {job['utterance']}"
        content, err = teacher.call(user_msg)
        if content is None:
            return ("err", err, job, None)
        targs, perr = parse_teacher_json(content)
        if targs is None:
            return ("err", perr, job, content)
        row, verr = validate_row(job, targs)
        if row is None:
            return ("reject", verr, job, content)
        return ("ok", "", job, row)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(work, jobs):
            if res is None:
                stats["skipped_budget"] += 1
                continue
            kind, reason, job, payload = res
            stats[kind] += 1
            if kind != "ok":
                with rejects_path.open("a") as f:
                    f.write(json.dumps({"reason": reason, "lang": job["lang"], "fn": job["fn"],
                                        "utterance": job["utterance"][:160],
                                        "teacher": (payload or "")[:400] if isinstance(payload, str)
                                        else None}, ensure_ascii=False) + "\n")
                continue
            rows.append(payload)
            if stats["ok"] % 200 == 0:
                print(f"  ok={stats['ok']} reject={stats['reject']} err={stats['err']} "
                      f"usage={ {k: v['prompt'] + v['completion'] for k, v in teacher.usage.items()} }")

    # dedup (same text may appear via directions+geocode or near-dup texts)
    seen, deduped = set(), []
    for r in rows:
        k = fold(r["utterance"])[:160] + "|" + r["function"]["name"]
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    rows = deduped
    random.Random(7).shuffle(rows)
    n_val = min(max(20, int(a.val_frac * len(rows))), len(rows) // 2)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    for name, rr in ((f"{a.out_prefix}_train", train_rows), (f"{a.out_prefix}_val", val_rows)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from build_extraction import stats as mk_stats
    st = mk_stats(train_rows, f"{a.out_prefix}_train")
    st["teacher"] = {"model": a.model, "outcome_counts": dict(stats),
                     "usage": teacher.usage}
    (OUT / f"{a.out_prefix}_train_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"[done] {dict(stats)} train={len(train_rows)} val={len(val_rows)} "
          f"langs={st['per_lang']} usage={teacher.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
