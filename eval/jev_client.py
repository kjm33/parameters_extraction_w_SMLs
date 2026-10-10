"""Minimal OpenRouter 'System One' (Jev decision model) client.

Shared by eval/jev_enum_probe.py and eval/jev_verify.py. Loads OPENROUTER_API_KEY
from the environment or .env files (repo root / parent), posts typed questions
(choice/noul/score, all evaluated in parallel against one shared state) to the
alpha decisions endpoint with retry/backoff, and provides a deterministic mock
mode for offline plumbing tests (JEV_MOCK=1 or mock=True).

Endpoint: POST https://openrouter.ai/api/alpha/decisions
Body:   {"model": ..., "state": ..., "questions": {name: {type, instructions, criteria}}}
Answer: {"model": ..., "answers": {name: {...}}, "usage": {input_tokens, output_tokens, cost}}

Usage:
  from jev_client import JevClient
  cli = JevClient(model="typesafe/jev-1.13")
  res = cli.ask(state={"utterance": "..."},
                questions={"lang": {"type": "choice", "instructions": "...",
                                    "criteria": {"en": "English"}}})
  p = res["answers"]["lang"]["probabilities"]["en"]
  results = cli.run_many([(state1, qs1), (state2, qs2)])   # ThreadPool, in order
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RETRY_STATUS = {402, 408, 429, 500, 502, 503, 504}


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


class JevClient:
    def __init__(self, model: str = "typesafe/jev-1.13", mock: bool = False,
                 workers: int = 8, max_retries: int = 6, timeout: int = 60):
        self.model = model
        self.mock = mock or os.environ.get("JEV_MOCK") == "1"
        self.workers = workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        if not self.mock:
            self.key = load_key()

    # --------------------------------------------------------------- mock
    def _mock_answer(self, name: str, q: dict, seed: str) -> dict:
        h = hashlib.md5((seed + "|" + name).encode()).digest()
        r = h[0] / 255.0
        if q.get("type") == "choice":
            labels = sorted(q.get("criteria") or {})
            if not labels:
                return {}
            pick = labels[h[1] % len(labels)]
            probs = {l: round(1.0 - i * 0.2 * r, 4) for i, l in enumerate(sorted(
                labels, key=lambda x: -abs(hash(x + seed) % 97)))}
            s = sum(probs.values()) or 1.0
            return {"choice": pick, "confidence": round(r, 4),
                    "probabilities": {k: round(v / s, 4) for k, v in probs.items()}}
        if q.get("type") == "noul":
            return {"noul": round(r, 4)}
        if q.get("type") == "score":
            return {"score": h[1] % 5 + 1, "confidence": round(r, 4)}
        return {}

    # --------------------------------------------------------------- call
    def ask(self, state, questions: dict) -> dict:
        body = {"model": self.model, "state": state, "questions": questions}
        if self.mock:
            seed = hashlib.md5(json.dumps(state, sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()[:12]
            return {"model": self.model, "mock": True,
                    "answers": {n: self._mock_answer(n, q, seed)
                                for n, q in questions.items()},
                    "usage": {"input_tokens": 500, "output_tokens": 0, "cost": 0.0}}
        delay = 0.5
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(ENDPOINT,
                                  headers={"Authorization": f"Bearer {self.key}",
                                           "Content-Type": "application/json"},
                                  json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = f"network: {e}"
            else:
                if r.status_code == 200:
                    out = r.json()
                    u = out.get("usage") or {}
                    self.usage["calls"] += 1
                    self.usage["input_tokens"] += u.get("input_tokens") or 0
                    self.usage["output_tokens"] += u.get("output_tokens") or 0
                    self.usage["cost"] += u.get("cost") or 0.0
                    return out
                try:
                    msg = r.json().get("error", {}).get("message", r.text[:200])
                except Exception:
                    msg = r.text[:200]
                last = f"HTTP {r.status_code}: {msg}"
                if r.status_code not in RETRY_STATUS:
                    raise RuntimeError(f"jev call failed: {last}")
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
            if attempt < self.max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError(f"jev call failed after {self.max_retries} attempts: {last}")

    # --------------------------------------------------------------- batch
    def run_many(self, tasks, progress_every: int = 50) -> list:
        """tasks: list of (state, questions) -> list of result dicts (input order)."""
        out: list = [None] * len(tasks)
        errors: list = [None] * len(tasks)
        done = 0
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.ask, s, q): i for i, (s, q) in enumerate(tasks)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    out[i] = fut.result()
                except Exception as e:
                    errors[i] = str(e)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  {done}/{len(tasks)} ({time.time() - t0:.0f}s, "
                          f"{sum(e is not None for e in errors)} failed)", flush=True)
        n_err = sum(e is not None for e in errors)
        if n_err:
            first = next(e for e in errors if e)
            print(f"WARNING: {n_err}/{len(tasks)} calls failed; first error: {first}",
                  file=sys.stderr)
        return out
