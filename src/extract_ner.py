"""Extract multilingual NER spans (LOC/PER) for NER-hint Experiment B.

Uses transformers.pipeline token-classification with Davlan/xlm-roberta-base-ner-hrl
(no venv changes). Runs over core_train + aug_train + core_test utterances (deduped by
folded utterance), keeps LOC first then PER, deduped, capped at 12 spans.
Writes data/ner_spans.json {row_id: [span, ...]} + prints a 20-row spot check.

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python src/extract_ner.py --gpu 0
  # or: --cpu (slow fallback for when both GPUs are full)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_ID = "Babelscape/wikineural-multilingual-ner"  # Davlan repo needs sentencepiece (absent in venv)
KEEP_LABELS = {"LOC", "PER"}
FILES = ["data/core_train.jsonl", "data/aug_train.jsonl", "data/core_test.jsonl"]
OUT = ROOT / "data" / "ner_spans.json"


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None, choices=[0, 1])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="dev smoke limit")
    a = ap.parse_args()
    if not a.cpu and a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from transformers import pipeline

    device = -1 if a.cpu else 0
    dtype = torch.float32 if a.cpu else torch.float16
    ner = pipeline("token-classification", model=MODEL_ID, device=device,
                   torch_dtype=dtype)  # aggregation_strategy passed at CALL time (v5.x ctor kwarg ignored)

    # collect unique utterances
    utt_by_id: dict[str, str] = {}
    for rel in FILES:
        with open(ROOT / rel) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                utt_by_id[d["id"]] = d["utterance"]
    uniq: dict[str, str] = {}
    for rid, u in utt_by_id.items():
        uniq.setdefault(fold(u), u)
    texts = list(uniq.values())
    if a.limit:
        texts = texts[:a.limit]
    print(f"rows={len(utt_by_id)} unique_utterances={len(texts)} device={'cpu' if a.cpu else 'gpu'}")

    t0 = time.time()
    spans_by_fold: dict[str, list[str]] = {}
    for s in range(0, len(texts), a.batch_size):
        batch = texts[s:s + a.batch_size]
        outs = ner(batch, aggregation_strategy="simple")
        for txt, ents in zip(batch, outs):
            def clean(w):
                w = w.strip().replace(" - ", "-").replace("##", "")
                return w if 2 <= len(w) <= 60 and not w.startswith("#") else None
            locs = [c for e in ents if e.get("entity_group") == "LOC"
                    for c in [clean(e["word"])] if c]
            pers = [c for e in ents if e.get("entity_group") == "PER"
                    for c in [clean(e["word"])] if c]
            seen, ordered = set(), []
            for spn in locs + pers:  # LOC first
                k = fold(spn)
                if k and k not in seen:
                    seen.add(k)
                    ordered.append(spn)
            spans_by_fold[fold(txt)] = ordered[:12]
        if (s // a.batch_size) % 50 == 0:
            print(f"  batch {s // a.batch_size + 1}/{(len(texts) + a.batch_size - 1) // a.batch_size}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    result = {rid: spans_by_fold.get(fold(u), []) for rid, u in utt_by_id.items()}
    n_with = sum(1 for v in result.values() if v)
    OUT.write_text(json.dumps(result, ensure_ascii=False))
    print(f"wrote {OUT}: {len(result)} ids, {n_with} with >=1 span "
          f"({time.time() - t0:.0f}s)")

    # spot check: 4 rows per lang
    per_lang: dict[str, list] = {}
    for rid, u in utt_by_id.items():
        pass
    rows = []
    for rel in FILES:
        with open(ROOT / rel) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    rows.append(d)
    by_lang: dict[str, list] = {}
    for d in rows:
        by_lang.setdefault(d.get("lang") or "?", []).append(d)
    print("\n--- spot check (4 per lang) ---")
    for lang in sorted(by_lang):
        import random
        rnd = random.Random(7)
        for d in rnd.sample(by_lang[lang], min(4, len(by_lang[lang]))):
            print(f"[{lang}] {d['utterance'][:90]!r}")
            print(f"      spans: {result.get(d['id'], [])[:8]}")


if __name__ == "__main__":
    sys.exit(main())
