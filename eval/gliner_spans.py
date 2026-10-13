"""GLiNER span dump: predict entity spans over dataset utterances (CPU).

Runs GLiNER multi-v2.1 over every row's utterance and writes one jsonl row per
dataset row: {"id", "lang", "spans": [{"text","label","score","start","end"}]}
(char offsets into the utterance, sorted by start; deduped by (start,end)
keeping the max score; spans whose text folds to empty dropped). The threshold
is kept LOW on purpose -- this dump is a coverage-oriented span pool; consumers
(eval/gliner_arbitrate.py) filter by score themselves.

Runs under the ISOLATED GLiNER venv, not the training venv:
  HF_HOME=/var/tmp/fcft_hf_cache /var/tmp/gliner_venv/bin/python \
    eval/gliner_spans.py --files data/core_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import fold  # noqa: E402

MODEL_ID = "urchade/gliner_multi-v2.1"
DEFAULT_LABELS = "city,street address,point of interest,location,category"


def rpath(part: str) -> pathlib.Path:
    p = pathlib.Path(part)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def default_out(parts: list[str]) -> pathlib.Path:
    stems = "_".join(pathlib.Path(p).stem for p in parts)
    return ROOT / "data" / "ner" / f"gliner_{stems}_spans.jsonl"


def clean_spans(entities: list[dict]) -> list[dict]:
    best: dict = {}
    for e in entities:
        if not fold(e.get("text") or ""):
            continue
        key = (e["start"], e["end"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e
    return [{"text": e["text"], "label": e["label"], "score": round(e["score"], 4),
             "start": e["start"], "end": e["end"]}
            for _k, e in sorted(best.items())]


def main():
    ap = argparse.ArgumentParser(
        description="GLiNER entity-span dump over dataset utterances")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="comma-separated dataset files")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="GLiNER predict threshold (keep low; consumers filter)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="output jsonl (default data/ner/gliner_<stem>_spans.jsonl)")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="comma-separated entity labels")
    args = ap.parse_args()
    parts = args.files.split(",")
    out = rpath(args.out) if args.out else default_out(parts)
    out.parent.mkdir(parents=True, exist_ok=True)

    from gliner import GLiNER
    model = GLiNER.from_pretrained(MODEL_ID)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    rows: list[dict] = []
    for part in parts:
        rows.extend(read_jsonl(rpath(part)))
    texts = [r["utterance"] for r in rows]
    batch = model.batch_predict_entities(texts, labels, threshold=args.threshold,
                                         batch_size=args.batch_size)
    label_hist: Counter = Counter()
    n_with = 0
    with open(out, "w") as f:
        for row, ents in zip(rows, batch):
            spans = clean_spans(ents)
            label_hist.update(s["label"] for s in spans)
            n_with += int(bool(spans))
            f.write(json.dumps({"id": row["id"], "lang": row.get("lang"),
                                "spans": spans}, ensure_ascii=False) + "\n")
    total = sum(label_hist.values())
    print(f"{MODEL_ID} threshold={args.threshold} labels={labels}")
    print(f"{args.files}: rows={len(rows)} spans={total} "
          f"rows_with_spans={n_with}/{len(rows)}")
    print("label_histogram=" + json.dumps(dict(label_hist.most_common()),
                                          ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
