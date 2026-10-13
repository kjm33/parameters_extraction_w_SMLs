"""Flair entity-span dump: predict entity spans over dataset utterances (CPU).

License note (checked 2026-10-12 on the HF model cards): the Flair 2.0 'entity-*'
models (entity-{english,german,multilingual}-{4,8}class) are Flukes NC 1.0 --
NONCOMMERCIAL; commercial use requires a separate license. This CONTRADICTS the
research note's '2.0 entity-* = commercial OK' claim. The classic 1.0 'ner-*'
models are also noncommercial. -> this dump is MEASUREMENT ONLY; do not ship
predictions from either model generation in a commercial pipeline.

Loading shim: the 'entity-*' hub checkpoints use a new container format
({'__cls__': 'TokenClassifier', nested 'state_dict', dropout floats, no
'embeddings' config}) that flair 0.15.1's SequenceTagger.load cannot parse
(KeyError 'use_crf'). model_card records flair_version 0.15.1 and a deberta
backbone per model (multi=mdeberta-v3-base, de=ikim-uk-essen/geberta-base,
en=microsoft/deberta-v3-base; identified via vocab size + key names), so we
rebuild a TransformerWordEmbeddings object for a candidate backbone, construct
TokenClassifier and strict-load the nested state_dict (first candidate whose
keys+shapes match wins).

Runs under the ISOLATED flair venv, not the training venv:
  HF_HOME=/var/tmp/fcft_hf_cache /var/tmp/flair_venv/bin/python \
    eval/flair_spans.py --files data/core_test.jsonl \
    --models flair/entity-multilingual-4class,flair/entity-german-4class,flair/entity-english-4class
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_extraction import fold  # noqa: E402

OUT_DIR = ROOT / "data" / "ner"

# candidate backbone per model-family substring; resolved by strict state_dict match
BACKBONES: dict[str, list[str]] = {
    "multilingual": ["microsoft/mdeberta-v3-base"],
    "german": ["ikim-uk-essen/geberta-base"],
    "english": ["microsoft/deberta-v3-base"],
}
BACKBONE_FALLBACK = [b for lst in BACKBONES.values() for b in lst]


def rpath(part: str) -> pathlib.Path:
    p = pathlib.Path(part)
    return p if p.is_absolute() else ROOT / p


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_entity_model(model_id: str):
    """Load a Flair 2.0 entity-* hub checkpoint into a flair TokenClassifier."""
    import torch
    from huggingface_hub import hf_hub_download

    repo = model_id if "/" in model_id else f"flair/{model_id}"
    path = hf_hub_download(repo_id=repo, filename="pytorch_model.bin")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "use_crf" in state:  # classic 1.x SequenceTagger format
        from flair.models import SequenceTagger
        return SequenceTagger.load(str(path)), "ner"

    import flair
    from flair.embeddings import TransformerWordEmbeddings
    from flair.models import TokenClassifier

    flair.device = torch.device("cpu")
    family = next((f for f in BACKBONES if f in model_id.lower()), None)
    candidates = BACKBONES.get(family, []) + [
        b for b in BACKBONE_FALLBACK if b not in BACKBONES.get(family, [])
    ]
    sd = state["state_dict"]
    label_dict, label_type = state["label_dictionary"], state.get("label_type", "ner")
    last_err: Exception | None = None
    for backbone in candidates:
        try:
            embeddings = TransformerWordEmbeddings(backbone, fine_tune=True)
            model = TokenClassifier._init_model_with_state_dict(
                {**state, "embeddings": embeddings})
            model.load_state_dict(sd)  # strict: validates the backbone guess
            model.eval()
            print(f"loaded {model_id} (backbone={backbone}, "
                  f"labels={len(label_dict.get_items())}, strict ok)")
            return model, label_type
        except Exception as e:  # try next backbone candidate
            last_err = e
    raise RuntimeError(f"no backbone matched {model_id}: {last_err}")


def clean_spans(spans: list) -> list[dict]:
    best: dict = {}
    for sp in spans:
        text = sp.text
        if not fold(text):
            continue
        lab = sp.get_label("ner")
        entry = {"text": text, "label": lab.value, "score": round(float(lab.score), 4),
                 "start": int(sp[0].start_position), "end": int(sp[-1].end_position)}
        key = (entry["start"], entry["end"])
        if key not in best or entry["score"] > best[key]["score"]:
            best[key] = entry
    return [best[k] for k in sorted(best)]


def main():
    ap = argparse.ArgumentParser(
        description="Flair entity-span dump over dataset utterances (measurement only; "
                    "entity-* models are Flukes NC 1.0 noncommercial)")
    ap.add_argument("--files", default="data/core_test.jsonl",
                    help="comma-separated dataset files")
    ap.add_argument("--models", default="flair/entity-multilingual-4class",
                    help="comma-separated hub model ids (flair/entity-*)")
    ap.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)),
                    help=f"output directory (default {OUT_DIR.relative_to(ROOT)})")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="prediction mini-batch size")
    ap.add_argument("--chunk", type=int, default=128,
                    help="sentences per predict() call (memory bound)")
    args = ap.parse_args()

    from flair.data import Sentence
    from flair.tokenization import SegtokTokenizer

    fr_tok = SegtokTokenizer(additional_split_characters=["'"])
    parts = args.files.split(",")
    rows: list[dict] = []
    for part in parts:
        rows.extend(read_jsonl(rpath(part)))
    stems = "_".join(pathlib.Path(p).stem for p in parts)
    out_dir = rpath(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models.split(","):
        t0 = time.time()
        model, label_type = load_entity_model(model_id.strip())
        modelshort = model_id.strip().split("/")[-1]
        out = out_dir / f"flair_{modelshort}_{stems}_spans.jsonl"
        sentences = [Sentence(r["utterance"],
                              use_tokenizer=fr_tok if r.get("lang") == "fr" else True)
                     for r in rows]
        label_hist: Counter = Counter()
        n_with = 0
        with open(out, "w") as f:
            for i in range(0, len(sentences), args.chunk):
                batch = sentences[i:i + args.chunk]
                model.predict(batch, mini_batch_size=args.batch_size)
                for row, sent in zip(rows[i:i + args.chunk], batch):
                    spans = clean_spans(sent.get_spans(label_type))
                    label_hist.update(s["label"] for s in spans)
                    n_with += int(bool(spans))
                    f.write(json.dumps({"id": row["id"], "lang": row.get("lang"),
                                        "spans": spans}, ensure_ascii=False) + "\n")
        total = sum(label_hist.values())
        print(f"{model_id}: rows={len(rows)} spans={total} "
              f"rows_with_spans={n_with}/{len(rows)} elapsed={time.time() - t0:.0f}s")
        print("label_histogram=" + json.dumps(dict(label_hist.most_common()),
                                              ensure_ascii=False))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
