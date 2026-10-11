"""Train the Track-B encoder tagger (round 4): JointBERT-style slot filling.

Backbone: bert-base-multilingual-cased (via cached Babelscape/wikineural-multilingual-ner
weights). Input = [CLS] schema-string [SEP] utterance [SEP] — parameter names (and enum
choices) injected before the utterance so attention aligns words to parameters (idea 1).
Heads: token classification over BIO(param[:enum]) labels on the first subtoken of each
utterance word + a 6-way lang head (en/de/pl/fr/es/null) on [CLS] for schemas with a lang
param. Decoding reuses decode_tags from src/build_tags (identical contract to Track A).

Usage (venv python, GPU):
  ../function_calling_edge_fine_tuning/train/venv/bin/python train/train_tagger.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_tags import props_of  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import AutoModel  # noqa: E402

BASE_ID = "Babelscape/wikineural-multilingual-ner"
LANGS = ["en", "de", "pl", "fr", "es", "null"]
MAX_LEN = 256


def schema_string(row) -> str:
    parts = []
    for k, v in props_of(row).items():
        enum = v.get("enum") or (v.get("items") or {}).get("enum")
        parts.append(f"{k}={'|'.join(map(str, enum))}" if enum else k)
    return ", ".join(parts)


def load_rows(paths):
    rows = []
    for p in paths:
        with open(ROOT / p) as f:
            rows.extend(json.loads(ln) for ln in f if ln.strip())
    return rows


def build_label_vocab(rows):
    labels = ["O"]
    seen = set()
    for r in rows:
        for t in r["bio_tags"]:
            if t.startswith(("B-", "I-")) and t not in seen:
                seen.add(t)
                labels.append(t)
    return labels


def encode(tok, row, label2id):
    """Manual [CLS] schema [SEP] subwords [SEP]; first-subtoken labels for utterance words."""
    schema_ids = tok(row.get("_schema_str") or schema_string(row),
                     add_special_tokens=False)["input_ids"]
    words = row["utterance"].split()
    word_ids = []
    utt_ids = []
    for wi, w in enumerate(words):
        ids = tok(w, add_special_tokens=False)["input_ids"] or tok("[UNK]")["input_ids"]
        word_ids.extend([wi] + [None] * (len(ids) - 1))
        utt_ids.extend(ids)
    cls, sep = tok.cls_token_id, tok.sep_token_id
    ids = [cls] + schema_ids[: MAX_LEN - len(utt_ids) - 3] + [sep] + utt_ids + [sep]
    n_utt = len(utt_ids)
    labels = [-100] * len(ids)
    # label position of first subtoken of word wi (offset inside the utterance segment)
    first_pos = {}
    for pos, wi in enumerate(word_ids):
        if wi is not None and wi not in first_pos:
            first_pos[wi] = pos
    base = 2 + len(schema_ids[: MAX_LEN - len(utt_ids) - 3])  # after [CLS] schema [SEP]
    for wi, pos in first_pos.items():
        if base + pos < len(ids) - 1:  # before final [SEP]
            has_lang_tag = row["bio_tags"] and str(row["bio_tags"][0]).startswith("lang:")
            t = row["bio_tags"][wi + 1] if has_lang_tag else row["bio_tags"][wi]
            labels[base + pos] = label2id.get(t, 0)
    lang_lab = -100
    if "lang" in row["target"]:
        v = row["target"]["lang"]
        lang_lab = LANGS.index("null" if v is None else v)
    return {"input_ids": ids, "labels": labels, "lang": lang_lab,
            "n_utt": n_utt, "n_words": len(words),
            "first_pos": first_pos, "base": base}


class Joint(nn.Module):
    def __init__(self, base_id, n_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_id)
        hid = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.tag_head = nn.Linear(hid, n_labels)
        self.lang_head = nn.Linear(hid, len(LANGS))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        return self.tag_head(self.drop(h)), self.lang_head(self.drop(h[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--train", default="data/core_train_tags.jsonl,data/aug_train_tags.jsonl")
    ap.add_argument("--val", default="data/core_val_tags.jsonl,data/aug_val_tags.jsonl")
    ap.add_argument("--out", default="train/runs/bert-tagger")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lang-weight", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    train_rows = load_rows(a.train.split(","))
    val_rows = load_rows(a.val.split(","))
    if a.limit:
        train_rows, val_rows = train_rows[:a.limit], val_rows[: max(8, a.limit // 8)]
    # precompute schema strings once (deterministic, cheap)
    for r in train_rows + val_rows:
        r["_schema_str"] = schema_string(r)
    labels = build_label_vocab(train_rows)
    label2id = {t: i for i, t in enumerate(labels)}
    print(f"rows train={len(train_rows)} val={len(val_rows)} labels={len(labels)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Joint(a.base, len(labels)).to(device)

    def encode_all(rows):
        enc = [encode(tok, r, label2id) for r in rows]
        return enc

    train_enc, val_enc = encode_all(train_rows), encode_all(val_rows)

    def batches(enc, bs, shuffle):
        idx = list(range(len(enc)))
        if shuffle:
            random.shuffle(idx)
        for s in range(0, len(idx), bs):
            chunk = [enc[i] for i in idx[s:s + bs]]
            mx = max(len(c["input_ids"]) for c in chunk)
            ids = torch.full((len(chunk), mx), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), mx), dtype=torch.long)
            lab = torch.full((len(chunk), mx), -100, dtype=torch.long)
            lang = torch.full((len(chunk),), -100, dtype=torch.long)
            for j, c in enumerate(chunk):
                n = len(c["input_ids"])
                ids[j, :n] = torch.tensor(c["input_ids"])
                att[j, :n] = 1
                lab[j, :n] = torch.tensor(c["labels"])
                lang[j] = c["lang"]
            yield ids.to(device), att.to(device), lab.to(device), lang.to(device)

    @torch.no_grad()
    def val_tag_acc():
        model.eval()
        hit = tot = 0
        for ids, att, lab, _ in batches(val_enc, 64, False):
            logits, _ = model(ids, att)
            pred = logits.argmax(-1)
            m = lab != -100
            hit += (pred[m] == lab[m]).sum().item()
            tot += m.sum().item()
        model.train()
        return hit / max(1, tot)

    steps = int(a.epochs * (len(train_enc) + a.bs - 1) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    tag_lossf = nn.CrossEntropyLoss(ignore_index=-100)
    lang_lossf = nn.CrossEntropyLoss(ignore_index=-100)

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, step, t0 = -1.0, 0, time.time()
    for epoch in range(int(a.epochs) + 1):
        if step >= steps:
            break
        for ids, att, lab, lang in batches(train_enc, a.bs, True):
            tag_logits, lang_logits = model(ids, att)
            loss = tag_lossf(tag_logits.transpose(1, 2), lab)
            if (lang != -100).any():
                loss = loss + a.lang_weight * lang_lossf(lang_logits, lang)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step >= steps:
                break
        acc = val_tag_acc()
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "base": a.base,
                        "labels": labels, "langs": LANGS}, out_dir / "best.pt")
            marker = " *"
        print(f"epoch {epoch + 1}: val_tag_acc={acc:.4f}{marker} "
              f"({time.time() - t0:.0f}s, step {step}/{steps})", flush=True)
    (out_dir / "labels.json").write_text(json.dumps({"labels": labels, "langs": LANGS,
                                                     "base": a.base}, ensure_ascii=False))
    print(f"best val_tag_acc={best_acc:.4f} -> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
