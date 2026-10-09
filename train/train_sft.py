"""LoRA/full SFT for slot-filling extractors (transformers 5.x + peft, single pinned GPU).

Bakes in fcft-verified caveats: CUDA_VISIBLE_DEVICES pinned before torch import;
manual prompt-masked labels (templates lack {% generation %}); no packing (short rows,
dynamic padding instead).

Usage:
  ../function_calling_edge_fine_tuning/train/venv/bin/python train/train_sft.py \
      --candidate qwen2.5-0.5b --gpu 1 [--full] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True, help="key in train/configs/<key>.json")
    p.add_argument("--train", default="data/core_train.jsonl,data/aug_train.jsonl")
    p.add_argument("--val", default="data/core_val.jsonl,data/aug_val.jsonl")
    p.add_argument("--out-dir", default=None, help="default train/runs/<candidate>[-full]")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--micro-bs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--gpu", type=int, default=1, choices=[0, 1])
    p.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    return a


def main():
    args = parse_args()
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback,
                              Trainer, TrainingArguments)
    sys.path.insert(0, str(ROOT / "src"))
    from render import render_ids

    cfg = json.loads((ROOT / "train" / "configs" / f"{args.candidate}.json").read_text())
    model_id = cfg["model_id"]
    tkw = cfg.get("template_kwargs") or {}
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT / "train" / "runs" / (args.candidate + ("-full" if args.full else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"train_{out_dir.name}.log"
    log_path.parent.mkdir(exist_ok=True)

    class Tee:
        def __init__(self, fh):
            self.fh = fh
        def write(self, s):
            sys.__stdout__.write(s)
            self.fh.write(s)
        def flush(self):
            sys.__stdout__.flush()
            self.fh.flush()

    with log_path.open("a") as fh:
        sys.stdout = Tee(fh)
        print(f"=== candidate={args.candidate} model={model_id} full={args.full} gpu={args.gpu} out={out_dir}")

        tok = AutoTokenizer.from_pretrained(model_id)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def load_rows(spec: str) -> list[dict]:
            rows = []
            for part in spec.split(","):
                with open(part) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            return rows

        def encode(rows: list[dict]) -> list[dict]:
            out, dropped = [], 0
            for r in rows:
                ids, labels, _ = render_ids(tok, r, tkw)
                if len(ids) > args.seq_len:
                    dropped += 1
                    continue
                out.append({"input_ids": ids, "labels": labels})
            if dropped:
                print(f"  dropped {dropped} rows > seq_len")
            return out

        train_enc = encode(load_rows(args.train))
        val_enc = encode(load_rows(args.val))
        print(f"train rows={len(train_enc)} val rows={len(val_enc)}")

        def collate(batch):
            maxlen = max(len(b["input_ids"]) for b in batch)
            input_ids, labels, attn = [], [], []
            for b in batch:
                pad = maxlen - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [pad_id] * pad)
                labels.append(b["labels"] + [-100] * pad)
                attn.append([1] * len(b["input_ids"]) + [0] * pad)
            return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                    "attention_mask": torch.tensor(attn)}

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        model.enable_input_require_grads()

        if args.full:
            print("full fine-tune (all params)")
        else:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules="all-linear", task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
            model.print_trainable_parameters()

        est_steps = max(1, int(len(train_enc) / (args.micro_bs * args.grad_accum) * args.epochs))
        warmup_steps = max(10, int(args.warmup_ratio * est_steps))
        targs = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.micro_bs,
            per_device_eval_batch_size=args.micro_bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=True,
            logging_steps=20,
            eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            seed=args.seed,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=Dataset.from_list(train_enc),
            eval_dataset=Dataset.from_list(val_enc),
            data_collator=collate,
            processing_class=tok,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        trainer.train()
        trainer.save_model(str(out_dir / "best"))
        tok.save_pretrained(str(out_dir / "best"))
        print(f"saved best -> {out_dir / 'best'}")


if __name__ == "__main__":
    sys.exit(main())
