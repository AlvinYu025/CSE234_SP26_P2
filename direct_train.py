"""Direct TRL SFT trainer for the schema-linking LoRA adapter.

Used instead of the RapidFire AI fit path, which leaves the LoRA adapter
non-trainable in this environment (grad_norm stays 0 and the saved lora_B
weights are all zero). Consumes the prepared prompt/completion JSONL and writes
a single LoRA adapter that main.py and the *_sft.py predict paths load unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def loadRows(path: str) -> Dataset:
    # prompt/completion only: TRL reads this as a prompt-completion dataset and
    # masks the prompt tokens from the loss (completion_only_loss).
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    return Dataset.from_list(rows)


def buildParser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True)
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=8.5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--maxLength", type=int, default=8192)
    ap.add_argument("--gradAccum", type=int, default=8)
    ap.add_argument("--warmupRatio", type=float, default=0.05)
    ap.add_argument("--evalSteps", type=int, default=20)
    ap.add_argument("--loggingSteps", type=int, default=5)
    return ap


def runTrain(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainData = loadRows(args.train_jsonl)
    evalData = loadRows(args.eval_jsonl)
    print(f"[direct_train] train={len(trainData)} eval={len(evalData)} -> {args.out}", flush=True)

    peftConfig = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    sftConfig = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradAccum,
        warmup_ratio=args.warmupRatio,
        lr_scheduler_type="cosine",
        logging_steps=args.loggingSteps,
        eval_strategy="steps",
        eval_steps=args.evalSteps,
        bf16=True,
        fp16=False,
        packing=False,
        completion_only_loss=True,
        max_length=args.maxLength,
        gradient_checkpointing=False,
        report_to="none",
        save_strategy="no",
    )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        args=sftConfig,
        train_dataset=trainData,
        eval_dataset=evalData,
        processing_class=tokenizer,
        peft_config=peftConfig,
    )

    # Guard against the rapidfireai failure mode: the adapter must be trainable.
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[direct_train] trainable params = {trainable:,}", flush=True)
    assert trainable > 0, "no trainable params -> LoRA not attached"

    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[direct_train] done -> {args.out}", flush=True)


if __name__ == "__main__":
    runTrain(buildParser().parse_args())
