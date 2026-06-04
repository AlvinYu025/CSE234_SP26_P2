# CSE/DSC 234 Project 2

Team 51

This project builds a local schema-linking model. Given a question and a
database schema, `main.py` predicts the tables and columns that the SQL answer
would reference.

## Setup

Inference needs only PyTorch, Transformers, and PEFT:

```bash
pip install torch transformers peft accelerate
```

The base model `Qwen/Qwen2.5-Coder-1.5B-Instruct` is public (not gated), so no
Hugging Face token is required; it is downloaded from the Hub on first run. The
trained LoRA adapter (~70 MB) ships in `adapter/` and needs no external storage.
A CUDA GPU is used automatically when available (bf16); the adapter and schemas
load from the repository root.

## Run

```bash
python3 main.py --input validation_input.json --output predictions.json
```

For local validation:

```bash
python3 main.py \
  --input validation_input.json \
  --output predictions.json \
  --gold validation_gold_schema_links.json
```

`main.py` expects `schemas/` and `adapter/` at the repository root. The public
base model is loaded from Hugging Face, and the local LoRA adapter is loaded
from `adapter/`.

## What `main.py` Does

`main.py` runs the final inference pipeline end to end (a thin wrapper over
`fast_predict.py`):

1. Load `Qwen/Qwen2.5-Coder-1.5B-Instruct` with the single local LoRA adapter in
   `adapter/`.
2. Build a pruned schema prompt for each input question using the schema files
   in `schemas/`.
3. Generate, in batches across questions, four task views from the same adapter:
   tables, full schema-links, columns (on the selected tables), and four
   clause-specific column views (select / join / filter / group-by-order).
4. Merge the views by deterministic voting (keep a table with one vote, keep a
   column only with two votes), parse the JSON, and filter identifiers against
   the target schema.

## Final Pipeline

| Item | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Adapter | single LoRA, rank 16, alpha 32, dropout 0.05 (multi-task "mixed") |
| Training | completion-only SFT, 2 epochs, learning rate `8.5e-5` (direct TRL) |
| Data | 301 released rows, balanced to 349, expanded to 5 task views each |
| Inference | one adapter, batched 4-view generation, deterministic voting (t=1, c=2) |
| Output cleanup | JSON parsing, duplicate removal, schema identifier filtering |

The pipeline uses a single multi-task adapter. Tables are kept when they appear
in any view; columns are kept only when at least two views agree, which raises
column precision. The per-column verifier was found to over-prune and is off.

Validation result for the submitted pipeline (~2.5 min for 101 questions):

| Metric | Score |
|---|---:|
| Overall | 0.7315 |
| Table | 0.8043 |
| Column | 0.6587 |

The previous 0.7150 view-vote pipeline is preserved as `main_backup_viewvote.py`
and `adapter_backup_viewvote/`.

## Training

The submitted adapter is the multi-task ("mixed") adapter. Data is prepared with
`mixed_task_sft.py prepare` (five task views per example: tables / full /
columns / clause / verify), then the single LoRA adapter is trained with a direct
TRL `SFTTrainer`:

```bash
python3 mixed_task_sft.py prepare
python3 direct_train.py \
  --train_jsonl data/sft_pruned/qwen-qwen2.5-coder-1.5b-instruct/mixed_coder_dbbal12_lr8p5e-5_r16_a32_ep1_len8192/train_sft.jsonl \
  --eval_jsonl  data/sft_pruned/qwen-qwen2.5-coder-1.5b-instruct/mixed_coder_dbbal12_lr8p5e-5_r16_a32_ep1_len8192/validation_sft.jsonl \
  --out adapter --epochs 2 --lr 8.5e-5 --rank 16 --alpha 32
```

`direct_train.py` is used instead of the RapidFire AI `fit` path because, in the
rebuilt environment (rapidfireai 0.16 + trl 0.21 / peft 0.19 / transformers
4.57), the RapidFire SFT path leaves the LoRA adapter non-trainable (`grad_norm`
stays 0.0 and the saved `lora_B` weights are all zero). The direct TRL trainer
produces the same single-LoRA-adapter artifact. Representative validation
experiments are summarized in `EXPERIMENTS.md`.

## AI Tools

AI tools were used for code inspection, debugging help, command organization,
and summarizing results from experiment runs. The adapter, predictions, and
reported scores come from local runs with the released evaluator.
