# CSE/DSC 234 Project 2

Team 51

This project builds a local schema-linking model. Given a question and a
database schema, `main.py` predicts the tables and columns that the SQL answer
would reference.

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

`main.py` runs the final inference pipeline end to end:

1. Load `Qwen/Qwen2.5-Coder-1.5B-Instruct` with the local LoRA adapter in
   `adapter/`.
2. Build pruned schema prompts for each input question using the schema files
   in `schemas/`.
3. Generate four predictions from the same adapter with default, wide,
   all-columns, and recall-oriented schema views.
4. Vote over the four predictions, parse the JSON, remove duplicate links, and
   filter table and column names against the target schema.

## Final Pipeline

| Item | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Adapter | LoRA, rank 16, alpha 32, dropout 0.05 |
| Training | completion-only SFT, 7 epochs, learning rate `8.5e-5` |
| Data | 301 released rows, upsampled to 349 rows with database target 12 |
| Inference | one adapter, four schema views, deterministic voting |
| Output cleanup | JSON parsing, duplicate removal, schema identifier filtering |

The final inference step uses the same adapter with four views of the schema:
default, wide, all-columns, and recall-oriented. Tables are kept if they appear
in any of the first three views. Columns are kept only when at least two views
agree and the table passed the table vote.

Validation result for the final submitted pipeline:

| Metric | Score |
|---|---:|
| Overall | 0.7150 |
| Table | 0.7869 |
| Column | 0.6431 |

## Training

The main training script is `coder_balanced_pruned_sft.py`.

```bash
python3 coder_balanced_pruned_sft.py train \
  --run_name sft_pruned_coder_dbbal12_lr8p5e-5_r16_a32_ep7_len8192 \
  --experiment_name project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_len8192 \
  --learning_rates 0.000085 \
  --epochs 7
```

Representative validation experiments are summarized in `EXPERIMENTS.md`.
RapidFire logs are under `logs/`.

## AI Tools

AI tools were used for code inspection, debugging help, command organization,
and summarizing results from experiment runs. The adapter, predictions, and
reported scores come from local runs with the released evaluator.
