# CSE234 Project 2

Schema linking for natural language questions over database schemas.

## Run Inference

```bash
python main.py \
  --input validation_input.json \
  --output predictions.json \
  --schemas_dir schemas
```

For local validation:

```bash
python main.py \
  --input validation_input.json \
  --output predictions.json \
  --schemas_dir schemas \
  --gold validation_gold_schema_links.json
```

`main.py` writes a list of predictions in this format:

```json
[
  {
    "question_id": 1,
    "schema_links": {
      "TableName": ["ColumnName"]
    }
  }
]
```

## Method

The final method uses:

- `Qwen/Qwen2.5-1.5B-Instruct`
- pruned schema prompting
- RapidFire SFT
- LoRA rank `16`, alpha `16`, dropout `0.05`
- learning rate `2e-4`
- 4 epochs
- completion-only loss
- database-balanced upsampling with target `12`

The database balancing step repeats examples from small databases until each
database has at least 12 training rows. It does not downsample larger
databases.

## Best Validation Result

```text
Leaderboard score: 0.6536
Table score:       0.7254
Column score:      0.5819
```

The best adapter path used locally was:

```text
rapidfire_experiments/project2_qwen25_15b_sft_pruned_dbbal12_r16a16_ep4_len6144/runs/2/checkpoints/final_checkpoint
```

The checkpoint is not committed in this repository.

## Important Files

```text
main.py                  inference entry point
balanced_pruned_sft.py   final training script
pruned_sft.py            schema pruning and SFT utilities
sft.py                   shared SFT utilities
eval.py                  validation scoring script
schemas/                 database schemas
train.json               training data
validation_input.json    validation questions
validation_gold_schema_links.json
EXPERIMENTS.md           experiment notes and scores
```

## Training Command

```bash
python balanced_pruned_sft.py train \
  --run_name sft_pruned_dbbal12_lr2e-4_r16_a16_ep4_len6144 \
  --experiment_name project2_qwen25_15b_sft_pruned_dbbal12_r16a16_ep4_len6144 \
  --max_length 6144 \
  --prompt_keep_head_tokens 800 \
  --top_tables 14 \
  --max_schema_columns 320 \
  --max_columns_per_table 48 \
  --db_balance_target 12 \
  --learning_rates 0.0002 \
  --lora_ranks 16 \
  --lora_alphas 16 \
  --lora_dropout 0.05 \
  --epochs 4 \
  --completion_only_loss
```

More experiment details are in `EXPERIMENTS.md`.
