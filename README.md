# CSE/DSC 234 Project 2

Schema linking for natural-language questions over database schemas.

The submitted inference path is `main.py`. It loads a public Qwen2.5-Coder
1.5B base model, attaches the local LoRA adapter in `adapter/`, prunes the
schema prompt, generates JSON, and filters identifiers against the schema.

## Run

```bash
python3 main.py --input validation_input.json --output predictions.json
```

Optional local scoring:

```bash
python3 main.py \
  --input validation_input.json \
  --output predictions.json \
  --gold validation_gold_schema_links.json
```

`main.py` expects schemas at `./schemas/` and loads `./adapter/` by default.
The base model is downloaded through `transformers`; the adapter is committed
in this repo.

## Output Format

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

If a table is referenced without a specific column, it is emitted with an empty
list.

## Final Setting

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Adapter | LoRA, rank 16, alpha 32, dropout 0.05 |
| Training | completion-only SFT, 7 epochs, lr `8.5e-5` |
| Data | 301 train rows, upsampled to 349 rows with database target 12 |
| Context | pruned schema, up to 14 tables and 320 columns |
| Decoding | greedy generation, schema filtering |

Validation score:

| Metric | Score |
|---|---:|
| Overall | 0.6869 |
| Table | 0.7753 |
| Column | 0.5985 |

## Training

The final training script is `coder_balanced_pruned_sft.py`.

```bash
python3 coder_balanced_pruned_sft.py train \
  --run_name sft_pruned_coder_dbbal12_lr8p5e-5_r16_a32_ep7_len8192 \
  --experiment_name project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_len8192 \
  --learning_rates 0.000085 \
  --epochs 7
```

Representative experiment results are summarized in `EXPERIMENTS.md`; raw logs
are under `logs/`.
