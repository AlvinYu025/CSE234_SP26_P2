# Project 2 Experiment Log

This log records the main validation experiments run for schema linking. Scores
are from `eval.py` on `validation_input.json` against
`validation_gold_schema_links.json`.

## Final Best Run

```text
Name: dbbal12_ep4
Base model: Qwen/Qwen2.5-1.5B-Instruct
Method: pruned schema SFT + light database-balanced upsampling
Training file: balanced_pruned_sft.py
Inference file: main.py
Adapter:
  rapidfire_experiments/project2_qwen25_15b_sft_pruned_dbbal12_r16a16_ep4_len6144/runs/2/checkpoints/final_checkpoint
Output:
  output/qwen-qwen2.5-1.5b-instruct/sft_pruned_dbbal12_lr2e-4_r16_a16_ep4_len6144/
```

Validation score:

```text
Leaderboard: 0.6536
Table:       0.7254
Column:      0.5819
```

## Best Run Settings

Schema representation:

- Use the pruned schema prompt from `pruned_sft.py`.
- Keep original table and column names.
- No alias serialization.
- No all-selected-table-columns expansion.
- Pruning settings: `top_tables=14`, `max_schema_columns=320`,
  `max_columns_per_table=48`.

Training:

- RapidFire SFT.
- Completion-only loss.
- LoRA rank `16`, alpha `16`, dropout `0.05`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`.
- Learning rate `2e-4`.
- Scheduler `cosine`.
- Warmup ratio `0.03`.
- Epochs `4`.
- Max length `6144`.
- Prompt head keep tokens `800`.
- Batch size per device `1`.
- Gradient accumulation steps `8`.
- BF16 enabled.

Data curation:

- Original training examples: `301`.
- Balanced training examples: `349`.
- Each database with fewer than 12 training rows was upsampled to 12 rows.
- Databases already above 12 rows were kept unchanged.
- No downsampling was used, because large-database examples are different
  questions and may cover unique tables, columns, and query patterns.

The resulting database counts were:

```text
ASIS_20161108_HerpInv_Database       24
ATBI                                 24
CratersWildlifeObservations          24
KlamathInvasiveSpecies               24
NTSB                                 60
NYSED_SRC2022                        37
NorthernPlainsFireManagement         24
PacificIslandLandbirds               24
SBODemoUS-Banking                    12
SBODemoUS-Business Partners          12
SBODemoUS-Finance                    12
SBODemoUS-General                    12
SBODemoUS-Human Resources            12
SBODemoUS-Inventory and Production   12
SBODemoUS-Reports                    12
SBODemoUS-Sales Opportunities        12
SBODemoUS-Service                    12
```

## Main Results

```text
Run                 Leaderboard  Table   Column  Notes
base_pruned_ep4     0.6347       0.7049  0.5644  Previous best before database balancing
hybrid_floor8_ep4   0.6191       0.6953  0.5429  Added extra candidate columns; hurt precision
dbbal18_ep4         0.6380       0.7249  0.5511  Stronger upsampling; helped table, hurt column
dbbal18_ep3         0.6016       0.6864  0.5168  Too few epochs for this balanced dataset
dbbal12_ep4         0.6536       0.7254  0.5819  Best result
dbbal14_ep4         0.6251       0.7135  0.5366  More upsampling than target=12 was worse
```

## Commands

Best training command:

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

Best validation command:

```bash
python main.py \
  --input validation_input.json \
  --output output/qwen-qwen2.5-1.5b-instruct/sft_pruned_dbbal12_lr2e-4_r16_a16_ep4_len6144/predictions.json \
  --schemas_dir schemas \
  --gold validation_gold_schema_links.json
```

## Interpretation

The best improvement came from changing the training distribution, not from
adding prompt rules or extra schema candidates. The training set is small and
database-specific. Some SBODemoUS databases had only 6 training examples, so
the model under-learned their table and column patterns. Light upsampling to
12 examples per database improved both table and column scores.

Over-balancing did not help. `target=18` and `target=14` both performed worse
than `target=12`, which suggests that repeated small-database examples can
start to distort column selection. The final setting keeps all original data
and applies only mild reweighting through repeated examples.
