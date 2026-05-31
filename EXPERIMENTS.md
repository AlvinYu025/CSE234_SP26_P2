# Project 2 Experiment Log

This log records the main validation experiments run for schema linking. Scores
are from `eval.py` on `validation_input.json` against
`validation_gold_schema_links.json`.

## Final Best Run

```text
Name: coder_dbbal12_lr8p5e-5_ep7
Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Method: pruned schema SFT + light database-balanced upsampling
Training file: coder_balanced_pruned_sft.py
Inference file: main.py
Adapter:
  rapidfire_experiments/project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_retry_detached_len8192/runs/1/checkpoints/final_checkpoint
Output:
  output/qwen-qwen2.5-coder-1.5b-instruct/sft_pruned_coder_dbbal12_lr8p5e-5_r16_a32_ep7_retry_detached_len8192/
```

Validation score:

```text
Leaderboard: 0.6869
Table:       0.7753
Column:      0.5985
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
- LoRA rank `16`, alpha `32`, dropout `0.05`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`.
- Learning rate `8.5e-5`.
- Scheduler `cosine`.
- Warmup ratio `0.05`.
- Epochs `7`.
- Max length `8192`.
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
dbbal12_ep4         0.6536       0.7254  0.5819  Best non-Coder run
dbbal14_ep4         0.6251       0.7135  0.5366  More upsampling than target=12 was worse
coder_dbbal12_ep6   0.6735       0.7524  0.5945  Coder base, lower LR, longer run
coder_lr7p5e-5_ep7  0.6787       0.7621  0.5952  Lower LR and 7 epochs
coder_lr8p5e-5_ep7  0.6869       0.7753  0.5985  Best result; local LR search around 7.5e-5
coder_lr9e-5_ep7    0.6531       0.7351  0.5711  Higher LR regressed sharply
coder_lr2e-4_ep6    0.6686       0.7411  0.5960  Friend-style higher LR; did not improve
coder_multicol_b3   0.6636       0.7393  0.5880  Upweighted complex answers; improved some small DBs, hurt NYSED/NTSB
deepseek_coder_ep7  0.5984       0.7138  0.4831  DeepSeek-Coder 1.3B, same setup; column score too low
deepseek_coder_ep2  0.5593       0.6618  0.4569  Short early-stop check; worse than ep7
qwen3_1.7b_ep7      n/a          n/a     n/a     Training stopped before a usable adapter was written
```

## Commands

Best training command:

```bash
python coder_balanced_pruned_sft.py train \
  --run_name sft_pruned_coder_dbbal12_lr8p5e-5_r16_a32_ep7_len8192 \
  --experiment_name project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_len8192 \
  --learning_rates 0.000085 \
  --epochs 7
```

Best validation command:

```bash
python main.py \
  --input validation_input.json \
  --output output/qwen-qwen2.5-coder-1.5b-instruct/sft_pruned_coder_dbbal12_lr8p5e-5_r16_a32_ep7_retry_detached_len8192/predictions.json \
  --schemas_dir schemas \
  --gold validation_gold_schema_links.json
```

## Interpretation

The best early improvement came from changing the training distribution, not
from adding prompt rules or extra schema candidates. The training set is small
and database-specific. Some SBODemoUS databases had only 6 training examples,
so the model under-learned their table and column patterns. Light upsampling to
12 examples per database improved both table and column scores.

Over-balancing did not help. `target=18` and `target=14` both performed worse
than `target=12`, which suggests that repeated small-database examples can
start to distort column selection. The final setting keeps all original data
and applies only mild reweighting through repeated examples.

Switching the base model to `Qwen2.5-Coder-1.5B-Instruct` improved both table
and column scores when paired with the same pruned and balanced data. The best
Coder run used `8.5e-5`, LoRA alpha `32`, max length `8192`, and 7 epochs.

The complex-answer upweighting run repeated rows with at least 5 gold columns
or 3 gold tables. It increased the weighted training set from 349 to 506 rows.
This helped some SBODemoUS databases, but it lowered the overall score because
NYSED_SRC2022, NTSB, and ASIS_20161108_HerpInv_Database regressed.

Two alternative sub-2B bases were checked after the best Coder run. Qwen3-1.7B
did not finish cleanly, so it was not scored. DeepSeek-Coder-1.3B-Instruct
trained and evaluated cleanly, but both 7 epochs and a shorter 2-epoch run were
well below the Qwen2.5-Coder-1.5B result.
