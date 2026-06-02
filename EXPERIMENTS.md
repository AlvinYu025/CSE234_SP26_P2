# Project 2 Experiments

Scores are from `eval.py` on `validation_input.json` and
`validation_gold_schema_links.json`.

## Final Submitted Setting

```text
Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Adapter: LoRA r16 alpha32 dropout0.05
Training: completion-only SFT, 7 epochs, lr 8.5e-5
Data: released 301 rows, database-balanced to 349 rows with target 12
Inference: one adapter with default, wide, all-columns, and recall views
```

Validation:

```text
Leaderboard: 0.7150
Table:       0.7869
Column:      0.6431
```

The single-adapter run before view voting scored 0.6948 on the same saved
generation set. View voting improved the column score by letting wider schema
views suggest additional candidates while requiring two-view agreement for
columns.

## Representative Runs

| Run | Main change | Table | Column | Overall |
|---|---|---:|---:|---:|
| No SFT | prompt-only Qwen2.5 1.5B | 0.4113 | 0.2579 | 0.3346 |
| Completion only | JSON target tokens only | 0.6523 | 0.4729 | 0.5626 |
| Pruned schema | remove most schema distractors | 0.7049 | 0.5644 | 0.6347 |
| Balanced to 12 | mild database upsampling | 0.7254 | 0.5819 | 0.6536 |
| Coder base | Qwen2.5-Coder 1.5B | 0.7524 | 0.5945 | 0.6735 |
| Best adapter | 7 epochs, lr 8.5e-5 | 0.7741 | 0.6156 | 0.6948 |
| Final view vote | one adapter, four inference views | 0.7869 | 0.6431 | 0.7150 |

## Notes

Completion-only SFT was the first large improvement. It focused the loss on the
JSON answer instead of the prompt text.

Schema pruning helped because full schemas contain many plausible but wrong
columns. The pruning step keeps the prompt shorter and lowers column noise.

Database balancing was mild by design. Databases with fewer than 12 examples
were upsampled to 12, but larger databases were not downsampled because their
rows cover distinct questions and schema patterns.

The final inference step keeps the official packaging simple: only one adapter
is required. The view vote uses different schema contexts at inference time,
then keeps tables by union across the reliable views and keeps columns only
when at least two views agree.
