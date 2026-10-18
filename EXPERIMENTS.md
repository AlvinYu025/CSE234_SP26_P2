# Project 2 Experiments

Scores are from `eval.py` on `validation_input.json` and
`validation_gold_schema_links.json`.

## Final Submitted Setting

```text
Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Adapter:    single LoRA r16 alpha32 dropout0.05  (method "mixed" multi-task)
Training:   completion-only SFT, 2 epochs, lr 8.5e-5  (direct TRL SFTTrainer)
Data:       released 301 rows, database-balanced to 349, expanded to 5 task
            views per example (tables / full / columns / clause / verify)
Inference:  one adapter, batched generation over tables+full+columns+4 clauses,
            deterministic voting (table_threshold=1, column_threshold=2),
            verifier OFF
```

Validation:

```text
Leaderboard: 0.7315
Table:       0.8043
Column:      0.6587
```

The packaged single-adapter "mixed" pipeline scores 0.7315, above the earlier
0.7150 view-vote submission. The multi-task adapter raises the table score to
0.80 while `column_threshold=2` raises column precision; the per-column verifier
was found to over-prune and is disabled.

## Representative Runs

| Run | Main change | Table | Column | Overall |
|---|---|---:|---:|---:|
| No SFT | prompt-only Qwen2.5 1.5B | 0.4113 | 0.2579 | 0.3346 |
| Completion only | JSON target tokens only | 0.6523 | 0.4729 | 0.5626 |
| Pruned schema | remove most schema distractors | 0.7049 | 0.5644 | 0.6347 |
| Coder base | Qwen2.5-Coder 1.5B | 0.7524 | 0.5945 | 0.6735 |
| View-vote (prev submission) | one adapter, four schema views | 0.7869 | 0.6431 | 0.7150 |
| two_stage ep7 | table-stage then column-stage (#1) | 0.7677 | 0.6592 | 0.7134 |
| two_stage ep7 r32 | + larger LoRA | 0.7667 | 0.6707 | 0.7187 |
| mixed ep2 (verifier on) | multi-task adapter (#7) | 0.8045 | 0.5810 | 0.6928 |
| **mixed ep2, votes t1c2** | **multi-task, verifier off, tuned votes** | **0.8094** | **0.6623** | **0.7358** |
| mixed ep2, batched (packaged) | same, batched generation | 0.8043 | 0.6587 | 0.7315 |

## Notes

The two tricks studied were method #1 (single-adapter two-stage: predict tables,
then predict columns on the selected tables) and method #7 (single-adapter
mixed multi-task: tables / full / columns / clause-specific columns / verify
yes-no). Both train one adapter from the same gold schema links — no new labels.

Method #7 won. Training the mixed adapter for 2 epochs gave the highest table
score (0.80); more epochs (ep3/ep4) or a larger rank (r32) overtrained and lost
table score. Column precision — the bottleneck — was fixed at inference by
requiring two view-votes to keep a column (`column_threshold=2`). The verifier
task hurt (it over-pruned columns, 0.66 -> 0.53) and is disabled.

Method #1 (two_stage) needed r32 and 7 epochs to clear 0.7150 (0.7187); it is a
clean single-view alternative kept in `adapters_direct/two_stage_ep7_r32`.

Training note: the RapidFire AI `fit` path could not be used for the final runs
in the rebuilt environment (rapidfireai 0.16 + trl 0.21 / peft 0.19 /
transformers 4.57 leaves the LoRA adapter non-trainable: `grad_norm` stays 0 and
the saved `lora_B` is all zeros). The adapters were therefore trained with a
direct TRL `SFTTrainer` (`direct_train.py`) on the same prepared JSONL, which
produces the same single-LoRA-adapter artifact. Inference, data prep, and
evaluation are unchanged.
