# Best single-adapter config so far

| rank | config | Table | Column | Leaderboard | vs 0.7150 |
|---|---|---:|---:|---:|---:|
| 1 | **mixed_ep2 + recombine(table_thr=1,col_thr=2)** | 0.8094 | 0.6623 | **0.7358** | +0.0208 |
| 2 | mixed_ep2 t2c2 | 0.8045 | 0.6623 | 0.7334 | +0.0184 |
| 3 | mixed_ep2 t1c1 (verifier-off) | 0.8094 | 0.6461 | 0.7278 | +0.0128 |
| - | two_stage_ep7 (single view) | 0.7677 | 0.6592 | 0.7134 | -0.0016 |

Winner artifact: adapters_direct/mixed_ep2 (single LoRA). Inference: mixed_task_sft.py predict (NO --use_verifier) then recombine with --table_vote_threshold 1 --column_vote_threshold 2.
