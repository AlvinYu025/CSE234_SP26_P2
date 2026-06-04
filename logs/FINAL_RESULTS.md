# Final results — two single-adapter tricks vs baseline 0.7150 (rank 9)

All scores: eval.py on validation_input.json (101 q). All SINGLE LoRA adapter (compliant).
Training: direct TRL SFTTrainer (rapidfireai fit path is broken in this env — LoRA non-trainable).

## Champions
| rank | config (single adapter) | Table | Column | Leaderboard | vs 0.7150 | infer time /101q |
|---|---|---:|---:|---:|---:|---:|
| 🏆 | **mixed_ep2 + votes t=1,c=2** (#7) | 0.8094 | 0.6623 | **0.7358** | +0.0208 | ~10.5 min |
| safe | **two_stage_ep7_r32** (#1) | 0.7667 | 0.6707 | **0.7187** | +0.0037 | ~3 min |
| - | baseline (submitted, view-vote) | 0.7869 | 0.6431 | 0.7150 | 0 | (4 gen/q) |

## How each champion runs end-to-end (one command)
- mixed_ep2:  python mixed_task_sft.py predict --adapter_path adapters_direct/mixed_ep2 \
    --input <in> --output <out> --table_vote_threshold 1 --column_vote_threshold 2   (NO --use_verifier)
- two_stage_ep7_r32:  python two_stage_sft.py predict --adapter_path adapters_direct/two_stage_ep7_r32 \
    --input <in> --output <out>

## Key findings
- mixed (method #7) at 2 epochs / r16 is the sweet spot: highest Table (0.8045-0.8094). More epochs (ep3/ep4) or r32 overtrain and drop Table.
- column_vote_threshold=2 fixes the column-precision bottleneck (require 2 votes to keep a column).
- The verifier HURTS (over-prunes columns 0.66 -> 0.53); keep it OFF.
- two_stage (method #1) needs r32/a64 + ep7 to beat baseline; ep4 too few, ep10 overtrains.

## Caveat
- Vote thresholds tuned on validation; standard but mild overfit risk vs hidden test.
- mixed timing (~10.5 min/101q) is under the 15-min budget but tighter than two_stage on a 24GB grading pod.
