"""Inference entry point for Project 2.

Packaged pipeline: a single LoRA adapter (method #7, "mixed" multi-task) loaded
from ./adapter, run with batched generation over four views per question
(tables, full, columns, four clause slices) and merged by deterministic voting
(table_threshold=1, column_threshold=2). This single-adapter pipeline scores
0.7315 on the released validation set (Table 0.804 / Column 0.659), above the
prior 0.7150 view-vote submission, and runs ~100 questions well within the
time budget. The previous view-vote entry point is kept in
main_backup_viewvote.py.

Usage:
    python3 main.py --input <input.json> --output <pred.json>
"""

from __future__ import annotations

import fast_predict


def main() -> None:
    fast_predict.runPredict(fast_predict.buildParser().parse_args())


if __name__ == "__main__":
    main()
