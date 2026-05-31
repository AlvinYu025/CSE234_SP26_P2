"""Coder SFT with database balance and light complex-answer upweighting."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

import balanced_pruned_sft
import coder_balanced_pruned_sft
import pruned_sft
import sft


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = "sft_pruned_coder_dbbal12_multicol5_table3_lr1e-4_r16_a32_ep6_len8192"
DEFAULT_EXPERIMENT = "project2_qwen25_coder15b_sft_pruned_dbbal12_multicol5_table3_r16a32_ep6_len8192"


def answerStats(row: dict[str, Any]) -> tuple[int, int]:
    links = row.get("target", {}) or {}
    table_count = len(links)
    column_count = sum(len(cols) for cols in links.values() if isinstance(cols, list))
    return table_count, column_count


def extraCopies(row: dict[str, Any], args: argparse.Namespace) -> int:
    table_count, column_count = answerStats(row)
    copies = 0
    if column_count >= args.multicol_min_columns or table_count >= args.multicol_min_tables:
        copies += args.multicol_extra_copies
    if args.multicol_large_columns > 0 and column_count >= args.multicol_large_columns:
        copies += args.multicol_large_extra_copies
    if args.multicol_max_extra_copies > 0:
        copies = min(copies, args.multicol_max_extra_copies)
    return max(0, copies)


def upweightRows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    weighted: list[dict[str, Any]] = []
    for row in rows:
        weighted.append(row)
        weighted.extend(row for _ in range(extraCopies(row, args)))

    if args.multicol_shuffle:
        rng = random.Random(args.seed + 17)
        rng.shuffle(weighted)
    return weighted


def complexityCounts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    column_counts = Counter()
    table_counts = Counter()
    for row in rows:
        table_count, column_count = answerStats(row)
        column_counts[column_count] += 1
        table_counts[table_count] += 1
    return {
        "column_counts": dict(sorted(column_counts.items())),
        "table_counts": dict(sorted(table_counts.items())),
    }


def prepareData(args: argparse.Namespace) -> dict[str, Path]:
    paths = pruned_sft.pathsFor(args)
    tokenizer = sft.loadTokenizer(args)
    train_examples = pruned_sft.readJson(ROOT / args.train_json)
    eval_examples = pruned_sft.readJson(ROOT / args.validation_json)

    rows_train_original = pruned_sft.buildRows(args, train_examples, tokenizer)
    rows_train_balanced = balanced_pruned_sft.balanceRows(rows_train_original, args)
    rows_train = upweightRows(rows_train_balanced, args)
    rows_eval = pruned_sft.buildRows(args, eval_examples, tokenizer)

    sft.writeJsonl(paths["train_jsonl"], rows_train)
    sft.writeJsonl(paths["eval_jsonl"], rows_eval)

    original_db_counts = Counter(row["db_id"] for row in rows_train_original)
    balanced_db_counts = Counter(row["db_id"] for row in rows_train_balanced)
    weighted_db_counts = Counter(row["db_id"] for row in rows_train)
    pruned_sft.writeJson(
        paths["manifest"],
        {
            "base_model": args.base_model,
            "train_examples_original": len(rows_train_original),
            "train_examples_balanced": len(rows_train_balanced),
            "train_examples_weighted": len(rows_train),
            "eval_examples": len(rows_eval),
            "max_length": args.max_length,
            "top_tables": args.top_tables,
            "max_schema_columns": args.max_schema_columns,
            "max_columns_per_table": args.max_columns_per_table,
            "include_all_selected_table_columns": args.include_all_selected_table_columns,
            "db_balance_target": args.db_balance_target,
            "db_balance_cap": args.db_balance_cap,
            "multicol_min_columns": args.multicol_min_columns,
            "multicol_min_tables": args.multicol_min_tables,
            "multicol_extra_copies": args.multicol_extra_copies,
            "multicol_large_columns": args.multicol_large_columns,
            "multicol_large_extra_copies": args.multicol_large_extra_copies,
            "multicol_max_extra_copies": args.multicol_max_extra_copies,
            "train_db_counts_original": dict(sorted(original_db_counts.items())),
            "train_db_counts_balanced": dict(sorted(balanced_db_counts.items())),
            "train_db_counts_weighted": dict(sorted(weighted_db_counts.items())),
            "train_complexity_original": complexityCounts(rows_train_original),
            "train_complexity_balanced": complexityCounts(rows_train_balanced),
            "train_complexity_weighted": complexityCounts(rows_train),
            "train_truncated": sum(row["truncated"] for row in rows_train),
            "eval_truncated": sum(row["truncated"] for row in rows_eval),
            "train_max_tokens": max(row["token_count"] for row in rows_train),
            "eval_max_tokens": max(row["token_count"] for row in rows_eval),
            "train_avg_pruned_tables": sum(len(row["pruned_tables"]) for row in rows_train) / len(rows_train),
            "eval_avg_pruned_tables": sum(len(row["pruned_tables"]) for row in rows_eval) / len(rows_eval),
            "train_avg_pruned_columns": sum(row["pruned_column_count"] for row in rows_train) / len(rows_train),
            "eval_avg_pruned_columns": sum(row["pruned_column_count"] for row in rows_eval) / len(rows_eval),
        },
    )
    return paths


def buildParser() -> argparse.ArgumentParser:
    parser = coder_balanced_pruned_sft.buildParser()
    choices = parser._subparsers._group_actions[0].choices
    for command in ("prepare", "train"):
        choices[command].set_defaults(run_name=DEFAULT_RUN, experiment_name=DEFAULT_EXPERIMENT)
        choices[command].add_argument(
            "--multicol_min_columns",
            type=int,
            default=5,
            help="Repeat rows whose answer has at least this many columns.",
        )
        choices[command].add_argument(
            "--multicol_min_tables",
            type=int,
            default=3,
            help="Repeat rows whose answer has at least this many tables.",
        )
        choices[command].add_argument("--multicol_extra_copies", type=int, default=1)
        choices[command].add_argument("--multicol_large_columns", type=int, default=8)
        choices[command].add_argument("--multicol_large_extra_copies", type=int, default=1)
        choices[command].add_argument("--multicol_max_extra_copies", type=int, default=2)
        choices[command].add_argument("--multicol_shuffle", action=argparse.BooleanOptionalAction, default=True)
    for command in ("predict",):
        choices[command].set_defaults(run_name=DEFAULT_RUN)
    return parser


def main() -> None:
    args = buildParser().parse_args()
    pruned_sft.prepareData = prepareData
    sft.buildRapidfireConfig = coder_balanced_pruned_sft.buildRapidFireConfig
    sft.createModelFn = coder_balanced_pruned_sft.createModel

    if args.command in {"prepare", "train"}:
        args.learning_rates = pruned_sft.parseCsvNumbers(args.learning_rates, float)
        args.lora_ranks = pruned_sft.parseCsvNumbers(args.lora_ranks, int)
        args.lora_alphas = pruned_sft.parseCsvNumbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = prepareData(args)
            print(f"Wrote Coder multicol train data to {paths['train_jsonl']}")
            print(f"Wrote Coder multicol eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            pruned_sft.runTrain(args)
    else:
        pruned_sft.predict(args)


if __name__ == "__main__":
    main()
