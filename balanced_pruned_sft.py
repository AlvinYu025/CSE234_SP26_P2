"""Train pruned SFT with light database-balanced sampling."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pruned_sft
import sft


ROOT = Path(__file__).resolve().parent


def balanceRows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    target = max(0, int(args.db_balance_target))
    cap = max(0, int(args.db_balance_cap))
    if target <= 0 and cap <= 0:
        return rows

    rng = random.Random(args.seed)
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)

    balanced: list[dict[str, Any]] = []
    for db_id in sorted(by_db):
        group = list(by_db[db_id])
        if cap > 0 and len(group) > cap:
            group = rng.sample(group, cap)
        if target > 0 and len(group) < target:
            # Repeat small-database rows instead of dropping large-database rows.
            group.extend(rng.choice(group) for _ in range(target - len(group)))
        balanced.extend(group)

    if args.db_balance_shuffle:
        rng.shuffle(balanced)
    return balanced


def prepareData(args: argparse.Namespace) -> dict[str, Path]:
    paths = pruned_sft.pathsFor(args)
    tokenizer = sft.loadTokenizer(args)
    train_examples = pruned_sft.readJson(ROOT / args.train_json)
    eval_examples = pruned_sft.readJson(ROOT / args.validation_json)
    rows_train_original = pruned_sft.buildRows(args, train_examples, tokenizer)
    rows_train = balanceRows(rows_train_original, args)
    rows_eval = pruned_sft.buildRows(args, eval_examples, tokenizer)

    sft.writeJsonl(paths["train_jsonl"], rows_train)
    sft.writeJsonl(paths["eval_jsonl"], rows_eval)

    train_db_counts = Counter(row["db_id"] for row in rows_train)
    original_db_counts = Counter(row["db_id"] for row in rows_train_original)
    pruned_sft.writeJson(
        paths["manifest"],
        {
            "base_model": args.base_model,
            "train_examples_original": len(rows_train_original),
            "train_examples_balanced": len(rows_train),
            "eval_examples": len(rows_eval),
            "max_length": args.max_length,
            "top_tables": args.top_tables,
            "max_schema_columns": args.max_schema_columns,
            "max_columns_per_table": args.max_columns_per_table,
            "include_all_selected_table_columns": args.include_all_selected_table_columns,
            "db_balance_target": args.db_balance_target,
            "db_balance_cap": args.db_balance_cap,
            "train_db_counts_original": dict(sorted(original_db_counts.items())),
            "train_db_counts_balanced": dict(sorted(train_db_counts.items())),
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
    parser = pruned_sft.buildArgParser()
    for command in ("prepare", "train"):
        sp = parser._subparsers._group_actions[0].choices[command]
        sp.add_argument(
            "--db_balance_target",
            type=int,
            default=0,
            help="Upsample each database to at least this many training rows. 0 disables upsampling.",
        )
        sp.add_argument(
            "--db_balance_cap",
            type=int,
            default=0,
            help="Optionally downsample databases above this count. 0 disables downsampling.",
        )
        sp.add_argument("--db_balance_shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = buildParser().parse_args()
    pruned_sft.prepareData = prepareData
    if args.command in {"prepare", "train"}:
        args.learning_rates = pruned_sft.parseCsvNumbers(args.learning_rates, float)
        args.lora_ranks = pruned_sft.parseCsvNumbers(args.lora_ranks, int)
        args.lora_alphas = pruned_sft.parseCsvNumbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = prepareData(args)
            print(f"Wrote balanced train data to {paths['train_jsonl']}")
            print(f"Wrote pruned eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            pruned_sft.runTrain(args)
    else:
        pruned_sft.predict(args)


if __name__ == "__main__":
    main()
