"""Pruned SFT variant with a few extra same-table column candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pruned_sft


ROOT = Path(__file__).resolve().parent
_BASE_PRUNE_SCHEMA = pruned_sft.prune_schema
_CURATION: argparse.Namespace | None = None


def columnOrder(schema: dict[str, Any]) -> dict[tuple[str, str], int]:
    return {
        (table, column): idx
        for table in schema["tables"]
        for idx, column in enumerate(schema["columns_by_table"][table])
    }


def addCandidate(
    candidates: dict[tuple[str, str], float],
    schema: dict[str, Any],
    table: str,
    column: str,
    priority: float,
) -> None:
    if column not in schema["columns_by_table"].get(table, []):
        return
    key = (table, column)
    candidates[key] = min(priority, candidates.get(key, priority))


def curatedPruneSchema(
    question: str,
    db_id: str,
    schema: dict[str, Any],
    train_index: dict[str, Any],
    top_tables: int,
    max_columns: int,
    max_columns_per_table: int,
    include_all_selected_table_columns: bool = False,
    force_links: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    cfg = _CURATION
    if cfg is None:
        return _BASE_PRUNE_SCHEMA(
            question,
            db_id,
            schema,
            train_index,
            top_tables,
            max_columns,
            max_columns_per_table,
            include_all_selected_table_columns,
            force_links,
        )

    base = _BASE_PRUNE_SCHEMA(
        question,
        db_id,
        schema,
        train_index,
        top_tables,
        max_columns,
        max_columns_per_table,
        False,
        force_links,
    )

    table_order = {table: idx for idx, table in enumerate(schema["tables"])}
    column_order = columnOrder(schema)
    selected_tables = list(base["tables"])
    candidates: dict[tuple[str, str], float] = {}

    # Keep every base-pruned column first. In training, force_links are already
    # included by the base pruner, so gold columns stay protected.
    for table in selected_tables:
        for column in base["columns_by_table"].get(table, []):
            addCandidate(candidates, schema, table, column, -1000.0)

    for table in selected_tables:
        columns = schema["columns_by_table"].get(table, [])
        if not columns:
            continue

        if cfg.small_table_all_columns and len(columns) <= cfg.small_table_all_columns:
            for idx, column in enumerate(columns):
                addCandidate(candidates, schema, table, column, 10.0 + idx / 1000.0)

        if cfg.column_neighbor_window > 0:
            base_columns = list(base["columns_by_table"].get(table, []))
            for column in base_columns:
                center = column_order.get((table, column))
                if center is None:
                    continue
                lo = max(0, center - cfg.column_neighbor_window)
                hi = min(len(columns), center + cfg.column_neighbor_window + 1)
                for idx in range(lo, hi):
                    distance = abs(idx - center)
                    addCandidate(candidates, schema, table, columns[idx], 20.0 + distance)

        if cfg.table_prior_columns > 0:
            priors = train_index["column_priors"].get(db_id, {})
            ranked = [
                (pair, count)
                for pair, count in priors.items()
                if pair[0] == table and pair[1] in columns
            ]
            ranked.sort(key=lambda item: (-item[1], column_order.get(item[0], 10**9)))
            for rank, ((_, column), _count) in enumerate(ranked[: cfg.table_prior_columns]):
                addCandidate(candidates, schema, table, column, 30.0 + rank / 1000.0)

        if cfg.table_column_floor > 0:
            for idx, column in enumerate(columns[: cfg.table_column_floor]):
                addCandidate(candidates, schema, table, column, 40.0 + idx / 1000.0)

    final_max_columns = cfg.curated_max_schema_columns or max_columns
    final_max_per_table = cfg.curated_max_columns_per_table or max_columns_per_table
    kept_by_table: dict[str, list[str]] = defaultdict(list)

    ranked_candidates = sorted(
        candidates.items(),
        key=lambda item: (
            item[1],
            table_order.get(item[0][0], 10**9),
            column_order.get(item[0], 10**9),
        ),
    )
    for (table, column), priority in ranked_candidates:
        # Never drop base-pruned columns due to the per-table cap. Only cap
        # lower-priority curated additions.
        if priority > -1000.0 and len(kept_by_table[table]) >= final_max_per_table:
            continue
        if sum(len(cols) for cols in kept_by_table.values()) >= final_max_columns and priority > -1000.0:
            continue
        if column not in kept_by_table[table]:
            kept_by_table[table].append(column)

    ordered_columns = {}
    for table in sorted(selected_tables, key=lambda value: table_order.get(value, 10**9)):
        ordered_columns[table] = sorted(
            kept_by_table.get(table, []),
            key=lambda column: column_order.get((table, column), 10**9),
        )

    return {
        "db_id": schema["db_id"],
        "tables": list(ordered_columns),
        "columns_by_table": ordered_columns,
        "types_by_pair": schema["types_by_pair"],
    }


def buildParser() -> argparse.ArgumentParser:
    parser = pruned_sft.build_arg_parser()
    for command in ("prepare", "train", "predict"):
        sp = parser._subparsers._group_actions[0].choices[command]
        sp.add_argument("--table_column_floor", type=int, default=0)
        sp.add_argument("--column_neighbor_window", type=int, default=0)
        sp.add_argument("--table_prior_columns", type=int, default=0)
        sp.add_argument("--small_table_all_columns", type=int, default=0)
        sp.add_argument("--curated_max_schema_columns", type=int, default=320)
        sp.add_argument("--curated_max_columns_per_table", type=int, default=64)
    return parser


def main() -> None:
    global _CURATION
    args = buildParser().parse_args()
    _CURATION = args
    pruned_sft.prune_schema = curatedPruneSchema
    if args.command in {"prepare", "train"}:
        args.learning_rates = pruned_sft.parse_csv_numbers(args.learning_rates, float)
        args.lora_ranks = pruned_sft.parse_csv_numbers(args.lora_ranks, int)
        args.lora_alphas = pruned_sft.parse_csv_numbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = pruned_sft.prepare_data(args)
            print(f"Wrote curated train data to {paths['train_jsonl']}")
            print(f"Wrote curated eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            pruned_sft.run_train(args)
    else:
        pruned_sft.predict(args)


if __name__ == "__main__":
    main()
