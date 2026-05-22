"""
Pruned-schema SFT and inference for Project 2.

This file is intentionally separate from sft.py. It keeps the same answer
format and the same original table/column names, but reduces large schemas
before prompting the model. No alias serialization is used.

Typical usage:

    conda activate rapidfire

    python pruned_sft.py train \
      --run_name sft_pruned_lr2e-4_r16_a16_ep2_len6144 \
      --experiment_name project2_qwen25_15b_sft_pruned_r16a16_len6144

    python pruned_sft.py predict \
      --adapter_path rapidfire_experiments/.../runs/1/checkpoints/final_checkpoint \
      --output output/qwen-qwen2.5-1.5b-instruct/sft_pruned.../predictions.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import baseline
import sft


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


STOPWORDS = {
    "a", "about", "above", "after", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "been", "before", "between", "both", "by", "can",
    "code", "codes", "count", "counts", "did", "display", "do", "does",
    "each", "entries", "entry", "exclude", "for", "from", "give", "has",
    "have", "having", "how", "in", "include", "including", "is", "it",
    "least", "less", "list", "make", "many", "more", "most", "name",
    "names", "not", "null", "number", "numbers", "of", "on", "one",
    "only", "or", "order", "ordered", "per", "record", "records", "result",
    "results", "return", "show", "should", "than", "that", "the", "their",
    "them", "there", "these", "those", "to", "total", "value", "values",
    "was", "were", "what", "where", "which", "with", "without", "no", "num",
}

SYNONYMS = {
    "average": {"avg", "mean", "average"},
    "avg": {"avg", "mean", "average"},
    "amount": {"amount", "amt", "sum", "total"},
    "category": {"category", "cat", "class", "type"},
    "class": {"class", "category", "type"},
    "coordinate": {"coordinate", "coord", "utm", "x", "y"},
    "coordinates": {"coordinate", "coord", "utm", "x", "y"},
    "date": {"date", "dt"},
    "description": {"description", "desc", "text"},
    "descriptions": {"description", "desc", "text"},
    "document": {"document", "doc"},
    "documents": {"document", "doc"},
    "employee": {"employee", "emp"},
    "employees": {"employee", "emp"},
    "event": {"event", "evt"},
    "events": {"event", "evt"},
    "id": {"id", "identifier"},
    "identifier": {"id", "identifier"},
    "injury": {"injury", "inj"},
    "injuries": {"injury", "inj"},
    "location": {"location", "loc"},
    "locations": {"location", "loc"},
    "maximum": {"maximum", "max"},
    "minimum": {"minimum", "min"},
    "note": {"note", "notes", "remark", "remarks"},
    "notes": {"note", "notes", "remark", "remarks"},
    "number": {"number", "num", "no"},
    "percent": {"percent", "pct", "rate"},
    "percentage": {"percent", "pct", "rate"},
    "quantity": {"quantity", "qty"},
    "rate": {"rate", "percent", "pct"},
    "remark": {"remark", "remarks", "note", "notes"},
    "remarks": {"remark", "remarks", "note", "notes"},
    "school": {"school", "sch"},
    "schools": {"school", "sch"},
    "species": {"species", "sp"},
    "station": {"station", "stn"},
    "stations": {"station", "stn"},
    "type": {"type", "category", "class"},
    "vehicle": {"vehicle", "veh"},
    "vehicles": {"vehicle", "veh"},
    "year": {"year", "yr"},
}

KNOWN_IDENTIFIER_WORDS = {
    "abs", "action", "adaptive", "age", "air", "airbag", "amount", "area",
    "avg", "bank", "body", "business", "card", "case", "category", "class",
    "code", "condition", "coord", "count", "crash", "credit", "date",
    "debit", "dept", "desc", "doc", "email", "emp", "employee", "entity",
    "error", "event", "font", "grade", "group", "height", "id", "injury",
    "item", "level", "loc", "location", "make", "max", "min", "model",
    "name", "no", "note", "notes", "open", "partner", "percent", "point",
    "price", "project", "qty", "quantity", "rate", "remark", "remarks",
    "school", "seat", "service", "solution", "species", "station", "sum",
    "text", "time", "tire", "total", "trans", "type", "utm", "veh",
    "vehicle", "volume", "wind", "x", "y", "year",
}


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def expand_token(token: str) -> set[str]:
    base = singularize(token.lower())
    out = {base}
    out.update(SYNONYMS.get(base, set()))
    return out


def split_identifier(identifier: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(identifier))
    raw = re.findall(r"[A-Za-z]+|\d+", text.replace("_", " ").replace("-", " "))
    tokens = []
    for part in raw:
        low = part.lower()
        if low:
            tokens.append(low)
        for word in KNOWN_IDENTIFIER_WORDS:
            if len(word) >= 2 and word in low and word != low:
                tokens.append(word)
    return tokens


def tokenize_text(text: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[A-Za-z]+|\d+", str(text).lower()):
        if singularize(raw) in STOPWORDS:
            continue
        for token in expand_token(raw):
            if token and token not in STOPWORDS:
                tokens.add(token)
    return tokens


def identifier_terms(identifier: str) -> set[str]:
    terms = set()
    for token in split_identifier(identifier):
        for expanded in expand_token(token):
            if expanded and expanded not in STOPWORDS:
                terms.add(expanded)
    compact = re.sub(r"[^A-Za-z0-9]+", "", str(identifier)).lower()
    if compact:
        terms.add(compact)
    return terms


def lexical_score(question_tokens: set[str], identifier_token_set: set[str], identifier: str) -> float:
    if not identifier_token_set:
        return 0.0
    overlap = question_tokens & identifier_token_set
    score = len(overlap) / math.sqrt(max(1, len(identifier_token_set))) if overlap else 0.0
    compact = re.sub(r"[^A-Za-z0-9]+", "", str(identifier)).lower()
    for token in sorted(question_tokens):
        if len(token) >= 3 and compact and token in compact:
            score += 0.35
        if len(token) >= 4:
            for ident_token in identifier_token_set:
                if len(ident_token) >= 4 and (token in ident_token or ident_token in token):
                    score += 0.20
                    break
    return score


def load_schema_with_terms(schemas_dir: str | Path, db_id: str) -> dict[str, Any]:
    schema = sft.load_schema(Path(schemas_dir), db_id)
    schema["table_terms"] = {table: identifier_terms(table) for table in schema["tables"]}
    schema["column_terms"] = {
        (table, column): identifier_terms(column)
        for table in schema["tables"]
        for column in schema["columns_by_table"][table]
    }
    return schema


def build_training_index(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_db = defaultdict(list)
    table_token_counts = defaultdict(Counter)
    column_token_counts = defaultdict(Counter)
    table_priors = defaultdict(Counter)
    column_priors = defaultdict(Counter)
    for item in examples:
        db_id = item.get("db_id")
        links = item.get("schema_links", {})
        if not db_id or not isinstance(links, dict):
            continue
        tokens = tokenize_text(item.get("question", ""))
        by_db[db_id].append({"tokens": tokens, "links": links})
        for table, cols in links.items():
            table_priors[db_id][table] += 1
            if isinstance(cols, list):
                for col in cols:
                    column_priors[db_id][(table, col)] += 1
        for token in sorted(tokens):
            for table, cols in links.items():
                table_token_counts[(db_id, token)][table] += 1
                if isinstance(cols, list):
                    for col in cols:
                        column_token_counts[(db_id, token)][(table, col)] += 1
    return {
        "by_db": dict(by_db),
        "table_token_counts": table_token_counts,
        "column_token_counts": column_token_counts,
        "table_priors": table_priors,
        "column_priors": column_priors,
    }


def add_training_signals(
    db_id: str,
    question_tokens: set[str],
    train_index: dict[str, Any],
    table_scores: defaultdict[str, float],
    column_scores: defaultdict[tuple[str, str], float],
) -> None:
    for token in sorted(question_tokens):
        for table, count in train_index["table_token_counts"].get((db_id, token), Counter()).most_common(10):
            table_scores[table] += min(0.45, 0.08 * count)
        for pair, count in train_index["column_token_counts"].get((db_id, token), Counter()).most_common(20):
            column_scores[pair] += min(0.35, 0.06 * count)


def prune_schema(
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
    question_tokens = tokenize_text(question)
    table_scores: defaultdict[str, float] = defaultdict(float)
    column_scores: defaultdict[tuple[str, str], float] = defaultdict(float)
    table_order = {table: idx for idx, table in enumerate(schema["tables"])}
    column_order = {
        (table, column): idx
        for table in schema["tables"]
        for idx, column in enumerate(schema["columns_by_table"][table])
    }

    for table in schema["tables"]:
        if table.lower() == "sysdiagrams":
            continue
        score = lexical_score(question_tokens, schema["table_terms"][table], table)
        if score > 0:
            table_scores[table] += 1.25 * score

    for table, columns in schema["columns_by_table"].items():
        if table.lower() == "sysdiagrams":
            continue
        table_hint = table_scores.get(table, 0.0)
        for column in columns:
            score = lexical_score(question_tokens, schema["column_terms"][(table, column)], column)
            if score > 0:
                column_scores[(table, column)] += score + min(0.35, 0.12 * table_hint)

    add_training_signals(db_id, question_tokens, train_index, table_scores, column_scores)

    table_candidates = sorted(
        table_scores.items(),
        key=lambda item: (-item[1], table_order.get(item[0], 10**9)),
    )
    column_candidates = sorted(
        column_scores.items(),
        key=lambda item: (-item[1], table_order.get(item[0][0], 10**9), column_order.get(item[0], 10**9)),
    )

    selected_tables = {table for table, _score in table_candidates[:top_tables]}
    selected_columns: dict[str, list[str]] = defaultdict(list)
    for (table, column), _score in column_candidates[:max_columns]:
        if len(selected_columns[table]) < max_columns_per_table:
            selected_columns[table].append(column)
            selected_tables.add(table)

    if force_links:
        for table, columns in force_links.items():
            if table in schema["columns_by_table"]:
                selected_tables.add(table)
                for column in columns:
                    if column in schema["columns_by_table"][table] and column not in selected_columns[table]:
                        selected_columns[table].append(column)

    if not selected_tables:
        priors = train_index["table_priors"].get(db_id, Counter())
        for table, _count in priors.most_common(1):
            if table in schema["columns_by_table"]:
                selected_tables.add(table)

    if include_all_selected_table_columns:
        for table in selected_tables:
            selected_columns[table] = list(schema["columns_by_table"][table])

    pruned_columns = {}
    for table in sorted(selected_tables, key=lambda value: table_order.get(value, 10**9)):
        columns = selected_columns.get(table, [])
        columns = sorted(set(columns), key=lambda column: column_order.get((table, column), 10**9))
        pruned_columns[table] = columns

    return {
        "db_id": schema["db_id"],
        "tables": list(pruned_columns),
        "columns_by_table": pruned_columns,
        "types_by_pair": schema["types_by_pair"],
    }


def serialize_pruned_schema(schema: dict[str, Any]) -> str:
    lines = [f"Database: {schema['db_id']}", "Schema:"]
    for table in schema["tables"]:
        columns = schema["columns_by_table"].get(table, [])
        lines.append(f"- {table}({', '.join(columns)})")
    return "\n".join(lines)


def build_rows(args: argparse.Namespace, examples: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    schema_cache = {}
    train_index = build_training_index(read_json(ROOT / args.train_json))
    rows = []
    for item in examples:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema_with_terms(ROOT / args.schemas_dir, db_id)
        full_schema = schema_cache[db_id]
        target_links = sft.canonical_schema_links(item.get("schema_links", {}), full_schema)
        pruned = prune_schema(
            item.get("question", ""),
            db_id,
            full_schema,
            train_index,
            args.top_tables,
            args.max_schema_columns,
            args.max_columns_per_table,
            args.include_all_selected_table_columns,
            force_links=target_links,
        )
        schema_text = serialize_pruned_schema(pruned)
        system, user = sft.build_prompt(item.get("question", ""), schema_text)
        assistant = json.dumps(target_links, ensure_ascii=False, separators=(",", ":"))
        answer_text = assistant + (getattr(tokenizer, "eos_token", None) or "")
        prompt_text = sft.apply_chat_prompt_template(tokenizer, system, user)
        prompt, completion, token_count, was_truncated = sft.fit_prompt_and_answer(
            tokenizer,
            prompt_text,
            answer_text,
            max_length=args.max_length,
            keep_head_tokens=args.prompt_keep_head_tokens,
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion,
                "question_id": item["question_id"],
                "db_id": db_id,
                "target": target_links,
                "pruned_tables": pruned["tables"],
                "pruned_column_count": sum(len(cols) for cols in pruned["columns_by_table"].values()),
                "token_count": token_count,
                "truncated": was_truncated,
            }
        )
    return rows


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    model_slug = sft.slugify(args.base_model)
    run_name = sft.slugify(args.run_name)
    data_dir = ROOT / args.data_dir / model_slug / run_name
    run_dir = ROOT / args.output_root / model_slug / run_name
    return {
        "data_dir": data_dir,
        "train_jsonl": data_dir / "train_sft.jsonl",
        "eval_jsonl": data_dir / "validation_sft.jsonl",
        "manifest": data_dir / "manifest.json",
        "run_dir": run_dir,
        "grid_config": run_dir / "rapidfire_grid.json",
        "results": run_dir / "rapidfire_results.csv",
        "runs_info": run_dir / "rapidfire_runs_info.csv",
        "run_config": run_dir / "run_config.json",
        "experiment_dir": ROOT / args.experiment_path / args.experiment_name,
    }


def prepare_data(args: argparse.Namespace) -> dict[str, Path]:
    paths = paths_for(args)
    tokenizer = sft.load_tokenizer(args)
    train_examples = read_json(ROOT / args.train_json)
    eval_examples = read_json(ROOT / args.validation_json)
    rows_train = build_rows(args, train_examples, tokenizer)
    rows_eval = build_rows(args, eval_examples, tokenizer)
    sft.write_jsonl(paths["train_jsonl"], rows_train)
    sft.write_jsonl(paths["eval_jsonl"], rows_eval)
    write_json(
        paths["manifest"],
        {
            "base_model": args.base_model,
            "train_examples": len(rows_train),
            "eval_examples": len(rows_eval),
            "max_length": args.max_length,
            "top_tables": args.top_tables,
            "max_schema_columns": args.max_schema_columns,
            "max_columns_per_table": args.max_columns_per_table,
            "include_all_selected_table_columns": args.include_all_selected_table_columns,
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


def run_train(args: argparse.Namespace) -> None:
    paths = prepare_data(args)
    param_config = sft.build_rapidfire_config(args)
    grid_summary = sft.summarize_runs(param_config, args.seed)
    write_json(paths["grid_config"], grid_summary)
    train_dataset = sft.load_dataset_from_jsonl(paths["train_jsonl"])
    eval_dataset = sft.load_dataset_from_jsonl(paths["eval_jsonl"])

    from rapidfireai import Experiment

    experiment = Experiment(
        args.experiment_name,
        mode="fit",
        experiment_path=str(ROOT / args.experiment_path),
    )
    experiment.run_fit(
        param_config=param_config,
        create_model_fn=sft.create_model_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_chunks=args.num_chunks,
        seed=args.seed,
        num_gpus=args.num_gpus,
    )
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    experiment.get_results().to_csv(paths["results"], index=False)
    experiment.get_runs_info().to_csv(paths["runs_info"], index=False)
    write_json(
        paths["run_config"],
        {
            "run_name": args.run_name,
            "experiment_name": args.experiment_name,
            "experiment_dir": str(paths["experiment_dir"]),
            "runs_info": str(paths["runs_info"]),
            "completion_only_loss": args.completion_only_loss,
            "pruning": {
                "top_tables": args.top_tables,
                "max_schema_columns": args.max_schema_columns,
                "max_columns_per_table": args.max_columns_per_table,
            },
        },
    )
    print(f"RapidFire experiment directory: {paths['experiment_dir']}")
    print(f"Run metadata directory: {paths['run_dir']}")


def load_adapter(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device, torch


def predict(args: argparse.Namespace) -> Path:
    model, tokenizer, device, torch = load_adapter(args)
    items = read_json(args.input)
    train_index = build_training_index(read_json(ROOT / args.train_json))
    schema_cache = {}
    predictions = []
    raw_rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(items, start=1):
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema_with_terms(ROOT / args.schemas_dir, db_id)
        full_schema = schema_cache[db_id]
        pruned = prune_schema(
            item.get("question", ""),
            db_id,
            full_schema,
            train_index,
            args.top_tables,
            args.max_schema_columns,
            args.max_columns_per_table,
            args.include_all_selected_table_columns,
        )
        schema_text = serialize_pruned_schema(pruned)
        system, user = sft.build_prompt(item.get("question", ""), schema_text)
        prompt = baseline.apply_chat_template(tokenizer, system, user)
        raw_text = baseline.generate_one(model, tokenizer, device, torch, prompt, args)
        raw_links = baseline.try_parse_mapping(raw_text)
        links = baseline.filter_schema_links(raw_links, full_schema)
        predictions.append({"question_id": item["question_id"], "schema_links": links})
        raw_rows.append(
            {
                "question_id": item["question_id"],
                "db_id": db_id,
                "raw_generation": raw_text,
                "parsed_schema_links": raw_links,
                "filtered_schema_links": links,
                "pruned_tables": pruned["tables"],
                "pruned_column_count": sum(len(cols) for cols in pruned["columns_by_table"].values()),
            }
        )
        print(f"[{idx}/{len(items)}] q{item['question_id']} {db_id}", file=sys.stderr)

    write_json(output, predictions)
    write_json(output.with_name("raw_generations.json"), raw_rows)
    if args.gold:
        per_question = output.with_name("per_question.csv")
        cmd = [
            sys.executable,
            str(ROOT / "eval.py"),
            "--predictions",
            str(output),
            "--gold",
            args.gold,
            "--schemas_dir",
            args.schemas_dir,
            "--questions_input",
            args.input,
            "--per_question_out",
            str(per_question),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
        print(proc.stdout)
        score = {"eval_stdout": proc.stdout, "per_question": str(per_question)}
        for line in proc.stdout.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            numeric = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip())
            if numeric:
                score[key] = float(numeric.group(0))
        write_json(output.with_name("score.json"), score)
    return output


def parse_csv_numbers(value: str, cast: Any) -> list[Any]:
    return [cast(piece.strip()) for piece in value.split(",") if piece.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "predict"):
        sp = sub.add_parser(name)
        sp.add_argument("--base_model", default=DEFAULT_MODEL)
        sp.add_argument("--train_json", default="train.json")
        sp.add_argument("--validation_json", default="validation.json")
        sp.add_argument("--schemas_dir", default="schemas")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--top_tables", type=int, default=14)
        sp.add_argument("--max_schema_columns", type=int, default=320)
        sp.add_argument("--max_columns_per_table", type=int, default=48)
        sp.add_argument("--include_all_selected_table_columns", action="store_true")
        sp.add_argument("--max_length", type=int, default=6144)
        sp.add_argument("--prompt_keep_head_tokens", type=int, default=512)
        sp.add_argument("--local_files_only", action="store_true")
        sp.add_argument("--trust_remote_code", action="store_true")

    for name in ("prepare", "train"):
        train = sub.choices[name]
        train.add_argument("--data_dir", default="data/sft_pruned")
        train.add_argument("--output_root", default="output")
        train.add_argument("--run_name", default="sft_pruned_lr2e-4_r16_a16_ep2_len6144")
        train.add_argument("--experiment_name", default="project2_qwen25_15b_sft_pruned_r16a16_len6144")
        train.add_argument("--experiment_path", default="rapidfire_experiments")
        train.add_argument("--num_chunks", type=int, default=1)
        train.add_argument("--num_gpus", type=int, default=1)
        train.add_argument("--epochs", type=float, default=2.0)
        train.add_argument("--per_device_train_batch_size", type=int, default=1)
        train.add_argument("--per_device_eval_batch_size", type=int, default=1)
        train.add_argument("--gradient_accumulation_steps", type=int, default=8)
        train.add_argument("--learning_rates", default="0.0002")
        train.add_argument("--lora_ranks", default="16")
        train.add_argument("--lora_alphas", default="16")
        train.add_argument("--lora_dropout", type=float, default=0.05)
        train.add_argument(
            "--target_modules",
            nargs="+",
            default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        train.add_argument("--warmup_ratio", type=float, default=0.03)
        train.add_argument("--lr_scheduler_type", default="cosine")
        train.add_argument("--logging_steps", type=int, default=5)
        train.add_argument("--eval_steps", type=int, default=10)
        train.add_argument("--model_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
        train.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
        train.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
        train.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
        train.add_argument("--completion_only_loss", action=argparse.BooleanOptionalAction, default=True)

    pred = sub.choices["predict"]
    pred.add_argument("--input", default="validation_input.json")
    pred.add_argument("--output", required=True)
    pred.add_argument("--adapter_path", required=True)
    pred.add_argument("--gold", default="validation_gold_schema_links.json")
    pred.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    pred.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    pred.add_argument("--max_input_tokens", type=int, default=6144)
    pred.add_argument("--max_new_tokens", type=int, default=384)
    pred.add_argument("--include_types", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command in {"prepare", "train"}:
        args.learning_rates = parse_csv_numbers(args.learning_rates, float)
        args.lora_ranks = parse_csv_numbers(args.lora_ranks, int)
        args.lora_alphas = parse_csv_numbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = prepare_data(args)
            print(f"Wrote pruned train data to {paths['train_jsonl']}")
            print(f"Wrote pruned eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            run_train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
