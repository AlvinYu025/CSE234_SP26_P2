"""
RapidFire SFT scaffold for Project 2 schema linking.

This file trains a <=2B local model with supervised fine-tuning using
RapidFire's fit framework and TRL SFTTrainer.  It is intentionally split into
small modes so the data pipeline can be validated before spending GPU time.

Typical usage:

    conda activate rapidfire

    # Build SFT JSONL files and a manifest.
    python sft.py --mode prepare

    # Inspect the 8 default RapidFire configs without training.
    python sft.py --mode list-configs

    # Launch RapidFire fit on the 1.5B base model.
    python sft.py --mode run

Outputs:
    data/sft/<model_slug>/...
    output/<model_slug>/<run_name>/...
    rapidfire_experiments/<experiment_name>/...
    logs/...

The final adapter chosen for submission should later be copied to ./adapter/
and loaded by main.py with PEFT, as required by the project statement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_EXPERIMENT = "project2_qwen25_15b_sft_grid8"

# Keep RapidFire artifacts inside this project instead of ~/rapidfireai.
# These must be set before importing rapidfireai.
os.environ.setdefault("RF_HOME", str(ROOT / "rapidfireai_sft"))
os.environ.setdefault("RF_LOG_PATH", str(ROOT / "logs"))
os.environ.setdefault("RF_DB_PATH", str(ROOT / "rapidfireai_sft" / "db"))
os.environ.setdefault("RF_EXPERIMENT_PATH", str(ROOT / "rapidfire_experiments"))
os.environ.setdefault(
    "RF_TENSORBOARD_LOG_DIR",
    str(ROOT / "rapidfire_experiments" / "tensorboard_logs"),
)
os.environ.setdefault("RF_TRAINER_OUTPUT", str(ROOT / "rapidfire_trainer_output"))
os.environ.setdefault("RF_TENSORBOARD_ENABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def readJson(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def writeJson(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def writeJsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "run"


def schemaFilename(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def loadSchema(schemas_dir: Path, db_id: str) -> dict[str, Any]:
    data = readJson(schemas_dir / schemaFilename(db_id))
    tables = list(data["table_names_original"])
    columns_by_table = {table: [] for table in tables}
    types_by_pair: dict[tuple[str, str], str] = {}
    column_types = data.get(
        "column_types",
        ["TEXT"] * len(data["column_names_original"]),
    )

    for (table_idx, column_name), column_type in zip(
        data["column_names_original"],
        column_types,
    ):
        if table_idx == -1:
            continue
        table = tables[table_idx]
        columns_by_table[table].append(column_name)
        types_by_pair[(table, column_name)] = column_type or "TEXT"

    return {
        "db_id": data.get("db_id", db_id),
        "tables": tables,
        "columns_by_table": columns_by_table,
        "types_by_pair": types_by_pair,
    }


def serializeSchema(schema: dict[str, Any], include_types: bool = False) -> str:
    lines = [f"Database: {schema['db_id']}", "Schema:"]
    for table in schema["tables"]:
        if table.lower() == "sysdiagrams":
            continue

        columns = []
        for column in schema["columns_by_table"][table]:
            if include_types:
                column_type = schema["types_by_pair"].get((table, column), "TEXT")
                columns.append(f"{column}:{column_type}")
            else:
                columns.append(column)
        lines.append(f"- {table}({', '.join(columns)})")
    return "\n".join(lines)


def buildPrompt(question: str, schema_text: str) -> tuple[str, str]:
    system = (
        "You are a schema-linking model. Given a natural-language question and "
        "a database schema, identify exactly which schema tables and columns "
        "the SQL query would reference."
    )
    user = f"""Return only a valid JSON object mapping schema table names to lists of schema column names.

Output shape:
{{"<exact table name>": ["<exact column name>", "..."], "<exact table name>": []}}

General rules:
- Choose the tables and columns that would be referenced by the SQL query answering the question.
- Use only identifiers that appear in the provided schema.
- Copy table and column names exactly as written in the schema, including casing.
- Include every referenced table, even when no specific column is referenced for that table.
- Use an empty list for a referenced table when the SQL would use only table-level access such as COUNT(*) or SELECT *.
- Do not treat * as meaning all columns.
- Return {{}} only if no table in the schema is relevant.
- Do not include SQL, explanations, markdown, comments, db_id, question_id, or any wrapper key.

{schema_text}

Question:
{question}
"""
    return system, user


def canonicalSchemaLinks(
    links: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, list[str]]:
    """Sort and filter gold links to exactly match the schema vocabulary."""
    if not isinstance(links, dict):
        return {}

    table_lookup = {table.lower(): table for table in schema["tables"]}
    column_lookup = {
        table: {column.lower(): column for column in schema["columns_by_table"][table]}
        for table in schema["tables"]
    }
    table_order = {table: idx for idx, table in enumerate(schema["tables"])}
    column_order = {
        (table, column): idx
        for table in schema["tables"]
        for idx, column in enumerate(schema["columns_by_table"][table])
    }

    selected: dict[str, list[str]] = {}
    for raw_table, raw_columns in links.items():
        table = table_lookup.get(str(raw_table).lower())
        if table is None or table.lower() == "sysdiagrams":
            continue

        columns = []
        seen = set()
        if isinstance(raw_columns, list):
            iterable = raw_columns
        elif isinstance(raw_columns, tuple):
            iterable = list(raw_columns)
        else:
            iterable = []

        for raw_column in iterable:
            column = column_lookup[table].get(str(raw_column).lower())
            if column is not None and column not in seen:
                seen.add(column)
                columns.append(column)
        selected[table] = sorted(
            columns,
            key=lambda column: column_order.get((table, column), 10**9),
        )

    return {
        table: selected[table]
        for table in sorted(selected, key=lambda table: table_order.get(table, 10**9))
    }


def applyChatTemplate(
    tokenizer: Any,
    system: str,
    user: str,
    assistant: str,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n{assistant}"


def applyChatPromptTemplate(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"


def buildPlainText(system: str, user: str, assistant: str) -> str:
    return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n{assistant}"


def buildPlainPrompt(system: str, user: str) -> str:
    return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"


def fitPromptAndAnswer(
    tokenizer: Any,
    prompt_text: str,
    answer_text: str,
    max_length: int,
    keep_head_tokens: int,
) -> tuple[str, str, int, bool]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length]
        return "", tokenizer.decode(answer_ids, skip_special_tokens=False), len(answer_ids), True

    prompt_budget = max_length - len(answer_ids)
    was_truncated = len(prompt_ids) > prompt_budget
    if was_truncated:
        head_count = min(keep_head_tokens, max(prompt_budget // 4, 0))
        tail_count = max(prompt_budget - head_count, 0)
        prompt_tail = prompt_ids[-tail_count:] if tail_count else []
        prompt_ids = prompt_ids[:head_count] + prompt_tail

    ids = prompt_ids + answer_ids
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    completion_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
    return prompt_text, completion_text, len(ids), was_truncated


def loadTokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def buildSftRows(
    examples: list[dict[str, Any]],
    schemas_dir: Path,
    tokenizer: Any | None,
    include_types: bool,
    template: str,
    max_length: int,
    keep_head_tokens: int,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    if max_examples is not None:
        examples = examples[:max_examples]

    schema_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for item in examples:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = loadSchema(schemas_dir, db_id)
        schema = schema_cache[db_id]
        schema_text = serializeSchema(schema, include_types=include_types)
        system, user = buildPrompt(item["question"], schema_text)
        target_links = canonicalSchemaLinks(item.get("schema_links", {}), schema)
        assistant = json.dumps(target_links, ensure_ascii=False, separators=(",", ":"))

        eos_token = getattr(tokenizer, "eos_token", None) if tokenizer else None
        answer_text = assistant + (eos_token or "")
        if template == "chat":
            prompt_text = applyChatPromptTemplate(tokenizer, system, user)
        else:
            prompt_text = buildPlainPrompt(system, user)

        prompt, completion, token_count, was_truncated = fitPromptAndAnswer(
            tokenizer,
            prompt_text,
            answer_text,
            max_length=max_length,
            keep_head_tokens=keep_head_tokens,
        )

        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion,
                "question_id": item["question_id"],
                "db_id": db_id,
                "target": target_links,
                "token_count": token_count,
                "truncated": was_truncated,
            }
        )
    return rows


def resolvePaths(args: argparse.Namespace) -> dict[str, Path]:
    model_slug = slugify(args.base_model)
    run_name = slugify(args.run_name)
    data_dir = ROOT / args.data_dir / model_slug
    run_dir = ROOT / args.output_root / model_slug / run_name
    experiment_path = ROOT / args.experiment_path
    return {
        "data_dir": data_dir,
        "run_dir": run_dir,
        "train_jsonl": data_dir / "train_sft.jsonl",
        "eval_jsonl": data_dir / "validation_sft.jsonl",
        "manifest": data_dir / "manifest.json",
        "run_config": run_dir / "run_config.json",
        "grid_config": run_dir / "rapidfire_grid.json",
        "results": run_dir / "rapidfire_results.csv",
        "runs_info": run_dir / "rapidfire_runs_info.csv",
        "experiment_path": experiment_path,
        "experiment_dir": experiment_path / args.experiment_name,
    }


def prepareData(args: argparse.Namespace) -> dict[str, Path]:
    paths = resolvePaths(args)
    tokenizer = loadTokenizer(args)

    train_examples = readJson(ROOT / args.train_json)
    eval_examples = readJson(ROOT / args.validation_json)
    random.Random(args.seed).shuffle(train_examples)

    train_rows = buildSftRows(
        train_examples,
        ROOT / args.schemas_dir,
        tokenizer,
        args.include_types,
        args.template,
        args.max_length,
        args.prompt_keep_head_tokens,
        args.max_train_examples,
    )
    eval_rows = buildSftRows(
        eval_examples,
        ROOT / args.schemas_dir,
        tokenizer,
        args.include_types,
        args.template,
        args.max_length,
        args.prompt_keep_head_tokens,
        args.max_eval_examples,
    )

    writeJsonl(paths["train_jsonl"], train_rows)
    writeJsonl(paths["eval_jsonl"], eval_rows)
    writeJson(
        paths["manifest"],
        {
            "base_model": args.base_model,
            "template": args.template,
            "include_types": args.include_types,
            "train_json": args.train_json,
            "validation_json": args.validation_json,
            "schemas_dir": args.schemas_dir,
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "max_length": args.max_length,
            "prompt_keep_head_tokens": args.prompt_keep_head_tokens,
            "train_truncated": sum(1 for row in train_rows if row["truncated"]),
            "eval_truncated": sum(1 for row in eval_rows if row["truncated"]),
            "train_max_tokens": max((row["token_count"] for row in train_rows), default=0),
            "eval_max_tokens": max((row["token_count"] for row in eval_rows), default=0),
            "train_jsonl": str(paths["train_jsonl"]),
            "eval_jsonl": str(paths["eval_jsonl"]),
            "seed": args.seed,
            "sample_target": train_rows[0]["target"] if train_rows else {},
        },
    )
    return paths


def requireSftDependencies() -> None:
    missing = []
    for module_name in ["accelerate", "bitsandbytes", "loguru", "peft", "trl"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing RapidFire SFT dependencies in this environment: "
            f"{', '.join(missing)}.\n"
            "Install them in the rapidfire env with:\n"
            f"  /home/fey006/.conda/envs/rapidfire/bin/python -m pip install {packages}"
        )


def torchDtypeFromString(dtype_name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def createModelFn(config_leaf: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config_leaf["model_name"]
    tokenizer_name = config_leaf.get("tokenizer") or model_name
    tokenizer_kwargs = dict(config_leaf.get("tokenizer_kwargs") or {})
    model_kwargs = dict(config_leaf.get("model_kwargs") or {})

    dtype = model_kwargs.get("torch_dtype")
    if isinstance(dtype, str) and dtype != "auto":
        model_kwargs["torch_dtype"] = torchDtypeFromString(dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    return model, tokenizer


def buildRapidfireConfig(args: argparse.Namespace) -> Any:
    requireSftDependencies()

    from rapidfireai.automl.datatypes import List
    from rapidfireai.automl.grid_search import RFGridSearch
    from rapidfireai.automl.model_config import RFModelConfig, RFLoraConfig, RFSFTConfig

    if RFLoraConfig is None or RFSFTConfig is None:
        raise SystemExit(
            "RapidFire imported without PEFT/TRL support. Restart the Python process "
            "after installing peft and trl."
        )

    training_args = RFSFTConfig(
        output_dir=str(ROOT / "rapidfire_trainer_output"),
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=args.completion_only_loss,
        assistant_only_loss=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=List(args.learning_rates),
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    peft_config = RFLoraConfig(
        r=List(args.lora_ranks),
        lora_alpha=List(args.lora_alphas),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    model_config = RFModelConfig(
        model_name=args.base_model,
        tokenizer=args.base_model,
        tokenizer_kwargs={
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "use_fast": True,
        },
        model_type="causal_lm",
        model_kwargs={
            "torch_dtype": args.model_dtype,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        },
        peft_config=peft_config,
        training_args=training_args,
        num_gpus=args.num_gpus,
    )
    return RFGridSearch(configs=[model_config], trainer_type="SFT")


def summarizeRuns(param_config: Any, seed: int) -> list[dict[str, Any]]:
    runs = param_config.get_runs(seed=seed)
    summary = []
    for idx, run in enumerate(runs, start=1):
        training = run.get("training_args", {})
        peft = run.get("peft_params", {})
        summary.append(
            {
                "index": idx,
                "model_name": run.get("model_name"),
                "learning_rate": training.get("learning_rate"),
                "epochs": training.get("num_train_epochs"),
                "max_length": training.get("max_length"),
                "batch_size": training.get("per_device_train_batch_size"),
                "grad_accum": training.get("gradient_accumulation_steps"),
                "completion_only_loss": training.get("completion_only_loss"),
                "lora_r": peft.get("r"),
                "lora_alpha": peft.get("lora_alpha"),
                "lora_dropout": peft.get("lora_dropout"),
                "target_modules": peft.get("target_modules"),
            }
        )
    return summary


def loadDatasetFromJsonl(path: Path) -> Any:
    from datasets import Dataset

    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def runRapidfire(args: argparse.Namespace) -> dict[str, Path]:
    paths = prepareData(args)
    param_config = buildRapidfireConfig(args)
    config_summary = summarizeRuns(param_config, args.seed)
    writeJson(paths["grid_config"], config_summary)

    train_dataset = loadDatasetFromJsonl(paths["train_jsonl"])
    eval_dataset = loadDatasetFromJsonl(paths["eval_jsonl"])

    from rapidfireai import Experiment

    experiment = Experiment(
        args.experiment_name,
        mode="fit",
        experiment_path=str(paths["experiment_path"]),
    )
    experiment.run_fit(
        param_config=param_config,
        create_model_fn=createModelFn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_chunks=args.num_chunks,
        seed=args.seed,
        num_gpus=args.num_gpus,
    )

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    try:
        results = experiment.get_results()
        results.to_csv(paths["results"], index=False)
    except Exception as exc:
        print(f"Warning: could not export RapidFire metric results: {exc}")

    try:
        runs_info = experiment.get_runs_info()
        runs_info.to_csv(paths["runs_info"], index=False)
    except Exception as exc:
        print(f"Warning: could not export RapidFire runs info: {exc}")

    writeJson(
        paths["run_config"],
        {
            "base_model": args.base_model,
            "run_name": args.run_name,
            "experiment_name": args.experiment_name,
            "experiment_path": str(paths["experiment_path"]),
            "experiment_dir": str(paths["experiment_dir"]),
            "train_jsonl": str(paths["train_jsonl"]),
            "eval_jsonl": str(paths["eval_jsonl"]),
            "rapidfire_grid": str(paths["grid_config"]),
            "results": str(paths["results"]),
            "runs_info": str(paths["runs_info"]),
            "num_configs": len(config_summary),
            "completion_only_loss": args.completion_only_loss,
            "note": (
                "RapidFire final adapters are under "
                f"{paths['experiment_dir']}/runs/<run_id>/checkpoints/final_checkpoint"
            ),
        },
    )
    return paths


def parseCsvNumbers(value: str, cast: Any) -> list[Any]:
    return [cast(piece.strip()) for piece in value.split(",") if piece.strip()]


def buildArgParser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="prepare",
        choices=["prepare", "list-configs", "run"],
        help="prepare data, list RapidFire grid configs, or run RapidFire SFT",
    )
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--validation_json", default="validation.json")
    ap.add_argument("--schemas_dir", default="schemas")
    ap.add_argument("--data_dir", default="data/sft")
    ap.add_argument("--output_root", default="output")
    ap.add_argument("--run_name", default="sft_grid8")
    ap.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--experiment_path", default="rapidfire_experiments")
    ap.add_argument("--template", default="chat", choices=["chat", "plain"])
    ap.add_argument("--include_types", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_eval_examples", type=int, default=None)
    ap.add_argument("--prompt_keep_head_tokens", type=int, default=512)
    ap.add_argument(
        "--completion_only_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute SFT loss only on the completion/answer tokens. Use --no-completion_only_loss for full LM loss.",
    )

    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rates", default="0.0001,0.0002")
    ap.add_argument("--lora_ranks", default="8,16")
    ap.add_argument("--lora_alphas", default="16,32")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=10)
    ap.add_argument(
        "--model_dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return ap


def main() -> None:
    args = buildArgParser().parse_args()
    args.learning_rates = parseCsvNumbers(args.learning_rates, float)
    args.lora_ranks = parseCsvNumbers(args.lora_ranks, int)
    args.lora_alphas = parseCsvNumbers(args.lora_alphas, int)

    if args.mode == "prepare":
        paths = prepareData(args)
        print(f"Wrote SFT train data to {paths['train_jsonl']}")
        print(f"Wrote SFT eval data to {paths['eval_jsonl']}")
        print(f"Wrote manifest to {paths['manifest']}")
        return

    if args.mode == "list-configs":
        param_config = buildRapidfireConfig(args)
        paths = prepareData(args)
        grid_summary = summarizeRuns(param_config, args.seed)
        writeJson(paths["grid_config"], grid_summary)
        print(json.dumps(grid_summary, indent=2))
        print(f"Wrote RapidFire grid summary to {paths['grid_config']}")
        return

    paths = runRapidfire(args)
    print(f"RapidFire experiment directory: {paths['experiment_dir']}")
    print(f"Run metadata directory: {paths['run_dir']}")


if __name__ == "__main__":
    main()
