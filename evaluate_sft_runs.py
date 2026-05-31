"""
Evaluate completed RapidFire SFT runs on the Project 2 validation split.

The script reads RapidFire's runs-info CSV, loads each completed LoRA adapter,
runs schema-link inference, and stores each result under an intuitive directory:

    output/<model_slug>/sft_lr<lr>_r<rank>_a<alpha>_ep<epochs>_len<maxlen>/

Usage after training:

    python evaluate_sft_runs.py \
        --runs_info output/qwen-qwen2.5-1.5b-instruct/sft_grid8_len6144/rapidfire_runs_info.csv \
        --experiment_dir rapidfire_experiments/project2_qwen25_15b_sft_grid8_len6144
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "run"


def compactFloat(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return slugify(value)
    if isinstance(value, float):
        text = f"{value:.0e}" if value < 0.01 else f"{value:g}"
        return text.replace("e-0", "e-").replace("e+0", "e")
    return str(value)


def readRunsInfo(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                config = ast.literal_eval(row.get("config", "{}"))
            except (SyntaxError, ValueError):
                config = {}
            row["config_obj"] = config
            rows.append(row)
    return rows


def isCompleted(row: dict[str, Any]) -> bool:
    return row.get("status") in {"RunStatus.COMPLETED", "COMPLETED", "completed"}


def jsonSafe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(jsonSafe(item) for item in value)
    if isinstance(value, tuple):
        return [jsonSafe(item) for item in value]
    if isinstance(value, list):
        return [jsonSafe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonSafe(item) for key, item in value.items()}
    return value


def configSlug(row: dict[str, Any]) -> str:
    config = row.get("config_obj", {})
    training = config.get("training_args", {})
    peft = config.get("peft_params", {})
    epochs = training.get("num_train_epochs") or row.get("num_epochs_completed")
    prefix = "sft_completion" if training.get("completion_only_loss") is True else "sft"
    return (
        f"{prefix}_lr{compactFloat(training.get('learning_rate'))}"
        f"_r{peft.get('r')}"
        f"_a{peft.get('lora_alpha')}"
        f"_ep{compactFloat(epochs)}"
        f"_len{training.get('max_length')}"
    )


def loadModelAndTokenizer(args: argparse.Namespace, adapter_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)
    model.eval()
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    return model, tokenizer, device, torch


def predictFile(args: argparse.Namespace, adapter_path: Path, output_dir: Path) -> Path:
    model, tokenizer, device, torch = loadModelAndTokenizer(args, adapter_path)
    items = baseline.readJson(args.input)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    raw_path = output_dir / "raw_generations.json"

    schema_cache = {}
    predictions = []
    raw_rows = []

    for idx, item in enumerate(items, start=1):
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = baseline.loadSchema(args.schemas_dir, db_id)
        schema = schema_cache[db_id]

        schema_text = baseline.serializeSchema(schema, include_types=args.include_types)
        system, user = baseline.buildPrompt(item.get("question", ""), schema_text)
        prompt = baseline.applyChatTemplate(tokenizer, system, user)
        raw_text = baseline.generateOne(model, tokenizer, device, torch, prompt, args)

        raw_links = baseline.tryParseMapping(raw_text)
        links = baseline.filterSchemaLinks(raw_links, schema)
        predictions.append(
            {
                "question_id": item["question_id"],
                "schema_links": links,
            }
        )
        raw_rows.append(
            {
                "question_id": item["question_id"],
                "db_id": db_id,
                "raw_generation": raw_text,
                "parsed_schema_links": raw_links,
                "filtered_schema_links": links,
            }
        )
        print(f"[{idx}/{len(items)}] {output_dir.name} q{item['question_id']} {db_id}", file=sys.stderr)

    with predictions_path.open("w") as f:
        json.dump(jsonSafe(predictions), f, indent=2)
    with raw_path.open("w") as f:
        json.dump(jsonSafe(raw_rows), f, indent=2)
    return predictions_path


def runEval(args: argparse.Namespace, predictions_path: Path, output_dir: Path) -> dict[str, Any]:
    per_question = output_dir / "per_question.csv"
    cmd = [
        sys.executable,
        str(ROOT / "eval.py"),
        "--predictions",
        str(predictions_path),
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
    eval_stdout = proc.stdout

    score = {
        "eval_stdout": eval_stdout,
        "per_question": str(per_question),
    }
    for line in eval_stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        value = value.strip()
        numeric = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if numeric:
            score[key] = float(numeric.group(0))
        else:
            score[key] = value

    with (output_dir / "score.json").open("w") as f:
        json.dump(score, f, indent=2)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_info", required=True)
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--base_model", default=DEFAULT_MODEL)
    ap.add_argument("--input", default="validation_input.json")
    ap.add_argument("--gold", default="validation_gold_schema_links.json")
    ap.add_argument("--schemas_dir", default="schemas")
    ap.add_argument("--output_root", default="output")
    ap.add_argument("--aggregate_name", default=None)
    ap.add_argument("--max_input_tokens", type=int, default=16000)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--include_types", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--only_run_id", default=None)
    args = ap.parse_args()

    model_slug = slugify(args.base_model)
    rows = readRunsInfo(Path(args.runs_info))
    completed = [row for row in rows if isCompleted(row)]
    if args.only_run_id is not None:
        completed = [row for row in completed if row.get("run_id") == str(args.only_run_id)]

    if not completed:
        raise SystemExit("No completed RapidFire runs found to evaluate.")

    aggregate = []
    for row in completed:
        run_id = row["run_id"]
        adapter_path = Path(args.experiment_dir) / "runs" / str(run_id) / "checkpoints" / "final_checkpoint"
        if not (adapter_path / "adapter_model.safetensors").exists():
            print(f"Skipping run {run_id}: adapter not found at {adapter_path}", file=sys.stderr)
            continue

        name = configSlug(row)
        output_dir = ROOT / args.output_root / model_slug / name
        predictions_path = predictFile(args, adapter_path, output_dir)
        score = runEval(args, predictions_path, output_dir)
        config = row.get("config_obj", {})
        training = config.get("training_args", {})
        peft = config.get("peft_params", {})
        summary = {
            "run_id": run_id,
            "name": name,
            "output_dir": str(output_dir),
            "adapter_path": str(adapter_path),
            "learning_rate": training.get("learning_rate"),
            "lora_r": peft.get("r"),
            "lora_alpha": peft.get("lora_alpha"),
            "epochs": training.get("num_train_epochs") or row.get("num_epochs_completed"),
            "max_length": training.get("max_length"),
            "completion_only_loss": training.get("completion_only_loss"),
            "leaderboard_score": score.get("leaderboard_score"),
            "table_score": score.get("table_score"),
            "column_score": score.get("column_score"),
        }
        aggregate.append(summary)
        with (output_dir / "run_config.json").open("w") as f:
            json.dump({**summary, "rapidfire_config": config}, f, indent=2)
        print(json.dumps(summary, indent=2))

    aggregate_name = args.aggregate_name or Path(args.runs_info).parent.name or "sft_runs"
    aggregate_dir = ROOT / args.output_root / model_slug / aggregate_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    with (aggregate_dir / "validation_scores.json").open("w") as f:
        json.dump(aggregate, f, indent=2)

    with (aggregate_dir / "validation_scores.csv").open("w", newline="") as f:
        fieldnames = [
            "run_id",
            "name",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "epochs",
            "max_length",
            "completion_only_loss",
            "leaderboard_score",
            "table_score",
            "column_score",
            "output_dir",
            "adapter_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote aggregate scores to {aggregate_dir / 'validation_scores.csv'}")


if __name__ == "__main__":
    main()
