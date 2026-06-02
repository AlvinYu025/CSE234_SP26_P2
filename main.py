"""Inference entry point for Project 2."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import baseline
import pruned_sft
import sft


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LEGACY_ADAPTER_PATH = (
    ROOT
    / "rapidfire_experiments"
    / "project2_qwen25_coder15b_sft_pruned_dbbal12_lr8p5e5_r16a32_ep7_retry_detached_len8192"
    / "runs"
    / "1"
    / "checkpoints"
    / "final_checkpoint"
)
DEFAULT_ADAPTER_PATH = ROOT / "adapter"
TABLE_VOTERS = ("best", "wide", "allcols")
COLUMN_VOTERS = ("best", "wide", "allcols", "recall")
TABLE_THRESHOLD = 1
COLUMN_THRESHOLD = 2
SCHEMA_VIEWS = {
    "wide": {
        "top_tables": 20,
        "max_schema_columns": 520,
        "max_columns_per_table": 80,
        "include_all_selected_table_columns": False,
    },
    "allcols": {
        "top_tables": 14,
        "max_schema_columns": 800,
        "max_columns_per_table": 160,
        "include_all_selected_table_columns": True,
    },
    "recall": {
        "top_tables": 14,
        "max_schema_columns": 800,
        "max_columns_per_table": 160,
        "include_all_selected_table_columns": True,
    },
}


def buildParser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")

    ap.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER_PATH))
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--view_vote", action=argparse.BooleanOptionalAction, default=True)

    # Best validation pruning setup.
    ap.add_argument("--top_tables", type=int, default=14)
    ap.add_argument("--max_schema_columns", type=int, default=320)
    ap.add_argument("--max_columns_per_table", type=int, default=48)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=384)

    # Local validation only; the hidden test runner will not pass this.
    ap.add_argument("--gold", default=None)
    return ap


def normalizeLinks(links: Any) -> dict[str, list[str]]:
    if not isinstance(links, dict):
        return {}
    clean = {}
    for table, cols in links.items():
        if not isinstance(cols, list):
            cols = []
        clean[str(table)] = [str(col) for col in cols]
    return clean


def votedTables(allLinks: dict[str, dict[str, list[str]]]) -> list[str]:
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    order = []
    for adapter in TABLE_VOTERS:
        for table in normalizeLinks(allLinks.get(adapter, {})):
            key = table.lower()
            counts[key] = counts.get(key, 0) + 1
            if key not in names:
                names[key] = table
                order.append(key)
    return [names[key] for key in order if counts[key] >= TABLE_THRESHOLD]


def votedColumns(allLinks: dict[str, dict[str, list[str]]], selectedTables: list[str]) -> dict[str, list[str]]:
    allowed = {table.lower(): table for table in selectedTables}
    counts: dict[tuple[str, str], int] = {}
    names: dict[tuple[str, str], tuple[str, str]] = {}
    for adapter in COLUMN_VOTERS:
        for table, cols in normalizeLinks(allLinks.get(adapter, {})).items():
            tableKey = table.lower()
            for col in cols:
                key = (tableKey, col.lower())
                counts[key] = counts.get(key, 0) + 1
                names[key] = (table, col)

    out = {table: [] for table in selectedTables}
    for (tableKey, colKey), count in counts.items():
        if count < COLUMN_THRESHOLD or tableKey not in allowed:
            continue
        _table, col = names[(tableKey, colKey)]
        out[allowed[tableKey]].append(col)
    return {table: sorted(set(cols)) for table, cols in out.items()}


def loadAdapterModel(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    baseModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )

    model = PeftModel.from_pretrained(baseModel, args.adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device, torch


def buildViewPrompt(tokenizer: Any, question: str, schemaText: str, viewName: str) -> str:
    if viewName != "recall":
        system, user = sft.buildPrompt(question, schemaText)
        return baseline.applyChatTemplate(tokenizer, system, user)

    system = (
        "You are a schema-linking model. Given a question and schema, "
        "identify all schema tables and columns that the SQL query would reference."
    )
    user = f"""Return only a valid JSON object mapping exact schema table names to lists of exact schema column names.

Prefer complete column recall: include columns used for SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, comparisons, aggregation arguments, and date or category filters. Still use only identifiers present in the schema. Do not add explanations or SQL.

Output shape:
{{"<exact table name>": ["<exact column name>", "..."], "<exact table name>": []}}

{schemaText}

Question:
{question}
"""
    return baseline.applyChatTemplate(tokenizer, system, user)


def runEval(args: argparse.Namespace, output: Path) -> None:
    if not args.gold:
        return
    perQuestion = output.with_name("per_question.csv")
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
        str(perQuestion),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    print(proc.stdout)
    score = {"eval_stdout": proc.stdout, "per_question": str(perQuestion)}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        numeric = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip())
        if numeric:
            score[key] = float(numeric.group(0))
    pruned_sft.writeJson(output.with_name("score.json"), score)


def runViewVote(args: argparse.Namespace) -> Path:
    model, tokenizer, device, torch = loadAdapterModel(args)
    items = pruned_sft.readJson(args.input)
    trainIndex = pruned_sft.buildTrainingIndex(pruned_sft.readJson(ROOT / args.train_json))
    schemaCache = {}
    predictions = []
    rawRows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items, start=1):
        dbId = item["db_id"]
        if dbId not in schemaCache:
            schemaCache[dbId] = pruned_sft.loadSchemaWithTerms(ROOT / args.schemas_dir, dbId)
        fullSchema = schemaCache[dbId]
        pruned = pruned_sft.pruneSchema(
            item.get("question", ""),
            dbId,
            fullSchema,
            trainIndex,
            args.top_tables,
            args.max_schema_columns,
            args.max_columns_per_table,
            args.include_all_selected_table_columns,
        )
        schemaText = pruned_sft.serializePrunedSchema(pruned)
        system, user = sft.buildPrompt(item.get("question", ""), schemaText)
        prompt = baseline.applyChatTemplate(tokenizer, system, user)

        adapterLinks = {}
        rawTextByAdapter = {}

        rawText = baseline.generateOne(model, tokenizer, device, torch, prompt, args)
        rawLinks = baseline.tryParseMapping(rawText)
        adapterLinks["best"] = baseline.filterSchemaLinks(rawLinks, fullSchema)
        rawTextByAdapter["best"] = rawText

        for viewName, view in SCHEMA_VIEWS.items():
            viewPruned = pruned_sft.pruneSchema(
                item.get("question", ""),
                dbId,
                fullSchema,
                trainIndex,
                view["top_tables"],
                view["max_schema_columns"],
                view["max_columns_per_table"],
                view["include_all_selected_table_columns"],
            )
            viewSchemaText = pruned_sft.serializePrunedSchema(viewPruned)
            viewPrompt = buildViewPrompt(tokenizer, item.get("question", ""), viewSchemaText, viewName)
            rawText = baseline.generateOne(model, tokenizer, device, torch, viewPrompt, args)
            rawLinks = baseline.tryParseMapping(rawText)
            adapterLinks[viewName] = baseline.filterSchemaLinks(rawLinks, fullSchema)
            rawTextByAdapter[viewName] = rawText

        selectedTables = votedTables(adapterLinks)
        links = votedColumns(adapterLinks, selectedTables)
        predictions.append({"question_id": item["question_id"], "schema_links": links})
        rawRows.append(
            {
                "question_id": item["question_id"],
                "db_id": dbId,
                "adapter_schema_links": adapterLinks,
                "raw_generation": rawTextByAdapter,
                "filtered_schema_links": links,
                "pruned_tables": pruned["tables"],
                "pruned_column_count": sum(len(cols) for cols in pruned["columns_by_table"].values()),
            }
        )
        print(f"[{idx}/{len(items)}] q{item['question_id']} {dbId}", file=sys.stderr)

    pruned_sft.writeJson(output, predictions)
    pruned_sft.writeJson(output.with_name("raw_generations.json"), rawRows)
    runEval(args, output)
    return output


def main() -> None:
    args = buildParser().parse_args()
    adapter_path = Path(args.adapter_path)
    using_default_adapter = str(adapter_path) == str(DEFAULT_ADAPTER_PATH)
    if using_default_adapter and not adapter_path.exists() and LEGACY_ADAPTER_PATH.exists():
        adapter_path = LEGACY_ADAPTER_PATH
        args.adapter_path = str(adapter_path)
    if not adapter_path.exists():
        raise SystemExit(
            "Adapter checkpoint not found at "
            f"{adapter_path}. Put the PEFT adapter in ./adapter or use --adapter_path."
        )

    # Keep inference on the pruned-schema path used by the best run.
    args.include_all_selected_table_columns = False
    if args.view_vote:
        runViewVote(args)
        return
    pruned_sft.predict(args)


if __name__ == "__main__":
    main()
