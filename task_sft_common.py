from __future__ import annotations

import argparse
import copy
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import balanced_pruned_sft
import baseline
import coder_balanced_pruned_sft
import pruned_sft
import sft


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def schemaForTables(schema: dict[str, Any], tables: list[str], columnCap: int = 0) -> dict[str, Any]:
    keep = []
    seen = set()
    tableLookup = {table.lower(): table for table in schema["tables"]}
    for table in tables:
        realTable = tableLookup.get(str(table).lower())
        if realTable and realTable.lower() not in seen and realTable.lower() != "sysdiagrams":
            seen.add(realTable.lower())
            keep.append(realTable)

    columnsByTable = {}
    for table in keep:
        cols = list(schema["columns_by_table"].get(table, []))
        if columnCap > 0:
            cols = cols[:columnCap]
        columnsByTable[table] = cols

    return {
        "db_id": schema["db_id"],
        "tables": keep,
        "columns_by_table": columnsByTable,
        "types_by_pair": schema["types_by_pair"],
    }


def buildTaskPrompt(
    task: str,
    question: str,
    schemaText: str,
    selectedTables: list[str] | None = None,
    clauseName: str | None = None,
    candidate: tuple[str, str] | None = None,
) -> tuple[str, str]:
    system = (
        "You are a schema-linking model. Given a natural-language question and "
        "a database schema, identify the exact schema identifiers needed for SQL."
    )
    if task == "tables":
        user = f"""Return only a valid JSON object whose keys are the relevant table names and whose values are empty lists.

Output shape:
{{"<exact table name>": [], "<exact table name>": []}}

Rules:
- Choose every table that the SQL query would reference.
- Prefer recall when a table may be needed for a join or filter.
- Do not include columns.
- Use only exact table names from the schema.
- Return {{}} if no table is relevant.
- Do not include SQL, explanations, or wrapper keys.

{schemaText}

Question:
{question}
"""
        return system, user

    if task == "columns":
        tableLines = "\n".join(f"- {table}" for table in selectedTables or [])
        user = f"""Return only a valid JSON object mapping the relevant candidate tables to the columns needed by the SQL query.

Candidate tables:
{tableLines}

Output shape:
{{"<exact table name>": ["<exact column name>", "..."], "<exact table name>": []}}

Rules:
- Use only the candidate tables.
- Omit candidate tables that the SQL query would not reference.
- Include columns used by SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, aggregation arguments, comparisons, and filters.
- Use an empty list when a relevant table is referenced but no specific column is needed.
- Copy table and column names exactly as written in the schema.
- Do not include SQL, explanations, or wrapper keys.

{schemaText}

Question:
{question}
"""
        return system, user

    if task == "clause":
        tableLines = "\n".join(f"- {table}" for table in selectedTables or [])
        user = f"""Return only a valid JSON object for columns used by the {clauseName} part of the SQL query.

Selected tables:
{tableLines}

Output shape:
{{"<exact table name>": ["<exact column name>", "..."]}}

Rules:
- Use only the selected tables.
- Include only columns needed for the {clauseName} part.
- Copy identifiers exactly as written in the schema.
- Return {{}} if this part does not need any column.
- Do not include SQL, explanations, or wrapper keys.

{schemaText}

Question:
{question}
"""
        return system, user

    if task == "verify":
        table, column = candidate or ("", "")
        user = f"""Answer only yes or no.

Candidate:
{table}.{column}

Question:
{question}

Does the SQL query need this exact table.column identifier?

{schemaText}
"""
        return system, user

    system, user = sft.buildPrompt(question, schemaText)
    return system, user


def makeRow(
    tokenizer: Any,
    task: str,
    item: dict[str, Any],
    schemaText: str,
    target: dict[str, list[str]] | str,
    args: argparse.Namespace,
    selectedTables: list[str] | None = None,
    clauseName: str | None = None,
    candidate: tuple[str, str] | None = None,
    prunedTables: list[str] | None = None,
    prunedColumnCount: int = 0,
) -> dict[str, Any]:
    system, user = buildTaskPrompt(
        task,
        item.get("question", ""),
        schemaText,
        selectedTables=selectedTables,
        clauseName=clauseName,
        candidate=candidate,
    )
    if isinstance(target, str):
        assistant = target
    else:
        assistant = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    answerText = assistant + (getattr(tokenizer, "eos_token", None) or "")
    promptText = sft.applyChatPromptTemplate(tokenizer, system, user)
    prompt, completion, tokenCount, wasTruncated = sft.fitPromptAndAnswer(
        tokenizer,
        promptText,
        answerText,
        max_length=args.max_length,
        keep_head_tokens=args.prompt_keep_head_tokens,
    )
    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "task": task,
        "question_id": item["question_id"],
        "db_id": item["db_id"],
        "target": target,
        "selected_tables": selectedTables or [],
        "clause": clauseName,
        "candidate": list(candidate) if candidate else None,
        "pruned_tables": prunedTables or [],
        "pruned_column_count": prunedColumnCount,
        "token_count": tokenCount,
        "truncated": wasTruncated,
    }


def sqlSlice(sql: str, startWord: str, endWords: tuple[str, ...]) -> str:
    upper = sql.upper()
    start = upper.find(startWord)
    if start < 0:
        return ""
    end = len(sql)
    for word in endWords:
        pos = upper.find(word, start + len(startWord))
        if pos >= 0:
            end = min(end, pos)
    return sql[start:end]


def clauseText(sql: str, clauseName: str) -> str:
    if clauseName == "select":
        return sqlSlice(sql, "SELECT", (" FROM ",))
    if clauseName == "join":
        fromPart = sqlSlice(sql, " FROM ", (" WHERE ", " GROUP BY ", " HAVING ", " ORDER BY ", " LIMIT "))
        return fromPart
    if clauseName == "filter":
        wherePart = sqlSlice(sql, " WHERE ", (" GROUP BY ", " HAVING ", " ORDER BY ", " LIMIT "))
        havingPart = sqlSlice(sql, " HAVING ", (" ORDER BY ", " LIMIT "))
        return f"{wherePart} {havingPart}".strip()
    if clauseName == "group_order":
        groupPart = sqlSlice(sql, " GROUP BY ", (" HAVING ", " ORDER BY ", " LIMIT "))
        orderPart = sqlSlice(sql, " ORDER BY ", (" LIMIT ",))
        return f"{groupPart} {orderPart}".strip()
    return ""


def sqlAliases(sql: str, targetLinks: dict[str, list[str]]) -> dict[str, str]:
    tableLookup = {table.lower(): table for table in targetLinks}
    aliases = {table.lower(): table for table in targetLinks}
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_$#]*)"
        r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_$#]*))?",
        flags=re.IGNORECASE,
    )
    stop = {"on", "where", "join", "left", "right", "inner", "outer", "full", "group", "order", "having", "limit"}
    for match in pattern.finditer(sql):
        tableName = match.group(1)
        realTable = tableLookup.get(tableName.lower())
        if not realTable:
            continue
        alias = match.group(2)
        if alias and alias.lower() not in stop:
            aliases[alias.lower()] = realTable
    return aliases


def normalizedSql(text: str) -> str:
    text = re.sub(r"[\[\]`\"]", "", text)
    return re.sub(r"[^A-Za-z0-9_.$#]+", " ", text).lower()


def targetForClause(targetLinks: dict[str, list[str]], sql: str, clauseName: str) -> dict[str, list[str]]:
    text = clauseText(sql, clauseName)
    if not text:
        return {}

    compact = normalizedSql(text)
    aliases = sqlAliases(sql, targetLinks)
    qualified: set[tuple[str, str]] = set()
    for owner, column in re.findall(r"\b([A-Za-z_][A-Za-z0-9_$#]*)\.([A-Za-z_][A-Za-z0-9_$#]*)\b", compact):
        table = aliases.get(owner.lower())
        if table:
            qualified.add((table, column.lower()))

    columnOwners: dict[str, list[str]] = defaultdict(list)
    for table, columns in targetLinks.items():
        for column in columns:
            columnOwners[str(column).lower()].append(table)

    out: dict[str, list[str]] = {}
    for table, columns in targetLinks.items():
        selected = []
        for column in columns:
            columnKey = str(column).lower()
            if (table, columnKey) in qualified:
                selected.append(column)
                continue
            if len(columnOwners[columnKey]) != 1:
                continue
            pattern = re.sub(r"[^A-Za-z0-9_]+", " ", str(column)).lower()
            if re.search(rf"(^|\s){re.escape(pattern)}($|\s)", compact):
                selected.append(column)
        if selected:
            out[table] = selected
    return out


def hardNegatives(
    schema: dict[str, Any],
    targetLinks: dict[str, list[str]],
    pruned: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    targetPairs = {(table, column) for table, cols in targetLinks.items() for column in cols}
    candidates = []
    for table in pruned["tables"]:
        for column in pruned["columns_by_table"].get(table, []):
            if (table, column) not in targetPairs:
                candidates.append((table, column))
    if len(candidates) <= limit:
        return candidates
    return rng.sample(candidates, limit)


def positivePairs(targetLinks: dict[str, list[str]], limit: int, rng: random.Random) -> list[tuple[str, str]]:
    pairs = [(table, column) for table, cols in targetLinks.items() for column in cols]
    if len(pairs) <= limit:
        return pairs
    return rng.sample(pairs, limit)


def candidateTables(targetTables: list[str], pruned: dict[str, Any], negativeCount: int, rng: random.Random) -> list[str]:
    selected = list(targetTables)
    targetKeys = {table.lower() for table in targetTables}
    negatives = [table for table in pruned["tables"] if table.lower() not in targetKeys]
    if negativeCount > 0 and len(negatives) > negativeCount:
        negatives = rng.sample(negatives, negativeCount)
    selected.extend(negatives)
    return selected


def balanceExamples(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    target = max(0, int(args.db_balance_target))
    cap = max(0, int(args.db_balance_cap))
    if target <= 0 and cap <= 0:
        return examples

    rng = random.Random(args.seed)
    byDb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        byDb[item["db_id"]].append(item)

    balanced = []
    for dbId in sorted(byDb):
        group = list(byDb[dbId])
        if cap > 0 and len(group) > cap:
            group = rng.sample(group, cap)
        if target > 0 and len(group) < target:
            group.extend(rng.choice(group) for _ in range(target - len(group)))
        balanced.extend(group)

    if args.db_balance_shuffle:
        rng.shuffle(balanced)
    return balanced


def buildRows(args: argparse.Namespace, examples: list[dict[str, Any]], tokenizer: Any, strategy: str) -> list[dict[str, Any]]:
    schemaCache = {}
    trainIndex = pruned_sft.buildTrainingIndex(pruned_sft.readJson(ROOT / args.train_json))
    rows = []
    rng = random.Random(args.seed + (0 if strategy == "two_stage" else 1000))

    for item in examples:
        dbId = item["db_id"]
        if dbId not in schemaCache:
            schemaCache[dbId] = pruned_sft.loadSchemaWithTerms(ROOT / args.schemas_dir, dbId)
        fullSchema = schemaCache[dbId]
        targetLinks = sft.canonicalSchemaLinks(item.get("schema_links", {}), fullSchema)
        targetTables = list(targetLinks)
        tableTarget = {table: [] for table in targetTables}

        pruned = pruned_sft.pruneSchema(
            item.get("question", ""),
            dbId,
            fullSchema,
            trainIndex,
            args.top_tables,
            args.max_schema_columns,
            args.max_columns_per_table,
            args.include_all_selected_table_columns,
            force_links=targetLinks,
        )
        prunedText = pruned_sft.serializePrunedSchema(pruned)
        prunedColumnCount = sum(len(cols) for cols in pruned["columns_by_table"].values())

        if strategy == "mixed":
            for _ in range(args.full_task_weight):
                rows.append(makeRow(tokenizer, "full", item, prunedText, targetLinks, args, prunedTables=pruned["tables"], prunedColumnCount=prunedColumnCount))

        for _ in range(args.table_task_weight):
            rows.append(makeRow(tokenizer, "tables", item, prunedText, tableTarget, args, prunedTables=pruned["tables"], prunedColumnCount=prunedColumnCount))

        selectedTables = candidateTables(targetTables, pruned, args.candidate_table_negatives, rng)
        selectedSchema = schemaForTables(fullSchema, selectedTables, args.stage_columns_cap)
        selectedText = pruned_sft.serializePrunedSchema(selectedSchema)
        selectedColumnCount = sum(len(cols) for cols in selectedSchema["columns_by_table"].values())
        for _ in range(args.column_task_weight):
            rows.append(
                makeRow(
                    tokenizer,
                    "columns",
                    item,
                    selectedText,
                    targetLinks,
                    args,
                    selectedTables=selectedTables,
                    prunedTables=selectedTables,
                    prunedColumnCount=selectedColumnCount,
                )
            )

        if strategy == "mixed":
            for clauseName in ("select", "join", "filter", "group_order"):
                clauseTarget = targetForClause(targetLinks, item.get("gold_sql", ""), clauseName)
                if clauseTarget:
                    for _ in range(args.clause_task_weight):
                        rows.append(
                            makeRow(
                                tokenizer,
                                "clause",
                                item,
                                selectedText,
                                clauseTarget,
                                args,
                                selectedTables=selectedTables,
                                clauseName=clauseName,
                                prunedTables=selectedTables,
                                prunedColumnCount=selectedColumnCount,
                            )
                        )

            for pair in positivePairs(targetLinks, args.verify_positive, rng):
                rows.append(makeRow(tokenizer, "verify", item, prunedText, "yes", args, candidate=pair, prunedTables=pruned["tables"], prunedColumnCount=prunedColumnCount))
            for pair in hardNegatives(fullSchema, targetLinks, pruned, args.verify_negative, rng):
                rows.append(makeRow(tokenizer, "verify", item, prunedText, "no", args, candidate=pair, prunedTables=pruned["tables"], prunedColumnCount=prunedColumnCount))

    return rows


def prepareData(args: argparse.Namespace, strategy: str) -> dict[str, Path]:
    paths = pruned_sft.pathsFor(args)
    tokenizer = sft.loadTokenizer(args)
    trainExamples = pruned_sft.readJson(ROOT / args.train_json)
    evalExamples = pruned_sft.readJson(ROOT / args.validation_json)
    trainExamplesBalanced = balanceExamples(trainExamples, args)
    rowsTrain = buildRows(args, trainExamplesBalanced, tokenizer, strategy)
    rowsEval = buildRows(args, evalExamples, tokenizer, strategy)

    sft.writeJsonl(paths["train_jsonl"], rowsTrain)
    sft.writeJsonl(paths["eval_jsonl"], rowsEval)
    trainTaskCounts = Counter(row["task"] for row in rowsTrain)
    evalTaskCounts = Counter(row["task"] for row in rowsEval)
    trainDbCounts = Counter(row["db_id"] for row in rowsTrain)
    pruned_sft.writeJson(
        paths["manifest"],
        {
            "strategy": strategy,
            "base_model": args.base_model,
            "train_examples_original": len(trainExamples),
            "train_examples_balanced": len(trainExamplesBalanced),
            "train_rows": len(rowsTrain),
            "eval_rows": len(rowsEval),
            "train_task_counts": dict(sorted(trainTaskCounts.items())),
            "eval_task_counts": dict(sorted(evalTaskCounts.items())),
            "train_db_counts": dict(sorted(trainDbCounts.items())),
            "max_length": args.max_length,
            "top_tables": args.top_tables,
            "max_schema_columns": args.max_schema_columns,
            "max_columns_per_table": args.max_columns_per_table,
            "stage_columns_cap": args.stage_columns_cap,
            "candidate_table_negatives": args.candidate_table_negatives,
            "db_balance_target": args.db_balance_target,
            "db_balance_cap": args.db_balance_cap,
            "full_task_weight": args.full_task_weight,
            "table_task_weight": args.table_task_weight,
            "column_task_weight": args.column_task_weight,
            "clause_task_weight": args.clause_task_weight,
            "verify_positive": args.verify_positive,
            "verify_negative": args.verify_negative,
            "train_truncated": sum(row["truncated"] for row in rowsTrain),
            "eval_truncated": sum(row["truncated"] for row in rowsEval),
            "train_max_tokens": max(row["token_count"] for row in rowsTrain),
            "eval_max_tokens": max(row["token_count"] for row in rowsEval),
            "train_avg_pruned_columns": sum(row["pruned_column_count"] for row in rowsTrain) / len(rowsTrain),
            "eval_avg_pruned_columns": sum(row["pruned_column_count"] for row in rowsEval) / len(rowsEval),
        },
    )
    return paths


def loadAdapter(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    baseModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(baseModel, args.adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device, torch


def tablesFromLinks(links: dict[str, list[str]], schema: dict[str, Any]) -> list[str]:
    tableLookup = {table.lower(): table for table in schema["tables"]}
    out = []
    seen = set()
    for table in links:
        realTable = tableLookup.get(str(table).lower())
        if realTable and realTable.lower() not in seen and realTable.lower() != "sysdiagrams":
            seen.add(realTable.lower())
            out.append(realTable)
    return out


def cleanLinksForTables(rawLinks: dict[str, Any], schema: dict[str, Any], tables: list[str]) -> dict[str, list[str]]:
    links = baseline.filterSchemaLinks(rawLinks, schema)
    allowed = {table.lower(): table for table in tables}
    out = {}
    for table, cols in links.items():
        key = table.lower()
        if key in allowed:
            out[allowed[key]] = cols
    return out


def mergeLinks(
    linkSets: list[dict[str, list[str]]],
    selectedTables: list[str],
    tableThreshold: int,
    columnThreshold: int,
) -> dict[str, list[str]]:
    allowed = {table.lower(): table for table in selectedTables}
    tableCounts: dict[str, int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    names: dict[tuple[str, str], str] = {}
    for links in linkSets:
        for table, columns in links.items():
            tableKey = table.lower()
            tableCounts[tableKey] += 1
            for column in columns:
                key = (tableKey, column.lower())
                counts[key] += 1
                names[key] = column
    out = {
        table: []
        for table in selectedTables
        if tableCounts.get(table.lower(), 0) >= tableThreshold
    }
    for (tableKey, _columnKey), count in counts.items():
        table = allowed.get(tableKey)
        if count >= columnThreshold and table in out:
            out[table].append(names[(tableKey, _columnKey)])
    return {table: sorted(set(columns)) for table, columns in out.items()}


def runPrompt(model: Any, tokenizer: Any, device: str, torch: Any, args: argparse.Namespace, task: str, item: dict[str, Any], schemaText: str, selectedTables: list[str] | None = None, clauseName: str | None = None) -> tuple[str, dict[str, Any]]:
    system, user = buildTaskPrompt(task, item.get("question", ""), schemaText, selectedTables=selectedTables, clauseName=clauseName)
    prompt = baseline.applyChatTemplate(tokenizer, system, user)
    generationArgs = copy.copy(args)
    if task == "tables":
        generationArgs.max_new_tokens = args.table_max_new_tokens
    elif task == "clause":
        generationArgs.max_new_tokens = args.clause_max_new_tokens
    rawText = baseline.generateOne(model, tokenizer, device, torch, prompt, generationArgs)
    return rawText, baseline.tryParseMapping(rawText)


def verifyLinks(model: Any, tokenizer: Any, device: str, torch: Any, args: argparse.Namespace, item: dict[str, Any], schemaText: str, links: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    kept: dict[str, list[str]] = {table: [] for table in links}
    traces = []
    for table, columns in links.items():
        for column in columns:
            system, user = buildTaskPrompt("verify", item.get("question", ""), schemaText, candidate=(table, column))
            prompt = baseline.applyChatTemplate(tokenizer, system, user)
            generationArgs = copy.copy(args)
            generationArgs.max_new_tokens = args.verify_max_new_tokens
            rawText = baseline.generateOne(model, tokenizer, device, torch, prompt, generationArgs).strip()
            decision = rawText.lower().split()
            yes = bool(decision and decision[0].strip(".,:;") == "yes")
            if yes:
                kept.setdefault(table, []).append(column)
            traces.append({"table": table, "column": column, "raw": rawText, "keep": yes})
    return {table: sorted(set(cols)) for table, cols in kept.items()}, traces


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


def predict(args: argparse.Namespace, strategy: str) -> Path:
    model, tokenizer, device, torch = loadAdapter(args)
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
        prunedText = pruned_sft.serializePrunedSchema(pruned)

        rawTable, parsedTable = runPrompt(model, tokenizer, device, torch, args, "tables", item, prunedText)
        tableLinks = baseline.filterSchemaLinks(parsedTable, fullSchema)
        selectedTables = tablesFromLinks(tableLinks, fullSchema)

        fullLinks = {}
        rawFull = ""
        if strategy == "mixed":
            rawFull, parsedFull = runPrompt(model, tokenizer, device, torch, args, "full", item, prunedText)
            fullLinks = baseline.filterSchemaLinks(parsedFull, fullSchema)
            for table in tablesFromLinks(fullLinks, fullSchema):
                if table not in selectedTables:
                    selectedTables.append(table)

        if not selectedTables:
            selectedTables = tablesFromLinks(fullLinks, fullSchema)

        selectedSchema = schemaForTables(fullSchema, selectedTables, args.stage_columns_cap)
        selectedText = pruned_sft.serializePrunedSchema(selectedSchema)
        rawColumn, parsedColumn = runPrompt(model, tokenizer, device, torch, args, "columns", item, selectedText, selectedTables=selectedTables)
        columnLinks = cleanLinksForTables(parsedColumn, fullSchema, selectedTables)

        clauseLinks = []
        rawClauses = {}
        if strategy == "mixed" and selectedTables:
            for clauseName in ("select", "join", "filter", "group_order"):
                rawClause, parsedClause = runPrompt(
                    model,
                    tokenizer,
                    device,
                    torch,
                    args,
                    "clause",
                    item,
                    selectedText,
                    selectedTables=selectedTables,
                    clauseName=clauseName,
                )
                rawClauses[clauseName] = rawClause
                clauseLinks.append(cleanLinksForTables(parsedClause, fullSchema, selectedTables))

        verifierTrace = []
        if strategy == "mixed":
            links = mergeLinks(
                [tableLinks, fullLinks, columnLinks] + clauseLinks,
                selectedTables,
                args.table_vote_threshold,
                args.column_vote_threshold,
            )
            if args.use_verifier:
                links, verifierTrace = verifyLinks(model, tokenizer, device, torch, args, item, selectedText, links)
        else:
            links = columnLinks

        predictions.append({"question_id": item["question_id"], "schema_links": links})
        rawRows.append(
            {
                "question_id": item["question_id"],
                "db_id": dbId,
                "raw_table": rawTable,
                "raw_full": rawFull,
                "raw_column": rawColumn,
                "raw_clauses": rawClauses,
                "table_links": tableLinks,
                "full_links": fullLinks,
                "column_links": columnLinks,
                "clause_links": clauseLinks,
                "verifier": verifierTrace,
                "filtered_schema_links": links,
                "selected_tables": selectedTables,
            }
        )
        print(f"[{idx}/{len(items)}] {strategy} q{item['question_id']} {dbId}", file=sys.stderr)

    pruned_sft.writeJson(output, predictions)
    pruned_sft.writeJson(output.with_name("raw_generations.json"), rawRows)
    runEval(args, output)
    return output


def recombineMixed(args: argparse.Namespace) -> Path:
    rawRows = pruned_sft.readJson(args.raw_generations)
    predictions = []
    for row in rawRows:
        selectedTables = list(row.get("selected_tables", []))
        linkSets = [
            row.get("table_links", {}),
            row.get("full_links", {}),
            row.get("column_links", {}),
        ]
        linkSets.extend(row.get("clause_links", []))
        links = mergeLinks(
            linkSets,
            selectedTables,
            args.table_vote_threshold,
            args.column_vote_threshold,
        )
        predictions.append({"question_id": row["question_id"], "schema_links": links})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pruned_sft.writeJson(output, predictions)
    runEval(args, output)
    return output


def addTaskArgs(parser: argparse.ArgumentParser, strategy: str) -> argparse.ArgumentParser:
    choices = parser._subparsers._group_actions[0].choices
    epochCount = 4.0 if strategy == "two_stage" else 1.0
    epochLabel = "4" if strategy == "two_stage" else "1"
    for command in ("prepare", "train"):
        sp = choices[command]
        sp.add_argument("--table_task_weight", type=int, default=1)
        sp.add_argument("--column_task_weight", type=int, default=1)
        sp.add_argument("--full_task_weight", type=int, default=2 if strategy == "mixed" else 0)
        sp.add_argument("--clause_task_weight", type=int, default=1 if strategy == "mixed" else 0)
        sp.add_argument("--verify_positive", type=int, default=1 if strategy == "mixed" else 0)
        sp.add_argument("--verify_negative", type=int, default=1 if strategy == "mixed" else 0)
        sp.add_argument("--stage_columns_cap", type=int, default=0)
        sp.add_argument("--candidate_table_negatives", type=int, default=2)
        sp.set_defaults(
            base_model=DEFAULT_MODEL,
            top_tables=14,
            max_schema_columns=320,
            max_columns_per_table=48,
            max_length=8192,
            prompt_keep_head_tokens=800,
            run_name=f"{strategy}_coder_dbbal12_lr8p5e-5_r16_a32_ep{epochLabel}_len8192",
            experiment_name=f"project2_{strategy}_coder_dbbal12_lr8p5e5_r16a32_ep{epochLabel}_len8192",
            epochs=epochCount,
            learning_rates="0.000085",
            lora_ranks="16",
            lora_alphas="32",
            lora_dropout=0.05,
            warmup_ratio=0.05,
            completion_only_loss=True,
            db_balance_target=12,
            db_balance_cap=0,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            model_dtype="bfloat16",
            bf16=True,
            fp16=False,
        )

    pred = choices["predict"]
    pred.add_argument("--stage_columns_cap", type=int, default=0)
    pred.add_argument("--table_vote_threshold", type=int, default=2)
    pred.add_argument("--column_vote_threshold", type=int, default=1)
    pred.add_argument("--use_verifier", action="store_true")
    pred.add_argument("--table_max_new_tokens", type=int, default=128)
    pred.add_argument("--clause_max_new_tokens", type=int, default=256)
    pred.add_argument("--verify_max_new_tokens", type=int, default=8)
    pred.set_defaults(
        base_model=DEFAULT_MODEL,
        top_tables=14,
        max_schema_columns=320,
        max_columns_per_table=48,
        max_input_tokens=8192,
        max_new_tokens=384,
    )

    if strategy == "mixed":
        subparsers = parser._subparsers._group_actions[0]
        recombine = subparsers.add_parser("recombine")
        recombine.add_argument("--raw_generations", required=True)
        recombine.add_argument("--output", required=True)
        recombine.add_argument("--input", default="validation_input.json")
        recombine.add_argument("--gold", default="validation_gold_schema_links.json")
        recombine.add_argument("--schemas_dir", default="schemas")
        recombine.add_argument("--table_vote_threshold", type=int, default=2)
        recombine.add_argument("--column_vote_threshold", type=int, default=1)
    return parser


def buildParser(strategy: str) -> argparse.ArgumentParser:
    parser = coder_balanced_pruned_sft.buildParser()
    return addTaskArgs(parser, strategy)


def runMain(strategy: str) -> None:
    args = buildParser(strategy).parse_args()
    if args.command == "recombine":
        recombineMixed(args)
        return

    pruned_sft.prepareData = lambda parsedArgs: prepareData(parsedArgs, strategy)
    sft.buildRapidfireConfig = coder_balanced_pruned_sft.buildRapidFireConfig
    sft.createModelFn = coder_balanced_pruned_sft.createModel

    if args.command in {"prepare", "train"}:
        args.learning_rates = pruned_sft.parseCsvNumbers(args.learning_rates, float)
        args.lora_ranks = pruned_sft.parseCsvNumbers(args.lora_ranks, int)
        args.lora_alphas = pruned_sft.parseCsvNumbers(args.lora_alphas, int)
        if args.command == "prepare":
            paths = prepareData(args, strategy)
            print(f"Wrote {strategy} train data to {paths['train_jsonl']}")
            print(f"Wrote {strategy} eval data to {paths['eval_jsonl']}")
            print(f"Wrote manifest to {paths['manifest']}")
        else:
            pruned_sft.runTrain(args)
    else:
        predict(args, strategy)
