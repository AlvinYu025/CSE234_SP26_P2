"""Batched inference for the packaged mixed-task adapter.

Same per-question views and voting as task_sft_common.predict('mixed') without
the verifier, but generation is batched across questions per stage so a full
run fits the grading time budget. Stages: (A) tables + full, (B) columns on the
selected tables, (C) four clause views; link sets are merged by vote
(table_threshold=1, column_threshold=2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import baseline
import pruned_sft
import task_sft_common as task

ROOT = Path(__file__).resolve().parent
CLAUSES = ("select", "join", "filter", "group_order")


def batchGenerate(model, tokenizer, device, torch, prompts, maxNewTokens, batchSize, maxInputTokens):
    """Greedy generation, batched with left padding. On CUDA out-of-memory the
    batch is halved and retried (down to one prompt) so a smaller grading GPU
    cannot terminate the run."""
    if not prompts:
        return []

    def generateChunk(chunk):
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=maxInputTokens)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=maxNewTokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        promptLen = encoded["input_ids"].shape[1]
        return [tokenizer.decode(generated[offset][promptLen:], skip_special_tokens=True).strip() for offset in range(len(chunk))]

    priorSide = tokenizer.padding_side
    tokenizer.padding_side = "left"
    decoded = []
    try:
        index = 0
        while index < len(prompts):
            size = batchSize
            while True:
                chunk = prompts[index : index + size]
                try:
                    decoded.extend(generateChunk(chunk))
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise
                    size = max(1, size // 2)
                    print(f"  [gen] OOM, retrying with batch {size}", file=sys.stderr)
            index += len(chunk)
            print(f"  [gen] {index}/{len(prompts)}", file=sys.stderr)
    finally:
        tokenizer.padding_side = priorSide
    return decoded


def runPredict(args: argparse.Namespace) -> Path:
    model, tokenizer, device, torch = task.loadAdapter(args)
    items = pruned_sft.readJson(args.input)
    trainIndex = pruned_sft.buildTrainingIndex(pruned_sft.readJson(ROOT / args.train_json))
    schemaCache: dict = {}

    # Pruned schema + serialized text for every question.
    contexts = []
    for item in items:
        dbId = item["db_id"]
        if dbId not in schemaCache:
            schemaCache[dbId] = pruned_sft.loadSchemaWithTerms(ROOT / args.schemas_dir, dbId)
        fullSchema = schemaCache[dbId]
        pruned = pruned_sft.pruneSchema(
            item.get("question", ""), dbId, fullSchema, trainIndex,
            args.top_tables, args.max_schema_columns, args.max_columns_per_table,
            args.include_all_selected_table_columns,
        )
        contexts.append({"item": item, "schema": fullSchema, "prunedText": pruned_sft.serializePrunedSchema(pruned)})

    def promptsFor(taskName, textKey, selectedKey=None, clauseName=None):
        built = []
        for context in contexts:
            selected = context.get(selectedKey) if selectedKey else None
            system, user = task.buildTaskPrompt(
                taskName, context["item"].get("question", ""), context[textKey],
                selectedTables=selected, clauseName=clauseName,
            )
            built.append(baseline.applyChatTemplate(tokenizer, system, user))
        return built

    batchSize, maxInput = args.batch_size, args.max_input_tokens

    # Stage A: tables + full -> selected tables and the per-question column-stage schema.
    tableRaw = batchGenerate(model, tokenizer, device, torch, promptsFor("tables", "prunedText"), args.table_max_new_tokens, batchSize, maxInput)
    fullRaw = batchGenerate(model, tokenizer, device, torch, promptsFor("full", "prunedText"), args.max_new_tokens, batchSize, maxInput)
    for context, tableText, fullText in zip(contexts, tableRaw, fullRaw):
        fullSchema = context["schema"]
        tableLinks = baseline.filterSchemaLinks(baseline.tryParseMapping(tableText), fullSchema)
        fullLinks = baseline.filterSchemaLinks(baseline.tryParseMapping(fullText), fullSchema)
        selected = task.tablesFromLinks(tableLinks, fullSchema)
        for tableName in task.tablesFromLinks(fullLinks, fullSchema):
            if tableName not in selected:
                selected.append(tableName)
        context["tableLinks"], context["fullLinks"], context["selected"] = tableLinks, fullLinks, selected
        selectedSchema = task.schemaForTables(fullSchema, selected, args.stage_columns_cap)
        context["selectedText"] = pruned_sft.serializePrunedSchema(selectedSchema)

    # Stage B: columns on the selected tables.
    columnRaw = batchGenerate(model, tokenizer, device, torch, promptsFor("columns", "selectedText", selectedKey="selected"), args.max_new_tokens, batchSize, maxInput)
    for context, columnText in zip(contexts, columnRaw):
        context["columnLinks"] = task.cleanLinksForTables(baseline.tryParseMapping(columnText), context["schema"], context["selected"])

    # Stage C: four clause views.
    clauseLinksByQuestion = [[] for _ in contexts]
    for clauseName in CLAUSES:
        clauseRaw = batchGenerate(model, tokenizer, device, torch, promptsFor("clause", "selectedText", selectedKey="selected", clauseName=clauseName), args.clause_max_new_tokens, batchSize, maxInput)
        for index, (context, clauseText) in enumerate(zip(contexts, clauseRaw)):
            clauseLinksByQuestion[index].append(task.cleanLinksForTables(baseline.tryParseMapping(clauseText), context["schema"], context["selected"]))

    # Merge link sets by vote.
    predictions = []
    for index, context in enumerate(contexts):
        linkSets = [context["tableLinks"], context["fullLinks"], context["columnLinks"]] + clauseLinksByQuestion[index]
        links = task.mergeLinks(linkSets, context["selected"], args.table_vote_threshold, args.column_vote_threshold)
        predictions.append({"question_id": context["item"]["question_id"], "schema_links": links})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pruned_sft.writeJson(output, predictions)
    if getattr(args, "gold", None):
        task.runEval(args, output)
    return output


def buildParser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter_path", default=str(ROOT / "adapter"))
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--schemas_dir", default="schemas")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--top_tables", type=int, default=14)
    ap.add_argument("--max_schema_columns", type=int, default=320)
    ap.add_argument("--max_columns_per_table", type=int, default=48)
    ap.add_argument("--include_all_selected_table_columns", action="store_true")
    ap.add_argument("--stage_columns_cap", type=int, default=0)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--table_max_new_tokens", type=int, default=128)
    ap.add_argument("--clause_max_new_tokens", type=int, default=256)
    ap.add_argument("--table_vote_threshold", type=int, default=1)
    ap.add_argument("--column_vote_threshold", type=int, default=2)
    ap.add_argument("--gold", default=None)
    return ap


if __name__ == "__main__":
    runPredict(buildParser().parse_args())
