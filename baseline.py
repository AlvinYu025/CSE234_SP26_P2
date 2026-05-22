"""
No-SFT model baseline for Project 2 schema linking.

This script runs a public <=2B instruct model directly, without loading a
LoRA/QLoRA adapter or full fine-tuned checkpoint. It is intended as the
"base model before SFT" baseline for comparison against later RapidFire SFT
experiments.

Required contract:

    python3 baseline.py --input validation_input.json

Optional common use:

    python3 baseline.py \
        --input validation_input.json \
        --schemas_dir schemas \
        --model_name Qwen/Qwen2.5-0.5B-Instruct

By default, outputs are written under:

    output/<model_slug>/baseline/

For SFT experiments, use descriptive run names such as:

    --run_name sft_lr2e-4_r16_epochs3_schema_full

Dependencies:
    pip install torch transformers
"""
import argparse
import ast
import json
import os
import re
import sys


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def read_json(path):
    with open(path) as f:
        return json.load(f)


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "run"


def resolve_output_paths(args):
    model_slug = slugify(args.model_name)
    run_name = slugify(args.run_name)
    run_dir = os.path.join(args.output_root, model_slug, run_name)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(run_dir, "predictions.json")
    else:
        run_dir = os.path.dirname(output_path) or "."

    raw_path = args.raw_generations
    if raw_path is None and not args.no_raw_generations:
        raw_path = os.path.join(run_dir, "raw_generations.json")

    config_path = os.path.join(run_dir, "run_config.json")
    return run_dir, output_path, raw_path, config_path


def schema_filename(db_id):
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def load_schema(schemas_dir, db_id):
    path = os.path.join(schemas_dir, schema_filename(db_id))
    data = read_json(path)

    tables = list(data["table_names_original"])
    columns_by_table = {table: [] for table in tables}
    types_by_pair = {}
    column_types = data.get("column_types", ["TEXT"] * len(data["column_names_original"]))

    for (tidx, col), col_type in zip(data["column_names_original"], column_types):
        if tidx == -1:
            continue
        table = tables[tidx]
        columns_by_table[table].append(col)
        types_by_pair[(table, col)] = col_type or "TEXT"

    return {
        "db_id": data.get("db_id", db_id),
        "tables": tables,
        "columns_by_table": columns_by_table,
        "types_by_pair": types_by_pair,
    }


def serialize_schema(schema, include_types=False):
    lines = [f"Database: {schema['db_id']}", "Schema:"]
    for table in schema["tables"]:
        if table.lower() == "sysdiagrams":
            continue

        columns = []
        for col in schema["columns_by_table"][table]:
            if include_types:
                col_type = schema["types_by_pair"].get((table, col), "TEXT")
                columns.append(f"{col}:{col_type}")
            else:
                columns.append(col)
        lines.append(f"- {table}({', '.join(columns)})")
    return "\n".join(lines)


def build_prompt(question, schema_text):
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


def apply_chat_template(tokenizer, system, user):
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


def strip_code_fences(text):
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_balanced_json_object(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    quote = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {"'", '"'}:
            in_string = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def try_parse_mapping(text):
    candidates = []
    cleaned = strip_code_fences(text)
    candidates.append(cleaned)

    balanced = extract_balanced_json_object(cleaned)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        for payload in (candidate, repaired):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(payload)
                except (SyntaxError, ValueError):
                    continue

            if isinstance(parsed, dict) and isinstance(parsed.get("schema_links"), dict):
                return parsed["schema_links"]
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict) and isinstance(first.get("schema_links"), dict):
                    return first["schema_links"]
    return {}


def normalize_column_list(cols):
    if isinstance(cols, list):
        return cols
    if isinstance(cols, tuple):
        return list(cols)
    if isinstance(cols, str):
        return [piece.strip() for piece in cols.split(",") if piece.strip()]
    return []


def filter_schema_links(raw_links, schema):
    if not isinstance(raw_links, dict):
        return {}

    table_lookup = {table.lower(): table for table in schema["tables"]}
    column_lookup = {
        table: {col.lower(): col for col in schema["columns_by_table"][table]}
        for table in schema["tables"]
    }

    selected = {}

    for raw_table, raw_cols in raw_links.items():
        table_key = str(raw_table).lower()
        if table_key not in table_lookup:
            continue
        table = table_lookup[table_key]
        if table.lower() == "sysdiagrams":
            continue

        cols = []
        seen = set()
        for raw_col in normalize_column_list(raw_cols):
            col_key = str(raw_col).lower()
            if col_key in column_lookup[table]:
                col = column_lookup[table][col_key]
                if col not in seen:
                    seen.add(col)
                    cols.append(col)
        selected[table] = cols

    table_order = {table: idx for idx, table in enumerate(schema["tables"])}
    column_order = {
        (table, col): idx
        for table in schema["tables"]
        for idx, col in enumerate(schema["columns_by_table"][table])
    }

    out = {}
    for table in sorted(selected, key=lambda t: table_order.get(t, 10**9)):
        out[table] = sorted(
            selected[table],
            key=lambda col: column_order.get((table, col), 10**9),
        )
    return out


def load_model_and_tokenizer(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install torch transformers"
        ) from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "auto":
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device, torch


def generate_one(model, tokenizer, device, torch, prompt, args):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[-1]
    generated = output_ids[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--output_root", default="output")
    ap.add_argument("--run_name", default="baseline")
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--max_input_tokens", type=int, default=16000)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--include_types", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--raw_generations", default=None)
    ap.add_argument("--no_raw_generations", action="store_true")
    args = ap.parse_args()

    run_dir, output_path, raw_path, config_path = resolve_output_paths(args)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if raw_path:
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)

    items = read_json(args.input)
    if not isinstance(items, list):
        raise ValueError("--input must be a JSON list")

    model, tokenizer, device, torch = load_model_and_tokenizer(args)

    schema_cache = {}
    predictions = []
    raw_rows = []

    for idx, item in enumerate(items, start=1):
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        schema = schema_cache[db_id]

        schema_text = serialize_schema(schema, include_types=args.include_types)
        system, user = build_prompt(item.get("question", ""), schema_text)
        prompt = apply_chat_template(tokenizer, system, user)
        raw_text = generate_one(model, tokenizer, device, torch, prompt, args)

        raw_links = try_parse_mapping(raw_text)
        links = filter_schema_links(raw_links, schema)
        predictions.append({
            "question_id": item["question_id"],
            "schema_links": links,
        })
        raw_rows.append({
            "question_id": item["question_id"],
            "db_id": db_id,
            "raw_generation": raw_text,
            "parsed_schema_links": raw_links,
            "filtered_schema_links": links,
        })

        print(f"[{idx}/{len(items)}] q{item['question_id']} {db_id}", file=sys.stderr)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote {len(predictions)} predictions to {output_path}")

    if raw_path:
        with open(raw_path, "w") as f:
            json.dump(raw_rows, f, indent=2)
        print(f"Wrote raw generations to {raw_path}")

    run_config = {
        "input": args.input,
        "output": output_path,
        "raw_generations": raw_path,
        "schemas_dir": args.schemas_dir,
        "model_name": args.model_name,
        "run_name": args.run_name,
        "output_root": args.output_root,
        "device": args.device,
        "dtype": args.dtype,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "include_types": args.include_types,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Wrote run config to {config_path}")


if __name__ == "__main__":
    main()
