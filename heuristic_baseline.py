"""
Runnable baseline for Project 2 schema linking.

This script follows the required grading contract:

    python3 main.py --input input.json --output predictions.json

It is intentionally dependency-free and does not call any external inference
API. The implementation is a conservative baseline, not the final SFT model:
it combines schema-name matching with weak signals from the released training
split, then filters every predicted identifier through the target schema so
the output never contains schema-invalid tables or columns.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict


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
    "was", "were", "what", "where", "which", "with", "without",
    "no", "num",
}

SYNONYMS = {
    "average": {"avg", "mean", "average"},
    "avg": {"avg", "mean", "average"},
    "amount": {"amount", "amt", "sum", "total"},
    "area": {"area", "zone", "region"},
    "body": {"body", "bod"},
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
    "avg", "bag", "bank", "body", "business", "card", "case", "category",
    "class", "code", "condition", "contract", "coord", "count", "crash",
    "credit", "date", "db", "debit", "decay", "dept", "desc", "distance",
    "doc", "email", "emp", "employee", "entity", "entry", "error", "event",
    "family", "final", "fire", "font", "grade", "height", "id", "injury",
    "item", "level", "loc", "location", "make", "max", "min", "model",
    "name", "no", "note", "notes", "obj", "observer", "open", "order",
    "partner", "percent", "plot", "point", "price", "project", "qty",
    "quantity", "rate", "region", "remark", "remarks", "role", "salary",
    "school", "seat", "service", "solution", "species", "state", "status",
    "station", "subplot", "sum", "table", "text", "time", "tire", "total",
    "trans", "type", "utm", "value", "veh", "vehicle", "ver", "wind", "x",
    "y", "year",
}


def schemaFilename(db_id):
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def readJson(path):
    with open(path) as f:
        return json.load(f)


def singularize(token):
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def expandToken(token):
    base = singularize(token.lower())
    out = {base}
    out.update(SYNONYMS.get(base, set()))
    return out


def splitIdentifier(identifier):
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


def tokenizeText(text):
    tokens = []
    for raw in re.findall(r"[A-Za-z]+|\d+", str(text).lower()):
        if singularize(raw) in STOPWORDS:
            continue
        for token in expandToken(raw):
            if token and token not in STOPWORDS:
                tokens.append(token)
    return set(tokens)


def identifierTerms(identifier):
    terms = set()
    for token in splitIdentifier(identifier):
        for expanded in expandToken(token):
            if expanded and expanded not in STOPWORDS:
                terms.add(expanded)
    compact = re.sub(r"[^A-Za-z0-9]+", "", str(identifier)).lower()
    if compact:
        terms.add(compact)
    return terms


def loadSchema(schemas_dir, db_id):
    path = os.path.join(schemas_dir, schemaFilename(db_id))
    data = readJson(path)
    tables = list(data["table_names_original"])
    columns_by_table = {table: [] for table in tables}
    column_types = data.get("column_types", ["TEXT"] * len(data["column_names_original"]))
    column_type_by_pair = {}
    for (tidx, col), ctype in zip(data["column_names_original"], column_types):
        if tidx == -1:
            continue
        table = tables[tidx]
        columns_by_table[table].append(col)
        column_type_by_pair[(table, col)] = ctype

    table_terms = {table: identifierTerms(table) for table in tables}
    column_terms = {}
    for table, columns in columns_by_table.items():
        for col in columns:
            column_terms[(table, col)] = identifierTerms(col)

    return {
        "tables": tables,
        "columns_by_table": columns_by_table,
        "column_types": column_type_by_pair,
        "table_terms": table_terms,
        "column_terms": column_terms,
    }


def loadTrainingExamples(path):
    if not path or not os.path.exists(path):
        return []
    data = readJson(path)
    if not isinstance(data, list):
        return []
    return data


def buildTrainingIndex(examples):
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
        tokens = tokenizeText(item.get("question", ""))
        by_db[db_id].append({"tokens": tokens, "links": links})
        for table, cols in links.items():
            table_priors[db_id][table] += 1
        for token in sorted(tokens):
            table_token_counts[(db_id, token)][table] += 1
            if isinstance(cols, list):
                for col in cols:
                    column_priors[db_id][(table, col)] += 1
                    for token in sorted(tokens):
                        column_token_counts[(db_id, token)][(table, col)] += 1

    return {
        "by_db": dict(by_db),
        "table_token_counts": table_token_counts,
        "column_token_counts": column_token_counts,
        "table_priors": table_priors,
        "column_priors": column_priors,
    }


def lexicalScore(question_tokens, identifier_token_set, identifier):
    if not identifier_token_set:
        return 0.0

    overlap = question_tokens & identifier_token_set
    score = 0.0
    if overlap:
        score += len(overlap) / math.sqrt(max(1, len(identifier_token_set)))

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


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def addTrainingSignals(db_id, question_tokens, train_index, table_scores, column_scores):
    examples = train_index["by_db"].get(db_id, [])
    nearest = []
    for ex in examples:
        sim = jaccard(question_tokens, ex["tokens"])
        if sim > 0:
            nearest.append((sim, ex["links"]))
    nearest.sort(key=lambda x: x[0], reverse=True)

    for rank, (sim, links) in enumerate(nearest[:5]):
        weight = sim * (1.4 / (rank + 1))
        for table, cols in links.items():
            table_scores[table] += weight
            if isinstance(cols, list):
                for col in cols:
                    column_scores[(table, col)] += weight

    for token in sorted(question_tokens):
        table_counts = train_index["table_token_counts"].get((db_id, token), Counter())
        for table, count in table_counts.most_common(5):
            table_scores[table] += min(0.45, 0.08 * count)
        column_counts = train_index["column_token_counts"].get((db_id, token), Counter())
        for pair, count in column_counts.most_common(8):
            column_scores[pair] += min(0.35, 0.06 * count)


def chooseFallbackTable(db_id, schema, train_index, table_scores):
    if table_scores:
        return max(table_scores.items(), key=lambda kv: kv[1])[0]
    priors = train_index["table_priors"].get(db_id)
    if priors:
        for table, _count in priors.most_common():
            if table in schema["columns_by_table"]:
                return table
    for table in schema["tables"]:
        if table.lower() != "sysdiagrams":
            return table
    return schema["tables"][0] if schema["tables"] else None


def predictSchemaLinks(question, db_id, schemas_dir, train_index):
    schema = loadSchema(schemas_dir, db_id)
    question_tokens = tokenizeText(question)
    table_scores = defaultdict(float)
    column_scores = defaultdict(float)
    table_order = {table: idx for idx, table in enumerate(schema["tables"])}
    column_order = {}
    order_idx = 0
    for table in schema["tables"]:
        for col in schema["columns_by_table"][table]:
            column_order[(table, col)] = order_idx
            order_idx += 1

    for table in schema["tables"]:
        if table.lower() == "sysdiagrams":
            continue
        score = lexicalScore(question_tokens, schema["table_terms"][table], table)
        if score > 0:
            table_scores[table] += 1.25 * score

    for table, columns in schema["columns_by_table"].items():
        if table.lower() == "sysdiagrams":
            continue
        table_hint = table_scores.get(table, 0.0)
        for col in columns:
            score = lexicalScore(question_tokens, schema["column_terms"][(table, col)], col)
            if score > 0:
                column_scores[(table, col)] += score + min(0.35, 0.12 * table_hint)

    addTrainingSignals(db_id, question_tokens, train_index, table_scores, column_scores)

    valid_columns = {
        (table, col)
        for table, columns in schema["columns_by_table"].items()
        for col in columns
    }
    column_candidates = [
        (pair, score)
        for pair, score in column_scores.items()
        if pair in valid_columns and pair[0].lower() != "sysdiagrams"
    ]
    column_candidates.sort(
        key=lambda x: (-x[1], table_order.get(x[0][0], 10**9), column_order.get(x[0], 10**9))
    )

    selected_columns = defaultdict(list)
    if column_candidates:
        best_col_score = column_candidates[0][1]
        cutoff = max(0.95, best_col_score * 0.50)
        for (table, col), score in column_candidates[:7]:
            if score >= cutoff:
                selected_columns[table].append(col)

    selected_tables = set(selected_columns)
    table_candidates = [
        (table, score)
        for table, score in table_scores.items()
        if table in schema["columns_by_table"] and table.lower() != "sysdiagrams"
    ]
    table_candidates.sort(key=lambda x: (-x[1], table_order.get(x[0], 10**9)))
    if table_candidates:
        best_table_score = table_candidates[0][1]
        cutoff = max(0.85, best_table_score * 0.60)
        for table, score in table_candidates[:3]:
            if score >= cutoff:
                selected_tables.add(table)

    if not selected_tables:
        fallback = chooseFallbackTable(db_id, schema, train_index, table_scores)
        if fallback:
            selected_tables.add(fallback)

    if len(selected_tables) > 3:
        def finalTableScore(table):
            best_col = max(
                (score for (t, _col), score in column_candidates if t == table),
                default=0.0,
            )
            return table_scores.get(table, 0.0) + 0.8 * best_col

        selected_tables = set(
            sorted(
                selected_tables,
                key=lambda t: (-finalTableScore(t), table_order.get(t, 10**9)),
            )[:3]
        )

    out = {}
    for table in sorted(selected_tables):
        cols = selected_columns.get(table, [])
        schema_order = {col: idx for idx, col in enumerate(schema["columns_by_table"][table])}
        cols = sorted(set(cols), key=lambda c: schema_order.get(c, 10**9))
        out[table] = cols
    return out


def defaultTrainingPath():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "train.json"),
        os.path.join(script_dir, "train.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--training_data", default=defaultTrainingPath())
    args = ap.parse_args()

    items = readJson(args.input)
    if not isinstance(items, list):
        raise ValueError("--input must be a JSON list")

    training_examples = loadTrainingExamples(args.training_data)
    train_index = buildTrainingIndex(training_examples)

    predictions = []
    for item in items:
        links = predictSchemaLinks(
            item.get("question", ""),
            item["db_id"],
            args.schemas_dir,
            train_index,
        )
        predictions.append({
            "question_id": item["question_id"],
            "schema_links": links,
        })

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
