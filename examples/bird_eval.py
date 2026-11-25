"""Isolated reference benchmark CLI for `birdsql/bird_mini_dev` evaluation.

Requires only core `group-by-vibes` for `score`; ML/HF packages (`examples` extra)
are imported lazily inside `generate`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from group_by_vibes import __version__ as gbv_version
from group_by_vibes import (
    compute_sql_bleu,
    compute_valid_sql_rate,
    compute_wildcard_detector,
    extract_fenced_or_last_sql,
    normalize_dialect,
)

DATASET_REPO = "birdsql/bird_mini_dev"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_PRESETS: dict[str, str] = {
    "qwen-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "xiyansql-3b": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
    "finer-sql-3b": "griffith-bigdata/FINER-SQL-3B-v2",
}

DIALECT_FILES: dict[str, tuple[str, str, str]] = {
    "sqlite": (
        "SQLite",
        "data/mini_dev_sqlite.jsonl",
        "baseline_inference/schemas/sqlite.json",
    ),
    "postgres": (
        "PostgreSQL",
        "data/mini_dev_postgresql.jsonl",
        "baseline_inference/schemas/postgresql.json",
    ),
    "mysql": (
        "MySQL",
        "data/mini_dev_mysql.jsonl",
        "baseline_inference/schemas/mysql.json",
    ),
}
PROMPT_TEMPLATE_FILE = "baseline_inference/prompt_template.txt"


def _ddl_to_schema_dict(ddl: str, dialect: str) -> dict[str, list[str]]:
    """Parse `CREATE TABLE` DDL blocks into `{table: [columns]}` for `ValidSqlRate`."""
    schema: dict[str, list[str]] = {}
    for stmt in sqlglot.parse(ddl, read=dialect):
        if stmt is None:
            continue
        if isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Schema):
            tbl = stmt.this.this
            table_name = tbl.name if isinstance(tbl, exp.Table) else str(tbl)
            cols = [
                col.name
                for col in stmt.this.expressions
                if isinstance(col, exp.ColumnDef) and col.name
            ]
            if table_name:
                schema[table_name] = cols
    return schema


def _proportional_counts(
    rows_by_diff: dict[str, list[dict[str, Any]]],
    sample_size: int,
) -> dict[str, int]:
    total = sum(len(items) for items in rows_by_diff.values())
    if sample_size <= 0 or sample_size > total:
        raise ValueError(f"`sample_size` must be between 1 and {total}, got {sample_size}.")

    raw_quotas = {k: (len(v) * sample_size) / total for k, v in rows_by_diff.items()}
    counts = {k: int(q) for k, q in raw_quotas.items()}
    remainder = sample_size - sum(counts.values())
    for k in sorted(raw_quotas, key=lambda name: raw_quotas[name] - counts[name], reverse=True):
        if remainder == 0:
            break
        if counts[k] < len(rows_by_diff[k]):
            counts[k] += 1
            remainder -= 1
    return counts


def _stratified_sample(
    questions: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        diff = str(row.get("difficulty", "unknown"))
        by_diff[diff].append(row)

    counts = _proportional_counts(by_diff, sample_size)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for diff in sorted(by_diff):
        pool = sorted(by_diff[diff], key=lambda item: int(item["question_id"]))
        sampled.extend(rng.sample(pool, counts[diff]))
    return sampled


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _format_prompt(
    template: str,
    *,
    engine: str,
    ddl: str,
    question: str,
    evidence: str,
) -> str:
    q_block = f"{question}\nExternal Knowledge: {evidence}" if evidence else question
    return template.format(engine=engine, schema=ddl, question_block=q_block)


def _resolve_model(model: str) -> str:
    """Resolve a preset alias to its Hugging Face repo ID, or return `model` unchanged."""
    return MODEL_PRESETS.get(model, model)


def run_generate(args: argparse.Namespace) -> int:
    """Download `bird_mini_dev`, optionally sample, run inference, and flush JSONL."""
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "`generate` requires the `examples` extras: `uv sync --extra examples`."
        ) from exc

    dialect = normalize_dialect(args.dialect)
    if dialect not in DIALECT_FILES:
        raise ValueError(
            f"Unsupported BIRD dialect {args.dialect!r}; choose sqlite/postgres/mysql."
        )
    engine_label, data_file, schema_file = DIALECT_FILES[dialect]

    data_path = Path(hf_hub_download(repo_id=DATASET_REPO, filename=data_file, repo_type="dataset"))
    schema_path = Path(
        hf_hub_download(repo_id=DATASET_REPO, filename=schema_file, repo_type="dataset")
    )
    prompt_path = Path(
        hf_hub_download(repo_id=DATASET_REPO, filename=PROMPT_TEMPLATE_FILE, repo_type="dataset")
    )

    questions = _load_jsonl(data_path)
    raw_ddl_by_db: dict[str, str] = json.loads(schema_path.read_text(encoding="utf-8"))
    parsed_schemas_by_db = {
        db_id: _ddl_to_schema_dict(ddl, dialect=dialect) for db_id, ddl in raw_ddl_by_db.items()
    }
    prompt_template = prompt_path.read_text(encoding="utf-8")
    if args.sample_size is None:
        sampled = sorted(questions, key=lambda item: int(item["question_id"]))
    else:
        sampled = _stratified_sample(questions, sample_size=args.sample_size, seed=args.seed)

    model_id = _resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
    model.eval()
    sqlglot_ver = importlib.metadata.version("sqlglot")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", buffering=1) as out_fh:
        for idx, row in enumerate(sampled, 1):
            t0 = time.monotonic()
            db_id = str(row["db_id"])
            ddl = raw_ddl_by_db[db_id]
            schema = parsed_schemas_by_db[db_id]
            question = str(row["question"])
            evidence = str(row.get("evidence") or "")

            prompt_text = _format_prompt(
                prompt_template,
                engine=engine_label,
                ddl=ddl,
                question=question,
                evidence=evidence,
            )
            if getattr(tokenizer, "chat_template", None):
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                chat_prompt = prompt_text

            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            prompt_len = int(inputs["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = generated[0][prompt_len:]
            truncated = len(gen_ids) >= args.max_new_tokens
            raw_completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

            record = {
                "question_id": row["question_id"],
                "db_id": db_id,
                "difficulty": row.get("difficulty"),
                "dialect": dialect,
                "question": question,
                "evidence": evidence,
                "schema": schema,
                "gold_sql": row["SQL"],
                "raw_completion": raw_completion,
                "truncated": truncated,
                "seed": args.seed,
                "model": model_id,
                "extractor": "extract_fenced_or_last_sql",
                "group_by_vibes_version": gbv_version,
                "sqlglot_version": sqlglot_ver,
            }
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            elapsed = time.monotonic() - t0
            print(
                f"[{idx}/{len(sampled)}] db={db_id} qid={row['question_id']} ({elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
    return 0


def run_score(args: argparse.Namespace) -> int:
    """Score saved BIRD JSONL completions strictly via core `group-by-vibes` APIs."""
    rows = _load_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"Input JSONL is empty: {args.input}")

    dialect = normalize_dialect(args.dialect or str(rows[0].get("dialect") or "sqlite"))
    completions = [str(row["raw_completion"]) for row in rows]
    gold_sqls = [str(row["gold_sql"]) for row in rows]
    schemas = [row.get("schema") for row in rows]

    sql_bleu_res = compute_sql_bleu(
        completions,
        gold_sqls,
        dialect=dialect,
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )
    valid_sql_res = compute_valid_sql_rate(
        completions,
        schemas,
        dialect=dialect,
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )
    wildcard_res = compute_wildcard_detector(
        completions,
        dialect=dialect,
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )

    by_diff: dict[str, dict[str, Any]] = {}
    diff_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        diff_groups[str(row.get("difficulty") or "unknown")].append(row)
    for diff, items in sorted(diff_groups.items()):
        d_preds = [str(r["raw_completion"]) for r in items]
        d_golds = [str(r["gold_sql"]) for r in items]
        d_schemas = [r.get("schema") for r in items]
        d_bleu = compute_sql_bleu(
            d_preds, d_golds, dialect=dialect, extractor=extract_fenced_or_last_sql
        )
        d_valid = compute_valid_sql_rate(
            d_preds, d_schemas, dialect=dialect, extractor=extract_fenced_or_last_sql
        )
        d_wild = compute_wildcard_detector(
            d_preds, dialect=dialect, extractor=extract_fenced_or_last_sql
        )
        by_diff[diff] = {
            "num_rows": len(items),
            "bleu": d_bleu["bleu"],
            "valid_sql_rate": d_valid["valid_sql_rate"],
            "syntax_valid_rate": d_valid["syntax_valid_rate"],
            "schema_grounded_rate": d_valid["schema_grounded_rate"],
            "wildcard_rate": d_wild["wildcard_rate"],
        }

    summary: dict[str, Any] = {
        "metadata": {
            "num_rows": len(rows),
            "dialect": dialect,
            "extractor": "extract_fenced_or_last_sql",
            "truncated_count": sum(1 for row in rows if row.get("truncated")),
            "group_by_vibes_version": gbv_version,
            "sqlglot_version": importlib.metadata.version("sqlglot"),
        },
        "sql_bleu": sql_bleu_res,
        "valid_sql_rate": valid_sql_res,
        "wildcard_detector": wildcard_res,
        "by_difficulty": by_diff,
    }

    rendered = json.dumps(summary, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BIRD Mini-Dev benchmark runner for group-by-vibes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate",
        help="Run full split or stratified sample generation to JSONL.",
    )
    gen.add_argument("--output", required=True, help="Output JSONL path.")
    gen.add_argument("--dialect", default="sqlite", help="sqlite, postgres, or mysql.")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size (omit to run the full split).",
    )
    gen.add_argument("--max-new-tokens", type=int, default=2048)
    gen.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Hugging Face model ID or preset ({', '.join(MODEL_PRESETS)}). "
            f"Default: {DEFAULT_MODEL}."
        ),
    )

    score = subparsers.add_parser("score", help="Score JSONL records with core evaluators.")
    score.add_argument("--input", required=True, help="Input JSONL path.")
    score.add_argument("--output", default=None, help="Optional output JSON report path.")
    score.add_argument("--dialect", default=None, help="Override dialect stored in JSONL rows.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    if args.command == "score":
        return run_score(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
