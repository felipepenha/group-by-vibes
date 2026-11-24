# group-by-vibes

[![CI](https://img.shields.io/github/actions/workflow/status/felipepenha/group-by-vibes/ci.yml?branch=main&label=CI)](https://github.com/felipepenha/group-by-vibes/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/github/license/felipepenha/group-by-vibes)](LICENSE)
[![Core dependency: sqlglot only](https://img.shields.io/badge/deps-sqlglot%20only-2E7D32)](pyproject.toml)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-1f6feb)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Standalone, `sqlglot`-only Python package for deterministic Text2SQL evaluation—both when golden SQL references exist (`SQL-BLEU`) and when screening unlabelled model outputs without executing queries against a database (`Valid SQL Rate`, `Wildcard Detector`).

> **Reference:** [Chen et al. (2019), NL2pSQL](https://aclanthology.org/D19-1262/) — Fuxiang Chen, Seung-won Hwang, Jaegul Choo, Jung-Woo Ha, and Sunghun Kim. *NL2pSQL: Generating Pseudo-SQL Queries from Under-Specified Natural Language Questions*. EMNLP-IJCNLP 2019, pages 2603–2613. DOI: `10.18653/v1/D19-1262`.

---

## Installation

```bash
uv add group-by-vibes
```

Core installation requires only `sqlglot>=30.0.0,<31.0.0` (zero runtime dependency on `evaluate`, `datasets`, `pyarrow`, `transformers`, or `torch`).

For optional benchmark inference utilities:

```bash
uv sync --extra examples
```

---

## Evaluators (`evaluate`-Shaped Library API)

`group-by-vibes` provides three synchronous evaluators following Hugging Face `evaluate` terminology—one `Metric` (`SqlBleu`) and two `Measurement`s (`ValidSqlRate`, `WildcardDetector`):

* **`SqlBleu` (`load("sql_bleu")`)**: Parses and canonicalizes SQL ASTs (`keyword casing`, whitespace, comment/semicolon normalization, lowercasing unquoted identifiers while preserving quoted identifiers, and sorting commutative `AND`/`OR`, `=`/`!=`, and subquery-free `IN (...)` expressions), skipping unparseable predictions (`None`) or rows where zero references parse while reporting `"translation_length"`, `parse_success_rate`, and `reference_parse_success_rate`.
* **`ValidSqlRate` (`load("valid_sql_rate")`)**: Performs lenient dialect-aware `sqlglot` AST parsing (rejecting empty queries, multiple statements, and `exp.Command` fallbacks) and optional per-query Python `dict` schema grounding via `sqlglot.optimizer.scope.build_scope`, reporting `valid_sql_rate`, `syntax_valid_rate`, `schema_grounded_rate`, and 8-key `failure_counts`.
* **`WildcardDetector` (`load("wildcard_detector")`)**: Detects projection wildcards (`SELECT *`, `table.*`, `* EXCEPT`/`EXCLUDE` -> `1`; ignoring `COUNT(*)`, direct `EXISTS (SELECT * ...)`, `RETURNING *`, and `*` arithmetic -> `0`; skipping unparseable rows -> `None`).

All evaluators prioritize **PostgreSQL (`"postgres"`)** and **SQLite (`"sqlite"`)** in documentation and examples, while accepting any SQL dialect supported by `sqlglot.Dialect.get_or_raise` (plus aliases `"postgresql"` and `"sqlite3"`, and generic ANSI `""`). Because v1 executes no queries against a database engine, these checks complement—rather than replace—execution accuracy (`EX`).

---

## Usage Examples

### 1. Golden-Set Evaluation (`SqlBleu`, `ValidSqlRate`, `WildcardDetector`)

```python
from group_by_vibes import load

sql_bleu = load("sql_bleu", dialect="postgres")
sql_bleu.add(
    prediction="select name, age from users where status = 'active' and role = 'admin'",
    reference=["SELECT name, age FROM users WHERE role = 'admin' AND status = 'active'"],
)
print(sql_bleu.compute())

valid_sql = load("valid_sql_rate", dialect="sqlite")
valid_sql.add(
    data='SELECT "Singer(s)" FROM "1-29135051-2"',
    schema={"1-29135051-2": ["Singer(s)", "Year"]},
)
print(valid_sql.compute())

wildcards = load("wildcard_detector", dialect="postgres", include_per_example=True)
wildcards.add_batch(["SELECT u.* FROM users u", "SELECT COUNT(*) FROM users"])
print(wildcards.compute())
```

### 2. Stateless Functions & Optional Extraction (`extract_fenced_or_last_sql`)

```python
from group_by_vibes import (
    compute_sql_bleu,
    compute_valid_sql_rate,
    compute_wildcard_detector,
    extract_fenced_or_last_sql,
)

raw_completions = ["```sql\nSELECT name FROM users WHERE id = 1\n```"]
schemas = [{"users": ["id", "name"]}]

print(compute_valid_sql_rate(raw_completions, schemas, dialect="sqlite", extractor=extract_fenced_or_last_sql))
print(compute_wildcard_detector(raw_completions, dialect="sqlite", extractor=extract_fenced_or_last_sql))
print(compute_sql_bleu(raw_completions, ["SELECT name FROM users WHERE 1 = id"], dialect="sqlite", extractor=extract_fenced_or_last_sql))
```

### 3. Out-of-Band Runtime Concurrency Note

Streaming `BaseEvaluator` instances (`SqlBleu`, `ValidSqlRate`, `WildcardDetector`) maintain single-writer mutable accumulators without locks. When screening runtime queries from background worker threads, instantiate one evaluator per worker, funnel updates through a single aggregation thread, or call stateless `compute_valid_sql_rate` / `compute_wildcard_detector` functions.

---

## Reference Benchmark (`examples/`)

See [examples/README.md](examples/README.md) and [examples/bird_eval.py](examples/bird_eval.py) for an isolated CLI evaluating `birdsql/bird_mini_dev` (SQLite, PostgreSQL, and MySQL) with `Qwen/Qwen2.5-Coder-3B-Instruct` (~6.2 GB BF16, suited to Apple Silicon and laptops) or `Snowflake/Arctic-Text2SQL-R1-7B` (~15.2 GB BF16, targeting dedicated 24 GB+ CUDA GPUs) for `generate`, plus offline scoring in a core-only environment (`score`).

---

## Development Checks

```bash
uv sync --group dev
uv run pytest -v --cov=group_by_vibes --cov-branch
uv run ruff check src/ tests/ examples/
uv run mypy src/group_by_vibes/
```
