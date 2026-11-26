<p align="center">
  <img src="https://raw.githubusercontent.com/felipepenha/group-by-vibes/main/docs/assets/banner-github.png" alt="group-by-vibes — Text2SQL Evals to Deal With AI Vibes" width="100%">
</p>

# group-by-vibes

[![PyPI](https://img.shields.io/pypi/v/group-by-vibes)](https://pypi.org/project/group-by-vibes/)
[![CI](https://img.shields.io/github/actions/workflow/status/felipepenha/group-by-vibes/ci.yml?branch=main&label=CI)](https://github.com/felipepenha/group-by-vibes/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-6c6ef5)](https://felipepenha.github.io/group-by-vibes/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/felipepenha/group-by-vibes/blob/main/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/github/license/felipepenha/group-by-vibes)](https://github.com/felipepenha/group-by-vibes/blob/main/LICENSE)
[![Core dependency: sqlglot only](https://img.shields.io/badge/deps-sqlglot%20only-2E7D32)](https://github.com/felipepenha/group-by-vibes/blob/main/pyproject.toml)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-1f6feb)](https://github.com/felipepenha/group-by-vibes/blob/main/pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Text2SQL (NL2SQL) is one of the most relevant agentic tasks in the enterprise, yet general-purpose LLM evaluation libraries rarely address SQL-specific failure modes.

**`group-by-vibes`** is a focused, niche Python package built specifically for deterministic Text2SQL evaluation—not a general NLP benchmark suite. It mirrors the [Hugging Face `evaluate`](https://github.com/huggingface/evaluate) interface (`load`, `add`, `add_batch`, `compute`) so teams can adopt it with familiar `Metric` and `Measurement` semantics, while providing specialized Text2SQL metrics and measurements not covered in `evaluate`, backed only by `sqlglot` with zero heavy ML runtime dependencies.

[Documentation](https://felipepenha.github.io/group-by-vibes/) · [Installation](#installation) · [Quickstart](#quickstart) · [Available Evaluators](#available-evaluators) · [Usage Patterns](#usage-patterns) · [Reference Benchmark](#reference-benchmark) · [Contributing](#contributing) · [References](#references)

---

## Why `group-by-vibes`?

* **Drop-in `evaluate` ergonomics:** Uses the same `load()` -> `add()` / `add_batch()` -> `compute()` lifecycle and `Metric` / `Measurement` terminology as Hugging Face `evaluate`, plus stateless `compute_*` functions.
* **SQL-aware, not string-based:** Evaluates parsed SQL ASTs across dialects via `sqlglot` rather than brittle text matching.
* **Zero database or ML stack required:** Ships with `sqlglot` as the sole runtime dependency (no `evaluate`, `datasets`, `pyarrow`, `transformers`, or `torch`), complementing execution accuracy (`EX`) in offline evals and runtime screening.
* **Built to grow for Text2SQL:** Designed as an extensible catalog of reference-based **Metrics** and reference-free **Measurements** tailored to enterprise NL2SQL systems.

---

## Installation

Install from [PyPI](https://pypi.org/project/group-by-vibes/) (requires Python `>=3.10`).

```bash
# With uv
uv add group-by-vibes

# Or with pip
pip install group-by-vibes
```

Core installation pulls only `sqlglot>=30.0.0,<31.0.0`. For optional benchmark inference utilities under [`examples/`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/README.md):

```bash
uv sync --extra examples
```

---

## Quickstart

Load evaluators by name using the same pattern as Hugging Face `evaluate`:

```python
import group_by_vibes as gbv

# 1. Metric (reference-based): compare generated SQL against golden SQL
sql_bleu = gbv.load("sql_bleu", dialect="postgres")
sql_bleu.add(
    prediction="select name, age from users where status = 'active' and role = 'admin'",
    reference=["SELECT name, age FROM users WHERE role = 'admin' AND status = 'active'"],
)
sql_bleu.compute()["bleu"]
# 1.0

# 2. Measurement (reference-free): screen unlabelled SQL without executing it
valid_sql = gbv.load("valid_sql_rate", dialect="sqlite")
valid_sql.add(
    data="SELECT name, age FROM users WHERE active = 1",
    schema={"users": ["id", "name", "age", "active"]},
)
valid_sql.compute()["valid_sql_rate"]
# 1.0
```

---

## Available Evaluators

| Name | Type | Class | Stateless Helper | What It Evaluates |
| :--- | :--- | :--- | :--- | :--- |
| `load("sql_bleu")` | `Metric` | `SqlBleu` | `compute_sql_bleu`, `sentence_sql_bleu` | AST-canonicalized n-gram overlap against reference SQL ([Chen et al., 2019](#references)); reports `bleu`, `precisions`, `translation_length`, `parse_success_rate`, and `reference_parse_success_rate`. |
| `load("component_matching")` | `Metric` | `ComponentMatching` | `compute_component_matching` | Clause-level exact-set precision, recall, and F1 across 12 SQL components and 5 Spider hardness buckets ([Yu et al., 2018](#references), dialect-general `sqlglot` AST adaptation); defaults to Spider value-blind mode with opt-in strict flags and optional `schema`/`foreign_keys`. |
| `load("exact_matching")` | `Metric` | `ExactMatching` | `compute_exact_matching` | Order-insensitive structural exact-match accuracy across all 12 SQL components plus canonical `FROM` table/subquery multisets ([Yu et al., 2018](#references), dialect-general `sqlglot` AST adaptation); supports Spider value-blind defaults, opt-in strict flags, and optional `schema`/`foreign_keys`. |
| `load("valid_sql_rate")` | `Measurement` | `ValidSqlRate` | `compute_valid_sql_rate` | Lenient dialect-aware SQL syntax validation (rejecting empty inputs, multi-statement strings, and `exp.Command` fallbacks) plus optional per-query schema grounding (tables & columns) with 8-key `failure_counts`. |
| `load("wildcard_detector")` | `Measurement` | `WildcardDetector` | `compute_wildcard_detector` | Projection wildcard rate (`SELECT *`, `table.*`, `* EXCEPT`/`EXCLUDE` -> `1`), ignoring `COUNT(*)`, `EXISTS (SELECT * ...)`, `RETURNING *`, and `*` arithmetic (`0`). |

* **Dialect support:** Documentation and examples prioritize **PostgreSQL (`"postgres"`)** and **SQLite (`"sqlite"`)**, while all evaluators accept any dialect supported by `sqlglot.Dialect.get_or_raise` (plus `"postgresql"`, `"sqlite3"`, and generic ANSI `""`).
* **Scope:** Evaluators execute no queries against a live database engine; they complement—rather than replace—execution accuracy (`EX`).

---

## Usage Patterns

### Batch Streaming & Schema Grounding

```python
from group_by_vibes import load

# Quoted identifiers and per-query schemas
valid_sql = load("valid_sql_rate", dialect="sqlite")
valid_sql.add(
    data='SELECT "Singer(s)" FROM "1-29135051-2"',
    schema={"1-29135051-2": ["Singer(s)", "Year"]},
)
print(valid_sql.compute())

# Batch screening with per-example scores
wildcards = load("wildcard_detector", dialect="postgres", include_per_example=True)
wildcards.add_batch(["SELECT u.* FROM users u", "SELECT COUNT(*) FROM users"])
print(wildcards.compute())
```

### Stateless Functions & Raw LLM Extraction

Use `extract_fenced_or_last_sql` (or pass a custom regex/callable via `extractor=...`) to strip `<think>` reasoning traces and Markdown code fences before evaluation:

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

### Component & Exact Matching (Spider Defaults vs. Strict Mode)

```python
from group_by_vibes import compute_component_matching, compute_exact_matching

# Default Spider value-blind mode with optional per-query schema and foreign keys
preds = ["SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id WHERE 'A' = u.status"]
refs = ["SELECT ord.user_id FROM users usr JOIN orders ord ON usr.id = ord.user_id WHERE usr.status = 'Z'"]
schemas = [{"users": ["id", "status"], "orders": ["user_id"]}]
fks = [[("users.id", "orders.user_id")]]

print(compute_exact_matching(preds, refs, schemas=schemas, foreign_keys=fks, dialect="postgres")["exact_match"])
# 1.0

# Opt-in strict mode enforces literal values, DISTINCT, LIMIT values, and JOIN ON predicates
strict = compute_component_matching(
    preds,
    refs,
    schemas=schemas,
    foreign_keys=fks,
    dialect="postgres",
    ignore_values=False,
    ignore_distinct=False,
    include_limit_value=True,
    include_join_conditions=True,
)
print(strict["components"]["where"]["f1"])
# 0.0
```

### Thread Safety (Out-of-Band Runtime Screening)

Stateful `BaseEvaluator` instances (`SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `ComponentMatching`, `ExactMatching`) maintain single-writer accumulators without locks. When screening queries across background worker threads, either instantiate one evaluator per worker, funnel updates through a single aggregator thread, or call the stateless `compute_*` functions directly.

---

## Reference Benchmark

See [`examples/README.md`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/README.md) and [`examples/bird_eval.py`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/bird_eval.py) for an isolated CLI evaluating `birdsql/bird_mini_dev` (SQLite, PostgreSQL, and MySQL) across local 3B models (`qwen-3b`, `xiyansql-3b`, `finer-sql-3b`), plus offline core-only scoring (`score`) and comparative results.

---

## Contributing

Contributions are welcome—please open an issue before submitting a pull request. See [`CONTRIBUTING.md`](https://github.com/felipepenha/group-by-vibes/blob/main/CONTRIBUTING.md) for our issue-first and fork-and-branch workflow, and [`docs/designs/group_by_vibes.md`](https://github.com/felipepenha/group-by-vibes/blob/main/docs/designs/group_by_vibes.md) for the package architecture.

```bash
uv sync --group dev
uv run pytest -v --cov=group_by_vibes --cov-branch
uv run ruff check src/ tests/ examples/ .agents/skills/
uv run mypy src/group_by_vibes/
```

---

## References

* **SQL-BLEU:** Fuxiang Chen, Seung-won Hwang, Jaegul Choo, Jung-Woo Ha, and Sunghun Kim (2019). [*NL2pSQL: Generating Pseudo-SQL Queries from Under-Specified Natural Language Questions*](https://aclanthology.org/D19-1262/). EMNLP-IJCNLP 2019, pages 2603–2613. DOI: `10.18653/v1/D19-1262`.
* **Component Matching & Exact Matching:** Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir R. Radev (2018). [*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*](https://arxiv.org/abs/1809.08887). EMNLP 2018, pages 3911–3921. Implemented as a dialect-general, bug-fixed `sqlglot` AST adaptation of [`taoyds/spider`](https://github.com/taoyds/spider) without `nltk` or flat dict AST limitations ([design doc](https://github.com/felipepenha/group-by-vibes/blob/main/docs/designs/component_and_exact_matching.md)).

---

## License

Licensed under the [Apache-2.0](https://github.com/felipepenha/group-by-vibes/blob/main/LICENSE) license.
