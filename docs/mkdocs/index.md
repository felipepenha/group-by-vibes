<p align="center">
  <img class="gbv-hero-banner" src="assets/banner-github-transparent.png" alt="group-by-vibes — Text2SQL Evals to Deal With AI Vibes">
</p>

# group-by-vibes

**`group-by-vibes`** is a standalone, deterministic Python package for **Text2SQL (NL2SQL)** evaluation powered solely by [`sqlglot`](https://github.com/tobymao/sqlglot). It mirrors the [Hugging Face `evaluate`](https://github.com/huggingface/evaluate) interface (`load`, `add`, `add_batch`, `compute`) with explicit **`Metric`** (reference-based) and **`Measurement`** (reference-free) evaluators—without pulling `evaluate`, `datasets`, `pyarrow`, `transformers`, `torch`, `nltk`, or `sqlparse`.

!!! tip "Evaluation Scope"
    All evaluators operate statically on SQL Abstract Syntax Trees (ASTs) with **zero live database execution**. They complement—rather than replace—Execution Accuracy (`EX`) during offline benchmarks and out-of-band runtime screening.

## Why `group-by-vibes`?

* **Familiar `evaluate` ergonomics:** Supports stateful streaming (`load()` -> `add()` / `add_batch()` -> `compute()` with automatic `reset()`) alongside stateless `compute_*` helpers.
* **SQL-aware, not string-based:** Canonicalizes, validates, and decomposes parsed `sqlglot` ASTs across SQL dialects rather than relying on brittle regex or whitespace splits.
* **Zero ML or NLP runtime bloat:** Core installation requires only `sqlglot>=30.0.0,<31.0.0` with zero runtime downloads.
* **Built for modern Text2SQL workloads:** Handles CTEs (`WITH`), window functions, `CASE WHEN`, `CAST(...)`, quoted identifiers, nested subqueries, and messy LLM reasoning completions.

## Installation

Requires **Python `>=3.10`**. Core installation pulls only `sqlglot>=30.0.0,<31.0.0`:

```bash
# With uv
uv add group-by-vibes

# Or with pip
pip install group-by-vibes
```

## Quickstart

Load reference-based **Metrics** and reference-free **Measurements** via `group_by_vibes.load()`:

```python
import group_by_vibes as gbv

# 1. Metric (reference-based): compare predicted SQL against golden SQL
sql_bleu = gbv.load("sql_bleu", dialect="postgres")
sql_bleu.add(
    prediction="select name, age from users where status = 'active' and role = 'admin'",
    reference=["SELECT name, age FROM users WHERE role = 'admin' AND status = 'active'"],
)
assert sql_bleu.compute()["bleu"] == 1.0

# 2. Metric (structural clause F1 / exact match): Yu et al. (2018) adapted to sqlglot
exact = gbv.load("exact_matching", dialect="postgres")
exact.add(
    prediction="SELECT u.id FROM users u WHERE 'A' = u.status",
    reference="SELECT users.id FROM users WHERE users.status = 'Z'",
    schema={"users": ["id", "status"]},
)
assert exact.compute()["exact_match"] == 1.0

# 3. Measurement (reference-free): screen unlabelled SQL without executing it
valid_sql = gbv.load("valid_sql_rate", dialect="sqlite")
valid_sql.add(
    data="SELECT name, age FROM users WHERE active = 1",
    schema={"users": ["id", "name", "age", "active"]},
)
assert valid_sql.compute()["valid_sql_rate"] == 1.0
```

## Evaluator Catalog

### Metrics (Reference-Based)

Compare generated SQL against one or more golden SQL references (`prediction` vs. `reference`):

| Name | `load(...)` ID | Class & Stateless Helpers | Summary |
| :--- | :--- | :--- | :--- |
| **[SQL-BLEU](metrics/sql-bleu.md)** | `"sql_bleu"` | `SqlBleu`, `compute_sql_bleu`, `sentence_sql_bleu` | AST-canonicalized n-gram overlap ([Chen et al., 2019](metrics/sql-bleu.md)) with commutative sorting (`=`, `!=`, `IN`, `AND`/`OR`). |
| **[Component Matching](metrics/component-matching.md)** | `"component_matching"` | `ComponentMatching`, `compute_component_matching` | Clause-level exact-set Precision, Recall, and F1 across 12 SQL components and 5 hardness buckets ([Yu et al., 2018](metrics/component-matching.md) adapted to `sqlglot`). |
| **[Exact Matching](metrics/exact-matching.md)** | `"exact_matching"` | `ExactMatching`, `compute_exact_matching` | Order-insensitive structural exact-match accuracy across all 12 components plus canonical `FROM` table/subquery multisets ([Yu et al., 2018](metrics/exact-matching.md) adapted to `sqlglot`). |

### Measurements (Reference-Free)

Score or screen generated SQL without requiring any reference query (`data` with optional `schema`):

| Name | `load(...)` ID | Class & Stateless Helper | Summary |
| :--- | :--- | :--- | :--- |
| **[Valid SQL Rate](measurements/valid-sql-rate.md)** | `"valid_sql_rate"` | `ValidSqlRate`, `compute_valid_sql_rate` | Lenient dialect-aware AST syntax validation plus optional per-query static schema grounding with an 8-key `failure_counts` taxonomy. |
| **[Wildcard Detector](measurements/wildcard-detector.md)** | `"wildcard_detector"` | `WildcardDetector`, `compute_wildcard_detector` | AST projection wildcard rate (`SELECT *`, `table.*`, `* EXCEPT`/`EXCLUDE`), ignoring `COUNT(*)`, `EXISTS (SELECT * ...)`, `RETURNING *`, and `*` arithmetic. |

## Core Lifecycle & Shared Utilities

### Streaming Lifecycle & Reset Semantics

All five evaluators inherit from `BaseEvaluator`:

1. Call `add(...)` repeatedly for streaming examples or `add_batch(...)` for vectorized batches (all sequence and metadata inputs are eagerly validated before mutating state).
2. Call `compute()` to finalize a JSON-serializable `dict`. Calling `compute()` with positional arguments raises `TypeError` on `ComponentMatching`, `ExactMatching`, `ValidSqlRate`, and `WildcardDetector` (`SqlBleu.compute(smooth=False)` accepts an optional `smooth` argument), and calling `compute()` before adding at least one example raises `ValueError` across all five evaluators.
3. `compute()` automatically calls `reset()` after computing the final payload so an evaluator instance can be reused cleanly across evaluation splits.

### Stateless Helpers & Thread Safety

`BaseEvaluator` instances (`SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `ComponentMatching`, `ExactMatching`) maintain single-writer accumulators without internal locks. For concurrent execution (e.g., `ThreadPoolExecutor` workers screening live agent queries):

* Instantiate one evaluator per worker thread,
* Funnel updates through a single aggregator thread, or
* Call the stateless `compute_*` functions (`compute_sql_bleu`, `compute_component_matching`, `compute_exact_matching`, `compute_valid_sql_rate`, `compute_wildcard_detector`), which instantiate fresh local state per call.

### Raw LLM Extraction (`extract_fenced_or_last_sql`)

Pass `extractor=extract_fenced_or_last_sql` (or any regex `str`, compiled `re.Pattern` with at most one capture group, or `Callable[[str], str | None]`) to evaluate raw completions containing `<think>...</think>` reasoning blocks and Markdown fences:

```python
import group_by_vibes as gbv

raw = "<think>Check table schema first.</think>\n```sql\nSELECT id FROM users;\n```"
extracted = gbv.extract_fenced_or_last_sql(raw)
assert extracted == "SELECT id FROM users;"

res = gbv.compute_valid_sql_rate(
    [raw],
    [{"users": ["id"]}],
    dialect="postgres",
    extractor=gbv.extract_fenced_or_last_sql,
)
assert res["valid_sql_rate"] == 1.0
```

Extraction order in `extract_fenced_or_last_sql`:

1. Strip closed `<think>...</think>` blocks; return `None` if an unclosed/stray `<think` or `</think` tag remains (capturing truncated reasoning completions).
2. Prefer the last ```` ```sql ```` fenced block; otherwise fall back to the last generic ```` ``` ```` block containing a SQL keyword (`SELECT`, `WITH`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`).
3. Otherwise match a standalone `WITH` / `SELECT` statement up to `;` or a blank line.

### Dialect Normalization & Bounds

* **Dialects:** Documentation focuses on **`"postgres"`** and **`"sqlite"`**, while `normalize_dialect` accepts any dialect supported by `sqlglot.Dialect.get_or_raise` (plus aliases `"postgresql"` -> `"postgres"`, `"sqlite3"` -> `"sqlite"`, and generic ANSI `""`).
* **Bounds:** All five evaluators validate `max_query_chars: int | None = 65_536` and `max_input_chars: int | None = 131_072`; `canonicalize_sql`, `try_canonicalize_sql`, `sql_tokenizer`, and `sentence_sql_bleu` accept `max_query_chars` (`sentence_sql_bleu` also accepts `max_input_chars`); dialect helpers (`normalize_dialect`, `split_qualified_table`) accept neither. Non-positive integers and booleans raise `ValueError`, preventing pathological parser inputs or superlinear regex hangs.
* **Additional Public Helpers:** `canonicalize_sql`, `try_canonicalize_sql`, `sql_tokenizer`, `normalize_dialect`, `split_qualified_table`, `ForeignKeyPair`, and `ForeignKeysLike`.
