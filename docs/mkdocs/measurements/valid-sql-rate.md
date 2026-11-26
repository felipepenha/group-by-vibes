# Valid SQL Rate

**`ValidSqlRate`** (`load("valid_sql_rate")`, `compute_valid_sql_rate`) validates SQL predictions without executing them or comparing against golden references. It combines:

1. **Optional SQL extraction** from messy LLM outputs (`extractor=...`),
2. **Lenient dialect-aware AST parsing** (`sqlglot.parse` rejecting empty queries, multiple statements, and `exp.Command` fallbacks), and
3. **Optional static schema grounding** against per-query `{table: columns}` catalogs using `sqlglot.optimizer.scope.build_scope`.

---

## Validation Pipeline

### 1. Syntax & Structural AST Gate (`parser.py`)

A query passes syntax validation only if it satisfies all checks in `parse_sql` ([source](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/parser.py)):

* Passes optional `extractor` (`extraction_failed` / `extraction_error` otherwise) and stays within `max_input_chars` / `max_query_chars`.
* Non-empty after whitespace stripping (`empty_input`).
* Parses cleanly with `sqlglot.parse(..., read=dialect)` without `SqlglotError` or `RecursionError` (`syntax_error`).
* Contains **exactly one** statement (`multi_statement` if >1 non-empty statements).
* Contains no `exp.Command` nodes—catching `sqlglot`'s unparsed command fallback (`command_fallback`).

### 2. Static Schema Grounding (`grounding.py`)

When a non-`None` per-row `schema: Mapping[str, Iterable[str]]` is provided, `check_schema_grounding` ([source](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/grounding.py)) walks query scopes via `sqlglot.optimizer.scope.build_scope` **without invoking `sqlglot.optimizer.qualify` AST rewrites**:

* **Supported table & column scopes:** Resolves physical table aliases, CTEs (including explicit column alias lists like `WITH c(a, b) AS (...)` and inferred `SELECT` projections), derived tables, correlated outer scopes, 1–3 part qualified table names (`table`, `schema.table`, `catalog.schema.table`), and qualified stars (`alias.*`).
* **CTE `SELECT *` & opaque subquery handling:** Treats CTEs with unresolvable expressions or `*` and opaque derived tables carefully so valid downstream column references are not falsely flagged.
* **`SELECT` output aliases in clauses:** Recognizes projection aliases (`SELECT COUNT(*) AS total ... ORDER BY total` / `GROUP BY total` / `HAVING total > 1`) without misreporting `unknown_column`.
* **Dialect identifier casing rules:** Applies `sqlglot`'s dialect `NormalizationStrategy` consistently—folding unquoted identifiers appropriately for SQLite (always case-insensitive lookup), PostgreSQL/BigQuery (`LOWERCASE` / `CASE_INSENSITIVE_EXCEPT_TABLES`), Snowflake (`UPPERCASE`), and preserving quoted identifiers (`"MixedCase"`, `` `My Col` ``).
* **SQLite implicit columns:** Recognizes SQLite's `rowid`, `_rowid_`, and `oid` on physical tables in `"sqlite"`.
* **Non-query statements (`CREATE`, `DROP`, `INSERT`, `UPDATE`, `DELETE`):** When `build_scope(ast)` returns `None`, syntax passes and schema grounding is recorded as `skipped` (`schema_valid = None` per example, incrementing `schema_skipped_count`) instead of falsely rejecting valid DDL/DML syntax.

!!! note "Failure Precedence"
    When both missing tables and missing columns occur in the same query, `failure_reason` reports `"unknown_table"` first, while **both** `missing_tables` and `missing_columns` lists are still populated in `per_example` diagnostics.

---

## 8-Key `failure_counts` Taxonomy

Every failed query increments **at most one** primary bucket in `failure_counts`:

| Key | Trigger Stage | Meaning |
| :--- | :--- | :--- |
| `extraction_failed` | Extractor | `apply_extractor` returned `None`/empty SQL, exceeded `max_input_chars`, or failed regex match. |
| `extraction_error` | Extractor | Non-`str` input when `extractor` expected text, callable raised an exception, or returned a non-`str` value. |
| `empty_input` | Parser | Extracted/raw SQL was empty, whitespace-only, or comment/semicolon-only (`stmts == []`). |
| `multi_statement` | Parser | Input contained 2 or more SQL statements (`SELECT 1; SELECT 2`). |
| `command_fallback` | Parser | `sqlglot` fell back to an unparsed `exp.Command` AST node. |
| `syntax_error` | Parser / Grounding | `sqlglot.parse` or `build_scope` raised `SqlglotError`/`RecursionError`, or `len(sql) > max_query_chars`. |
| `unknown_table` | Schema Grounding | Query referenced a table name or star qualifier not found in `schema` (or visible CTE/derived scopes). |
| `unknown_column` | Schema Grounding | Query referenced a column name not found on its resolved physical table or visible CTE columns. |

---

## API Reference

### Class & Stateless Signatures

* **`gbv.load("valid_sql_rate", **kwargs) -> ValidSqlRate`** / **`ValidSqlRate(...)`**
* **`add(data: str, schema: SchemaDict | None = None) -> None`**
* **`add_batch(data: Sequence[str], schemas: Sequence[SchemaDict | None] | None = None) -> None`**
* **`compute_valid_sql_rate(data, schemas=None, *, ...)`**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"postgres"` | Target `sqlglot` dialect (`"postgres"`, `"sqlite"`, etc.). |
| `extractor` | `ExtractorLike` | `None` | Optional extractor applied to each `data` item (`extract_fenced_or_last_sql`, regex, callable). |
| `max_query_chars` | `int \| None` | `65_536` | Maximum character length per SQL query before rejecting as `syntax_error`. |
| `max_input_chars` | `int \| None` | `131_072` | Maximum raw input length when `extractor` is active. |
| `include_per_example` | `bool` | `False` | Include detailed per-row diagnostics in `per_example`. |

### `compute()` Output Schema

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `valid_sql_rate` | `float` | `valid_sql_count / num_examples` (syntax valid **and**, if schema evaluated, grounded cleanly). |
| `syntax_valid_rate` | `float` | `syntax_valid_count / num_examples` (independent of schema grounding). |
| `schema_grounded_rate` | `float \| None` | `schema_grounded_count / schema_evaluated_count`, or `None` if `schema_evaluated_count == 0`. |
| `schema_grounding_status` | `str` | `"skipped"` (`schema_evaluated_count == 0`), `"applied"` (all rows evaluated), or `"partial"`. |
| `schema_evaluated_count` | `int` | Rows where non-`None` schema grounding actually ran (excludes unparseable & DDL/DML skipped rows). |
| `schema_grounded_count` | `int` | Evaluated rows passing table and column grounding with zero missing items. |
| `schema_skipped_count` | `int` | `num_examples - schema_evaluated_count`. |
| `syntax_valid_count` | `int` | Rows passing parsing and AST gates. |
| `valid_sql_count` | `int` | Rows passing both syntax checks and any applicable schema grounding. |
| `num_examples` | `int` | Total rows processed. |
| `dialect` | `str` | Normalized `sqlglot` dialect name. |
| `failure_counts` | `dict[str, int]` | Exact tally across all 8 failure keys above. |
| `per_example` | `list[dict[str, Any]]` | *(Only when `include_per_example=True`)* Dict per row with `syntax_valid`, `schema_valid`, `failure_reason`, `missing_tables`, and `missing_columns` (raw SQL omitted). |

---

## Examples

### Quoted Identifiers, CTEs, and Missing-Column Diagnostics

```python
import group_by_vibes as gbv

measurement = gbv.load("valid_sql_rate", dialect="sqlite", include_per_example=True)
measurement.add(
    data='WITH active AS (SELECT "Singer(s)" FROM "1-29135051-2") SELECT "Singer(s)" FROM active',
    schema={"1-29135051-2": ["Singer(s)", "Year"]},
)
measurement.add(
    data="SELECT missing_col FROM users",
    schema={"users": ["id", "name"]},
)

res = measurement.compute()
assert res["syntax_valid_rate"] == 1.0
assert res["valid_sql_rate"] == 0.5
assert res["schema_grounding_status"] == "applied"
assert res["failure_counts"]["unknown_column"] == 1
assert res["per_example"][1]["missing_columns"] == ["missing_col"]
```
