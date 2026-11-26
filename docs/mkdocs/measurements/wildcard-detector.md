# Wildcard Detector

**`WildcardDetector`** (`load("wildcard_detector")`, `compute_wildcard_detector`) flags generated SQL queries that leak broad **`SELECT *`** or **`table.*`** projections—a common failure mode in enterprise Text2SQL agents where LLMs fetch wide sensitive or expensive tables instead of naming target columns.

---

## AST Detection Rules

Unlike naive string checks (`"*" in sql`), `WildcardDetector` inspects `sqlglot` AST `exp.Select` nodes ([source](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/wildcard_detector.py)) so structural wildcards are separated cleanly from safe SQL constructs:

| SQL Pattern | Flagged (`1` / `0` / `None`) | Rationale |
| :--- | :---: | :--- |
| `SELECT * FROM users` | `1` | Direct projection `exp.Star`. |
| `SELECT u.* FROM users u` | `1` | Table-qualified projection `exp.Column(this=exp.Star)`. |
| `SELECT * EXCEPT (secret) FROM users` | `1` | BigQuery/DuckDB/Snowflake `* EXCEPT`/`EXCLUDE` parses with an `exp.Star` projection target. |
| `WITH c AS (SELECT * FROM users) SELECT id FROM c` | `1` | Projection wildcard inside a CTE or derived subquery projection (`ast.find_all(exp.Select)`). |
| `SELECT COUNT(*) FROM users` | `0` | Aggregate argument `exp.Count(this=exp.Star)` is not a `SELECT` projection expression. |
| `SELECT id FROM users u WHERE EXISTS (SELECT * FROM orders o WHERE o.user_id = u.id)` | `0` | Explicit AST exception: `_is_direct_exists_target(sel)` skips `SELECT` clauses inside `EXISTS (...)`. |
| `DELETE FROM users WHERE id = 1 RETURNING *` | `0` | DML `RETURNING *` parses under `exp.Returning`, not `exp.Select`. |
| `SELECT price * qty AS total FROM items` | `0` | Multiplication operator (`exp.Mul`), not `exp.Star`. |
| Syntax error / `multi_statement` / failed extraction | `None` | Excluded from `wildcard_rate` denominator (`parsed_count`), counted in `parse_failed_count`. |

### Rate Denominator Semantics

`wildcard_rate` equals `wildcard_count / parsed_count`. If `parsed_count == 0` (all inputs failed extraction or parsing), `wildcard_rate` returns `None` instead of dividing by zero or returning a misleading `0.0`.

---

## API Reference

### Class & Stateless Signatures

* **`gbv.load("wildcard_detector", **kwargs) -> WildcardDetector`** / **`WildcardDetector(...)`**
* **`add(data: str) -> None`** / **`add_batch(data: Sequence[str]) -> None`**
* **`compute_wildcard_detector(data, *, ...)`**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"postgres"` | Target `sqlglot` dialect (`"postgres"`, `"sqlite"`, etc.). |
| `extractor` | `ExtractorLike` | `None` | Optional SQL extractor (`extract_fenced_or_last_sql`, regex, or callable). |
| `max_query_chars` | `int \| None` | `65_536` | Maximum character length per SQL query before flagging parse failure. |
| `max_input_chars` | `int \| None` | `131_072` | Maximum raw input length when `extractor` is active. |
| `include_per_example` | `bool` | `False` | Include `per_example: list[int \| None]` (`1`, `0`, or `None`) in `compute()` output. |

### `compute()` Output Schema

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `wildcard_rate` | `float \| None` | `wildcard_count / parsed_count`, or `None` when `parsed_count == 0`. |
| `wildcard_count` | `int` | Count of parsed queries containing at least one non-`EXISTS` projection wildcard. |
| `parsed_count` | `int` | Count of inputs passing extraction and single-statement AST parsing. |
| `parse_failed_count` | `int` | Count of inputs failing extraction, char bounds, or AST parsing. |
| `extraction_failed_count` | `int` | Subset of `parse_failed_count` caused specifically by `extractor` returning `None`/error. |
| `num_examples` | `int` | Total rows processed. |
| `dialect` | `str` | Normalized `sqlglot` dialect name. |
| `per_example` | `list[int \| None]` | *(Only when `include_per_example=True`)* `1` (wildcard found), `0` (clean), or `None` (unparseable). |

---

## Example

```python
import group_by_vibes as gbv

queries = [
    "SELECT u.* FROM users u",
    "SELECT COUNT(*) FROM users",
    "SELECT id FROM users u WHERE EXISTS (SELECT * FROM orders o WHERE o.user_id = u.id)",
    "SELECT * FROM (",
]

res = gbv.compute_wildcard_detector(
    queries,
    dialect="postgres",
    include_per_example=True,
)
assert res["wildcard_count"] == 1
assert res["parsed_count"] == 3
assert res["parse_failed_count"] == 1
assert res["wildcard_rate"] == 1 / 3
assert res["per_example"] == [1, 0, 0, None]
```
