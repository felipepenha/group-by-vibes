# Measurements (Reference-Free)

Reference-free **Measurements** inspect generated SQL strings directly—without needing golden reference queries—making them practical for both unlabelled offline audits and out-of-band runtime guardrails around live SQL agents.

## Available Measurements

| Measurement | `load(...)` ID | Purpose | Schema Required | Key Output Fields |
| :--- | :--- | :--- | :--- | :--- |
| **[Valid SQL Rate](valid-sql-rate.md)** | `"valid_sql_rate"` | Checks SQL extractability, strict single-statement AST parsing (`sqlglot.parse`), and optional per-query static table/column grounding. | Optional (`{table: [columns]}`) | `valid_sql_rate`, `syntax_valid_rate`, `schema_grounded_rate`, `failure_counts` (8 keys) |
| **[Wildcard Detector](wildcard-detector.md)** | `"wildcard_detector"` | Detects projection wildcards (`SELECT *`, `table.*`, `* EXCEPT`/`EXCLUDE`) while ignoring `COUNT(*)`, `EXISTS`, `RETURNING *`, and `*` math. | No | `wildcard_rate`, `wildcard_count`, `parsed_count`, `parse_failed_count` |

## Offline Screening Pattern

Both measurements accept streaming `add()` / `add_batch()` updates as well as raw LLM completions via `extractor=extract_fenced_or_last_sql`:

```python
import group_by_vibes as gbv

completions = [
    "```sql\nSELECT u.name, COUNT(*) FROM users u GROUP BY u.name\n```",
    "```sql\nSELECT u.* FROM users u\n```",
]
schemas = [
    {"users": ["id", "name"]},
    {"users": ["id", "name"]},
]

validity = gbv.compute_valid_sql_rate(
    completions,
    schemas,
    dialect="postgres",
    extractor=gbv.extract_fenced_or_last_sql,
)
wildcards = gbv.compute_wildcard_detector(
    completions,
    dialect="postgres",
    extractor=gbv.extract_fenced_or_last_sql,
)

assert validity["valid_sql_rate"] == 1.0
assert wildcards["wildcard_rate"] == 0.5
```

## Runtime Concurrency Note

Because stateful `ValidSqlRate` and `WildcardDetector` instances keep lock-free single-writer counters, multi-threaded screening services should either keep one instance per worker thread or call `compute_valid_sql_rate` / `compute_wildcard_detector` directly.
