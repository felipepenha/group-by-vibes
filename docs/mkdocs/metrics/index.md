# Metrics (Reference-Based)

Reference-based **Metrics** compare a predicted SQL query against one or more golden reference SQL queries (`prediction` vs. `reference`). Unlike live database execution (`EX`), every Metric in `group-by-vibes` runs deterministically in memory using `sqlglot` ASTs.

## Available Metrics

| Metric | Primary Signal | Multi-Ref | Optional Schema & FKs | Value Blind by Default | Unparseable Prediction Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[SQL-BLEU](sql-bleu.md)** | Canonicalized AST n-gram overlap (0.0–1.0) | Yes (`Counter` max union) | No | No (literals retained as AST tokens) | Excluded from `scored_count`; tracked via `parse_success_rate` |
| **[Component Matching](component-matching.md)** | Clause-level exact-set Precision / Recall / F1 across 12 SQL components & 5 hardness buckets | Yes (best matching reference) | Yes (`schema`, `foreign_keys`) | Yes (`ignore_values=True`; opt-in strict flags) | Scored as empty SQL miss by default (`skip_unparseable_predictions=False`) |
| **[Exact Matching](exact-matching.md)** | Order-insensitive whole-query structural exact match across 12 components + `FROM` tables/subqueries | Yes (best matching reference) | Yes (`schema`, `foreign_keys`) | Yes (`ignore_values=True`; opt-in strict flags) | Scored as `0` by default (`skip_unparseable_predictions=False`) |

## Multi-Reference Evaluation

Every Metric accepts either a single SQL string or a non-empty sequence of equivalent golden queries per example:

```python
import group_by_vibes as gbv

refs = [
    "SELECT u.name FROM users u WHERE u.active = 1",
    "SELECT name FROM users WHERE 1 = active",
]

em = gbv.compute_exact_matching(
    ["SELECT name FROM users WHERE active = 999"],
    [refs],
    schemas=[{"users": ["name", "active"]}],
    dialect="sqlite",
)
assert em["exact_match"] == 1.0
```

* **`SqlBleu`** merges clipped reference n-gram counts across valid references (standard multi-reference BLEU) and picks the closest reference token length.
* **`ComponentMatching` and `ExactMatching`** evaluate the prediction against every valid reference in the group and select the deterministically best-scoring reference (first maximizing `exact_match`, then matched active components, then total F1 across all 12 components, then earliest index).

## Shared Yu et al. (2018) Engine

`ComponentMatching` and `ExactMatching` are standalone evaluators sharing one dialect-general, bug-fixed `sqlglot` AST engine. Beyond reading the individual metric pages, see **[Adapting Spider to `sqlglot`](spider-to-sqlglot.md)** for the full architectural walkthrough, why `taoyds/spider/process_sql.py` and flat-dict lowering were rejected, and a complete line-by-line summary of the five upstream `taoyds/spider/evaluation.py` defects fixed here.
