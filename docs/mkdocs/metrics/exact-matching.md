# Exact Matching

**`ExactMatching`** (`load("exact_matching")`, `compute_exact_matching`) measures **order-insensitive structural exact-match accuracy** across complete SQL queries—overall and stratified across Yu et al. (2018)'s four Spider hardness buckets (`easy`, `medium`, `hard`, `extra`, plus `all`). It is built on the same **dialect-general, bug-fixed `sqlglot` AST engine** as [`ComponentMatching`](component-matching.md).

> **Reference:** Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir R. Radev (2018). [*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*](https://arxiv.org/abs/1809.08887). EMNLP 2018, pages 3911–3921. Reference implementation: [`taoyds/spider`](https://github.com/taoyds/spider) (`evaluation.py`).

!!! info "See Also: Shared `sqlglot` AST Engine Deep Dive"
    See **[Adapting Spider to `sqlglot`](spider-to-sqlglot.md)** for the end-to-end 6-pass canonicalization pipeline, measured failures of `taoyds/spider/process_sql.py` and flat-dict lowering on `bird_mini_dev`, two-layer identifier normalization, and transitive union-find `foreign_keys` (`kmap`).

---

## Exact Match Criterion

In `score_pair` ([source](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py#L1914-L1936)), a prediction scores `exact_match = 1` against a reference **if and only if all of the following hold**:

1. **Empty consistency:** `pred.is_empty == ref.is_empty` (unparseable empty fallback queries never falsely match real queries).
2. **All 12 components match (`cs.f1 == 1`):** Every key in `select`, `select(no AGG)`, `where`, `where(no OP)`, `group(no Having)`, `group`, `order`, `and/or`, `IUEN`, `keywords`, `cte`, and `join_on` has per-example `f1 == 1` (inactive/both-empty components emit `f1 = 1` per example).
3. **`FROM` physical & CTE tables match as a multiset:** `Counter(pred.from_tables) == Counter(ref.from_tables)` (preserving duplicate counts on self-joins like `FROM emp e1 JOIN emp e2` -> `("emp", "emp")`, and alpha-renamed CTE placeholders like `"__cte_0__"`).
4. **Derived `FROM` subqueries match recursively:** `len(pred.from_subqueries) == len(ref.from_subqueries)` and the multisets match order-insensitively under recursive `eval_exact_match == 1`.

Dataset accuracy is then `exact_match = exact_match_count / scored_count` (or `None` when `scored_count == 0`).

---

## How We Adapted Yu et al. (2018) to `sqlglot`: Choices & Bug Fixes

### Exact-Match Bug Fixes vs. `taoyds/spider/evaluation.py`

1. **Python 3 crash (`TypeError`) on `>= 2` `FROM` subqueries (`evaluation.py:387-388`):**
   * *Upstream bug:* `eval_exact_match` executed `sorted(label['from']['table_units'])`. When a query joins two or more derived subqueries, `table_units` contains `('sql', dict)` tuples; Python 3 raises `TypeError: '<' not supported between instances of 'dict' and 'dict'`.
   * *Our fix ([`score_pair`](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py#L1918-L1925)):* Normalizes derived tables deterministically via `canonical_signature` (`__subq_0__`, ...) and compares `from_subqueries` using order-insensitive recursive multiset matching (`_multiset_match_count`), never invoking `<` over Python dicts.
2. **Skipped `FROM` verification when gold has empty `table_units` (`evaluation.py:386`):**
   * *Upstream bug:* `if len(label['from']['table_units']) > 0:` bypassed checking `pred['from']['table_units']` whenever the reference query had no `FROM` clause (e.g., `SELECT 1`), allowing `SELECT 1 FROM users` to score `1`.
   * *Our fix:* Always checks `Counter(pred.from_tables) == Counter(ref.from_tables)` and `from_subqueries` equality unconditionally.
3. **Hardness aggregate inflation (`evaluation.py:333, 338`):**
   * *Upstream bug:* `count_others` miscounted `NOT` flags in `WHERE` and `'and'`/`'or'` strings in `HAVING` as aggregate functions, shifting queries into `medium`/`hard`/`extra` incorrectly.
   * *Our fix:* Counts genuine current-level `exp.AggFunc` AST nodes before computing Spider's 4-level hardness bucket, keeping `by_hardness` breakdowns accurate.
4. **In-place mutation corrupting repeated reference checks (`evaluation.py:123`):**
   * *Upstream bug:* `eval_sel` deleted elements in place from `label['select'][1]`.
   * *Our fix:* Immutable `NormalizedQuery` decompositions allow safe reuse across nested subqueries, CTEs, `IUEN` branches, and multi-reference groups.

### Architectural & Generalization Choices

* **Why standalone `ExactMatching` alongside `ComponentMatching`?** We rejected exposing a single `SpiderMatch` evaluator or an `include_components=True` overload on `ExactMatching` so `group-by-vibes` maintains strict Hugging Face `evaluate` symmetry (one evaluator concept per `load()` name, non-overlapping output schemas). The deliberate trade-off is parsing twice if a caller runs `ComponentMatching` and `ExactMatching` in separate calls.
* **Wave-by-wave CTE alpha-renaming (`WITH a AS ...` vs `WITH b AS ...`):** Instead of comparing CTE aliases verbatim or inlining CTEs (which alters query structure and hardness), `matching.py` partitions CTEs into Kahn topological waves, binds recursive self-references to `__self__`, preserves explicit column aliases (`WITH c(x, y)`), assigns canonical `__cte_0__`, `__cte_1__`, ... IDs sorted by `(body.canonical_signature, col_aliases)`, and verifies both CTE definitions (`cte`) and main-query references.
* **Nested subquery canonicalization without string dependence:** Nested `exp.Query` expressions in `WHERE`, `HAVING`, `SELECT`, or `JOIN ON` are recursively decomposed and replaced by deterministic hashes `__subq_<32-char sha256>__`, making `WHERE a IN (SELECT x, y FROM t) AND b = 1` match `WHERE b = 2 AND a IN (SELECT y, x FROM t)` structurally.
* **Spider value-blind defaults vs. strict LLM evaluation:** By default (`ignore_values=True`, `ignore_distinct=True`, `include_limit_value=False`, `include_join_conditions=False`), `ExactMatching` asks *"did the model build the right structural SQL skeleton?"*. For production regression suites where constants (`'active'` vs `'deleted'`), `DISTINCT`, `LIMIT 10`, or `JOIN ... ON` keys matter, enable the four strict flags—all five flags are echoed into every `compute()` output dictionary so runs cannot be silently compared across mismatched configurations.
* **Unparseable predictions (`skip_unparseable_predictions=False` vs. `True`):** Defaulting to `False` follows Spider by scoring unparseable predictions as `0` in `scored_count` (avoiding survivorship bias where a model parsing `1/100` queries scores `100%` exact match). Setting `True` lets you isolate structural equivalence strictly on syntactically valid queries while still reporting `parse_success_rate`.
* **Multi-reference hardness-anchoring trade-off:** When a row supplies multiple references (`[ref1, ref2]`), the winning reference (`(exact_match, matched_active_components, all_components_f1_sum, -ref_index)`) supplies both the score and the row's hardness bucket. On heterogeneous multi-reference datasets where two acceptable references have different hardness levels, `by_hardness[level]["count"]` can vary slightly across models; single-reference benchmarks (`Spider`, `bird_mini_dev`) are unaffected.
* **Known static limitation:** Like Spider itself, `ExactMatching` checks canonical AST structure rather than algebraic/runtime equivalence (`1.0 * a / b` vs. `CAST(a AS REAL) / b`, `IIF(c, x, y)` vs. `CASE WHEN c THEN x ELSE y END`, or `JOIN` vs. `IN (SELECT ...)`). Pair `ExactMatching` with Execution Accuracy (`EX`) when semantically rewritten plans are expected.

---

## API Reference

### Signatures

* **`gbv.load("exact_matching", **kwargs) -> ExactMatching`** / **`ExactMatching(...)`**
* **`add(prediction, reference, schema=None, foreign_keys=None) -> None`**
* **`add_batch(predictions, references, schemas=None, foreign_keys=None) -> None`**
* **`compute_exact_matching(predictions, references, schemas=None, foreign_keys=None, *, ...)`**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"postgres"` | Target `sqlglot` dialect (`"postgres"`, `"sqlite"`, `"mysql"`, `"snowflake"`, `"bigquery"`, etc.). |
| `ignore_values` | `bool` | `True` | Mask non-exempt literals to `__val__` and `COUNT(1)` -> `COUNT(*)`. Set `False` for strict literal matching. |
| `ignore_distinct` | `bool` | `True` | Strip `DISTINCT` and collapse `UNION`/`INTERSECT`/`EXCEPT ALL` distinctions. Set `False` for strict matching. |
| `include_limit_value` | `bool` | `False` | Require matching canonical `LIMIT` / `OFFSET` expressions rather than only presence. |
| `include_join_conditions` | `bool` | `False` | Require matching `JOIN ... ON` and `USING(...)` predicates across joins. |
| `skip_unparseable_predictions` | `bool` | `False` | Skip unparseable predictions (`None` in `per_example`) instead of counting them as `0`. |
| `extractor` | `ExtractorLike` | `None` | Optional extractor run on predictions only (`extract_fenced_or_last_sql`, regex, callable). |
| `max_query_chars` | `int \| None` | `65_536` | Character limit per SQL query before rejecting parse. |
| `max_input_chars` | `int \| None` | `131_072` | Character limit on raw prediction input passed to `extractor`. |
| `include_per_example` | `bool` | `False` | Include sanitized per-example dicts in `compute()["per_example"]`. |

### `compute()` Output Schema

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `exact_match` | `float \| None` | Overall accuracy `exact_match_count / scored_count`, or `None` when `scored_count == 0`. |
| `exact_match_count` | `int` | Count of scored examples with `exact_match == 1`. |
| `by_hardness` | `dict[str, dict[str, Any]]` | Keys `"easy"`, `"medium"`, `"hard"`, `"extra"`, `"all"`; each maps to `{"count": int, "exact_match": float \| None, "exact_match_count": int}`. |
| `scored_count` | `int` | Examples contributing to `exact_match`. |
| `parsed_count` | `int` | Predictions passing extraction, char limits, and query-AST parsing. |
| `unparseable_count` | `int` | Predictions failing extraction, char limits, parsing, or query-AST gate. |
| `parse_success_rate` | `float` | `parsed_count / num_examples`. |
| `reference_parse_success_rate` | `float` | Parsed references divided by total supplied reference strings. |
| `schema_applied_count` | `int` | Scored examples evaluated with a non-`None` `schema`. |
| `foreign_keys_applied_count` | `int` | Scored examples evaluated with non-empty `foreign_keys`. |
| `num_examples` | `int` | Total rows processed. |
| `dialect` | `str` | Normalized `sqlglot` dialect name. |
| `ignore_values` | `bool` | Recorded mode flag value. |
| `ignore_distinct` | `bool` | Recorded mode flag value. |
| `include_limit_value` | `bool` | Recorded mode flag value. |
| `include_join_conditions` | `bool` | Recorded mode flag value. |
| `skip_unparseable_predictions` | `bool` | Recorded mode flag value. |
| `per_example` | `list[dict[str, Any] \| None]` | *(Only when `include_per_example=True`)* `None` if skipped; else `{"exact_match": int, "hardness": str, "prediction_parsed": bool, "schema_applied": bool, "foreign_keys_applied": bool}` (raw SQL and literals omitted). |

---

## Examples

### Reordered CTEs, Multiple `FROM` Subqueries, and Hardness Buckets

```python
import group_by_vibes as gbv

pred_cte = (
    "WITH active_users(uid, name) AS (SELECT id, name FROM users WHERE active = 1) "
    "SELECT name, uid FROM active_users"
)
ref_cte = (
    "WITH renamed(uid, name) AS (SELECT u.id, u.name FROM users u WHERE 99 = u.active) "
    "SELECT uid, name FROM renamed"
)

pred_subq = (
    "SELECT a.x, b.y FROM (SELECT id AS x FROM t1) a "
    "JOIN (SELECT id AS y FROM t2) b ON a.x = b.y"
)
ref_subq = (
    "SELECT b.y, a.x FROM (SELECT id AS y FROM t2) b "
    "JOIN (SELECT id AS x FROM t1) a ON b.y = a.x"
)

res = gbv.compute_exact_matching(
    [pred_cte, pred_subq],
    [ref_cte, ref_subq],
    schemas=[{"users": ["id", "name", "active"]}, {"t1": ["id"], "t2": ["id"]}],
    dialect="postgres",
    include_per_example=True,
)
assert res["exact_match"] == 1.0
assert res["exact_match_count"] == 2
assert res["by_hardness"]["extra"]["count"] == 2
```
