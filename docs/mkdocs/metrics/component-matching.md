# Component Matching

**`ComponentMatching`** (`load("component_matching")`, `compute_component_matching`) evaluates Text2SQL predictions at the **SQL clause level** by reporting exact-set **Precision, Recall, and F1** across **12 SQL components** and **4 Spider hardness levels** (`easy`, `medium`, `hard`, `extra`, plus overall `all`). It is a **dialect-general, bug-fixed `sqlglot` AST adaptation** of Yu et al. (EMNLP 2018).

> **Reference:** Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir R. Radev (2018). [*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*](https://arxiv.org/abs/1809.08887). EMNLP 2018, pages 3911–3921. Reference implementation: [`taoyds/spider`](https://github.com/taoyds/spider) (`evaluation.py`, `process_sql.py`).

!!! info "See Also: Full AST Engine Deep Dive"
    `ComponentMatching` and [`ExactMatching`](exact-matching.md) share one `sqlglot` AST decomposition engine ([`matching.py`](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py)). For the 6-pass AST pipeline diagram, why Spider's `process_sql.py` + `nltk` and flat-dict lowering fail on modern SQL, two-layer identifier normalization, and `foreign_keys` union-find equivalence, read **[Adapting Spider to `sqlglot`](spider-to-sqlglot.md)**.

---

## Mathematical Definition & Why F1

### 1. Per-Example Exact-Set Rule (`0` or `1`)

For each SQL component \(c\), both the predicted query and golden reference are decomposed into **order-independent multisets** (or ordered tuples for `order`) of canonicalized sub-components. On example \(i\), component \(c\) scores **binary `1` or `0`**—there is no partial credit *within* a single clause:

$$
m_{i,c} =
\begin{cases}
1 & \text{if } \text{pred\_total}_{i,c} = \text{label\_total}_{i,c} \;\text{and}\; \text{overlap}_{i,c} = \text{pred\_total}_{i,c} \\
0 & \text{otherwise}
\end{cases}
$$

Per example, if \(m_{i,c} = 1\) (including empty-vs-empty \(\text{pred\_total}_{i,c} = \text{label\_total}_{i,c} = 0\)), the per-example `ComponentScore` sets `precision = recall = f1 = 1`; otherwise `precision = recall = f1 = 0`.

### 2. Corpus Precision, Recall, and F1

Across the dataset (and inside each hardness bucket), rows only enter a denominator when that side actually contains component \(c\):

$$
P_c = \frac{\sum_{i:\,\text{pred\_total}_{i,c} > 0} m_{i,c}}{\left|\{i : \text{pred\_total}_{i,c} > 0\}\right|}, \qquad
R_c = \frac{\sum_{i:\,\text{label\_total}_{i,c} > 0} m_{i,c}}{\left|\{i : \text{label\_total}_{i,c} > 0\}\right|}, \qquad
F_{1,c} = \frac{2 P_c R_c}{P_c + R_c}
$$

* Rows where **neither** prediction nor reference contains component \(c\) (`pred_total == 0` and `label_total == 0`) are **excluded from both `pred_count` and `label_count`**.
* If `pred_count == 0` and `label_count == 0` for a component in a bucket, `precision`, `recall`, and `f1` return `None`.
* If `pred_count > 0` or `label_count > 0` and `precision + recall == 0.0`, **`f1` returns `0.0`** (fixing an upstream Spider bug—see below).

### 3. Why F1 Instead of Plain Accuracy?

Most SQL clauses (`WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `IUEN`, `WITH` CTEs, `and/or`) are **optional and sparse** across a benchmark:

* **Plain accuracy is dominated by true negatives:** If `GROUP BY` appears in 40 of 500 queries, a trivial model that never emits `GROUP BY` agrees on all 460 empty cases and would get `92.0%` accuracy while having `0%` `GROUP BY` capability.
* **Precision alone rewards under-predicting:** Emitting `ORDER BY` on 1 trivial query and omitting it everywhere else achieves `Precision = 1.0`.
* **Recall alone hides hallucinated clauses:** Counting only `gold > 0` ignores every spurious `WHERE` or `GROUP BY` hallucinated on simple queries.
* **F1 balances omission and hallucination:** Excluding empty-vs-empty queries removes true-negative inflation, and the harmonic mean requires both high precision (clauses emitted match gold) and high recall (gold clauses are not dropped).
* *(Special case:)* Because every valid SQL query has a `SELECT` clause, when every prediction parses (`unparseable_count == 0`), `pred_count == label_count == N`, so for `select` and `select(no AGG)` specifically, \(P = R = F_1\) equals exact-set clause accuracy.

---

## The 12 Evaluated Components

Under Spider defaults (`ignore_values=True`, `ignore_distinct=True`, `include_limit_value=False`, `include_join_conditions=False`), components `1`–`10` align with `taoyds/spider/evaluation.py`, while `11` (`cte`) and `12` (`join_on`, opt-in) extend coverage to modern SQL:

| # | Component Key | What Is Extracted & Compared |
| :--- | :--- | :--- |
| 1 | `select` | Multiset of canonicalized `SELECT` projection expressions (`AS` output aliases stripped; includes outer aggregates; includes `DISTINCT` / `DISTINCT ON` when `ignore_distinct=False`). |
| 2 | `select(no AGG)` | Multiset of `SELECT` expressions with the outermost aggregate wrapper (`MAX`, `MIN`, `COUNT`, `SUM`, `AVG`) unwrapped; non-aggregate projections remain unchanged. |
| 3 | `where` | Multiset of flattened top-level `WHERE` predicate atoms (splitting `AND`/`OR` chains while keeping compound `NOT (a AND b)` intact; respects `ignore_values` and recursive `__subq_<sig>__` hashing). |
| 4 | `where(no OP)` | Multiset of target expressions extracted from each `WHERE` atom (unwrapping outer `NOT`/parens): non-literal LHS for binary comparisons (`=`, `<>`, `<`, `<=`, `>`, `>=`, `LIKE`, `ILIKE`, `IS`), `BETWEEN`, and `IN`; `"exists"` for `EXISTS (...)`; canonicalized expression otherwise. |
| 5 | `group(no Having)` | Multiset of `GROUP BY` expressions with leading table/schema/catalog qualifiers stripped on bare columns (matching Spider `eval_group`). |
| 6 | `group` | Presence unit (`1` if `GROUP BY` or `HAVING` is non-empty, else `0`); matches (`1`) iff `group_by_full` multiset (with normalized table qualifiers & FK collapse) AND `having_preds` multiset AND `having_and_or` connector set all match. |
| 7 | `order` | Presence unit (`1` if `ORDER BY` or `LIMIT`/`OFFSET`/`FETCH` is present, else `0`); matches (`1`) iff ordered `(expr, "asc"\|"desc", "first"\|"last"\|"default")` list (defaulting omitted direction to `"asc"`) matches AND `has_limit` matches (plus canonical `LIMIT`/`OFFSET` text when `include_limit_value=True`). |
| 8 | `and/or` | Set of boolean connectors (`"and"`, `"or"`) in `WHERE`. When both sets are empty (`frozenset()`), emits `pred_total = 0, label_total = 0` with per-example `f1 = 1` so corpus denominators are never inflated. |
| 9 | `IUEN` | Set-operation branches (`union`, `intersect`, `except`, or `*_all` when `ignore_distinct=False`) decomposed recursively via left-associative chain flattening; matched as an order-insensitive multiset via recursive exact match. |
| 10 | `keywords` | Set of structural SQL keywords at the **current query level only** (no recursion into subqueries/CTEs/IUEN): `where`, `group`, `having`, `order`, `asc`, `desc`, `limit`, `intersect`, `union`, `except`, `with`, plus `or`, `not` (`NOT`, `NOT IN`, `NOT LIKE`, `IS NOT`), `in`, `like` (`LIKE`/`ILIKE`) inside current-level `JOIN ON`/`WHERE`/`HAVING`. |
| 11 | `cte` | Multiset of `(column_aliases, body_nq)` pairs from `WITH` clauses after wave-by-wave topological alpha-renaming (`__cte_0__`, ...), matched recursively via exact match. |
| 12 | `join_on` | When `include_join_conditions=True`, multiset of canonicalized `JOIN ... ON` atoms and `USING(col)` tokens at the current query level. When `False` (default), inactive (`pred_count = label_count = 0`). |

---

## Spider Hardness Stratification (`by_hardness`)

Following Yu et al. (2018), every scored example is classified by its **reference query** into `easy`, `medium`, `hard`, or `extra` (plus the corpus summary `all`) using three counts at the current query level:

* **`count_comp1`:** presence of `WHERE` (`+1`), `GROUP BY`/`HAVING` (`+1`), `ORDER BY` (`+1`), `LIMIT` (`+1`), extra tables/subqueries `max(0, len(from_tables) + len(from_subqueries) - 1)`, plus count of current-level `OR` and `LIKE`/`ILIKE` nodes.
* **`count_comp2`:** `len(from_subqueries)` + direct nested query nodes in `WHERE`/`HAVING`/`SELECT`/`JOIN ON` + `len(iuen_branches)` + `len(ctes)`.
* **`count_others`:** `+1` if current-level `exp.AggFunc` count > 1, `+1` if `len(select_items) > 1`, `+1` if `len(where_preds) > 1`, `+1` if `len(group_by_full) > 1`.

| Bucket | Exact Threshold Rule |
| :--- | :--- |
| `easy` | `count_comp1 <= 1 and count_others == 0 and count_comp2 == 0` |
| `medium` | `(count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0)` or `(count_comp1 <= 2 and count_others < 2 and count_comp2 == 0)` |
| `hard` | `(count_others > 2 and count_comp1 <= 2 and count_comp2 == 0)` or `(2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0)` or `(count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1)` |
| `extra` | All remaining queries (e.g., `count_comp2 >= 2`, or CTE/IUEN + multiple clauses). |

---

## How We Adapted Yu et al. (2018) to `sqlglot`: Choices & Bug Fixes

Instead of cloning `taoyds/spider/process_sql.py` (which uses `nltk.word_tokenize`, downloads `punkt` at runtime, and fails on `>90%` of `bird_mini_dev` gold SQL) or lowering `sqlglot` ASTs into Spider's flat Python dicts (which fails on `376/500 = 75.2%` of `bird_mini_dev` gold SQL), `ComponentMatching` decomposes `sqlglot` ASTs directly while fixing five verified defects in `taoyds/spider/evaluation.py`:

### Bug Fixes vs. `taoyds/spider/evaluation.py`

1. **Zero-match corpus `f1 = 1` bug (`evaluation.py:604-605`):**
   * *Upstream bug:* When `acc == 0 and rec == 0`, Spider unconditionally sets `scores[level]['partial'][type_]['f1'] = 1`—even when `acc_count > 0` or `rec_count > 0` (meaning predictions or references *had* the component and matched **zero** times!). On `bird_mini_dev`, this bug would report `IUEN = 1.000` and `cte = 1.000` despite `0` matches!
   * *Our fix ([`finalize_component_bucket`](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py#L1987-L2011)):* Returns `None` (`precision`, `recall`, `f1`) **only** when `pred_count == 0 and label_count == 0`; returns `0.0` whenever either count `> 0` and `precision + recall == 0.0`.
2. **Swapped denominators & empty-set `(1, 1, 1)` inflation in `eval_and_or` (`evaluation.py:195-203, 419`):**
   * *Upstream bug:* On mismatch, `eval_and_or` returned `(len(pred_ao), len(label_ao), 0)`, but `eval_partial_match` unpacked `label_total, pred_total, cnt = eval_and_or(...)`—incrementing Precision's denominator using gold counts and Recall's using prediction counts. It also returned `(1, 1, 1)` when `pred_ao == label_ao == set()`, inflating `and/or` F1 with every query lacking boolean connectors.
   * *Our fix ([`score_pair`](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py#L1870-L1877)):* Assigns `ao_label_total = len(ref.where_and_or)` and `ao_pred_total = len(pred.where_and_or)` in the right slots, and returns `(pred_total=0, label_total=0, f1=1)` when both sets are empty so corpus `and/or` F1 measures actual `AND`/`OR` queries cleanly without breaking `ExactMatching`.
3. **In-place mutation of parsed gold dicts (`evaluation.py:123`):**
   * *Upstream bug:* `eval_sel` mutates `label['select'][1]` via `.remove(unit)`, corrupting gold dicts reused across multi-reference or repeated calls.
   * *Our fix:* All decomposed `NormalizedQuery` fields are immutable tuples/frozensets and overlap uses `collections.Counter` intersections.
4. **Hardness aggregate over-counting (`evaluation.py:333, 338`):**
   * *Upstream bug:* `count_others` ran `count_agg(unit[0] != 0)` on `where[::2]` (where index `0` is a boolean `not_op` flag!) and on unsliced `having` (containing `'and'`/`'or'` strings, where `'and'[0] != 0` is truthy), misclassifying queries with `NOT` or compound `HAVING` into higher hardness buckets.
   * *Our fix ([`_count_current_level_aggs`](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/matching.py#L1085-L1101)):* Counts actual `exp.AggFunc` AST nodes at the current query level across `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, and `HAVING`.
5. **Silent clause omissions in `eval_having` / `eval_order` (`evaluation.py:166-192`):**
   * *Upstream gaps:* Upstream `eval_having` ignored `HAVING` on queries without `GROUP BY` and compared `groupBy` and `having` as order-sensitive Python lists; `eval_order` ignored `LIMIT` on queries without `ORDER BY`.
   * *Our adaptations:* `group` triggers on `(group_by_full or having_preds)` and compares `Counter` multisets for `GROUP BY` and `HAVING` (+ `having_and_or`); `order` triggers on `(order_by or has_limit)` (handling `LIMIT`-only and `FETCH FIRST` queries cleanly).

### Intentional Generalizations & Configuration Choices

* **Non-literal-LHS canonicalization (`where` & `where(no OP)`):** Spider puts the column expression in `unit[2]` and literal in `unit[3]`. In `sqlglot`, `WHERE 'S' = s.rtype` puts `'S'` on `this` (LHS). Our predicate canonicalizer swaps `EQ`/`NEQ` and flips ordered inequalities (`10 < age` -> `age > 10`) so non-literal expressions stay on `this`, making `where(no OP)` work naturally without colliding unrelated columns (`WHERE a = 1` vs `WHERE b = 2` both yielding normalized column targets `a` and `b`).
* **`12` component keys (`+ cte`, `+ join_on`):** Adds `cte` (covering `WITH` queries present in `30%` of `bird_mini_dev`) and opt-in `join_on` (`include_join_conditions=True`) rather than hiding CTEs inside subqueries or losing `JOIN ON` signal.
* **Spider value-blind defaults + auditable strict flags:** Preserves `ignore_values=True`, `ignore_distinct=True`, `include_limit_value=False`, and `include_join_conditions=False` by default for Spider continuity, while exposing all flags as explicit keyword toggles and stamping them onto every `compute()` output dictionary.
* **Multi-reference deterministic winner selection:** Supports `str | Sequence[str]` references per example, picking the reference maximizing `(exact_match, matched_active_components, all_components_f1_sum, -ref_index)` (both score and hardness come from that winning reference).
* **Unparseable-prediction penalty (`skip_unparseable_predictions=False`):** Matches Spider's empty-query penalty (`0` on all gold components in `label_count` without shrinking recall denominators) by default, with an opt-in `skip_unparseable_predictions=True` toggle when evaluating syntax-valid subsets.

---

## API Reference

### Signatures

* **`gbv.load("component_matching", **kwargs) -> ComponentMatching`** / **`ComponentMatching(...)`**
* **`add(prediction, reference, schema=None, foreign_keys=None) -> None`**
* **`add_batch(predictions, references, schemas=None, foreign_keys=None) -> None`**
* **`compute_component_matching(predictions, references, schemas=None, foreign_keys=None, *, ...)`**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"postgres"` | Target `sqlglot` dialect (`"postgres"`, `"sqlite"`, `"mysql"`, `"snowflake"`, `"bigquery"`, etc.). |
| `ignore_values` | `bool` | `True` | Mask non-exempt literals to `__val__` and rewrite `COUNT(1)` -> `COUNT(*)`. Set `False` to check exact constants. |
| `ignore_distinct` | `bool` | `True` | Strip `DISTINCT` and collapse `UNION`/`INTERSECT`/`EXCEPT ALL` distinctions. Set `False` for strict matching. |
| `include_limit_value` | `bool` | `False` | Match exact `LIMIT`/`OFFSET` expressions instead of only presence/absence (`has_limit`). |
| `include_join_conditions` | `bool` | `False` | Evaluate flattened `JOIN ... ON` and `USING(...)` predicates in the `join_on` component. |
| `skip_unparseable_predictions` | `bool` | `False` | Skip unparseable predictions (`None` in `per_example`) instead of scoring against an empty query. |
| `extractor` | `ExtractorLike` | `None` | Optional extractor run on predictions only (`extract_fenced_or_last_sql`, regex, callable). |
| `max_query_chars` | `int \| None` | `65_536` | Character limit per SQL query before rejecting parse. |
| `max_input_chars` | `int \| None` | `131_072` | Character limit on raw prediction input passed to `extractor`. |
| `include_per_example` | `bool` | `False` | Include sanitized per-example component score dicts in `compute()["per_example"]`. |

### `compute()` Output Schema

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `components` | `dict[str, dict[str, float \| int \| None]]` | Overall (`"all"`) breakdown for all 12 keys; each maps to `{"precision", "recall", "f1", "pred_count", "label_count"}` (`None` when both counts are `0`). |
| `by_hardness` | `dict[str, dict[str, Any]]` | Keys `"easy"`, `"medium"`, `"hard"`, `"extra"`, `"all"`; each maps to `{"count": int, "components": <12-key dict>}`. |
| `scored_count` | `int` | Examples contributing to scores. |
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
| `per_example` | `list[dict[str, Any] \| None]` | *(Only when `include_per_example=True`)* `None` if skipped; else `{"hardness", "prediction_parsed", "schema_applied", "foreign_keys_applied", "components"}` (no raw SQL or literals). |

---

## Examples

### Spider Value-Blind Defaults vs. Opt-In Strict Mode

```python
import group_by_vibes as gbv

preds = [
    "SELECT u.id, COUNT(1) FROM users u JOIN orders o ON u.id = o.user_id "
    "WHERE 'A' = u.status GROUP BY u.id ORDER BY COUNT(1) LIMIT 10"
]
refs = [
    "SELECT ord.user_id, COUNT(*) FROM users usr JOIN orders ord ON usr.id = ord.user_id "
    "WHERE usr.status = 'Z' GROUP BY ord.user_id ORDER BY COUNT(*) LIMIT 99"
]
schemas = [{"users": ["id", "status"], "orders": ["user_id"]}]
fks = [[("users.id", "orders.user_id")]]

# Default Spider value-blind mode + FK equivalence collapse
default_res = gbv.compute_component_matching(
    preds, refs, schemas=schemas, foreign_keys=fks, dialect="postgres"
)
assert default_res["components"]["select"]["f1"] == 1.0
assert default_res["components"]["where"]["f1"] == 1.0
assert default_res["components"]["group"]["f1"] == 1.0
assert default_res["components"]["order"]["f1"] == 1.0
assert default_res["components"]["cte"]["f1"] is None

# Strict mode enforces exact literal values, COUNT(1) != COUNT(*), and LIMIT values
strict_res = gbv.compute_component_matching(
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
assert strict_res["components"]["select"]["f1"] == 0.0
assert strict_res["components"]["where"]["f1"] == 0.0
assert strict_res["components"]["order"]["f1"] == 0.0
```
