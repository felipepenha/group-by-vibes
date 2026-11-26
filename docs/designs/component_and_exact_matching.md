# Component Matching and Exact Matching: Dialect-General `sqlglot` AST Adaptation of Yu et al. (2018)

*   **Author:** Felipe Campos Penha
*   **Status:** Proposed
*   **Date:** 2026-11-24

--------------------------------------------------------------------------------

## Overview

This design adds two standalone, reference-based `Metric` evaluators to `group-by-vibes` post-`v0.0.1`—**`ComponentMatching`** (`load("component_matching")`, `compute_component_matching`) and **`ExactMatching`** (`load("exact_matching")`, `compute_exact_matching`)—sharing one internal, dialect-general `sqlglot` Abstract Syntax Tree (AST) decomposition and scoring engine.

Both metrics adapt the evaluation methodology of Yu et al. (EMNLP 2018) and its official reference script (`taoyds/spider`, Apache-2.0; whose `--etype match` scoring logic is identical in `taoyds/test-suite-sql-eval`):

> **Reference:** [Yu et al. (2018), Spider](https://arxiv.org/abs/1809.08887) — Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir R. Radev. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. EMNLP 2018, pages 3911–3921. Official reference code: [`taoyds/spider`](https://github.com/taoyds/spider) and [`taoyds/test-suite-sql-eval`](https://github.com/taoyds/test-suite-sql-eval).

Following the same adaptation principle established by `SqlBleu` (which adapts Chen et al., 2019 onto `sqlglot` ASTs), this design intentionally builds a **dialect-general, bug-fixed `sqlglot` AST adaptation** rather than an unmodified clone of `taoyds/spider`. Four explicit architectural choices govern this scope, and their rationale is documented upfront:

1.  **`sqlglot` AST decomposition instead of `taoyds/spider/process_sql.py` + `nltk`:**
    *   *Why:* `group-by-vibes` maintains `sqlglot` as its sole core runtime dependency (`pyproject.toml`), supports offline execution without runtime data downloads, and evaluates any dialect resolved by `sqlglot.Dialect.get_or_raise`. `taoyds/spider/process_sql.py` pulls `nltk.word_tokenize` (requiring `nltk` and runtime `punkt`/`punkt_tab` downloads), has no dialect awareness, and rejects `INNER`/`LEFT` joins, implicit table aliases, `SELECT` column `AS` aliases, `IS NULL`, `<>`, and quoted identifiers.
    *   *Measured evidence on this repository:* Across the 500 SQLite gold queries in `examples/logs/bird_qwen_3b.jsonl`, `process_sql.py` fails on >90% of queries. Even if parsed with `sqlglot`, lowering into Spider's flat dict AST fails on **376/500 (75.2%)** of `bird_mini_dev` gold queries because Spider's dict schema has no slots for scalar/cast functions (`CAST`, `ROUND`, `IF`/`IIF`, `SUBSTR`, `COALESCE`: 324/500), CTEs (`WITH`: 150/500), `CASE WHEN` (111/500), nested or literal arithmetic such as `1.0 * a / b` (111/500), or window functions (25/500). By contrast, `group_by_vibes.parser.parse_sql` parses **500/500 (100.0%)** of `bird_mini_dev` gold queries (498 `Select`, 2 `Union`) and **500/500 (100.0%)** in `examples/logs/bird_finer_sql_3b.jsonl`. Operating directly on `sqlglot` ASTs gives full coverage across `bird_mini_dev` and multi-dialect enterprise SQL.
2.  **Fixing verified defects in `taoyds/spider/evaluation.py` while retaining Spider's scoring method:**
    *   *Why:* Line-by-line auditing of `taoyds/spider/evaluation.py` identified two scoring-math defects affecting Component Matching, one hardness-bucketing defect affecting both metrics, and two runtime safety defects:
        *   **Zero-match corpus `f1 = 1` (`evaluation.py:594-609`, Component Matching only):** Assigns corpus `f1 = 1` whenever `acc == 0` and `rec == 0`, even when `acc_count > 0` or `rec_count > 0` (meaning every predicted or gold component failed to match).
        *   **Swapped `(label_total, pred_total)` in `eval_and_or` (`evaluation.py:195-203, 419`, Component Matching only):** Returns `(len(pred_ao), len(label_ao), 0)` on mismatch while the caller unpacks `(label_total, pred_total, cnt)`, incrementing the wrong precision/recall denominator.
        *   **Hardness aggregate counting (`evaluation.py:338, 343`, both metrics' hardness buckets):** `count_others` calls `count_agg` (`unit[0] != 0`) on `where[::2]` (counting `not_op` booleans rather than aggregates) and on unsliced `having` (counting `'and'`/`'or'` tokens and `not_op` booleans).
        *   **Runtime mutation and sort crash (`evaluation.py:123, 387-388`, both metrics):** `eval_sel` mutates `label['select'][1]` in place via `.remove()`, and `eval_exact_match` runs `sorted(label['from']['table_units'])`, raising `TypeError` in Python 3 when `FROM` contains two or more subquery dicts.
    *   Fixing these defects prevents false `f1 = 1.0` CI readings, inverted denominators, misbucketed hardness counts, and Python 3 crashes while keeping Yu et al. (2018)'s exact-set clause matching intact.
3.  **Spider value-blind defaults with opt-in strict matching flags:**
    *   *Why:* Yu et al. (2018) intentionally excludes literal values (`DISABLE_VALUE = True`), `DISTINCT` (`DISABLE_DISTINCT = True`), `LIMIT` integer values, and `JOIN ... ON` predicates because Spider's 2018 task did not require value prediction. Keeping those defaults preserves methodological continuity with Yu et al. (2018) out of the box, while opt-in strict flags allow modern LLM benchmarks (`bird_mini_dev`, enterprise SQL) to enforce exact literal values, `DISTINCT`, `LIMIT` counts, and `JOIN ... ON` conditions without silent false positives.
4.  **Two standalone `Metric` evaluators (`ComponentMatching` and `ExactMatching`) sharing one internal engine:**
    *   *Why:* Matches the two metric names in Yu et al. (2018) and `group-by-vibes`'s Hugging Face `evaluate` convention of one evaluator per `load()` name, avoiding a benchmark-named monolithic class while sharing all AST parsing, canonicalization, clause extraction, and matching code internally.

Because neither metric executes SQL against a database engine, both serve as deterministic structural diagnostics and exact-set benchmarks that complement—rather than replace—Execution Accuracy (`EX`).

## The Problem

AI Engineers evaluating Text2SQL LLMs with `group-by-vibes` `v0.0.1` and existing Spider tooling face three gaps:

1.  **No clause-level diagnostics or structural exact matching in `group-by-vibes`:** `v0.0.1` provides `SqlBleu` (corpus n-gram overlap), `ValidSqlRate` (lenient syntax + static grounding), and `WildcardDetector` (`SELECT *` rate). None of these tells an engineer *which* SQL clauses (`SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `WITH`, set operations, or keywords) a model gets right or misses, nor whether a generated query matches the reference structurally independent of clause element ordering.
2.  **`taoyds/spider` scripts fail on multi-dialect SQL and modern benchmarks:** The official `taoyds/spider` scripts require `nltk` plus runtime `punkt` downloads, reject standard dialect syntax (`INNER`/`LEFT JOIN`, column aliases, quoted identifiers, `IS NULL`), and use a flat dict AST unable to represent 75.2% of `bird_mini_dev` gold queries. Teams evaluating BIRD or enterprise PostgreSQL/SQLite workloads cannot run `taoyds/spider` without dropping most of their dataset.
3.  **Upstream defects and inflexible value-blindness in `taoyds/spider/evaluation.py`:** Upstream `evaluation.py` reports `f1 = 1.0` for 0%-matching components, swaps `and/or` denominators on mismatch, crashes on multiple `FROM` subqueries, mutates parsed gold dicts in place, miscounts aggregates during hardness classification, and hardcodes `DISABLE_VALUE = True` / `DISABLE_DISTINCT = True` with no way for modern LLM pipelines to check literals, `DISTINCT`, `LIMIT` counts, or `JOIN ... ON` keys.

`group-by-vibes` solves all three gaps by shipping dialect-general, bug-fixed `ComponentMatching` and `ExactMatching` metrics over `sqlglot` ASTs with Spider value-blind defaults and opt-in strict matching flags.

## Goals

*   **Two Standalone `Metric` Evaluators (`src/group_by_vibes/`):** Ship `ComponentMatching` (`load("component_matching")`, `compute_component_matching`) and `ExactMatching` (`load("exact_matching")`, `compute_exact_matching`) sharing a single private AST decomposition and scoring engine, following `BaseEvaluator`'s synchronous `add()` / `add_batch()` / `compute()` / `reset()` lifecycle, JSON-serializable `dict` outputs, shared optional `extractor`, input/query char limits, and opt-in `include_per_example`.
*   **Zero New Core Dependencies & 100% `bird_mini_dev` Gold Coverage:** Keep `sqlglot>=30.0.0,<31.0.0` as the sole runtime dependency (no `nltk`, no `sqlparse`, no runtime downloads), accept every dialect supported by `sqlglot.Dialect.get_or_raise`, and parse and decompose 100% of `bird_mini_dev` gold queries (500/500 in SQLite logs) including CTEs (`WITH`), `CASE WHEN`, scalar/cast functions, window functions, and nested arithmetic.
*   **Yu et al. (2018) Method Adaptation with Fixed Upstream Defects:** Preserve exact-set clause matching, hardness stratification (`easy`, `medium`, `hard`, `extra`, `all`), and per-component precision/recall/F1 reporting while fixing all five verified defects in `taoyds/spider/evaluation.py` (zero-match corpus F1, `eval_and_or` swapped denominators, `eval_sel` in-place mutation, `FROM` subquery sort crash, and `count_others` hardness bug) and documenting every intentional divergence clearly.
*   **Spider Value-Blind Defaults with Auditable Opt-In Strict Flags:** Default to Spider's structural matching semantics (ignoring literal values, `DISTINCT`, `LIMIT` counts, and `JOIN ... ON` predicates) and expose explicit opt-in constructor flags to enforce strict matching on each, recording the active flag configuration in every `compute()` output `dict`.
*   **Parity, Regression, and Reference Benchmark Integration:** Validate alignment against `taoyds/spider/evaluation.py` on non-buggy Spider-subset test cases, add explicit regression tests for every fixed upstream bug and modern SQL construct, and wire both metrics into `examples/bird_eval.py` for full offline re-scoring of `examples/logs/` without compromising core packaging isolation.

## Non-Goals

*   Porting `taoyds/spider/process_sql.py` (hand-rolled parser + `nltk.word_tokenize`) or reproducing `taoyds/spider/evaluation.py`'s arithmetic bugs for raw stdout equality on buggy edge cases.
*   Adding `nltk`, `sqlparse`, or any runtime corpus download to core or optional package dependencies.
*   Database-executed metrics (`Execution Accuracy`, `Test Suite Accuracy` from `taoyds/test-suite-sql-eval`, `EXPLAIN`, or `DRY RUN`).
*   Semantic or algebraic query-equivalence proving across distinct SQL plans (e.g., proving `1.0 * a / b` equivalent to `CAST(a AS REAL) / b`, `IIF` equivalent to `CASE WHEN`, `JOIN` equivalent to `IN (SELECT ...)`, or CTE equivalent to derived subquery).
*   Multi-turn conversational SQL session evaluation (SParC / CoSQL `turn 1..>4` and `joint_all` metrics).
*   A third combined single-pass `SpiderMatch` evaluator class (two standalone evaluators—`ComponentMatching` and `ExactMatching`—were chosen).
*   Multi-statement SQL evaluation or component matching for non-query DDL/DML statements (`CREATE`, `DROP`, `INSERT`, `UPDATE`, `DELETE`).

## Open Questions

None — all architectural questions have been resolved and incorporated into the Technical Plan below.

## Critical User Journeys (CUJs)

### CUJ 1: Diagnose Clause-Level Failures Across Dialects (`ComponentMatching`)

*   **Given:** An AI Engineer has SQL predictions and golden references for any `sqlglot` dialect (including PostgreSQL or SQLite queries with CTEs, `CASE WHEN`, `CAST`, and window functions) and loads `gbv.load("component_matching", dialect="sqlite")`.
*   **When:** The engineer adds prediction/reference pairs via `add()` / `add_batch()` (with optional extraction via `extractor=extract_fenced_or_last_sql`, using default Spider value-blind mode or opt-in strict flags) and calls `compute()`.
*   **Then:** `ComponentMatching` decomposes each query's `sqlglot` AST into clause sets, returns JSON-serializable per-component precision, recall, and F1 overall and stratified by hardness (`easy`, `medium`, `hard`, `extra`, `all`) without the upstream zero-match `f1 = 1` or swapped `and/or` denominator bugs, records active mode flags, and resets accumulator state.

### CUJ 2: Measure Order-Insensitive Structural Exact Match (`ExactMatching`)

*   **Given:** An AI Engineer evaluates whether predicted queries match golden queries structurally regardless of clause item ordering (e.g., reordered `SELECT` projections or `WHERE` predicates), using `gbv.load("exact_matching", dialect="postgres")`.
*   **When:** The engineer runs `add()` / `add_batch()` and calls `compute()`, either with Spider value-blind defaults (for structural skeleton comparison) or with opt-in strict flags enabled (to require matching literal values, `DISTINCT`, `LIMIT` counts, and `JOIN ... ON` predicates).
*   **Then:** `ExactMatching` evaluates all components and table/clause structure using the shared AST engine, returns overall and hardness-stratified exact-match accuracy plus active mode flags and optional `per_example` flags (`include_per_example=True`), and resets state after `compute()`.

### CUJ 3: Re-Score Full `bird_mini_dev` Model Logs Offline (`examples/bird_eval.py`)

*   **Given:** An AI Engineer runs offline evaluation on existing 500-row `bird_mini_dev` JSONL logs (`examples/logs/bird_qwen_3b.jsonl`, `bird_xiyansql_3b.jsonl`, `bird_finer_sql_3b.jsonl`) using core `group-by-vibes` only (no ML packages, no `nltk`).
*   **When:** `examples/bird_eval.py score` passes extracted predictions and golden SQL through `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `ComponentMatching`, and `ExactMatching`.
*   **Then:** All 500 `bird_mini_dev` gold queries parse and decompose deterministically across both new metrics (100% gold coverage, compared to <25% under Spider's dict schema), producing reproducible comparative scores with zero extra core dependencies.

## Technical Plan (Architecture)

### High-Level Data Flow

```text
  Caller Inputs (prediction, reference: str | Sequence[str], dialect,
                 optional per-row schema & foreign_keys, mode flags)
         |
         v
  +-----------------------------------------------------------------------------------+
  | 1. Optional Shared SQL Extractor (`extract.py`)                                   |
  |    - Runs on predictions only when `extractor is not None`                        |
  |    - Enforces `max_input_chars` pre-extraction                                    |
  +-----------------------------------------------------------------------------------+
         |
         v
  +-----------------------------------------------------------------------------------+
  | 2. Isolated sqlglot Parser (`parser.py` + `dialects.py`)                          |
  |    - Parses prediction + all references under normalized `sqlglot` dialect        |
  |    - Rejects empty, multi-statement, `exp.Command`, non-query DDL/DML ASTs        |
  |    - Unparseable references dropped; row skipped if 0 valid references remain     |
  |    - Unparseable prediction -> empty decomposition (`skip=False`) or skip (`True`)|
  +-----------------------------------------------------------------------------------+
         |
         v
  +-----------------------------------------------------------------------------------+
  | 3. Shared Dialect-General AST Matching Engine (`matching.py`)                     |
  |    a. Per-row eager validation & normalization of optional `schema`               |
  |       (`grounding.normalize_schema`) and optional `foreign_keys` (union-find kmap)|
  |    b. Scope-aware AST canonicalization (no `sqlglot.optimizer.qualify` rewrites): |
  |       - Unquoted identifier folding, table-alias -> physical table resolution     |
  |       - Optional `schema` unqualified column qualification & `kmap` FK collapse   |
  |       - Deterministic CTE name alpha-renaming (`__cte_0__`, `__cte_1__`, ...)     |
  |       - Commutative predicate sorting (`=`, `!=`, `IN`, `AND`/`OR`), `<>` -> `!=` |
  |       - Mode flags: `ignore_values`, `ignore_distinct`,                           |
  |         `include_limit_value`, `include_join_conditions`                          |
  |    c. 12-component clause extraction + fixed hardness classification              |
  |    d. Multi-reference deterministic best-reference selection                      |
  +-----------------------------------------------------------------------------------+
         |                                                |
         v                                                v
  +-------------------------------+        +------------------------------------------+
  | ComponentMatching (`Metric`)  |        | ExactMatching (`Metric`)                 |
  | `component_matching.py`       |        | `exact_matching.py`                      |
  | - 12 component types x        |        | - Requires all 12 components `f1 == 1`   |
  |   5 hardness buckets          |        |   AND canonical `FROM` table multiset    |
  | - Fixed zero-match corpus F1  |        | - Stratified across 4 hardness levels    |
  | - Fixed `and/or` denominators |        |   + `all`                                |
  +-------------------------------+        +------------------------------------------+
         |                                                |
         +------------------------+-----------------------+
                                  v
  +-----------------------------------------------------------------------------------+
  | 4. Single-Writer Streaming Accumulators (`base.py`)                               |
  |    - O(1)-counter accumulators (`add` / `add_batch` / `compute` / `reset`)        |
  |    - Opt-in `include_per_example` in O(N) memory; no raw SQL/literals in output   |
  +-----------------------------------------------------------------------------------+
                                  |
                                  v
                    JSON-Serializable `dict` Outputs
```

### Module Boundaries & Core Engine (`src/group_by_vibes/`)

1.  **`matching.py` — Shared AST Normalization, Decomposition, Hardness & Comparison Engine:**
    *   **Query AST Gate:** Calls `parser.parse_sql` on predictions (post-`extract.apply_extractor` when `apply_extract=True` and `extractor is not None`) and on every reference string (`apply_extract=False`). Accepts only query root expressions (`exp.Query`: `exp.Select`, `exp.Union`, `exp.Intersect`, `exp.Except`, unwrapping outer `exp.Subquery` parens). Rejects non-query statements (`CREATE`, `DROP`, `INSERT`, `UPDATE`, `DELETE`) as unparseable for query matching. Catches `(SqlglotError, RecursionError)` across all AST walks so malformed or deep trees never crash a batch.
    *   **Two-Layer Identifier Normalization, Optional `schema`, and Optional `foreign_keys` (`kmap`):**
        *   *Lookup Layer (for `schema` and `kmap` resolution):* Validates `schema: SchemaDict | None` eagerly per row via `grounding.normalize_schema(schema, dialect=dialect)` (zero cross-row caching). Validates `foreign_keys: ForeignKeysLike | None` eagerly per row: accepts `None` or an empty sequence `[]` (producing empty `kmap = {}` and `foreign_keys_applied = False`); rejects `str`/`bytes`/`Mapping` outer containers with `TypeError`; requires every element to be a 2-element non-`str`/non-`bytes`/non-`Mapping` sequence `(left, right)` of non-empty `"table.col"` strings (`TypeError` on wrong types, `ValueError` on length != 2 or blank strings). Parses each endpoint via `dialects.split_qualified_table_parts(endpoint, dialect=dialect)`, rejecting 1-part (`db is None`) or 3-part (`catalog is not None`) names with `ValueError`. Normalizes table and column lookup keys using `dialects.normalize_grounding_identifier(table_name, table_quoted, dialect, is_table=True)` and `dialects.normalize_grounding_identifier(col_name, col_quoted, dialect, is_table=False)`. Builds transitive union-find equivalence classes mapping every lookup key `(norm_table, norm_col)` to the lexicographically smallest canonical AST representative `((canon_table_name, canon_table_quoted), (canon_col_name, canon_col_quoted))`.
        *   *Canonical AST Layer (for rendered expressions):* After scope-aware lookup/rewriting on a deep AST copy (leaving caller inputs untouched), folds all unquoted `exp.Identifier` nodes via `ident.this.lower()` while preserving exact text on quoted identifiers (`ident.quoted == True`) across every dialect (matching `canonicalize.py` so e.g. SQLite quoted `"Free Meal Count (Ages 5-17)"` and backticked `` `Free Meal Count (Ages 5-17)` `` match identically while lookup keys still resolve case-insensitively on SQLite and case-appropriately on Snowflake/PostgreSQL/MySQL/BigQuery). Canonical `kmap` representatives store unquoted parts lowercased (`quoted=False`) and quoted parts with preserved case (`quoted=True`).
        *   *Scope-Aware Alias, Qualification, Self-Join & `SELECT`-Alias Rules:* Traverses `sqlglot.optimizer.scope.build_scope` on the copied AST:
            *   Maps table aliases (`FROM users AS u` or `FROM users u`) to canonical physical table names (`u.name` -> `users.name`). Self-joins (`FROM emp e1 JOIN emp e2`) map both aliases to physical table `emp` (`from_tables` multiset `("emp", "emp")`). Derived-table aliases (`FROM (SELECT ...) AS sub`) rewrite qualifiers to deterministic positional placeholders `__subq_0__`, `__subq_1__`, ... ordered by each derived table's `canonical_signature`.
            *   Before qualifying unqualified columns inside `GROUP BY`, `HAVING`, or `ORDER BY`, checks `SELECT` output aliases in the enclosing `exp.Select` (reusing `grounding._select_alias_names` semantics) so references to projection aliases (e.g. `SELECT COUNT(*) AS cnt FROM users ORDER BY cnt`) are never mis-qualified to a physical table.
            *   Qualifies remaining unqualified columns to `table.col` when (a) `schema` is provided and exactly one physical table in the current scope owns the column's Layer-A lookup key, or (b) `schema` is `None` and the scope contains exactly one non-opaque physical table (and no CTE/derived sources). Leaves ambiguous or unresolvable columns unqualified.
            *   Applies `kmap` FK replacement to resolved `table.col` references only when the referenced physical table appears in the current query scope's `FROM`/`JOIN` sources (matching Spider's `build_valid_col_units` scoping rule).
        *   *Wave-by-Wave Deterministic CTE Alpha-Renaming:* For `WITH` clauses (`exp.With`), partitions CTEs into Kahn topological waves (wave 0 = CTEs referencing no other CTEs; wave `w` = CTEs whose CTE dependencies lie in waves `< w`; recursive self-references temporarily bind to `__self__` during body normalization). Within each wave, normalizes every CTE body substituting earlier-wave `__cte_i__` names, includes any explicit CTE column aliases (`WITH c(x, y) AS (...)`) in scope binding and canonical signature, sorts wave entries by `(body.canonical_signature, canonical_column_aliases)`, assigns consecutive `__cte_0__`, `__cte_1__`, ... IDs, and rewrites references in downstream waves and the main query.
    *   **Mode Flags (Spider Value-Blind Defaults + Opt-In Strict Matching):**
        *   `ignore_values: bool = True`: Runs **before** predicate canonicalization. Replaces non-subquery literal nodes (`exp.Literal`, `exp.Boolean`, `exp.Null`, `exp.Neg(this=exp.Literal)`) with `exp.Var(this="__val__")` (rendering unquoted `__val__` across dialects). Normalizes `COUNT(1)` (`exp.Count(this=exp.Literal.number(1))`) to `COUNT(*)` (`exp.Count(this=exp.Star())`) when `ignore_values=True` (preserving `COUNT(1)` when `False`). Explicitly exempts non-value literal AST nodes from masking: `exp.DataTypeParam` (e.g. `CAST(x AS VARCHAR(10))`), `exp.Limit`/`exp.Offset` expressions (governed by `include_limit_value`), `exp.WindowSpec` frame bounds, and positional integer ordinals directly in `exp.Group.expressions` or `exp.Ordered.this` (`GROUP BY 1`, `ORDER BY 1`). Preserves nested subquery ASTs.
        *   `ignore_distinct: bool = True`: Strips `exp.Distinct` on `SELECT` projections and aggregate function arguments, and collapses `UNION ALL` / `INTERSECT ALL` / `EXCEPT ALL` to `"union"` / `"intersect"` / `"except"`. When `False`, preserves `DISTINCT` and `*_all` set-op kinds.
        *   `include_limit_value: bool = False`: When `False`, `order` compares only `has_limit` (presence/absence of `LIMIT`). When `True`, compares canonicalized `LIMIT` and `OFFSET` expressions (`limit_value`).
        *   `include_join_conditions: bool = False`: When `False`, `JOIN ... ON` and `USING` predicates are excluded from `join_on` and `ExactMatching`. When `True`, flattened `ON` predicates and normalized `USING` column tokens populate `join_on` and are enforced in `ExactMatching`.
    *   **Non-Literal-LHS Predicate Canonicalization & Recursive Subquery Embedding:**
        *   `matching.py` implements its own post-order `_canonicalize_predicates(ast, dialect)` helper (no cross-module dependency on private `canonicalize._rewrite_commutative`):
            *   Both `<>` and `!=` parse to `exp.NEQ` in `sqlglot` and render consistently under the active dialect.
            *   Classifies operands as literal-like (`exp.Literal`, `exp.Boolean`, `exp.Null`, `exp.Placeholder`, `exp.Var(this="__val__")`, `exp.Neg(this=exp.Literal)`). For `exp.EQ` and `exp.NEQ`, if exactly one side is literal-like, always places the **non-literal** expression on `this` (LHS) and the literal-like operand on `expression` (RHS) (so `'S' = s.rtype` becomes `satscores.rtype = 'S'` / `satscores.rtype = __val__`); when both sides have the same literal-like class, orders deterministically by rendered SQL (`left_sql <= right_sql`). For ordered comparisons (`<`, `<=`, `>`, `>=`), flips operator and operands when LHS is literal-like and RHS is non-literal (e.g. `10 < age` -> `age > 10`).
            *   Sorts subquery-free `exp.In` expressions by rendered SQL and flattens/sorts same-operator `exp.And`/`exp.Or` chains while preserving mixed-operator precedence parentheses (`_LOGICAL_CLASSES = (exp.And, exp.Or, exp.Xor)`).
        *   Strips `SELECT` output aliases (`SELECT expr AS alias` -> `expr`) after resolving `GROUP BY`/`HAVING`/`ORDER BY` alias references.
        *   Every nested `exp.Query` inside `WHERE`, `HAVING`, `SELECT`, or `JOIN ON` is recursively decomposed via `decompose_query_ast` into a `NormalizedQuery` and replaced in the enclosing expression by `exp.Var(this=f"__subq_({sub_nq.canonical_signature})__")`, making nested subqueries order-insensitive across clauses without raw Python dict comparison.

2.  **12-Component Clause Decomposition (`COMPONENT_TYPES`):**
    All per-example component scores use Spider's binary exact-set rule (`get_scores(cnt, pred_total, label_total)` returns `(1, 1, 1)` iff `pred_total == label_total` and `cnt == pred_total`, else `(0, 0, 0)`), operating on immutable multisets (`collections.Counter`) so no reference state is ever mutated:

    | # | Component Key | Per-Example Extraction & Match Rule |
    | :--- | :--- | :--- |
    | 1 | `select` | Multiset of canonicalized `SELECT` projection expressions (including outer aggregates and `DISTINCT` when `ignore_distinct=False`). |
    | 2 | `select(no AGG)` | Multiset of `SELECT` projection expressions with outermost standard aggregate wrapper (`MAX`, `MIN`, `COUNT`, `SUM`, `AVG`) stripped; non-aggregate items stay unchanged. |
    | 3 | `where` | Multiset of flattened top-level `WHERE` predicate atoms (splitting `AND`/`OR` chains while unwrapping `exp.Paren`; compound `NOT (a AND b)` stays a single atom; respecting `ignore_values` and recursive `__subq_(...)__` canonicalization). |
    | 4 | `where(no OP)` | Multiset of canonical target expressions extracted from each `WHERE` predicate atom (unwrapping outer `exp.Not`/`exp.Paren`): `p.this` (guaranteed non-literal LHS when one operand is non-literal) for binary comparisons (`=`, `<>`, `<`, `<=`, `>`, `>=`, `LIKE`, `ILIKE`, `IS`), `BETWEEN`, and `IN`; `"exists"` for `exp.Exists`; canonicalized inner expression for compound `NOT (...)` and bare boolean columns/calls (`WHERE is_active`). |
    | 5 | `group(no Having)` | Multiset of canonical `GROUP BY` expressions with leading table qualifiers stripped on bare columns (matching Spider's `eval_group`). |
    | 6 | `group` | Presence unit (`1` if `GROUP BY` or `HAVING` is non-empty, else `0`); matches (`1`) iff canonical `GROUP BY` multiset (with table qualifiers preserved after FK canonicalization) matches AND canonical `HAVING` predicate multiset + `HAVING` `AND`/`OR` connector set match. |
    | 7 | `order` | Presence unit (`1` if `ORDER BY` or `LIMIT` is present, else `0`); matches (`1`) iff ordered `ORDER BY` `(expr, "asc"\|"desc", "first"\|"last"\|"default")` tuple list (defaulting omitted direction to `"asc"`) matches AND `LIMIT` matches (`has_limit` presence if `include_limit_value=False`, else canonical `LIMIT` + `OFFSET` string `limit_value`). |
    | 8 | `and/or` | Set of boolean connectors (`"and"`, `"or"`) in `WHERE`. **Bug-fixed & non-inflated:** returns `(label_total=len(label_ao), pred_total=len(pred_ao), cnt=len(label_ao) if pred_ao == label_ao else 0)` (correcting Spider's swapped return order). When both sets are empty (`set()`), returns `(0, 0, 0)` with per-example `(precision=1, recall=1, f1=1)` so `ExactMatching` passes without inflating corpus `and/or` denominators or rewarding unparseable empty fallbacks. |
    | 9 | `IUEN` | Left-associative `UNION` / `INTERSECT` / `EXCEPT` chains decompose their leftmost base `Select` into the outer query's `select`/`from`/`where`/etc. (matching Spider `evaluation.py:235-242`) and record each RHS branch as `(op_kind, NormalizedQuery)` in `iuen_branches` (merging any root-level `WITH`, `ORDER BY`, and `LIMIT`/`OFFSET` onto the outer query). Matched as an order-insensitive multiset via recursive `exact_match`. |
    | 10 | `keywords` | Set of structural SQL keywords at the **current query level only** (not recursing into subqueries/CTEs/IUEN branches, matching Spider `get_keywords`): `where`, `group`, `having`, `order`, `asc` (whenever any `ORDER BY` item defaults to or specifies `ASC`), `desc`, `limit`, `intersect`, `union`, `except`, `with`, plus from current-level `FROM`/`WHERE`/`HAVING` conditions: `or`, `not` (any `exp.Not`, `NOT IN`, `NOT LIKE`, `IS NOT`), `in`, and `like` (`exp.Like` or `exp.ILike`). |
    | 11 | `cte` | Multiset of canonicalized `WITH` CTE body decompositions (post wave-by-wave alpha-renaming to `__cte_i__`), matched recursively via `exact_match`. |
    | 12 | `join_on` | When `include_join_conditions=True`, multiset of canonicalized `JOIN ... ON` / `USING` predicates across all joins at the current query level. When `include_join_conditions=False`, returns `(0, 0, 0)` with per-example `(precision=1, recall=1, f1=1)` (inactive). |

3.  **Exact Matching Rule (`eval_exact_match`):**
    A pair scores `exact_match = 1` iff:
    *   All 12 components have per-example `f1 == 1` (a component with `pred_total == 0 and label_total == 0` has per-example `(precision=1, recall=1, f1=1)`), AND
    *   The multisets of canonical `FROM` table names (`from_tables`) match AND `from_subqueries` match as an order-insensitive multiset via recursive `exact_match` (deterministic multiset matching instead of Python 3 `sorted()` over dicts).

4.  **Corrected Hardness Classification (`eval_hardness`):**
    Preserves Yu et al. (2018)'s 4 levels (`easy`, `medium`, `hard`, `extra`) and threshold tree (`evaluation.py:367-377`), correcting `count_others`'s aggregate-counting defect:
    *   `count_component1`: `+1` each for non-empty `where_preds`, `group_by_full` (or `having_preds`), `order_by`, `has_limit`, plus `max(0, len(from_tables) + len(from_subqueries) - 1)`, plus count of `exp.Or` and `exp.Like`/`exp.ILike` nodes in current-level `FROM`/`WHERE`/`HAVING`.
    *   `count_component2`: count of **direct** nested query nodes (`len(from_subqueries)` + direct `exp.Query` nodes in current-level `WHERE`/`HAVING`/`SELECT`/`JOIN ON`) + `len(iuen_branches)` + `len(ctes)`.
    *   `count_others`: `+1` if count of current-level `exp.AggFunc` nodes across `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `HAVING` (excluding nodes inside nested subqueries and never counting `not_op` booleans or `'and'`/`'or'` tokens) > 1; `+1` if `len(select_items) > 1`; `+1` if `len(where_preds) > 1`; `+1` if `len(group_by_full) > 1`.

5.  **Corpus Aggregation (Corrected Zero-Match F1) & Multi-Reference Selection:**
    *   **Corpus component aggregation per hardness bucket (`easy`, `medium`, `hard`, `extra`, `all`):**
        *   Accumulate `acc_sum`, `acc_count` (rows with `pred_total > 0`), `rec_sum`, `rec_count` (rows with `label_total > 0`) in O(1) counters per `(level, component)`.
        *   Finalize each `(level, component)`:
            *   If `acc_count == 0 and rec_count == 0`: `precision = None`, `recall = None`, `f1 = None`.
            *   Otherwise: `precision = acc_sum / acc_count if acc_count > 0 else 0.0`, `recall = rec_sum / rec_count if rec_count > 0 else 0.0`, and `f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0`.
    *   **Unparseable handling:**
        *   Invalid/non-query references are dropped per row (`parsed_refs_count / total_refs_count` tracked). If zero valid references remain, skip row (`None` in `per_example`).
        *   Invalid/non-query prediction on a valid reference: if `skip_unparseable_predictions=False` (default), score against an empty query decomposition (`exact = 0`, all reference components count as misses in `rec_count`, `acc_count` unchanged, `scored_count += 1`); if `skip_unparseable_predictions=True`, skip row (`None` in `per_example`).
    *   **Multi-reference deterministic best-reference selection:**
        *   Score the prediction (or empty fallback) against every valid reference in the row's reference group and select the winning reference maximizing `(exact_match, matched_active_components, all_components_f1_sum, -ref_index)`, where `"active component"` means `label_total > 0 or pred_total > 0`, `matched_active_components` counts active components with `f1 == 1`, and `all_components_f1_sum` sums `f1` across all 12 components (distinguishing references with fewer extra unmatched components).
        *   The winning reference determines both the exact/component tallies and the row's hardness bucket. *(Documented trade-off: on multi-reference datasets where references for the same question have different hardness levels, per-hardness `count`s can vary across models depending on which reference matched.)*

6.  **`component_matching.py`, `exact_matching.py`, and `__init__.py`:**
    *   `ComponentMatching(BaseEvaluator)` and `ExactMatching(BaseEvaluator)` wrap `matching.py` with single-writer O(1) streaming counters, eager batch sequence/length validation before state mutation, post-`compute()` reset, and stateless `compute_component_matching` / `compute_exact_matching` helpers.
    *   `__init__.py` exports `ComponentMatching`, `ExactMatching`, `compute_component_matching`, `compute_exact_matching`, and registers `@overload` `load("component_matching", **kwargs)` and `load("exact_matching", **kwargs)`.

### Public Function & Class Signatures & Output Schemas

#### Public API Signatures (`src/group_by_vibes/`)

```python
ForeignKeyPair = tuple[str, str]
ForeignKeysLike = Sequence[ForeignKeyPair]
# Constructor bounds inherit BaseEvaluator validation (`max_query_chars`, `max_input_chars` = None or positive int, not bool).

class ComponentMatching(BaseEvaluator):
    def __init__(
        self,
        dialect: str = "postgres",
        *,
        ignore_values: bool = True,
        ignore_distinct: bool = True,
        include_limit_value: bool = False,
        include_join_conditions: bool = False,
        skip_unparseable_predictions: bool = False,
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None: ...
    def add(
        self,
        prediction: str,
        reference: str | Sequence[str],
        schema: SchemaDict | None = None,
        foreign_keys: ForeignKeysLike | None = None,
    ) -> None: ...
    def add_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str] | Sequence[Sequence[str]],
        schemas: Sequence[SchemaDict | None] | None = None,
        foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    ) -> None: ...
    def compute(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class ExactMatching(BaseEvaluator):
    def __init__(
        self,
        dialect: str = "postgres",
        *,
        ignore_values: bool = True,
        ignore_distinct: bool = True,
        include_limit_value: bool = False,
        include_join_conditions: bool = False,
        skip_unparseable_predictions: bool = False,
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None: ...
    def add(
        self,
        prediction: str,
        reference: str | Sequence[str],
        schema: SchemaDict | None = None,
        foreign_keys: ForeignKeysLike | None = None,
    ) -> None: ...
    def add_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str] | Sequence[Sequence[str]],
        schemas: Sequence[SchemaDict | None] | None = None,
        foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    ) -> None: ...
    def compute(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

def compute_component_matching(
    predictions: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    schemas: Sequence[SchemaDict | None] | None = None,
    foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    *,
    dialect: str = "postgres",
    ignore_values: bool = True,
    ignore_distinct: bool = True,
    include_limit_value: bool = False,
    include_join_conditions: bool = False,
    skip_unparseable_predictions: bool = False,
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]: ...

def compute_exact_matching(
    predictions: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    schemas: Sequence[SchemaDict | None] | None = None,
    foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    *,
    dialect: str = "postgres",
    ignore_values: bool = True,
    ignore_distinct: bool = True,
    include_limit_value: bool = False,
    include_join_conditions: bool = False,
    skip_unparseable_predictions: bool = False,
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]: ...

@overload
def load(name: Literal["component_matching"], **kwargs: Any) -> ComponentMatching: ...
@overload
def load(name: Literal["exact_matching"], **kwargs: Any) -> ExactMatching: ...
```

#### Exact `compute()` JSON-Serializable Output Schemas

##### 1. `ComponentMatching.compute()` (`Metric`)

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `components` | `dict[str, dict[str, float \| int \| None]]` | Overall (`"all"`) breakdown across all 12 component keys (`select`, `select(no AGG)`, `where`, `where(no OP)`, `group(no Having)`, `group`, `order`, `and/or`, `IUEN`, `keywords`, `cte`, `join_on`); each maps to `{"precision": float \| None, "recall": float \| None, "f1": float \| None, "pred_count": int, "label_count": int}` (`precision`/`recall` correspond to Spider's internal `acc`/`rec`; all three are `None` when `pred_count == 0 and label_count == 0`). |
| `by_hardness` | `dict[str, dict[str, Any]]` | Keys `"easy"`, `"medium"`, `"hard"`, `"extra"`, `"all"`; each maps to `{"count": int, "components": <same 12-key dict>}`. |
| `scored_count` | `int` | Rows contributing to scores (all valid-reference rows when `skip_unparseable_predictions=False`; valid-prediction + valid-reference pairs when `True`). |
| `parsed_count` | `int` | Predictions passing extraction, char limits, and query-AST parsing. |
| `unparseable_count` | `int` | Predictions failing extraction, char limits, parsing, or query-AST check. |
| `parse_success_rate` | `float` | `parsed_count / num_examples`. |
| `reference_parse_success_rate` | `float` | Fraction of all supplied reference strings passing query-AST parsing. |
| `schema_applied_count` | `int` | Scored rows evaluated with a non-`None` `schema`. |
| `foreign_keys_applied_count` | `int` | Scored rows evaluated with non-empty `foreign_keys`. |
| `num_examples` | `int` | Total example rows processed. |
| `dialect` | `str` | Normalized `sqlglot` dialect used. |
| `ignore_values` | `bool` | Active value-blind flag (`True` by default). |
| `ignore_distinct` | `bool` | Active `DISTINCT`-blind flag (`True` by default). |
| `include_limit_value` | `bool` | Active `LIMIT` value-matching flag (`False` by default). |
| `include_join_conditions` | `bool` | Active `JOIN ... ON` matching flag (`False` by default). |
| `skip_unparseable_predictions` | `bool` | Active unparseable-prediction policy (`False` by default). |
| `per_example` | `list[dict[str, Any] \| None]` | *(Only when `include_per_example=True`)* `None` if skipped; else `{"hardness": str, "prediction_parsed": bool, "schema_applied": bool, "foreign_keys_applied": bool, "components": {name: {"precision": int, "recall": int, "f1": int, "label_total": int, "pred_total": int}}}`. Raw SQL and literal values omitted. |

##### 2. `ExactMatching.compute()` (`Metric`)

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `exact_match` | `float \| None` | Overall exact-match accuracy (`exact_match_count / scored_count`; `None` if `scored_count == 0`). |
| `exact_match_count` | `int` | Number of scored rows achieving `exact_match == 1`. |
| `by_hardness` | `dict[str, dict[str, Any]]` | Keys `"easy"`, `"medium"`, `"hard"`, `"extra"`, `"all"`; each maps to `{"count": int, "exact_match": float \| None, "exact_match_count": int}`. |
| `scored_count` | `int` | Rows contributing to `exact_match`. |
| `parsed_count` | `int` | Predictions passing extraction, char limits, and query-AST parsing. |
| `unparseable_count` | `int` | Predictions failing extraction, char limits, parsing, or query-AST check. |
| `parse_success_rate` | `float` | `parsed_count / num_examples`. |
| `reference_parse_success_rate` | `float` | Fraction of all supplied reference strings passing query-AST parsing. |
| `schema_applied_count` | `int` | Scored rows evaluated with a non-`None` `schema`. |
| `foreign_keys_applied_count` | `int` | Scored rows evaluated with non-empty `foreign_keys`. |
| `num_examples` | `int` | Total example rows processed. |
| `dialect` | `str` | Normalized `sqlglot` dialect used. |
| `ignore_values` | `bool` | Active value-blind flag (`True` by default). |
| `ignore_distinct` | `bool` | Active `DISTINCT`-blind flag (`True` by default). |
| `include_limit_value` | `bool` | Active `LIMIT` value-matching flag (`False` by default). |
| `include_join_conditions` | `bool` | Active `JOIN ... ON` matching flag (`False` by default). |
| `skip_unparseable_predictions` | `bool` | Active unparseable-prediction policy (`False` by default). |
| `per_example` | `list[dict[str, Any] \| None]` | *(Only when `include_per_example=True`)* `None` if skipped; else `{"exact_match": int, "hardness": str, "prediction_parsed": bool, "schema_applied": bool, "foreign_keys_applied": bool}`. Raw SQL and literal values omitted. |

### Reference Benchmark Integration (`examples/bird_eval.py`)

*   Extend `_ddl_to_schema_dict` (or add sibling `_ddl_to_foreign_keys`) in `examples/bird_eval.py` to extract `exp.ForeignKey` pairs from the already-downloaded `baseline_inference/schemas/<dialect>.json` DDL (yielding 104 FK pairs across all 11 SQLite databases in `bird_mini_dev`) and store `foreign_keys` in newly generated JSONL logs.
*   Wire `compute_component_matching` and `compute_exact_matching` into `examples/bird_eval.py score`, passing `schemas=[row["schema"] ...]` and `foreign_keys=[row.get("foreign_keys") ...]` (falling back gracefully to `None` on existing v0.0.1 logs that predate `foreign_keys`), while keeping `examples/` strictly isolated from `src/group_by_vibes/`.

## Alternatives Considered

### Alternative 1: Pure-Python Port of `taoyds/spider/process_sql.py` With a Deterministic Replacement for `nltk.word_tokenize`

*   **Pros:** Keeps `sqlglot` as the sole runtime dependency, runs offline without NLTK data downloads, and preserves `process_sql.py`'s recursive-descent grammar and Spider dict schema.
*   **Cons:** Rejects >90% of `bird_mini_dev` gold queries (`INNER`/`LEFT JOIN`, `SELECT ... AS` column aliases, implicit table aliases, `IS NULL`, quoted identifiers), lacks multi-dialect support, and still risks subtle tokenization divergence from NLTK's `Punkt` + `TreebankWordTokenizer`.
*   **Reason for Rejection:** Rejected by author: standardized on `sqlglot` AST parsing and dialect-general decomposition so `group-by-vibes` maintains multi-dialect support and 100% `bird_mini_dev` gold coverage.

### Alternative 2: Verbatim `taoyds/spider/process_sql.py` Using `nltk.word_tokenize` Behind an Optional `[spider]` Extra

*   **Pros:** Achieves bit-for-bit tokenizer and parser parity with `taoyds/spider` while keeping base installs free of `nltk`.
*   **Cons:** Fragments `load()` so registered evaluators fail on default installs, requires runtime NLTK `punkt`/`punkt_tab` downloads (breaking air-gapped CI), and inherits all `process_sql.py` grammar failures (>90% failure on `bird_mini_dev`).
*   **Reason for Rejection:** Rejected by author: kept all registered metrics working on default `sqlglot`-only installs with zero runtime downloads.

### Alternative 3: Lowering `sqlglot` ASTs Into Spider's Flat Dict Schema (Pure or Extended Hybrid)

*   **Pros:** Reuses `sqlglot` for dialect syntax sugar (`INNER JOIN`, `<>`, quoted identifiers) while preserving Spider's dict tuple representation for `evaluation.py`.
*   **Cons:** Pure lowering cannot represent 376/500 (75.2%) of `bird_mini_dev` gold queries (scalar/cast functions: 324, CTEs: 150, `CASE`: 111, nested/literal arithmetic: 111, windows: 25). Extending Spider's dict with embedded `sqlglot` expression slots mixes two AST models and complicates literal stripping inside `CASE`, `CAST`, and multi-term arithmetic.
*   **Reason for Rejection:** Rejected by author: decomposed `sqlglot` ASTs directly into normalized clauses so 100% of `bird_mini_dev` (500/500) and enterprise SQL queries are scored under one uniform AST pipeline.

### Alternative 4: Bug-for-Bug `taoyds/spider/evaluation.py` Scoring Math (or an Opt-In Bug-Compatibility Toggle)

*   **Pros:** Matches raw `taoyds/spider/evaluation.py` stdout on 0%-matching components and `and/or` mismatches, or lets callers toggle between buggy and fixed math.
*   **Cons:** Reports `f1 = 1.0` when a component has zero matches (`acc == 0 and rec == 0`), swaps `and/or` precision and recall denominators on mismatch, and miscounts `not_op`/`and`/`or` tokens as aggregates during hardness bucketing. Adding a compatibility toggle doubles the test matrix and risks incomparable logged scores.
*   **Reason for Rejection:** Rejected by author: fixed the verified arithmetic, hardness, in-place mutation, and Python 3 subquery-sort defects unconditionally while preserving Yu et al. (2018)'s exact-set clause matching method.

### Alternative 5: Single Combined `SpiderMatch` Evaluator (or Opt-In `include_components` on `ExactMatching`)

*   **Pros:** Parses and decomposes each query pair once when callers want both metrics, reducing CPU work and class count.
*   **Cons:** A single `SpiderMatch` class hides Yu et al. (2018)'s two metric names behind a benchmark-specific name and breaks `group-by-vibes`'s one-metric-per-`load()` convention; adding `include_components=True` on `ExactMatching` creates overlapping APIs with duplicate output schemas.
*   **Reason for Rejection:** Rejected by author: shipped two standalone `Metric` evaluators (`ComponentMatching` and `ExactMatching`) sharing one internal `matching.py` engine, accepting 2x parse work when both are invoked independently in exchange for single-responsibility APIs.

### Alternative 6: Full SQL Clause Fidelity by Default (or Omitting Value-Blind Mode Toggles)

*   **Pros:** Eliminates Spider's default false positives (`WHERE age > 10` matching `WHERE age > 999`, `LIMIT 1` matching `LIMIT 100`, dropped `DISTINCT`, mismatched `JOIN ... ON` keys) out of the box, or reduces constructor surface by removing flags altogether.
*   **Cons:** Changes default scoring semantics away from Yu et al. (2018)'s value-blind structural definition, and removing toggles eliminates Spider-style template comparison entirely.
*   **Reason for Rejection:** Rejected by author: preserved Spider's value-blind defaults (`ignore_values=True`, `ignore_distinct=True`, `include_limit_value=False`, `include_join_conditions=False`) while exposing opt-in strict flags for modern LLM benchmarks.

### Alternative 7: Schema-Free AST-Only Matching or `schema`-Only Without Optional `foreign_keys` (`kmap`)

*   **Pros:** Keeps `add(prediction, reference)` identical to `SqlBleu` (zero schema parameters) or reuses `ValidSqlRate`'s `schema` parameter without introducing `foreign_keys`.
*   **Cons:** Without `schema`, multi-table queries mixing qualified (`t.col`) and unqualified (`col`) columns fail to unify (468/500 `bird_mini_dev` gold queries touch >=2 tables and 228/500 contain unqualified columns). Without `foreign_keys`, equivalent joined keys (`t1.stu_id` vs. `t2.stu_id`) cannot collapse as in Yu et al. (2018), despite 11/11 `bird_mini_dev` DDL schemas declaring foreign keys (104 pairs in SQLite).
*   **Reason for Rejection:** Rejected by author: added optional per-query `schema` and optional `foreign_keys` with AST-only fallback when omitted.

### Alternative 8: Unconditional Skip or Unconditional Penalty on Unparseable Predictions

*   **Pros:** Always skipping mirrors `SqlBleu` uniformly; always penalizing avoids an extra constructor flag.
*   **Cons:** Always skipping introduces survivorship bias into `ExactMatching` (a 1%-parseable model can score `1.0`); always penalizing prevents isolating structural matching on syntax-valid predictions without external pre-filtering.
*   **Reason for Rejection:** Rejected by author: defaulted to Spider's `0` empty-SQL penalty on unparseable predictions while exposing an opt-in `skip_unparseable_predictions: bool = False` flag.

### Alternative 9: Spider's 10 Component Keys Only, Empty-Set `(1, 1, 1)` on `and/or`, Single-Reference Inputs, or First-Reference Hardness Anchoring

*   **Pros:** Keeps Spider's exact 10-key table and single-gold simplicity, or keeps multi-reference hardness `count`s invariant across models.
*   **Cons:** Limiting output to 10 keys leaves `WITH` CTEs (150/500 BIRD queries) and opt-in `JOIN ... ON` predicates (386/500 BIRD equi-joins) invisible in `ComponentMatching`; keeping `(1, 1, 1)` on empty `and/or` sets inflates `and/or` F1 on single-predicate queries; single-reference inputs break `SqlBleu` signature symmetry and reject multi-gold benchmarks; decoupling hardness from the winning reference tallies matches against a secondary `hard` reference inside a `medium` first reference's bucket.
*   **Reason for Rejection:** Rejected by author: extended to 12 component keys (`+ cte`, `+ join_on`), excluded empty `and/or` sets from corpus denominators, and supported `str | Sequence[str]` references where the winning reference determines both scores and hardness.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Security & Privacy:** Never log or emit raw SQL strings or literal values in `per_example` payloads or debug logs (redacted structural diagnostics only, matching `ValidSqlRate` and `parser.py`). Enforce `max_input_chars` pre-extraction and `max_query_chars` pre-parse, catch `(SqlglotError, RecursionError)` on all AST traversals, and validate `schema` and `foreign_keys` eagerly without cross-row global caching.
*   **Performance & Latency:** Aggregate streaming counters use O(1) memory per evaluator; `include_per_example=True` uses O(N) list memory. Multi-reference groups parse and decompose `O(K)` references per row, and invoking `ComponentMatching` and `ExactMatching` separately parses each corpus twice (pure-Python `sqlglot` latency applies; callers needing higher throughput can install `sqlglot[c]` out-of-band or run stateless helpers across worker threads). Shared `BaseEvaluator` instances remain single-writer and non-thread-safe.
*   **Compatibility & Auditability:** Additive post-`v0.0.1` public API with zero breaking changes to `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, or `pyproject.toml` core dependencies (`sqlglot>=30.0.0,<31.0.0`). Every `compute()` payload records `dialect`, `ignore_values`, `ignore_distinct`, `include_limit_value`, `include_join_conditions`, `skip_unparseable_predictions`, `schema_applied_count`, and `foreign_keys_applied_count` so logged benchmark runs cannot be silently mixed across incompatible configurations.
*   **Key Risks & Mitigations:**
    *   *Default value-blind false positives:* Under Spider defaults (`ignore_values=True`, `ignore_distinct=True`, `include_limit_value=False`, `include_join_conditions=False`), wrong filter constants, `LIMIT` counts, missing `DISTINCT`, or wrong `JOIN ... ON` keys still score `1`. *Mitigation:* Document defaults prominently and echo all four flags in every `compute()` output.
    *   *Static equivalence false negatives on BIRD idioms:* Equivalent rewrites (`1.0 * a / b` vs. `CAST(a AS REAL) / b`, `IIF` vs. `CASE WHEN`, `JOIN` vs. `IN (SELECT ...)`) score `0`. *Mitigation:* Position both metrics as structural diagnostics complementing Execution Accuracy (`EX`), and support multi-reference inputs.
    *   *Prediction-dependent hardness counts on heterogeneous multi-reference datasets:* Because the winning reference sets the row's hardness bucket, two models matching different-difficulty references for the same question produce different per-hardness `count`s. *Mitigation:* Document explicitly and keep single-reference datasets (Spider, `bird_mini_dev`) unaffected.

--------------------------------------------------------------------------------

## Additional Resources

*   [Yu et al. (2018), *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task* (arXiv:1809.08887)](https://arxiv.org/abs/1809.08887)
*   Official Spider reference evaluator: [`taoyds/spider`](https://github.com/taoyds/spider) (`evaluation.py`, `process_sql.py`, Apache-2.0) and [`taoyds/test-suite-sql-eval`](https://github.com/taoyds/test-suite-sql-eval)
*   Existing package architecture design: [group_by_vibes.md](./group_by_vibes.md)
*   External benchmark runner: [bird_eval.py](../../examples/bird_eval.py)

## Future Improvements

*   **Execution-Backed Complement:** Pair `ExactMatching` and `ComponentMatching` with optional sandboxed Execution Accuracy (`EX`) / Test Suite Accuracy adapters once DB-execution metrics enter a future release.
*   **Dialect Expression Equivalences:** Optional opt-in AST rewrite normalizers for common dialect idioms (e.g., SQLite `IIF(cond, a, b)` -> `CASE WHEN cond THEN a ELSE b END`, `1.0 * a` vs. `CAST(a AS REAL)`) while keeping default structural matching deterministic and rewrite-free.
*   **Single-Pass Multi-Evaluator Pipeline:** Optional shared-parse helper or batch context allowing callers running `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `ComponentMatching`, and `ExactMatching` together to parse each SQL string once.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Phased Execution Order

1.  **Phase 1 — Shared AST Matching Engine:** Create `src/group_by_vibes/matching.py` implementing query-AST validation, two-layer identifier normalization, eager `foreign_keys` validation and union-find `kmap` construction, shared `coerce_reference_group`, scope-aware alias/schema resolution, wave-by-wave CTE alpha-renaming, mode-flag normalization (`ignore_values`, `ignore_distinct`, `include_limit_value`, `include_join_conditions`), non-literal-LHS predicate canonicalization, 12-component clause decomposition, corrected hardness bucketing, binary exact-set comparison, multi-reference winner selection, and corrected corpus F1 aggregation. *Checkpoint:* `uv run ruff check src/group_by_vibes/matching.py && uv run mypy src/group_by_vibes/ && uv run pytest` exits `0`.
2.  **Phase 2 — Public Evaluators & Exports:** Create `src/group_by_vibes/component_matching.py` and `src/group_by_vibes/exact_matching.py`, then update `src/group_by_vibes/__init__.py` (`__all__`, typed `@overload`s, and widened concrete `load()` signature/dispatch). *Checkpoint:* `uv run ruff check src/group_by_vibes/ && uv run mypy src/group_by_vibes/ && uv run pytest` exits `0`.
3.  **Phase 3 — Unit, Spider-Parity, Bug-Regression & Isolation Tests:** Create `tests/test_matching_and_evaluators.py` and update `tests/test_packaging_isolation.py` (including `"nltk"` and `"sqlparse"` in forbidden imports). *Checkpoint:* `uv run pytest -v --cov=group_by_vibes --cov-branch` passes with `>= 90%` line and `>= 85%` branch coverage on `src/group_by_vibes/` (inspected in the `pytest-cov` report).
4.  **Phase 4 — Reference Benchmark, Sync Script, Docs & CI:** Update `examples/bird_eval.py` (top-level imports, DDL FK extraction, scoring integration), `.agents/skills/sync-examples/scripts/sync_examples.py` (report formatting for `component_matching` and `exact_matching`), regenerate the four score JSON artifacts and comparative tables in `examples/README.md`, update static prose in `examples/README.md` (lines 44, 71, 75, and `### Key Comparative Takeaways`), and update `README.md`, `CHANGELOG.md`, and `.github/workflows/ci.yml`. *Checkpoint:* `sync_examples.py audit`, `sync_examples.py verify`, explicit 500/500 gold-coverage JSON assertion, `ruff`, `mypy`, `pytest`, and isolated wheel build checks all exit `0`.

### Target Files to Create/Modify

#### 1. `src/group_by_vibes/matching.py` `(Create)`

*   **Type / Scope:** Internal Python library module (AST normalization, decomposition, hardness, and matching engine).
*   **Proposed Changes:**
    *   *Constants & Type Aliases:* Define `COMPONENT_TYPES` (the 12 ordered keys), `HARDNESS_LEVELS` (`"easy"`, `"medium"`, `"hard"`, `"extra"`), `ALL_HARDNESS_LEVELS` (`*HARDNESS_LEVELS, "all"`), `ForeignKeyPair = tuple[str, str]`, `ForeignKeysLike = Sequence[ForeignKeyPair]`, `_CanonIdent = tuple[str, bool]`, and `_ForeignKeyMap = dict[tuple[str, str], tuple[_CanonIdent, _CanonIdent]]`.
    *   *Frozen Dataclasses:*
        *   `MatchingConfig(dialect: str, ignore_values: bool = True, ignore_distinct: bool = True, include_limit_value: bool = False, include_join_conditions: bool = False, skip_unparseable_predictions: bool = False)`.
        *   `ComponentScore(precision: int, recall: int, f1: int, label_total: int, pred_total: int)`.
        *   `NormalizedQuery(is_empty: bool, select_distinct: bool, select_items: tuple[str, ...], select_no_agg: tuple[str, ...], from_tables: tuple[str, ...], from_subqueries: tuple[NormalizedQuery, ...], join_on: tuple[str, ...], where_preds: tuple[str, ...], where_no_op: tuple[str, ...], where_and_or: frozenset[str], group_by_no_table: tuple[str, ...], group_by_full: tuple[str, ...], having_preds: tuple[str, ...], having_and_or: frozenset[str], order_by: tuple[tuple[str, str, str], ...], has_limit: bool, limit_value: str | None, iuen_branches: tuple[tuple[str, NormalizedQuery], ...], ctes: tuple[NormalizedQuery, ...], keywords: frozenset[str], hardness: str, canonical_signature: str)`.
        *   `PairMatchResult(exact_match: int, hardness: str, components: dict[str, ComponentScore], prediction_parsed: bool, schema_applied: bool, foreign_keys_applied: bool)`.
    *   *Validation & Coercion Helpers:*
        *   `coerce_reference_group(reference: Any, index: int | None = None) -> list[str]` (shared by both evaluators, mirroring `SqlBleu._coerce_reference_group` semantics without duplicating code or editing `sql_bleu.py`).
        *   `normalize_foreign_keys(foreign_keys: ForeignKeysLike | None, dialect: str) -> _ForeignKeyMap`: returns `{}` for `None` or empty `[]`; rejects `str`/`bytes`/`Mapping` outer values with `TypeError`; requires each entry to be a 2-element non-`str`/non-`bytes`/non-`Mapping` sequence of non-empty `"table.col"` strings (`TypeError`/`ValueError`); splits each endpoint with `dialects.split_qualified_table_parts(endpoint, dialect=dialect)` and rejects `catalog is not None or db is None` with `ValueError`; builds lookup keys via `dialects.normalize_grounding_identifier(..., is_table=True)` for table and `is_table=False` for column; runs deterministic union-find mapping each lookup key to the lexicographically smallest canonical AST representative `((table_name, table_quoted), (col_name, col_quoted))` (unquoted lowercased, quoted case preserved).
    *   *Query Parsing & AST Decomposition:*
        *   `parse_query_ast(raw_sql, *, dialect, max_query_chars, max_input_chars, compiled_extractor, apply_extract)` wrapping `extract.apply_extractor` (when `apply_extract=True`, for predictions only) and `parser.parse_sql`, accepting only `exp.Query` roots (`Select`, `Union`, `Intersect`, `Except`, unwrapping outer `exp.Subquery` parens) and rejecting DDL/DML statements.
        *   `decompose_query_ast(ast, *, dialect, config, norm_schema, kmap)` operates on a deep AST copy under `(SqlglotError, RecursionError)` in deterministic pass order:
            1.  *Wave-by-wave CTE alpha-renaming:* Partition `exp.With` CTEs into Kahn topological waves (binding recursive self-references to `__self__`), normalize each body recursively, preserve CTE column aliases (`WITH c(x, y)`), sort each wave by `(canonical_signature, canonical_column_aliases)`, assign `__cte_0__`, `__cte_1__`, ..., and rewrite downstream references.
            2.  *Left-associative `IUEN` decomposition:* Flatten `Union`/`Intersect`/`Except` chains so the leftmost base `Select` supplies top-level clauses, RHS branches populate `iuen_branches`, and root-level `WITH`/`ORDER BY`/`LIMIT`/`OFFSET` merge onto the outer query.
            3.  *Scope-aware alias, schema, and `kmap` resolution:* Traverse `sqlglot.optimizer.scope.build_scope` using Layer-A lookup keys (`normalize_grounding_identifier` + `grounding.normalize_schema`); rewrite physical aliases (mapping self-joins to physical table name and derived tables to `__subq_0__`... ordered by `canonical_signature`); skip qualifying `GROUP BY`/`HAVING`/`ORDER BY` names matching `SELECT` aliases (`grounding._select_alias_names` semantics); qualify unambiguous unqualified columns via `norm_schema` (or sole physical table when `norm_schema is None` with no CTE/derived sources); apply `kmap` only to tables active in the current scope's `FROM`/`JOIN`; then fold all unquoted `exp.Identifier` nodes via `.lower()` while keeping quoted identifier text unchanged (Layer B).
            4.  *Value & `DISTINCT` normalization:* Strip `SELECT` output aliases; if `ignore_distinct=True`, strip `exp.Distinct` and collapse `*_all` set-op kinds; if `ignore_values=True`, rewrite `COUNT(1)` to `COUNT(*)` and replace non-exempt literals with `exp.Var(this="__val__")` (exempting `exp.DataTypeParam`, `exp.Limit`/`exp.Offset`, `exp.WindowSpec` frame bounds, and positional `GROUP BY`/`ORDER BY` integer ordinals).
            5.  *Recursive subqueries & non-literal-LHS predicate canonicalization:* Replace nested `exp.Query` nodes with `exp.Var(this=f"__subq_({sub_nq.canonical_signature})__")`; run internal `_canonicalize_predicates` (orienting non-literal on `this` and literal-like on `expression` for `EQ`/`NEQ` and flipping ordered inequalities when LHS is literal-like, sorting same-class operands by rendered SQL, sorting subquery-free `IN` lists, flattening/sorting `AND`/`OR` chains while keeping mixed-operator parens; zero coupling to private `canonicalize._rewrite_commutative`).
            6.  *12-component extraction, `canonical_signature`, & corrected hardness:* Extract all 12 immutable component multisets/sets (defaulting omitted `ORDER BY` direction to `"asc"`, extracting `where(no OP)` targets per Technical Plan rules, scanning `keywords` at the current query level only), build `canonical_signature`, and compute hardness via current-level `exp.AggFunc` counts (excluding boolean/connector tokens and nested subqueries) and direct nested-query counts.
        *   `empty_query_decomposition()` returning an empty `NormalizedQuery(is_empty=True, ...)` used for unparseable predictions when `skip_unparseable_predictions=False`.
    *   *Scoring, Multi-Reference Selection & Aggregation:*
        *   `score_pair(pred_decomp, ref_decomp, config)` and `select_best_reference(pred_decomp, ref_decomps, config)`: apply binary exact-set `get_scores` (emitting `ComponentScore(1, 1, 1, 0, 0)` for empty/inactive `and/or` and `join_on`), corrected `and/or` return order, exact-match check across all 12 components plus `from_tables` and recursive `from_subqueries` multiset equality, and deterministic multi-reference tie-breaking maximizing `(exact_match, matched_active_components, all_components_f1_sum, -ref_index)`.
        *   `finalize_component_bucket(acc_sum, acc_count, rec_sum, rec_count)` implementing the bug-fixed zero-match corpus precision/recall/F1 rule (`None` when both counts are `0`; `0.0` when either count `> 0` and `precision + recall == 0`).
*   **Rationale:** Centralizes all dialect-general AST decomposition, upstream defect fixes, and Spider scoring semantics in a single internal module so `ComponentMatching` and `ExactMatching` share identical semantics.

#### 2. `src/group_by_vibes/component_matching.py` `(Create)`

*   **Type / Scope:** Public Python `Metric` evaluator module.
*   **Proposed Changes:** Implement `ComponentMatching(BaseEvaluator)` and stateless `compute_component_matching(...)` matching the signatures and JSON-serializable output schema in the Technical Plan. Validate `prediction`/`predictions`, `reference`/`references` (`str | Sequence[str]` per row, coerced via `matching.coerce_reference_group`), optional `schema`/`schemas`, and optional `foreign_keys` eagerly before mutating state: raise `TypeError` on non-`str` `prediction` when `extractor is None`, and record it as unparseable via `apply_extractor` when `extractor is not None` (matching `SqlBleu`/`ValidSqlRate`/`WildcardDetector`). Maintain O(1) counters per `(level, component)` plus parse/reference/schema/FK counts and opt-in O(N) `per_example` records (`{"precision": 1, "recall": 1, "f1": 1, "label_total": 0, "pred_total": 0}` for empty/inactive components; raw SQL and literals omitted).
*   **Rationale:** Exposes clause-level precision, recall, and F1 diagnostics across all 12 component types and 5 hardness buckets (`easy`, `medium`, `hard`, `extra`, `all`) with an `evaluate`-shaped streaming interface.

#### 3. `src/group_by_vibes/exact_matching.py` `(Create)`

*   **Type / Scope:** Public Python `Metric` evaluator module.
*   **Proposed Changes:** Implement `ExactMatching(BaseEvaluator)` and stateless `compute_exact_matching(...)` matching the signatures and JSON-serializable output schema in the Technical Plan. Share `matching.py` decomposition, `coerce_reference_group`, conditional `prediction` validation (`TypeError` when `extractor is None`; unparseable via `apply_extractor` when `extractor is not None`), unparseable-prediction handling, and multi-reference winner selection while maintaining O(1) exact-match counters per hardness bucket plus opt-in O(N) `per_example` records.
*   **Rationale:** Exposes order-insensitive structural exact match accuracy as a standalone `Metric` aligned with Yu et al. (2018) and `group-by-vibes`'s one-metric-per-`load()` convention.

#### 4. [__init__.py](../../src/group_by_vibes/__init__.py) `(Modify)`

*   **Type / Scope:** Package entrypoint and public exports.
*   **Proposed Changes:** Import and re-export `ComponentMatching`, `ExactMatching`, `compute_component_matching`, `compute_exact_matching`, `ForeignKeyPair`, and `ForeignKeysLike` in `__all__`. Add `@overload` definitions for `load(name: Literal["component_matching"], **kwargs: Any) -> ComponentMatching` and `load(name: Literal["exact_matching"], **kwargs: Any) -> ExactMatching`, widen the concrete `load()` signature (`name: Literal["sql_bleu", "valid_sql_rate", "wildcard_detector", "component_matching", "exact_matching"] | str` and return union `SqlBleu | ValidSqlRate | WildcardDetector | ComponentMatching | ExactMatching` for `mypy strict = true`), extend dispatch, and update its unknown-evaluator `ValueError` message.
*   **Rationale:** Registers both new metrics in the public API and `load()` registry without altering existing exports.

#### 5. `tests/test_matching_and_evaluators.py` `(Create)`

*   **Type / Scope:** Unit, Spider-parity, regression, and multi-dialect test suite (`pytest`).
*   **Proposed Changes:** Add tests covering:
    *   All 12 components (`select`, `select(no AGG)`, `where`, `where(no OP)`, `group(no Having)`, `group`, `order`, `and/or`, `IUEN`, `keywords`, `cte`, `join_on`) and `ExactMatching` (`from_tables` and recursive `from_subqueries` multiset equality).
    *   Static Spider-subset parity fixtures matching `taoyds/spider/evaluation.py` on non-buggy SQL pairs (order-insensitive clauses, `select(no AGG)`, `where(no OP)`, `order` + `LIMIT` presence, `IUEN` left-branch/RHS decomposition, `keywords`, FK `kmap` equivalence, and 4 hardness levels).
    *   Regression tests for all five addressed upstream defects: zero-match corpus F1 (`0.0` vs. `None`), `eval_and_or` non-swapped denominators and empty-set `(0, 0, 0)` exclusion (including unparseable empty fallback), non-mutation of caller `schema`/`foreign_keys` inputs and reused reference decompositions, >=2 `FROM` subqueries without `TypeError`, and corrected `count_others` hardness aggregate counting (`NOT` in `WHERE` and `AND`/`OR` in `HAVING`).
    *   Multi-dialect SQL constructs (`PostgreSQL`, `SQLite`, `MySQL`, `Snowflake`, `BigQuery`): two-layer identifier normalization (SQLite quoted vs. backticked case preservation, Snowflake unquoted uppercase lookup vs. lowercase AST, BigQuery table vs. column casing), non-literal-LHS orientation (`'S' = s.rtype` vs. `s.rtype = 'S'`, `10 < age` vs. `age > 10`, `WHERE a = 1` vs. `WHERE b = 2` non-collision on `where(no OP)`), wave-by-wave CTE alpha-renaming (`WITH a` vs. `WITH b`, multi-wave dependencies, CTE column aliases), `CASE WHEN`, `CAST` type-param exemption, `COUNT(1)` vs. `COUNT(*)`, `ROUND`/`COALESCE`/`SUBSTR`, window functions (`OVER`) with frame-bound exemption, nested/literal arithmetic (`1.0 * a / b`), `INNER`/`LEFT` joins, `<>` vs. `!=`, `SELECT` alias preservation in `ORDER BY` under `schema=None`, self-joins, and order-insensitive nested `WHERE` subqueries.
    *   All five mode flags (`ignore_values`, `ignore_distinct`, `include_limit_value`, `include_join_conditions`, `skip_unparseable_predictions`) and verification that active flag values are recorded in `compute()` output.
    *   Optional `schema` qualification, optional `foreign_keys` (`[]` accepted; 1-part/3-part rejected; transitive union-find collapse), AST-only fallback when omitted, conditional non-`str` `prediction` handling (`TypeError` vs. `extractor` unparseable), eager validation errors on malformed `schema`/`foreign_keys`/batch length mismatches, multi-reference `str | Sequence[str]` deterministic tie-breaking and hardness selection, zero-valid-reference skipping, non-query DDL/DML rejection, `RecursionError` handling, `per_example` `(1, 1, 1)` inactive-component values and SQL/literal redaction, empty `compute()` `ValueError`, and post-`compute()` reset.
    *   End-to-end fixture test over [bird_sample.jsonl](../../examples/fixtures/bird_sample.jsonl) verifying 5/5 gold query decomposition (including literal arithmetic and `'S' = s.rtype` in row 0, and CTE in row 2), plus targeted unit tests for `examples/bird_eval.py`'s `_ddl_to_foreign_keys` (table-level, inline column `REFERENCES`, composite FKs, omitted target column lists, quoted identifiers with spaces) and `sync_examples.py`'s `_render_component_matching_section` / `None`-safe rendering.
*   **Rationale:** Verifies scoring logic, Spider-subset parity, upstream defect regressions, dialect coverage, helper parsing/rendering, and API invariants across both evaluators and `matching.py`.

#### 6. [test_packaging_isolation.py](../../tests/test_packaging_isolation.py) `(Modify)`

*   **Type / Scope:** Packaging isolation, `load()`, bounds, and concurrency tests.
*   **Proposed Changes:** Add `"nltk"` and `"sqlparse"` to the forbidden `sys.modules` set in the fresh-process isolation check and invoke `compute_component_matching` and `compute_exact_matching` inside that subprocess script. Add `load("component_matching")` / `load("exact_matching")` assertions, `max_query_chars`/`max_input_chars` `ValueError` checks, empty/argument `compute()` guards, and `ThreadPoolExecutor` stateless concurrency checks for both new evaluators.
*   **Rationale:** Guards the `sqlglot`-only zero-download core invariant and thread-safe stateless contract.

#### 7. [bird_eval.py](../../examples/bird_eval.py) `(Modify)`

*   **Type / Scope:** Isolated external reference benchmark CLI.
*   **Proposed Changes:**
    *   Import `compute_component_matching`, `compute_exact_matching`, and `dialects.primary_identifier_delimiter` (or `primary_identifier_delimiter` helper) at module top level alongside `compute_sql_bleu`, `compute_valid_sql_rate`, and `compute_wildcard_detector` (keeping ML imports lazy inside `run_generate`).
    *   Add `_ddl_to_foreign_keys(ddl: str, dialect: str) -> list[list[str]]` parsing `CREATE TABLE` DDL via `sqlglot.parse(ddl, read=dialect)` from the `baseline_inference/schemas/<dialect>.json` DDL already fetched during `run_generate`: extracts both table-level `exp.ForeignKey` nodes (pairing `node.expressions` positionally with `ref.this.expressions` when `ref.this` is `exp.Schema`, skipping bare `REFERENCES parent` without column lists) and inline column-level `exp.ColumnConstraint` nodes where `kind` is `exp.Reference`; delimits table/column identifiers with the dialect's primary identifier delimiter whenever a name contains `.`, whitespace, or quote characters; deduplicates and sorts pairs deterministically; writes `"foreign_keys"` into newly generated JSONL rows in `run_generate`.
    *   In `run_score`, convert `[row.get("foreign_keys") for row in rows]` (converting each non-`None` `[a, b]` list to `(a, b)` tuple, passing `None` when absent on older logs), call `compute_component_matching` and `compute_exact_matching` with `schemas=schemas`, `foreign_keys=foreign_keys_arg`, `dialect=dialect`, `extractor=extract_fenced_or_last_sql`, `include_per_example=True` (using default Spider value-blind flags), emit top-level `"component_matching"` and `"exact_matching"` sections, and add `"exact_match"` and `"exact_match_count"` to each `by_difficulty[diff]` dict (computed via `compute_exact_matching` over that slice with `extractor=extract_fenced_or_last_sql`). Keep zero ML imports inside `run_score`.
*   **Rationale:** Satisfies `sync_examples.py audit` (every exported `compute_*` evaluator wired into `bird_eval.py`) and enables offline full-split scoring of `bird_mini_dev`.

#### 8. [sync_examples.py](../../.agents/skills/sync-examples/scripts/sync_examples.py) `(Modify)`

*   **Type / Scope:** Benchmark report synchronization and CI verification script.
*   **Proposed Changes:**
    *   Add `"component_matching"` and `"exact_matching"` to `BUILTIN_EVAL_KEYS`.
    *   In `_render_headline_section`: add **`**Exact Match Accuracy**`** formatted as `` `XX.X%` (`matched/scored`) `` when `exact_match is not None` else `` `N/A` (`0/0`) `` (with 2-run delta `_fmt_delta_pp` + `(_fmt_delta_int)` when both are non-`None`, else `""`).
    *   In `_render_difficulty_section`: append `{label} Exact Match` column(s) (plus `Delta Exact Match` on 2-run cohorts), sourcing `exact_match` and `exact_match_count` from `by_difficulty[diff]` and formatting `None` as `` `N/A` (`0`) ``.
    *   Add `_render_component_matching_section(runs, delta_tag)` inserted in `_render_cohort_report` immediately after `_render_headline_section` with heading `### Component Matching F1`, columns `["Component", *model_headers]` (+ `Delta ({delta_tag})` on 2-run cohorts), iterating the 12 `COMPONENT_TYPES` keys and keeping only keys where at least one run has `f1 is not None`, formatting cells as `` `0.XXX` `` (or `` `N/A` `` when `None`) and 2-run deltas via `_fmt_delta_float(..., 3)` (or `""` when either is `None`). Keep `FORBIDDEN_README_SUBSTRINGS` and `BANNED_PROMO_WORDS` checks intact.
*   **Rationale:** Prevents scalar-only fallback rows (which omit nested `components` and `by_hardness` tables) and keeps `.github/workflows/ci.yml`'s `sync_examples.py verify` check passing.

#### 9. [bird_sample_scores.json](../../examples/logs/bird_sample_scores.json), `examples/logs/bird_qwen_3b_scores.json`, `examples/logs/bird_xiyansql_3b_scores.json`, `examples/logs/bird_finer_sql_3b_scores.json`, and [examples/README.md](../../examples/README.md) `(Modify — Generated & Refreshed)`

*   **Type / Scope:** Exact four benchmark score JSON files and comparative benchmark report.
*   **Proposed Changes:** Run `uv run python .agents/skills/sync-examples/scripts/sync_examples.py run --update-readme` to regenerate all four score JSON files and auto-generated Markdown tables with `component_matching` and `exact_matching`. Then update static prose in `examples/README.md` outside `_update_readme_markdown`'s replacement range—Line 44 (`**Schema Mapping:**` noting `foreign_keys` DDL extraction alongside `{table: [columns]}`), Line 71 (`### 2. Offline Re-Scoring` adding `ComponentMatching` and `ExactMatching`), Line 75 (`## Saved JSONL Schema` adding `foreign_keys`), and `### Key Comparative Takeaways`—while preserving all report hygiene rules.
*   **Rationale:** Keeps checked-in example score files, comparative tables, and static benchmark documentation synchronized with the package's exported evaluators for CI verification.

#### 10. [README.md](../../README.md) `(Modify)`

*   **Type / Scope:** Top-level package documentation.
*   **Proposed Changes:** Add `ComponentMatching` (`load("component_matching")`) and `ExactMatching` (`load("exact_matching")`) rows to the Available Evaluators table, update the stateful evaluator list in `### Thread Safety` (`README.md:136`) to include `ComponentMatching` and `ExactMatching`, add a Usage Patterns snippet demonstrating Spider value-blind defaults, opt-in strict flags, and optional `schema`/`foreign_keys`, and add the Yu et al. (2018) citation under References explicitly noting the dialect-general, bug-fixed `sqlglot` AST adaptation.
*   **Rationale:** Documents the new evaluators, thread-safety contract, default semantics, and intentional divergences from `taoyds/spider` at the package entry point.

#### 11. [CHANGELOG.md](../../CHANGELOG.md) `(Modify)`

*   **Type / Scope:** Release notes.
*   **Proposed Changes:** Add `ComponentMatching`, `ExactMatching`, shared AST matching engine, DDL foreign-key extraction in `examples/bird_eval.py`, and `sync-examples` reporting updates under `## [Unreleased]`.
*   **Rationale:** Tracks post-`v0.0.1` additions under Keep a Changelog format.

#### 12. [ci.yml](../../.github/workflows/ci.yml) `(Modify)`

*   **Type / Scope:** GitHub Actions CI workflow.
*   **Proposed Changes:** Extend the isolated wheel verification snippet in the `build` job to assert: `assert isinstance(gbv.load("component_matching"), gbv.ComponentMatching)`, `assert isinstance(gbv.load("exact_matching"), gbv.ExactMatching)`, `assert gbv.compute_exact_matching(["SELECT a, b FROM t"], ["SELECT b, a FROM t"], dialect="postgres")["exact_match"] == 1.0`, and `assert gbv.compute_component_matching(["SELECT a FROM t"], ["SELECT a FROM t"], dialect="postgres")["components"]["select"]["f1"] == 1.0`.
*   **Rationale:** Verifies end-to-end packaging of the new modules in CI with `sqlglot` as the sole installed dependency.

### Verification Plan

*   **Unit, Parity & Regression Tests:** `uv run pytest -v --cov=group_by_vibes --cov-branch` exits `0` across all suites including `tests/test_matching_and_evaluators.py` and `tests/test_packaging_isolation.py`.
*   **Test Coverage Requirement:** `>= 90%` line coverage and `>= 85%` branch coverage on `src/group_by_vibes/` (to be inspected in the `pytest-cov` report across `src/group_by_vibes/` and individually on `matching.py`, `component_matching.py`, and `exact_matching.py`, as `pyproject.toml` defines no `fail_under` threshold).
*   **Static Analysis & Linting:** `uv run ruff check src/ tests/ examples/ .agents/skills/sync-examples/scripts/` exits `0`, and `uv run mypy src/group_by_vibes/` (`strict = true`) exits `0`.
*   **Compilation / Packaging & Dependency Isolation:** `uv build`, inspect built wheel `METADATA` via `zipfile` to confirm `Requires-Dist` lists only `sqlglot>=30.0.0,<31.0.0` (zero `nltk` or `sqlparse`), and run the isolated wheel smoke test.
*   **E2E Benchmark, Sync & Gold-Coverage Verification:**
    *   `uv run python examples/bird_eval.py score --input examples/fixtures/bird_sample.jsonl > /dev/null` exits `0`.
    *   `uv run python .agents/skills/sync-examples/scripts/sync_examples.py audit` exits `0` (`unwired_evaluators == []`).
    *   `uv run python .agents/skills/sync-examples/scripts/sync_examples.py verify` exits `0` (confirming report freshness, evaluator wiring, and `examples/README.md` hygiene).
    *   Explicit JSON coverage check via `uv run python -c` asserting `reference_parse_success_rate == 1.0` and `scored_count == 500` for both `component_matching` and `exact_matching` across `examples/logs/bird_qwen_3b_scores.json`, `examples/logs/bird_xiyansql_3b_scores.json`, and `examples/logs/bird_finer_sql_3b_scores.json`.

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
