# group-by-vibes: Standalone Text2SQL Evaluation Package

*   **Author:** Felipe Campos Penha
*   **Status:** Proposed
*   **Date:** 2026-11-23

---

## Overview

`group-by-vibes` is a standalone, `sqlglot`-only Python package for AI Engineers evaluating Text2SQL Large Language Models (LLMs). While v1 prioritizes PostgreSQL and SQLite in docs and tests, it accepts any SQL dialect resolved by `sqlglot.Dialect.get_or_raise`. It provides deterministic, SQL-aware evaluation both when golden SQL references exist (benchmarks or validated prior runs) and when evaluating unlabelled runtime outputs.

Following Hugging Face `evaluate` terminology, v1 organizes three dialect-dependent evaluators inside `src/group_by_vibes/`—one `Metric` and two `Measurement`s:

*   **Metric (`SQL-BLEU`, reference-based):** Adapted from [sql_bleu.py](../../src/sql_bleu.py); canonicalizes parsed Abstract Syntax Trees (ASTs) and tokenizes syntactically under the chosen dialect, skipping rows whose prediction fails to parse or where zero supplied references parse while reporting corpus scores and parse-rate counts.
    > **Reference:** [Chen et al. (2019), NL2pSQL](https://aclanthology.org/D19-1262/) — Fuxiang Chen, Seung-won Hwang, Jaegul Choo, Jung-Woo Ha, and Sunghun Kim. *NL2pSQL: Generating Pseudo-SQL Queries from Under-Specified Natural Language Questions*. EMNLP-IJCNLP 2019, pages 2603–2613. DOI: `10.18653/v1/D19-1262`.
*   **Measurement (`Valid SQL Rate`, reference-free):** Performs lenient, dialect-aware `sqlglot` AST parsing (rejecting empty inputs, multi-statement inputs, and `exp.Command` nodes—`sqlglot`'s unparsed fallback AST representation) and optionally checks table/column grounding via `sqlglot.optimizer.scope.build_scope` against per-query Python `dict` schemas, without executing SQL, parsing `CREATE TABLE` DDL, or type checking.
*   **Measurement (`Wildcard Detector`, reference-free):** Scans parsed query ASTs and outputs `1` when `SELECT *` or `table.*` projections appear, `0` otherwise (excluding `COUNT(*)`, `EXISTS (SELECT * ...)`, and `*` arithmetic), and `None` on unparseable rows.

All three evaluators expose a synchronous `evaluate`-shaped `add()` / `add_batch()` / `compute()` / `reset()` lifecycle (`prediction`/`reference` and `"translation_length"` on `SqlBleu`; `data=` on Measurements; returning JSON-serializable `dict`s and resetting state after `compute()`) layered over stateless helper functions, share an optional regex/callable SQL `extractor`, and carry zero dependency on `evaluate`, `datasets`, or `pyarrow`. Benchmark inference on `birdsql/bird_mini_dev` (with local 3B models such as `Qwen/Qwen2.5-Coder-3B-Instruct`, `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`, or `griffith-bigdata/FINER-SQL-3B-v2`) lives outside core in `examples/`.

Because v1 executes no queries against a database engine, these evaluators act as deterministic pre-execution guardrails rather than substitutes for execution accuracy (`EX`).

## The Problem

AI Engineers evaluating Text2SQL LLMs face two gaps in standard tooling:

1.  **Text metrics fail on SQL equivalence and syntax:** Hugging Face `evaluate`'s `bleu` tokenizes SQL as natural-language text, penalizing keyword casing, whitespace, comments, semicolons, and commutative predicate order (`AND`/`OR`, `=`/`!=`, `IN (...)`). Meanwhile, [sql_bleu.py](../../src/sql_bleu.py) remains unpackaged ([README.md](../../README.md) still references non-existent `utils/sql_bleu.py` paths), carries an optional `evaluate` backend loader, omits reference parse rates, and contains AST mutation and string-quoting bugs.
2.  **Unlabelled outputs lack DB-free checks:** Standard metrics offer no `sqlglot` AST check, no per-query schema grounding check, and no AST filter distinguishing projection wildcards (`SELECT *`, `table.*`) from `COUNT(*)` or `*` arithmetic. Running untrusted queries on live databases (`EX`, `EXPLAIN`, `DRY RUN`) adds latency and safety exposure, while dataset libraries (`datasets`, `pyarrow`) and model stacks pull dependencies unsuited to a scoring package. LLM generations under prompts such as `bird_mini_dev` (fenced ```` ```sql ```` blocks) or reasoning traces (`<think>...</think>`) also require optional extraction before parsing.

`group-by-vibes` addresses both gaps in one `sqlglot`-only package using an `evaluate`-shaped API without loading models, datasets, or database drivers.

## Goals

*   **Standalone Core (`src/group_by_vibes/`):** Ship `pyproject.toml` with `sqlglot>=30.0.0,<31.0.0` as the sole runtime dependency, migrating [sql_bleu.py](../../src/sql_bleu.py), retiring the `evaluate` backend loader, and fixing stale `utils/` paths in [README.md](../../README.md).
*   **Reference-Based `SQL-BLEU`:** Compute canonicalized AST corpus BLEU over valid pairs (skipping invalid prediction or all-invalid reference rows, tracking prediction/reference parse rates, `prediction`/`reference` inputs, `"translation_length"` key).
*   **Reference-Free `Valid SQL Rate`:** Score lenient dialect-aware `sqlglot` AST parsing and optional per-query `dict` grounding via `build_scope` (`data=` inputs, skipped vs. `{}` empty catalogs, `schema_grounded_rate`, and single-reason `failure_counts`).
*   **Reference-Free `Wildcard Detector`:** Score projection wildcards (`1` / `0` / `None`, `data=` inputs) over parsed queries, reporting `wildcard_rate`.
*   **All `sqlglot` Dialects Accepted:** Accept any dialect supported by `sqlglot` (aliases and ANSI `""` included), prioritizing PostgreSQL and SQLite in docs, examples, and tests.
*   **Streaming & Stateless Parity:** Provide synchronous O(1)-counter accumulators (`add` / `add_batch` / `compute` / `reset`, opt-in `include_per_example` in O(N) list memory, post-compute reset) over stateless `compute_*` / `sentence_sql_bleu` helpers.
*   **Shared Extraction & Bounded Parsing:** Share an optional regex/callable `extractor`, fix the detached-node `And`/`Or` bug in [sql_bleu.py](../../src/sql_bleu.py), enforce input/query length bounds, catch parser/recursion errors, and omit raw SQL query literals from logs.
*   **External Reference Benchmark (`examples/`):** Ship an isolated CLI evaluating `birdsql/bird_mini_dev` (SQLite, PostgreSQL, MySQL) with local 3B models (`Qwen/Qwen2.5-Coder-3B-Instruct`, `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`, `griffith-bigdata/FINER-SQL-3B-v2`) and saving JSONL logs for zero-ML offline re-scoring.

## Non-Goals

*   ISO SQL:2016 EBNF (`sql-2016.ebnf`) Context-Free Grammar (CFG) conformance and prefix metrics (`Raw SQL:2016 Context-Free Grammar Conformance`, `Valid Context-Free Grammar Prefix`, `Extracted Context-Free Grammar Conformance`).
*   Custom Apache Impala dialect support in v1 (`sqlglot` ships no built-in `impala` dialect; PostgreSQL and SQLite are the v1 priority).
*   Database-executed metrics (`BIRD Execution accuracy (EX)`, `SQLite Executable` — executing SQL against SQLite and checking for zero runtime errors, `EXPLAIN`, `DRY RUN`) or DB connection management in v1.
*   Static data-type checking (column, literal, or aggregate type inference without a planner).
*   `CREATE TABLE` DDL string parsing in v1 (Python `dict` schemas only).
*   Backward-compatibility shims for legacy `utils/` paths, `backend="evaluate"`, the `SqlBleuResult` dataclass (replaced by JSON-serializable `dict` outputs), or [sql_bleu.py](../../src/sql_bleu.py)'s `main()` CLI entrypoint (the core package ships a Python library API only).
*   Multi-statement SQL evaluation (queries parsed by v1 must be single statements).
*   Runtime dependency on or subclassing of Hugging Face `evaluate`, `datasets`, or `pyarrow`.
*   Built-in LLM inference, weight loading, or dataset downloading inside the core package.
*   LLM-as-a-judge scoring, `SQL Component F1` (clause-level keyword/identifier F1 matching), or additional metrics in v1 beyond `SQL-BLEU`, `Valid SQL Rate`, and `Wildcard Detector`.
*   Native `async def` wrappers around CPU-bound parsing (callers run evaluation asynchronously out-of-band).
*   Custom `offline` / `online` naming (replaced by Hugging Face `evaluate`'s `Metric` / `Measurement` taxonomy).

## Open Questions

None — all initial design questions have been resolved and incorporated into the Technical Plan below.

## Critical User Journeys (CUJs)

### CUJ 1: Evaluate a Fine-Tuned Model Against a Golden Set (`SQL-BLEU`, `Valid SQL Rate`, and `Wildcard Detector`)

*   **Given:** An AI Engineer has generated SQL predictions, golden SQL references, and optional per-query `dict` schemas for a `sqlglot` dialect, and initializes `SqlBleu`, `ValidSqlRate`, and `WildcardDetector`.
*   **When:** The engineer passes predictions/references into `SqlBleu` (`add`/`add_batch`) and predictions (`data=`) plus optional `schemas=` into `ValidSqlRate` and `WildcardDetector`, then calls `compute()`.
*   **Then:** `SqlBleu` canonicalizes parsed ASTs, skips rows with failed predictions or zero valid references, computes corpus BLEU over `scored_count` pairs, resets accumulators after `compute()` succeeds, and returns the `SqlBleu` result `dict`. Without reading golden SQL, `ValidSqlRate` scores lenient parsing/grounding (`valid_sql_rate`, `schema_grounded_rate`, `failure_counts`) and `WildcardDetector` returns `wildcard_rate` (plus opt-in per-example `0`/`1`/`None` flags when `include_per_example=True`).

### CUJ 2: Screen Unlabelled Queries Out-of-Band at Runtime (`Valid SQL Rate` and `Wildcard Detector`)

*   **Given:** An AI Engineer evaluates unlabelled runtime SQL—with per-query `dict` schemas or `schema=None` (parse-only)—without blocking a live agent loop.
*   **When:** The host application dispatches `data=` and optional `schemas=` asynchronously out-of-band using per-worker instances, a serialized aggregation queue, or stateless `compute_valid_sql_rate` / `compute_wildcard_detector` calls (since shared `BaseEvaluator` instances are single-writer and not thread-safe).
*   **Then:** `group-by-vibes` runs lenient dialect-aware `sqlglot` AST parsing, verifies query table/column grounding via `build_scope` on syntax-valid query ASTs with non-`None` schemas (skipping `schema=None` and non-query ASTs), checks projection wildcards, and returns JSON-serializable `dict`s covering `valid_sql_rate`, `syntax_valid_rate`, `schema_grounded_rate` (`None` if `schema_evaluated_count == 0`), `wildcard_rate`, and all eight `failure_counts` categories.

### CUJ 3: Score Reasoning-Model Outputs on the External Reference Benchmark

*   **Given:** An AI Engineer runs `examples/bird_eval.py` with a local 3B model (such as `Qwen/Qwen2.5-Coder-3B-Instruct`, `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`, or `griffith-bigdata/FINER-SQL-3B-v2`) on `birdsql/bird_mini_dev` (500 queries across 11 relational databases in `"sqlite"`, `"postgres"`, or `"mysql"`), where outputs may contain ```` ```sql ```` fences or optional `<think>...</think>` reasoning blocks.
*   **When:** The script parses `baseline_inference/schemas/<dialect>.json` DDL into per-database `dict` schemas, logs completions and schemas to JSONL, and scores outputs across all three evaluators sharing `extract_fenced_or_last_sql`.
*   **Then:** All three evaluators process the identical extracted SQL string per row without importing ML packages into core, enabling reproducible, auditable offline re-scoring across package releases.

## Technical Plan (Architecture)

### High-Level Data Flow

```text
  Caller Inputs (raw SQL / completions, dialect, optional schema dicts & golden refs)
         |
         v
  +-----------------------------------------------------------------------------------+
  | 1. Optional Shared SQL Extractor (`extract.py`)                                   |
  |    - Runs when `extractor is not None`; rejects inputs > `max_input_chars` first  |
  |    - Failures isolated with `failure_reason` (`extraction_failed` / `..._error`)  |
  +-----------------------------------------------------------------------------------+
         |  SQL string (raw input if `extractor=None`, else extracted SQL)
         v
  +-----------------------------------------------------------------------------------+
  | 2. Isolated sqlglot Parser Adapter (`parser.py` + `dialects.py`)                  |
  |    - Dialect lookup via `sqlglot.Dialect.get_or_raise` (plus aliases & ANSI "")   |
  |    - Rejects > `max_query_chars`, empty input, multi-statement, and `exp.Command` |
  |    - Owns `sqlglot.parse` and `tokenize_sql`; `(SqlglotError, RecursionError)`    |
  +-----------------------------------------------------------------------------------+
         |
         +------------------------------+------------------------------+
         v                              v                              v
  +--------------------+     +--------------------+     +-----------------------------+
  | SqlBleu (`Metric`) |     | ValidSqlRate       |     | WildcardDetector            |
  | `sql_bleu.py` +    |     | (`Measurement`)    |     | (`Measurement`)             |
  | `canonicalize.py`  |     | `valid_sql_rate.py`|     | `wildcard_detector.py`      |
  | - Skip failed pred |     | + `grounding.py`   |     | - Projection star scan      |
  |   or 0 valid refs  |     | - Query `build_    |     | - Excludes `COUNT(*)`,      |
  | - `scored_count`   |     |   scope` traversal |     |   `EXISTS`, `*` arithmetic  |
  | - No eval backend  |     | - Per-query `dict` |     | - Skip unparseable (`None`) |
  |                    |     |   grounding        |     | - `wildcard_rate` over ASTs |
  +--------------------+     +--------------------+     +-----------------------------+
         |                              |                              |
         +------------------------------+------------------------------+
                                        v
  +-----------------------------------------------------------------------------------+
  | 3. Streaming Accumulators (`base.py`)                                             |
  |    - Single-writer `add()` / `add_batch()` / `compute()` / `reset()`              |
  |    - Resets counters after `compute()` succeeds; opt-in `include_per_example`     |
  +-----------------------------------------------------------------------------------+
                                        |
                                        v
                          JSON-Serializable `dict` Outputs

  -------------------------------------------------------------------------------------
  External Reference Boundary (`examples/bird_eval.py`, outside core pkg):
  [Local 3B SQL models + bird_mini_dev] -> JSONL log -> core API
```

### Package Architecture & Module Boundaries (`src/group_by_vibes/`)

1.  **`dialects.py` — Dialect Validation & Identifiers:** `normalize_dialect` checks `isinstance(dialect, str)`, strips/lowercases input, maps `"postgresql"`->`"postgres"` and `"sqlite3"`->`"sqlite"`, keeps ANSI `""`, and validates via `sqlglot.Dialect.get_or_raise` (`ValueError` otherwise). `split_qualified_table` parses 1–3 part names across dialect quotes inside `dialects.py` without `sqlglot.parse_one` (handling hyphenated keys like `"1-29135051-2"`). `normalize_grounding_identifier` applies case-insensitive `.lower()` on `sqlite` regardless of quotes; non-SQLite dialects keep quoted case exact and fold unquoted names via `sqlglot` `NORMALIZATION_STRATEGY` (lowercase on `postgres`/`duckdb`/`bigquery`, uppercase on `snowflake`/`oracle`, case-insensitive on `mysql`/`tsql`).
2.  **`extract.py` — Shared Optional Extractor:** Compiles regexes eagerly (`re.error` on invalid syntax, `ValueError` on `> 1` capture group). Runs `apply_extractor` only when `extractor is not None`, checking `isinstance(raw_text, str)` and `len(raw_text) <= max_input_chars` (`None` disables) before execution (`None`/blank/oversized -> `extraction_failed`; non-`str` or callable exception except `KeyboardInterrupt`/`SystemExit` -> `extraction_error`). When `extractor=None`, inputs route directly to `parser.py` (`max_query_chars`). Exports `extract_fenced_or_last_sql(text: str) -> str | None`: rejects unclosed `<think>`, strips `<think>...</think>`, prefers last ```` ```sql ```` fence, else last SQL-keyword fence, else slices standalone `WITH`/`SELECT` to `;` or blank-line break, omitting trailing prose.
3.  **`parser.py` — `sqlglot` Boundary:** Sole caller of `sqlglot.parse` and `sqlglot.tokenize` (`sqlglot>=30.0.0,<31.0.0`). `parse_sql` rejects `len(sql) > max_query_chars` (`syntax_error`), blank inputs (`empty_input`), >1 statement after stripping trailing `;` `None` items (`multi_statement`), and any `exp.Command` node (`command_fallback`), catching `(SqlglotError, RecursionError)` (`syntax_error`) and logging SQL-redacted diagnostics with zero parse cache. `tokenize_sql` owns `sqlglot.tokenize`, re-quotes `TokenType.STRING` (`"'" + tok.text.replace("'", "''") + "'"`), keeps quoted identifier delimiters, filters `;` (`include_semicolons`) and comments (`strip_comments`), and falls back under `(SqlglotError, RecursionError)` to linear quote-aware regex tokenization with dialect-scoped `[...]` brackets.
4.  **`canonicalize.py` — AST Canonicalization & Tokenizer Wrapper:** Normalizes ASTs without `sqlglot.optimizer`: keyword casing (`normalize=True`), comment/semicolon flags (`strip_comments`, `include_semicolons`), deterministic `.lower()` on all unquoted identifiers (`if not ident.quoted` across all dialects, keeping quoted case intact), and optional commutative sorting (`=`, `!=`, subquery-free `exp.In` where `query is None`, flattened `AND`/`OR` chains preserving `_LOGICAL_CLASSES` precedence). Post-order recursive transform returning updated root (fixing detached chains and bare roots), catches `(SqlglotError, RecursionError)` -> `("", False)`, and delegates `sql_tokenizer` to `parser.tokenize_sql`.
5.  **`grounding.py` — Static Query Schema Grounding (`build_scope`):** Validates `None` vs. `Mapping[str, Iterable[str]]` eagerly (`TypeError`/`ValueError` on `str`/`bytes` column values or non-`str` entries; zero global cache). Skips non-query ASTs where `build_scope(ast) is None` (`schema_valid = None`). Traverses `scope.traverse()` + `scope.external_columns` under `(SqlglotError, RecursionError)`, resolving physical tables, CTEs, CTE/derived `exp.TableAlias.columns`, aliases, `SELECT` aliases in `GROUP BY`/`HAVING`/`ORDER BY`, and SQLite implicit columns (`rowid`, `_rowid_`, `oid` when `dialect == "sqlite"`). Treats `SELECT *` CTEs, `VALUES`, `UNNEST`, TVFs, and column-less derived tables as opaque (reducing false positives: unqualified columns pass in mixed opaque+physical scopes; `alias.*` checks qualifiers). Matches 3-part `(catalog, db, table)` and 2-part `(db, table)` keys without falling back across mismatched qualifiers; returns sorted deduplicated missing identifiers (`bad_alias.col` adds `"bad_alias"` to `missing_tables`).
6.  **`base.py` — Streaming Evaluator Base:** Single-writer O(1)-counter accumulators (`add`, `add_batch`, `compute`, `reset`; opt-in `include_per_example` in O(N) list memory). Rejects `str`/`bytes` batch inputs and length mismatches before state updates; `compute()` rejects empty state (`ValueError`) and resets after returning `dict`.
7.  **`sql_bleu.py` — `SqlBleu` Metric:** Computes Papineni corpus BLEU (`m_k / c_k` or `(m_k + 1.0) / (c_k + 1.0)` when `smooth=True`; `BP = 1.0` if `translation_length > reference_length` else `exp(1 - reference_length / translation_length)`). Accepts `str | Sequence[str]` non-empty references; applies `extractor` to predictions only. Under `canonicalize=False`, runs extraction/size checks, marks failed extractions unparseable, and calls `sql_tokenizer` (`include_semicolons`, `strip_comments`). Under `canonicalize=True`, post-parse canonicalize errors count as unparseable; drops invalid references; skips rows with invalid prediction or zero valid references (`None`); returns `None` score fields if `scored_count == 0` or `reference_length == 0`, `bleu=0.0` if `translation_length == 0` (`scored_count > 0`, `reference_length > 0`), and smoothed `per_example_bleu` (`smooth=True`).
8.  **`valid_sql_rate.py` — `ValidSqlRate` Measurement:** Normalizes non-`None` schemas per row (no cross-row caching), grounds syntax-valid query ASTs (`schema_evaluated_count + schema_skipped_count == num_examples`), zero-initializes all 8 `failure_counts` keys (`max_input_chars -> extraction_failed`, `max_query_chars -> syntax_error`), and emits structural `per_example` dicts without raw SQL or literals.
9.  **`wildcard_detector.py` — `WildcardDetector` Measurement:** Walks `ast.find_all(exp.Select)` under `(SqlglotError, RecursionError)`, skipping only direct `exp.Exists.this` targets (`Select` or `Subquery(Select)`) while scanning deeper nested subqueries inside `EXISTS`. Unwraps `exp.Alias`, flags direct `exp.Star` (incl. `* EXCEPT`/`EXCLUDE`) or `exp.Column(this=exp.Star())` as `1`, ignores `RETURNING *`, `COUNT(*)`, and `*` math (`0`), and skips unparseable rows (`None`).
10. **`__init__.py` — Public API:** Exports evaluators, stateless helpers, core utilities, and `@overload` `load(name, **kwargs)` for `Literal["sql_bleu"]`, `Literal["valid_sql_rate"]`, `Literal["wildcard_detector"]`, attaching `NullHandler` to `"group_by_vibes"`.

### Public Function & Class Signatures & Output Schemas

#### Public API Signatures (`src/group_by_vibes/`)

```python
ExtractorLike = str | re.Pattern[str] | Callable[[str], str | None] | None
SchemaDict = Mapping[str, Iterable[str]]
# Bounds: max_order >= 1; max_query_chars/max_input_chars = None or int > 0 (not bool); invalid -> ValueError.

def normalize_dialect(dialect: str = "postgres") -> str: ...
def extract_fenced_or_last_sql(text: str) -> str | None: ...
def try_canonicalize_sql(query: str, *, dialect: str = "postgres", sort_commutative: bool = True, normalize_identifiers: bool = True, strip_comments: bool = True, include_semicolons: bool = False, max_query_chars: int | None = 65_536) -> tuple[str, bool]: ...
def canonicalize_sql(query: str, dialect: str = "postgres", *, sort_commutative: bool = True, normalize_identifiers: bool = True, strip_comments: bool = True, include_semicolons: bool = False, max_query_chars: int | None = 65_536) -> str: ...
def sql_tokenizer(text: str, *, dialect: str = "postgres", include_semicolons: bool = False, strip_comments: bool = True, max_query_chars: int | None = 65_536) -> list[str]: ...

class SqlBleu:
    def __init__(self, dialect: str = "postgres", max_order: int = 4, canonicalize: bool = True, sort_commutative: bool = True, normalize_identifiers: bool = True, strip_comments: bool = True, include_semicolons: bool = False, extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> None: ...
    def add(self, prediction: str, reference: str | Sequence[str]) -> None: ...
    def add_batch(self, predictions: Sequence[str], references: Sequence[str] | Sequence[Sequence[str]]) -> None: ...
    def compute(self, smooth: bool = False) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class ValidSqlRate:
    def __init__(self, dialect: str = "postgres", extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> None: ...
    def add(self, data: str, schema: SchemaDict | None = None) -> None: ...
    def add_batch(self, data: Sequence[str], schemas: Sequence[SchemaDict | None] | None = None) -> None: ...
    def compute(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class WildcardDetector:
    def __init__(self, dialect: str = "postgres", extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> None: ...
    def add(self, data: str) -> None: ...
    def add_batch(self, data: Sequence[str]) -> None: ...
    def compute(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

def compute_sql_bleu(predictions: Sequence[str], references: Sequence[str] | Sequence[Sequence[str]], *, dialect: str = "postgres", max_order: int = 4, smooth: bool = False, canonicalize: bool = True, sort_commutative: bool = True, normalize_identifiers: bool = True, strip_comments: bool = True, include_semicolons: bool = False, extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> dict[str, Any]: ...
def sentence_sql_bleu(prediction: str, reference: str | Sequence[str], *, dialect: str = "postgres", max_order: int = 4, smooth: bool = True, canonicalize: bool = True, sort_commutative: bool = True, normalize_identifiers: bool = True, strip_comments: bool = True, include_semicolons: bool = False, extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072) -> float | None: ...
def compute_valid_sql_rate(data: Sequence[str], schemas: Sequence[SchemaDict | None] | None = None, *, dialect: str = "postgres", extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> dict[str, Any]: ...
def compute_wildcard_detector(data: Sequence[str], *, dialect: str = "postgres", extractor: ExtractorLike = None, max_query_chars: int | None = 65_536, max_input_chars: int | None = 131_072, include_per_example: bool = False) -> dict[str, Any]: ...

@overload
def load(name: Literal["sql_bleu"], **kwargs: Any) -> SqlBleu: ...
@overload
def load(name: Literal["valid_sql_rate"], **kwargs: Any) -> ValidSqlRate: ...
@overload
def load(name: Literal["wildcard_detector"], **kwargs: Any) -> WildcardDetector: ...
def load(name: Literal["sql_bleu", "valid_sql_rate", "wildcard_detector"] | str, **kwargs: Any) -> Any: ...
```

#### Exact `compute()` JSON-Serializable Output Schemas

##### 1. `SqlBleu.compute()` (`Metric`)

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `bleu` | `float \| None` | Corpus SQL-BLEU in `[0.0, 1.0]` (`None` if `scored_count == 0` or `reference_length == 0`; `0.0` if `translation_length == 0`) |
| `precisions` | `list[float] \| None` | Modified 1-gram .. `max_order`-gram precisions (`None` if `scored_count == 0` or `reference_length == 0`) |
| `brevity_penalty` | `float \| None` | Corpus brevity penalty (`None` if `scored_count == 0` or `reference_length == 0`) |
| `length_ratio` | `float \| None` | `translation_length / reference_length` (`None` if `scored_count == 0` or `reference_length == 0`) |
| `translation_length` | `int` | Total SQL token count across scored pairs (matches HF `evaluate` `bleu` key) |
| `reference_length` | `int` | Closest valid reference SQL token count across scored pairs |
| `scored_count` | `int` | Pairs contributing to corpus BLEU (valid prediction and >= 1 valid reference) |
| `parsed_count` | `int` | Predictions passing extraction, size check, and parsing/tokenization (`canonicalize` flag dependent) |
| `unparseable_count` | `int` | Predictions failing extraction, size check, or parsing/canonicalization |
| `parse_success_rate` | `float` | `parsed_count / num_examples` |
| `reference_parse_success_rate` | `float` | Fraction of all supplied golden references passing validation (`parse_success` / non-blank tokens) |
| `num_examples` | `int` | Total example pairs processed |
| `dialect` | `str` | Normalized `sqlglot` dialect used |
| `max_order` | `int` | Maximum n-gram order |
| `smooth` | `bool` | Whether add-1 smoothing (`(m + 1.0) / (p + 1.0)`) was applied to corpus BLEU |
| `per_example_bleu` | `list[float \| None]` | *(Only when `include_per_example=True`)* Smoothed (`smooth=True`) per-pair score, `None` when skipped |

##### 2. `ValidSqlRate.compute()` (`Measurement`)

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `valid_sql_rate` | `float` | Rows passing lenient AST parsing and applicable query schema grounding divided by `num_examples` |
| `syntax_valid_rate` | `float` | Fraction passing extraction and lenient dialect-aware `sqlglot` AST parsing |
| `schema_grounded_rate` | `float \| None` | `schema_grounded_count / schema_evaluated_count` (`None` if `schema_evaluated_count == 0`) |
| `schema_grounding_status` | `str` | `"applied"` (`schema_evaluated_count == num_examples`), `"partial"` (`0 < evaluated < num_examples`), or `"skipped"` (`schema_evaluated_count == 0`) |
| `schema_evaluated_count` | `int` | Syntax-valid query AST rows (`build_scope(ast)` non-`None`) with non-`None` schema |
| `schema_grounded_count` | `int` | Rows in `schema_evaluated_count` passing static table and column checks |
| `schema_skipped_count` | `int` | `num_examples - schema_evaluated_count` (`schema=None`, syntax failures, or `build_scope(ast) is None`) |
| `syntax_valid_count` | `int` | Rows passing extraction and lenient `sqlglot` AST parsing |
| `valid_sql_count` | `int` | Rows passing syntax validation and applicable schema grounding (`schema_valid` is `True` or `None`) |
| `num_examples` | `int` | Total examples processed |
| `dialect` | `str` | Normalized `sqlglot` dialect used |
| `failure_counts` | `dict[str, int]` | Zero-initialized 8-key dict summing to `num_examples - valid_sql_count`: `extraction_failed` (includes `max_input_chars`), `extraction_error`, `empty_input`, `multi_statement`, `command_fallback`, `syntax_error` (includes `max_query_chars` and AST scope errors), `unknown_table`, `unknown_column` |
| `per_example` | `list[dict[str, Any]]` | *(Only when `include_per_example=True`)* Dict with `syntax_valid` (`bool`), `schema_valid` (`bool \| None`), `failure_reason` (`str \| None`), sorted deduplicated `missing_tables` (`list[str]`), `missing_columns` (`list[str]`); raw SQL and literal values omitted |

##### 3. `WildcardDetector.compute()` (`Measurement`)

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `wildcard_rate` | `float \| None` | `wildcard_count / parsed_count` (`None` if `parsed_count == 0`) |
| `wildcard_count` | `int` | Parsed queries containing one or more `SELECT *` / `table.*` projections |
| `parsed_count` | `int` | Queries passing extraction and `sqlglot` AST parsing |
| `parse_failed_count` | `int` | Queries skipped due to extraction or parse failure |
| `extraction_failed_count` | `int` | Subset of `parse_failed_count` caused by extractor failure or error |
| `num_examples` | `int` | Total examples processed |
| `dialect` | `str` | Normalized `sqlglot` dialect used |
| `per_example` | `list[int \| None]` | *(Only when `include_per_example=True`)* `1` (projection wildcard found), `0` (absent), or `None` (unparseable) |

### External Reference Benchmark Boundary (`examples/`)

*   **Strict Isolation:** `examples/bird_eval.py` uses only `[project.optional-dependencies]` `examples` in `pyproject.toml`, remaining outside `src/group_by_vibes/` so ML/dataset packages never enter core imports.
*   **Deterministic Stratification & Inference:** Samples up to 500 queries from `birdsql/bird_mini_dev` (`data/mini_dev_sqlite.jsonl`, `data/mini_dev_postgresql.jsonl`, or `data/mini_dev_mysql.jsonl`) across 11 multi-table databases, stratified by `difficulty` (`simple`, `moderate`, `challenging`), sorting each stratum by `question_id` before `random.Random(42).sample(...)`. Loads `baseline_inference/schemas/<dialect>.json` via `huggingface_hub.hf_hub_download`, parses `CREATE TABLE` DDL into per-database `{table: [columns]}` `dict` schemas, and records `gold_sql = row["SQL"]` plus `evidence`. Formats prompts with `baseline_inference/prompt_template.txt` (`apply_chat_template` when present) using `torch.bfloat16` (`batch_size=1`, `max_new_tokens=2048`), defaulting to `Qwen/Qwen2.5-Coder-3B-Instruct` (~6.2 GB BF16 for local Apple Silicon/laptops, alongside presets for `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` and `griffith-bigdata/FINER-SQL-3B-v2`).
*   **Reproducible JSONL Log:** Flushes per-row records (`question_id`, `db_id`, `difficulty`, `dialect`, `question`, `evidence`, `schema`, `gold_sql`, `raw_completion`, `truncated`, `seed`, `model`, `extractor`, `group_by_vibes_version`, `sqlglot_version`) per query before offline `score` evaluation across all three evaluators plus per-`difficulty` breakdown.

## Alternatives Considered

### Alternative 1: ISO SQL:2016 EBNF Grammar and CFG Prefix Metrics

*   **Pros:** Measures conformance against the published SQL:2016 EBNF grammar ([sql-2016.ebnf](https://github.com/ronsavage/SQL/blob/master/sql-2016.ebnf)) via Context-Free Grammar metrics.
*   **Cons:** `sql-2016.ebnf` is a railroad-diagram specification rather than an executable parser and rejects valid dialect syntax such as PostgreSQL/SQLite `LIMIT` clauses and SQLite bracket/backtick quoting.
*   **Reason for Rejection:** Rejected by author: standardized on dialect-aware `sqlglot` parsing so valid SQL across dialects is not flagged as grammar violations.

### Alternative 2: Database-Executed Metrics (`BIRD Execution accuracy (EX)`, `SQLite Executable`, `EXPLAIN`, `DRY RUN`) in v1

*   **Pros:** Verifies semantic execution equivalence and planner/runtime behavior directly on a database engine.
*   **Cons:** Requires DB drivers, sandboxes, timeouts, and read-only isolation against untrusted queries.
*   **Reason for Rejection:** Deferred by author: excluded from v1 to keep the core library DB-free and lightweight; static `sqlglot` checks run first without connection setup.

### Alternative 3: Static Data-Type Checking in `Valid SQL Rate` (v1)

*   **Pros:** Catches type mismatches (e.g., numeric aggregates over text columns) offline.
*   **Cons:** Without a planner, SQLite dynamic typing and PostgreSQL implicit casts produce false alarms, and callers may supply only column names.
*   **Reason for Rejection:** Deferred by author: excluded until planner-backed checks or typed schemas are added later; trading type precision for zero-type-schema compatibility.

### Alternative 4: Hard Runtime Dependency on / Subclassing Hugging Face `evaluate`

*   **Pros:** Supports `evaluate.load(...)` registry plugins and PyArrow disk caching out of the box.
*   **Cons:** Pulls `evaluate`, `datasets`, `pyarrow`, and Hub dependencies into lightweight or offline pipelines.
*   **Reason for Rejection:** Rejected by author: mirroring `add()` / `add_batch()` / `compute()` locally keeps `sqlglot` as the sole core runtime dependency while giving up Hub plugin discovery.

### Alternative 5: Custom `offline` / `online` Module Taxonomy

*   **Pros:** Separates golden-set batch scripts from live monitoring scripts directly by usage setting.
*   **Cons:** Misleads callers when reference-free checks (`ValidSqlRate`, `WildcardDetector`) also run on golden benchmarks.
*   **Reason for Rejection:** Rejected by author: adopted Hugging Face `evaluate`'s `Metric` (reference-based) vs. `Measurement` (reference-free) split, accepting slight domain indirection for API familiarity.

### Alternative 6: Built-In LLM Inference and Benchmark Runner Inside Core Package

*   **Pros:** Provides a single pip install and CLI command from Hugging Face weights to scores.
*   **Cons:** Forces `torch`, `transformers`, `accelerate`, and `datasets` into every lightweight evaluation environment.
*   **Reason for Rejection:** Rejected by author: isolated benchmark inference under `examples/` so production scoring environments avoid heavy GPU/ML dependencies.

### Alternative 7: `CREATE TABLE` DDL String Parsing and Global-Only Schema in `Valid SQL Rate` (v1)

*   **Pros:** Accepts raw `.sql` dumps and simplifies single-DB constructor calls.
*   **Cons:** Single global schemas cannot score multi-database benchmarks (`birdsql/bird_mini_dev` spans 11 databases) in one batch, and DDL parsing adds dialect statement complexity.
*   **Reason for Rejection:** Rejected/deferred by author: chose per-query Python `dict` schemas (`None` skipped vs. `{}` failed) for heterogeneous batch support, deferring `.sql` DDL parsing to post-v1 adapters.

### Alternative 8: Custom Apache Impala Dialect Subclass or Restricting Dialects Strictly to PostgreSQL and SQLite

*   **Pros:** Covers custom Impala syntax or limits test scope to two engines.
*   **Cons:** `sqlglot` ships no built-in `impala` dialect, while rejecting other engines breaks valid callers on BigQuery, DuckDB, Snowflake, MySQL, or T-SQL.
*   **Reason for Rejection:** Rejected by author: accepts all dialects resolved by `sqlglot.Dialect.get_or_raise`, excludes custom Impala maintenance, and prioritizes PostgreSQL and SQLite in docs and tests.

### Alternative 9: Mandatory Regex SQL Extractor Parameter

*   **Pros:** Forces callers to think explicitly about extraction before scoring noisy text.
*   **Cons:** Adds boilerplate when scoring unformatted raw SQL queries and misses fenced Markdown variants without custom regexes.
*   **Reason for Rejection:** Rejected by author: defaulted to raw SQL (`extractor=None`) with an opt-in regex/callable `extractor` so simple SQL inputs require zero configuration.

### Alternative 10: Native `async def` Wrappers Around `sqlglot` Evaluation

*   **Pros:** Permits `await metric.compute()` syntax inside async loops.
*   **Cons:** `sqlglot` work is CPU-bound Python with zero I/O, so `async def` still blocks event loops unless offloaded to worker threads or processes.
*   **Reason for Rejection:** Rejected by author: core evaluators remain synchronous while advising callers to offload runtime evaluation out-of-band.

### Alternative 11: Strict Legacy Compatibility with `src/sql_bleu.py` / `utils/` References, CLI, `SqlBleuResult`, and `backend="evaluate"`

*   **Pros:** Zero import changes for legacy scripts passing `backend="evaluate"`, reading `SqlBleuResult` attributes, or invoking [sql_bleu.py](../../src/sql_bleu.py) via CLI.
*   **Cons:** Preserves heavyweight `evaluate` backend plumbing, a mutable global cache, dual output types (`SqlBleuResult` vs. `dict`), and CLI flag maintenance inside a library API.
*   **Reason for Rejection:** Rejected by author: relocated core scoring into `src/group_by_vibes/` as a library-only Python API returning JSON-serializable `dict`s (`"translation_length"` key), retiring legacy backend/dataclass/CLI entrypoints at v0.1.

### Alternative 12: `sqlglot.optimizer.qualify` for Static Schema Grounding

*   **Pros:** Automates `SELECT *` and column lineage expansion across CTEs and subqueries.
*   **Cons:** Adds rewrite overhead, mutates ASTs, and raises `OptimizeError` when column metadata or dialect constructs cannot be qualified.
*   **Reason for Rejection:** Rejected by author: chose bounded query-scope traversal via `sqlglot.optimizer.scope.build_scope` without optimizer AST rewrites (`build_scope(ast) is None` non-query statements skip schema grounding).

### Alternative 13: Plain-Text Fallback or `0.0` Scoring on Unparseable SQL

*   **Pros:** Produces numeric scores on invalid SQL (`0.0` penalizes invalid SQL directly; plain-text fallback awards partial string overlap).
*   **Cons:** `0.0` collapses average BLEU when parse rates are low, plain-text fallback compares unparsed text against canonical SQL tokens, and regex wildcard matching misreads comments/strings.
*   **Reason for Rejection:** Rejected by author: `SqlBleu` and `WildcardDetector` skip unparseable predictions (`None`) and unparseable reference groups so aggregate metrics evaluate over valid SQL alongside explicit parse-rate counts.

### Alternative 14: SQL-Domain-Only (`predictions=` / `"prediction_length"`) or Hybrid API Naming

*   **Pros:** Uses uniform parameter names across Metrics and Measurements and preserves `"prediction_length"` from [sql_bleu.py](../../src/sql_bleu.py).
*   **Cons:** Diverges from Hugging Face `evaluate`'s `Metric` (`prediction`/`reference`, `"translation_length"`) vs. `Measurement` (`data=`) conventions.
*   **Reason for Rejection:** Rejected by author: standardized on Hugging Face `evaluate` parameter and `"translation_length"` output key naming for interoperability, trading internal uniformity across evaluator signatures.

### Alternative 15: Deferring `Wildcard Detector` or Folding `SELECT *` Into `Valid SQL Rate`

*   **Pros:** Shrinks v1 from three classes to two without adding a separate `WildcardDetector` API.
*   **Cons:** `SELECT *` is syntactically valid SQL, so folding it into `ValidSqlRate` conflates validity with projection style, while deferring omits projection-star checks in v1.
*   **Reason for Rejection:** Rejected by author: kept `WildcardDetector` as a third standalone `Measurement` (`0` / `1` / `None`) in v1 for explicit signal separation.

### Alternative 16: Broader Initial Metric Suite (`SQL Component F1` and Additional Metrics) in v1

*   **Pros:** Adds clause-level partial matching (`SELECT`/`WHERE`/`GROUP BY` keyword-identifier F1) from day one.
*   **Cons:** Multiplies AST decomposition edge cases before validating the core three-evaluator API.
*   **Reason for Rejection:** Deferred by author: excluded clause-level F1 metrics until post-v1 releases, prioritizing `SqlBleu`, `ValidSqlRate`, and `WildcardDetector` first.

### Alternative 17: Alternate Python SQL Parsers (`sqlfluff`, `sqlparse`, `sqloxide`) and Bundling `sqlglot[c]`

*   **Pros:** `sqlparse` has zero C extensions and tolerates arbitrary partial text; `sqlfluff` checks formatting; `sqloxide` (`sqlparser-rs`) and `sqlglot[c]` offer native C/Rust parsing throughput.
*   **Cons:** `sqlparse` provides no semantic AST/scope analysis; `sqlfluff` pulls heavy linter rules and config overhead; `sqloxide` lacks `build_scope` / AST canonicalization; bundling `sqlglot[c]` introduces compiled wheel portability constraints in core dependencies.
*   **Reason for Rejection:** Rejected by author: retained pure-Python `sqlglot` for its multi-dialect AST, tokenizer, and `build_scope` engine while letting callers install `sqlglot[c]` out-of-band if compiled speedup is desired.

## Risks & Mitigations

*   **Lenient `sqlglot` Parsing vs. DB Engines:** `sqlglot` is a permissive transpiler and accepts queries real engines reject. *Mitigation:* Document `ValidSqlRate` strictly as lenient dialect-aware AST parsing (not `EX`/`EXPLAIN`) and scan entire AST subtrees for `exp.Command`.
*   **DDL/DML Pass-Through:** Syntax-valid destructive SQL (`DROP`, `DELETE`) parses with `syntax_valid = True` (`build_scope(ast) is None` skips schema checks on non-query ASTs, yielding `valid_sql_rate = 1.0`). *Mitigation:* Document zero read-only filtering and leave execution authorization to callers.
*   **`SqlBleu` Survivorship Bias:** Skipping unparseable rows lets a model parsing 1% of queries score `bleu = 1.0`. *Mitigation:* Return `scored_count`, `parsed_count`, `parse_success_rate`, and `reference_parse_success_rate` alongside `bleu`.
*   **Custom Regex ReDoS & Oversized Input Rejection:** `max_input_chars` limits scan size, not exponential backtracking on short strings, while `max_query_chars=65_536` and Python recursion limits reject unusually long or deeply nested valid queries via `syntax_error`. *Mitigation:* Document safe linear caller regex guidance, apply `max_input_chars` pre-execution, and catch `(SqlglotError, RecursionError)` without silent truncation.
*   **Shared-Accumulator Worker Races:** `BaseEvaluator` instances are single-writer without internal locks; mutating one instance from concurrent worker threads causes lost updates or reset races. *Mitigation:* Document single-writer semantics explicitly; use per-worker instances, a serialized aggregation queue, or stateless `compute_*` helpers.
*   **Per-Example Memory & Parse Latency:** While aggregate counters take O(1) memory, `include_per_example=True` retains `O(N)` list state, and complex multi-clause queries can take ~0.41–0.47 s per query in pure-Python `sqlglot`. *Mitigation:* Keep `include_per_example=False` by default and note optional out-of-band `sqlglot[c]` installation for throughput-sensitive pipelines.
*   **`sqlglot` API Drift Across Minor Versions:** Upstream `sqlglot` occasionally introduces breaking AST or optimizer changes on `MINOR` releases, so `<31.0.0` alone does not prevent minor-bump breaks and `parser.py` does not shield downstream `exp.*` / `build_scope` consumers. *Mitigation:* Pin exact `sqlglot` versions in lockfiles (`uv.lock`), isolate `sqlglot.parse`/`tokenize` inside `parser.py`, and record `sqlglot_version` in `examples/` JSONL artifacts.

## Non-Functional Requirements

*   **Security & Privacy:** The core package opens no database connections and executes no SQL. Size caps (`max_query_chars=65_536`, `max_input_chars=131_072`, or `None` to disable) guard oversized inputs pre-parse, and `(SqlglotError, RecursionError)` catches handle deep-nesting failures. Diagnostic logs and `per_example` outputs omit raw SQL query strings and raw literal values (`missing_tables` and `missing_columns` record identifier strings only).
*   **Performance & Latency:** Aggregate accumulator counters run in O(1) memory, while opt-in `include_per_example=True` grows `O(N)` with batch size. Example generation uses `torch.bfloat16` (`batch_size=1`, `max_new_tokens=2048`) and logs truncation.
*   **Packaging & Rollback:** Core `pyproject.toml` requires only `sqlglot>=30.0.0,<31.0.0`. Static golden snapshots verify parity before deleting [sql_bleu.py](../../src/sql_bleu.py), while `examples/` JSONL logs record raw generations, schemas, and versions for offline rollback checks.

## Future Improvements

*   **Sandboxed Database Execution:** Optional `EX`, `EXPLAIN`, and `DRY RUN` modules (starting with read-only stdlib `sqlite3` and timeout-guarded adapters).
*   **`CREATE TABLE` DDL Adapter:** Compiled DDL-to-dict helper cached by `(ddl_hash, dialect)` for `.sql` schema dumps.
*   **Typed Column/Planner Checking:** Opt-in type compatibility checks once typed schemas or planners exist.
*   **Additional Metrics & `evaluate` Adapter:** `SQL Component F1`, AST edit distance, and an optional standalone wrapper for `evaluate.load(...)`.
*   **Custom Dialect Extensions:** Opt-in `Impala(Hive)` subclass or dialect registry plugin.

## Additional Resources

*   **SQL-BLEU Paper:** [Chen et al. (2019), NL2pSQL](https://aclanthology.org/D19-1262/) (DOI: `10.18653/v1/D19-1262`)
*   **ISO SQL:2016 Grammar Reference:** [sql-2016.ebnf](https://github.com/ronsavage/SQL/blob/master/sql-2016.ebnf)
*   **SQL Parser & Scope Engine:** [sqlglot](https://github.com/tobymao/sqlglot)
*   **API Reference Model:** [Hugging Face evaluate](https://huggingface.co/docs/evaluate)
*   **External Benchmark & Models:** [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct), [XGenerationLab/XiYanSQL-QwenCoder-3B-2504](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-3B-2504), and [griffith-bigdata/FINER-SQL-3B-v2](https://huggingface.co/griffith-bigdata/FINER-SQL-3B-v2)
*   **Existing Prototype:** [sql_bleu.py](../../src/sql_bleu.py)

---

## Detailed Implementation Plan

### Phased Execution Order

1.  **Phase 1 — Packaging & Core Modules:** Create `pyproject.toml`, initial `src/group_by_vibes/__init__.py` (`__version__`, stdlib `logging.NullHandler()`, Phase 1 exports), `dialects.py`, `extract.py`, `parser.py`, `canonicalize.py`, `grounding.py`, `base.py`, and `tests/test_dialects_extract_parser.py`. *Checkpoint:* `uv sync --group dev` succeeds, wheel metadata lists only `sqlglot>=30.0.0,<31.0.0`, and Phase 1 tests pass.
2.  **Phase 2 — Top-Level Evaluators & Golden Snapshots:** Implement top-level evaluator modules `sql_bleu.py`, `valid_sql_rate.py`, `wildcard_detector.py`, and typed `load()`, updating `src/group_by_vibes/__init__.py`. Capture reference canonicalized outputs and corpus BLEU numbers from `src/sql_bleu.py` into permanent static golden snapshots in `tests/test_canonicalize_and_sql_bleu.py` (without importing `src/sql_bleu.py`), alongside `tests/test_valid_sql_rate.py` and `tests/test_wildcard_detector.py`. *Checkpoint:* Phase 1–2 unit and snapshot suites pass.
3.  **Phase 3 — Prototype Retirement & Isolation:** Delete [sql_bleu.py](../../src/sql_bleu.py), update [README.md](../../README.md), and add `tests/test_packaging_isolation.py`. *Checkpoint:* `uv build`, wheel `METADATA` verification, `pytest --cov-branch`, `ruff`, and `mypy` pass with zero errors.
4.  **Phase 4 — Reference Benchmark (`examples/`):** Add `examples/bird_eval.py`, `examples/fixtures/bird_sample.jsonl`, and `examples/README.md` (`[project.optional-dependencies]` `examples`). *Checkpoint:* `uv run python examples/bird_eval.py score --input examples/fixtures/bird_sample.jsonl` passes in a core-only environment.

### Target Files to Create/Modify/Delete

1.  **`pyproject.toml` `(Create)`** — *Packaging:* Configure `[build-system]` (`requires = ["hatchling"]`, `build-backend = "hatchling.build"`), `[tool.hatch.build.targets.wheel]` (`packages = ["src/group_by_vibes"]`), PEP 621 `name = "group-by-vibes"`, `version = "0.1.0"`, `requires-python = ">=3.10"`, `dependencies = ["sqlglot>=30.0.0,<31.0.0"]`, `[dependency-groups]` `dev` (`pytest>=8.0`, `pytest-cov>=5.0`, `ruff>=0.8`, `mypy>=1.11`), `[project.optional-dependencies]` `examples` (`datasets`, `huggingface_hub`, `transformers`, `torch`, `accelerate`), `[tool.pytest.ini_options]`, `[tool.coverage.run]` / `[tool.coverage.report]`, `[tool.ruff.lint]` (`select = ["E", "F", "I", "UP", "B", "SIM"]` on `src/`, `tests/`, `examples/`), and `[tool.mypy]` strict mode on `src/group_by_vibes/`.
2.  **`src/group_by_vibes/__init__.py` `(Create Phase 1, Modify Phase 2)`** — *Entrypoint:* Phase 1 defines `__version__`, attaches `logging.NullHandler()` to `"group_by_vibes"`, and exports Phase 1 utilities (`normalize_dialect`, `extract_fenced_or_last_sql`, `canonicalize_sql`, `try_canonicalize_sql`, `sql_tokenizer`). Phase 2 adds `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `compute_sql_bleu`, `sentence_sql_bleu`, `compute_valid_sql_rate`, `compute_wildcard_detector`, and `load(name, **kwargs)` typed via `typing.overload` (`Literal["sql_bleu"] -> SqlBleu`, `Literal["valid_sql_rate"] -> ValidSqlRate`, `Literal["wildcard_detector"] -> WildcardDetector`, raising `ValueError` on unknown names).
3.  **`src/group_by_vibes/dialects.py` `(Create)`** — *Dialects & Identifiers:* `normalize_dialect` validates `isinstance(dialect, str)`, strips whitespace, lowercases, maps `"postgresql"`->`"postgres"` and `"sqlite3"`->`"sqlite"`, keeps ANSI `""`, calls `sqlglot.Dialect.get_or_raise`, and raises descriptive `ValueError` otherwise. `normalize_grounding_identifier(name, quoted, dialect)` applies case-insensitive `.lower()` for `"sqlite"` and delegates non-SQLite unquoted identifiers to the dialect's `sqlglot` `NORMALIZATION_STRATEGY` while preserving quoted case (used exclusively by `grounding.py`). `split_qualified_table(raw_name, dialect)` parses 1–3 part `(catalog, db, table)` keys plus per-part quoted flags directly inside `dialects.py` across quoted dots without calling `sqlglot.parse_one`.
4.  **`src/group_by_vibes/extract.py` `(Create)`** — *Extractor:* Define frozen `ExtractionResult(sql, ok, failure_reason)`, `compile_extractor(extractor: str | re.Pattern[str] | Callable[[str], str | None] | None)`, and `apply_extractor(raw_text, compiled, max_input_chars=131_072)`. Compile regexes eagerly (`re.error` on invalid regex, `ValueError` if `pattern.groups > 1`). Run `apply_extractor` only when `extractor is not None`: verify `isinstance(raw_text, str)` and `len(raw_text) <= max_input_chars` (`None` disables) before regex/callable execution (oversized -> `extraction_failed`). Regex uses `re.search` (`group(1)` if `groups == 1` else `group(0)`; `None`/blank -> `extraction_failed`). Callables returning `None`/blank -> `extraction_failed`; non-`str` return or exception -> `extraction_error` (`KeyboardInterrupt`/`SystemExit` re-raised). `extract_fenced_or_last_sql(text: str) -> str | None` rejects unclosed `<think>`, strips `<think>...</think>`, returns last ```` ```sql ```` fence, else last generic SQL-keyword fence (`SELECT`/`WITH`/`INSERT`/`UPDATE`/`DELETE`/`CREATE`), else slices from standalone `WITH`/`SELECT` to terminating `;` or blank-line paragraph break so trailing prose is excluded (`None` if blank/unmatched).
5.  **`src/group_by_vibes/parser.py` `(Create)`** — *`sqlglot` Adapter:* Sole caller of `sqlglot.parse` and `sqlglot.tokenize`. Define frozen `ParsedSql(ast: exp.Expression | None, ok: bool, failure_reason: str | None)`, `parse_sql(sql, dialect, max_query_chars=65_536)`, and `tokenize_sql(sql, dialect, *, include_semicolons=False, strip_comments=True, max_query_chars=65_536) -> list[str]`. Pre-parse: reject `len(sql) > max_query_chars` (`syntax_error`) and blank strings (`empty_input`); strip trailing `;` `None` entries from `sqlglot.parse`; reject empty (`empty_input`), `len(stmts) > 1` (`multi_statement`), and `next(ast.find_all(exp.Command), None) is not None` (`command_fallback`); catch `(SqlglotError, RecursionError)` (`syntax_error`), log SQL-redacted diagnostics, with zero parse cache. `tokenize_sql` validates `max_query_chars`, re-quotes `TokenType.STRING` via `"'" + tok.text.replace("'", "''") + "'"`, preserves quoted identifiers, filters `TokenType.SEMICOLON` unless `include_semicolons=True` and comment tokens when `strip_comments=True`, and falls back under `(SqlglotError, RecursionError)` to linear quote-aware regex tokenization with dialect-scoped `[...]` brackets (`sqlite`, `tsql`, `bigquery`).
6.  **`src/group_by_vibes/canonicalize.py` `(Create)`** — *Canonicalizer & Tokenizer Wrapper:* Export `try_canonicalize_sql`, `canonicalize_sql`, and thin `sql_tokenizer` wrapper calling `parser.tokenize_sql` with matching kwargs. Inline transforms: keyword normalization (`normalize=True`), comment/semicolon stripping (`comments=not strip_comments`, preserving trailing `;` when `include_semicolons=True`), deterministic `.lower()` on all unquoted identifiers (`if not ident.quoted`, keeping quoted case intact across dialects), and optional commutative sorting of `exp.EQ`/`exp.NEQ`, subquery-free `exp.In` (`query is None`), and flattened `exp.And`/`exp.Or` chains preserving `_LOGICAL_CLASSES = (exp.And, exp.Or, exp.Xor)` precedence parentheses. Recursive post-order AST transform returning updated root (fixing detached nodes and bare roots) without `sqlglot.optimizer`, guarded under `(SqlglotError, RecursionError)` returning `("", False)`. Note fallback regex scopes `[...]` brackets to bracket dialects so PostgreSQL `arr[1]` does not merge `[1]` into a bracketed identifier token.
7.  **`src/group_by_vibes/grounding.py` `(Create)`** — *Static Grounding:* Internal `_NormalizedSchema`, frozen `GroundingResult(ok, failure_reason, missing_tables, missing_columns, skipped)`, `normalize_schema(schema, dialect)`, `check_schema_grounding(ast, normalized_schema, dialect)`. Validate `Mapping[str, Iterable[str]]` eagerly (reject `str`/`bytes` column values and non-`str` entries, zero global cache). When `build_scope(ast)` returns `None` (non-query AST), return `skipped=True` (`schema_valid=None`). Otherwise traverse `scope.traverse()` and `scope.external_columns` under `(SqlglotError, RecursionError)`, resolving CTEs, CTE/derived `exp.TableAlias.columns`, table aliases, `SELECT` aliases in `GROUP BY`/`HAVING`/`ORDER BY`, and SQLite implicit `rowid`/`_rowid_`/`oid` (`dialect == "sqlite"`). Treat `SELECT *` CTEs, `VALUES`, `UNNEST`, TVFs, and column-less derived tables as opaque sources (unqualified columns pass in mixed opaque+physical scopes; `unknown_column` triggers only when all active sources have known columns; `alias.*` still verifies qualifier resolution). Resolve 3-part `(catalog, db, table)` and 2-part `(db, table)` keys without falling back across mismatched explicit qualifiers; collect all missing tables and columns sorted and deduplicated (`bad_alias.col` records `"bad_alias"` in `missing_tables`; `failure_reason = "unknown_table"` if `missing_tables` else `"unknown_column"`).
8.  **`src/group_by_vibes/base.py` `(Create)`** — *Streaming Base:* `BaseEvaluator` (`add`, `add_batch`, `compute`, `reset`) with single-writer counters (not thread-safe). Validate `max_query_chars` / `max_input_chars` (`None` or `int > 0`, rejecting `bool` / `<= 0` with `ValueError`), reject invalid batch sequences (`str`/`bytes`/mapping) and length mismatches before mutating accumulators; `compute()` accepts zero data args, raises `ValueError` on empty state, and resets state only after returning `dict`.
9.  **`src/group_by_vibes/sql_bleu.py` `(Create)`** — *`SQL-BLEU`:* Implement `SqlBleu` (`add`, `add_batch`, `compute`, `reset`), `compute_sql_bleu(predictions, references, *, dialect, max_order, smooth, canonicalize, sort_commutative, normalize_identifiers, strip_comments, include_semicolons, extractor, max_query_chars, max_input_chars, include_per_example) -> dict[str, Any]`, and `sentence_sql_bleu(prediction, reference, *, dialect, max_order, smooth, canonicalize, sort_commutative, normalize_identifiers, strip_comments, include_semicolons, extractor, max_query_chars, max_input_chars) -> float | None` matching Public API Signatures defaults. Retire legacy backend, `SqlBleuResult`, and `main()` CLI. Accept `str | Sequence[str]` references; run `extractor` on predictions only. Under `canonicalize=False`, run `apply_extractor`, reject blank/oversized rows as unparseable, and tokenize via `sql_tokenizer` (`include_semicolons`, `strip_comments`, `max_query_chars`). Under `canonicalize=True`, classify post-parse canonicalize errors as unparseable, drop failed references, skip pairs with invalid prediction or zero valid references (`None`), implement Papineni smoothing (`(m + 1.0) / (p + 1.0)`), return `bleu=0.0` if `translation_length == 0` (`reference_length > 0`), `None` score fields if `scored_count == 0` or `reference_length == 0`, smoothed `per_example_bleu`, and `"translation_length"`.
10. **`src/group_by_vibes/valid_sql_rate.py` `(Create)`** — *`Valid SQL Rate`:* Implement `ValidSqlRate` (`add`, `add_batch`, `compute`, `reset`) and `compute_valid_sql_rate(data, schemas=None, *, dialect, extractor, max_query_chars, max_input_chars, include_per_example) -> dict[str, Any]` matching Public API Signatures defaults. Normalize each non-`None` schema mapping per row; ground syntax-valid query AST rows (`schema_evaluated_count`); keep `schema_evaluated_count + schema_skipped_count == num_examples`; zero-initialize all 8 `failure_counts` keys in precedence order (`extraction_failed` [incl. `max_input_chars`] / `extraction_error` -> `empty_input` -> `multi_statement` -> `command_fallback` -> `syntax_error` [incl. `max_query_chars` / scope errors] -> `unknown_table` -> `unknown_column`); output exact `per_example` dict schema (`syntax_valid`, `schema_valid`, `failure_reason`, `missing_tables`, `missing_columns`).
11. **`src/group_by_vibes/wildcard_detector.py` `(Create)`** — *`Wildcard Detector`:* Implement `WildcardDetector` (`add`, `add_batch`, `compute`, `reset`) and `compute_wildcard_detector(data, *, dialect, extractor, max_query_chars, max_input_chars, include_per_example) -> dict[str, Any]` matching Public API Signatures defaults. Skip unparseable rows (`None`); walk `ast.find_all(exp.Select)` inside `(SqlglotError, RecursionError)`, excluding only direct `exp.Exists.this` targets (`Select` or `Subquery(Select)`) while scanning deeper nested subqueries, unwrapping `exp.Alias` on direct `sel.expressions`, emitting `1` iff direct `exp.Star` (including `* EXCEPT`/`EXCLUDE`) or `exp.Column(this=exp.Star())` exists (ignoring `RETURNING *`, `COUNT(*)`, `*` math -> `0`).
12. **`src/sql_bleu.py` `(Delete)`** — *Prototype Removal:* Delete in Phase 3 after permanent static golden snapshot tests pass.
13. **[README.md](../../README.md) `(Modify)`** — *Root Docs:* Replace stale `utils/sql_bleu.py` references with `group-by-vibes` installation, evaluator/`load()` examples, PostgreSQL/SQLite priority, runtime out-of-band notes, Chen et al. (2019) citation, and `examples/` link.
14. **`tests/test_dialects_extract_parser.py` `(Create)`** — *Core Tests:* Cover `normalize_dialect` (whitespace/case/aliases/errors), quoted-dot and 3-part `split_qualified_table` without `parse_one`, regex (`re.search`, `group(1)`, `re.error`, `groups > 1`) and callable (`str | None`) extractor errors, `max_input_chars` (`extractor is not None`), `extract_fenced_or_last_sql` (unclosed `<think>`, trailing prose termination), `empty_input`, trailing `;`, `multi_statement`, nested `exp.Command`, `max_query_chars`, `RecursionError`, `max_query_chars`/`max_input_chars` `ValueError` validation, and `tokenize_sql` string/quote handling.
15. **`tests/test_canonicalize_and_sql_bleu.py` `(Create)`** — *Metric Tests:* Static golden canonicalization/BLEU snapshots matching legacy `src/sql_bleu.py` (captured before Phase 3 deletion without importing `src/sql_bleu.py`), detached `And`/`Or` and `_LOGICAL_CLASSES` fixes, deterministic lowercase canonicalization on Postgres/SQLite/Snowflake preserving quoted case, subquery-free `exp.In` sorting, `'O''Reilly'` quoting, Postgres `arr[1]` tokenization, unparseable/empty skipping (`scored_count == 0`, `reference_length == 0`, `translation_length == 0` -> `0.0`), multi-reference `Sequence[str]`, `canonicalize=False`, `strip_comments`, `include_semicolons`, Papineni `smooth`, smoothed `per_example_bleu`, `"translation_length"`, empty `compute()` `ValueError`, and reset.
16. **`tests/test_valid_sql_rate.py` `(Create)`** — *Grounding Tests:* Test `data=`/`schemas=`, `None` skipped vs. `{}` vs. `"partial"`, `schema_evaluated_count + schema_skipped_count == num_examples`, non-query `build_scope(ast) is None` skipping, SQLite `rowid`/`_rowid_`/`oid`, CTE `exp.TableAlias.columns`, `scope.traverse()` + `external_columns`, mixed opaque+physical scopes, 3-part `(catalog, db, table)` keys without silent wrong-db fallback, `bad_alias.*` & `bad_alias.col`, missing identifier sorting/dedup, zero-initialized 8-key `failure_counts`, `str` col iterable rejection, `schemas=dict`/length mismatches, empty `compute()`, and omission of raw SQL/literals from `per_example`.
17. **`tests/test_wildcard_detector.py` `(Create)`** — *Wildcard Tests:* Test `SELECT *`, `t.*`, `* EXCEPT`/`EXCLUDE`, subquery/CTE wildcards, and nested subquery inside `EXISTS (SELECT 1 FROM (SELECT * FROM t))` (`1`), `COUNT(*)`, `price * qty`, `RETURNING *`, and direct `EXISTS (SELECT * ...)` (`0`), unparseable/multi-statement skips (`None`), `include_per_example`, and empty `compute()`.
18. **`tests/test_packaging_isolation.py` `(Create)`** — *Isolation & Config Tests:* Assert zero `evaluate`/`datasets`/`pyarrow`/`torch`/`transformers` imports in `sys.modules`, `load()` typed dispatch and unknown-name `ValueError`, single-writer `compute_*` concurrency checks, and `max_query_chars`/`max_input_chars` validation (`None`, `0`, `-1`, `True`).
19. **`examples/bird_eval.py` `(Create)`** — *Benchmark CLI:* Implement `generate --output <jsonl> [--dialect sqlite|postgres|mysql] --seed 42 --sample-size 100 --max-new-tokens 2048 --model Qwen/Qwen2.5-Coder-3B-Instruct` (stratified by `difficulty` across `birdsql/bird_mini_dev` sorted by `question_id`, `baseline_inference/schemas/<dialect>.json` DDL parsed via `sqlglot.parse` into `{table: [cols]}`, `gold_sql = row["SQL"]`, `baseline_inference/prompt_template.txt`, `torch.bfloat16`, line-buffered per-row flush + `stderr` progress) and `score --input <jsonl> [--output <json>] [--dialect <dialect>]` (offline scoring across all 3 evaluators + `by_difficulty` via `extract_fenced_or_last_sql` with zero ML dependencies).
20. **`examples/fixtures/bird_sample.jsonl` `(Create)`** — *E2E Fixture:* 5 synthetic multi-table BIRD JSONL rows matching `#19` schema (`<think>` block, fenced SQL + trailing prose, wildcard projection, unclosed `<think>` fail, missing column) for core-only tests.
21. **`examples/README.md` `(Create)`** — *Example Docs:* Document `uv sync --extra examples`, `generate` vs. `score` flags, local 3B/1.5B guidance, JSONL schema, and `difficulty` breakdown.

### Verification Plan

*   **Compilation/Build:** `uv sync --group dev && uv build`, plus inspecting built wheel `METADATA` via Python `zipfile` to verify `Requires-Dist` lists only `sqlglot>=30.0.0,<31.0.0`.
*   **Unit & Snapshot Tests:** Phase 2 captures reference parity outputs from `src/sql_bleu.py` into static golden vectors in `tests/test_canonicalize_and_sql_bleu.py` before retiring `src/sql_bleu.py`; Phase 3+ runs `uv run pytest -v --cov=group_by_vibes --cov-branch` across all five permanent suites (`#14`–`#18`).
*   **Static Analysis & Linting:** `uv run ruff check src/ tests/ examples/` and `uv run mypy src/group_by_vibes/` exit `0`.
*   **Coverage Target:** `>= 90%` line coverage and `>= 85%` branch coverage on `src/group_by_vibes/`.
*   **End-to-End Core-Only Smoke Test:** `uv run python examples/bird_eval.py score --input examples/fixtures/bird_sample.jsonl` exits `0` without `examples` optional dependencies installed.

*Drafted with EGM*
