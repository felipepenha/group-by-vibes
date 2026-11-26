# SQL-BLEU

**`SqlBleu`** (`load("sql_bleu")`, `compute_sql_bleu`, `sentence_sql_bleu`) measures syntactic and structural similarity between predicted and reference SQL queries using **AST-canonicalized n-gram overlap**, adapting Chen et al. (EMNLP-IJCNLP 2019) onto `sqlglot` ASTs.

> **Reference:** Fuxiang Chen, Seung-won Hwang, Jaegul Choo, Jung-Woo Ha, and Sunghun Kim (2019). [*NL2pSQL: Generating Pseudo-SQL Queries from Under-Specified Natural Language Questions*](https://aclanthology.org/D19-1262/). EMNLP-IJCNLP 2019, pages 2603–2613. DOI: `10.18653/v1/D19-1262`.

---

## How SQL-BLEU Works

### 1. Mathematical Formulation

Given tokenized canonical predictions and reference queries up to order \(K = \text{max\_order}\) (default `4`), SQL-BLEU computes clipped n-gram precision \(p_n\) aggregated across all scored examples:

$$
p_n =
\begin{cases}
\dfrac{\sum_i \text{clipped\_matches}_{i,n}}{\sum_i \text{possible}_{i,n}} & \text{if } \text{smooth} = \text{False} \\[1em]
\dfrac{1 + \sum_i \text{clipped\_matches}_{i,n}}{1 + \sum_i \text{possible}_{i,n}} & \text{if } \text{smooth} = \text{True}
\end{cases}
$$

Let \(C = \sum_i |\text{pred}_i|\) (`translation_length`) and \(R = \sum_i |\text{ref}^*_i|\) (`reference_length`, using the closest-length valid reference per example, breaking ties toward the shortest reference). The brevity penalty \(\text{BP}\) and final score are:

$$
\text{BP} =
\begin{cases}
1.0 & \text{if } C > R \\
\exp\!\left(1 - \dfrac{R}{C}\right) & \text{otherwise}
\end{cases}
\qquad
\text{SQL-BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{K} \frac{1}{K} \log p_n\right)
$$

* When `min(p_n) == 0` (common on unsmoothed short batches with zero 4-gram overlap), the geometric mean collapses to `0.0`.
* **`sentence_sql_bleu` and `per_example_bleu`:** Single-example BLEU always uses add-1 smoothing (`smooth=True` by default on `sentence_sql_bleu`, and unconditionally for each non-`None` entry in `per_example_bleu`) so individual short queries retain graded signal even without exact 4-gram matches.

### 2. Adaptations to `sqlglot` ASTs

Raw text BLEU penalizes harmless SQL formatting differences—casing, indentation, comment noise, or reordered associative clauses. Before tokenization, `SqlBleu` normalizes both predictions and references through `canonicalize.py` ([source](https://github.com/felipepenha/group-by-vibes/blob/main/src/group_by_vibes/canonicalize.py)):

1. **Dialect-aware AST parsing (`parse_sql`):** Parses each SQL string under the configured `sqlglot` dialect, rejecting empty strings, multi-statement batches, and `exp.Command` parser fallbacks.
2. **Identifier normalization (`normalize_identifiers=True`):** Folds **unquoted** `exp.Identifier` names to lowercase while preserving **exact casing** on **quoted** identifiers (for example `"MixedCase"` or `"Order Date"`); quote delimiters normalize to the target dialect's primary identifier quote (so `` `Order Date` `` and `[Order Date]` under `"sqlite"` both render `"Order Date"`).
3. **Bottom-up commutative predicate sorting (`sort_commutative=True`):**
   * Normalizes binary `=` (`exp.EQ`) and `!=` / `<>` (`exp.NEQ`) operands lexicographically by rendered SQL (`id = 1` becomes `1 = id`).
   * Sorts literal-list `IN (...)` expressions (`status IN ('b', 'a')` becomes `status IN ('a', 'b')`) while leaving `IN (SELECT ...)` subqueries intact.
   * Flattens associative `AND` / `OR` trees and sorts their operands lexicographically while **preserving parentheses around mixed logical operators** (`(a OR b) AND c` never loses precedence).
4. **Comment and semicolon normalization (`strip_comments=True`, `include_semicolons=False`):** Drops `--` and `/* ... */` comments and trailing `;` delimiters before `sqlglot` AST rendering and tokenization.
5. **Quote-preserving tokenization (`sqlglot.tokenize`):** Produces dialect-aware syntactic tokens (re-quoting `TokenType.STRING` literals cleanly, falling back to a bounded linear regex tokenizer on tokenization edge cases).

!!! note "SQL-BLEU vs. Component/Exact Matching"
    Unlike `ComponentMatching` and `ExactMatching`, `SqlBleu` does **not** mask literal values (`'active'` vs. `'inactive'`) or collapse foreign keys—it stays faithful to Chen et al. (2019)'s token-level n-gram comparison over canonicalized SQL.

### 3. Unparseable Query Handling

Unparseable predictions or references are **excluded** from n-gram accumulators (`scored_count`), while `parsed_count`, `unparseable_count`, `parse_success_rate`, and `reference_parse_success_rate` are always reported explicitly alongside `bleu` so silent survivorship bias is immediately visible. If `scored_count == 0` or `reference_length == 0`, `bleu`, `precisions`, `brevity_penalty`, and `length_ratio` return `None`.

---

## API Reference

### Class & Stateless Signatures

* **`gbv.load("sql_bleu", **kwargs) -> SqlBleu`** / **`SqlBleu(...)`**
* **`compute_sql_bleu(predictions, references, *, ...)`**
* **`sentence_sql_bleu(prediction, reference, *, smooth=True, ...)`**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"postgres"` | Target `sqlglot` dialect (or `"postgresql"`, `"sqlite3"`, ANSI `""`). |
| `max_order` | `int` | `4` | Maximum n-gram order (`>= 1`). |
| `smooth` | `bool` | `False` (`True` on `sentence_sql_bleu`) | Apply add-1 smoothing; on `SqlBleu`, pass to `compute(smooth=...)`. |
| `canonicalize` | `bool` | `True` | Parse and canonicalize ASTs before tokenization; if `False`, tokenizes raw SQL text directly. |
| `sort_commutative` | `bool` | `True` | Sort operands of `=`, `!=`, literal `IN (...)`, `AND`, and `OR`. |
| `normalize_identifiers` | `bool` | `True` | Lowercase unquoted identifiers while preserving quoted identifier case. |
| `strip_comments` | `bool` | `True` | Strip SQL comments before tokenization. |
| `include_semicolons` | `bool` | `False` | Retain `;` tokens during canonical rendering and tokenization. |
| `extractor` | `ExtractorLike` | `None` | Optional extractor run on predictions only (`extract_fenced_or_last_sql`, regex, or callable). |
| `max_query_chars` | `int \| None` | `65_536` | Maximum character length per SQL query before rejecting/failing parse. |
| `max_input_chars` | `int \| None` | `131_072` | Maximum raw completion character length accepted by `extractor`. |
| `include_per_example` | `bool` | `False` | Include `per_example_bleu: list[float \| None]` (add-1 smoothed) in `compute()` output (not accepted by `sentence_sql_bleu`). |

### `compute()` Output Schema

| Key | Type | Semantics |
| :--- | :--- | :--- |
| `bleu` | `float \| None` | Corpus SQL-BLEU score in `[0.0, 1.0]`, or `None` when `scored_count == 0` / `reference_length == 0`. |
| `precisions` | `list[float] \| None` | Clipped n-gram precision for orders `1..max_order` (length `max_order`), or `None`. |
| `brevity_penalty` | `float \| None` | Corpus brevity penalty in `(0.0, 1.0]` (or `0.0` if `translation_length == 0`), or `None`. |
| `length_ratio` | `float \| None` | `translation_length / reference_length`, or `None`. |
| `translation_length` | `int` | Total canonical prediction tokens across `scored_count` pairs. |
| `reference_length` | `int` | Sum of closest reference token lengths across `scored_count` pairs. |
| `scored_count` | `int` | Number of rows where the prediction and at least one reference parsed cleanly. |
| `parsed_count` | `int` | Number of predictions passing extraction, char bounds, and AST parsing. |
| `unparseable_count` | `int` | Number of predictions failing extraction, char bounds, or AST parsing. |
| `parse_success_rate` | `float` | `parsed_count / num_examples`. |
| `reference_parse_success_rate` | `float` | Parsed reference queries divided by total supplied reference queries. |
| `num_examples` | `int` | Total rows passed via `add()` / `add_batch()`. |
| `dialect` | `str` | Normalized `sqlglot` dialect name. |
| `max_order` | `int` | Configured maximum n-gram order. |
| `smooth` | `bool` | Whether add-1 smoothing was applied to corpus precisions. |
| `per_example_bleu` | `list[float \| None]` | *(Only when `include_per_example=True`)* Per-row add-1 smoothed BLEU, or `None` if unparseable. |

---

## Examples

### Corpus, Commutative Canonicalization & Multi-Reference Scoring

```python
import group_by_vibes as gbv

metric = gbv.load("sql_bleu", dialect="postgres", include_per_example=True)
metric.add_batch(
    predictions=[
        "SELECT name, age FROM users WHERE status = 'active' AND 1 = role",
        "SELECT name FROM users WHERE id IN (3, 1, 2)",
    ],
    references=[
        ["SELECT name, age FROM users WHERE role = 1 AND status = 'active'"],
        [
            "SELECT email FROM accounts",
            "SELECT name FROM users WHERE id IN (1, 2, 3)",
        ],
    ],
)
res = metric.compute()
assert res["bleu"] == 1.0
assert res["parse_success_rate"] == 1.0
assert res["per_example_bleu"] == [1.0, 1.0]
```

### Single-Pair `sentence_sql_bleu` With Raw LLM Completions

```python
import group_by_vibes as gbv

raw_completion = (
    "<think>\nUse active status filter.\n</think>\n"
    "```sql\nSELECT name FROM users WHERE status = 'active';\n```"
)

score = gbv.sentence_sql_bleu(
    raw_completion,
    "SELECT name FROM users WHERE status = 'active'",
    dialect="sqlite",
    extractor=gbv.extract_fenced_or_last_sql,
)
assert score == 1.0
```
