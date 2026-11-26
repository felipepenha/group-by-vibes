# External Reference Benchmark (`examples/`)

This directory holds reference evaluation scripts for **[birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev)** across **SQLite**, **PostgreSQL**, and **MySQL**, isolated from `src/group_by_vibes/` so ML and Hub libraries remain outside the core runtime package.

## Subcommands

### 1. Generate Model Outputs (`generate`, requires `examples` extras)

For local laptops and Apple Silicon (16–24 GB unified RAM), use 3B models (`Qwen/Qwen2.5-Coder-3B-Instruct`, `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`, or `griffith-bigdata/FINER-SQL-3B-v2`, each ~6.2 GB in `torch.bfloat16`) or `Qwen/Qwen2.5-Coder-1.5B-Instruct` (~3 GB in `torch.bfloat16`):

```bash
uv sync --extra examples

# Baseline coding model
uv run python examples/bird_eval.py generate \
  --output examples/logs/bird_qwen_3b.jsonl \
  --dialect sqlite \
  --model Qwen/Qwen2.5-Coder-3B-Instruct

# SQL-specialized 3B fine-tune
uv run python examples/bird_eval.py generate \
  --output examples/logs/bird_xiyansql_3b.jsonl \
  --dialect sqlite \
  --model XGenerationLab/XiYanSQL-QwenCoder-3B-2504

# RL-trained SQL 3B model (FINER-SQL)
uv run python examples/bird_eval.py generate \
  --output examples/logs/bird_finer_sql_3b.jsonl \
  --dialect sqlite \
  --model griffith-bigdata/FINER-SQL-3B-v2
```

Omitting `--sample-size` evaluates the full `bird_mini_dev` split (`500` rows for SQLite). You may optionally pass `--sample-size <N>` to run a deterministic stratified subset (for example, `--sample-size 100` with `--seed 42` for a faster benchmark pass, or `--sample-size 3` for a quick probe), or pass `--dialect postgres` or `--dialect mysql` for other dialects. `--model` accepts any Hugging Face repo ID or the built-in presets (`qwen-3b`, `xiyansql-3b`, `finer-sql-3b`).

Each row flushes to disk immediately and logs `[i/N]` progress to `stderr`, so you can follow progress live with:

```bash
tail -f examples/logs/bird_qwen_3b.jsonl
# or: tail -f examples/logs/bird_xiyansql_3b.jsonl
# or: tail -f examples/logs/bird_finer_sql_3b.jsonl
```

* **Sampling:** Evaluates the full split sorted by `question_id` by default, or takes a deterministic stratified sample across `difficulty` (`simple`, `moderate`, `challenging`) when `--sample-size` is specified.
* **Schema Mapping:** Parses `baseline_inference/schemas/<dialect>.json` `CREATE TABLE` DDL via `sqlglot.parse` into per-database `{table: [columns]}` dicts for `ValidSqlRate` / matching qualification and `foreign_keys` (`[child_table.col, parent_table.col]` pairs) for `ComponentMatching` / `ExactMatching` equivalence collapse.
* **Prompting:** Uses `bird_mini_dev`'s `baseline_inference/prompt_template.txt` with `question` and `evidence`.

### 2. Offline Re-Scoring (`score`, zero ML dependencies required)

```bash
uv sync --group dev

# Smoke check on the bundled 5-row fixture
uv run python examples/bird_eval.py score \
  --input examples/fixtures/bird_sample.jsonl \
  --output examples/logs/bird_sample_scores.json

# Score full or sampled model runs to JSON
uv run python examples/bird_eval.py score \
  --input examples/logs/bird_qwen_3b.jsonl \
  --output examples/logs/bird_qwen_3b_scores.json

uv run python examples/bird_eval.py score \
  --input examples/logs/bird_xiyansql_3b.jsonl \
  --output examples/logs/bird_xiyansql_3b_scores.json

uv run python examples/bird_eval.py score \
  --input examples/logs/bird_finer_sql_3b.jsonl \
  --output examples/logs/bird_finer_sql_3b_scores.json
```

Omit `--output` to print the JSON report to `stdout` instead of writing a file. Scoring evaluates saved JSONL rows across `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, `ComponentMatching`, `ExactMatching`, and per-`difficulty` slices using `extract_fenced_or_last_sql`.

## Saved JSONL Schema

Each row stores: `question_id`, `db_id`, `difficulty`, `dialect`, `question`, `evidence`, `schema`, `foreign_keys` (emitted on newly generated logs), `gold_sql`, `raw_completion`, `truncated`, `seed`, `model`, `extractor`, `group_by_vibes_version`, and `sqlglot_version`.

## Comparative Evaluation Report (SQLite, Full Split, n=500)

Evaluation on `birdsql/bird_mini_dev` (`sqlite`, `n = 500`): `griffith-bigdata/FINER-SQL-3B-v2` vs. `Qwen/Qwen2.5-Coder-3B-Instruct` vs. `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`.

### Run Metadata

* **Dataset & Split:** `birdsql/bird_mini_dev` (`sqlite`, full `500` rows: `460` challenging, `32` moderate, `8` simple across `11` databases)
* **Models Compared:** `griffith-bigdata/FINER-SQL-3B-v2` vs. `Qwen/Qwen2.5-Coder-3B-Instruct` vs. `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`
* **Extractor & Versions:** `extract_fenced_or_last_sql`, `group_by_vibes` `0.1.3`, `sqlglot` `30.33.0`
* **Score Reports:** `examples/logs/bird_finer_sql_3b_scores.json`, `examples/logs/bird_qwen_3b_scores.json`, `examples/logs/bird_xiyansql_3b_scores.json`

### Headline Comparison

| Metric | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| --- | ---: | ---: | ---: |
| **SQL-BLEU** | `0.2646` | `0.2130` | `0.1807` |
| **Parsed Queries (SQL-BLEU)** | `495/500` (`99.0%`) | `478/500` (`95.6%`) | `494/500` (`98.8%`) |
| **SQL-BLEU precisions (1 / 2 / 3 / 4)** | `0.782 / 0.551 / 0.397 / 0.283` | `0.714 / 0.447 / 0.278 / 0.171` | `0.677 / 0.398 / 0.230 / 0.129` |
| **AST Token Length (Pred / Gold)** | `28,062 / 44,147` | `28,252 / 42,364` | `29,311 / 44,071` |
| **Length Ratio / Brevity Penalty** | `0.636` / `0.564` | `0.667` / `0.607` | `0.665` / `0.604` |
| **Per-example SQL-BLEU (min / p25 / median / mean / p75 / max)** | `0.0004 / 0.1431 / 0.2539 / 0.2874 / 0.3967 / 1.0000` | `0.0013 / 0.0948 / 0.1961 / 0.2351 / 0.3177 / 1.0000` | `0.0004 / 0.0895 / 0.1640 / 0.2074 / 0.2695 / 1.0000` |
| **Exact Canonical Matches (`BLEU = 1.0`)** | `1` | `1` | `1` |
| **Exact Match Accuracy** | `1.2%` (`6/500`) | `1.2%` (`6/500`) | `0.8%` (`4/500`) |
| **Valid SQL Rate** | `84.2%` (`421/500`) | `56.2%` (`281/500`) | `73.6%` (`368/500`) |
| **Wildcard Rate (`SELECT *`)** | `0.0%` (`0/495`) | `0.0%` (`0/478`) | `0.0%` (`0/494`) |
| **Truncated Completions** | `4` | `1` | `2` |

### Component Matching F1

| Component | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| --- | ---: | ---: | ---: |
| `select` | `0.279` | `0.219` | `0.245` |
| `select(no AGG)` | `0.281` | `0.219` | `0.254` |
| `where` | `0.126` | `0.102` | `0.097` |
| `where(no OP)` | `0.155` | `0.129` | `0.121` |
| `group(no Having)` | `0.420` | `0.352` | `0.389` |
| `group` | `0.303` | `0.215` | `0.253` |
| `order` | `0.097` | `0.085` | `0.092` |
| `and/or` | `0.640` | `0.655` | `0.612` |
| `IUEN` | `0.000` | `0.000` | `0.000` |
| `keywords` | `0.241` | `0.271` | `0.247` |
| `cte` | `0.000` | `0.000` | `0.000` |

### Breakdown by Difficulty

| Difficulty | Rows | FINER-SQL SQL-BLEU | Qwen SQL-BLEU | XiYanSQL SQL-BLEU | FINER-SQL Valid SQL | Qwen Valid SQL | XiYanSQL Valid SQL | FINER-SQL Exact Match | Qwen Exact Match | XiYanSQL Exact Match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `simple` | 8 | `0.3549` | `0.3167` | `0.2906` | `87.5%` (`7`) | `87.5%` (`7`) | `100.0%` (`8`) | `12.5%` (`1`) | `25.0%` (`2`) | `12.5%` (`1`) |
| `moderate` | 32 | `0.3058` | `0.2783` | `0.2526` | `96.9%` (`31`) | `71.9%` (`23`) | `87.5%` (`28`) | `6.2%` (`2`) | `3.1%` (`1`) | `0.0%` (`0`) |
| `challenging` | 460 | `0.2619` | `0.2091` | `0.1765` | `83.3%` (`383`) | `54.6%` (`251`) | `72.2%` (`332`) | `0.7%` (`3`) | `0.7%` (`3`) | `0.7%` (`3`) |

### Breakdown by Database

| Database | Rows | FINER-SQL SQL-BLEU | Qwen SQL-BLEU | XiYanSQL SQL-BLEU | FINER-SQL Valid SQL | Qwen Valid SQL | XiYanSQL Valid SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `california_schools` | 49 | `0.1682` | `0.1607` | `0.1416` | `46.9%` (`23`) | `22.4%` (`11`) | `55.1%` (`27`) |
| `card_games` | 69 | `0.3222` | `0.3005` | `0.2157` | `92.8%` (`64`) | `66.7%` (`46`) | `85.5%` (`59`) |
| `codebase_community` | 31 | `0.2364` | `0.2172` | `0.1852` | `93.5%` (`29`) | `61.3%` (`19`) | `80.6%` (`25`) |
| `debit_card_specializing` | 14 | `0.2643` | `0.2496` | `0.2123` | `92.9%` (`13`) | `92.9%` (`13`) | `85.7%` (`12`) |
| `european_football_2` | 31 | `0.1137` | `0.0895` | `0.1042` | `87.1%` (`27`) | `48.4%` (`15`) | `71.0%` (`22`) |
| `financial` | 46 | `0.2643` | `0.2330` | `0.1843` | `80.4%` (`37`) | `45.7%` (`21`) | `54.3%` (`25`) |
| `formula_1` | 55 | `0.2930` | `0.1995` | `0.1492` | `89.1%` (`49`) | `58.2%` (`32`) | `78.2%` (`43`) |
| `student_club` | 37 | `0.2635` | `0.2388` | `0.1523` | `100.0%` (`37`) | `70.3%` (`26`) | `64.9%` (`24`) |
| `superhero` | 50 | `0.3829` | `0.2117` | `0.2396` | `92.0%` (`46`) | `62.0%` (`31`) | `92.0%` (`46`) |
| `thrombosis_prediction` | 86 | `0.2573` | `0.2020` | `0.1891` | `75.6%` (`65`) | `55.8%` (`48`) | `69.8%` (`60`) |
| `toxicology` | 32 | `0.3384` | `0.2639` | `0.2376` | `96.9%` (`31`) | `59.4%` (`19`) | `78.1%` (`25`) |

### Failure Breakdown (`valid_sql_rate.failure_counts`)

| Failure Reason | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| --- | ---: | ---: | ---: |
| `unknown_column` | `67` | `184` | `121` |
| `syntax_error` | `5` | `19` | `5` |
| `unknown_table` | `7` | `13` | `5` |
| `multi_statement` | `0` | `3` | `0` |
| `extraction_failed` | `0` | `0` | `1` |
| `extraction_error` | `0` | `0` | `0` |
| `empty_input` | `0` | `0` | `0` |
| `command_fallback` | `0` | `0` | `0` |

### SQL Style & Structural Patterns (`n = 500`)

| Pattern | Gold SQL | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| --- | ---: | ---: | ---: | ---: |
| `T1`/`T2` table aliases | `93` (`18.6%`) | `0` (`0.0%`) | `286` (`57.2%`) | `427` (`85.4%`) |
| Explicit `INNER JOIN` | `129` (`25.8%`) | `0` (`0.0%`) | `36` (`7.2%`) | `348` (`69.6%`) |
| `WITH` CTEs | `150` (`30.0%`) | `1` (`0.2%`) | `3` (`0.6%`) | `2` (`0.4%`) |
| `CAST(...)` expressions | `145` (`29.0%`) | `11` (`2.2%`) | `11` (`2.2%`) | `39` (`7.8%`) |
| `CASE` / `IIF` conditionals | `116` (`23.2%`) | `81` (`16.2%`) | `89` (`17.8%`) | `93` (`18.6%`) |
| Mean extracted query length (chars) | `371.6` | `244.1` | `287.7` | `251.6` |

### Key Comparative Takeaways

1. **`FINER-SQL` leads `Valid SQL Rate` (`84.2%`, `421/500`), `SQL-BLEU` (`0.2646`), and major structural component clauses:** Across the identical `500` SQLite questions, `FINER-SQL` outpaces `XiYanSQL` (`73.6%`, `+10.6 pp` / `+53` valid queries) and `Qwen` (`56.2%`, `+28.0 pp` / `+140` valid queries) on `Valid SQL Rate`, leads `SQL-BLEU` over `Qwen` (`0.2130`, `+0.0516`) and `XiYanSQL` (`0.1807`, `+0.0839`), and posts the highest `Component Matching` F1 on `select` (`0.279` vs `0.245` / `0.219`), `select(no AGG)` (`0.281`), `where` (`0.126`), `where(no OP)` (`0.155`), `group(no Having)` (`0.420`), `group` (`0.303`), and `order` (`0.097`). `Qwen` leads on `and/or` (`0.655` vs `0.640` / `0.612`) and `keywords` (`0.271` vs `0.241` / `0.247`), while all three models score `0.000` F1 on `IUEN` and `cte` (consistent with emitting `0.2%`–`0.6%` CTEs against `30.0%` in gold SQL).
2. **`Exact Match Accuracy` stays near `1%` across all three 3B models (`bird_mini_dev` `0.8%`–`1.2%`):** `FINER-SQL` and `Qwen` tie overall at `1.2%` (`6/500`), ahead of `XiYanSQL` at `0.8%` (`4/500`), with `reference_parse_success_rate = 1.0` (`500/500` gold queries decomposed across all three runs). By difficulty, `Qwen` reaches `25.0%` (`2/8`) on `simple`, `FINER-SQL` reaches `6.2%` (`2/32`) on `moderate`, and all three match on `0.7%` (`3/460`) on `challenging` (`92%` of the split). Even under Spider value-blind defaults (`ignore_values=True`), BIRD queries mix multi-table joins, nested conditionals (`CASE`/`IIF`), literal arithmetic (`1.0 * a / b`), and CTEs (`30.0%`), so structural AST skeletons rarely align end-to-end despite higher clause-level F1 on `select` and `group`. Note that the checked-in `v0.0.1` `500`-row JSONL logs predate DDL `foreign_keys` extraction (`foreign_keys_applied_count == 0`), unlike newly generated logs.
3. **Error reduction and 3-way agreement:** `FINER-SQL` cuts `unknown_column` failures to `67` (vs `121` for `XiYanSQL` and `184` for `Qwen`), matches `XiYanSQL` on low `syntax_error` counts (`5` each vs `19` for `Qwen`), and records `0` `multi_statement` or `extraction_failed` errors (`unknown_table`: `7` vs `5` for `XiYanSQL` and `13` for `Qwen`). Across all `500` questions, `221` (`44.2%`) pass for all three models, `37` (`7.4%`) fail for all three, and `FINER-SQL` uniquely passes `51` queries (`10.2%`) where both `Qwen` and `XiYanSQL` fail (vs `24` unique passes for `XiYanSQL` and `2` for `Qwen`); head-to-head on `Valid SQL Rate`, `FINER-SQL` beats `XiYanSQL` `93` to `40` (`328` both valid) and beats `Qwen` `158` to `18` (`263` both valid). On per-example `SQL-BLEU`, `FINER-SQL` holds the sole top score on `249` questions (`49.8%`) vs `129` (`25.8%`) for `Qwen`, `108` (`21.6%`) for `XiYanSQL`, and `14` ties (`2.8%`).
4. **SQL style drives n-gram precision differences:** `FINER-SQL` avoids synthetic `T1`/`T2` aliases (`0.0%` vs `57.2%` for `Qwen`, `85.4%` for `XiYanSQL`, `18.6%` in gold) and explicit `INNER JOIN` boilerplate (`0.0%` vs `7.2%` for `Qwen`, `69.6%` for `XiYanSQL`, `25.8%` in gold), lifting `SQL-BLEU` n-gram precisions at every order (`0.782 / 0.551 / 0.397 / 0.283` vs `0.714 / 0.447 / 0.278 / 0.171` for `Qwen` and `0.677 / 0.398 / 0.230 / 0.129` for `XiYanSQL`) despite emitting the shortest queries (`244.1` chars, brevity penalty `0.564`). Between `Qwen` and `XiYanSQL`, that same alias and join verbosity explains why `Qwen` retains a higher `SQL-BLEU` (`0.2130` vs `0.1807`) despite lower `Valid SQL Rate` (`56.2%` vs `73.6%`).
5. **Shared structural brevity and zero wildcards:** All three 3B models emit ~33–36% fewer AST tokens than gold SQL (`length_ratio` `0.636`–`0.667`, `brevity_penalty` `0.564`–`0.607`), rarely generating `WITH` CTEs (`0.2%`–`0.6%` vs `30.0%` in gold) or `CAST(...)` expressions (`2.2%`–`7.8%` vs `29.0%` in gold), and all three emit `0.0%` projection wildcards (`SELECT *`). All `4` truncated `FINER-SQL` completions still extract to valid SQL via `extract_fenced_or_last_sql` because a fenced SQL block appears before trailing reasoning hits `max_new_tokens`.
6. **Metric complementarity:** `Valid SQL Rate`, `SQL-BLEU`, `ComponentMatching`, and `ExactMatching` capture orthogonal signals—schema grounding vs. AST token overlap vs. per-clause F1 vs. whole-query structural equivalence—and complement execution accuracy (`EX`).
