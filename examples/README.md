# External Reference Benchmark (`examples/`)

This directory holds reference evaluation scripts for **[birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev)** across **SQLite**, **PostgreSQL**, and **MySQL**, isolated from `src/group_by_vibes/` so ML and Hub libraries remain outside the core runtime package.

## Subcommands

### 1. Generate Model Outputs (`generate`, requires `examples` extras)

For local laptops and Apple Silicon (16–24 GB unified RAM), use 3B models (`Qwen/Qwen2.5-Coder-3B-Instruct` or the SQL-specialized `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`, each ~6.2 GB in `torch.bfloat16`) or `Qwen/Qwen2.5-Coder-1.5B-Instruct` (~3 GB in `torch.bfloat16`):

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
```

Omitting `--sample-size` evaluates the full `bird_mini_dev` split (`500` rows for SQLite). You may optionally pass `--sample-size <N>` to run a deterministic stratified subset (for example, `--sample-size 100` with `--seed 42` for a faster benchmark pass, or `--sample-size 3` for a quick probe), pass `--dialect postgres` or `--dialect mysql` for other dialects, or use `Snowflake/Arctic-Text2SQL-R1-7B` (~15.2 GB BF16) on dedicated 24 GB+ CUDA GPUs.

Each row flushes to disk immediately and logs `[i/N]` progress to `stderr`, so you can follow progress live with:

```bash
tail -f examples/logs/bird_qwen_3b.jsonl
# or: tail -f examples/logs/bird_xiyansql_3b.jsonl
```

* **Sampling:** Evaluates the full split sorted by `question_id` by default, or takes a deterministic stratified sample across `difficulty` (`simple`, `moderate`, `challenging`) when `--sample-size` is specified.
* **Schema Mapping:** Parses `baseline_inference/schemas/<dialect>.json` `CREATE TABLE` DDL via `sqlglot.parse` into per-database `{table: [columns]}` dicts for `ValidSqlRate`.
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
```

Omit `--output` to print the JSON report to `stdout` instead of writing a file. Scoring evaluates saved JSONL rows across `SqlBleu`, `ValidSqlRate`, `WildcardDetector`, and per-`difficulty` slices using `extract_fenced_or_last_sql`.

## Saved JSONL Schema

Each row stores: `question_id`, `db_id`, `difficulty`, `dialect`, `question`, `evidence`, `schema`, `gold_sql`, `raw_completion`, `truncated`, `seed`, `model`, `extractor`, `group_by_vibes_version`, and `sqlglot_version`.

## Comparative Evaluation Report (SQLite, Full Split, n=500)

Paired full-split evaluation on `birdsql/bird_mini_dev` (`sqlite`), comparing `Qwen/Qwen2.5-Coder-3B-Instruct` vs. `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` over the identical `500` questions.

### Run Metadata

* **Dataset & Split:** `birdsql/bird_mini_dev` (`sqlite`, full `500` rows: `460` challenging, `32` moderate, `8` simple across `11` databases)
* **Models Compared:** `Qwen/Qwen2.5-Coder-3B-Instruct` vs. `XGenerationLab/XiYanSQL-QwenCoder-3B-2504`
* **Extractor & Versions:** `extract_fenced_or_last_sql`, `group_by_vibes` `0.1.0`, `sqlglot` `30.33.0`
* **Score Reports:** `examples/logs/bird_qwen_3b_scores.json`, `examples/logs/bird_xiyansql_3b_scores.json`

### Headline Comparison

| Metric | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` | Delta (`XiYan - Qwen`) |
| --- | ---: | ---: | ---: |
| **SQL-BLEU** | `0.2130` | `0.1807` | `-0.0324` |
| **Parsed Queries (SQL-BLEU)** | `478/500` (`95.6%`) | `494/500` (`98.8%`) | `+3.2 pp` (`+16`) |
| **SQL-BLEU precisions (1 / 2 / 3 / 4)** | `0.714 / 0.447 / 0.278 / 0.171` | `0.677 / 0.398 / 0.230 / 0.129` | `-0.037 / -0.049 / -0.048 / -0.042` |
| **AST Token Length (Pred / Gold)** | `28,252 / 42,364` | `29,311 / 44,071` | `+1,059` pred tokens |
| **Length Ratio / Brevity Penalty** | `0.667` / `0.607` | `0.665` / `0.604` | `-0.002` / `-0.002` |
| **Per-example SQL-BLEU (min / p25 / median / mean / p75 / max)** | `0.0013 / 0.0948 / 0.1961 / 0.2351 / 0.3177 / 1.0000` | `0.0004 / 0.0895 / 0.1640 / 0.2074 / 0.2695 / 1.0000` | median `-0.0321`, mean `-0.0276` |
| **Exact Canonical Matches (`BLEU = 1.0`)** | `1` | `1` | `0` |
| **Valid SQL Rate** | `56.2%` (`281/500`) | `73.6%` (`368/500`) | `+17.4 pp` (`+87`) |
| **Wildcard Rate (`SELECT *`)** | `0.0%` (`0/478`) | `0.0%` (`0/494`) | `0.0 pp` |
| **Truncated Completions** | `1` | `2` | `+1` |

### Breakdown by Difficulty

| Difficulty | Rows | Qwen SQL-BLEU | XiYanSQL SQL-BLEU | Delta BLEU | Qwen Valid SQL | XiYanSQL Valid SQL | Delta Valid SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `simple` | 8 | `0.3167` | `0.2906` | `-0.0261` | `87.5%` (`7`) | `100.0%` (`8`) | `+12.5 pp` |
| `moderate` | 32 | `0.2783` | `0.2526` | `-0.0257` | `71.9%` (`23`) | `87.5%` (`28`) | `+15.6 pp` |
| `challenging` | 460 | `0.2091` | `0.1765` | `-0.0327` | `54.6%` (`251`) | `72.2%` (`332`) | `+17.6 pp` |

### Breakdown by Database

| Database | Rows | Qwen SQL-BLEU | XiYanSQL SQL-BLEU | Delta BLEU | Qwen Valid SQL | XiYanSQL Valid SQL | Delta Valid SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `california_schools` | 49 | `0.1607` | `0.1416` | `-0.0191` | `22.4%` (`11`) | `55.1%` (`27`) | `+32.7 pp` |
| `card_games` | 69 | `0.3005` | `0.2157` | `-0.0849` | `66.7%` (`46`) | `85.5%` (`59`) | `+18.8 pp` |
| `codebase_community` | 31 | `0.2172` | `0.1852` | `-0.0320` | `61.3%` (`19`) | `80.6%` (`25`) | `+19.4 pp` |
| `debit_card_specializing` | 14 | `0.2496` | `0.2123` | `-0.0374` | `92.9%` (`13`) | `85.7%` (`12`) | `-7.1 pp` |
| `european_football_2` | 31 | `0.0895` | `0.1042` | `+0.0147` | `48.4%` (`15`) | `71.0%` (`22`) | `+22.6 pp` |
| `financial` | 46 | `0.2330` | `0.1843` | `-0.0487` | `45.7%` (`21`) | `54.3%` (`25`) | `+8.7 pp` |
| `formula_1` | 55 | `0.1995` | `0.1492` | `-0.0503` | `58.2%` (`32`) | `78.2%` (`43`) | `+20.0 pp` |
| `student_club` | 37 | `0.2388` | `0.1523` | `-0.0865` | `70.3%` (`26`) | `64.9%` (`24`) | `-5.4 pp` |
| `superhero` | 50 | `0.2117` | `0.2396` | `+0.0279` | `62.0%` (`31`) | `92.0%` (`46`) | `+30.0 pp` |
| `thrombosis_prediction` | 86 | `0.2020` | `0.1891` | `-0.0129` | `55.8%` (`48`) | `69.8%` (`60`) | `+14.0 pp` |
| `toxicology` | 32 | `0.2639` | `0.2376` | `-0.0264` | `59.4%` (`19`) | `78.1%` (`25`) | `+18.8 pp` |

### Paired Head-to-Head Outcomes (Same 500 Questions)

#### Valid SQL Rate Agreement

| Outcome Category | Count | Share |
| --- | ---: | ---: |
| Both models valid | `237` | `47.4%` |
| XiYanSQL valid only | `131` | `26.2%` |
| Qwen valid only | `44` | `8.8%` |
| Neither model valid | `88` | `17.6%` |

#### Per-Example SQL-BLEU Head-to-Head (`n = 500`)

Unparseable predictions rank below parsed predictions; across the `473` jointly parsed queries, mean per-example `SQL-BLEU` is `0.2334` for Qwen vs. `0.2083` for XiYanSQL (`-0.0251`).

| Outcome Category | Count | Share |
| --- | ---: | ---: |
| Qwen higher BLEU | `266` | `53.2%` |
| XiYanSQL higher BLEU | `221` | `44.2%` |
| Tied | `13` | `2.6%` |

### Failure Breakdown (`valid_sql_rate.failure_counts`)

| Failure Reason | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` | Delta (`XiYan - Qwen`) |
| --- | ---: | ---: | ---: |
| `unknown_column` | `184` | `121` | `-63` |
| `syntax_error` | `19` | `5` | `-14` |
| `unknown_table` | `13` | `5` | `-8` |
| `multi_statement` | `3` | `0` | `-3` |
| `extraction_failed` | `0` | `1` | `+1` |
| `extraction_error` | `0` | `0` | `0` |
| `empty_input` | `0` | `0` | `0` |
| `command_fallback` | `0` | `0` | `0` |

### SQL Style & Structural Patterns (`n = 500`)

| Pattern | Gold SQL | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| --- | ---: | ---: | ---: |
| `T1`/`T2` table aliases | `93` (`18.6%`) | `286` (`57.2%`) | `427` (`85.4%`) |
| Explicit `INNER JOIN` | `129` (`25.8%`) | `36` (`7.2%`) | `348` (`69.6%`) |
| `WITH` CTEs | `150` (`30.0%`) | `3` (`0.6%`) | `2` (`0.4%`) |
| `CAST(...)` expressions | `145` (`29.0%`) | `11` (`2.2%`) | `39` (`7.8%`) |
| `CASE` / `IIF` conditionals | `116` (`23.2%`) | `89` (`17.8%`) | `93` (`18.6%`) |
| Mean extracted query length (chars) | `371.6` | `287.7` | `251.6` |

### Key Comparative Takeaways

1. **XiYanSQL wins on `Valid SQL Rate` (`+17.4 pp`, `73.6%` vs `56.2%`):** Across identical questions, `131` queries pass only for XiYanSQL vs `44` only for Qwen, winning on `9` of `11` databases (`california_schools` `+32.7 pp`, `superhero` `+30.0 pp`) while reducing `unknown_column` failures (`184` -> `121`), `unknown_table` (`13` -> `5`), and parse failures (`syntax_error` `19` -> `5`, `multi_statement` `3` -> `0`).
2. **`Qwen` scores higher on `SQL-BLEU` (`0.2130` vs `0.1807`) despite lower validity:** XiYanSQL standardizes on `T1`/`T2` aliases (`85.4%` vs Qwen `57.2%` vs gold `18.6%`) and explicit `INNER JOIN` (`69.6%` vs Qwen `7.2%` vs gold `25.8%`), which lowers token n-gram precision because `SQL-BLEU` does not alpha-rename table aliases. XiYanSQL also emits shorter queries (`251.6` vs `287.7` chars), occasionally dropping `WHERE` filters or `GROUP BY` clauses.
3. **Shared brevity penalty:** Both 3B models emit ~33% fewer AST tokens than gold SQL—rarely generating `WITH` CTEs (`0.6%` and `0.4%` vs `30.0%` in gold) or `CAST(...)` expressions (`2.2%` and `7.8%` vs `29.0%`)—yielding nearly identical brevity penalties (`0.607` vs `0.604`).
4. **Metric complementarity:** `Valid SQL Rate` and `SQL-BLEU` capture orthogonal signals (schema grounding vs. clause/lexical similarity to reference SQL) and complement execution accuracy (`EX`).
