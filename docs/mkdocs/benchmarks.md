# Reference Benchmarks (`birdsql/bird_mini_dev`)

All five evaluators are exercised together in the isolated benchmark harness under [`examples/`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/README.md) against **[`birdsql/bird_mini_dev`](https://huggingface.co/datasets/birdsql/bird_mini_dev)** (`sqlite`, full `500` rows: `460` challenging, `32` moderate, `8` simple across `11` databases).

> **Note:** ML inference dependencies (`transformers`, `torch`, `datasets`, `huggingface_hub`, `accelerate`) live strictly under `[project.optional-dependencies] examples` so the core package stays `sqlglot`-only. Offline `score` runs execute with zero ML dependencies.

---

## Offline Re-Scoring Command

```bash
# Score the bundled 5-row smoke fixture with core dependencies only
uv run python examples/bird_eval.py score \
  --input examples/fixtures/bird_sample.jsonl
```

See [`examples/README.md`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/README.md) and [`examples/bird_eval.py`](https://github.com/felipepenha/group-by-vibes/blob/main/examples/bird_eval.py) for `generate` / `score` CLI flags and DDL `{table: [columns]}` + `foreign_keys` extraction details.

---

## Comparative Summary (SQLite, `n = 500`)

Evaluated with `extract_fenced_or_last_sql`, `group_by_vibes` `0.0.1`, and `sqlglot` `30.33.0`:

| Metric | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| :--- | ---: | ---: | ---: |
| **SQL-BLEU** | `0.2646` | `0.2130` | `0.1807` |
| **Parsed Queries (SQL-BLEU)** | `495/500` (`99.0%`) | `478/500` (`95.6%`) | `494/500` (`98.8%`) |
| **Exact Match Accuracy** | `1.2%` (`6/500`) | `1.2%` (`6/500`) | `0.8%` (`4/500`) |
| **Valid SQL Rate** | `84.2%` (`421/500`) | `56.2%` (`281/500`) | `73.6%` (`368/500`) |
| **Wildcard Rate (`SELECT *`)** | `0.0%` (`0/495`) | `0.0%` (`0/478`) | `0.0%` (`0/494`) |

### Component Matching F1 (`bird_mini_dev`, `n = 500`)

Every single one of the **`500/500` (`100.0%`)** gold SQLite queries in `bird_mini_dev` parses and decomposes cleanly through `matching.py`—whereas `taoyds/spider/process_sql.py` fails on `>90%` and flat-dict lowering fails on `75.2%` (`376/500`).

| Component | `griffith-bigdata/FINER-SQL-3B-v2` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` |
| :--- | ---: | ---: | ---: |
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

!!! info "Bug Fix Visible in Real Numbers"
    In upstream `taoyds/spider/evaluation.py:604-605`, both `IUEN` and `cte` would have reported **`1.000`** because all three models had `precision = 0.0` and `recall = 0.0` despite emitting `CTE` or `IUEN` clauses (`0.2%`–`0.6%` CTE rate vs. `30.0%` in gold). Fixing the zero-match F1 condition reports the true **`0.000`** score.
