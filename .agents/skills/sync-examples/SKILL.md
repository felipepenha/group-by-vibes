---
name: sync-examples
description: >-
  Audits benchmark logs in `examples/logs/` against exported `group_by_vibes`
  metrics and measurements, wires new evaluators into `examples/bird_eval.py`
  when needed, re-scores model outputs with the latest package code, and updates
  the comparative evaluation report in `examples/README.md`. Use when logs
  change, new models are evaluated, or package metrics or versions change.
---

# Sync Examples (`sync-examples`)

Keep `examples/logs/*_scores.json` and the comparative evaluation report in
`examples/README.md` synchronized with the current `group_by_vibes` package
evaluators, version, and benchmark logs using
[`sync_examples.py`](./scripts/sync_examples.py).

## Steps

1. **Audit evaluators and logs (`audit`)**:

   ```bash
   uv run python .agents/skills/sync-examples/scripts/sync_examples.py audit
   ```

   * Checks every `compute_*` evaluator exported in
     `src/group_by_vibes/__init__.py` against the top-level evaluator keys
     emitted by `examples/bird_eval.py score`.
   * Lists every `examples/logs/*.jsonl` run with row counts, model IDs,
     dialects, probe status (`*_probe.jsonl`), and whether its `_scores.json`
     file is missing or stale.
   * If `audit` exits non-zero because a newly added package evaluator is not
     yet wired into `examples/bird_eval.py`, update `run_score` in
     `examples/bird_eval.py` (top-level summary and `by_difficulty`, keeping zero
     ML imports in `score`), extend `scripts/sync_examples.py` if custom report
     formatting is needed, and re-run `audit` until it exits `0`.

2. **Re-score all logs and update tables (`run --update-readme`)**:

   ```bash
   uv run python .agents/skills/sync-examples/scripts/sync_examples.py run --update-readme
   ```

   * Re-scores `examples/fixtures/bird_sample.jsonl` to
     `examples/logs/bird_sample_scores.json` and every non-probe
     `examples/logs/*.jsonl` to `examples/logs/<stem>_scores.json` using
     `examples/bird_eval.py score`.
   * Regenerates the Markdown tables in `examples/README.md` from
     `## Comparative Evaluation Report` (or `## Sample Evaluation Report`) up to
     `### Key Comparative Takeaways`, grouped by `(dialect, num_rows)` cohort,
     while preserving CLI usage above and prose takeaways below.

3. **Refresh CLI commands and prose takeaways in `examples/README.md`**:

   * If new non-probe model logs were added to `examples/logs/`, ensure
     `### 1. Generate Model Outputs` includes their `generate` command (omitting
     `--sample-size` so full splits remain the default) and
     `### 2. Offline Re-Scoring` includes their `score --output ...json` command.
   * Update `### Key Comparative Takeaways` so observations, paired wins, failure
     deltas, and metric trade-offs reflect the freshly generated tables and any
     newly added evaluators.
   * **Report exclusions & hygiene**: Never include `Syntax Valid`,
     `Schema Grounded`, `syntax_valid_rate`, or `schema_grounded_rate` anywhere
     in `examples/README.md`, and keep zero `/Users/` paths, `artifacts/` paths,
     `llmsql` references, or promotional adjectives.

4. **Verify (`verify` and repo checks)**:

   ```bash
   uv run python .agents/skills/sync-examples/scripts/sync_examples.py verify
   uv run ruff check src/ tests/ examples/ .agents/skills/sync-examples/scripts/
   uv run pytest -v
   ```

   Confirm every non-probe log has fresh scores matching the current
   `group_by_vibes` and `sqlglot` versions, `examples/README.md` passes all
   exclusion and hygiene checks, linting passes, and all tests pass.
