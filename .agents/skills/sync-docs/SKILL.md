---
name: sync-docs
description: >-
  Audits and synchronizes `group-by-vibes` documentation (`README.md`,
  `mkdocs.yml`, `docs/mkdocs/`, and `docs/mkdocs/assets/`) against exported
  evaluators in `src/group_by_vibes/`, `pyproject.toml` metadata, and synced
  benchmark reports in `examples/README.md`, verifies zero relative links in
  `README.md` for PyPI rendering, executes Python doc snippets, and validates
  strict MkDocs builds. Use when adding or modifying evaluators, changing
  dependencies or versions, updating benchmark logs, or refreshing brand
  assets.
---

# Sync Docs (`sync-docs`)

Keep `README.md`, `mkdocs.yml`, `docs/mkdocs/`, and mirrored `docs/mkdocs/assets/` synchronized with the current `group-by-vibes` package exports, `pyproject.toml` metadata, and benchmark results using [`sync_docs.py`](./scripts/sync_docs.py).

## Dependencies

* **`sync-examples`:** When evaluators or `examples/logs/` changed, run `sync-examples` first so `examples/README.md` and `examples/logs/*_scores.json` reflect the latest scores and package version before syncing `docs/mkdocs/benchmarks.md`.
* **`docs.yml`:** Pushes to `main` touching `mkdocs.yml`, `docs/mkdocs/**`, `pyproject.toml`, `uv.lock`, or `.github/workflows/docs.yml` automatically deploy the MkDocs site to GitHub Pages.

## Documentation Surfaces

| Source of Truth | Synchronized Targets | Verified By |
| :--- | :--- | :--- |
| `src/group_by_vibes/__init__.py` (`__all__`) | `README.md`, `docs/mkdocs/index.md`, `docs/mkdocs/{metrics,measurements}/index.md`, dedicated evaluator pages, `mkdocs.yml` `nav:` | `sync_docs.py audit` / `verify` |
| `examples/README.md` (`## Comparative Evaluation Report`) | `docs/mkdocs/benchmarks.md` (version line, Comparative Summary table, Component Matching F1 table) | `sync_docs.py run` / `verify` |
| `docs/assets/` | `docs/mkdocs/assets/` (`banner-github-transparent.png`, `icon-256-transparent.png`, `mkdocs-favicon-64.png`) | `sync_docs.py run` / `verify` |
| `pyproject.toml` (`requires-python`, `dependencies`, `version`) | `README.md`, `docs/mkdocs/index.md`, `docs/mkdocs/benchmarks.md` | `sync_docs.py audit` / `verify` |
| `README.md` links & image sources | Absolute `https://github.com/felipepenha/group-by-vibes/blob/main/...` and `raw.githubusercontent.com` URLs for PyPI rendering | `sync_docs.py audit` / `verify` |

## Steps

1. **Refresh benchmark reports first (if evaluators or logs changed):**

   ```bash
   uv run --no-sync python .agents/skills/sync-examples/scripts/sync_examples.py run --update-readme
   ```

2. **Audit documentation drift (`audit`):**

   ```bash
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py audit
   ```

   Inspects exported evaluator coverage, `mkdocs.yml` `nav:` parity with `docs/mkdocs/**/*.md`, `README.md` PyPI link hygiene, `pyproject.toml` metadata parity, `docs/mkdocs/benchmarks.md` table/version alignment, `docs/mkdocs/assets/` SHA-256 parity, and executes all `group_by_vibes` Python code blocks in `README.md` and `docs/mkdocs/`.

3. **Synchronize deterministic tables and brand assets (`run`):**

   ```bash
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py run
   ```

   Updates `docs/mkdocs/benchmarks.md` from `examples/README.md` while preserving explanatory prose and the closing `!!! info` admonition, and copies any updated brand assets from `docs/assets/` to `docs/mkdocs/assets/`.

4. **Update prose, API pages, and navigation manually for structural changes:**

   * When adding a new evaluator, add `docs/mkdocs/metrics/<slug>.md` or `docs/mkdocs/measurements/<slug>.md`, register it in `mkdocs.yml` `nav:`, link it from the parent `index.md`, add it to `docs/mkdocs/index.md` and `README.md`, and update `docs/designs/` if needed.
   * Keep all `README.md` image `src` and file links as absolute GitHub URLs (`#anchor` links may stay relative).

5. **Verify (`verify`, strict MkDocs build, and dev env restore):**

   ```bash
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py verify
   uv run --locked --only-group docs mkdocs build --strict
   uv sync --locked --group dev
   rm -rf site/
   ```

## Common Pitfalls

* **Relative links in `README.md`:** PyPI does not rewrite relative repo paths. Any relative `](file)` or `src="..."` outside fenced code blocks breaks on `pypi.org`.
* **Orphaned MkDocs pages:** Adding a `.md` file under `docs/mkdocs/` without adding it to `mkdocs.yml` `nav:` causes navigation gaps and fails `sync_docs.py verify`.
* **Environment switching for MkDocs:** `uv run --locked --only-group docs ...` uninstalls `dev` dependencies (`pytest`, `ruff`, `mypy`) from `.venv`; always run `uv sync --locked --group dev` and remove `site/` afterward.
