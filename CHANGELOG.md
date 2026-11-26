# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-25

### Added

- **Release & Documentation Sync Skills (`.agents/skills/`):** Added `release` (`.agents/skills/release/scripts/release.py`) and `sync-docs` (`.agents/skills/sync-docs/scripts/sync_docs.py`) workspace skills with offline unit tests (`tests/test_release_and_sync_docs.py`), widened CI `ruff` linting to `.agents/skills/`, and wired `sync_docs.py verify` into `.github/workflows/ci.yml`.

### Changed

- **PyPI Installation Docs & Site Links (`README.md`, `docs/mkdocs/index.md`, `mkdocs.yml`):** Updated installation instructions from `git+https://...` to `uv add group-by-vibes` / `pip install group-by-vibes`, added the PyPI badge and homepage link in `README.md` and `docs/mkdocs/index.md`, added GitHub/PyPI footer links in `mkdocs.yml`, and aligned `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md` verification commands with CI.

### Fixed

- **README Example Typo (`README.md`):** Fixed the table alias typo (`o.user_id` -> `ord.user_id`) in the Component/Exact Matching example so it outputs `1.0` as documented.

## [0.1.0] - 2026-11-25

### Added

- **`ComponentMatching` (`load("component_matching")`, `compute_component_matching`) and `ExactMatching` (`load("exact_matching")`, `compute_exact_matching`):** Dialect-general, bug-fixed `sqlglot` AST adaptations of Yu et al. (2018) Component Matching and Exact Matching sharing `src/group_by_vibes/matching.py`, supporting 12 SQL component keys (`select`, `select(no AGG)`, `where`, `where(no OP)`, `group(no Having)`, `group`, `order`, `and/or`, `IUEN`, `keywords`, `cte`, `join_on`), corrected corpus F1 / `and/or` denominators / hardness aggregate counting, Spider value-blind defaults with opt-in strict flags (`ignore_values`, `ignore_distinct`, `include_limit_value`, `include_join_conditions`, `skip_unparseable_predictions`), optional per-query `schema` and union-find `foreign_keys` (`kmap`) equivalence collapse, and `ForeignKeyPair` / `ForeignKeysLike` exports.
- **Reference Benchmark & Report Sync (`examples/`):** Added DDL foreign-key extraction (`_ddl_to_foreign_keys`) and Component/Exact Matching scoring in `examples/bird_eval.py`, plus `**Exact Match Accuracy**`, `### Component Matching F1`, and per-difficulty Exact Match reporting in `.agents/skills/sync-examples/scripts/sync_examples.py`.
- **Documentation Site (`docs/mkdocs/`, `mkdocs.yml`):** Material for MkDocs documentation site and automated GitHub Pages deployment (`.github/workflows/docs.yml`).
- **PyPI Packaging & Release Automation:** Added complete PyPI metadata (PEP 639 `license-files`, author, classifiers, project URLs), lean sdist rules excluding `examples/`/`docs/`/`.agents/`, PEP 561 `py.typed` marker, absolute README asset/file links, and OIDC Trusted Publishing workflow (`.github/workflows/publish.yml`).

## [0.0.1] - 2026-11-24

Initial public pre-release of **group-by-vibes** — a standalone, `sqlglot`-only Python package for deterministic Text2SQL (NL2SQL) evaluation mirroring the Hugging Face `evaluate` interface.

### Added

- **Core Evaluator API (`src/group_by_vibes/`):** Synchronous `Metric` / `Measurement` lifecycle (`load`, `add`, `add_batch`, `compute`, `reset`) layered over stateless `compute_*` functions, with `sqlglot>=30.0.0,<31.0.0` as the sole runtime dependency.
- **`SqlBleu` (`load("sql_bleu")`, `compute_sql_bleu`, `sentence_sql_bleu`):** Reference-based AST-canonicalized corpus and sentence SQL-BLEU metric (Chen et al., 2019) with commutative expression sorting and parse-rate reporting.
- **`ValidSqlRate` (`load("valid_sql_rate")`, `compute_valid_sql_rate`):** Reference-free lenient dialect-aware SQL syntax validator and optional per-query schema grounding check with 8-key `failure_counts`.
- **`WildcardDetector` (`load("wildcard_detector")`, `compute_wildcard_detector`):** Reference-free AST projection wildcard detector (`SELECT *`, `table.*`, `* EXCEPT`/`EXCLUDE`), ignoring `COUNT(*)`, `EXISTS (SELECT * ...)`, `RETURNING *`, and `*` arithmetic.
- **Shared Utilities:** Optional SQL extraction (`extract_fenced_or_last_sql`, regex/callable `extractor` support), dialect normalization (`normalize_dialect`), and qualified table parsing (`split_qualified_table`).
- **Reference Benchmark (`examples/`):** Isolated `birdsql/bird_mini_dev` CLI (`examples/bird_eval.py`) supporting `qwen-3b`, `xiyansql-3b`, and `finer-sql-3b` presets, bundled 5-row fixture, 500-row SQLite Qwen/XiYanSQL logs and score reports, comparative report in `examples/README.md`, and `.agents/skills/sync-examples/`.
- **CI, Governance & Brand Assets:** GitHub Actions CI (`.github/workflows/ci.yml` running `ruff`, `mypy`, `pytest --cov`, core smoke checks, and package builds across Python 3.10–3.12), `CONTRIBUTING.md`, `.github/CODEOWNERS`, issue/PR templates, and reproducible multi-platform brand assets under `docs/assets/`.

[Unreleased]: https://github.com/felipepenha/group-by-vibes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/felipepenha/group-by-vibes/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/felipepenha/group-by-vibes/releases/tag/v0.0.1
