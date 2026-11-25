# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/felipepenha/group-by-vibes/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/felipepenha/group-by-vibes/releases/tag/v0.0.1
