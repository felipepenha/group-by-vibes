# Contributing to group-by-vibes

Thank you for your interest in contributing to **group-by-vibes**! This guide covers our issue-first workflow, fork-and-branch Git flow, architecture rules, and local verification steps.

## Before Opening a Pull Request

* **Open an issue first:** Please [open a GitHub issue](https://github.com/felipepenha/group-by-vibes/issues) before starting work on bug fixes, new evaluators, features, or architectural changes so scope and design are aligned up front.
* **Trivial changes:** Only minor fixes such as typos or broken links may skip a prior issue.
* **Link the issue:** Reference the issue in every PR description using `Closes #<issue>` or `Fixes #<issue>`.
* **Design specifications:** For non-trivial features or new evaluators, include or update a design specification under `docs/designs/` (following [docs/designs/group_by_vibes.md](docs/designs/group_by_vibes.md)).

## Git Workflow (Fork & Branch into `main`)

Contributions follow a fork-and-branch workflow targeting `main`. Direct pushes to `main` are not allowed.

1. **Fork and clone:** Fork `https://github.com/felipepenha/group-by-vibes` on GitHub and clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/group-by-vibes.git
   cd group-by-vibes
   ```

2. **Add the upstream remote:**

   ```bash
   git remote add upstream https://github.com/felipepenha/group-by-vibes.git
   ```

3. **Sync your local `main` with `upstream/main`:**

   ```bash
   git fetch upstream
   git checkout main
   git merge --ff-only upstream/main
   ```

4. **Create a focused branch off `main`:** Use a descriptive prefix such as `feat/`, `fix/`, `docs/`, or `test/`:

   ```bash
   git checkout -b feat/<short-description>
   ```

5. **Commit with Conventional Commits:** Format commit messages using a Conventional Commit prefix (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
6. **Stay up to date with `upstream/main`:** Before opening or updating your PR, fetch `upstream` and rebase onto (or merge) `upstream/main`:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

7. **Push and open a Pull Request:** Push the branch to your fork and open a PR targeting `felipepenha/group-by-vibes:main`:

   ```bash
   git push -u origin feat/<short-description>
   ```

8. **Review and merge:** Wait for automated checks and Code Owner (`@felipepenha`) review before merging into `main`.

## Architecture & Contribution Guidelines

* **Core-only runtime dependency:** `src/group_by_vibes/` must depend only on `sqlglot` (see [`pyproject.toml`](pyproject.toml)). Never import ML or Hub packages (`datasets`, `transformers`, `torch`, `evaluate`, `pyarrow`, `huggingface_hub`) inside `src/group_by_vibes/`; keep them isolated in `examples/` under `[project.optional-dependencies] examples`.
* **Python & typing:** Target Python `>=3.10`. Core code in `src/group_by_vibes/` must pass `mypy` with `strict = true`, and all code must pass `ruff` (`line-length = 100`, rules `E`, `F`, `I`, `UP`, `B`, `SIM`).
* **Offline, deterministic tests:** All tests in `tests/` must run offline without live database servers, network calls, or `examples` ML extras.
* **Small, focused PRs:** Keep each pull request scoped to a single issue or cohesive change.

## Local Setup & Verification

Install development dependencies using `uv`:

```bash
uv sync --group dev
```

Run all required checks before opening a pull request:

```bash
uv run pytest -v --cov=group_by_vibes --cov-branch
uv run ruff check src/ tests/ examples/ .agents/skills/sync-examples/scripts/
uv run mypy src/group_by_vibes/
```

If you modify evaluators or anything under `examples/`, also verify that example score logs and [`examples/README.md`](examples/README.md) are in sync:

```bash
uv run python .agents/skills/sync-examples/scripts/sync_examples.py verify
```

To install optional benchmark inference dependencies for running `examples/bird_eval.py generate`:

```bash
uv sync --extra examples
```

## Licensing

By contributing, you agree that your contributions will be licensed under the repository's [Apache-2.0](LICENSE) license.
