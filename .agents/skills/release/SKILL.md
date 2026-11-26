---
name: release
description: >-
  Prepares, validates, and publishes a new PyPI release of `group-by-vibes`
  via GitHub Releases and OIDC Trusted Publishing. Bumps `pyproject.toml`,
  `src/group_by_vibes/__init__.py`, `CHANGELOG.md`, `uv.lock`, example score
  logs (`sync-examples`), and documentation (`sync-docs`), verifies built
  sdist/wheel artifacts and CI checks, cuts a matching `v<version>` GitHub
  Release, and verifies the live PyPI upload. Use when cutting a release,
  bumping the package version, or publishing to PyPI.
---

# Release (`release`)

Prepare, verify, and publish a PyPI release of `group-by-vibes` using [`release.py`](./scripts/release.py), GitHub Releases, and `.github/workflows/publish.yml` (OIDC Trusted Publishing via the `pypi` GitHub environment).

## Dependencies

* **`sync-examples`:** Re-scores `examples/logs/` and updates `examples/README.md` with the bumped `group_by_vibes` version.
* **`sync-docs`:** Syncs `docs/mkdocs/benchmarks.md` and verifies `README.md`, `mkdocs.yml`, `docs/mkdocs/`, and Python code snippets.
* **Tools:** `uv` and authenticated `gh` CLI (`repo` and `workflow` scopes).

## Steps

1. **Preflight (`preflight`):**

   Ensure all release notes for `<X.Y.Z>` are drafted under `## [Unreleased]` in `CHANGELOG.md`, then validate branch, git state, tags, changelog, `gh` auth, and PyPI immutability:

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py preflight --version <X.Y.Z>
   ```

2. **Bump version & promote changelog (`bump`):**

   Updates `[project].version` in `pyproject.toml`, `__version__` in `src/group_by_vibes/__init__.py`, and promotes `## [Unreleased]` in `CHANGELOG.md` to `## [<X.Y.Z>] - YYYY-MM-DD` with compare links:

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py bump --version <X.Y.Z>
   ```

3. **Refresh lockfile, examples, and documentation:**

   ```bash
   uv lock
   uv sync --locked --group dev
   uv run --no-sync python .agents/skills/sync-examples/scripts/sync_examples.py run --update-readme
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py run
   ```

   If evaluators, dependencies, or public APIs changed, complete any manual documentation updates flagged by `sync_docs.py audit`.

4. **Build distributions & run full local CI gate:**

   ```bash
   rm -rf dist/
   uv build
   uvx twine check --strict dist/*
   uv run --no-sync python .agents/skills/release/scripts/release.py verify --version <X.Y.Z> --check-dist
   uv run --isolated --no-project --with dist/*.whl python -c "import group_by_vibes as gbv; assert gbv.__version__ == '<X.Y.Z>'"
   uv run --no-sync ruff check src/ tests/ examples/ .agents/skills/
   uv run --no-sync mypy src/group_by_vibes/
   uv run --no-sync pytest -v --cov=group_by_vibes --cov-branch
   uv run --no-sync python .agents/skills/sync-examples/scripts/sync_examples.py verify
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py verify
   uv run --locked --only-group docs mkdocs build --strict
   uv sync --locked --group dev
   rm -rf dist/ site/
   ```

5. **Commit, push to `main`, and verify GitHub CI:**

   Commit the release prep files with `chore(release): prepare v<X.Y.Z> for PyPI publishing`, push to `origin main`, and wait for `.github/workflows/ci.yml` on `HEAD` to pass before tagging or releasing:

   ```bash
   gh run watch <ci-run-id> --exit-status
   ```

6. **Publish GitHub Release `v<X.Y.Z>` (triggers PyPI upload automatically):**

   Extract release notes from `CHANGELOG.md` and publish a non-draft GitHub Release targeting `main` (`.github/workflows/publish.yml` runs on `release: types: [published]` and requires a matching `v<X.Y.Z>` tag):

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py release-notes --version <X.Y.Z> --output /tmp/gbv-release-notes.md
   gh release create v<X.Y.Z> --target main --title "v<X.Y.Z>" --notes-file /tmp/gbv-release-notes.md
   rm -f /tmp/gbv-release-notes.md
   gh run watch <publish-run-id> --exit-status
   ```

7. **Verify live on PyPI:**

   Confirm `https://pypi.org/pypi/group-by-vibes/json` lists `<X.Y.Z>` and verify a clean isolated install directly from PyPI:

   ```bash
   uv run --isolated --no-project --no-cache --with "group-by-vibes==<X.Y.Z>" python -c "import group_by_vibes as gbv; assert gbv.__version__ == '<X.Y.Z>'"
   ```

## Common Pitfalls

* **PyPI versions are immutable:** Never re-tag or try to overwrite an existing version on PyPI; always bump to the next patch/minor version.
* **`v` prefix vs. PEP 440:** Git tags and GitHub Releases use `v<X.Y.Z>`, whereas `pyproject.toml`, `__init__.py`, `CHANGELOG.md` headers, and PyPI use `<X.Y.Z>` without `v`.
* **PEP 639 license metadata:** `pyproject.toml` uses `license = "Apache-2.0"` and `license-files = ["LICENSE"]` — never add legacy `License ::` Trove classifiers.
* **Lean source distributions:** Keep `[tool.hatch.build.targets.sdist]` in `pyproject.toml` so `examples/`, `docs/`, and `.agents/` stay out of the `.tar.gz` uploaded to PyPI.
* **Version checks in CI:** Both `sync_examples.py verify` and `sync_docs.py verify` fail CI if `examples/logs/*_scores.json`, `examples/README.md`, or `docs/mkdocs/benchmarks.md` drift from `group_by_vibes.__version__`.
