---
name: release
description: >-
  Prepares, validates, and publishes a new PyPI release of `group-by-vibes`
  via GitHub Releases and OIDC Trusted Publishing. Always invokes the
  `sync-docs` skill first to ensure `README.md`, `mkdocs.yml`, and
  `docs/mkdocs/` are up to date before bumping `pyproject.toml`,
  `src/group_by_vibes/__init__.py`, `CHANGELOG.md`, `uv.lock`, and example
  score logs (`sync-examples`), verifies built sdist/wheel artifacts and CI
  checks, cuts a matching `v<version>` GitHub Release, and verifies the live
  PyPI upload. Use when cutting a release, bumping the package version, or
  publishing to PyPI.
---

# Release (`release`)

Prepare, verify, and publish a PyPI release of `group-by-vibes` using the `sync-docs` skill, [`release.py`](./scripts/release.py), GitHub Releases, and `.github/workflows/publish.yml` (OIDC Trusted Publishing via the `pypi` GitHub environment).

## Dependencies

* **`sync-docs` (Required Step 1):** Must always be invoked as the first step to ensure `README.md` (rendered on PyPI), `mkdocs.yml`, `docs/mkdocs/`, `CONTRIBUTING.md`, and mirrored `docs/mkdocs/assets/` reflect every feature, evaluator, and dependency change in the release—and re-run after bumping to refresh versioned benchmark tables.
* **`sync-examples`:** Re-scores `examples/logs/` and updates `examples/README.md` with the bumped `group_by_vibes` version.
* **Tools:** `uv` and authenticated `gh` CLI (`repo` and `workflow` scopes).

## Steps

1. **Sync documentation first (mandatory — invoke `sync-docs`):**

   Always read and execute the [`sync-docs`](../sync-docs/SKILL.md) skill **before** preflight or bumping versions so every release publishes accurate `README.md` (which PyPI freezes at the release commit), `mkdocs.yml`, `docs/mkdocs/`, and brand assets with zero relative links:

   ```bash
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py audit
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py run
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py verify
   ```

   If `sync-docs` finds missing pages, outdated API/quickstart snippets, or broken links, update the affected docs and commit/push them to `main` before Step 2 so `preflight` starts from a clean working tree (or carry them into the release commit using `--allow-dirty` intentionally).

2. **Preflight (`preflight`):**

   Ensure all release notes for `<X.Y.Z>` are drafted under `## [Unreleased]` in `CHANGELOG.md`, then validate branch, git state, tags, changelog, `gh` auth, and PyPI immutability:

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py preflight --version <X.Y.Z>
   ```

3. **Bump version & promote changelog (`bump`):**

   Updates `[project].version` in `pyproject.toml`, `__version__` in `src/group_by_vibes/__init__.py`, and promotes `## [Unreleased]` in `CHANGELOG.md` to `## [<X.Y.Z>] - YYYY-MM-DD` with compare links:

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py bump --version <X.Y.Z>
   ```

4. **Refresh lockfile, examples, and versioned benchmark docs:**

   ```bash
   uv lock
   uv sync --locked --group dev
   uv run --no-sync python .agents/skills/sync-examples/scripts/sync_examples.py run --update-readme
   uv run --no-sync python .agents/skills/sync-docs/scripts/sync_docs.py run
   ```

5. **Build distributions & run full local CI gate:**

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

6. **Commit, push to `main`, and verify GitHub CI & Docs deploy:**

   Commit the release prep files with `chore(release): prepare v<X.Y.Z> for PyPI publishing`, push to `origin main`, and wait for `.github/workflows/ci.yml` on `HEAD` to pass before tagging or releasing (`.github/workflows/docs.yml` also deploys GitHub Pages automatically when `docs/mkdocs/benchmarks.md`, `pyproject.toml`, and `uv.lock` are pushed):

   ```bash
   gh run watch <ci-run-id> --exit-status
   ```

7. **Publish GitHub Release `v<X.Y.Z>` (triggers PyPI upload automatically):**

   Extract release notes from `CHANGELOG.md` and publish a non-draft GitHub Release targeting `main` (`.github/workflows/publish.yml` runs on `release: types: [published]` and requires a matching `v<X.Y.Z>` tag):

   ```bash
   uv run --no-sync python .agents/skills/release/scripts/release.py release-notes --version <X.Y.Z> --output /tmp/gbv-release-notes.md
   gh release create v<X.Y.Z> --target main --title "v<X.Y.Z>" --notes-file /tmp/gbv-release-notes.md
   rm -f /tmp/gbv-release-notes.md
   gh run watch <publish-run-id> --exit-status
   ```

8. **Verify live on PyPI:**

   Confirm `https://pypi.org/pypi/group-by-vibes/json` lists `<X.Y.Z>` and verify a clean isolated install directly from PyPI:

   ```bash
   uv run --isolated --no-project --no-cache --with "group-by-vibes==<X.Y.Z>" python -c "import group_by_vibes as gbv; assert gbv.__version__ == '<X.Y.Z>'"
   ```

## Common Pitfalls

* **Skipping `sync-docs` before cutting a release:** PyPI freezes the `README.md` project description from the tagged release commit and never updates it from later commits on `main`; always run `sync-docs` in Step 1 so PyPI and MkDocs ship current docs.
* **PyPI versions are immutable:** Never re-tag or try to overwrite an existing version on PyPI; always bump to the next patch/minor version.
* **`v` prefix vs. PEP 440:** Git tags and GitHub Releases use `v<X.Y.Z>`, whereas `pyproject.toml`, `__init__.py`, `CHANGELOG.md` headers, and PyPI use `<X.Y.Z>` without `v`.
* **PEP 639 license metadata:** `pyproject.toml` uses `license = "Apache-2.0"` and `license-files = ["LICENSE"]` — never add legacy `License ::` Trove classifiers.
* **Lean source distributions:** Keep `[tool.hatch.build.targets.sdist]` in `pyproject.toml` so `examples/`, `docs/`, and `.agents/` stay out of the `.tar.gz` uploaded to PyPI.
* **Version checks in CI:** Both `sync_examples.py verify` and `sync_docs.py verify` fail CI if `examples/logs/*_scores.json`, `examples/README.md`, or `docs/mkdocs/benchmarks.md` drift from `group_by_vibes.__version__`.
