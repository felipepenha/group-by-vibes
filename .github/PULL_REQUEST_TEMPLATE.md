## Linked Issue

<!-- Open an issue before submitting a PR (only trivial typo/link fixes may skip this). -->
Closes #

## Summary of Changes

<!-- Describe what changed and why. Keep PRs focused on a single issue or cohesive change. -->

-

## Design Specification

<!-- For non-trivial features or new evaluators, link the spec under `docs/designs/`. Otherwise write `N/A`. -->

- N/A

## Verification

Confirm all applicable local checks pass:

- [ ] `uv run pytest -v --cov=group_by_vibes --cov-branch`
- [ ] `uv run ruff check src/ tests/ examples/ .agents/skills/`
- [ ] `uv run mypy src/group_by_vibes/`
- [ ] `uv run python .agents/skills/sync-examples/scripts/sync_examples.py verify` *(required if evaluators or `examples/` changed)*
- [ ] `uv run python .agents/skills/sync-docs/scripts/sync_docs.py verify` *(required if evaluators, dependencies, or docs changed)*

## Checklist

- [ ] Branch created from up-to-date `main` and PR targets `main`
- [ ] Linked to an open issue (or trivial typo/link fix)
- [ ] `src/group_by_vibes/` remains `sqlglot`-only (no ML/Hub runtime imports)
- [ ] Added or updated offline, deterministic unit tests in `tests/`
- [ ] Updated relevant docs (`README.md`, `docs/mkdocs/`, `examples/README.md`, or `docs/designs/`)
