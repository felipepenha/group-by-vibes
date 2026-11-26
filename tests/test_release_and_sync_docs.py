"""Offline tests for the `.agents/skills/release` and `.agents/skills/sync-docs` helpers."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

import group_by_vibes

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_SCRIPT = REPO_ROOT / ".agents" / "skills" / "release" / "scripts" / "release.py"
SYNC_DOCS_SCRIPT = REPO_ROOT / ".agents" / "skills" / "sync-docs" / "scripts" / "sync_docs.py"


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scaffold_release_repo(root: Path, version: str = "0.1.0") -> None:
    (root / "src" / "group_by_vibes").mkdir(parents=True)
    (root / "examples" / "logs").mkdir(parents=True)
    (root / "docs" / "mkdocs").mkdir(parents=True)

    (root / "pyproject.toml").write_text(
        (
            "[project]\n"
            'name = "group-by-vibes"\n'
            f'version = "{version}"\n'
            'requires-python = ">=3.10"\n'
            'dependencies = ["sqlglot>=30.0.0,<31.0.0"]\n\n'
            "[tool.hatch.build.targets.sdist]\n"
            'include = ["/src", "/LICENSE", "/README.md", "/CHANGELOG.md", "/pyproject.toml"]\n'
        ),
        encoding="utf-8",
    )
    (root / "src" / "group_by_vibes" / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (root / "src" / "group_by_vibes" / "py.typed").write_text("", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        (
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "### Added\n\n"
            "- Feature X for deterministic SQL checks.\n\n"
            f"## [{version}] - 2026-11-25\n\n"
            "### Added\n\n"
            "- Initial PyPI release.\n\n"
            f"[Unreleased]: https://github.com/felipepenha/group-by-vibes/compare/v{version}...HEAD\n"
            f"[{version}]: https://github.com/felipepenha/group-by-vibes/compare/v0.0.1...v{version}\n"
        ),
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        f'version = 1\n\n[[package]]\nname = "group-by-vibes"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "examples" / "logs" / "bird_sample_scores.json").write_text(
        json.dumps({"metadata": {"group_by_vibes_version": version}}),
        encoding="utf-8",
    )
    (root / "docs" / "mkdocs" / "benchmarks.md").write_text(
        f"Evaluated with `extract_fenced_or_last_sql`, `group_by_vibes` `{version}`, "
        "and `sqlglot` `30.33.0`:\n",
        encoding="utf-8",
    )


def _write_dist_artifacts(
    root: Path,
    version: str,
    *,
    include_leaked_examples: bool = False,
) -> None:
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)
    prefix = f"group_by_vibes-{version}"

    sdist_path = dist_dir / f"{prefix}.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as tf:
        for rel in (
            "src/group_by_vibes/py.typed",
            "LICENSE",
            "README.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ):
            data = b"placeholder\n"
            info = tarfile.TarInfo(name=f"{prefix}/{rel}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if include_leaked_examples:
            data = b"{}\n"
            info = tarfile.TarInfo(name=f"{prefix}/examples/foo.jsonl")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    wheel_path = dist_dir / f"{prefix}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("group_by_vibes/py.typed", "")
        zf.writestr(f"{prefix}.dist-info/licenses/LICENSE", "Apache-2.0\n")


def _scaffold_docs_repo(root: Path) -> None:
    shutil.copy2(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "README.md", root / "README.md")
    shutil.copy2(REPO_ROOT / "mkdocs.yml", root / "mkdocs.yml")
    shutil.copytree(REPO_ROOT / "docs", root / "docs")
    (root / "examples").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "examples" / "README.md", root / "examples" / "README.md")


def test_release_parse_version_and_validation() -> None:
    release_mod = _load_script_module("test_release_mod", RELEASE_SCRIPT)

    assert release_mod.parse_version("0.2.0") > release_mod.parse_version("0.1.0")
    assert release_mod.parse_version("1.0.0") > release_mod.parse_version("1.0.0rc1")
    assert release_mod.parse_version("1.0.0rc1") > release_mod.parse_version("1.0.0b2")

    with pytest.raises(ValueError, match="must not start with 'v'"):
        release_mod.parse_version("v0.2.0")
    with pytest.raises(ValueError, match="Invalid version"):
        release_mod.parse_version("0.2")


def test_release_bump_notes_verify_and_dist_checks(tmp_path: Path) -> None:
    release_mod = _load_script_module("test_release_mod", RELEASE_SCRIPT)
    _scaffold_release_repo(tmp_path, version="0.1.0")

    # Equal or older target version fails
    with pytest.raises(ValueError, match="strictly greater"):
        release_mod.bump_version(tmp_path, target_version="0.1.0", release_date="2026-12-01")

    # Bump 0.1.0 -> 0.2.0
    prev = release_mod.bump_version(tmp_path, target_version="0.2.0", release_date="2026-12-01")
    assert prev == "0.1.0"
    assert 'version = "0.2.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "0.2.0"' in (
        tmp_path / "src" / "group_by_vibes" / "__init__.py"
    ).read_text(encoding="utf-8")

    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [0.2.0] - 2026-12-01" in changelog
    assert (
        "[Unreleased]: https://github.com/felipepenha/group-by-vibes/compare/v0.2.0...HEAD"
        in changelog
    )
    assert (
        "[0.2.0]: https://github.com/felipepenha/group-by-vibes/compare/v0.1.0...v0.2.0"
        in changelog
    )

    notes = release_mod.extract_release_notes(tmp_path, target_version="0.2.0")
    assert "- Feature X for deterministic SQL checks." in notes
    assert (
        "**Full Changelog:** https://github.com/felipepenha/group-by-vibes/compare/v0.1.0...v0.2.0"
        in notes
    )

    # Before refreshing lock/scores/benchmarks, verify detects all three stale files
    stale_errors = release_mod.verify_release(tmp_path, target_version="0.2.0", check_dist=False)
    assert any("uv.lock" in err for err in stale_errors)
    assert any("bird_sample_scores.json" in err for err in stale_errors)
    assert any("docs/mkdocs/benchmarks.md" in err for err in stale_errors)

    # Sync remaining versioned files and build clean dists
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "group-by-vibes"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "examples" / "logs" / "bird_sample_scores.json").write_text(
        json.dumps({"metadata": {"group_by_vibes_version": "0.2.0"}}),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "mkdocs" / "benchmarks.md").write_text(
        "Evaluated with `extract_fenced_or_last_sql`, `group_by_vibes` `0.2.0`, "
        "and `sqlglot` `30.33.0`:\n",
        encoding="utf-8",
    )
    _write_dist_artifacts(tmp_path, "0.2.0", include_leaked_examples=False)
    assert release_mod.verify_release(tmp_path, target_version="0.2.0", check_dist=True) == []

    # Leaking examples/ into sdist is rejected
    _write_dist_artifacts(tmp_path, "0.2.0", include_leaked_examples=True)
    leak_errors = release_mod.verify_release(tmp_path, target_version="0.2.0", check_dist=True)
    assert any("examples/" in err for err in leak_errors)


def test_sync_docs_verify_real_repo() -> None:
    sync_docs_mod = _load_script_module("test_sync_docs_mod", SYNC_DOCS_SCRIPT)
    assert sync_docs_mod.main(["verify"]) == 0


def test_sync_docs_run_restores_benchmarks_and_is_idempotent(tmp_path: Path) -> None:
    sync_docs_mod = _load_script_module("test_sync_docs_mod", SYNC_DOCS_SCRIPT)
    _scaffold_docs_repo(tmp_path)

    benchmarks_path = tmp_path / "docs" / "mkdocs" / "benchmarks.md"
    expected_benchmarks = benchmarks_path.read_text(encoding="utf-8")

    # Introduce stale version & metric number while keeping prose + admonition intact
    stale_benchmarks = expected_benchmarks.replace(
        f"`group_by_vibes` `{group_by_vibes.__version__}`", "`group_by_vibes` `0.0.1`"
    ).replace(
        "| **SQL-BLEU** | `0.2646` |",
        "| **SQL-BLEU** | `0.9999` |",
    )
    benchmarks_path.write_text(stale_benchmarks, encoding="utf-8")
    assert sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues

    assert sync_docs_mod.main(["--repo-root", str(tmp_path), "run"]) == 0
    assert benchmarks_path.read_text(encoding="utf-8") == expected_benchmarks
    assert sync_docs_mod.sync_benchmarks_and_assets(tmp_path) == []
    assert sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues == ()


def test_sync_docs_detects_pypi_links_assets_and_nav_drift(tmp_path: Path) -> None:
    sync_docs_mod = _load_script_module("test_sync_docs_mod", SYNC_DOCS_SCRIPT)
    _scaffold_docs_repo(tmp_path)

    # 1. Relative links inside fenced code blocks are ignored, while prose links are flagged
    readme_path = tmp_path / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(
        readme_text
        + "\n```markdown\n[ignored](docs/ignored.md)\n```\n\nSee [broken](docs/broken.md).\n",
        encoding="utf-8",
    )
    issues = sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues
    assert any("docs/broken.md" in item for item in issues)
    assert not any("docs/ignored.md" in item for item in issues)

    # Reset README and test asset parity + auto-repair
    readme_path.write_text(readme_text, encoding="utf-8")
    mirrored_icon = tmp_path / "docs" / "mkdocs" / "assets" / "icon-256-transparent.png"
    mirrored_icon.write_bytes(b"stale-bytes")
    asset_issues = sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues
    assert any("icon-256-transparent.png" in item for item in asset_issues)

    changed = sync_docs_mod.sync_benchmarks_and_assets(tmp_path)
    assert changed == ["docs/mkdocs/assets/icon-256-transparent.png"]
    assert sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues == ()

    # 3. Unlisted MkDocs page is flagged
    (tmp_path / "docs" / "mkdocs" / "unlisted.md").write_text("# Unlisted\n", encoding="utf-8")
    nav_issues = sync_docs_mod.collect_issues(tmp_path, check_snippets=False).issues
    assert any("unlisted.md" in item for item in nav_issues)
