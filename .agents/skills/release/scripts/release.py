"""Preflight, version bump, artifact verification, and release notes for `group-by-vibes`."""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$")
PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}
INIT_VERSION_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', flags=re.MULTILINE)
REPO_SLUG = "felipepenha/group-by-vibes"


def default_repo_root() -> Path:
    """Locate the repository root containing `pyproject.toml` and `src/group_by_vibes`."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "group_by_vibes" / "__init__.py"
        ).is_file():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    raise RuntimeError("Could not locate repository root containing `pyproject.toml`.")


def parse_version(raw: str) -> tuple[int, int, int, int, int]:
    """Parse a strict PEP 440 `X.Y.Z` (with optional `a|b|rcN`) version tuple."""
    if raw.startswith(("v", "V")):
        raise ValueError(
            f"Invalid version {raw!r}: package version must not start with 'v' "
            "(use PEP 440 'X.Y.Z'; the Git tag/release uses 'vX.Y.Z')."
        )
    match = SEMVER_RE.fullmatch(raw)
    if not match:
        raise ValueError(
            f"Invalid version {raw!r}: expected `X.Y.Z` or `X.Y.Z{{a|b|rc}}N`."
        )
    major, minor, patch, pre_tag, pre_num = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        PRE_RANK[pre_tag],
        int(pre_num) if pre_num is not None else 0,
    )


def _strip_toml_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote == '"':
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            return line[:idx]
    return line


def _read_pyproject(repo_root: Path) -> dict[str, Any]:
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        return dict(tomllib.loads(text))

    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    lines = iter(text.splitlines())
    for raw_line in lines:
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section_parts = [p.strip().strip('"').strip("'") for p in line[1:-1].split(".")]
            node = root
            for part in section_parts:
                node = node.setdefault(part, {})
            current = node
            continue
        if "=" not in line:
            continue
        key, val = (part.strip() for part in line.split("=", 1))
        while val.startswith("[") and not val.endswith("]"):
            try:
                next_line = _strip_toml_comment(next(lines)).strip()
            except StopIteration:
                break
            if next_line:
                val = f"{val} {next_line}"
        try:
            current[key] = ast.literal_eval(val)
        except (SyntaxError, ValueError):
            current[key] = val.strip('"').strip("'")
    return root


def current_package_version(repo_root: Path) -> str:
    data = _read_pyproject(repo_root)
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise ValueError("Missing `[project].version` in `pyproject.toml`.")
    return str(project["version"])


def _run_cmd(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _extract_unreleased_body(changelog_text: str) -> tuple[int, int, str]:
    """Return `(body_start, body_end, body_text)` for `## [Unreleased]` in `CHANGELOG.md`."""
    header = re.search(r"^## \[Unreleased\][ \t]*\n?", changelog_text, flags=re.MULTILINE)
    if not header:
        raise ValueError("Missing `## [Unreleased]` heading in `CHANGELOG.md`.")
    body_start = header.end()
    next_section = re.search(
        r"^(?:## \[|\[[^\]]+\]:\s*https?://)",
        changelog_text[body_start:],
        flags=re.MULTILINE,
    )
    body_end = body_start + next_section.start() if next_section else len(changelog_text)
    return body_start, body_end, changelog_text[body_start:body_end]


def _check_pypi_absent(version: str) -> str | None:
    url = "https://pypi.org/pypi/group-by-vibes/json"
    req = urllib.request.Request(url, headers={"User-Agent": "group-by-vibes-release-check"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        return f"PyPI lookup returned HTTP {exc.code}: {exc.reason}"
    except OSError as exc:
        return f"Failed to query PyPI ({url}): {exc}"

    releases = payload.get("releases", {})
    if version in releases:
        return (
            f"PyPI already has release `{version}` for `group-by-vibes` "
            "(PyPI versions are immutable)."
        )
    return None


def run_preflight(
    repo_root: Path,
    *,
    target_version: str,
    allow_dirty: bool = False,
    skip_network: bool = False,
) -> list[str]:
    """Validate repository, changelog, git, and PyPI readiness before bumping."""
    errors: list[str] = []
    try:
        target_tuple = parse_version(target_version)
    except ValueError as exc:
        return [str(exc)]

    current_ver = current_package_version(repo_root)
    try:
        current_tuple = parse_version(current_ver)
    except ValueError as exc:
        errors.append(f"Current pyproject.toml version invalid: {exc}")
    else:
        if target_tuple <= current_tuple:
            errors.append(
                f"Target version `{target_version}` must be strictly greater than "
                f"current `{current_ver}`."
            )

    branch_proc = _run_cmd(["git", "branch", "--show-current"], cwd=repo_root)
    if branch_proc.returncode != 0:
        errors.append(f"Failed to check git branch: {branch_proc.stderr.strip()}")
    elif branch_proc.stdout.strip() != "main":
        branch_name = branch_proc.stdout.strip()
        errors.append(
            f"Current git branch is {branch_name!r}; releases must be cut from 'main'."
        )

    if not allow_dirty:
        status_proc = _run_cmd(["git", "status", "--porcelain"], cwd=repo_root)
        if status_proc.returncode != 0:
            errors.append(f"Failed to check git status: {status_proc.stderr.strip()}")
        elif status_proc.stdout.strip():
            errors.append(
                "Working tree is dirty; commit or stash changes (or pass `--allow-dirty`)."
            )

    tag_name = f"v{target_version}"
    local_tag = _run_cmd(["git", "tag", "-l", tag_name], cwd=repo_root)
    if local_tag.returncode == 0 and local_tag.stdout.strip() == tag_name:
        errors.append(f"Local git tag `{tag_name}` already exists.")

    changelog_path = repo_root / "CHANGELOG.md"
    if not changelog_path.is_file():
        errors.append("Missing `CHANGELOG.md`.")
    else:
        changelog_text = changelog_path.read_text(encoding="utf-8")
        try:
            _, _, body = _extract_unreleased_body(changelog_text)
            if not body.strip():
                errors.append(
                    "`CHANGELOG.md` `## [Unreleased]` section is empty; "
                    "document release changes first."
                )
        except ValueError as exc:
            errors.append(str(exc))

    if not skip_network:
        head_proc = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
        remote_main = _run_cmd(["git", "ls-remote", "origin", "refs/heads/main"], cwd=repo_root)
        if remote_main.returncode != 0 or not remote_main.stdout.strip():
            errors.append(
                f"Could not verify `origin/main` via git ls-remote: {remote_main.stderr.strip()}"
            )
        else:
            remote_sha = remote_main.stdout.split()[0]
            local_sha = head_proc.stdout.strip()
            if local_sha != remote_sha:
                errors.append(
                    f"Local HEAD ({local_sha[:7]}) does not match origin/main ({remote_sha[:7]})."
                )

        remote_tag = _run_cmd(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag_name}"],
            cwd=repo_root,
        )
        if remote_tag.returncode == 0 and remote_tag.stdout.strip():
            errors.append(f"Remote git tag `{tag_name}` already exists on `origin`.")

        gh_auth = _run_cmd(["gh", "auth", "status"], cwd=repo_root)
        if gh_auth.returncode != 0:
            errors.append("GitHub CLI is not authenticated (`gh auth status` failed).")

        pypi_err = _check_pypi_absent(target_version)
        if pypi_err:
            errors.append(pypi_err)

    return errors


def _update_pyproject_version(pyproject_text: str, new_version: str) -> str:
    """Replace `version = "..."` inside `[project]` in `pyproject.toml`."""
    lines = pyproject_text.splitlines(keepends=True)
    in_project = False
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
        elif in_project and not replaced and re.match(r'^version\s*=\s*"[^"]+"', stripped):
            line = re.sub(
                r'^(version\s*=\s*")[^"]+(")',
                rf"\g<1>{new_version}\g<2>",
                line,
                count=1,
            )
            replaced = True
        out.append(line)
    if not replaced:
        raise ValueError("Could not update `version = ...` under `[project]` in `pyproject.toml`.")
    return "".join(out)


def _promote_changelog(
    changelog_text: str,
    *,
    prev_version: str,
    new_version: str,
    release_date: str,
) -> str:
    """Promote `## [Unreleased]` to `## [new_version] - release_date` and update links."""
    if re.search(rf"^## \[{re.escape(new_version)}\]", changelog_text, flags=re.MULTILINE):
        raise ValueError(f"`CHANGELOG.md` already contains a `## [{new_version}]` section.")

    body_start, body_end, body = _extract_unreleased_body(changelog_text)
    cleaned_body = body.strip()
    if not cleaned_body:
        raise ValueError(
            "`CHANGELOG.md` `## [Unreleased]` section is empty; add entries before bumping."
        )

    before = changelog_text[:body_start].rstrip("\n") + "\n\n"
    promoted = f"## [{new_version}] - {release_date}\n\n{cleaned_body}\n\n"
    after = changelog_text[body_end:].lstrip("\n")
    updated = before + promoted + after

    unreleased_link_re = re.compile(
        r"^(\[Unreleased\]:\s*https://github\.com/[^/]+/[^/]+/compare/)([^\.\s]+)(\.\.\.HEAD\s*)$",
        flags=re.MULTILINE,
    )
    link_match = unreleased_link_re.search(updated)
    if link_match:
        base_compare, old_tag, suffix = link_match.groups()
        new_links = (
            f"{base_compare}v{new_version}{suffix.rstrip()}\n"
            f"[{new_version}]: {base_compare}{old_tag}...v{new_version}"
        )
        updated = unreleased_link_re.sub(new_links, updated, count=1)
    else:
        compare_url = (
            f"https://github.com/{REPO_SLUG}/compare/v{prev_version}...v{new_version}"
        )
        updated = (
            updated.rstrip()
            + f"\n\n[Unreleased]: https://github.com/{REPO_SLUG}/compare/v{new_version}...HEAD\n"
            + f"[{new_version}]: {compare_url}\n"
        )
    return updated


def bump_version(
    repo_root: Path,
    *,
    target_version: str,
    release_date: str | None = None,
) -> str:
    """Bump package version across `pyproject.toml`, `__init__.py`, and `CHANGELOG.md`."""
    target_tuple = parse_version(target_version)
    prev_version = current_package_version(repo_root)
    if target_tuple <= parse_version(prev_version):
        raise ValueError(
            f"Target version `{target_version}` must be strictly greater than "
            f"current `{prev_version}`."
        )

    date_str = release_date or datetime.date.today().isoformat()
    datetime.date.fromisoformat(date_str)

    pyproject_path = repo_root / "pyproject.toml"
    init_path = repo_root / "src" / "group_by_vibes" / "__init__.py"
    changelog_path = repo_root / "CHANGELOG.md"

    new_pyproject = _update_pyproject_version(
        pyproject_path.read_text(encoding="utf-8"),
        target_version,
    )
    init_text = init_path.read_text(encoding="utf-8")
    new_init, init_subs = INIT_VERSION_RE.subn(rf"\g<1>{target_version}\g<3>", init_text, count=1)
    if init_subs != 1:
        raise ValueError("Could not update `__version__` in `src/group_by_vibes/__init__.py`.")

    new_changelog = _promote_changelog(
        changelog_path.read_text(encoding="utf-8"),
        prev_version=prev_version,
        new_version=target_version,
        release_date=date_str,
    )

    pyproject_path.write_text(new_pyproject, encoding="utf-8")
    init_path.write_text(new_init, encoding="utf-8")
    changelog_path.write_text(new_changelog, encoding="utf-8")
    return prev_version


def _uv_lock_package_version(uv_lock_text: str) -> str | None:
    """Extract the locked version for `name = \"group-by-vibes\"` from `uv.lock`."""
    match = re.search(
        r'^\[\[package\]\]\s*\nname\s*=\s*"group-by-vibes"\s*\nversion\s*=\s*"([^"]+)"',
        uv_lock_text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def verify_release(
    repo_root: Path,
    *,
    target_version: str,
    check_dist: bool = False,
) -> list[str]:
    """Verify synchronized version strings, packaging metadata, and optional `dist/` artifacts."""
    errors: list[str] = []
    try:
        parse_version(target_version)
    except ValueError as exc:
        return [str(exc)]

    # 1. pyproject.toml
    pyproject_data = _read_pyproject(repo_root)
    project = pyproject_data.get("project", {})
    pyproject_ver = project.get("version") if isinstance(project, dict) else None
    if pyproject_ver != target_version:
        errors.append(
            f"pyproject.toml: version is {pyproject_ver!r}, expected {target_version!r}."
        )
    tool_hatch = (
        pyproject_data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
        if isinstance(pyproject_data.get("tool"), dict)
        else {}
    )
    if not isinstance(tool_hatch, dict) or "sdist" not in tool_hatch:
        errors.append("pyproject.toml: missing `[tool.hatch.build.targets.sdist]` configuration.")

    # 2. src/group_by_vibes/__init__.py + py.typed
    init_path = repo_root / "src" / "group_by_vibes" / "__init__.py"
    init_match = INIT_VERSION_RE.search(init_path.read_text(encoding="utf-8"))
    init_ver = init_match.group(2) if init_match else None
    if init_ver != target_version:
        errors.append(
            f"src/group_by_vibes/__init__.py: `__version__` is {init_ver!r}, "
            f"expected {target_version!r}."
        )
    if not (repo_root / "src" / "group_by_vibes" / "py.typed").is_file():
        errors.append("Missing PEP 561 marker `src/group_by_vibes/py.typed`.")

    # 3. uv.lock
    uv_lock_path = repo_root / "uv.lock"
    if not uv_lock_path.is_file():
        errors.append("Missing `uv.lock`.")
    else:
        lock_ver = _uv_lock_package_version(uv_lock_path.read_text(encoding="utf-8"))
        if lock_ver != target_version:
            errors.append(
                f"uv.lock: `group-by-vibes` version is {lock_ver!r}, "
                f"expected {target_version!r} (run `uv lock`)."
            )

    # 4. CHANGELOG.md
    changelog_path = repo_root / "CHANGELOG.md"
    changelog_text = changelog_path.read_text(encoding="utf-8")
    heading_re = rf"^## \[{re.escape(target_version)}\] - \d{{4}}-\d{{2}}-\d{{2}}"
    if not re.search(heading_re, changelog_text, flags=re.MULTILINE):
        errors.append(
            f"CHANGELOG.md: missing dated release heading `## [{target_version}] - YYYY-MM-DD`."
        )
    link_re = rf"^\[{re.escape(target_version)}\]:\s*https?://"
    if not re.search(link_re, changelog_text, flags=re.MULTILINE):
        errors.append(
            f"CHANGELOG.md: missing reference link `[{target_version}]: https://...`."
        )

    # 5. Non-probe examples/logs/*_scores.json
    logs_dir = repo_root / "examples" / "logs"
    score_files = (
        sorted(p for p in logs_dir.glob("*_scores.json") if not p.stem.endswith("_probe_scores"))
        if logs_dir.is_dir()
        else []
    )
    if not score_files:
        errors.append("Missing non-probe `examples/logs/*_scores.json` files.")
    for score_path in score_files:
        data = json.loads(score_path.read_text(encoding="utf-8"))
        meta_ver = data.get("metadata", {}).get("group_by_vibes_version")
        if meta_ver != target_version:
            rel_score = score_path.relative_to(repo_root)
            errors.append(
                f"{rel_score}: `group_by_vibes_version` is {meta_ver!r}, "
                f"expected {target_version!r}."
            )

    # 6. docs/mkdocs/benchmarks.md
    benchmarks_path = repo_root / "docs" / "mkdocs" / "benchmarks.md"
    benchmarks_text = benchmarks_path.read_text(encoding="utf-8")
    expected_token = f"`group_by_vibes` `{target_version}`"
    if expected_token not in benchmarks_text:
        errors.append(f"docs/mkdocs/benchmarks.md: missing {expected_token!r}.")

    # 7. Optional dist/ validation
    if check_dist:
        sdist_path = repo_root / "dist" / f"group_by_vibes-{target_version}.tar.gz"
        wheel_path = repo_root / "dist" / f"group_by_vibes-{target_version}-py3-none-any.whl"
        if not sdist_path.is_file():
            errors.append(f"Missing sdist artifact `{sdist_path.relative_to(repo_root)}`.")
        else:
            with tarfile.open(sdist_path, "r:gz") as tf:
                members = tf.getnames()
            for req_suffix in (
                "/src/group_by_vibes/py.typed",
                "/LICENSE",
                "/README.md",
                "/CHANGELOG.md",
                "/pyproject.toml",
            ):
                if not any(m.endswith(req_suffix) for m in members):
                    errors.append(f"{sdist_path.name}: missing expected member `*{req_suffix}`.")
            for forbidden in ("examples", "docs", ".agents"):
                leaked = [
                    m for m in members if f"/{forbidden}/" in m or m.endswith(f"/{forbidden}")
                ]
                if leaked:
                    errors.append(
                        f"{sdist_path.name}: unexpectedly bundles `{forbidden}/` ({leaked[0]})."
                    )

        if not wheel_path.is_file():
            errors.append(f"Missing wheel artifact `{wheel_path.relative_to(repo_root)}`.")
        else:
            with zipfile.ZipFile(wheel_path, "r") as zf:
                wheel_names = set(zf.namelist())
            for req_member in (
                "group_by_vibes/py.typed",
                f"group_by_vibes-{target_version}.dist-info/licenses/LICENSE",
            ):
                if req_member not in wheel_names:
                    errors.append(f"{wheel_path.name}: missing required file `{req_member}`.")

    return errors


def extract_release_notes(repo_root: Path, *, target_version: str) -> str:
    """Extract release notes for `target_version` from `CHANGELOG.md`."""
    parse_version(target_version)
    changelog_text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")

    heading_match = re.search(
        rf"^## \[{re.escape(target_version)}\][^\n]*\n",
        changelog_text,
        flags=re.MULTILINE,
    )
    if not heading_match:
        raise ValueError(f"Could not find `## [{target_version}]` section in `CHANGELOG.md`.")

    start = heading_match.end()
    end_match = re.search(
        r"^(?:## \[|\[[^\]]+\]:\s*https?://)",
        changelog_text[start:],
        flags=re.MULTILINE,
    )
    end = start + end_match.start() if end_match else len(changelog_text)
    body = changelog_text[start:end].strip()
    if not body:
        raise ValueError(f"`## [{target_version}]` section in `CHANGELOG.md` is empty.")

    link_match = re.search(
        rf"^\[{re.escape(target_version)}\]:\s*(\S+)\s*$",
        changelog_text,
        flags=re.MULTILINE,
    )
    if link_match:
        return f"{body}\n\n**Full Changelog:** {link_match.group(1)}\n"
    return f"{body}\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight, bump, verify, and extract release notes for group-by-vibes."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root override (defaults to workspace root).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre_p = sub.add_parser(
        "preflight",
        help="Check branch, git clean state, changelog, and PyPI.",
    )
    pre_p.add_argument(
        "--version",
        required=True,
        help="Target PEP 440 version (without leading 'v').",
    )
    pre_p.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted changes.")
    pre_p.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip remote git/gh/PyPI checks.",
    )

    bump_p = sub.add_parser(
        "bump",
        help="Bump version in pyproject.toml, __init__.py, and CHANGELOG.md.",
    )
    bump_p.add_argument(
        "--version",
        required=True,
        help="Target PEP 440 version (without leading 'v').",
    )
    bump_p.add_argument("--date", default=None, help="Release date YYYY-MM-DD (defaults to today).")

    ver_p = sub.add_parser(
        "verify",
        help="Verify version parity across repo files and optional dist/.",
    )
    ver_p.add_argument("--version", required=True, help="Expected PEP 440 version.")
    ver_p.add_argument(
        "--check-dist",
        action="store_true",
        help="Validate built sdist and wheel under dist/.",
    )

    notes_p = sub.add_parser(
        "release-notes",
        help="Extract GitHub Release notes from CHANGELOG.md.",
    )
    notes_p.add_argument("--version", required=True, help="Released PEP 440 version.")
    notes_p.add_argument("--output", type=Path, default=None, help="Optional output file path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve() if args.repo_root else default_repo_root()

    if args.command == "preflight":
        errors = run_preflight(
            root,
            target_version=args.version,
            allow_dirty=args.allow_dirty,
            skip_network=args.skip_network,
        )
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        cur = current_package_version(root)
        print(f"PREFLIGHT=OK current={cur} target={args.version} tag=v{args.version}")
        return 0

    if args.command == "bump":
        try:
            prev = bump_version(root, target_version=args.version, release_date=args.date)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        date_str = args.date or datetime.date.today().isoformat()
        print(f"BUMP=OK prev={prev} version={args.version} date={date_str}")
        return 0

    if args.command == "verify":
        errors = verify_release(root, target_version=args.version, check_dist=args.check_dist)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print(f"VERIFY=OK version={args.version} dist_checked={args.check_dist}")
        return 0

    if args.command == "release-notes":
        try:
            notes = extract_release_notes(root, target_version=args.version)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.output:
            args.output.write_text(notes, encoding="utf-8")
        else:
            print(notes, end="")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
