"""Audit and synchronize `group-by-vibes` documentation with package exports and logs."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import group_by_vibes

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None

CURATED_HEADLINE_METRICS = (
    "**SQL-BLEU**",
    "**Parsed Queries (SQL-BLEU)**",
    "**Exact Match Accuracy**",
    "**Valid SQL Rate**",
    "**Wildcard Rate (`SELECT *`)**",
)

FENCED_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)
HTML_SRC_RE = re.compile(r'src="([^"]+)"')
MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")
EXTRACTOR_VERSIONS_RE = re.compile(
    r"^\*\s+\*\*Extractor & Versions:\*\*\s+`([^`]+)`\s*,\s*"
    r"`group_by_vibes`\s+`([^`]+)`\s*,\s*`sqlglot`\s+`([^`]+)`",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class EvaluatorDocTarget:
    """Metadata for a public evaluator discovered from `group_by_vibes`."""

    eval_id: str
    class_name: str
    kind: str  # "metric" | "measurement"
    slug: str

    @property
    def page_rel_path(self) -> str:
        return f"{self.kind}s/{self.slug}.md"


@dataclass(frozen=True)
class ExampleReportTables:
    """Parsed summary metadata and tables from `examples/README.md`."""

    extractor: str
    gbv_version: str
    sqlglot_version: str
    headline_table: str
    component_table: str

    @property
    def version_line(self) -> str:
        return (
            f"Evaluated with `{self.extractor}`, "
            f"`group_by_vibes` `{self.gbv_version}`, "
            f"and `sqlglot` `{self.sqlglot_version}`:"
        )


@dataclass(frozen=True)
class AuditSummary:
    """Result of auditing documentation surfaces against the package and logs."""

    evaluators: tuple[EvaluatorDocTarget, ...]
    nav_pages: tuple[str, ...]
    asset_count: int
    snippet_count: int
    issues: tuple[str, ...]


def default_repo_root() -> Path:
    """Locate the repository root containing `pyproject.toml` and `mkdocs.yml`."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "mkdocs.yml").is_file():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "mkdocs.yml").is_file():
        return cwd
    raise RuntimeError(
        "Could not locate repository root containing `pyproject.toml` and `mkdocs.yml`."
    )


def discover_evaluators() -> list[EvaluatorDocTarget]:
    """Discover all exported evaluators dynamically from `group_by_vibes.__all__`."""
    exported = getattr(group_by_vibes, "__all__", [])
    targets: list[EvaluatorDocTarget] = []
    for name in exported:
        if not name.startswith("compute_"):
            continue
        eval_id = name.removeprefix("compute_")
        inst = group_by_vibes.load(eval_id)
        params = set(inspect.signature(inst.add).parameters)
        kind = "metric" if "reference" in params else "measurement"
        targets.append(
            EvaluatorDocTarget(
                eval_id=eval_id,
                class_name=type(inst).__name__,
                kind=kind,
                slug=eval_id.replace("_", "-"),
            )
        )
    return sorted(targets, key=lambda item: (item.kind, item.eval_id))


def parse_mkdocs_nav(mkdocs_text: str) -> list[str]:
    """Parse `.md` document paths from the `nav:` section of `mkdocs.yml`."""
    pages: list[str] = []
    in_nav = False
    for raw_line in mkdocs_text.splitlines():
        no_comment = raw_line.split("#", 1)[0].rstrip()
        if not no_comment.strip():
            continue
        if not in_nav:
            if re.match(r"^nav:\s*$", no_comment):
                in_nav = True
            continue
        if not no_comment[0].isspace():
            break
        match = re.search(r"(?:^-\s+|:\s+)([^\s'\"]+\.md)\s*$", no_comment.strip())
        if match:
            pages.append(match.group(1))
    return pages


def _extract_heading_table(section_text: str, heading_prefix: str) -> list[str]:
    """Extract the first Markdown table block following `heading_prefix`."""
    lines = section_text.splitlines()
    found_heading = False
    table_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not found_heading:
            if stripped.startswith(heading_prefix):
                found_heading = True
            continue
        if not table_lines:
            if not stripped:
                continue
            if stripped.startswith("|"):
                table_lines.append(line)
                continue
            if stripped.startswith("#"):
                break
        else:
            if stripped.startswith("|"):
                table_lines.append(line)
            else:
                break
    if len(table_lines) < 3:
        raise ValueError(f"Could not find Markdown table after heading {heading_prefix!r}.")
    return table_lines


def _with_left_aligned_first_col(align_row: str) -> str:
    """Normalize column 0 alignment in a Markdown separator row to `:---`."""
    cells = [cell.strip() for cell in align_row.strip().strip("|").split("|")]
    if cells:
        cells[0] = ":---"
    return "| " + " | ".join(cells) + " |"


def parse_example_report_tables(examples_readme_text: str) -> ExampleReportTables:
    """Parse primary comparative report metadata and tables from `examples/README.md`."""
    report_match = re.search(
        r"^## Comparative Evaluation Report\b",
        examples_readme_text,
        flags=re.MULTILINE,
    )
    if not report_match:
        raise ValueError(
            "Missing `## Comparative Evaluation Report` section in `examples/README.md`."
        )
    section = examples_readme_text[report_match.start() :]
    next_h2 = re.search(r"^## ", section[3:], flags=re.MULTILINE)
    if next_h2:
        section = section[: 3 + next_h2.start()]

    ver_match = EXTRACTOR_VERSIONS_RE.search(section)
    if not ver_match:
        raise ValueError(
            "Missing `* **Extractor & Versions:**` line under `## Comparative Evaluation Report`."
        )
    extractor, gbv_version, sqlglot_version = ver_match.groups()

    raw_headline = _extract_heading_table(section, "### Headline Comparison")
    headline_header = raw_headline[0]
    headline_align = _with_left_aligned_first_col(raw_headline[1])
    row_by_metric: dict[str, str] = {}
    for row in raw_headline[2:]:
        first_cell = row.strip().strip("|").split("|", 1)[0].strip()
        row_by_metric[first_cell] = row
    missing_metrics = [m for m in CURATED_HEADLINE_METRICS if m not in row_by_metric]
    if missing_metrics:
        raise ValueError(
            f"Missing curated headline rows in `examples/README.md`: {', '.join(missing_metrics)}."
        )
    headline_table = "\n".join(
        [headline_header, headline_align, *(row_by_metric[m] for m in CURATED_HEADLINE_METRICS)]
    )

    raw_comp = _extract_heading_table(section, "### Component Matching F1")
    component_table = "\n".join(
        [raw_comp[0], _with_left_aligned_first_col(raw_comp[1]), *raw_comp[2:]]
    )

    return ExampleReportTables(
        extractor=extractor,
        gbv_version=gbv_version,
        sqlglot_version=sqlglot_version,
        headline_table=headline_table,
        component_table=component_table,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _load_pyproject(path: Path) -> dict[str, Any]:
    """Load `pyproject.toml` using `tomllib` (3.11+) or a minimal stdlib parser (3.10)."""
    text = path.read_text(encoding="utf-8")
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


def _check_readme_pypi_links(readme_text: str) -> list[str]:
    """Ensure `README.md` has no relative links or image sources outside code blocks."""
    issues: list[str] = []
    stripped_text = FENCED_BLOCK_RE.sub(
        lambda m: "\n" * m.group(0).count("\n"),
        readme_text,
    )
    for line_no, line in enumerate(stripped_text.splitlines(), 1):
        for match in HTML_SRC_RE.finditer(line):
            target = match.group(1).strip()
            if not target.startswith(("http://", "https://", "#")):
                issues.append(f"README.md:{line_no}: relative `src` URL {target!r} breaks on PyPI.")
        for match in MD_LINK_RE.finditer(line):
            target = match.group(1).strip().split()[0]
            if not target.startswith(("http://", "https://", "#")):
                issues.append(
                    f"README.md:{line_no}: relative Markdown link {target!r} breaks on PyPI."
                )
    return issues


def _run_doc_snippets(repo_root: Path) -> tuple[int, list[str]]:
    """Execute Python fenced blocks referencing `group_by_vibes` across docs."""
    doc_files = [repo_root / "README.md", *sorted((repo_root / "docs" / "mkdocs").rglob("*.md"))]
    count = 0
    issues: list[str] = []
    for doc_path in doc_files:
        if not doc_path.is_file():
            continue
        rel_path = doc_path.relative_to(repo_root).as_posix()
        text = doc_path.read_text(encoding="utf-8")
        for idx, block in enumerate(PYTHON_BLOCK_RE.findall(text), 1):
            if "import group_by_vibes" not in block and "from group_by_vibes" not in block:
                continue
            count += 1
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(block, f"{rel_path}#block{idx}", "exec"), {})
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    f"{rel_path}#block{idx}: snippet failed ({type(exc).__name__}: {exc})"
                )
    return count, issues


def collect_issues(repo_root: Path, *, check_snippets: bool = True) -> AuditSummary:
    """Run all documentation drift and parity checks against `repo_root`."""
    issues: list[str] = []
    evaluators = discover_evaluators()

    readme_path = repo_root / "README.md"
    pyproject_path = repo_root / "pyproject.toml"
    mkdocs_yml_path = repo_root / "mkdocs.yml"
    docs_dir = repo_root / "docs" / "mkdocs"
    examples_readme_path = repo_root / "examples" / "README.md"
    benchmarks_path = docs_dir / "benchmarks.md"

    readme_text = readme_path.read_text(encoding="utf-8")
    mkdocs_text = mkdocs_yml_path.read_text(encoding="utf-8")
    nav_pages = parse_mkdocs_nav(mkdocs_text)
    nav_set = set(nav_pages)

    mkdocs_index_text = (docs_dir / "index.md").read_text(encoding="utf-8")
    metrics_index_text = (docs_dir / "metrics" / "index.md").read_text(encoding="utf-8")
    measurements_index_text = (docs_dir / "measurements" / "index.md").read_text(encoding="utf-8")

    # 1. Evaluator coverage
    for ev in evaluators:
        if (
            f'load("{ev.eval_id}")' not in readme_text
            or f"compute_{ev.eval_id}" not in readme_text
        ):
            issues.append(
                f'README.md: missing `load("{ev.eval_id}")` or `compute_{ev.eval_id}` reference.'
            )
        if (
            f'"{ev.eval_id}"' not in mkdocs_index_text
            or f"compute_{ev.eval_id}" not in mkdocs_index_text
        ):
            issues.append(
                f'docs/mkdocs/index.md: missing `"{ev.eval_id}"` or `compute_{ev.eval_id}`.'
            )
        parent_index_text = metrics_index_text if ev.kind == "metric" else measurements_index_text
        if f"{ev.slug}.md" not in parent_index_text:
            issues.append(
                f"docs/mkdocs/{ev.kind}s/index.md: missing link to `{ev.slug}.md`."
            )
        if not (docs_dir / ev.page_rel_path).is_file():
            issues.append(f"docs/mkdocs/{ev.page_rel_path}: missing evaluator page.")
        if ev.page_rel_path not in nav_set:
            issues.append(f"mkdocs.yml: `nav:` missing `{ev.page_rel_path}`.")

    # 2. MkDocs nav completeness
    for page in nav_pages:
        if not (docs_dir / page).is_file():
            issues.append(f"mkdocs.yml: `nav:` points to missing file `docs/mkdocs/{page}`.")
    for md_file in sorted(docs_dir.rglob("*.md")):
        rel_md = md_file.relative_to(docs_dir).as_posix()
        if rel_md not in nav_set:
            issues.append(f"mkdocs.yml: `nav:` omits existing page `docs/mkdocs/{rel_md}`.")

    # 3. README PyPI link hygiene
    issues.extend(_check_readme_pypi_links(readme_text))

    # 4. pyproject.toml metadata parity
    project_meta = _load_pyproject(pyproject_path).get("project", {})
    req_python = str(project_meta.get("requires-python", ""))
    pyproject_ver = str(project_meta.get("version", ""))
    sqlglot_deps = [d for d in project_meta.get("dependencies", []) if str(d).startswith("sqlglot")]

    if req_python:
        if req_python not in readme_text:
            issues.append(f"README.md: missing `requires-python` string {req_python!r}.")
        if req_python not in mkdocs_index_text:
            issues.append(f"docs/mkdocs/index.md: missing `requires-python` string {req_python!r}.")
    for dep in sqlglot_deps:
        if dep not in readme_text:
            issues.append(f"README.md: missing dependency constraint {dep!r}.")
        if dep not in mkdocs_index_text:
            issues.append(f"docs/mkdocs/index.md: missing dependency constraint {dep!r}.")

    # 5. Benchmark table & version parity
    try:
        parsed_report = parse_example_report_tables(
            examples_readme_path.read_text(encoding="utf-8")
        )
        benchmarks_text = benchmarks_path.read_text(encoding="utf-8")
        if pyproject_ver and parsed_report.gbv_version != pyproject_ver:
            issues.append(
                f"examples/README.md: version `{parsed_report.gbv_version}` != "
                f"`pyproject.toml` version `{pyproject_ver}` (run sync-examples first)."
            )
        if parsed_report.gbv_version != group_by_vibes.__version__:
            issues.append(
                f"examples/README.md: version `{parsed_report.gbv_version}` != "
                f"`group_by_vibes.__version__` `{group_by_vibes.__version__}`."
            )
        if parsed_report.version_line not in benchmarks_text:
            issues.append(
                "docs/mkdocs/benchmarks.md: version/extractor line out of sync "
                "with examples/README.md."
            )
        if parsed_report.headline_table not in benchmarks_text:
            issues.append(
                "docs/mkdocs/benchmarks.md: Comparative Summary table out of sync "
                "with examples/README.md."
            )
        if parsed_report.component_table not in benchmarks_text:
            issues.append(
                "docs/mkdocs/benchmarks.md: Component Matching F1 table out of sync "
                "with examples/README.md."
            )
    except ValueError as exc:
        issues.append(str(exc))

    # 6. Brand asset parity
    mkdocs_assets_dir = docs_dir / "assets"
    source_assets_dir = repo_root / "docs" / "assets"
    asset_files = (
        sorted(p for p in mkdocs_assets_dir.iterdir() if p.is_file())
        if mkdocs_assets_dir.is_dir()
        else []
    )
    for mk_asset in asset_files:
        src_asset = source_assets_dir / mk_asset.name
        if not src_asset.is_file():
            issues.append(
                f"docs/assets/{mk_asset.name}: missing source asset matching "
                f"docs/mkdocs/assets/{mk_asset.name}."
            )
        elif _sha256(mk_asset) != _sha256(src_asset):
            issues.append(
                f"docs/mkdocs/assets/{mk_asset.name}: SHA-256 differs from "
                f"docs/assets/{mk_asset.name}."
            )

    # 7. Python doc snippets
    snippet_count = 0
    if check_snippets:
        snippet_count, snippet_issues = _run_doc_snippets(repo_root)
        issues.extend(snippet_issues)

    return AuditSummary(
        evaluators=tuple(evaluators),
        nav_pages=tuple(nav_pages),
        asset_count=len(asset_files),
        snippet_count=snippet_count,
        issues=tuple(issues),
    )


def _replace_markdown_table_after_heading(text: str, heading_prefix: str, new_table: str) -> str:
    """Replace the first Markdown table following `heading_prefix` while preserving prose."""
    lines = text.splitlines(keepends=True)
    found_heading = False
    start_idx: int | None = None
    end_idx: int | None = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not found_heading:
            if stripped.startswith(heading_prefix):
                found_heading = True
            continue
        if start_idx is None:
            if not stripped:
                continue
            if stripped.startswith("|"):
                start_idx = idx
            elif stripped.startswith("#"):
                break
        else:
            if stripped.startswith("|"):
                continue
            end_idx = idx
            break

    if start_idx is None:
        raise ValueError(
            f"Could not locate table under {heading_prefix!r} in `docs/mkdocs/benchmarks.md`."
        )
    if end_idx is None:
        end_idx = len(lines)

    replacement_lines = [f"{row}\n" for row in new_table.splitlines()]
    return "".join([*lines[:start_idx], *replacement_lines, *lines[end_idx:]])


def sync_benchmarks_and_assets(repo_root: Path) -> list[str]:
    """Synchronize `docs/mkdocs/benchmarks.md` and `docs/mkdocs/assets/` from their sources."""
    changed: list[str] = []
    examples_readme_path = repo_root / "examples" / "README.md"
    benchmarks_path = repo_root / "docs" / "mkdocs" / "benchmarks.md"

    parsed = parse_example_report_tables(examples_readme_path.read_text(encoding="utf-8"))
    original = benchmarks_path.read_text(encoding="utf-8")

    updated, ver_subs = re.subn(
        r"^Evaluated with `[^`]+`, `group_by_vibes` `[^`]+`, and `sqlglot` `[^`]+`:$",
        parsed.version_line,
        original,
        count=1,
        flags=re.MULTILINE,
    )
    if ver_subs != 1:
        raise ValueError(
            "Could not locate `Evaluated with ...:` version line in `docs/mkdocs/benchmarks.md`."
        )

    updated = _replace_markdown_table_after_heading(
        updated, "## Comparative Summary", parsed.headline_table
    )
    updated = _replace_markdown_table_after_heading(
        updated, "### Component Matching F1", parsed.component_table
    )

    if updated != original:
        benchmarks_path.write_text(updated, encoding="utf-8")
        changed.append("docs/mkdocs/benchmarks.md")

    mkdocs_assets_dir = repo_root / "docs" / "mkdocs" / "assets"
    source_assets_dir = repo_root / "docs" / "assets"
    if mkdocs_assets_dir.is_dir():
        for mk_asset in sorted(p for p in mkdocs_assets_dir.iterdir() if p.is_file()):
            src_asset = source_assets_dir / mk_asset.name
            if src_asset.is_file() and _sha256(mk_asset) != _sha256(src_asset):
                shutil.copy2(src_asset, mk_asset)
                changed.append(f"docs/mkdocs/assets/{mk_asset.name}")

    return changed


def cmd_audit(repo_root: Path, *, skip_snippets: bool) -> int:
    summary = collect_issues(repo_root, check_snippets=not skip_snippets)
    eval_list = ", ".join(e.eval_id for e in summary.evaluators)
    print(f"Evaluators ({len(summary.evaluators)}): {eval_list}")
    print(f"MkDocs nav pages: {len(summary.nav_pages)}")
    print(f"Mirrored assets: {summary.asset_count}")
    if not skip_snippets:
        print(f"Executed Python doc snippets: {summary.snippet_count}")
    if summary.issues:
        print(f"Issues ({len(summary.issues)}):")
        for issue in summary.issues:
            print(f"  - {issue}")
        return 1
    print("AUDIT=OK")
    return 0


def cmd_run(repo_root: Path) -> int:
    changed = sync_benchmarks_and_assets(repo_root)
    if changed:
        print(f"SYNC=UPDATED files={','.join(changed)}")
    else:
        print("SYNC=UP_TO_DATE")
    return 0


def cmd_verify(repo_root: Path, *, skip_snippets: bool) -> int:
    summary = collect_issues(repo_root, check_snippets=not skip_snippets)
    if summary.issues:
        for issue in summary.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print(
        f"VERIFY=OK evaluators={len(summary.evaluators)} "
        f"nav_pages={len(summary.nav_pages)} "
        f"snippets={summary.snippet_count} "
        f"assets={summary.asset_count} "
        f"gbv={group_by_vibes.__version__}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and synchronize group-by-vibes documentation with package exports and logs."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root override (defaults to workspace root).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser(
        "audit",
        help="Audit documentation coverage, links, tables, assets, and snippets.",
    )
    audit_p.add_argument(
        "--skip-snippets",
        action="store_true",
        help="Skip executing Python doc snippets.",
    )

    sub.add_parser(
        "run",
        help="Synchronize docs/mkdocs/benchmarks.md and mirrored docs/mkdocs/assets/.",
    )

    verify_p = sub.add_parser("verify", help="Verify all documentation parity checks pass.")
    verify_p.add_argument(
        "--skip-snippets",
        action="store_true",
        help="Skip executing Python doc snippets.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve() if args.repo_root else default_repo_root()
    if args.command == "audit":
        return cmd_audit(root, skip_snippets=args.skip_snippets)
    if args.command == "run":
        return cmd_run(root)
    if args.command == "verify":
        return cmd_verify(root, skip_snippets=args.skip_snippets)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
