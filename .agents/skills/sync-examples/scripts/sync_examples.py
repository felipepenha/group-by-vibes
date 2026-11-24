"""Synchronize `examples/logs/` scores and `examples/README.md` with `group-by-vibes`."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import group_by_vibes
from group_by_vibes import compute_sql_bleu, extract_fenced_or_last_sql, normalize_dialect

BUILTIN_EVAL_KEYS = frozenset({"sql_bleu", "valid_sql_rate", "wildcard_detector"})
IGNORED_REPORT_SUBFIELDS = frozenset({"syntax_valid_rate", "schema_grounded_rate"})
FAILURE_KEY_ORDER = (
    "unknown_column",
    "syntax_error",
    "unknown_table",
    "multi_statement",
    "extraction_failed",
    "extraction_error",
    "empty_input",
    "command_fallback",
)
DIFFICULTY_ORDER = ("simple", "moderate", "challenging")
FORBIDDEN_README_SUBSTRINGS = (
    "Syntax Valid",
    "Schema Grounded",
    "syntax_valid_rate",
    "schema_grounded_rate",
    "/Users/",
    "artifacts/",
)
BANNED_PROMO_WORDS = (
    "seamless",
    "seamlessly",
    "robust",
    "blazingly",
    "effortlessly",
    "literally",
    "drastically",
    "obviously",
    "brilliant",
    "great",
    "smart",
    "correct",
    "correctly",
    "perfect",
    "excellent",
    "spot-on",
    "clean",
    "cleanly",
    "elegant",
    "comprehensive",
    "powerful",
)
BANNED_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in BANNED_PROMO_WORDS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScoredRun:
    """Loaded JSONL completions and sibling `*_scores.json` report for one model run."""

    stem: str
    jsonl_path: Path
    score_path: Path
    model: str
    dialect: str
    num_rows: int
    rows: list[dict[str, Any]]
    scores: dict[str, Any]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "examples" / "bird_eval.py"
        ).is_file():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    raise RuntimeError("Could not locate repository root containing `examples/bird_eval.py`.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _expected_evaluators() -> list[str]:
    exported = getattr(group_by_vibes, "__all__", [])
    return sorted(name.removeprefix("compute_") for name in exported if name.startswith("compute_"))


def _run_bird_score(
    root: Path, input_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(root / "examples" / "bird_eval.py"),
        "score",
        "--input",
        str(input_path),
    ]
    if output_path is not None:
        cmd.extend(["--output", str(output_path)])
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    if output_path is not None:
        return json.loads(output_path.read_text(encoding="utf-8"))
    return json.loads(proc.stdout)


def _score_eval_keys(payload: dict[str, Any]) -> set[str]:
    return set(payload.keys()) - {"metadata", "by_difficulty"}


def _inspect_log_file(
    root: Path,
    jsonl_path: Path,
    expected_evals: Sequence[str],
    gbv_ver: str,
    sqlglot_ver: str,
) -> dict[str, Any]:
    rows = _load_jsonl(jsonl_path)
    models = sorted({str(r.get("model", "")) for r in rows if r.get("model")})
    dialects = sorted({normalize_dialect(str(r.get("dialect") or "sqlite")) for r in rows})
    score_path = jsonl_path.with_name(f"{jsonl_path.stem}_scores.json")
    is_probe = jsonl_path.stem.endswith("_probe")

    score_exists = score_path.is_file()
    score_gbv_ver: str | None = None
    score_sg_ver: str | None = None
    score_rows: int | None = None
    missing_evals: list[str] = list(expected_evals)
    stale_reason: list[str] = []

    if score_exists:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        meta = payload.get("metadata", {})
        score_gbv_ver = meta.get("group_by_vibes_version")
        score_sg_ver = meta.get("sqlglot_version")
        score_rows = meta.get("num_rows")
        present = _score_eval_keys(payload)
        missing_evals = sorted(set(expected_evals) - present)
        if score_gbv_ver != gbv_ver:
            stale_reason.append(f"gbv_version {score_gbv_ver!r} != {gbv_ver!r}")
        if score_sg_ver != sqlglot_ver:
            stale_reason.append(f"sqlglot_version {score_sg_ver!r} != {sqlglot_ver!r}")
        if score_rows != len(rows):
            stale_reason.append(f"num_rows {score_rows} != {len(rows)}")
        if jsonl_path.stat().st_mtime > score_path.stat().st_mtime:
            stale_reason.append("jsonl newer than score json")
        if missing_evals:
            stale_reason.append(f"missing evaluators: {missing_evals}")
    else:
        stale_reason.append("missing score json")

    return {
        "path": str(jsonl_path.relative_to(root)),
        "stem": jsonl_path.stem,
        "is_probe": is_probe,
        "num_rows": len(rows),
        "models": models,
        "dialects": dialects,
        "score_path": str(score_path.relative_to(root)),
        "score_exists": score_exists,
        "score_gbv_version": score_gbv_ver,
        "score_sqlglot_version": score_sg_ver,
        "missing_evaluators": missing_evals,
        "stale": bool(stale_reason),
        "stale_reasons": stale_reason,
    }


def cmd_audit(root: Path) -> int:
    """Audit exported `group_by_vibes` evaluators against `bird_eval.py` and `examples/logs/`."""
    gbv_ver = group_by_vibes.__version__
    sqlglot_ver = importlib.metadata.version("sqlglot")
    expected = _expected_evaluators()

    sample_path = root / "examples" / "fixtures" / "bird_sample.jsonl"
    sample_payload = _run_bird_score(root, sample_path)
    bird_eval_keys = sorted(_score_eval_keys(sample_payload))
    unwired = sorted(set(expected) - set(bird_eval_keys))

    logs_dir = root / "examples" / "logs"
    jsonl_files = sorted(logs_dir.glob("*.jsonl")) if logs_dir.is_dir() else []
    log_reports = [
        _inspect_log_file(root, path, expected, gbv_ver, sqlglot_ver) for path in jsonl_files
    ]

    report = {
        "group_by_vibes_version": gbv_ver,
        "sqlglot_version": sqlglot_ver,
        "exported_evaluators": expected,
        "bird_eval_evaluators": bird_eval_keys,
        "unwired_evaluators": unwired,
        "logs": log_reports,
    }
    print(json.dumps(report, indent=2))
    return 1 if unwired else 0


def _short_labels(models: Sequence[str]) -> tuple[list[str], list[str]]:
    raw_short: list[str] = []
    raw_delta: list[str] = []
    for model in models:
        low = model.lower()
        if "xiyansql" in low:
            raw_short.append("XiYanSQL")
            raw_delta.append("XiYan")
        elif "finer-sql" in low or "finersql" in low:
            raw_short.append("FINER-SQL")
            raw_delta.append("FINER-SQL")
        elif "qwen" in low:
            raw_short.append("Qwen")
            raw_delta.append("Qwen")
        else:
            tail = model.split("/")[-1]
            raw_short.append(tail)
            raw_delta.append(tail)

    if len(set(raw_short)) < len(raw_short):
        raw_short = [m.split("/")[-1] for m in models]
        raw_delta = list(raw_short)
    return raw_short, raw_delta


def _fmt_delta_float(val: float, digits: int = 4) -> str:
    if abs(val) < 0.5 * (10.0 ** (-digits)):
        return f"`{0.0:.{digits}f}`"
    return f"`{val:+.{digits}f}`"


def _fmt_delta_pp(val: float, digits: int = 1) -> str:
    pp = val * 100.0
    if abs(pp) < 0.5 * (10.0 ** (-digits)):
        return f"`{0.0:.{digits}f} pp`"
    return f"`{pp:+.{digits}f} pp`"


def _fmt_delta_int(val: int) -> str:
    if val == 0:
        return "`0`"
    return f"`{val:+d}`"


def _dialect_display(dialect: str) -> str:
    return {"sqlite": "SQLite", "postgres": "PostgreSQL", "mysql": "MySQL"}.get(
        dialect, dialect.upper()
    )


def _bleu_stats(per_example: Sequence[float | None]) -> dict[str, Any]:
    vals = sorted(float(v) for v in per_example if v is not None)
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "exact": 0,
        }
    return {
        "n": n,
        "min": vals[0],
        "p25": vals[n // 4],
        "median": vals[n // 2],
        "mean": sum(vals) / n,
        "p75": vals[(3 * n) // 4],
        "max": vals[-1],
        "exact": sum(1 for x in vals if x >= 1.0 - 1e-9),
    }


def _md_table(
    headers: Sequence[str], alignments: Sequence[str], rows: Sequence[Sequence[str]]
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _extract_patterns(sql_list: Sequence[str]) -> dict[str, float | int]:
    n = len(sql_list)
    if n == 0:
        return {"t1": 0, "inner": 0, "with": 0, "cast": 0, "cond": 0, "avg_chars": 0.0}
    return {
        "t1": sum(" T1" in s or " AS T1" in s for s in sql_list),
        "inner": sum("INNER JOIN" in s.upper() for s in sql_list),
        "with": sum(s.lstrip().upper().startswith("WITH ") for s in sql_list),
        "cast": sum("CAST(" in s.upper() for s in sql_list),
        "cond": sum(("CASE " in s.upper() or "IIF(" in s.upper()) for s in sql_list),
        "avg_chars": sum(len(s) for s in sql_list) / n,
    }


def _render_headline_section(runs: Sequence[ScoredRun], delta_tag: str | None) -> list[str]:
    is_pair = len(runs) == 2 and delta_tag is not None
    headers = ["Metric"] + [f"`{r.model}`" for r in runs]
    aligns = ["---"] + ["---:"] * len(runs)
    if is_pair:
        headers.append(f"Delta ({delta_tag})")
        aligns.append("---:")

    rows: list[list[str]] = []
    sb_list = [r.scores["sql_bleu"] for r in runs]
    vs_list = [r.scores["valid_sql_rate"] for r in runs]
    wd_list = [r.scores["wildcard_detector"] for r in runs]
    meta_list = [r.scores["metadata"] for r in runs]
    bstats = [_bleu_stats(sb["per_example_bleu"]) for sb in sb_list]

    def add_row(label: str, vals: list[str], delta: str | None = None) -> None:
        if is_pair:
            rows.append([label, *vals, delta or ""])
        else:
            rows.append([label, *vals])

    add_row(
        "**SQL-BLEU**",
        [f"`{sb['bleu']:.4f}`" for sb in sb_list],
        _fmt_delta_float(sb_list[1]["bleu"] - sb_list[0]["bleu"], 4) if is_pair else None,
    )
    add_row(
        "**Parsed Queries (SQL-BLEU)**",
        [
            f"`{sb['parsed_count']}/{sb['num_examples']}` (`{sb['parse_success_rate']:.1%}`)"
            for sb in sb_list
        ],
        (
            f"{_fmt_delta_pp(sb_list[1]['parse_success_rate'] - sb_list[0]['parse_success_rate'])} "
            f"({_fmt_delta_int(sb_list[1]['parsed_count'] - sb_list[0]['parsed_count'])})"
        )
        if is_pair
        else None,
    )
    add_row(
        "**SQL-BLEU precisions (1 / 2 / 3 / 4)**",
        ["`" + " / ".join(f"{p:.3f}" for p in sb["precisions"]) + "`" for sb in sb_list],
        (
            "`"
            + " / ".join(
                f"{p1 - p0:+.3f}"
                for p0, p1 in zip(sb_list[0]["precisions"], sb_list[1]["precisions"], strict=True)
            )
            + "`"
        )
        if is_pair
        else None,
    )
    add_row(
        "**AST Token Length (Pred / Gold)**",
        [f"`{sb['translation_length']:,} / {sb['reference_length']:,}`" for sb in sb_list],
        f"`{sb_list[1]['translation_length'] - sb_list[0]['translation_length']:+,}` pred tokens"
        if is_pair
        else None,
    )
    add_row(
        "**Length Ratio / Brevity Penalty**",
        [f"`{sb['length_ratio']:.3f}` / `{sb['brevity_penalty']:.3f}`" for sb in sb_list],
        (
            f"{_fmt_delta_float(sb_list[1]['length_ratio'] - sb_list[0]['length_ratio'], 3)} / "
            f"{_fmt_delta_float(sb_list[1]['brevity_penalty'] - sb_list[0]['brevity_penalty'], 3)}"
        )
        if is_pair
        else None,
    )
    add_row(
        "**Per-example SQL-BLEU (min / p25 / median / mean / p75 / max)**",
        [
            "`"
            + " / ".join(f"{st[k]:.4f}" for k in ("min", "p25", "median", "mean", "p75", "max"))
            + "`"
            for st in bstats
        ],
        (
            f"median {_fmt_delta_float(bstats[1]['median'] - bstats[0]['median'], 4)}, "
            f"mean {_fmt_delta_float(bstats[1]['mean'] - bstats[0]['mean'], 4)}"
        )
        if is_pair
        else None,
    )
    add_row(
        "**Exact Canonical Matches (`BLEU = 1.0`)**",
        [f"`{st['exact']}`" for st in bstats],
        _fmt_delta_int(bstats[1]["exact"] - bstats[0]["exact"]) if is_pair else None,
    )
    add_row(
        "**Valid SQL Rate**",
        [
            f"`{vs['valid_sql_rate']:.1%}` (`{vs['valid_sql_count']}/{vs['num_examples']}`)"
            for vs in vs_list
        ],
        (
            f"{_fmt_delta_pp(vs_list[1]['valid_sql_rate'] - vs_list[0]['valid_sql_rate'])} "
            f"({_fmt_delta_int(vs_list[1]['valid_sql_count'] - vs_list[0]['valid_sql_count'])})"
        )
        if is_pair
        else None,
    )
    add_row(
        "**Wildcard Rate (`SELECT *`)**",
        [
            f"`{wd['wildcard_rate']:.1%}` (`{wd['wildcard_count']}/{wd['parsed_count']}`)"
            for wd in wd_list
        ],
        _fmt_delta_pp(wd_list[1]["wildcard_rate"] - wd_list[0]["wildcard_rate"])
        if is_pair
        else None,
    )

    extra_keys = sorted(set(_score_eval_keys(runs[0].scores)) - BUILTIN_EVAL_KEYS)
    for eval_key in extra_keys:
        dicts = [r.scores.get(eval_key, {}) for r in runs]
        scalar_fields = [
            k
            for k, v in dicts[0].items()
            if k not in IGNORED_REPORT_SUBFIELDS
            and not k.startswith("per_example")
            and k not in {"num_examples", "dialect"}
            and isinstance(v, (int, float, bool, str))
        ]
        for field in scalar_fields:
            vals = [d.get(field) for d in dicts]
            rendered = [f"`{v:.4f}`" if isinstance(v, float) else f"`{v}`" for v in vals]
            delta_str: str | None = None
            if is_pair and isinstance(vals[0], (int, float)) and isinstance(vals[1], (int, float)):
                if isinstance(vals[0], float) or isinstance(vals[1], float):
                    delta_str = _fmt_delta_float(float(vals[1]) - float(vals[0]), 4)
                else:
                    delta_str = _fmt_delta_int(int(vals[1]) - int(vals[0]))
            add_row(f"**{eval_key}.{field}**", rendered, delta_str)

    add_row(
        "**Truncated Completions**",
        [f"`{m['truncated_count']}`" for m in meta_list],
        _fmt_delta_int(int(meta_list[1]["truncated_count"]) - int(meta_list[0]["truncated_count"]))
        if is_pair
        else None,
    )

    title = "### Headline Comparison" if len(runs) >= 2 else "### Headline Metrics"
    return [title, "", *_md_table(headers, aligns, rows)]


def _render_difficulty_section(runs: Sequence[ScoredRun], short_labels: Sequence[str]) -> list[str]:
    is_pair = len(runs) == 2
    diff_keys_all = {k for r in runs for k in r.scores.get("by_difficulty", {})}
    ordered_diffs = [d for d in DIFFICULTY_ORDER if d in diff_keys_all] + sorted(
        diff_keys_all - set(DIFFICULTY_ORDER)
    )

    headers = ["Difficulty", "Rows"]
    aligns = ["---", "---:"]
    for label in short_labels:
        headers.append(f"{label} SQL-BLEU")
        aligns.append("---:")
    if is_pair:
        headers.append("Delta BLEU")
        aligns.append("---:")
    for label in short_labels:
        headers.append(f"{label} Valid SQL")
        aligns.append("---:")
    if is_pair:
        headers.append("Delta Valid SQL")
        aligns.append("---:")

    table_rows: list[list[str]] = []
    for diff in ordered_diffs:
        slices = [r.scores["by_difficulty"][diff] for r in runs]
        num_rows = slices[0]["num_rows"]
        valid_counts = [
            sum(
                1
                for i, item in enumerate(r.rows)
                if str(item.get("difficulty") or "unknown") == diff
                and r.scores["valid_sql_rate"]["per_example"][i]["syntax_valid"]
                and r.scores["valid_sql_rate"]["per_example"][i]["schema_valid"] is not False
            )
            for r in runs
        ]
        row: list[str] = [f"`{diff}`", str(num_rows)]
        row.extend(f"`{s['bleu']:.4f}`" for s in slices)
        if is_pair:
            row.append(_fmt_delta_float(slices[1]["bleu"] - slices[0]["bleu"], 4))
        row.extend(
            f"`{s['valid_sql_rate']:.1%}` (`{cnt}`)"
            for s, cnt in zip(slices, valid_counts, strict=True)
        )
        if is_pair:
            row.append(_fmt_delta_pp(slices[1]["valid_sql_rate"] - slices[0]["valid_sql_rate"]))
        table_rows.append(row)

    return ["### Breakdown by Difficulty", "", *_md_table(headers, aligns, table_rows)]


def _render_database_section(
    runs: Sequence[ScoredRun], short_labels: Sequence[str], dialect: str
) -> list[str]:
    is_pair = len(runs) == 2
    by_db: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(runs[0].rows):
        by_db[str(item["db_id"])].append(idx)

    headers = ["Database", "Rows"]
    aligns = ["---", "---:"]
    for label in short_labels:
        headers.append(f"{label} SQL-BLEU")
        aligns.append("---:")
    if is_pair:
        headers.append("Delta BLEU")
        aligns.append("---:")
    for label in short_labels:
        headers.append(f"{label} Valid SQL")
        aligns.append("---:")
    if is_pair:
        headers.append("Delta Valid SQL")
        aligns.append("---:")

    table_rows: list[list[str]] = []
    for db in sorted(by_db):
        idxs = by_db[db]
        n = len(idxs)
        bleus: list[float] = []
        valid_counts: list[int] = []
        valid_rates: list[float] = []

        for r in runs:
            row_map = {item["question_id"]: i for i, item in enumerate(r.rows)}
            mapped_idxs = [row_map[runs[0].rows[i]["question_id"]] for i in idxs]
            preds = [str(r.rows[i]["raw_completion"]) for i in mapped_idxs]
            refs = [[str(r.rows[i]["gold_sql"])] for i in mapped_idxs]
            b_res = compute_sql_bleu(
                preds, refs, dialect=dialect, extractor=extract_fenced_or_last_sql
            )
            bleus.append(float(b_res["bleu"]) if b_res["bleu"] is not None else 0.0)
            v_cnt = sum(
                1
                for i in mapped_idxs
                if r.scores["valid_sql_rate"]["per_example"][i]["syntax_valid"]
                and r.scores["valid_sql_rate"]["per_example"][i]["schema_valid"] is not False
            )
            valid_counts.append(v_cnt)
            valid_rates.append(v_cnt / n)

        row: list[str] = [f"`{db}`", str(n)]
        row.extend(f"`{b:.4f}`" for b in bleus)
        if is_pair:
            row.append(_fmt_delta_float(bleus[1] - bleus[0], 4))
        row.extend(
            f"`{vr:.1%}` (`{cnt}`)" for vr, cnt in zip(valid_rates, valid_counts, strict=True)
        )
        if is_pair:
            row.append(_fmt_delta_pp(valid_rates[1] - valid_rates[0]))
        table_rows.append(row)

    return ["### Breakdown by Database", "", *_md_table(headers, aligns, table_rows)]


def _render_paired_section(runs: Sequence[ScoredRun], short_labels: Sequence[str]) -> list[str]:
    if len(runs) != 2:
        return []
    r0, r1 = runs[0], runs[1]
    l0, l1 = short_labels[0], short_labels[1]
    qids0 = {item["question_id"]: i for i, item in enumerate(r0.rows)}
    qids1 = {item["question_id"]: i for i, item in enumerate(r1.rows)}
    if set(qids0) != set(qids1):
        return []

    n = len(qids0)
    both_ok = r1_only = r0_only = neither = 0
    r0_wins = r1_wins = ties = 0
    joint_b0: list[float] = []
    joint_b1: list[float] = []

    for qid in sorted(qids0):
        i0, i1 = qids0[qid], qids1[qid]
        v0 = r0.scores["valid_sql_rate"]["per_example"][i0]
        v1 = r1.scores["valid_sql_rate"]["per_example"][i1]
        ok0 = bool(v0["syntax_valid"] and v0["schema_valid"] is not False)
        ok1 = bool(v1["syntax_valid"] and v1["schema_valid"] is not False)
        if ok0 and ok1:
            both_ok += 1
        elif ok1:
            r1_only += 1
        elif ok0:
            r0_only += 1
        else:
            neither += 1

        b0 = r0.scores["sql_bleu"]["per_example_bleu"][i0]
        b1 = r1.scores["sql_bleu"]["per_example_bleu"][i1]
        if b0 is not None and b1 is not None:
            joint_b0.append(float(b0))
            joint_b1.append(float(b1))
        sb0 = -1.0 if b0 is None else float(b0)
        sb1 = -1.0 if b1 is None else float(b1)
        if abs(sb1 - sb0) < 1e-9:
            ties += 1
        elif sb0 > sb1:
            r0_wins += 1
        else:
            r1_wins += 1

    valid_rows = [
        ["Both models valid", f"`{both_ok}`", f"`{both_ok / n:.1%}`"],
        [f"{l1} valid only", f"`{r1_only}`", f"`{r1_only / n:.1%}`"],
        [f"{l0} valid only", f"`{r0_only}`", f"`{r0_only / n:.1%}`"],
        ["Neither model valid", f"`{neither}`", f"`{neither / n:.1%}`"],
    ]
    bleu_rows = [
        [f"{l0} higher BLEU", f"`{r0_wins}`", f"`{r0_wins / n:.1%}`"],
        [f"{l1} higher BLEU", f"`{r1_wins}`", f"`{r1_wins / n:.1%}`"],
        ["Tied", f"`{ties}`", f"`{ties / n:.1%}`"],
    ]

    mean_note = ""
    if joint_b0:
        m0 = sum(joint_b0) / len(joint_b0)
        m1 = sum(joint_b1) / len(joint_b1)
        mean_note = (
            f"Unparseable predictions rank below parsed predictions; across the `{len(joint_b0)}` "
            f"jointly parsed queries, mean per-example `SQL-BLEU` is `{m0:.4f}` for {l0} vs. "
            f"`{m1:.4f}` for {l1} ({_fmt_delta_float(m1 - m0, 4)})."
        )

    lines = [
        f"### Paired Head-to-Head Outcomes (Same {n} Questions)",
        "",
        "#### Valid SQL Rate Agreement",
        "",
        *_md_table(["Outcome Category", "Count", "Share"], ["---", "---:", "---:"], valid_rows),
        "",
        f"#### Per-Example SQL-BLEU Head-to-Head (`n = {n}`)",
        "",
    ]
    if mean_note:
        lines.extend([mean_note, ""])
    lines.extend(
        _md_table(["Outcome Category", "Count", "Share"], ["---", "---:", "---:"], bleu_rows)
    )
    return lines


def _render_failure_section(runs: Sequence[ScoredRun], delta_tag: str | None) -> list[str]:
    is_pair = len(runs) == 2 and delta_tag is not None
    all_keys = {k for r in runs for k in r.scores["valid_sql_rate"]["failure_counts"]}
    ordered = [k for k in FAILURE_KEY_ORDER if k in all_keys] + sorted(
        all_keys - set(FAILURE_KEY_ORDER)
    )

    headers = ["Failure Reason"] + [f"`{r.model}`" for r in runs]
    aligns = ["---"] + ["---:"] * len(runs)
    if is_pair:
        headers.append(f"Delta ({delta_tag})")
        aligns.append("---:")

    table_rows: list[list[str]] = []
    for key in ordered:
        counts = [int(r.scores["valid_sql_rate"]["failure_counts"].get(key, 0)) for r in runs]
        row = [f"`{key}`", *(f"`{c}`" for c in counts)]
        if is_pair:
            row.append(_fmt_delta_int(counts[1] - counts[0]))
        table_rows.append(row)

    return [
        "### Failure Breakdown (`valid_sql_rate.failure_counts`)",
        "",
        *_md_table(headers, aligns, table_rows),
    ]


def _render_style_section(runs: Sequence[ScoredRun]) -> list[str]:
    n = runs[0].num_rows
    gold_stats = _extract_patterns([str(item["gold_sql"]) for item in runs[0].rows])
    model_stats = [
        _extract_patterns(
            [extract_fenced_or_last_sql(str(item["raw_completion"])) or "" for item in r.rows]
        )
        for r in runs
    ]
    headers = ["Pattern", "Gold SQL"] + [f"`{r.model}`" for r in runs]
    aligns = ["---", "---:"] + ["---:"] * len(runs)
    specs = (
        ("`T1`/`T2` table aliases", "t1", False),
        ("Explicit `INNER JOIN`", "inner", False),
        ("`WITH` CTEs", "with", False),
        ("`CAST(...)` expressions", "cast", False),
        ("`CASE` / `IIF` conditionals", "cond", False),
        ("Mean extracted query length (chars)", "avg_chars", True),
    )

    table_rows: list[list[str]] = []
    for label, key, is_float in specs:
        all_stats = [gold_stats, *model_stats]
        if is_float:
            vals = [f"`{float(st[key]):.1f}`" for st in all_stats]
        else:
            vals = [f"`{int(st[key])}` (`{int(st[key]) / n:.1%}`)" for st in all_stats]
        table_rows.append([label, *vals])

    return [
        f"### SQL Style & Structural Patterns (`n = {n}`)",
        "",
        *_md_table(headers, aligns, table_rows),
    ]


def _render_cohort_report(dialect: str, num_rows: int, runs: Sequence[ScoredRun]) -> list[str]:
    dialect_label = _dialect_display(dialect)
    split_desc = "Full Split" if num_rows == 500 else "Stratified Sample"
    heading = (
        f"## Comparative Evaluation Report ({dialect_label}, {split_desc}, n={num_rows})"
        if len(runs) >= 2
        else f"## Evaluation Report ({dialect_label}, {split_desc}, n={num_rows})"
    )

    short_labels, delta_labels = _short_labels([r.model for r in runs])
    delta_tag = f"`{delta_labels[1]} - {delta_labels[0]}`" if len(runs) == 2 else None

    diff_counts = Counter(str(item.get("difficulty") or "unknown") for item in runs[0].rows)
    diff_order = ["challenging", "moderate", "simple"]
    ordered_diffs = [d for d in diff_order if d in diff_counts] + sorted(
        set(diff_counts) - set(diff_order)
    )
    diff_summary = ", ".join(f"`{diff_counts[d]}` {d}" for d in ordered_diffs)
    db_count = len({str(item["db_id"]) for item in runs[0].rows})

    meta0 = runs[0].scores["metadata"]
    gbv_ver = meta0["group_by_vibes_version"]
    sg_ver = meta0["sqlglot_version"]
    score_list = ", ".join(f"`{r.score_path.as_posix()}`" for r in runs)
    models_str = " vs. ".join(f"`{r.model}`" for r in runs)

    if len(runs) == 2 and {r["question_id"] for r in runs[0].rows} == {
        r["question_id"] for r in runs[1].rows
    }:
        intro = (
            f"Paired {'full-split ' if num_rows == 500 else ''}evaluation on "
            f"`birdsql/bird_mini_dev` (`{dialect}`), comparing {models_str} over the identical "
            f"`{num_rows}` questions."
        )
    else:
        intro = (
            f"Evaluation on `birdsql/bird_mini_dev` (`{dialect}`, `n = {num_rows}`): {models_str}."
        )

    split_prefix = f"full `{num_rows}`" if num_rows == 500 else f"`{num_rows}`"
    dataset_line = (
        f"* **Dataset & Split:** `birdsql/bird_mini_dev` (`{dialect}`, {split_prefix} rows: "
        f"{diff_summary} across `{db_count}` databases)"
    )
    extractor_line = (
        f"* **Extractor & Versions:** `{meta0['extractor']}`, "
        f"`group_by_vibes` `{gbv_ver}`, `sqlglot` `{sg_ver}`"
    )
    sections: list[list[str]] = [
        [heading, "", intro],
        [
            "### Run Metadata",
            "",
            dataset_line,
            f"* **{'Models Compared' if len(runs) >= 2 else 'Model'}:** {models_str}",
            extractor_line,
            f"* **Score Reports:** {score_list}",
        ],
        _render_headline_section(runs, delta_tag),
        _render_difficulty_section(runs, short_labels),
        _render_database_section(runs, short_labels, dialect),
    ]
    paired = _render_paired_section(runs, short_labels)
    if paired:
        sections.append(paired)
    sections.append(_render_failure_section(runs, delta_tag))
    sections.append(_render_style_section(runs))

    lines: list[str] = []
    for idx, block in enumerate(sections):
        if idx > 0:
            lines.append("")
        lines.extend(block)
    return lines


def _update_readme_markdown(readme_path: Path, generated_block: str) -> None:
    content = readme_path.read_text(encoding="utf-8")
    start_match = re.search(
        r"^## (?:Comparative|Sample) Evaluation Report\b", content, flags=re.MULTILINE
    )
    takeaway_match = re.search(r"^### Key Comparative Takeaways\b", content, flags=re.MULTILINE)

    if start_match and takeaway_match and takeaway_match.start() > start_match.start():
        updated = (
            content[: start_match.start()]
            + generated_block.rstrip()
            + "\n\n"
            + content[takeaway_match.start() :]
        )
    elif start_match:
        updated = content[: start_match.start()] + generated_block.rstrip() + "\n"
    else:
        updated = (
            content.rstrip()
            + "\n\n"
            + generated_block.rstrip()
            + "\n\n### Key Comparative Takeaways\n\n* Update takeaways based on the tables above.\n"
        )
    readme_path.write_text(updated, encoding="utf-8")


def cmd_run(root: Path, *, update_readme: bool, include_probes: bool) -> int:
    """Re-score fixture + logs via `bird_eval.py` and optionally update `examples/README.md`."""
    sample_in = root / "examples" / "fixtures" / "bird_sample.jsonl"
    sample_out = root / "examples" / "logs" / "bird_sample_scores.json"
    _run_bird_score(root, sample_in, sample_out)

    logs_dir = root / "examples" / "logs"
    jsonl_files = sorted(logs_dir.glob("*.jsonl")) if logs_dir.is_dir() else []
    runs: list[ScoredRun] = []

    for jsonl_path in jsonl_files:
        if jsonl_path.stem.endswith("_probe") and not include_probes:
            continue
        score_path = jsonl_path.with_name(f"{jsonl_path.stem}_scores.json")
        scores = _run_bird_score(root, jsonl_path, score_path)
        rows = _load_jsonl(jsonl_path)
        if not rows:
            continue
        dialect = normalize_dialect(str(scores["metadata"]["dialect"]))
        model = str(rows[0].get("model") or jsonl_path.stem)
        runs.append(
            ScoredRun(
                stem=jsonl_path.stem,
                jsonl_path=jsonl_path.relative_to(root),
                score_path=score_path.relative_to(root),
                model=model,
                dialect=dialect,
                num_rows=len(rows),
                rows=rows,
                scores=scores,
            )
        )

    cohorts: dict[tuple[str, int], list[ScoredRun]] = defaultdict(list)
    for run in runs:
        cohorts[(run.dialect, run.num_rows)].append(run)

    rendered_blocks: list[str] = []
    for (dialect, num_rows), cohort_runs in sorted(
        cohorts.items(), key=lambda item: (item[0][0], -item[0][1])
    ):
        cohort_runs_sorted = sorted(cohort_runs, key=lambda r: r.stem)
        rendered_blocks.append(
            "\n".join(_render_cohort_report(dialect, num_rows, cohort_runs_sorted))
        )

    full_block = "\n\n".join(rendered_blocks)
    if update_readme:
        _update_readme_markdown(root / "examples" / "README.md", full_block)
        print(
            f"Updated examples/README.md across {len(runs)} log run(s) "
            f"and {len(cohorts)} cohort(s); refreshed {sample_out.relative_to(root)}."
        )
    else:
        print(full_block)
    return 0


def cmd_verify(root: Path, *, include_probes: bool) -> int:
    """Verify score freshness, evaluator coverage, and `examples/README.md` hygiene."""
    gbv_ver = group_by_vibes.__version__
    sqlglot_ver = importlib.metadata.version("sqlglot")
    expected = set(_expected_evaluators())
    errors: list[str] = []

    sample_in = root / "examples" / "fixtures" / "bird_sample.jsonl"
    sample_score = root / "examples" / "logs" / "bird_sample_scores.json"
    score_targets: list[tuple[Path, Path]] = [(sample_in, sample_score)]

    logs_dir = root / "examples" / "logs"
    for jsonl_path in sorted(logs_dir.glob("*.jsonl")) if logs_dir.is_dir() else []:
        if jsonl_path.stem.endswith("_probe") and not include_probes:
            continue
        score_targets.append((jsonl_path, jsonl_path.with_name(f"{jsonl_path.stem}_scores.json")))

    readme_path = root / "examples" / "README.md"
    readme = readme_path.read_text(encoding="utf-8")

    for jsonl_path, score_path in score_targets:
        rel_score = score_path.relative_to(root).as_posix()
        if not score_path.is_file():
            errors.append(f"Missing score file: {rel_score}")
            continue
        rows = _load_jsonl(jsonl_path)
        payload = json.loads(score_path.read_text(encoding="utf-8"))
        meta = payload.get("metadata", {})
        if meta.get("group_by_vibes_version") != gbv_ver:
            errors.append(
                f"{rel_score}: version {meta.get('group_by_vibes_version')!r} != {gbv_ver!r}"
            )
        if meta.get("sqlglot_version") != sqlglot_ver:
            errors.append(
                f"{rel_score}: sqlglot {meta.get('sqlglot_version')!r} != {sqlglot_ver!r}"
            )
        if meta.get("num_rows") != len(rows):
            errors.append(f"{rel_score}: num_rows {meta.get('num_rows')} != {len(rows)}")
        if jsonl_path.stat().st_mtime > score_path.stat().st_mtime:
            errors.append(f"{rel_score}: older than {jsonl_path.relative_to(root).as_posix()}")
        missing = sorted(expected - _score_eval_keys(payload))
        if missing:
            errors.append(f"{rel_score}: missing evaluators {missing}")
        if rel_score not in readme:
            errors.append(f"examples/README.md missing reference to {rel_score}")

    if gbv_ver not in readme:
        errors.append(f"examples/README.md missing group_by_vibes version {gbv_ver!r}")
    if sqlglot_ver not in readme:
        errors.append(f"examples/README.md missing sqlglot version {sqlglot_ver!r}")

    for term in FORBIDDEN_README_SUBSTRINGS:
        if term in readme:
            errors.append(f"examples/README.md contains forbidden substring: {term!r}")
    if "llmsql" in readme.lower():
        errors.append("examples/README.md contains forbidden `llmsql` reference")
    banned_hits = BANNED_WORD_RE.findall(readme)
    if banned_hits:
        errors.append(f"examples/README.md contains banned words: {sorted(set(banned_hits))}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(
        f"VERIFY=OK files={len(score_targets)} gbv={gbv_ver} "
        f"sqlglot={sqlglot_ver} lines={len(readme.splitlines())}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize examples/logs scores and examples/README.md with group-by-vibes."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Audit exported evaluators vs bird_eval.py and log files.")

    run_p = sub.add_parser("run", help="Re-score logs and render or update examples/README.md.")
    run_p.add_argument(
        "--update-readme",
        action="store_true",
        help="Write rendered comparative report sections back into examples/README.md.",
    )
    run_p.add_argument(
        "--include-probes",
        action="store_true",
        help="Include `*_probe.jsonl` files when scoring and rendering.",
    )

    ver_p = sub.add_parser(
        "verify", help="Verify score freshness, evaluator coverage, and README hygiene."
    )
    ver_p.add_argument(
        "--include-probes",
        action="store_true",
        help="Require fresh scores and README links for `*_probe.jsonl` files as well.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    if args.command == "audit":
        return cmd_audit(root)
    if args.command == "run":
        return cmd_run(
            root,
            update_readme=bool(args.update_readme),
            include_probes=bool(args.include_probes),
        )
    if args.command == "verify":
        return cmd_verify(root, include_probes=bool(args.include_probes))
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
