"""Optional SQL extraction helpers shared across evaluators."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

ExtractorCallable: TypeAlias = Callable[[str], str | None]
ExtractorLike: TypeAlias = str | re.Pattern[str] | ExtractorCallable | None
CompiledExtractor: TypeAlias = re.Pattern[str] | ExtractorCallable | None

_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_STRAY_THINK_RE = re.compile(r"</?think\b", flags=re.IGNORECASE)
_FENCED_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", flags=re.DOTALL)
_SQL_KEYWORD_RE = re.compile(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b", flags=re.IGNORECASE)
_STANDALONE_QUERY_RE = re.compile(
    r"(?:^|(?<=\n)|(?<=:\s)|(?<=\s))\b((?:WITH|SELECT)\b[\s\S]*?)(?=;|\n\s*\n|\Z)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of running a compiled SQL extractor on a raw text input."""

    sql: str | None
    ok: bool
    failure_reason: str | None


def compile_extractor(extractor: ExtractorLike) -> CompiledExtractor:
    """Validate and compile a regex or callable SQL extractor."""
    if extractor is None:
        return None
    if isinstance(extractor, str):
        compiled = re.compile(extractor)
        if compiled.groups > 1:
            raise ValueError(
                f"Regex extractor may define at most 1 capture group, got {compiled.groups}."
            )
        return compiled
    if isinstance(extractor, re.Pattern):
        if extractor.groups > 1:
            raise ValueError(
                f"Regex extractor may define at most 1 capture group, got {extractor.groups}."
            )
        return extractor
    if callable(extractor):
        return extractor
    raise TypeError(
        f"`extractor` must be str, re.Pattern, callable, or None, got {type(extractor).__name__}."
    )


def apply_extractor(
    raw_text: Any,
    compiled: CompiledExtractor,
    max_input_chars: int | None = 131_072,
) -> ExtractionResult:
    """Apply a compiled extractor when non-None while enforcing `max_input_chars`."""
    if compiled is None:
        if not isinstance(raw_text, str):
            return ExtractionResult(sql=None, ok=False, failure_reason="extraction_error")
        return ExtractionResult(sql=raw_text, ok=True, failure_reason=None)

    if not isinstance(raw_text, str):
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_error")
    if max_input_chars is not None and len(raw_text) > max_input_chars:
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_failed")

    if isinstance(compiled, re.Pattern):
        match = compiled.search(raw_text)
        if match is None:
            return ExtractionResult(sql=None, ok=False, failure_reason="extraction_failed")
        captured: str | None = match.group(1) if compiled.groups == 1 else match.group(0)
        if not isinstance(captured, str) or not captured.strip():
            return ExtractionResult(sql=None, ok=False, failure_reason="extraction_failed")
        return ExtractionResult(sql=captured, ok=True, failure_reason=None)

    try:
        result = compiled(raw_text)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_error")

    if result is None:
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_failed")
    if not isinstance(result, str):
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_error")
    if not result.strip():
        return ExtractionResult(sql=None, ok=False, failure_reason="extraction_failed")
    return ExtractionResult(sql=result, ok=True, failure_reason=None)


def extract_fenced_or_last_sql(text: str) -> str | None:
    """Extract SQL from reasoning/Markdown completions or return `None`."""
    if not isinstance(text, str):
        return None
    stripped = _CLOSED_THINK_RE.sub("", text)
    if _STRAY_THINK_RE.search(stripped):
        return None

    sql_fences: list[str] = []
    generic_sql_fences: list[str] = []
    for lang, body in _FENCED_BLOCK_RE.findall(stripped):
        lang_norm = lang.strip().lower()
        body_stripped = body.strip()
        if not body_stripped:
            continue
        if lang_norm == "sql":
            sql_fences.append(body_stripped)
        elif not lang_norm and _SQL_KEYWORD_RE.search(body_stripped):
            generic_sql_fences.append(body_stripped)

    if sql_fences:
        return sql_fences[-1]
    if generic_sql_fences:
        return generic_sql_fences[-1]

    matches = _STANDALONE_QUERY_RE.findall(stripped)
    if not matches:
        return None
    # Prefer line-start WITH/SELECT candidate if present; fallback to first standalone match.
    candidate = matches[0].strip()
    return candidate if candidate else None
