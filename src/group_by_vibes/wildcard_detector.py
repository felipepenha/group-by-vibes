"""Reference-free Wildcard Detector measurement for `SELECT *` and `table.*`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlglot import exp
from sqlglot.errors import SqlglotError

from .base import BaseEvaluator
from .extract import ExtractorLike, apply_extractor
from .parser import parse_sql


def _is_direct_exists_target(sel: exp.Select) -> bool:
    parent = sel.parent
    if isinstance(parent, exp.Exists):
        return True
    return (
        isinstance(parent, exp.Subquery)
        and isinstance(parent.parent, exp.Exists)
        and parent.parent.this is parent
    )


def _has_projection_wildcard(ast: exp.Expr) -> bool:
    """Return True if any non-EXISTS SELECT clause projects `*` or `table.*`."""
    for sel in ast.find_all(exp.Select):
        if _is_direct_exists_target(sel):
            continue
        for expr in sel.expressions:
            target = expr.this if isinstance(expr, exp.Alias) else expr
            if isinstance(target, exp.Star):
                return True
            if isinstance(target, exp.Column) and isinstance(target.this, exp.Star):
                return True
    return False


class WildcardDetector(BaseEvaluator):
    """Streaming Measurement flagging projection wildcards (`1`/`0`/`None`)."""

    def __init__(
        self,
        dialect: str = "postgres",
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None:
        super().__init__(
            dialect=dialect,
            extractor=extractor,
            max_query_chars=max_query_chars,
            max_input_chars=max_input_chars,
            include_per_example=include_per_example,
        )
        self._wildcard_count = 0
        self._parsed_count = 0
        self._parse_failed_count = 0
        self._extraction_failed_count = 0
        self._per_example: list[int | None] = []
        self.reset()

    def reset(self) -> None:
        """Reset streaming counters and optional per-example flags."""
        self._num_examples = 0
        self._wildcard_count = 0
        self._parsed_count = 0
        self._parse_failed_count = 0
        self._extraction_failed_count = 0
        self._per_example = []

    def _record_one(self, raw_data: Any) -> None:
        sql_text: str
        if self._compiled_extractor is not None:
            ext = apply_extractor(raw_data, self._compiled_extractor, self.max_input_chars)
            if not ext.ok or ext.sql is None:
                self._num_examples += 1
                self._parse_failed_count += 1
                self._extraction_failed_count += 1
                if self.include_per_example:
                    self._per_example.append(None)
                return
            sql_text = ext.sql
        else:
            if not isinstance(raw_data, str):
                raise TypeError(f"`data` must be a str, got {type(raw_data).__name__}.")
            sql_text = raw_data

        parsed = parse_sql(sql_text, dialect=self.dialect, max_query_chars=self.max_query_chars)
        if not parsed.ok or parsed.ast is None:
            self._num_examples += 1
            self._parse_failed_count += 1
            if self.include_per_example:
                self._per_example.append(None)
            return

        try:
            flagged = 1 if _has_projection_wildcard(parsed.ast) else 0
        except (SqlglotError, RecursionError):
            self._num_examples += 1
            self._parse_failed_count += 1
            if self.include_per_example:
                self._per_example.append(None)
            return

        self._num_examples += 1
        self._parsed_count += 1
        self._wildcard_count += flagged
        if self.include_per_example:
            self._per_example.append(flagged)

    def add(self, data: str) -> None:
        """Add one SQL string or raw completion for wildcard detection."""
        if self._compiled_extractor is None and not isinstance(data, str):
            raise TypeError(f"`data` must be a str, got {type(data).__name__}.")
        self._record_one(data)

    def add_batch(self, data: Sequence[str]) -> None:
        """Validate and scan a batch of SQL strings or raw completions."""
        queries = self._validate_sequence("data", data)
        if self._compiled_extractor is None:
            for idx, query in enumerate(queries):
                if not isinstance(query, str):
                    raise TypeError(f"`data[{idx}]` must be str, got {type(query).__name__}.")
        for query in queries:
            self._record_one(query)

    def compute(self) -> dict[str, Any]:
        """Compute Wildcard Detector summary metrics and reset accumulators."""
        return super().compute()

    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        rate: float | None = (
            self._wildcard_count / self._parsed_count if self._parsed_count > 0 else None
        )
        payload: dict[str, Any] = {
            "wildcard_rate": rate,
            "wildcard_count": self._wildcard_count,
            "parsed_count": self._parsed_count,
            "parse_failed_count": self._parse_failed_count,
            "extraction_failed_count": self._extraction_failed_count,
            "num_examples": self._num_examples,
            "dialect": self.dialect,
        }
        if self.include_per_example:
            payload["per_example"] = list(self._per_example)
        return payload


def compute_wildcard_detector(
    data: Sequence[str],
    *,
    dialect: str = "postgres",
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]:
    """Compute Wildcard Detector measurement over a batch of SQL inputs."""
    measurement = WildcardDetector(
        dialect=dialect,
        extractor=extractor,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        include_per_example=include_per_example,
    )
    measurement.add_batch(data)
    return measurement.compute()
