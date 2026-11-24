"""Reference-free Valid SQL Rate measurement with static query grounding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias

from .base import BaseEvaluator
from .extract import ExtractorLike, apply_extractor
from .grounding import _NormalizedSchema, check_schema_grounding, normalize_schema
from .parser import parse_sql

SchemaDict: TypeAlias = Mapping[str, Iterable[str]]

_FAILURE_KEYS: tuple[str, ...] = (
    "extraction_failed",
    "extraction_error",
    "empty_input",
    "multi_statement",
    "command_fallback",
    "syntax_error",
    "unknown_table",
    "unknown_column",
)


class ValidSqlRate(BaseEvaluator):
    """Streaming Measurement combining `sqlglot` AST checks and static query grounding."""

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
        self._syntax_valid_count = 0
        self._schema_evaluated_count = 0
        self._schema_grounded_count = 0
        self._valid_sql_count = 0
        self._failure_counts: dict[str, int] = {}
        self._per_example: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """Reset streaming counters and failure tallies."""
        self._num_examples = 0
        self._syntax_valid_count = 0
        self._schema_evaluated_count = 0
        self._schema_grounded_count = 0
        self._valid_sql_count = 0
        self._failure_counts = {key: 0 for key in _FAILURE_KEYS}
        self._per_example = []

    def _record_one(self, raw_data: Any, norm_schema: _NormalizedSchema | None) -> None:
        sql_text: str
        if self._compiled_extractor is not None:
            ext = apply_extractor(raw_data, self._compiled_extractor, self.max_input_chars)
            if not ext.ok or ext.sql is None:
                reason = ext.failure_reason or "extraction_failed"
                self._num_examples += 1
                self._failure_counts[reason] += 1
                if self.include_per_example:
                    self._per_example.append(
                        {
                            "syntax_valid": False,
                            "schema_valid": None,
                            "failure_reason": reason,
                            "missing_tables": [],
                            "missing_columns": [],
                        }
                    )
                return
            sql_text = ext.sql
        else:
            if not isinstance(raw_data, str):
                raise TypeError(f"`data` must be a str, got {type(raw_data).__name__}.")
            sql_text = raw_data

        parsed = parse_sql(sql_text, dialect=self.dialect, max_query_chars=self.max_query_chars)
        if not parsed.ok or parsed.ast is None:
            reason = parsed.failure_reason or "syntax_error"
            self._num_examples += 1
            self._failure_counts[reason] += 1
            if self.include_per_example:
                self._per_example.append(
                    {
                        "syntax_valid": False,
                        "schema_valid": None,
                        "failure_reason": reason,
                        "missing_tables": [],
                        "missing_columns": [],
                    }
                )
            return

        if norm_schema is None:
            self._num_examples += 1
            self._syntax_valid_count += 1
            self._valid_sql_count += 1
            if self.include_per_example:
                self._per_example.append(
                    {
                        "syntax_valid": True,
                        "schema_valid": None,
                        "failure_reason": None,
                        "missing_tables": [],
                        "missing_columns": [],
                    }
                )
            return

        grounding = check_schema_grounding(parsed.ast, norm_schema, dialect=self.dialect)
        self._num_examples += 1

        if grounding.failure_reason == "syntax_error":
            self._failure_counts["syntax_error"] += 1
            if self.include_per_example:
                self._per_example.append(
                    {
                        "syntax_valid": False,
                        "schema_valid": None,
                        "failure_reason": "syntax_error",
                        "missing_tables": [],
                        "missing_columns": [],
                    }
                )
            return

        self._syntax_valid_count += 1
        if grounding.skipped:
            self._valid_sql_count += 1
            if self.include_per_example:
                self._per_example.append(
                    {
                        "syntax_valid": True,
                        "schema_valid": None,
                        "failure_reason": None,
                        "missing_tables": [],
                        "missing_columns": [],
                    }
                )
            return

        self._schema_evaluated_count += 1
        if grounding.ok:
            self._schema_grounded_count += 1
            self._valid_sql_count += 1
        elif grounding.failure_reason is not None:
            self._failure_counts[grounding.failure_reason] += 1

        if self.include_per_example:
            self._per_example.append(
                {
                    "syntax_valid": True,
                    "schema_valid": grounding.ok,
                    "failure_reason": grounding.failure_reason,
                    "missing_tables": list(grounding.missing_tables),
                    "missing_columns": list(grounding.missing_columns),
                }
            )

    def add(self, data: str, schema: SchemaDict | None = None) -> None:
        """Add one SQL prediction and optional `{table: columns}` schema dict."""
        if self._compiled_extractor is None and not isinstance(data, str):
            raise TypeError(f"`data` must be a str, got {type(data).__name__}.")
        norm_schema = normalize_schema(schema, dialect=self.dialect) if schema is not None else None
        self._record_one(data, norm_schema)

    def add_batch(
        self,
        data: Sequence[str],
        schemas: Sequence[SchemaDict | None] | None = None,
    ) -> None:
        """Validate and score a sequence of queries and optional per-row schemas."""
        queries = self._validate_sequence("data", data)
        if schemas is None:
            norm_schemas: list[_NormalizedSchema | None] = [None] * len(queries)
        else:
            schema_seq = self._validate_sequence("schemas", schemas)
            if len(queries) != len(schema_seq):
                raise ValueError(
                    f"Length mismatch: got {len(queries)} queries and {len(schema_seq)} schemas."
                )
            norm_schemas = [
                normalize_schema(schema, dialect=self.dialect) if schema is not None else None
                for schema in schema_seq
            ]
        if self._compiled_extractor is None:
            for idx, query in enumerate(queries):
                if not isinstance(query, str):
                    raise TypeError(f"`data[{idx}]` must be str, got {type(query).__name__}.")

        for query, norm_schema in zip(queries, norm_schemas, strict=True):
            self._record_one(query, norm_schema)

    def compute(self) -> dict[str, Any]:
        """Compute Valid SQL Rate summary metrics and reset accumulators."""
        return super().compute()

    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        schema_skipped = self._num_examples - self._schema_evaluated_count
        if self._schema_evaluated_count == 0:
            status = "skipped"
            grounded_rate: float | None = None
        elif self._schema_evaluated_count == self._num_examples:
            status = "applied"
            grounded_rate = self._schema_grounded_count / self._schema_evaluated_count
        else:
            status = "partial"
            grounded_rate = self._schema_grounded_count / self._schema_evaluated_count

        payload: dict[str, Any] = {
            "valid_sql_rate": self._valid_sql_count / self._num_examples,
            "syntax_valid_rate": self._syntax_valid_count / self._num_examples,
            "schema_grounded_rate": grounded_rate,
            "schema_grounding_status": status,
            "schema_evaluated_count": self._schema_evaluated_count,
            "schema_grounded_count": self._schema_grounded_count,
            "schema_skipped_count": schema_skipped,
            "syntax_valid_count": self._syntax_valid_count,
            "valid_sql_count": self._valid_sql_count,
            "num_examples": self._num_examples,
            "dialect": self.dialect,
            "failure_counts": dict(self._failure_counts),
        }
        if self.include_per_example:
            payload["per_example"] = [dict(row) for row in self._per_example]
        return payload


def compute_valid_sql_rate(
    data: Sequence[str],
    schemas: Sequence[SchemaDict | None] | None = None,
    *,
    dialect: str = "postgres",
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]:
    """Compute Valid SQL Rate across queries and optional per-query schemas."""
    measurement = ValidSqlRate(
        dialect=dialect,
        extractor=extractor,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        include_per_example=include_per_example,
    )
    measurement.add_batch(data, schemas=schemas)
    return measurement.compute()
