"""Reference-based Exact Matching metric adapted onto `sqlglot` ASTs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .extract import ExtractorLike
from .matching import (
    ALL_HARDNESS_LEVELS,
    ForeignKeysLike,
    PairMatchResult,
    SchemaDict,
    _BaseMatchingEvaluator,
)


class ExactMatching(_BaseMatchingEvaluator):
    """Streaming Exact Matching metric (Yu et al., 2018, adapted via `sqlglot`)."""

    def __init__(
        self,
        dialect: str = "postgres",
        *,
        ignore_values: bool = True,
        ignore_distinct: bool = True,
        include_limit_value: bool = False,
        include_join_conditions: bool = False,
        skip_unparseable_predictions: bool = False,
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None:
        self._exact_counts: dict[str, int] = {}
        self._per_example: list[dict[str, Any] | None] = []
        super().__init__(
            dialect=dialect,
            ignore_values=ignore_values,
            ignore_distinct=ignore_distinct,
            include_limit_value=include_limit_value,
            include_join_conditions=include_join_conditions,
            skip_unparseable_predictions=skip_unparseable_predictions,
            extractor=extractor,
            max_query_chars=max_query_chars,
            max_input_chars=max_input_chars,
            include_per_example=include_per_example,
        )
        self.reset()

    def reset(self) -> None:
        """Reset streaming exact-match counters and optional per-example records."""
        self._reset_common()
        self._exact_counts = {level: 0 for level in ALL_HARDNESS_LEVELS}
        self._per_example = []

    def _record_match_result(self, res: PairMatchResult | None) -> None:
        if res is None:
            if self.include_per_example:
                self._per_example.append(None)
            return

        if res.exact_match:
            self._exact_counts[res.hardness] += 1
            self._exact_counts["all"] += 1

        if self.include_per_example:
            self._per_example.append(
                {
                    "exact_match": res.exact_match,
                    "hardness": res.hardness,
                    "prediction_parsed": res.prediction_parsed,
                    "schema_applied": res.schema_applied,
                    "foreign_keys_applied": res.foreign_keys_applied,
                }
            )

    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        by_hardness: dict[str, dict[str, Any]] = {}
        for level in ALL_HARDNESS_LEVELS:
            count = self._hardness_counts[level]
            em_count = self._exact_counts[level]
            by_hardness[level] = {
                "count": count,
                "exact_match": (em_count / count) if count > 0 else None,
                "exact_match_count": em_count,
            }

        overall_exact: float | None = (
            self._exact_counts["all"] / self._scored_count if self._scored_count > 0 else None
        )
        payload: dict[str, Any] = {
            "exact_match": overall_exact,
            "exact_match_count": self._exact_counts["all"],
            "by_hardness": by_hardness,
            **self._common_metadata_payload(),
        }
        if self.include_per_example:
            payload["per_example"] = [
                None if row is None else dict(row) for row in self._per_example
            ]
        return payload


def compute_exact_matching(
    predictions: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    schemas: Sequence[SchemaDict | None] | None = None,
    foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    *,
    dialect: str = "postgres",
    ignore_values: bool = True,
    ignore_distinct: bool = True,
    include_limit_value: bool = False,
    include_join_conditions: bool = False,
    skip_unparseable_predictions: bool = False,
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]:
    """Compute overall and hardness-stratified Exact Matching accuracy."""
    metric = ExactMatching(
        dialect=dialect,
        ignore_values=ignore_values,
        ignore_distinct=ignore_distinct,
        include_limit_value=include_limit_value,
        include_join_conditions=include_join_conditions,
        skip_unparseable_predictions=skip_unparseable_predictions,
        extractor=extractor,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        include_per_example=include_per_example,
    )
    metric.add_batch(predictions, references, schemas=schemas, foreign_keys=foreign_keys)
    return metric.compute()
