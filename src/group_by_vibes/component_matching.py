"""Reference-based Component Matching metric adapted onto `sqlglot` ASTs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .extract import ExtractorLike
from .matching import (
    ALL_HARDNESS_LEVELS,
    COMPONENT_TYPES,
    ForeignKeysLike,
    PairMatchResult,
    SchemaDict,
    _BaseMatchingEvaluator,
    finalize_component_bucket,
)


class ComponentMatching(_BaseMatchingEvaluator):
    """Streaming Component Matching metric (Yu et al., 2018, adapted via `sqlglot`)."""

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
        self._acc_sums: dict[str, dict[str, int]] = {}
        self._acc_counts: dict[str, dict[str, int]] = {}
        self._rec_sums: dict[str, dict[str, int]] = {}
        self._rec_counts: dict[str, dict[str, int]] = {}
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
        """Reset streaming component accumulators and optional per-example records."""
        self._reset_common()
        self._acc_sums = {
            level: {comp: 0 for comp in COMPONENT_TYPES} for level in ALL_HARDNESS_LEVELS
        }
        self._acc_counts = {
            level: {comp: 0 for comp in COMPONENT_TYPES} for level in ALL_HARDNESS_LEVELS
        }
        self._rec_sums = {
            level: {comp: 0 for comp in COMPONENT_TYPES} for level in ALL_HARDNESS_LEVELS
        }
        self._rec_counts = {
            level: {comp: 0 for comp in COMPONENT_TYPES} for level in ALL_HARDNESS_LEVELS
        }
        self._per_example = []

    def _record_match_result(self, res: PairMatchResult | None) -> None:
        if res is None:
            if self.include_per_example:
                self._per_example.append(None)
            return

        for level in (res.hardness, "all"):
            acc_sums = self._acc_sums[level]
            acc_counts = self._acc_counts[level]
            rec_sums = self._rec_sums[level]
            rec_counts = self._rec_counts[level]
            for comp, cs in res.components.items():
                if cs.pred_total > 0:
                    acc_sums[comp] += cs.precision
                    acc_counts[comp] += 1
                if cs.label_total > 0:
                    rec_sums[comp] += cs.recall
                    rec_counts[comp] += 1

        if self.include_per_example:
            self._per_example.append(
                {
                    "hardness": res.hardness,
                    "prediction_parsed": res.prediction_parsed,
                    "schema_applied": res.schema_applied,
                    "foreign_keys_applied": res.foreign_keys_applied,
                    "components": {
                        comp: {
                            "precision": cs.precision,
                            "recall": cs.recall,
                            "f1": cs.f1,
                            "label_total": cs.label_total,
                            "pred_total": cs.pred_total,
                        }
                        for comp, cs in res.components.items()
                    },
                }
            )

    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        by_hardness: dict[str, dict[str, Any]] = {}
        for level in ALL_HARDNESS_LEVELS:
            comp_dict = {
                comp: finalize_component_bucket(
                    self._acc_sums[level][comp],
                    self._acc_counts[level][comp],
                    self._rec_sums[level][comp],
                    self._rec_counts[level][comp],
                )
                for comp in COMPONENT_TYPES
            }
            by_hardness[level] = {
                "count": self._hardness_counts[level],
                "components": comp_dict,
            }

        overall_components = {
            comp: dict(stats) for comp, stats in by_hardness["all"]["components"].items()
        }
        payload: dict[str, Any] = {
            "components": overall_components,
            "by_hardness": by_hardness,
            **self._common_metadata_payload(),
        }
        if self.include_per_example:
            payload["per_example"] = [
                None
                if row is None
                else {
                    **row,
                    "components": {k: dict(v) for k, v in row["components"].items()},
                }
                for row in self._per_example
            ]
        return payload


def compute_component_matching(
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
    """Compute corpus and hardness-stratified Component Matching F1 metrics."""
    metric = ComponentMatching(
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
