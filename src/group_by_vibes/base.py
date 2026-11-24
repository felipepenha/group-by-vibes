"""Single-writer streaming evaluator base class and batch validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from .dialects import normalize_dialect, validate_char_limit
from .extract import ExtractorLike, compile_extractor


class BaseEvaluator(ABC):
    """Single-writer streaming evaluator base implementing `evaluate`-style lifecycle."""

    def __init__(
        self,
        *,
        dialect: str = "postgres",
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None:
        self.dialect = normalize_dialect(dialect)
        self.extractor = extractor
        self._compiled_extractor = compile_extractor(extractor)
        self.max_query_chars = validate_char_limit("max_query_chars", max_query_chars)
        self.max_input_chars = validate_char_limit("max_input_chars", max_input_chars)
        self.include_per_example = bool(include_per_example)
        self._num_examples = 0

    @staticmethod
    def _validate_sequence(name: str, values: Any) -> Sequence[Any]:
        if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(values, Sequence):
            raise TypeError(
                f"`{name}` must be a non-str/non-mapping Sequence, got {type(values).__name__}."
            )
        return values

    def compute(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compute accumulated summary metrics and reset state after succeeding."""
        if args:
            raise TypeError(
                "`compute()` takes no positional arguments; use `add()`/`add_batch()`."
            )
        if self._num_examples == 0:
            raise ValueError("Cannot call `compute()` before adding at least one example.")
        result = self._finalize(**kwargs)
        self.reset()
        return result

    @abstractmethod
    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        """Build the JSON-serializable result dict without mutating state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset streaming counters and optional per-example state."""
