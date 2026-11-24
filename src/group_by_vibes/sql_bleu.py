"""Reference-based SQL-BLEU evaluator and stateless scoring helpers."""

from __future__ import annotations

import collections
import math
from collections.abc import Iterable, Sequence
from typing import Any

from .base import BaseEvaluator
from .canonicalize import canonicalize_ast
from .dialects import validate_positive_int
from .extract import ExtractorLike, apply_extractor
from .parser import parse_sql, tokenize_sql


def _get_ngrams(
    segment: Sequence[str],
    max_order: int,
) -> collections.Counter[tuple[str, ...]]:
    counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    seg_len = len(segment)
    for order in range(1, max_order + 1):
        for idx in range(seg_len - order + 1):
            counts[tuple(segment[idx : idx + order])] += 1
    return counts


def _corpus_bleu_from_counts(
    matches_by_order: Sequence[int],
    possible_matches_by_order: Sequence[int],
    translation_length: int,
    reference_length: int,
    scored_count: int,
    max_order: int,
    smooth: bool,
) -> tuple[float | None, list[float] | None, float | None, float | None]:
    if scored_count == 0 or reference_length == 0:
        return None, None, None, None
    if translation_length == 0:
        return 0.0, [0.0] * max_order, 0.0, 0.0

    precisions = [0.0] * max_order
    for idx in range(max_order):
        if smooth:
            precisions[idx] = (matches_by_order[idx] + 1.0) / (
                possible_matches_by_order[idx] + 1.0
            )
        elif possible_matches_by_order[idx] > 0:
            precisions[idx] = matches_by_order[idx] / possible_matches_by_order[idx]

    if min(precisions) > 0:
        geo_mean = math.exp(sum((1.0 / max_order) * math.log(p) for p in precisions))
    else:
        geo_mean = 0.0

    ratio = translation_length / reference_length
    bp = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / ratio)
    return geo_mean * bp, precisions, bp, ratio


def _per_example_smoothed_bleu(
    pred_tokens: Sequence[str],
    ref_tokens_list: Sequence[Sequence[str]],
    max_order: int,
) -> float | None:
    pred_len = len(pred_tokens)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - pred_len), r_len))
    if closest_ref_len == 0:
        return None
    if pred_len == 0:
        return 0.0

    merged_ref_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    for ref_tokens in ref_tokens_list:
        merged_ref_counts |= _get_ngrams(ref_tokens, max_order)

    pred_counts = _get_ngrams(pred_tokens, max_order)
    overlap = pred_counts & merged_ref_counts
    matches = [0] * max_order
    for ngram, count in overlap.items():
        matches[len(ngram) - 1] += count
    possible = [max(pred_len - order + 1, 0) for order in range(1, max_order + 1)]

    bleu, _, _, _ = _corpus_bleu_from_counts(
        matches,
        possible,
        translation_length=pred_len,
        reference_length=closest_ref_len,
        scored_count=1,
        max_order=max_order,
        smooth=True,
    )
    return bleu


class SqlBleu(BaseEvaluator):
    """Streaming SQL-BLEU metric (Chen et al., 2019, adapted via `sqlglot`)."""

    def __init__(
        self,
        dialect: str = "postgres",
        max_order: int = 4,
        canonicalize: bool = True,
        sort_commutative: bool = True,
        normalize_identifiers: bool = True,
        strip_comments: bool = True,
        include_semicolons: bool = False,
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
        self.max_order = validate_positive_int("max_order", max_order)
        self.canonicalize = bool(canonicalize)
        self.sort_commutative = bool(sort_commutative)
        self.normalize_identifiers = bool(normalize_identifiers)
        self.strip_comments = bool(strip_comments)
        self.include_semicolons = bool(include_semicolons)

        self._matches_by_order: list[int] = []
        self._possible_matches_by_order: list[int] = []
        self._translation_length = 0
        self._reference_length = 0
        self._scored_count = 0
        self._parsed_count = 0
        self._unparseable_count = 0
        self._total_refs_count = 0
        self._parsed_refs_count = 0
        self._per_example_bleu: list[float | None] = []
        self.reset()

    def reset(self) -> None:
        """Reset accumulated corpus BLEU statistics and per-example scores."""
        self._num_examples = 0
        self._matches_by_order = [0] * self.max_order
        self._possible_matches_by_order = [0] * self.max_order
        self._translation_length = 0
        self._reference_length = 0
        self._scored_count = 0
        self._parsed_count = 0
        self._unparseable_count = 0
        self._total_refs_count = 0
        self._parsed_refs_count = 0
        self._per_example_bleu = []

    @staticmethod
    def _coerce_reference_group(reference: Any, index: int | None = None) -> list[str]:
        loc = f" at index {index}" if index is not None else ""
        if isinstance(reference, str):
            return [reference]
        if isinstance(reference, (bytes, bytearray)) or not isinstance(reference, Iterable):
            raise TypeError(
                f"Expected str or Sequence[str] reference{loc}, got {type(reference).__name__}."
            )
        refs = list(reference)
        if not refs:
            raise ValueError(f"Reference group{loc} must not be empty.")
        for item in refs:
            if not isinstance(item, str):
                raise TypeError(
                    f"Reference entries{loc} must be str, got {type(item).__name__}."
                )
        return refs

    def _prepare_tokens(self, raw_sql: str, *, apply_extract: bool) -> tuple[list[str], bool]:
        sql_text = raw_sql
        if apply_extract and self._compiled_extractor is not None:
            ext = apply_extractor(raw_sql, self._compiled_extractor, self.max_input_chars)
            if not ext.ok or ext.sql is None:
                return [], False
            sql_text = ext.sql
        elif not isinstance(sql_text, str):
            return [], False

        if self.canonicalize:
            parsed = parse_sql(sql_text, dialect=self.dialect, max_query_chars=self.max_query_chars)
            if not parsed.ok or parsed.ast is None:
                return [], False
            canon, ok = canonicalize_ast(
                parsed.ast,
                dialect=self.dialect,
                sort_commutative=self.sort_commutative,
                normalize_identifiers=self.normalize_identifiers,
                strip_comments=self.strip_comments,
                include_semicolons=self.include_semicolons,
            )
            if not ok:
                return [], False
            tokens = tokenize_sql(
                canon,
                dialect=self.dialect,
                include_semicolons=self.include_semicolons,
                strip_comments=self.strip_comments,
                max_query_chars=None,
            )
            return tokens, True

        if self.max_query_chars is not None and len(sql_text) > self.max_query_chars:
            return [], False
        if not sql_text.strip():
            return [], False
        tokens = tokenize_sql(
            sql_text,
            dialect=self.dialect,
            include_semicolons=self.include_semicolons,
            strip_comments=self.strip_comments,
            max_query_chars=self.max_query_chars,
        )
        return tokens, True

    def _record_pair(self, prediction: str, ref_group: list[str]) -> None:
        if self._compiled_extractor is None and not isinstance(prediction, str):
            raise TypeError(f"`prediction` must be a str, got {type(prediction).__name__}.")

        pred_tokens, pred_ok = self._prepare_tokens(prediction, apply_extract=True)
        valid_ref_tokens: list[list[str]] = []
        self._total_refs_count += len(ref_group)
        for ref in ref_group:
            ref_tokens, ref_ok = self._prepare_tokens(ref, apply_extract=False)
            if ref_ok:
                self._parsed_refs_count += 1
                valid_ref_tokens.append(ref_tokens)

        self._num_examples += 1
        if pred_ok:
            self._parsed_count += 1
        else:
            self._unparseable_count += 1

        if not pred_ok or not valid_ref_tokens:
            if self.include_per_example:
                self._per_example_bleu.append(None)
            return

        self._scored_count += 1
        pred_len = len(pred_tokens)
        ref_lens = [len(r) for r in valid_ref_tokens]
        closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - pred_len), r_len))
        self._translation_length += pred_len
        self._reference_length += closest_ref_len

        merged_ref_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
        for r_toks in valid_ref_tokens:
            merged_ref_counts |= _get_ngrams(r_toks, self.max_order)

        pred_counts = _get_ngrams(pred_tokens, self.max_order)
        overlap = pred_counts & merged_ref_counts
        for ngram, count in overlap.items():
            self._matches_by_order[len(ngram) - 1] += count
        for order in range(1, self.max_order + 1):
            possible = pred_len - order + 1
            if possible > 0:
                self._possible_matches_by_order[order - 1] += possible

        if self.include_per_example:
            self._per_example_bleu.append(
                _per_example_smoothed_bleu(pred_tokens, valid_ref_tokens, self.max_order)
            )

    def add(self, prediction: str, reference: str | Sequence[str]) -> None:
        """Add one prediction and its reference string or non-empty reference list."""
        ref_group = self._coerce_reference_group(reference)
        if self._compiled_extractor is None and not isinstance(prediction, str):
            raise TypeError(f"`prediction` must be a str, got {type(prediction).__name__}.")
        self._record_pair(prediction, ref_group)

    def add_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str] | Sequence[Sequence[str]],
    ) -> None:
        """Validate and accumulate a batch of predictions and references."""
        preds = self._validate_sequence("predictions", predictions)
        refs = self._validate_sequence("references", references)
        if len(preds) != len(refs):
            raise ValueError(
                f"Length mismatch: got {len(preds)} predictions and {len(refs)} references."
            )
        if self._compiled_extractor is None:
            for idx, pred in enumerate(preds):
                if not isinstance(pred, str):
                    raise TypeError(
                        f"`predictions[{idx}]` must be str, got {type(pred).__name__}."
                    )
        coerced_refs = [
            self._coerce_reference_group(ref, index=idx) for idx, ref in enumerate(refs)
        ]
        for pred, ref_group in zip(preds, coerced_refs, strict=True):
            self._record_pair(pred, ref_group)

    def compute(self, smooth: bool = False) -> dict[str, Any]:
        """Compute corpus SQL-BLEU (optionally add-1 smoothed) and reset accumulators."""
        return super().compute(smooth=smooth)

    def _finalize(self, **kwargs: Any) -> dict[str, Any]:
        smooth_raw = kwargs.pop("smooth", False)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        smooth_flag = bool(smooth_raw)
        bleu, precisions, bp, ratio = _corpus_bleu_from_counts(
            self._matches_by_order,
            self._possible_matches_by_order,
            translation_length=self._translation_length,
            reference_length=self._reference_length,
            scored_count=self._scored_count,
            max_order=self.max_order,
            smooth=smooth_flag,
        )
        payload: dict[str, Any] = {
            "bleu": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": self._translation_length,
            "reference_length": self._reference_length,
            "scored_count": self._scored_count,
            "parsed_count": self._parsed_count,
            "unparseable_count": self._unparseable_count,
            "parse_success_rate": self._parsed_count / self._num_examples,
            "reference_parse_success_rate": (
                self._parsed_refs_count / self._total_refs_count
                if self._total_refs_count > 0
                else 0.0
            ),
            "num_examples": self._num_examples,
            "dialect": self.dialect,
            "max_order": self.max_order,
            "smooth": smooth_flag,
        }
        if self.include_per_example:
            payload["per_example_bleu"] = list(self._per_example_bleu)
        return payload


def compute_sql_bleu(
    predictions: Sequence[str],
    references: Sequence[str] | Sequence[Sequence[str]],
    *,
    dialect: str = "postgres",
    max_order: int = 4,
    smooth: bool = False,
    canonicalize: bool = True,
    sort_commutative: bool = True,
    normalize_identifiers: bool = True,
    strip_comments: bool = True,
    include_semicolons: bool = False,
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
    include_per_example: bool = False,
) -> dict[str, Any]:
    """Compute corpus SQL-BLEU for sequences of predictions and references."""
    metric = SqlBleu(
        dialect=dialect,
        max_order=max_order,
        canonicalize=canonicalize,
        sort_commutative=sort_commutative,
        normalize_identifiers=normalize_identifiers,
        strip_comments=strip_comments,
        include_semicolons=include_semicolons,
        extractor=extractor,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        include_per_example=include_per_example,
    )
    metric.add_batch(predictions, references)
    return metric.compute(smooth=smooth)


def sentence_sql_bleu(
    prediction: str,
    reference: str | Sequence[str],
    *,
    dialect: str = "postgres",
    max_order: int = 4,
    smooth: bool = True,
    canonicalize: bool = True,
    sort_commutative: bool = True,
    normalize_identifiers: bool = True,
    strip_comments: bool = True,
    include_semicolons: bool = False,
    extractor: ExtractorLike = None,
    max_query_chars: int | None = 65_536,
    max_input_chars: int | None = 131_072,
) -> float | None:
    """Compute smoothed single-pair SQL-BLEU (or `None` if unparseable)."""
    res = compute_sql_bleu(
        [prediction],
        [reference],
        dialect=dialect,
        max_order=max_order,
        smooth=smooth,
        canonicalize=canonicalize,
        sort_commutative=sort_commutative,
        normalize_identifiers=normalize_identifiers,
        strip_comments=strip_comments,
        include_semicolons=include_semicolons,
        extractor=extractor,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        include_per_example=False,
    )
    bleu: float | None = res["bleu"]
    return bleu
