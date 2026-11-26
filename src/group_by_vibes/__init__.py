"""Standalone deterministic Text2SQL evaluation package powered by `sqlglot`."""

from __future__ import annotations

import logging
from typing import Any, Literal, overload

from .canonicalize import canonicalize_sql, sql_tokenizer, try_canonicalize_sql
from .component_matching import ComponentMatching, compute_component_matching
from .dialects import normalize_dialect, split_qualified_table
from .exact_matching import ExactMatching, compute_exact_matching
from .extract import extract_fenced_or_last_sql
from .matching import ForeignKeyPair, ForeignKeysLike
from .sql_bleu import SqlBleu, compute_sql_bleu, sentence_sql_bleu
from .valid_sql_rate import ValidSqlRate, compute_valid_sql_rate
from .wildcard_detector import WildcardDetector, compute_wildcard_detector

__version__ = "0.1.3"

logging.getLogger("group_by_vibes").addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "SqlBleu",
    "ValidSqlRate",
    "WildcardDetector",
    "ComponentMatching",
    "ExactMatching",
    "compute_sql_bleu",
    "sentence_sql_bleu",
    "compute_valid_sql_rate",
    "compute_wildcard_detector",
    "compute_component_matching",
    "compute_exact_matching",
    "ForeignKeyPair",
    "ForeignKeysLike",
    "load",
    "normalize_dialect",
    "split_qualified_table",
    "extract_fenced_or_last_sql",
    "canonicalize_sql",
    "try_canonicalize_sql",
    "sql_tokenizer",
]


@overload
def load(name: Literal["sql_bleu"], **kwargs: Any) -> SqlBleu: ...


@overload
def load(name: Literal["valid_sql_rate"], **kwargs: Any) -> ValidSqlRate: ...


@overload
def load(name: Literal["wildcard_detector"], **kwargs: Any) -> WildcardDetector: ...


@overload
def load(name: Literal["component_matching"], **kwargs: Any) -> ComponentMatching: ...


@overload
def load(name: Literal["exact_matching"], **kwargs: Any) -> ExactMatching: ...


def load(
    name: Literal[
        "sql_bleu",
        "valid_sql_rate",
        "wildcard_detector",
        "component_matching",
        "exact_matching",
    ]
    | str,
    **kwargs: Any,
) -> SqlBleu | ValidSqlRate | WildcardDetector | ComponentMatching | ExactMatching:
    """Instantiate an evaluator by registered Hugging Face `evaluate`-style name."""
    if name == "sql_bleu":
        return SqlBleu(**kwargs)
    if name == "valid_sql_rate":
        return ValidSqlRate(**kwargs)
    if name == "wildcard_detector":
        return WildcardDetector(**kwargs)
    if name == "component_matching":
        return ComponentMatching(**kwargs)
    if name == "exact_matching":
        return ExactMatching(**kwargs)
    raise ValueError(
        f"Unknown evaluator {name!r}; expected 'sql_bleu', 'valid_sql_rate', "
        "'wildcard_detector', 'component_matching', or 'exact_matching'."
    )
