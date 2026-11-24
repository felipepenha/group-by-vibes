"""Standalone deterministic Text2SQL evaluation package powered by `sqlglot`."""

from __future__ import annotations

import logging
from typing import Any, Literal, overload

from .canonicalize import canonicalize_sql, sql_tokenizer, try_canonicalize_sql
from .dialects import normalize_dialect, split_qualified_table
from .extract import extract_fenced_or_last_sql
from .sql_bleu import SqlBleu, compute_sql_bleu, sentence_sql_bleu
from .valid_sql_rate import ValidSqlRate, compute_valid_sql_rate
from .wildcard_detector import WildcardDetector, compute_wildcard_detector

__version__ = "0.1.0"

logging.getLogger("group_by_vibes").addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "SqlBleu",
    "ValidSqlRate",
    "WildcardDetector",
    "compute_sql_bleu",
    "sentence_sql_bleu",
    "compute_valid_sql_rate",
    "compute_wildcard_detector",
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


def load(
    name: Literal["sql_bleu", "valid_sql_rate", "wildcard_detector"] | str,
    **kwargs: Any,
) -> SqlBleu | ValidSqlRate | WildcardDetector:
    """Instantiate an evaluator by registered Hugging Face `evaluate`-style name."""
    if name == "sql_bleu":
        return SqlBleu(**kwargs)
    if name == "valid_sql_rate":
        return ValidSqlRate(**kwargs)
    if name == "wildcard_detector":
        return WildcardDetector(**kwargs)
    raise ValueError(
        f"Unknown evaluator {name!r}; expected 'sql_bleu', 'valid_sql_rate', "
        "or 'wildcard_detector'."
    )
