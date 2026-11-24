"""Dialect validation and identifier normalization helpers."""

from __future__ import annotations

from sqlglot.dialects.dialect import Dialect, NormalizationStrategy

_DIALECT_ALIASES: dict[str, str] = {
    "postgresql": "postgres",
    "sqlite3": "sqlite",
}


def validate_char_limit(name: str, value: int | None) -> int | None:
    """Validate optional positive integer character limit arguments."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive int or None, got {value!r}.")
    return value


def validate_positive_int(name: str, value: int) -> int:
    """Validate required positive integer configuration parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{name}` must be an int >= 1, got {value!r}.")
    return value


def normalize_dialect(dialect: str = "postgres") -> str:
    """Normalize and validate a `sqlglot` dialect name or alias."""
    if not isinstance(dialect, str):
        raise ValueError(f"`dialect` must be a str, got {type(dialect).__name__}.")
    cleaned = dialect.strip().lower()
    mapped = _DIALECT_ALIASES.get(cleaned, cleaned)
    try:
        Dialect.get_or_raise(mapped)
    except Exception as exc:
        raise ValueError(f"Unsupported sqlglot dialect: {dialect!r}.") from exc
    return mapped


def _identifier_delimiters(dialect: str) -> dict[str, str]:
    """Return mapping of opening -> closing identifier quote delimiters for `dialect`."""
    cls = Dialect.get_or_raise(dialect)
    delimiters: dict[str, str] = {}
    for entry in cls.tokenizer_class.IDENTIFIERS:
        if isinstance(entry, tuple) and len(entry) == 2:
            delimiters[entry[0]] = entry[1]
        elif isinstance(entry, str) and len(entry) == 1:
            delimiters[entry] = entry
    if '"' not in delimiters:
        delimiters.setdefault('"', '"')
    return delimiters


def primary_identifier_delimiter(dialect: str) -> tuple[str, str]:
    """Return primary (open, close) identifier quote delimiters for `dialect`."""
    cls = Dialect.get_or_raise(dialect)
    for entry in cls.tokenizer_class.IDENTIFIERS:
        if isinstance(entry, tuple) and len(entry) == 2:
            return entry[0], entry[1]
        if isinstance(entry, str) and entry:
            return entry, entry
    return '"', '"'


def split_qualified_table_parts(
    raw_name: str,
    dialect: str = "postgres",
) -> tuple[tuple[str, bool] | None, tuple[str, bool] | None, tuple[str, bool]]:
    """Split a 1-3 part table identifier into `(catalog, db, table)` with quoted flags.

    Parses dialect identifier quotes directly without invoking `sqlglot.parse_one`,
    preserving hyphenated unquoted table names such as `"1-29135051-2"`.
    """
    norm_dialect = normalize_dialect(dialect)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"Invalid empty table name: {raw_name!r}.")

    delimiters = _identifier_delimiters(norm_dialect)
    text = raw_name.strip()
    n = len(text)
    idx = 0
    parts: list[tuple[str, bool]] = []

    while True:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            raise ValueError(f"Invalid qualified table name: {raw_name!r}.")

        ch = text[idx]
        if ch in delimiters:
            close_ch = delimiters[ch]
            idx += 1
            buf: list[str] = []
            closed = False
            while idx < n:
                cur = text[idx]
                if cur == close_ch:
                    if idx + 1 < n and text[idx + 1] == close_ch:
                        buf.append(close_ch)
                        idx += 2
                        continue
                    idx += 1
                    closed = True
                    break
                buf.append(cur)
                idx += 1
            if not closed or not buf:
                raise ValueError(f"Invalid quoted segment in table name: {raw_name!r}.")
            parts.append(("".join(buf), True))
            while idx < n and text[idx].isspace():
                idx += 1
        else:
            start = idx
            while idx < n and text[idx] != ".":
                if text[idx] in delimiters:
                    raise ValueError(f"Unexpected quote in unquoted segment: {raw_name!r}.")
                idx += 1
            seg = text[start:idx].strip()
            if not seg:
                raise ValueError(f"Empty segment in table name: {raw_name!r}.")
            parts.append((seg, False))

        if idx >= n:
            break
        if text[idx] != ".":
            raise ValueError(f"Unexpected trailing characters in table name: {raw_name!r}.")
        idx += 1
        if idx >= n:
            raise ValueError(f"Trailing dot in table name: {raw_name!r}.")

    if len(parts) == 1:
        return None, None, parts[0]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Table name has more than 3 parts: {raw_name!r}.")


def split_qualified_table(
    raw_name: str,
    dialect: str = "postgres",
) -> tuple[str | None, str | None, str]:
    """Split a 1-3 part table name into `(catalog, db, table)` strings."""
    catalog, db, table = split_qualified_table_parts(raw_name, dialect=dialect)
    return (
        catalog[0] if catalog is not None else None,
        db[0] if db is not None else None,
        table[0],
    )


def normalize_grounding_identifier(
    name: str,
    quoted: bool,
    dialect: str,
    *,
    is_table: bool = False,
) -> str:
    """Normalize an identifier exclusively for schema-grounding lookups."""
    norm_dialect = normalize_dialect(dialect)
    if norm_dialect == "sqlite":
        return name.lower()
    if quoted:
        return name

    strategy = Dialect.get_or_raise(norm_dialect).NORMALIZATION_STRATEGY
    if strategy in (NormalizationStrategy.LOWERCASE, NormalizationStrategy.CASE_INSENSITIVE):
        return name.lower()
    if strategy in (
        NormalizationStrategy.UPPERCASE,
        NormalizationStrategy.CASE_INSENSITIVE_UPPERCASE,
    ):
        return name.upper()
    if strategy == NormalizationStrategy.CASE_INSENSITIVE_EXCEPT_TABLES:
        return name if is_table else name.lower()
    return name
