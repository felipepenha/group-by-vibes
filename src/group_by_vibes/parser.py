"""Isolated `sqlglot` parse and tokenization boundary."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import TokenType, exp
from sqlglot.errors import SqlglotError

from .dialects import normalize_dialect, primary_identifier_delimiter, validate_char_limit

logger = logging.getLogger("group_by_vibes")

_BRACKET_DIALECTS = frozenset({"sqlite", "tsql", "bigquery"})
_COMMENT_TOKEN_TYPE = getattr(TokenType, "COMMENT", None)
_FALLBACK_WORD_RE_DEFAULT = re.compile(
    r"--[^\n]*|/\*[\s\S]*?\*/|'[^']*(?:''[^']*)*'|\"[^\"]*(?:\"\"[^\"]*)*\""
    r"|`[^`]*`|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]"
)
_FALLBACK_WORD_RE_BRACKET = re.compile(
    r"--[^\n]*|/\*[\s\S]*?\*/|'[^']*(?:''[^']*)*'|\"[^\"]*(?:\"\"[^\"]*)*\""
    r"|`[^`]*`|\[[^\]\n]*\]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]"
)


@dataclass(frozen=True)
class ParsedSql:
    """Result of parsing a single SQL query under a validated dialect."""

    ast: exp.Expr | None
    ok: bool
    failure_reason: str | None


def parse_sql(
    sql: str,
    dialect: str = "postgres",
    max_query_chars: int | None = 65_536,
) -> ParsedSql:
    """Parse one SQL statement via `sqlglot`, rejecting empty, multi-statement, or commands."""
    norm_dialect = normalize_dialect(dialect)
    limit = validate_char_limit("max_query_chars", max_query_chars)
    if not isinstance(sql, str):
        raise TypeError(f"`sql` must be a str, got {type(sql).__name__}.")
    if limit is not None and len(sql) > limit:
        logger.debug("Query rejected pre-parse: length=%d exceeds limit=%d", len(sql), limit)
        return ParsedSql(ast=None, ok=False, failure_reason="syntax_error")
    if not sql.strip():
        return ParsedSql(ast=None, ok=False, failure_reason="empty_input")

    try:
        raw_stmts = sqlglot.parse(sql, read=norm_dialect)
    except (SqlglotError, RecursionError) as exc:
        logger.debug(
            "Parse failed (dialect=%r, length=%d, error=%s)",
            norm_dialect,
            len(sql),
            type(exc).__name__,
        )
        return ParsedSql(ast=None, ok=False, failure_reason="syntax_error")

    stmts = [stmt for stmt in raw_stmts if stmt is not None]
    if not stmts:
        return ParsedSql(ast=None, ok=False, failure_reason="empty_input")
    if len(stmts) > 1:
        return ParsedSql(ast=None, ok=False, failure_reason="multi_statement")

    ast = stmts[0]
    if next(ast.find_all(exp.Command), None) is not None:
        return ParsedSql(ast=None, ok=False, failure_reason="command_fallback")
    return ParsedSql(ast=ast, ok=True, failure_reason=None)


def _format_quoted_identifier(text: str, dialect: str) -> str:
    open_q, close_q = primary_identifier_delimiter(dialect)
    escaped = text.replace(close_q, close_q * 2)
    return f"{open_q}{escaped}{close_q}"


def _fallback_regex_tokenize(
    sql: str,
    dialect: str,
    *,
    include_semicolons: bool,
    strip_comments: bool,
) -> list[str]:
    pattern = (
        _FALLBACK_WORD_RE_BRACKET if dialect in _BRACKET_DIALECTS else _FALLBACK_WORD_RE_DEFAULT
    )
    out: list[str] = []
    for tok in pattern.findall(sql):
        if tok == ";" and not include_semicolons:
            continue
        if strip_comments and (tok.startswith("--") or tok.startswith("/*")):
            continue
        out.append(tok)
    return out


def tokenize_sql(
    sql: str,
    dialect: str = "postgres",
    *,
    include_semicolons: bool = False,
    strip_comments: bool = True,
    max_query_chars: int | None = 65_536,
) -> list[str]:
    """Tokenize SQL via `sqlglot.tokenize` with quote preservation and linear regex fallback."""
    norm_dialect = normalize_dialect(dialect)
    limit = validate_char_limit("max_query_chars", max_query_chars)
    if not isinstance(sql, str):
        raise TypeError(f"`sql` must be a str, got {type(sql).__name__}.")
    if limit is not None and len(sql) > limit:
        raise ValueError(f"Query length {len(sql)} exceeds max_query_chars={limit}.")
    if not sql.strip():
        return []

    try:
        tokens = sqlglot.tokenize(sql, read=norm_dialect)
    except (SqlglotError, RecursionError) as exc:
        logger.debug(
            "Tokenize fallback triggered (dialect=%r, length=%d, error=%s)",
            norm_dialect,
            len(sql),
            type(exc).__name__,
        )
        return _fallback_regex_tokenize(
            sql,
            norm_dialect,
            include_semicolons=include_semicolons,
            strip_comments=strip_comments,
        )

    out: list[str] = []
    for tok in tokens:
        if tok.token_type == TokenType.SEMICOLON and not include_semicolons:
            continue
        if _COMMENT_TOKEN_TYPE is not None and tok.token_type == _COMMENT_TOKEN_TYPE:
            if not strip_comments:
                out.append(tok.text)
            continue
        if not strip_comments and tok.comments:
            for comment in tok.comments:
                out.append(f"/*{comment}*/")
        if tok.token_type == TokenType.STRING:
            escaped = tok.text.replace("'", "''")
            out.append(f"'{escaped}'")
        elif tok.token_type == TokenType.IDENTIFIER:
            out.append(_format_quoted_identifier(tok.text, norm_dialect))
        else:
            out.append(tok.text)
    return out
