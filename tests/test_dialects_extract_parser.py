"""Tests for dialects, extraction, parser boundary, canonicalize helpers, and base validation."""

from __future__ import annotations

import logging
import re

import pytest
from sqlglot import exp

from group_by_vibes import (
    canonicalize_sql,
    extract_fenced_or_last_sql,
    normalize_dialect,
    split_qualified_table,
    sql_tokenizer,
    try_canonicalize_sql,
)
from group_by_vibes.dialects import normalize_grounding_identifier, validate_char_limit
from group_by_vibes.extract import apply_extractor, compile_extractor
from group_by_vibes.grounding import check_schema_grounding, normalize_schema
from group_by_vibes.parser import parse_sql


def test_normalize_dialect_aliases_whitespace_and_errors() -> None:
    assert normalize_dialect("  PostgreSQL  ") == "postgres"
    assert normalize_dialect("SQLite3") == "sqlite"
    assert normalize_dialect("") == ""
    assert normalize_dialect("Snowflake") == "snowflake"
    with pytest.raises(ValueError):
        normalize_dialect("not_a_real_dialect_xyz")
    with pytest.raises(ValueError):
        normalize_dialect(123)  # type: ignore[arg-type]


def test_validate_char_limit() -> None:
    assert validate_char_limit("max_query_chars", None) is None
    assert validate_char_limit("max_query_chars", 1024) == 1024
    for bad in (0, -1, True, False, "10"):
        with pytest.raises(ValueError):
            validate_char_limit("max_query_chars", bad)  # type: ignore[arg-type]


def test_split_qualified_table_and_grounding_normalization() -> None:
    assert split_qualified_table("1-29135051-2", "sqlite") == (None, None, "1-29135051-2")
    assert split_qualified_table('"my.db"."Users"', "postgres") == (None, "my.db", "Users")
    assert split_qualified_table("cat.sch.tbl", "postgres") == ("cat", "sch", "tbl")
    assert split_qualified_table("[dbo].[My.Tbl]", "tsql") == (None, "dbo", "My.Tbl")

    for bad in ("", "  ", ".users", "a..b", "a.b.c.d", '"unclosed'):
        with pytest.raises(ValueError):
            split_qualified_table(bad, "postgres")

    assert normalize_grounding_identifier("Users", False, "postgres") == "users"
    assert normalize_grounding_identifier("Users", True, "postgres") == "Users"
    assert normalize_grounding_identifier("Singer(s)", True, "sqlite") == "singer(s)"
    assert normalize_grounding_identifier("users", False, "snowflake") == "USERS"
    assert normalize_grounding_identifier("Users", True, "snowflake") == "Users"


def test_extractor_compilation_limits_and_fenced_sql() -> None:
    with pytest.raises(re.error):
        compile_extractor("[unclosed")
    with pytest.raises(ValueError):
        compile_extractor(r"(a)(b)")
    with pytest.raises(TypeError):
        compile_extractor(42)  # type: ignore[arg-type]

    regex_pat = compile_extractor(r"SQL:\s*(SELECT .*)$")
    res = apply_extractor("Prefix SQL: SELECT 1", regex_pat, max_input_chars=100)
    assert res.ok and res.sql == "SELECT 1"

    oversized = apply_extractor("SQL: SELECT 1" + "x" * 200, regex_pat, max_input_chars=50)
    assert not oversized.ok and oversized.failure_reason == "extraction_failed"

    no_match = apply_extractor("no sql here", regex_pat)
    assert not no_match.ok and no_match.failure_reason == "extraction_failed"

    def bad_callable(_: str) -> str | None:
        raise RuntimeError("boom")

    err_res = apply_extractor("SELECT 1", compile_extractor(bad_callable))
    assert not err_res.ok and err_res.failure_reason == "extraction_error"

    def returns_non_str(_: str) -> str | None:
        return 123  # type: ignore[return-value]

    type_err = apply_extractor("SELECT 1", compile_extractor(returns_non_str))
    assert not type_err.ok and type_err.failure_reason == "extraction_error"

    none_res = apply_extractor("anything", compile_extractor(lambda _: None))
    assert not none_res.ok and none_res.failure_reason == "extraction_failed"

    # KeyboardInterrupt must propagate rather than turn into extraction_error
    def interrupt_fn(_: str) -> str | None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        apply_extractor("SELECT 1", compile_extractor(interrupt_fn))

    # extract_fenced_or_last_sql checks
    assert extract_fenced_or_last_sql("<think>unclosed SELECT 1") is None
    fenced = "<think>draft: SELECT 0</think>\n```sql\nSELECT 1\n```\nDone!"
    assert extract_fenced_or_last_sql(fenced) == "SELECT 1"
    generic = "```python\nx = 1\n```\n```\nWITH a AS (SELECT 1) SELECT * FROM a\n```"
    assert extract_fenced_or_last_sql(generic) == "WITH a AS (SELECT 1) SELECT * FROM a"
    unfenced = "Here is the query:\nSELECT a FROM t WHERE x = 1;\nThis skips prose."
    assert extract_fenced_or_last_sql(unfenced) == "SELECT a FROM t WHERE x = 1"
    para_break = "WITH c AS (SELECT 1)\nSELECT * FROM c\n\nExplanation follows here."
    assert extract_fenced_or_last_sql(para_break) == "WITH c AS (SELECT 1)\nSELECT * FROM c"


def test_parse_sql_guards_and_redacted_logging(caplog: pytest.LogCaptureFixture) -> None:
    assert parse_sql("   \n;  ", "postgres").failure_reason == "empty_input"
    assert parse_sql("-- comment only", "postgres").failure_reason == "empty_input"
    ok_semi = parse_sql("SELECT 1;", "postgres")
    assert ok_semi.ok and isinstance(ok_semi.ast, exp.Select)
    assert parse_sql("SELECT 1; SELECT 2;", "postgres").failure_reason == "multi_statement"
    assert parse_sql("EXPLAIN SELECT 1", "sqlite").failure_reason == "command_fallback"
    assert parse_sql("RESET ALL", "postgres").failure_reason == "command_fallback"

    with caplog.at_level(logging.DEBUG, logger="group_by_vibes"):
        oversized = parse_sql("SELECT 'TOP_SECRET_LITERAL'", "postgres", max_query_chars=10)
        bad_syntax = parse_sql("SELECT 'SECRET_ERR' FROM (", "postgres")
    assert not oversized.ok and oversized.failure_reason == "syntax_error"
    assert not bad_syntax.ok and bad_syntax.failure_reason == "syntax_error"
    assert "TOP_SECRET_LITERAL" not in caplog.text
    assert "SECRET_ERR" not in caplog.text


def test_parse_sql_recursion_error_handled() -> None:
    deep_sql = "SELECT " + "(" * 5000 + "1" + ")" * 5000
    parsed = parse_sql(deep_sql, "postgres", max_query_chars=None)
    assert not parsed.ok and parsed.failure_reason == "syntax_error"


def test_tokenize_sql_quotes_comments_and_brackets() -> None:
    toks = sql_tokenizer("SELECT \"Singer(s)\", 'O''Reilly' FROM \"1-2\";", dialect="sqlite")
    assert toks == ["SELECT", '"Singer(s)"', ",", "'O''Reilly'", "FROM", '"1-2"']

    with_semi = sql_tokenizer("SELECT 1;", dialect="postgres", include_semicolons=True)
    assert with_semi == ["SELECT", "1", ";"]

    arr_toks = sql_tokenizer("SELECT arr[1] FROM t", dialect="postgres")
    assert arr_toks == ["SELECT", "arr", "[", "1", "]", "FROM", "t"]

    # Force regex fallback on arr[1] vs tsql [bracket ident]
    pg_fallback = sql_tokenizer("SELECT arr[1] FROM 'unterminated", dialect="postgres")
    assert pg_fallback[1:5] == ["arr", "[", "1", "]"]
    tsql_fallback = sql_tokenizer("SELECT [col 1] FROM 'unterminated", dialect="tsql")
    assert "[col 1]" in tsql_fallback

    comment_diff_1 = sql_tokenizer("SELECT /* a */ 1", dialect="postgres", strip_comments=False)
    comment_diff_2 = sql_tokenizer("SELECT /* b */ 1", dialect="postgres", strip_comments=False)
    assert comment_diff_1 != comment_diff_2

    assert canonicalize_sql("SELECT a FROM t WHERE y = 2 AND x = 1", dialect="postgres") != ""
    assert try_canonicalize_sql("SELECT (((", dialect="postgres") == ("", False)


def test_grounding_core_smoke() -> None:
    schema = normalize_schema({"users": ["id", "name"]}, dialect="postgres")
    ast = parse_sql("SELECT u.id FROM users u", "postgres").ast
    assert ast is not None
    res = check_schema_grounding(ast, schema, dialect="postgres")
    assert res.ok and not res.skipped


def test_additional_dialect_extract_and_parser_branches() -> None:
    from group_by_vibes.parser import tokenize_sql

    assert split_qualified_table('"a""b"."c"', "postgres") == (None, 'a"b', "c")
    with pytest.raises(ValueError):
        split_qualified_table('"users" extra', "postgres")
    with pytest.raises(ValueError):
        split_qualified_table('us"ers', "postgres")
    with pytest.raises(ValueError):
        split_qualified_table("users.", "postgres")

    assert normalize_grounding_identifier("MyTbl", False, "bigquery", is_table=True) == "MyTbl"
    assert normalize_grounding_identifier("MyCol", False, "bigquery", is_table=False) == "mycol"
    assert normalize_grounding_identifier("MyCol", False, "mysql", is_table=False) == "MyCol"

    pre_pat = compile_extractor(re.compile(r"SELECT\s+.*"))
    assert apply_extractor(123, None).failure_reason == "extraction_error"
    assert apply_extractor("SELECT 1", None).sql == "SELECT 1"
    assert apply_extractor("SELECT 1", pre_pat).ok
    with pytest.raises(ValueError):
        compile_extractor(re.compile(r"(a)(b)"))

    assert extract_fenced_or_last_sql(123) is None  # type: ignore[arg-type]
    assert extract_fenced_or_last_sql("```sql\n   \n```\nplain text") is None
    assert tokenize_sql("   ", "postgres") == []
    assert tokenize_sql("SELECT 1 -- hi\n; FROM 'unterminated", "postgres", include_semicolons=True)
    with pytest.raises(TypeError):
        parse_sql(123, "postgres")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenize_sql(123, "postgres")  # type: ignore[arg-type]
    assert try_canonicalize_sql(123) == ("", False)  # type: ignore[arg-type]
