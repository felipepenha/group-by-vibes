"""Tests for `ValidSqlRate` parsing, schema grounding, partitions, and privacy."""

from __future__ import annotations

import math

import pytest

from group_by_vibes import ValidSqlRate, compute_valid_sql_rate, extract_fenced_or_last_sql


def test_sqlite_rowid_and_recursive_cte_columns() -> None:
    schema = {"users": ["id", "name", "dept"], "events": ["id", "tag"]}
    queries = [
        "SELECT rowid, _rowid_, oid, name FROM users",
        (
            "WITH RECURSIVE s(id, w, rest, st) AS ("
            "  SELECT rowid, name, dept, 1 FROM users "
            "  UNION ALL "
            "  SELECT s.id + 1, s.w, s.rest, s.st + 1 FROM s WHERE s.st < 3"
            ") SELECT s.id, s.w FROM s"
        ),
        "SELECT d.x FROM (SELECT 1, 2) AS d(x, y)",
        "SELECT name AS label FROM users GROUP BY label HAVING COUNT(*) > 1 ORDER BY label",
    ]
    res = compute_valid_sql_rate(queries, [schema] * len(queries), dialect="sqlite")
    assert res["valid_sql_rate"] == 1.0
    assert res["schema_grounding_status"] == "applied"
    assert res["schema_evaluated_count"] == len(queries)
    assert res["schema_skipped_count"] == 0

    # In postgres, rowid is not an implicit table column
    pg_res = compute_valid_sql_rate(["SELECT rowid FROM users"], [schema], dialect="postgres")
    assert pg_res["failure_counts"]["unknown_column"] == 1


def test_subquery_correlation_unions_and_mixed_opaque_sources() -> None:
    schema = {"t": ["id", "t_col", "arr"], "u": ["x"], "t1": ["a"], "t2": ["b"]}
    res = compute_valid_sql_rate(
        [
            "SELECT 1 FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t_col)",
            "SELECT 1 FROM t a WHERE EXISTS (SELECT 1 FROM u WHERE u.x = a.t_col)",
            "SELECT a FROM t1 UNION ALL SELECT b FROM t2",
            "WITH c AS (SELECT * FROM t) SELECT c.anything FROM c",
            "SELECT dyn FROM t, UNNEST(t.arr) AS u",
        ],
        [schema] * 5,
        dialect="postgres",
    )
    assert res["valid_sql_rate"] == 1.0

    # Bad correlated column and missing qualified col inside mixed opaque+physical scope
    fail_res = compute_valid_sql_rate(
        [
            "SELECT 1 FROM t WHERE EXISTS (SELECT z FROM u)",
            "SELECT t.missing FROM t, UNNEST(t.arr) AS u",
            "SELECT a FROM t1 UNION ALL SELECT missing_b FROM t2",
        ],
        [schema] * 3,
        dialect="postgres",
        include_per_example=True,
    )
    assert fail_res["valid_sql_count"] == 0
    assert fail_res["failure_counts"]["unknown_column"] == 3
    assert fail_res["per_example"][0]["missing_columns"] == ["z"]
    assert fail_res["per_example"][1]["missing_columns"] == ["t.missing"]
    assert fail_res["per_example"][2]["missing_columns"] == ["missing_b"]


def test_qualified_catalog_keys_and_dialect_case_rules() -> None:
    schema = {"cat.db.users": ["id"], "public.orders": ["oid"]}
    ok_res = compute_valid_sql_rate(
        [
            "SELECT cat.db.users.id FROM cat.db.users",
            "SELECT users.id FROM cat.db.users",
            "SELECT oid FROM orders",
        ],
        [schema] * 3,
        dialect="postgres",
    )
    assert ok_res["valid_sql_count"] == 3

    # Explicit wrong_db must NOT fall back to users/orders
    bad_res = compute_valid_sql_rate(
        ["SELECT wrong_db.orders.oid FROM wrong_db.orders"],
        [schema],
        dialect="postgres",
        include_per_example=True,
    )
    assert bad_res["failure_counts"]["unknown_table"] == 1
    assert bad_res["per_example"][0]["missing_tables"] == ["wrong_db.orders"]

    # Quoted mixed case on Postgres vs Snowflake vs SQLite
    mix = {'"1-29135051-2"': ["Singer(s)", "Year"]}
    pg = compute_valid_sql_rate(
        [
            'SELECT "Singer(s)" FROM "1-29135051-2"',
            'SELECT Singer FROM "1-29135051-2"',
        ],
        [mix, mix],
        dialect="postgres",
    )
    assert pg["valid_sql_count"] == 1
    assert pg["failure_counts"]["unknown_column"] == 1

    sf = compute_valid_sql_rate(
        ["SELECT id FROM users"],
        [{"USERS": ["ID"]}],
        dialect="snowflake",
    )
    assert sf["valid_sql_rate"] == 1.0


def test_partition_invariants_failure_precedence_and_bad_alias() -> None:
    schema = {"users": ["id", "name"]}
    m = ValidSqlRate(
        dialect="postgres",
        extractor=extract_fenced_or_last_sql,
        max_query_chars=120,
        max_input_chars=250,
        include_per_example=True,
    )
    m.add(
        "SELECT 'SECRET_SQL' FROM bad_tbl, users WHERE bad_alias.x = 1 AND users.bad_col = 2",
        schema,
    )
    m.add("SELECT bad.* FROM users", schema)
    m.add("```sql\nDELETE FROM users WHERE id = 1\n```", schema)  # build_scope is None -> skipped
    m.add("SELECT id FROM users", None)  # schema=None -> skipped
    m.add("SELECT " + "a" * 150, schema)  # max_query_chars -> syntax_error
    m.add("<think>unclosed SELECT 'HIDDEN_LIT'", schema)  # extraction_failed
    m.add("SELECT 1 FROM users WHERE name = 'PRIVATE_VALUE'", {})  # empty catalog -> unknown_table

    res = m.compute()
    assert res["num_examples"] == 7
    assert res["schema_evaluated_count"] + res["schema_skipped_count"] == res["num_examples"]
    assert res["schema_grounding_status"] == "partial"
    assert res["valid_sql_count"] == 2
    assert sum(res["failure_counts"].values()) == res["num_examples"] - res["valid_sql_count"]
    assert set(res["failure_counts"]) == {
        "extraction_failed",
        "extraction_error",
        "empty_input",
        "multi_statement",
        "command_fallback",
        "syntax_error",
        "unknown_table",
        "unknown_column",
    }
    assert math.isclose(res["schema_grounded_rate"], 0.0, abs_tol=1e-12)

    ex0 = res["per_example"][0]
    assert ex0["failure_reason"] == "unknown_table"
    assert ex0["missing_tables"] == ["bad_alias", "bad_tbl"]
    assert ex0["missing_columns"] == ["users.bad_col"]
    assert ex0["schema_valid"] is False
    assert "SECRET_SQL" not in str(res["per_example"])
    assert "PRIVATE_VALUE" not in str(res["per_example"])
    assert res["per_example"][2]["schema_valid"] is None
    assert res["per_example"][4]["syntax_valid"] is False
    assert res["per_example"][4]["failure_reason"] == "syntax_error"


def test_valid_sql_rate_validation_and_callable_errors() -> None:
    with pytest.raises(TypeError):
        compute_valid_sql_rate(["SELECT 1"], schemas=[{"users": "name"}])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_valid_sql_rate(["SELECT 1"], schemas=[{"users": [""]}])
    with pytest.raises(TypeError):
        compute_valid_sql_rate(["SELECT 1"], schemas=[["users"]])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_valid_sql_rate(["SELECT 1", "SELECT 2"], schemas=[None])
    with pytest.raises(TypeError):
        compute_valid_sql_rate("SELECT 1")  # type: ignore[arg-type]

    def exploding_extractor(_: str) -> str | None:
        raise ValueError("extract failure")

    err_res = compute_valid_sql_rate(
        ["raw", "x" * 20],
        extractor=exploding_extractor,
        max_input_chars=10,
    )
    assert err_res["failure_counts"]["extraction_error"] == 1
    assert err_res["failure_counts"]["extraction_failed"] == 1

    # No-schema run -> skipped status
    no_schema = compute_valid_sql_rate(["SELECT 1", "   ", "SELECT 1; SELECT 2;"])
    assert no_schema["schema_grounding_status"] == "skipped"
    assert no_schema["schema_grounded_rate"] is None
    assert no_schema["failure_counts"]["empty_input"] == 1
    assert no_schema["failure_counts"]["multi_statement"] == 1

    # Pure valid batch -> applied
    all_ok = compute_valid_sql_rate(["SELECT id FROM users"], [{"users": ["id"]}])
    assert all_ok["schema_grounding_status"] == "applied"
    assert all_ok["valid_sql_rate"] == 1.0


def test_aliased_cte_join_lookup() -> None:
    query = (
        "WITH high_loans AS (SELECT account_id FROM loan WHERE amount > 10000) "
        "SELECT a.account_id FROM account AS a "
        "JOIN high_loans AS h ON a.account_id = h.account_id"
    )
    schemas = [
        {
            "account": ["account_id", "district_id"],
            "loan": ["loan_id", "account_id", "amount"],
        }
    ]
    result = compute_valid_sql_rate([query], schemas)
    assert result["valid_sql_rate"] == 1.0
    assert result["schema_grounding_status"] == "applied"
