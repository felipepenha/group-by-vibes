"""Tests for canonicalization fixes, tokenizer contracts, and `SqlBleu`."""

from __future__ import annotations

import math

import pytest
from sqlglot.errors import SqlglotError

from group_by_vibes import (
    SqlBleu,
    canonicalize_sql,
    compute_sql_bleu,
    extract_fenced_or_last_sql,
    sentence_sql_bleu,
    sql_tokenizer,
    try_canonicalize_sql,
)


def test_legacy_golden_snapshots_without_importing_prototype() -> None:
    # Exact canonical strings & corpus BLEU values captured from src/sql_bleu.py
    # on unaffected queries prior to prototype retirement.
    assert (
        canonicalize_sql("SELECT id FROM orders WHERE total > 100;", dialect="postgres")
        == "SELECT id FROM orders WHERE total > 100"
    )
    assert (
        canonicalize_sql("SELECT id FROM t WHERE k IN (30, 10, 20)", dialect="sqlite")
        == "SELECT id FROM t WHERE k IN (10, 20, 30)"
    )
    assert (
        canonicalize_sql("SELECT a /* c */ FROM t -- e", dialect="postgres") == "SELECT a FROM t"
    )

    preds = [
        "select a, b from users where age > 20",
        "SELECT id FROM orders WHERE total > 100;",
        "SELECT x, y FROM items WHERE id IN (3, 1, 2)",
    ]
    refs = [
        ["SELECT a, b FROM users WHERE age > 20"],
        ["SELECT id, amount FROM orders WHERE total > 50"],
        ["SELECT x, y FROM items WHERE id IN (1, 2, 3)", "SELECT x FROM items"],
    ]
    un_res = compute_sql_bleu(
        preds, refs, dialect="postgres", smooth=False, include_per_example=True
    )
    assert math.isclose(un_res["bleu"], 0.8665372965209636, abs_tol=1e-12)
    assert un_res["translation_length"] == 34
    assert un_res["reference_length"] == 36
    assert un_res["scored_count"] == 3
    assert un_res["parsed_count"] == 3
    assert un_res["unparseable_count"] == 0
    assert un_res["parse_success_rate"] == 1.0
    assert un_res["reference_parse_success_rate"] == 1.0
    for got, exp_p in zip(
        un_res["precisions"],
        [0.9705882352941176, 0.9354838709677419, 0.8928571428571429, 0.88],
        strict=True,
    ):
        assert math.isclose(got, exp_p, abs_tol=1e-12)
    assert math.isclose(un_res["brevity_penalty"], 0.9428731438548749, abs_tol=1e-12)
    assert math.isclose(un_res["length_ratio"], 0.9444444444444444, abs_tol=1e-12)
    # Note: per_example_bleu always uses add-1 smoothing matching sentence_sql_bleu
    assert math.isclose(un_res["per_example_bleu"][1], 0.5145013914965352, abs_tol=1e-12)

    sm_res = compute_sql_bleu(
        preds, refs, dialect="postgres", smooth=True, include_per_example=True
    )
    assert math.isclose(sm_res["bleu"], 0.8692231063994053, abs_tol=1e-12)
    sent = sentence_sql_bleu(preds[1], refs[1], dialect="postgres", smooth=True)
    assert sent is not None and math.isclose(sent, 0.5145013914965352, abs_tol=1e-12)


def test_postorder_detached_boolean_and_precedence_fixes() -> None:
    q1 = "SELECT a, b FROM t WHERE k IN (3, 1, 2) AND y != 5"
    q2 = "SELECT a, b FROM t WHERE 5 != y AND k IN (1, 2, 3)"
    assert canonicalize_sql(q1) == canonicalize_sql(q2)
    assert canonicalize_sql(q1) == "SELECT a, b FROM t WHERE 5 <> y AND k IN (1, 2, 3)"

    p1 = "SELECT id FROM t WHERE (x = 1 OR y = 2) AND z = 3"
    p2 = "SELECT id FROM t WHERE z = 3 AND (y = 2 OR x = 1)"
    assert canonicalize_sql(p1) == canonicalize_sql(p2)
    assert canonicalize_sql(p1) != canonicalize_sql(
        "SELECT id FROM t WHERE x = 1 OR y = 2 AND z = 3"
    )

    # Subquery IN is preserved in place, while subquery-free identifier/negative IN sorts
    assert canonicalize_sql("SELECT id FROM t WHERE a IN (-1, z, b)") == (
        "SELECT id FROM t WHERE a IN (-1, b, z)"
    )
    assert "SELECT u FROM users" in canonicalize_sql(
        "SELECT id FROM t WHERE a IN (SELECT u FROM users)"
    )

    # Bare expression root rewrites cleanly
    bare = canonicalize_sql("b = 2 AND a = 1")
    assert bare == "1 = a AND 2 = b"


def test_deterministic_lowercase_canonicalization_across_dialects() -> None:
    for dialect in ("postgres", "sqlite", "snowflake", "oracle"):
        c, ok = try_canonicalize_sql(
            'SELECT MyCol, "MixedCase" FROM MyTable WHERE Id = 1', dialect=dialect
        )
        assert ok
        assert "mycol" in c and "mytable" in c and '"MixedCase"' in c

    mysql_c, mysql_ok = try_canonicalize_sql(
        "SELECT MyCol, `MixedCase` FROM MyTable WHERE Id = 1", dialect="mysql"
    )
    assert mysql_ok
    assert "mycol" in mysql_c and "mytable" in mysql_c and "`MixedCase`" in mysql_c


def test_symmetric_unparseable_skipping_and_edge_guards() -> None:
    metric = SqlBleu(
        dialect="sqlite", extractor=extract_fenced_or_last_sql, include_per_example=True
    )
    metric.add("not sql at all", "SELECT id FROM t")
    metric.add("```sql\nSELECT id FROM t\n```", ["not_sql(", "SELECT id FROM t"])
    metric.add("```sql\nSELECT id FROM t\n```", "SELECT FROM")
    metric.add("```sql\nSELECT id FROM t\n```", "SELECT id FROM t")

    res = metric.compute()
    assert res["num_examples"] == 4
    assert res["parsed_count"] == 3
    assert res["unparseable_count"] == 1
    assert res["scored_count"] == 2
    assert math.isclose(res["parse_success_rate"], 0.75, abs_tol=1e-12)
    assert math.isclose(res["reference_parse_success_rate"], 3 / 5, abs_tol=1e-12)
    assert res["per_example_bleu"] == [None, 1.0, None, 1.0]
    assert math.isclose(res["bleu"], 1.0, abs_tol=1e-12)

    # Second compute without new data raises because compute() reset state
    with pytest.raises(ValueError):
        metric.compute()

    # Zero valid scored pairs -> None score metrics
    none_res = compute_sql_bleu(["SELECT FROM"], ["SELECT 1"])
    assert none_res["scored_count"] == 0
    assert none_res["bleu"] is None
    assert none_res["precisions"] is None
    assert none_res["brevity_penalty"] is None
    assert none_res["length_ratio"] is None

    # Comments-only prediction under canonicalize=False -> translation_length == 0 -> bleu == 0.0
    zero_len = compute_sql_bleu(
        ["-- just comment"],
        ["SELECT 1"],
        canonicalize=False,
        strip_comments=True,
    )
    assert zero_len["scored_count"] == 1
    assert zero_len["translation_length"] == 0
    assert zero_len["bleu"] == 0.0


def test_canonicalize_false_semicolons_comments_and_post_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = compute_sql_bleu(
        ["SELECT 1 FROM t; -- c"],
        ["SELECT 1 FROM t; -- c"],
        canonicalize=False,
        strip_comments=False,
        include_semicolons=True,
    )
    assert raw["scored_count"] == 1 and math.isclose(raw["bleu"], 1.0, abs_tol=1e-12)
    assert sql_tokenizer("SELECT 1;", include_semicolons=True)[-1] == ";"
    assert canonicalize_sql("SELECT 1", include_semicolons=True) == "SELECT 1;"

    def fail_rewrite(*_args: object, **_kwargs: object) -> object:
        raise SqlglotError("boom rewrite")

    monkeypatch.setattr("group_by_vibes.canonicalize._rewrite_commutative", fail_rewrite)
    assert try_canonicalize_sql("SELECT a FROM t WHERE x = 1") == ("", False)
    res = compute_sql_bleu(["SELECT a FROM t WHERE x = 1"], ["SELECT 1"])
    assert res["unparseable_count"] == 1 and res["scored_count"] == 0


def test_sql_bleu_validation_errors() -> None:
    with pytest.raises(ValueError):
        SqlBleu(max_order=0)
    m = SqlBleu()
    with pytest.raises(ValueError):
        m.add("SELECT 1", [])
    with pytest.raises(TypeError):
        m.add("SELECT 1", 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        m.add_batch(["SELECT 1"], ["SELECT 1", "SELECT 2"])
    with pytest.raises(TypeError):
        m.add_batch("SELECT 1", ["SELECT 1"])  # type: ignore[arg-type]
