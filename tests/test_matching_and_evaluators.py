"""Tests for `matching`, `ComponentMatching`, `ExactMatching`, parity, and regressions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from group_by_vibes import (
    ComponentMatching,
    ExactMatching,
    compute_component_matching,
    compute_exact_matching,
    extract_fenced_or_last_sql,
    load,
)
from group_by_vibes.extract import compile_extractor
from group_by_vibes.matching import (
    COMPONENT_TYPES,
    MatchingConfig,
    coerce_reference_group,
    decompose_query_ast,
    empty_query_decomposition,
    normalize_foreign_keys,
    parse_query_ast,
    select_best_reference,
)
from group_by_vibes.parser import parse_sql

REPO_ROOT = Path(__file__).resolve().parent.parent


def _f1_map(result: dict[str, Any]) -> dict[str, float | None]:
    return {comp: stats["f1"] for comp, stats in result["components"].items()}


def test_all_12_components_and_exact_matching() -> None:
    pred = (
        "WITH c_cte(x) AS (SELECT id FROM t0 WHERE flag = 'A') "
        "SELECT DISTINCT COUNT(t1.id), SUM(t1.amt) "
        "FROM t1 JOIN t2 ON t1.id = t2.id AND (t1.k = 1 OR t2.k = 2) "
        "JOIN c_cte ON t1.id = c_cte.x "
        "WHERE t1.a = 1 AND t2.b IN (3, 2, 1) "
        "GROUP BY t1.dept HAVING COUNT(*) > 5 "
        "ORDER BY t1.dept DESC LIMIT 10 "
        "UNION SELECT 1 FROM t3"
    )
    ref = (
        "WITH d_cte(x) AS (SELECT id FROM t0 WHERE 'Z' = flag) "
        "SELECT SUM(t1.amt), COUNT(t1.id) "
        "FROM d_cte JOIN t1 ON d_cte.x = t1.id "
        "JOIN t2 ON (t2.k = 9 OR t1.k = 8) AND t2.id = t1.id "
        "WHERE t2.b IN (1, 2, 3) AND 99 = t1.a "
        "GROUP BY t1.dept HAVING 2 < COUNT(*) "
        "ORDER BY t1.dept DESC LIMIT 25 "
        "UNION SELECT 999 FROM t3"
    )
    cm = compute_component_matching(
        [pred],
        [ref],
        dialect="postgres",
        include_join_conditions=True,
        include_per_example=True,
    )
    em = compute_exact_matching(
        [pred],
        [ref],
        dialect="postgres",
        include_join_conditions=True,
        include_per_example=True,
    )
    f1s = _f1_map(cm)
    assert set(f1s.keys()) == {
        "select",
        "select(no AGG)",
        "where",
        "where(no OP)",
        "group(no Having)",
        "group",
        "order",
        "and/or",
        "IUEN",
        "keywords",
        "cte",
        "join_on",
    }
    for comp, val in f1s.items():
        assert val == 1.0, f"Component {comp} expected 1.0, got {val}"
    assert em["exact_match"] == 1.0
    assert em["exact_match_count"] == 1
    assert em["per_example"][0]["exact_match"] == 1


def test_spider_subset_parity_and_hardness_levels() -> None:
    schema = {
        "student": ["stuid", "lname", "age", "advisor"],
        "has_pet": ["stuid", "petid"],
        "pets": ["petid", "pettype", "weight"],
    }
    fks = [("student.stuid", "has_pet.stuid"), ("pets.petid", "has_pet.petid")]

    # Easy (count_comp1 <= 1, count_others == 0, count_comp2 == 0)
    easy_p = "SELECT lname FROM student WHERE age > 20"
    easy_r = "SELECT lname FROM student WHERE 10 < age"

    # Medium (2 where conditions + 1 join -> comp1=2, others=1)
    med_p = (
        "SELECT s.stuid FROM student s JOIN has_pet h ON s.stuid = h.stuid "
        "WHERE s.age > 18 AND h.petid = 1"
    )
    med_r = (
        "SELECT h.stuid FROM student s JOIN has_pet h ON s.stuid = h.stuid "
        "WHERE h.petid = 2 AND s.age > 25"
    )

    # Hard (IUEN comp2=1, comp1=1)
    hard_p = "SELECT student.stuid FROM student EXCEPT SELECT stuid FROM has_pet"
    hard_r = "SELECT stuid FROM student EXCEPT SELECT has_pet.stuid FROM has_pet"

    # Extra (nested subquery + multiple conditions)
    extra_p = (
        "SELECT s.lname FROM student s JOIN has_pet h ON s.stuid = h.stuid "
        "WHERE s.age > 10 AND h.petid IN (SELECT p.petid FROM pets p WHERE p.weight > 5) "
        "AND s.advisor = 1"
    )
    extra_r = (
        "SELECT s.lname FROM student s JOIN has_pet h ON s.stuid = h.stuid "
        "WHERE s.advisor = 9 AND h.petid IN (SELECT p.petid FROM pets p WHERE p.weight > 99) "
        "AND 1 < s.age"
    )

    cm_eval = load("component_matching", dialect="sqlite", include_per_example=True)
    em_eval = load("exact_matching", dialect="sqlite", include_per_example=True)
    pairs = [(easy_p, easy_r), (med_p, med_r), (hard_p, hard_r), (extra_p, extra_r)]
    for p, r in pairs:
        cm_eval.add(p, r, schema=schema, foreign_keys=fks)
        em_eval.add(p, r, schema=schema, foreign_keys=fks)

    cm_res = cm_eval.compute()
    em_res = em_eval.compute()

    # After compute(), evaluators must reset
    with pytest.raises(ValueError, match="before adding"):
        cm_eval.compute()
    with pytest.raises(ValueError, match="before adding"):
        em_eval.compute()

    assert em_res["exact_match"] == 1.0
    assert [row["hardness"] for row in em_res["per_example"]] == [
        "easy",
        "medium",
        "hard",
        "extra",
    ]
    assert cm_res["by_hardness"]["easy"]["count"] == 1
    assert cm_res["by_hardness"]["medium"]["count"] == 1
    assert cm_res["by_hardness"]["hard"]["count"] == 1
    assert cm_res["by_hardness"]["extra"]["count"] == 1
    assert cm_res["by_hardness"]["all"]["count"] == 4
    assert cm_res["per_example"][0]["components"]["join_on"] == {
        "precision": 1,
        "recall": 1,
        "f1": 1,
        "label_total": 0,
        "pred_total": 0,
    }


def test_select_no_agg_where_no_op_group_order_and_keywords_parity() -> None:
    # select(no AGG) matches when aggregate wrapper differs
    cm = compute_component_matching(
        ["SELECT MIN(a), MAX(b) FROM t"],
        ["SELECT a, MIN(b) FROM t"],
    )
    assert cm["components"]["select"]["f1"] == 0.0
    assert cm["components"]["select(no AGG)"]["f1"] == 1.0

    # where(no OP) keeps target column on LHS even for scalar/subquery/unary/EXISTS/NOT/IS NULL
    cm_w = compute_component_matching(
        [
            "SELECT a FROM t WHERE x = (SELECT MAX(y) FROM u) AND NOT y LIKE '%k%' "
            "AND z IS NOT NULL AND EXISTS (SELECT 1 FROM u) AND is_active AND NOT (p = 1 AND q = 2)"
        ],
        [
            "SELECT a FROM t WHERE x > (SELECT MAX(y) FROM u) AND NOT y = 'm' "
            "AND z IS NULL AND EXISTS (SELECT 2 FROM u) AND is_active AND NOT (q = 9 AND p = 8)"
        ],
    )
    assert cm_w["components"]["where(no OP)"]["f1"] == 1.0
    assert cm_w["components"]["where"]["f1"] == 0.0
    kw_res = compute_component_matching(
        [
            "SELECT a FROM t WHERE b NOT IN (1, 2) AND c NOT LIKE '%x%' "
            "GROUP BY d HAVING COUNT(*) > 1 ORDER BY d"
        ],
        [
            "SELECT a FROM t WHERE b NOT IN (3, 4) AND c NOT LIKE '%y%' "
            "GROUP BY d HAVING COUNT(*) > 9 ORDER BY d ASC"
        ],
    )
    assert kw_res["components"]["keywords"]["f1"] == 1.0
    assert kw_res["components"]["order"]["f1"] == 1.0
    assert kw_res["components"]["group"]["f1"] == 1.0

    # group(no Having) strips table qualifier while `group` retains it and compares HAVING + AND/OR
    cm_grp = compute_component_matching(
        ["SELECT COUNT(*) FROM t1 JOIN t2 ON t1.id = t2.id GROUP BY t1.dept"],
        ["SELECT COUNT(*) FROM t1 JOIN t2 ON t1.id = t2.id GROUP BY t2.dept"],
    )
    assert cm_grp["components"]["group(no Having)"]["f1"] == 1.0
    assert cm_grp["components"]["group"]["f1"] == 0.0


def test_five_spider_defect_regressions() -> None:
    # 1. Zero-match corpus F1 must be 0.0 (not Spider's f1=1), and None when both counts are 0
    cm_zero = compute_component_matching(
        ["SELECT a FROM t WHERE a = 1"],
        ["SELECT b FROM t WHERE b = 2"],
    )
    assert cm_zero["components"]["select"]["precision"] == 0.0
    assert cm_zero["components"]["select"]["recall"] == 0.0
    assert cm_zero["components"]["select"]["f1"] == 0.0
    assert cm_zero["components"]["IUEN"]["f1"] is None
    assert cm_zero["components"]["and/or"]["f1"] is None
    assert cm_zero["components"]["and/or"]["pred_count"] == 0
    assert cm_zero["components"]["and/or"]["label_count"] == 0

    # 2. and/or denominator must not be swapped + empty set excluded from corpus denominator
    cm_ao = compute_component_matching(
        ["SELECT a FROM t WHERE a = 1", "NOT VALID SQL"],
        ["SELECT a FROM t WHERE a = 1 AND b = 2", "SELECT a FROM t WHERE a = 1 OR b = 2"],
    )
    # 0 pred_total on both rows -> pred_count = 0, label_count = 2 -> rec=0.0, prec=0.0, f1=0.0
    assert cm_ao["components"]["and/or"]["pred_count"] == 0
    assert cm_ao["components"]["and/or"]["label_count"] == 2
    assert cm_ao["components"]["and/or"]["recall"] == 0.0
    assert cm_ao["components"]["and/or"]["f1"] == 0.0

    # Unparseable prediction vs single-predicate gold does NOT inflate and/or
    cm_unp = compute_component_matching(["NOT VALID SQL"], ["SELECT a FROM t WHERE a = 1"])
    assert cm_unp["components"]["and/or"]["pred_count"] == 0
    assert cm_unp["components"]["and/or"]["label_count"] == 0
    assert cm_unp["components"]["and/or"]["f1"] is None

    # 3. Non-mutation of caller containers across repeated evaluations
    schema = {"t": ["a", "b"]}
    fks = [("t.a", "t.a")]
    ev = ComponentMatching(dialect="postgres")
    for _ in range(3):
        ev.add("SELECT a, b FROM t", "SELECT a, b FROM t", schema=schema, foreign_keys=fks)
    assert ev.compute()["components"]["select"]["f1"] == 1.0
    assert schema == {"t": ["a", "b"]}
    assert fks == [("t.a", "t.a")]

    # 4. >= 2 FROM subqueries must not raise TypeError in Python 3 (Spider sorted() crash)
    em_subq = compute_exact_matching(
        ["SELECT s1.a, s2.b FROM (SELECT a FROM t1) s1 JOIN (SELECT b FROM t2) s2 ON s1.a = s2.b"],
        ["SELECT q2.b, q1.a FROM (SELECT b FROM t2) q2 JOIN (SELECT a FROM t1) q1 ON q2.b = q1.a"],
    )
    assert em_subq["exact_match"] == 1.0

    # 5. count_others hardness must not count NOT / AND / OR or subquery aggregates
    em_hard = compute_exact_matching(
        ["SELECT a FROM t WHERE NOT a = 1 HAVING a > 1 AND a < 5"],
        ["SELECT a FROM t WHERE NOT a = 9 HAVING a > 2 AND a < 8"],
        include_per_example=True,
    )
    assert em_hard["per_example"][0]["hardness"] == "medium"

    # Subquery containing >1 aggregates does NOT push an otherwise simple outer query past hard
    em_agg_subq = compute_exact_matching(
        ["SELECT a FROM t WHERE a = (SELECT MIN(x) + MAX(y) FROM u)"],
        ["SELECT a FROM t WHERE a = (SELECT MIN(x) + MAX(y) FROM u)"],
        include_per_example=True,
    )
    assert em_agg_subq["per_example"][0]["hardness"] == "hard"


def test_multidialect_normalization_and_sql_constructs() -> None:
    # SQLite quoted `"..."` vs backticked `` `...` ``
    assert (
        compute_exact_matching(
            ['SELECT `Free Meal (Y/N)` FROM `My Table` WHERE `My Table`.`Id` = 1'],
            ['SELECT "Free Meal (Y/N)" FROM "My Table" WHERE "Id" = 9'],
            schemas=[{"My Table": ["Id", "Free Meal (Y/N)"]}],
            dialect="sqlite",
        )["exact_match"]
        == 1.0
    )

    # Snowflake uppercase unquoted lookup + lowercase canonical AST + quoted case preserved
    assert (
        compute_exact_matching(
            ['SELECT ID, "MixedCase" FROM USERS WHERE "MixedCase" = \'A\''],
            ['SELECT USERS.id, "MixedCase" FROM users WHERE \'B\' = users."MixedCase"'],
            schemas=[{"USERS": ["ID", "MixedCase"]}],
            dialect="snowflake",
        )["exact_match"]
        == 1.0
    )

    # BigQuery backticks & MySQL LIMIT offset syntax
    assert (
        compute_exact_matching(
            ["SELECT a FROM t ORDER BY a LIMIT 5, 10"],
            ["SELECT a FROM t ORDER BY a LIMIT 10 OFFSET 5"],
            dialect="mysql",
            include_limit_value=True,
        )["exact_match"]
        == 1.0
    )
    assert (
        compute_exact_matching(
            ["SELECT Col FROM proj.ds.Tbl", "SELECT `Col` FROM `proj.ds.Tbl`"],
            ["SELECT Tbl.col FROM proj.ds.Tbl", "SELECT `Tbl`.`Col` FROM `proj.ds.Tbl`"],
            schemas=[{"proj.ds.Tbl": ["Col"]}, None],
            dialect="bigquery",
        )["exact_match"]
        == 1.0
    )
    # BigQuery table lookup is case-sensitive (`Tbl` != `tbl`), whereas column `Col` folds
    assert (
        compute_exact_matching(
            ["SELECT Col FROM proj.ds.Tbl"],
            ["SELECT Col FROM proj.ds.tbl"],
            schemas=[{"proj.ds.Tbl": ["Col"]}],
            dialect="bigquery",
        )["exact_match"]
        == 0.0
    )

    # Multi-part table retention and ambiguous multi-schema table protection
    assert (
        compute_exact_matching(
            ["SELECT id FROM db1.users"],
            ["SELECT id FROM db2.users"],
            dialect="postgres",
        )["exact_match"]
        == 0.0
    )
    assert (
        compute_exact_matching(
            ["SELECT u.id FROM db1.users u JOIN db1.orders o ON u.id = o.user_id"],
            ["SELECT o.user_id FROM db1.users u JOIN db1.orders o ON u.id = o.user_id"],
            foreign_keys=[[("users.id", "orders.user_id")]],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )
    assert (
        compute_exact_matching(
            ["SELECT s1.users.id FROM s1.users JOIN s2.users ON s1.users.id = s2.users.id"],
            ["SELECT s2.users.id FROM s1.users JOIN s2.users ON s1.users.id = s2.users.id"],
            dialect="postgres",
        )["exact_match"]
        == 0.0
    )

    # CTEs: 2-wave dependencies, recursive __self__, cycle fallback, col aliases, left-UNION CTE
    assert (
        compute_exact_matching(
            [
                "WITH b AS (SELECT x FROM a), a AS (SELECT id AS x FROM t) SELECT x FROM b",
                (
                    "WITH RECURSIVE r(n) AS "
                    "(SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT n FROM r"
                ),
                (
                    "WITH RECURSIVE p(x) AS (SELECT x FROM q), "
                    "q(x) AS (SELECT x FROM p) SELECT x FROM p"
                ),
                "((WITH c AS (SELECT 1 AS a) SELECT a FROM c) UNION SELECT 2 FROM t) LIMIT 5",
            ],
            [
                "WITH u AS (SELECT id AS x FROM t), v AS (SELECT x FROM u) SELECT x FROM v",
                (
                    "WITH RECURSIVE s(n) AS "
                    "(SELECT 9 UNION ALL SELECT n + 2 FROM s WHERE 10 > n) SELECT n FROM s"
                ),
                (
                    "WITH RECURSIVE a(x) AS (SELECT x FROM b), "
                    "b(x) AS (SELECT x FROM a) SELECT x FROM a"
                ),
                "WITH c AS (SELECT 9 AS a) SELECT a FROM c UNION SELECT 3 FROM t LIMIT 5",
            ],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )

    # Modern BIRD/enterprise expressions: CASE, CAST, COUNT(1), ROUND/COALESCE/SUBSTR, window
    assert (
        compute_exact_matching(
            [
                "SELECT CASE WHEN a = 1 THEN 'x' ELSE 'y' END, CAST(b AS VARCHAR(10)), "
                "ROUND(COALESCE(c, 0), 2), SUBSTR(d, 1, 3), COUNT(1), "
                "SUM(e) OVER (PARTITION BY f ORDER BY g ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING) "
                "FROM t WHERE -5 = a GROUP BY 1 ORDER BY 1"
            ],
            [
                "SELECT CASE WHEN 9 = a THEN 'p' ELSE 'q' END, CAST(t.b AS VARCHAR(10)), "
                "ROUND(COALESCE(c, 99), 8), SUBSTR(d, 4, 9), COUNT(*), "
                "SUM(e) OVER (PARTITION BY f ORDER BY g ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING) "
                "FROM t WHERE t.a = -999 GROUP BY 1 ORDER BY 1"
            ],
            dialect="sqlite",
        )["exact_match"]
        == 1.0
    )

    # SELECT alias in ORDER BY must NOT be mis-qualified under schema=None
    assert (
        compute_exact_matching(
            ["SELECT COUNT(*) AS cnt FROM users AS u ORDER BY cnt"],
            ["SELECT COUNT(*) AS cnt FROM users ORDER BY cnt ASC"],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )

    # Correlated subquery & opaque source & FETCH FIRST & root Subquery hoisting
    assert (
        compute_exact_matching(
            [
                "SELECT u.id FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.uid = u.id)",
                "SELECT x FROM UNNEST(ARRAY[1, 2]) AS arr(x)",
                "(SELECT a FROM t) ORDER BY a DESC OFFSET 2 ROWS FETCH FIRST 3 ROWS ONLY",
            ],
            [
                (
                    "SELECT usr.id FROM users usr "
                    "WHERE EXISTS (SELECT 2 FROM orders ord WHERE usr.id = ord.uid)"
                ),
                "SELECT x FROM UNNEST(ARRAY[1, 2]) AS other_arr(x)",
                "SELECT a FROM t ORDER BY a DESC OFFSET 2 ROWS FETCH FIRST 3 ROWS ONLY",
            ],
            dialect="postgres",
            include_limit_value=True,
        )["exact_match"]
        == 1.0
    )


def test_mode_flags_and_strict_toggles() -> None:
    # ignore_values=False differentiates literals and COUNT(1) vs COUNT(*)
    res_val = compute_exact_matching(
        ["SELECT COUNT(1) FROM t WHERE a = 1 AND b = -2", "SELECT a FROM t WHERE c = 'x'"],
        ["SELECT COUNT(*) FROM t WHERE a = 1 AND b = -2", "SELECT a FROM t WHERE c = 'x'"],
        ignore_values=False,
    )
    assert res_val["exact_match"] == 0.5
    assert res_val["ignore_values"] is False
    assert res_val["ignore_distinct"] is True
    assert res_val["include_limit_value"] is False
    assert res_val["include_join_conditions"] is False
    assert res_val["skip_unparseable_predictions"] is False

    # ignore_distinct=False enforces DISTINCT, DISTINCT ON, COUNT(DISTINCT a, b), UNION ALL
    cm_dist = compute_component_matching(
        [
            "SELECT DISTINCT ON (a) a, b FROM t",
            "SELECT COUNT(DISTINCT a, b) FROM t",
            "SELECT a FROM t1 UNION ALL SELECT b FROM t2",
        ],
        [
            "SELECT a, b FROM t",
            "SELECT COUNT(a, b) FROM t",
            "SELECT a FROM t1 UNION SELECT b FROM t2",
        ],
        dialect="postgres",
        ignore_distinct=False,
    )
    assert cm_dist["ignore_distinct"] is False
    assert cm_dist["components"]["select"]["f1"] == pytest.approx(1.0 / 3.0)
    assert cm_dist["components"]["IUEN"]["f1"] == 0.0

    # COUNT(DISTINCT a) unwraps to COUNT(a) when ignore_distinct=True and keeps DISTINCT when False
    assert (
        compute_exact_matching(
            ["SELECT COUNT(DISTINCT a) FROM t"],
            ["SELECT COUNT(a) FROM t"],
            ignore_distinct=True,
        )["exact_match"]
        == 1.0
    )
    assert (
        compute_exact_matching(
            ["SELECT COUNT(DISTINCT a) FROM t"],
            ["SELECT COUNT(a) FROM t"],
            ignore_distinct=False,
        )["exact_match"]
        == 0.0
    )

    # include_limit_value=True and standalone OFFSET
    assert (
        compute_exact_matching(
            ["SELECT a FROM t LIMIT 5"],
            ["SELECT a FROM t LIMIT 10"],
            include_limit_value=True,
        )["exact_match"]
        == 0.0
    )
    assert (
        compute_component_matching(
            ["SELECT a FROM t OFFSET 10"],
            ["SELECT a FROM t OFFSET 10"],
            include_limit_value=True,
        )["components"]["order"]["f1"]
        == 1.0
    )

    # include_join_conditions=True distinguishes mismatched ON predicates vs USING
    assert (
        compute_component_matching(
            ["SELECT a FROM t1 JOIN t2 ON t1.id = t2.id"],
            ["SELECT a FROM t1 JOIN t2 ON t1.other = t2.other"],
            include_join_conditions=True,
        )["components"]["join_on"]["f1"]
        == 0.0
    )

    # skip_unparseable_predictions=True vs False + DDL/DML/VALUES rejection
    for bad in ("", "SELECT FROM", "VALUES (1, 2)", "CREATE TABLE t (id INT)", "DELETE FROM t"):
        penalized = compute_exact_matching([bad], ["SELECT a FROM t"])
        assert penalized["exact_match"] == 0.0
        assert penalized["scored_count"] == 1
        assert penalized["unparseable_count"] == 1
        assert penalized["parsed_count"] == 0

        skipped = compute_exact_matching(
            [bad],
            ["SELECT a FROM t"],
            skip_unparseable_predictions=True,
            include_per_example=True,
        )
        assert skipped["exact_match"] is None
        assert skipped["scored_count"] == 0
        assert skipped["unparseable_count"] == 1
        assert skipped["per_example"] == [None]


def test_validation_fk_multiref_redaction_and_error_guards() -> None:
    # Transitive union-find + duplicate key tie-breaking + empty Sequences
    assert normalize_foreign_keys([], dialect="postgres") == {}
    assert normalize_foreign_keys((), dialect="postgres") == {}
    kmap = normalize_foreign_keys(
        [("c.k", "b.k"), ("a.k", "b.k"), ("A.k", "b.k")],
        dialect="postgres",
    )
    assert kmap[("c", "k")] == kmap[("a", "k")] == kmap[("b", "k")] == (("a", False), ("k", False))

    for bad_fk in (
        "a.id",
        b"a.id",
        {"a.id": "b.id"},
        ["bad"],
        [("a.id",)],
        [("a.id", "b.id", "c.id")],
        [("id_only", "b.id")],
        [("cat.db.tbl.col", "b.id")],
        [(" ", "b.id")],
        [(123, "b.id")],
    ):
        with pytest.raises((TypeError, ValueError)):
            normalize_foreign_keys(bad_fk)  # type: ignore[arg-type]

    for bad_ref in (123, b"SELECT 1", {"a": 1}, [], ["SELECT 1", 2]):
        with pytest.raises((TypeError, ValueError)):
            coerce_reference_group(bad_ref)

    # Eager batch validation must NOT mutate state on failure
    ev = ComponentMatching(dialect="postgres")
    ev.add("SELECT a FROM t", "SELECT a FROM t")
    with pytest.raises(ValueError, match="Length mismatch"):
        ev.add_batch(["SELECT a FROM t"], ["SELECT a FROM t", "SELECT b FROM t"])
    with pytest.raises(ValueError, match="Length mismatch"):
        ev.add_batch(["SELECT a FROM t"], ["SELECT a FROM t"], schemas=[])
    with pytest.raises(ValueError, match="Length mismatch"):
        ev.add_batch(["SELECT a FROM t"], ["SELECT a FROM t"], foreign_keys=[])
    with pytest.raises(TypeError, match="must be str"):
        ev.add_batch([123], ["SELECT a FROM t"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="must be a str"):
        ev.add(123, "SELECT a FROM t")  # type: ignore[arg-type]
    # Existing valid row must remain intact after rejected bad batch
    assert ev.compute()["num_examples"] == 1

    # Conditional non-str prediction with extractor becomes unparseable (no TypeError)
    ev_ext = ExactMatching(dialect="postgres", extractor=extract_fenced_or_last_sql)
    ev_ext.add(123, "SELECT a FROM t")  # type: ignore[arg-type]
    ext_res = ev_ext.compute()
    assert ext_res["unparseable_count"] == 1
    assert ext_res["exact_match"] == 0.0

    # Multi-reference best selection picks matching hardness + score, and drops invalid refs
    em_multi = compute_exact_matching(
        ["SELECT a FROM t WHERE b = 1", "SELECT a FROM t"],
        [
            ["INVALID REF", "SELECT a FROM t WHERE b = 1 AND c = 2", "SELECT a FROM t WHERE b = 9"],
            ["ALSO INVALID"],
        ],
        include_per_example=True,
    )
    assert em_multi["scored_count"] == 1
    assert em_multi["num_examples"] == 2
    assert em_multi["reference_parse_success_rate"] == 0.5
    assert em_multi["per_example"][0]["exact_match"] == 1
    assert em_multi["per_example"][1] is None

    # Redaction check: raw SQL / literals must not appear in JSON per_example
    secret_sql = "SELECT secret_col FROM secret_tbl WHERE token = 'TOP_SECRET_VALUE_999'"
    cm_priv = compute_component_matching(
        [secret_sql], [secret_sql], include_per_example=True
    )
    dumped = json.dumps(cm_priv)
    assert "TOP_SECRET_VALUE_999" not in dumped
    assert "secret_tbl" not in dumped

    # RecursionError inside parse_sql / decompose_query_ast returns None safely
    with patch("group_by_vibes.matching._decompose_query_ast", side_effect=RecursionError):
        ast_res = parse_sql("SELECT a FROM t", dialect="postgres")
        assert ast_res.ast is not None
        assert (
            decompose_query_ast(
                ast_res.ast,
                dialect="postgres",
                config=MatchingConfig(dialect="postgres"),
                norm_schema=None,
                kmap={},
            )
            is None
        )
    with patch("group_by_vibes.matching._unwrap_query_subquery", side_effect=RecursionError):
        bad_p = parse_query_ast(
            "SELECT a FROM t",
            dialect="postgres",
            max_query_chars=65_536,
            max_input_chars=131_072,
            compiled_extractor=None,
            apply_extract=False,
        )
        assert not bad_p.ok
        assert bad_p.failure_reason == "syntax_error"

    # Reject unused import lint warning on COMPONENT_TYPES check
    assert len(COMPONENT_TYPES) == 12


def test_bird_sample_fixture_end_to_end() -> None:
    sample_path = REPO_ROOT / "examples" / "fixtures" / "bird_sample.jsonl"
    raw_lines = sample_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines if line]
    preds = [str(r["raw_completion"]) for r in rows]
    golds = [str(r["gold_sql"]) for r in rows]
    schemas = [r.get("schema") for r in rows]

    cm = compute_component_matching(
        preds,
        golds,
        schemas=schemas,
        dialect="sqlite",
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )
    em = compute_exact_matching(
        preds,
        golds,
        schemas=schemas,
        dialect="sqlite",
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )
    assert cm["reference_parse_success_rate"] == 1.0
    assert em["reference_parse_success_rate"] == 1.0
    assert em["scored_count"] == 5
    # Row 0 has `'S' = s.rtype` + backticks matching `s.rtype = 'S'` + double quotes
    assert em["per_example"][0]["exact_match"] == 1
    assert em["schema_applied_count"] == 5
    assert em["foreign_keys_applied_count"] == 0
    # Ensure `sys` used cleanly for lint (placeholder until script helper tests arrive)
    assert sys.version_info >= (3, 10)


def test_additional_ast_and_evaluator_edge_cases() -> None:
    # ComponentMatching skipped row with include_per_example=True + unexpected kwargs guard
    cm_skip = ComponentMatching(
        dialect="postgres",
        skip_unparseable_predictions=True,
        include_per_example=True,
        extractor=extract_fenced_or_last_sql,
    )
    cm_skip.add("<think>unclosed", "SELECT a FROM t")
    skip_res = cm_skip.compute()
    assert skip_res["per_example"] == [None]
    assert skip_res["components"]["select"]["f1"] is None
    with pytest.raises(TypeError, match="Unexpected keyword"):
        cm_skip._finalize(extra_arg=1)
    em_skip = ExactMatching(dialect="postgres", include_per_example=True)
    with pytest.raises(TypeError, match="Unexpected keyword"):
        em_skip._finalize(extra_arg=1)

    # Scalar subqueries in SELECT, ORDER BY, NULLS FIRST, and merged nested WITH clauses
    assert (
        compute_exact_matching(
            [
                "WITH outer_c AS (SELECT id FROM u) "
                "((WITH inner_c AS (SELECT id FROM v) "
                "SELECT a, (SELECT MAX(x) FROM outer_c) FROM inner_c)) "
                "ORDER BY a NULLS FIRST, (SELECT MIN(z) FROM w) LIMIT 2",
                "WITH c1 AS (SELECT id FROM u) SELECT a FROM t UNION ALL SELECT b FROM c1",
            ],
            [
                "WITH c1 AS (SELECT id FROM u), c2 AS (SELECT id FROM v) "
                "SELECT a, (SELECT MAX(x) FROM c1) FROM c2 "
                "ORDER BY a NULLS FIRST, (SELECT MIN(z) FROM w) LIMIT 5",
                "WITH c2 AS (SELECT id FROM u) SELECT a FROM t UNION ALL SELECT b FROM c2",
            ],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )

    # 3-part catalog.db.table resolution + 2-part lookup + correlated outer qual + HAVING OR
    assert (
        compute_exact_matching(
            [
                "SELECT cat.db.tbl.id, db.tbl.val FROM cat.db.tbl "
                "WHERE b = a AND EXISTS (SELECT 1 FROM o WHERE o.id = db.tbl.id) "
                "AND NOT (x = 1 OR y = 2) "
                "GROUP BY tbl.g + 1 HAVING COUNT(*) > 1 OR SUM(tbl.val) > 5"
            ],
            [
                "SELECT db.tbl.id, tbl.val FROM cat.db.tbl "
                "WHERE a = b AND EXISTS (SELECT 1 FROM o WHERE cat.db.tbl.id = o.id) "
                "AND NOT (y = 9 OR x = 8) "
                "GROUP BY tbl.g + 9 HAVING SUM(tbl.val) > 8 OR COUNT(*) > 2"
            ],
            schemas=[{"cat.db.tbl": ["id", "val", "g"], "o": ["id"]}],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )

    # DISTINCT ON with non-ignored distinct + JOIN USING + CTE table in HAVING + non-query subquery
    cm_dist_on = compute_component_matching(
        ["SELECT DISTINCT ON (a) a, b FROM t JOIN u ON t.id = u.id"],
        ["SELECT DISTINCT ON (b) a, b FROM t JOIN u ON u.id = t.id"],
        dialect="postgres",
        ignore_distinct=False,
    )
    assert cm_dist_on["components"]["select"]["f1"] == 0.0

    # VALUES inside subclauses (UNION / FROM / IN / EXISTS) decompose cleanly without crashing
    values_in_sub = compute_exact_matching(
        [
            "SELECT a FROM t UNION VALUES (1)",
            "SELECT x FROM (VALUES (1)) AS v(x)",
            "SELECT a FROM (WITH c AS (SELECT 1) SELECT * FROM c) s WHERE s.x = 1 UNION VALUES (1)",
            "SELECT a FROM t WHERE x IN (VALUES (1))",
            "SELECT a FROM t WHERE EXISTS (VALUES (1))",
        ],
        ["SELECT a FROM t"] * 5,
        dialect="postgres",
    )
    assert values_in_sub["parsed_count"] == 5
    assert values_in_sub["exact_match"] == 0.0

    # parse_query_ast edge cases & select_best_reference empty guard
    assert not parse_query_ast(
        123,
        dialect="postgres",
        max_query_chars=65_536,
        max_input_chars=131_072,
        compiled_extractor=None,
        apply_extract=False,
    ).ok
    bad_ext = parse_query_ast(
        "SELECT 1",
        dialect="postgres",
        max_query_chars=65_536,
        max_input_chars=131_072,
        compiled_extractor=compile_extractor(lambda _: None),
        apply_extract=True,
    )
    assert not bad_ext.ok and bad_ext.failure_reason == "extraction_failed"
    with pytest.raises(ValueError, match="at least one"):
        select_best_reference(
            empty_query_decomposition(),
            [],
            MatchingConfig(dialect="postgres"),
            prediction_parsed=True,
            schema_applied=False,
            foreign_keys_applied=False,
        )

    # Opaque FROM expression without alias and custom table function call
    assert (
        compute_exact_matching(
            ["SELECT x FROM UNNEST(ARRAY[1, 2])"],
            ["SELECT x FROM UNNEST(ARRAY[1, 2])"],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )

    # Multi-item IN vs subquery IN + WHERE x = (SELECT ...) + subq in SELECT args
    assert (
        compute_exact_matching(
            ["SELECT a, (SELECT MAX(y) FROM u) FROM t WHERE x IN (SELECT k FROM v WHERE k = 1)"],
            ["SELECT (SELECT MAX(y) FROM u), a FROM t WHERE x IN (SELECT k FROM v WHERE k = 9)"],
            dialect="postgres",
        )["exact_match"]
        == 1.0
    )


def _load_script_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_bird_eval_ddl_to_foreign_keys_and_sync_examples_rendering() -> None:
    bird_eval = _load_script_module("test_bird_eval_mod", REPO_ROOT / "examples" / "bird_eval.py")
    ddl = """
    CREATE TABLE "Parent Tbl" (
        id INT PRIMARY KEY,
        "Space Col" INT,
        "a.b" TEXT,
        "q""c" INT
    );
    CREATE TABLE child (
        id INT,
        pid INT REFERENCES "Parent Tbl"(id),
        bare_pid INT REFERENCES "Parent Tbl",
        "Sp Ref" INT,
        col_dot TEXT,
        col_q INT,
        CONSTRAINT fk_comp FOREIGN KEY ("Sp Ref", col_dot)
            REFERENCES "Parent Tbl"("Space Col", "a.b"),
        FOREIGN KEY (col_q) REFERENCES "Parent Tbl"("q""c"),
        FOREIGN KEY (id) REFERENCES "Parent Tbl"
    );
    """
    fks = bird_eval._ddl_to_foreign_keys(ddl, "sqlite")
    assert fks == [
        ["child.\"Sp Ref\"", "\"Parent Tbl\".\"Space Col\""],
        ["child.col_dot", "\"Parent Tbl\".\"a.b\""],
        ["child.col_q", "\"Parent Tbl\".\"q\"\"c\""],
        ["child.pid", "\"Parent Tbl\".id"],
    ]
    # Every extracted pair must be accepted cleanly by normalize_foreign_keys
    kmap = normalize_foreign_keys([(a, b) for a, b in fks], dialect="sqlite")
    assert ("child", "pid") in kmap

    # Also test MySQL backtick escaping
    mysql_ddl = """
    CREATE TABLE p (`a``b` INT PRIMARY KEY);
    CREATE TABLE c (x INT, FOREIGN KEY (x) REFERENCES p(`a``b`));
    """
    mysql_fks = bird_eval._ddl_to_foreign_keys(mysql_ddl, "mysql")
    assert mysql_fks == [["c.x", "p.`a``b`"]]
    assert ("c", "x") in normalize_foreign_keys([(a, b) for a, b in mysql_fks], dialect="mysql")

    # sync_examples rendering tests (2-run with None/non-None & 3-run)
    sync_mod = _load_script_module(
        "test_sync_examples_mod",
        REPO_ROOT / ".agents" / "skills" / "sync-examples" / "scripts" / "sync_examples.py",
    )

    def make_run(
        stem: str,
        model: str,
        *,
        em: float | None,
        em_cnt: int,
        scored: int,
        diff_em: float | None,
        select_f1: float | None,
        order_f1: float | None,
    ) -> Any:
        comps = {c: {"f1": None} for c in COMPONENT_TYPES}
        comps["select"] = {"f1": select_f1}
        comps["order"] = {"f1": order_f1}
        return sync_mod.ScoredRun(
            stem=stem,
            jsonl_path=Path(f"examples/logs/{stem}.jsonl"),
            score_path=Path(f"examples/logs/{stem}_scores.json"),
            model=model,
            dialect="sqlite",
            num_rows=1,
            rows=[
                {
                    "question_id": 1,
                    "db_id": "d1",
                    "difficulty": "simple",
                    "gold_sql": "SELECT 1",
                    "raw_completion": "SELECT 1",
                }
            ],
            scores={
                "metadata": {
                    "truncated_count": 0,
                    "group_by_vibes_version": "0.0.1",
                    "sqlglot_version": "30.33.0",
                    "extractor": "extract_fenced_or_last_sql",
                },
                "sql_bleu": {
                    "bleu": 0.5,
                    "parsed_count": 1,
                    "num_examples": 1,
                    "parse_success_rate": 1.0,
                    "precisions": [0.5, 0.5, 0.5, 0.5],
                    "translation_length": 10,
                    "reference_length": 10,
                    "length_ratio": 1.0,
                    "brevity_penalty": 1.0,
                    "per_example_bleu": [0.5],
                },
                "valid_sql_rate": {
                    "valid_sql_rate": 1.0,
                    "valid_sql_count": 1,
                    "num_examples": 1,
                    "per_example": [{"syntax_valid": True, "schema_valid": True}],
                },
                "wildcard_detector": {"wildcard_rate": 0.0, "wildcard_count": 0, "parsed_count": 1},
                "component_matching": {"components": comps},
                "exact_matching": {
                    "exact_match": em,
                    "exact_match_count": em_cnt,
                    "scored_count": scored,
                },
                "by_difficulty": {
                    "simple": {
                        "num_rows": 1,
                        "bleu": 0.5,
                        "valid_sql_rate": 1.0,
                        "exact_match": diff_em,
                        "exact_match_count": 1 if diff_em else 0,
                    }
                },
            },
        )

    r_none = make_run(
        "r0",
        "org/Qwen-3B",
        em=None,
        em_cnt=0,
        scored=0,
        diff_em=None,
        select_f1=0.25,
        order_f1=None,
    )
    r_ok = make_run(
        "r1",
        "org/FINER-SQL-3B",
        em=0.5,
        em_cnt=1,
        scored=2,
        diff_em=1.0,
        select_f1=0.75,
        order_f1=0.5,
    )
    r_third = make_run(
        "r2",
        "org/XiYanSQL-3B",
        em=0.25,
        em_cnt=1,
        scored=4,
        diff_em=0.0,
        select_f1=0.5,
        order_f1=0.25,
    )

    head_2 = "\n".join(sync_mod._render_headline_section([r_none, r_ok], "FINER - Qwen"))
    assert "| **Exact Match Accuracy** | `N/A` (`0/0`) | `50.0%` (`1/2`) |  |" in head_2

    comp_2 = "\n".join(sync_mod._render_component_matching_section([r_none, r_ok], "FINER - Qwen"))
    assert "### Component Matching F1" in comp_2
    assert "| `select` | `0.250` | `0.750` | `+0.500` |" in comp_2
    assert "| `order` | `N/A` | `0.500` |  |" in comp_2
    assert "`where`" not in comp_2

    diff_2 = "\n".join(sync_mod._render_difficulty_section([r_none, r_ok], ["Qwen", "FINER-SQL"]))
    expected_diff_row = (
        "| `simple` | 1 | `0.5000` | `0.5000` | `0.0000` | "
        "`100.0%` (`1`) | `100.0%` (`1`) | `0.0 pp` | `N/A` (`0`) | `100.0%` (`1`) |  |"
    )
    assert expected_diff_row in diff_2

    head_3 = "\n".join(sync_mod._render_headline_section([r_ok, r_third, r_none], None))
    assert (
        "| **Exact Match Accuracy** | `50.0%` (`1/2`) | `25.0%` (`1/4`) | `N/A` (`0/0`) |"
        in head_3
    )

    comp_3 = "\n".join(sync_mod._render_component_matching_section([r_ok, r_third, r_none], None))
    assert "| `select` | `0.750` | `0.500` | `0.250` |" in comp_3


