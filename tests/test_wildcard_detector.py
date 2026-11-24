"""Tests for `WildcardDetector` AST projection checks and edge cases."""

from __future__ import annotations

import math

import pytest

from group_by_vibes import (
    WildcardDetector,
    compute_wildcard_detector,
    extract_fenced_or_last_sql,
)


def test_wildcard_projections_vs_safe_stars() -> None:
    flagged_queries = [
        ("SELECT * FROM users", "postgres"),
        ("SELECT u.* FROM users u", "postgres"),
        ("SELECT * EXCEPT (secret) FROM users", "bigquery"),
        ("SELECT * EXCLUDE (secret) FROM users", "snowflake"),
        ("WITH c AS (SELECT * FROM users) SELECT 1 FROM c", "postgres"),
        ("SELECT 1 FROM users WHERE EXISTS (SELECT 1 FROM (SELECT * FROM logs) sub)", "postgres"),
    ]
    for query, dialect in flagged_queries:
        res = compute_wildcard_detector([query], dialect=dialect, include_per_example=True)
        assert res["per_example"] == [1], query
        assert res["wildcard_rate"] == 1.0

    safe_queries = [
        ("SELECT COUNT(*) FROM users", "postgres"),
        ("SELECT price * qty AS total FROM orders", "postgres"),
        ("SELECT 1 FROM users WHERE EXISTS (SELECT * FROM orders)", "postgres"),
        ("SELECT 1 FROM users WHERE NOT EXISTS (SELECT u.* FROM orders u)", "sqlite"),
        ("DELETE FROM users RETURNING *", "postgres"),
    ]
    for query, dialect in safe_queries:
        res = compute_wildcard_detector([query], dialect=dialect, include_per_example=True)
        assert res["per_example"] == [0], query
        assert res["wildcard_rate"] == 0.0


def test_wildcard_unparseable_extraction_and_streaming() -> None:
    detector = WildcardDetector(
        dialect="sqlite",
        extractor=extract_fenced_or_last_sql,
        include_per_example=True,
    )
    detector.add("```sql\nSELECT * FROM t\n```")
    detector.add_batch(
        [
            "```sql\nSELECT COUNT(*) FROM t\n```",
            "no sql block",
            "```sql\nSELECT 1; SELECT 2;\n```",
        ]
    )

    res = detector.compute()
    assert res["num_examples"] == 4
    assert res["parsed_count"] == 2
    assert res["wildcard_count"] == 1
    assert res["parse_failed_count"] == 2
    assert res["extraction_failed_count"] == 1
    assert math.isclose(res["wildcard_rate"], 0.5, abs_tol=1e-12)
    assert res["per_example"] == [1, 0, None, None]

    with pytest.raises(ValueError):
        detector.compute()

    # All failed -> wildcard_rate is None
    all_fail = compute_wildcard_detector(["SELECT FROM"])
    assert all_fail["wildcard_rate"] is None
    assert all_fail["parse_failed_count"] == 1

    with pytest.raises(TypeError):
        compute_wildcard_detector("SELECT * FROM t")  # type: ignore[arg-type]
