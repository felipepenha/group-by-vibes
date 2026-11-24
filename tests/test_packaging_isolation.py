"""Tests for packaging isolation, `load()` dispatch, concurrency, and bounds validation."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys

import pytest

from group_by_vibes import (
    SqlBleu,
    ValidSqlRate,
    WildcardDetector,
    canonicalize_sql,
    compute_sql_bleu,
    compute_valid_sql_rate,
    compute_wildcard_detector,
    load,
    sentence_sql_bleu,
    sql_tokenizer,
    try_canonicalize_sql,
)


def test_core_package_has_no_heavy_ml_imports_in_fresh_process() -> None:
    script = """
import sys
import group_by_vibes as gbv
assert gbv.compute_sql_bleu(["SELECT a FROM t"], ["SELECT a FROM t"])["bleu"] is not None
assert gbv.compute_valid_sql_rate(["SELECT id FROM u"], [{"u": ["id"]}])["valid_sql_rate"] == 1.0
assert gbv.compute_wildcard_detector(["SELECT * FROM u"])["wildcard_rate"] == 1.0
forbidden = {"evaluate", "datasets", "pyarrow", "torch", "transformers", "huggingface_hub"}
loaded = forbidden.intersection(sys.modules)
assert not loaded, f"Unexpected ML dependencies imported: {sorted(loaded)}"
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_load_dispatch_and_unknown_name_error() -> None:
    assert isinstance(load("sql_bleu"), SqlBleu)
    assert isinstance(load("valid_sql_rate"), ValidSqlRate)
    assert isinstance(load("wildcard_detector"), WildcardDetector)
    with pytest.raises(ValueError):
        load("unknown_metric")


def test_char_limit_validation_across_public_api() -> None:
    for bad in (0, -5, True, False):
        with pytest.raises(ValueError):
            SqlBleu(max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ValidSqlRate(max_input_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            WildcardDetector(max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            try_canonicalize_sql("SELECT 1", max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            canonicalize_sql("SELECT 1", max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sql_tokenizer("SELECT 1", max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            compute_sql_bleu(["SELECT 1"], ["SELECT 1"], max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sentence_sql_bleu("SELECT 1", "SELECT 1", max_input_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            compute_valid_sql_rate(["SELECT 1"], max_query_chars=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            compute_wildcard_detector(["SELECT 1"], max_input_chars=bad)  # type: ignore[arg-type]


def test_stateless_helpers_safe_across_worker_threads_and_base_compute_guards() -> None:
    m = ValidSqlRate()
    with pytest.raises(TypeError):
        m.compute("SELECT 1")
    with pytest.raises(TypeError):
        WildcardDetector().compute(extra_arg=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [
            pool.submit(
                compute_valid_sql_rate,
                ["SELECT id FROM users", "SELECT bad FROM users"],
                [{"users": ["id"]}, {"users": ["id"]}],
                dialect="sqlite",
            )
            for _ in range(8)
        ]
        w_futs = [
            pool.submit(compute_wildcard_detector, ["SELECT * FROM t", "SELECT COUNT(*) FROM t"])
            for _ in range(8)
        ]
        for f in futs:
            assert f.result()["valid_sql_rate"] == 0.5
        for wf in w_futs:
            assert wf.result()["wildcard_rate"] == 0.5
