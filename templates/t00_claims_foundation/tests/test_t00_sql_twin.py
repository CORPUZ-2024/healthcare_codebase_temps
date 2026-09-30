"""The SQL twin must produce the same answer as the pandas implementation."""
import pandas as pd
import pytest

from claims_foundation import methods

duckdb = pytest.importorskip("duckdb")
from claims_foundation.sqltwin import run_sql  # noqa: E402


def test_latest_version_sql_matches_pandas(universe):
    med = universe["medical"]
    py = methods.collapse_versions_latest(med).sort_values(["claim_id", "line_seq"]).reset_index(drop=True)
    sql = run_sql("01_claims_latest_version.sql", {"claims": med}).reset_index(drop=True)
    assert len(py) == len(sql)
    assert py["paid_amt"].sum() == pytest.approx(sql["paid_amt"].sum())


def test_member_months_sql_matches_pandas(universe):
    enr = universe["enrollment"]
    py = methods.member_months_daily(enr, "2023-01-01", "2023-12-31")
    sql = run_sql("02_member_months.sql", {"enrollment": enr}, period_start="2023-01-01", period_end="2023-12-31")
    assert sql["member_months"].sum() == pytest.approx(py["member_months"].sum())
    assert len(sql) == len(py)


def test_service_category_sql_matches_pandas(universe):
    latest = methods.collapse_versions_latest(universe["medical"])
    py = (methods.service_category_claim(latest)[["claim_id", "service_category"]]
          .drop_duplicates().sort_values("claim_id").reset_index(drop=True))
    sql = run_sql("03_service_category.sql", {"claims": latest}).reset_index(drop=True)
    pd.testing.assert_frame_equal(py, sql, check_dtype=False)
