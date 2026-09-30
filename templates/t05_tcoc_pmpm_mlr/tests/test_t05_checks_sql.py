import pandas as pd
import pytest

from tcoc_pmpm_mlr import checks, methods


def test_claims_without_exposure():
    mm = pd.DataFrame({"member_id": ["A"], "month": pd.PeriodIndex(["2025-01"], freq="M")})
    c = pd.DataFrame({"member_id": ["A", "B"], "month": pd.PeriodIndex(["2025-01", "2025-01"], freq="M")})
    assert checks.check_claims_without_exposure(c, mm)[0].n_rows == 1


def test_high_cost_concentration():
    s = pd.Series([1.0] * 99 + [1_000.0])
    assert checks.check_high_cost_concentration(s)[0].check_id == "ANL-009"
    assert not checks.check_high_cost_concentration(pd.Series([1.0] * 100))


def test_small_exposure():
    t = pd.DataFrame({"member_months": [50.0, 5_000.0]})
    assert checks.check_small_exposure(t)[0].n_rows == 1


duckdb = pytest.importorskip("duckdb")
from tcoc_pmpm_mlr.sqltwin import run_sql  # noqa: E402


def _sql_inputs(prepared):
    c, mm = prepared
    c2 = c.assign(month_start=c["month"].dt.start_time)[["member_id", "month_start", "paid_amt"]]
    mm2 = mm.assign(month_start=mm["month"].dt.start_time)[["member_id", "month_start", "lob_cd", "member_months"]]
    return c2, mm2


def test_pmpm_sql_matches_pandas(prepared):
    c, mm = prepared
    c2, mm2 = _sql_inputs(prepared)
    sql = run_sql("01_pmpm_by_group.sql", {"claims": c2, "member_months": mm2}).set_index("lob_cd")
    py = methods.pmpm_ratio_of_sums(c, mm, by=["lob_cd"]).set_index("lob_cd")
    assert sql["pmpm"].to_dict() == pytest.approx(py["pmpm"].to_dict())


def test_wrong_join_understates_pmpm(prepared):
    c2, mm2 = _sql_inputs(prepared)
    right = run_sql("01_pmpm_by_group.sql", {"claims": c2, "member_months": mm2})["pmpm"]
    wrong = run_sql("02_pmpm_wrong_join.sql", {"claims": c2, "member_months": mm2})["pmpm"]
    assert (wrong.to_numpy() < right.to_numpy()).all()
