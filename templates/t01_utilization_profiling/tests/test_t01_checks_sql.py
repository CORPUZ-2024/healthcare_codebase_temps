import pandas as pd
import pytest

from utilization_profiling import checks, methods


def test_ip_missing_dates_check(ip_claims):
    c = ip_claims([("a", "M1", "2025-01-01", "2025-01-05")])
    c.loc[0, "discharge_dt"] = pd.NaT
    assert checks.check_ip_missing_dates(c)[0].check_id == "VAL-015"


def test_small_denominator():
    assert checks.check_small_denominator(12) and not checks.check_small_denominator(300)


def test_runout_check_fires_on_synthetic(prepared):
    _, claims, mm = prepared
    monthly = methods.monthly_rates(methods.count_events(claims), mm, "OFFICE_VISIT")
    assert [f.check_id for f in checks.check_runout(monthly)] == ["VAL-031"]


duckdb = pytest.importorskip("duckdb")
from utilization_profiling.sqltwin import run_sql  # noqa: E402


def test_ed_monthly_sql_matches_pandas(prepared):
    _, claims, mm = prepared
    py = methods.monthly_rates(methods.count_events(claims), mm, "ED_VISIT")
    sql = run_sql("01_ed_per_1000_by_month.sql",
                  {"claims": claims, "member_months": mm.assign(month_start=mm["month"].dt.start_time).drop(columns="month")})
    assert sql["events"].tolist() == py["events"].tolist()
    assert sql["rate_per_1000"].to_numpy() == pytest.approx(py["rate_per_1000"].to_numpy())


def test_readmission_sql_matches_pandas(prepared):
    u, claims, _ = prepared
    stays = methods.build_stays(claims)
    py = methods.readmissions_per_index(stays, u["enrollment"].groupby("member_id")["enroll_end_dt"].max())
    sql = run_sql("02_readmissions.sql", {"stays": stays}, window_days=30)
    assert sql["readmit_flag"].sum() == py["readmit_flag"].sum()
