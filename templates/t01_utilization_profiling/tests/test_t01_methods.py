import numpy as np
import pandas as pd
import pytest

from conftest import ip_claims
from utilization_profiling import methods


def test_build_stays_merges_transfers_only():
    c = ip_claims([("a", "M1", "2025-01-01", "2025-01-05"), ("b", "M1", "2025-01-06", "2025-01-09"),
                   ("c", "M1", "2025-01-12", "2025-01-14"), ("d", "M2", "2025-01-06", "2025-01-09")])
    s = methods.build_stays(c)
    assert len(s) == 3
    m1 = s[s["member_id"] == "M1"].sort_values("admit_dt")
    assert m1["los_days"].tolist() == [8, 2] and m1["n_claims"].tolist() == [2, 1]


def test_readmission_window_boundaries():
    c = ip_claims([("a", "M1", "2025-01-01", "2025-01-05"), ("b", "M1", "2025-02-04", "2025-02-06"),   # day 30
                   ("c", "M1", "2025-03-10", "2025-03-12")])                                           # day 32
    s = methods.build_stays(c)
    idx = methods.readmissions_per_index(s, pd.Series({"M1": pd.Timestamp("2025-12-31")}),
                                         data_end=pd.Timestamp("2025-12-31"))
    assert idx["readmit_flag"].tolist() == [1, 0, 0]
    assert idx["days_to_readmit"].iloc[0] == 30


def test_ineligible_when_not_enrolled_through_window():
    c = ip_claims([("a", "M1", "2025-01-01", "2025-01-05")])
    idx = methods.readmissions_per_index(methods.build_stays(c), pd.Series({"M1": pd.Timestamp("2025-01-20")}),
                                         data_end=pd.Timestamp("2025-12-31"))
    assert idx["eligible_flag"].tolist() == [0]


def test_poisson_exact_ci_known_answer():
    r = methods.rate_ci_poisson_exact(0, 1_200)
    assert r["lo"] == 0 and r["hi"] == pytest.approx(36.89, abs=0.01)      # 3.689 x 10


def test_bootstrap_wider_than_poisson_under_overdispersion():
    rng = np.random.default_rng(0)
    ev = pd.Series(np.where(rng.random(500) < 0.05, rng.poisson(15, 500), 0), index=[f"m{i}" for i in range(500)])
    mm = pd.Series(12.0, index=ev.index)
    p = methods.rate_ci_poisson_exact(int(ev.sum()), mm.sum())
    b = methods.rate_ci_cluster_bootstrap(ev, mm, n_boot=1_000)
    assert (b["hi"] - b["lo"]) > 2 * (p["hi"] - p["lo"])
    assert b["rate"] == pytest.approx(p["rate"])


def test_per_1000_alternative_counts_all_readmits():
    idx = pd.DataFrame({"readmit_flag": [1, 1, 0]})
    assert methods.readmissions_per_1000(idx, 1_200)["rate"] == pytest.approx(20.0)


def test_monthly_rates_reindex_missing_months():
    ev = pd.DataFrame({"member_id": ["a"], "event_type": ["ED_VISIT"], "event_dt": pd.to_datetime(["2025-01-10"])})
    mm = pd.DataFrame({"member_id": ["a", "a", "a"], "month": pd.PeriodIndex(["2025-01", "2025-02", "2025-04"], freq="M"),
                       "member_months": [1.0, 1.0, 1.0]})
    out = methods.monthly_rates(ev, mm, "ED_VISIT")
    assert len(out) == 4                                            # March reindexed, not skipped
    assert np.isnan(out.loc[pd.Period("2025-03", "M"), "rate_per_1000"])
    assert out.loc[pd.Period("2025-02", "M"), "rate_per_1000"] == 0


def test_trend_blanks_incomplete_months(prepared):
    _, claims, mm = prepared
    monthly = methods.monthly_rates(methods.count_events(claims), mm, "ED_VISIT")
    cut = monthly.index.max() - 3
    r12 = methods.trend_rolling12(monthly, cut)
    assert r12.loc[r12.index > cut, "r12_rate_per_1000"].isna().all()
    assert r12["r12_rate_per_1000"].notna().sum() == len(monthly) - 3 - 11
    yoy = methods.trend_yoy_same_month(monthly, cut)
    assert yoy["yoy_pct"].notna().sum() == len(monthly) - 12 - 3


def test_frequent_ed_users_rolling_window():
    d = pd.to_datetime(["2025-01-01", "2025-03-01", "2025-06-01", "2025-09-01", "2026-06-01"])
    ev = pd.DataFrame({"member_id": "a", "event_type": "ED_VISIT", "event_dt": d})
    assert methods.frequent_ed_users(ev, 4)["max_visits_in_window"].tolist() == [4]
    assert methods.frequent_ed_users(ev, 5).empty


def test_utilization_table_consistency(prepared):
    _, claims, mm = prepared
    stays = methods.build_stays(claims)
    t = methods.utilization_table(methods.count_events(claims), mm, stays).set_index("event_type")
    assert t.loc["IP_ADMIT", "events"] == len(stays)
    assert t.loc["IP_ADMIT", "days_per_1000"] == pytest.approx(t.loc["IP_ADMIT", "rate"] * t.loc["IP_ADMIT", "alos_days"])
