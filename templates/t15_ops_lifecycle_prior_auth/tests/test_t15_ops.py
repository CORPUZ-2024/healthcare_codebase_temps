"""Funnel, censoring-aware time to start, PA metrics against CMS-0057-F targets, cohorts, Pareto, checks."""
import numpy as np
import pandas as pd
import pytest

from ops_lifecycle_prior_auth import checks, methods


def test_km_matches_hand_calculation():
    assert methods.km_curve(np.array([2, 3, 4, 5]), np.array([1, 0, 1, 0]))["survival"].tolist() == pytest.approx([0.75, 0.375])


def test_km_recovers_truth_despite_open_referrals(sim, cfg):
    obs, truth = sim["observed"], sim["truth"]
    km = methods.time_to_start(obs, cfg.as_of, (14, 30))
    true_days = (truth["soc_dt"] - truth["referral_dt"]).dt.days
    for d in (14, 30):
        assert km["by_day"][d] == pytest.approx((true_days <= d).mean(), abs=0.01)


def test_recent_cohort_naive_share_is_badly_biased_km_is_not(sim, cfg):
    obs, truth = sim["observed"], sim["truth"]
    dec = obs["referral_dt"] >= pd.Timestamp(cfg.as_of).replace(day=1)
    true30 = ((truth.loc[dec, "soc_dt"] - truth.loc[dec, "referral_dt"]).dt.days <= 30).mean()
    naive = obs.loc[dec, "soc_dt"].notna().mean()
    km30 = methods.time_to_start(obs[dec], cfg.as_of, (30,))["by_day"][30]
    assert naive < true30 - 0.2 and abs(km30 - true30) < 0.05


def test_funnel_counts_monotone_and_mature_view_higher(sim, cfg):
    f_all = methods.funnel(sim["observed"])
    f_mat = methods.funnel(sim["observed"], cfg.as_of, min_age_days=60)
    main = f_all[f_all["stage"] != "retained_90d"]
    assert main["n"].is_monotonic_decreasing and main["n"].iloc[0] == cfg.n_referrals
    s = "started_care"
    assert f_mat.set_index("stage").loc[s, "conv_from_referral"] > f_all.set_index("stage").loc[s, "conv_from_referral"]


def test_funnel_hand_example():
    ts = pd.Timestamp
    df = pd.DataFrame({"referral_dt": [ts("2025-01-01")] * 4,
                       "assessment_dt": [ts("2025-01-03"), ts("2025-01-05"), ts("2025-01-04"), pd.NaT],
                       "pa_request_ts": [ts("2025-01-04"), ts("2025-01-06"), pd.NaT, pd.NaT],
                       "pa_decision_ts": [ts("2025-01-05"), ts("2025-01-08"), pd.NaT, pd.NaT],
                       "soc_dt": [ts("2025-01-10"), pd.NaT, pd.NaT, pd.NaT], "retained_90d_flag": [1.0, np.nan, np.nan, np.nan]})
    f = methods.funnel(df).set_index("stage")
    assert f["n"].tolist() == [4, 3, 2, 2, 1, 1]
    assert f.loc["assessed", "median_days_from_prev"] == 3.0 and f.loc["pa_requested", "conv_from_prev"] == pytest.approx(2 / 3)


def test_pa_metrics_recover_generator(sim, cfg):
    pa = methods.pa_metrics(sim["observed"], cfg.expedited_target_hours, cfg.standard_target_hours).set_index("priority")
    assert pa["pct_denied"].between(0.12, 0.25).all()
    assert pa["pct_overturned_of_appealed"].between(0.35, 0.65).all()                  # generator: 50%
    assert pa.loc["expedited", "median_hours"] < pa.loc["standard", "median_hours"]
    assert (pa["pct_approved"] + pa["pct_denied"]).to_numpy() == pytest.approx([1, 1])


def test_cohort_table_marks_immature(sim, cfg):
    c = methods.cohort_conversion(sim["observed"], cfg.as_of, cfg.cohort_window_days)
    assert c["mature_flag"].iloc[-1] == 0 and np.isnan(c["conversion_rate"].iloc[-1])
    assert c.loc[c["mature_flag"] == 1, "conversion_rate"].between(0.5, 0.75).all()


def test_denial_pareto(sim):
    p = methods.denial_pareto(sim["observed"])
    assert p["n"].is_monotonic_decreasing and p["cum_share"].iloc[-1] == pytest.approx(1.0)
    assert p["denial_reason"].iloc[0] == "missing clinical documentation"


def test_checks(sim, cfg):
    pa = methods.pa_metrics(sim["observed"])
    assert checks.check_turnaround(pa)[0].check_id == "OPS-001"
    assert checks.check_overturns(pa)[0].check_id == "OPS-004"
    assert checks.check_immature_cohorts(methods.cohort_conversion(sim["observed"], cfg.as_of))
    assert not checks.check_timestamp_order(sim["observed"])
    broken = sim["observed"].head(20).copy()
    broken.loc[broken.index[0], "soc_dt"] = pd.Timestamp("2020-01-01")
    assert checks.check_timestamp_order(broken)[0].n_rows == 1
    assert checks.check_open_cases(0.38) and not checks.check_open_cases(0.05)


def test_observed_never_shows_the_future(sim, cfg):
    cut = pd.Timestamp(cfg.as_of)
    for c in ("assessment_dt", "pa_request_ts", "pa_decision_ts", "soc_dt"):
        assert (sim["observed"][c].dropna() <= cut).all()
    later = sim["truth"]["soc_dt"] > cut
    assert later.sum() > 0 and sim["observed"].loc[later, "soc_dt"].isna().all()
