"""O/E, funnel limits, shrinkage and the mixed model against known provider quality."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from provider_benchmark_profiling import checks, data, methods


def test_oe_known_answer_and_exact_ci():
    pts = pd.DataFrame({"provider_id": ["A"] * 10, "readmit_flag": [1] * 3 + [0] * 7})
    r = methods.oe_table(pts, pd.Series([0.2] * 10)).iloc[0]
    assert (r["observed"], r["expected"], r["oe"]) == (3, 2.0, 1.5)
    assert r["oe_lo"] == pytest.approx(stats.chi2.ppf(0.025, 6) / 2 / 2)
    assert r["oe_hi"] == pytest.approx(stats.chi2.ppf(0.975, 8) / 2 / 2)


def test_risk_model_calibrated_in_development_data(profile):
    assert profile["info"]["overall_oe"] == pytest.approx(1.0, abs=1e-6)
    assert 0.6 < profile["info"]["c_statistic"] < 0.8


def test_funnel_narrows_with_volume():
    lim = methods.funnel_limits(np.array([5.0, 50.0, 500.0]), (0.95,))
    width = lim["hi_0.95"] - lim["lo_0.95"]
    assert width.is_monotonic_decreasing


def test_funnel_false_alarm_rate_without_provider_effects():
    """No true differences (provider_sd ~ 0): ~5% outside the 95% limits, ~0.2% outside 99.8%."""
    d = data.generate(n_providers=200, provider_sd=1e-6, seed=5)
    p, _ = methods.risk_model(d["patients"])
    f = methods.funnel_flags(methods.oe_table(d["patients"], p))
    assert (f["flag_0.95"] != 0).mean() < 0.10 and (f["flag_0.998"] != 0).mean() < 0.02


def test_shrinkage_and_mixed_model_beat_raw_oe(profile):
    t = profile["truth"]["true_ratio"]
    raw = methods.estimation_error(profile["eb"]["oe"].reindex(t.index).clip(lower=0.01), t)
    eb = methods.estimation_error(profile["eb"]["eb_oe"].reindex(t.index), t)
    mx = methods.estimation_error(profile["mixed"]["pe_ratio"].reindex(t.index), t)
    assert eb["rmse_log"] < raw["rmse_log"] / 2 and mx["rmse_log"] < raw["rmse_log"] / 2
    assert abs(eb["rmse_log"] - mx["rmse_log"]) < 0.05                  # the two shrinkage routes agree


def test_risk_adjustment_beats_crude_ranking(profile, sim):
    t = profile["truth"]["true_ratio"]
    crude = profile["eb"]["crude_rate"].reindex(t.index)
    assert stats.spearmanr(profile["eb"]["oe"].reindex(t.index), t).statistic > stats.spearmanr(crude, t).statistic + 0.1


def test_shrinkage_strongest_for_small_providers(profile):
    eb = profile["eb"]
    moved = (eb["eb_oe"] - eb["oe"]).abs()
    small, large = eb["n_cases"] < 60, eb["n_cases"] > 400
    assert moved[small].mean() > 3 * moved[large].mean()
    assert eb.loc[small, "reliability"].max() < eb.loc[large, "reliability"].min()


def test_eb_tau2_near_truth(profile, cfg):
    """Method-of-moments between-provider variance is in the right range (true log-ratio SD ~ 0.2)."""
    assert 0.01 < profile["eb"].attrs["tau2"] < 0.09


def test_beta_binomial_eb():
    k, n = pd.Series([0, 5, 50, 1]), pd.Series([5, 50, 500, 3])
    r = methods.eb_beta_binomial(k, n)
    assert (r["eb_rate"].between(0, 1)).all()
    assert abs(r["eb_rate"].iloc[2] - 0.1) < abs(r["eb_rate"].iloc[0] - 0.0)    # big n barely moves


def test_hrrp_loader_and_benchmark(tmp_path):
    f = tmp_path / "hrrp.csv"
    data.generate_fake_hrrp(seed=13).to_csv(f, index=False)         # 2,500 hospitals; a few suppressed (< 25)
    b = data.load_hrrp(f)
    assert b["excess_readmission_ratio"].isna().sum() == (b["n_discharges"].isna()).sum() > 0     # suppressed
    assert b["predicted_rate"].max() < 1 and b["end_date"].iloc[0] == pd.Timestamp("2024-06-30")
    pct = methods.benchmark_percentile(1.0, b["excess_readmission_ratio"])
    assert 30 < pct["percentile"] < 70


def test_checks(profile, cfg):
    assert checks.check_small_volume(profile["oe"], cfg.min_cases)[0].check_id == "BEN-001"
    assert checks.check_overdispersion(profile["oe"])[0].check_id == "BEN-002"
    assert not checks.check_risk_model(profile["info"])
    assert checks.check_risk_model({"c_statistic": 0.55, "overall_oe": 1.1})[1].severity == "error"
    assert checks.check_rank_disagreement(profile["eb"]["crude_rate"], profile["eb"]["oe"])
    assert checks.check_benchmark_vintage("2021-07-01", "2024-06-30", "2024-07-01", "2025-06-30")[0].check_id == "ANL-013"
    assert not checks.check_benchmark_vintage("2024-07-01", "2025-06-30", "2024-07-01", "2025-06-30")


def test_funnel_plot_writes_png(profile, tmp_path):
    pytest.importorskip("matplotlib")
    methods.funnel_plot(profile["oe"], tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 5_000
