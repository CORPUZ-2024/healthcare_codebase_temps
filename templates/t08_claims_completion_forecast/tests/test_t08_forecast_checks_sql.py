import numpy as np
import pandas as pd
import pytest

from claims_completion_forecast import checks, data, methods


# --- forecasts -----------------------------------------------------------------------------------

def test_forecast_shapes_and_intervals(completed):
    for f in (methods.forecast_sarimax, methods.forecast_ets):
        fc = f(completed["completed_pmpm"], 12)
        assert len(fc) == 12 and (fc["lo"] < fc["mean"]).all() and (fc["mean"] < fc["hi"]).all()
        assert fc.index[0] == completed.index[-1] + pd.DateOffset(months=1)


def test_sarimax_interval_widens_with_horizon(completed):
    fc = methods.forecast_sarimax(completed["completed_pmpm"], 12)
    width = fc["hi"] - fc["lo"]
    assert width.iloc[-1] > width.iloc[0]


def test_forecast_recovers_trend_and_season(completed):
    """Next January should be above next July (winter peak) and above this January (6% trend)."""
    fc = methods.forecast_sarimax(completed["completed_pmpm"], 12)["mean"]
    assert fc.iloc[0] > fc.iloc[6]
    assert fc.iloc[0] > completed["completed_pmpm"].iloc[-12]


def test_forecast_backtest_accuracy(completed):
    bt = methods.backtest_forecasts(completed["true_pmpm"], 6)
    assert bt["mape_sarimax"] < 0.08 and bt["mape_ets"] < 0.08


def test_forecasting_paid_pmpm_is_worse(completed):
    """Feeding raw paid PMPM (incomplete months) drags the forecast down."""
    good = methods.forecast_sarimax(completed["completed_pmpm"], 3)["mean"].mean()
    bad = methods.forecast_sarimax(completed["paid_pmpm"], 3)["mean"].mean()
    assert bad < 0.9 * good


# --- checks --------------------------------------------------------------------------------------

def test_immature_check():
    df = pd.DataFrame({"completion_pct": [0.99, 0.35], "method": ["CL", "CL"]})
    assert checks.check_immature_months(df)[0].n_rows == 1
    assert not checks.check_immature_months(df.assign(method=["CL", "BF"]))


def test_factor_volatility_check():
    f = pd.DataFrame({"lag": [0, 1], "cv_individual": [0.25, 0.02]})
    assert checks.check_factor_volatility(f)[0].n_rows == 1


def test_negative_incrementals_check():
    t = pd.DataFrame({0: [10.0, 5.0], 1: [-2.0, np.nan]})
    assert checks.check_negative_incrementals(t)[0].n_rows == 1


def test_lag_shift_check(synth, cfg):
    cum = methods.cumulative(methods.build_triangle(synth["claims"], synth["as_of"], cfg.max_lag))
    assert not checks.check_lag_shift(cum, methods.chain_ladder(cum)[1])
    sp = data.generate(cfg.start, cfg.months, cfg.members, cfg.base_pmpm, cfg.annual_trend, cfg.max_lag, speedup=0.5, seed=cfg.seed)
    cum_sp = methods.cumulative(methods.build_triangle(sp["claims"], sp["as_of"], cfg.max_lag))
    assert checks.check_lag_shift(cum_sp, methods.chain_ladder(cum_sp)[1])[0].check_id == "IBNR-004"


def test_forecast_input_check(completed):
    assert not checks.check_forecast_input(completed)
    assert checks.check_forecast_input(completed.tail(18))[0].check_id == "FC-001"
    assert any(f.check_id == "FC-002" for f in checks.check_forecast_input(completed.drop(columns="completed_pmpm")))


# --- SQL twin ------------------------------------------------------------------------------------

duckdb = pytest.importorskip("duckdb")
from claims_completion_forecast.sqltwin import run_sql  # noqa: E402


def test_triangle_sql_matches_pandas(synth, cfg):
    sql = run_sql("01_lag_triangle.sql", {"claims": synth["claims"]}, as_of=str(synth["as_of"].date()))
    wide = sql.pivot(index="incurred_month_start", columns="lag", values="paid_amt")
    py = methods.build_triangle(synth["claims"], synth["as_of"], cfg.max_lag)
    for lag in (0, 1, 5):
        common = wide[lag].dropna().index
        assert np.allclose(wide.loc[common, lag].to_numpy(), py.loc[common, lag].to_numpy())


def test_paid_vs_incurred_sql(synth):
    t = run_sql("02_paid_vs_incurred_pmpm.sql", {"claims": synth["claims"], "exposure": synth["exposure"]})
    last = t.iloc[-1]
    assert last["pmpm_incurred_basis"] < 0.5 * last["pmpm_paid_basis"]     # runout vs. mixed-month view
