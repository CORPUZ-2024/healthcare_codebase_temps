"""Triangle, chain ladder, Bornhuetter-Ferguson: known answers and recovery of the synthetic truth."""
import numpy as np
import pandas as pd
import pytest

from claims_completion_forecast import data, methods


def _claims(rows):
    return pd.DataFrame(rows, columns=["incurred_month_start", "paid_month_start", "paid_amt"]).assign(
        incurred_month_start=lambda d: pd.to_datetime(d["incurred_month_start"]),
        paid_month_start=lambda d: pd.to_datetime(d["paid_month_start"]))


TOY = _claims([("2025-01-01", "2025-01-01", 50), ("2025-01-01", "2025-02-01", 30), ("2025-01-01", "2025-03-01", 20),
               ("2025-02-01", "2025-02-01", 60), ("2025-02-01", "2025-03-01", 30), ("2025-03-01", "2025-03-01", 40)])


def test_triangle_zero_vs_future():
    c = _claims([("2025-01-01", "2025-01-01", 10), ("2025-01-01", "2025-03-01", 5), ("2025-02-01", "2025-02-01", 7)])
    t = methods.build_triangle(c, as_of="2025-03-31")
    assert t.loc["2025-01-01", 1] == 0.0                      # observable, nothing paid -> 0
    assert np.isnan(t.loc["2025-02-01", 2])                   # future -> NaN
    assert t.loc["2025-03-01", 0] == 0.0 and t.shape == (3, 3)  # month with no claims yet still gets a row


def test_as_of_rebuilds_history():
    t = methods.build_triangle(TOY, as_of="2025-02-28")
    assert list(t.index) == list(pd.to_datetime(["2025-01-01", "2025-02-01"]))
    assert t.loc["2025-01-01", 1] == 30 and t.columns.tolist() == [0, 1]     # lag 2 not observable yet
    assert np.isnan(t.loc["2025-02-01", 1])


def test_lag_is_calendar_months():
    c = _claims([("2025-01-01", "2025-02-01", 1.0)])           # e.g. Jan 31 service, Feb 1 payment
    assert methods.build_triangle(c).columns.tolist() == [0, 1]
    assert methods.build_triangle(c).loc["2025-01-01", 1] == 1.0


def test_chain_ladder_known_answer():
    cl, f = methods.chain_ladder(methods.cumulative(methods.build_triangle(TOY)), n_avg=None)
    assert f["factor"].tolist() == pytest.approx([170 / 110, 1.25])
    assert cl.loc["2025-02-01", "ultimate_amt"] == pytest.approx(90 * 1.25)
    assert cl.loc["2025-03-01", "completion_pct"] == pytest.approx(1 / (170 / 110 * 1.25))
    assert cl["ibnr_amt"].iloc[0] == 0.0


def test_tail_factor_scales_everything():
    cum = methods.cumulative(methods.build_triangle(TOY))
    base, _ = methods.chain_ladder(cum, None)
    tail, _ = methods.chain_ladder(cum, None, tail_factor=1.02)
    assert (tail["ultimate_amt"] / base["ultimate_amt"]).to_numpy() == pytest.approx([1.02] * 3)


def test_bf_is_between_paid_and_prior():
    cum = methods.cumulative(methods.build_triangle(TOY))
    cl, _ = methods.chain_ladder(cum, None)
    bf = methods.bornhuetter_ferguson(cl, pd.Series(1.0, index=cl.index), pd.Series(120.0, index=cl.index))
    assert (bf >= cl["paid_to_date"]).all()
    assert bf.loc["2025-01-01"] == 100.0                      # fully paid month: BF = paid


def test_paid_basis_understates_recent_months(completed):
    last = completed.iloc[-1]
    assert last["paid_pmpm"] < 0.5 * last["true_pmpm"]        # the "costs are dropping!" illusion


def test_completed_pmpm_close_to_truth(completed):
    err = (completed["ultimate_amt"] / completed["true_ult"] - 1).abs()
    assert err.max() < 0.05
    assert (completed["method"].tail(3) == "BF").all() and (completed["method"].iloc[:-3] == "CL").all()


def test_total_ibnr_close_to_truth(completed):
    true_ibnr = (completed["true_ult"] - completed["paid_to_date"]).sum()
    assert completed["ibnr_amt"].sum() == pytest.approx(true_ibnr, rel=0.05)


def test_backtest_bf_beats_cl_for_the_newest_month(synth, cfg):
    cuts = pd.date_range(synth["as_of"] - pd.DateOffset(months=12), periods=12, freq="ME")
    bt = methods.backtest_ibnr(synth["claims"], synth["exposure"], synth["truth"], cuts, cfg.cl_avg_months, cfg.bf_max_age, cfg.max_lag)
    mae = bt.groupby("age")[["pct_error_cl", "pct_error_bf"]].agg(lambda x: np.abs(x).mean())
    assert mae.loc[0, "pct_error_bf"] < mae.loc[0, "pct_error_cl"]
    assert mae.loc[6, "pct_error_cl"] < mae.loc[0, "pct_error_cl"]          # error shrinks with age
    assert set(bt["cutoff"]) == set(cuts)


def test_speedup_biases_ibnr_up(cfg):
    """Faster processing in recent months + factors from the old pattern -> IBNR overstated."""
    sp = data.generate(cfg.start, cfg.months, cfg.members, cfg.base_pmpm, cfg.annual_trend, cfg.max_lag, speedup=0.5, seed=cfg.seed)
    r = methods.complete(sp["claims"], sp["exposure"], sp["as_of"], bf_max_age=-1, max_lag=cfg.max_lag)   # pure CL
    true_ult = sp["truth"].set_index("incurred_month_start")["ultimate_paid_amt"]
    assert (r["ultimate_amt"] / true_ult - 1).iloc[-1] > 0.10


def test_lag_pattern_sums_to_one():
    p = data.lag_pattern(18)
    assert p.sum() == pytest.approx(1.0) and 0.2 < p[0] < 0.45 and p[-1] < 0.001


def test_load_claim_lines(tmp_path):
    f = tmp_path / "lines.csv"
    pd.DataFrame({"svc_from_dt": ["2025-01-15", "2025-01-31", "2025-03-02"], "paid_dt": ["2025-02-03", "2025-02-01", "2025-02-20"],
                  "paid_amt": ["100.5", "20", "9"]}).to_csv(f, index=False)
    df = data.load_claim_lines(f, "svc_from_dt", "paid_dt", "paid_amt")
    assert len(df) == 2                                        # paid-before-service row dropped
    assert (df["paid_month_start"] == pd.Timestamp("2025-02-01")).all()
