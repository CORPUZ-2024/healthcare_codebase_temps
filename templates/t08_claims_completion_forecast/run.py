"""
t08 — Claims completion (IBNR) and completed-PMPM forecast. Synthetic data with a known ultimate.

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from claims_completion_forecast import checks, data, methods  # noqa: E402
from claims_completion_forecast.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", module="statsmodels")
    d = data.generate(cfg.start, cfg.months, cfg.members, cfg.base_pmpm, cfg.annual_trend, cfg.max_lag, seed=cfg.seed)
    tri = methods.build_triangle(d["claims"], d["as_of"], cfg.max_lag)
    cum = methods.cumulative(tri)
    _, factors = methods.chain_ladder(cum, cfg.cl_avg_months, cfg.tail_factor)
    res = methods.complete(d["claims"], d["exposure"], d["as_of"], cfg.cl_avg_months, cfg.tail_factor, cfg.bf_max_age, cfg.max_lag)
    truth = d["truth"].set_index("incurred_month_start")
    res["true_pmpm"] = truth["true_pmpm"]

    print(f"== Development factors (volume-weighted, last {cfg.cl_avg_months} months) ==")
    print(factors.head(8).round(4).to_string(index=False))
    print(f"\n== Last 8 incurred months as of {d['as_of'].date()} (STANDARD chain ladder; ALTERNATIVE BF for age <= {cfg.bf_max_age}) ==")
    show = res.tail(8)[["age", "completion_pct", "paid_pmpm", "completed_pmpm", "method", "true_pmpm"]].copy()
    show["cl_pmpm"] = (res["ultimate_cl"] / res["member_months"]).tail(8)
    show["bf_pmpm"] = (res["ultimate_bf"] / res["member_months"]).tail(8)
    show.index = show.index.strftime("%Y-%m")
    print(show.round(3).to_string())
    true_ibnr = float((truth["ultimate_paid_amt"] - res["paid_to_date"]).sum())
    print(f"\nIBNR estimate ${res['ibnr_amt'].sum():,.0f} vs. TRUE ${true_ibnr:,.0f} (synthetic only); "
          f"paid-basis PMPM for the last month understates by {1 - res['paid_pmpm'].iloc[-1] / res['true_pmpm'].iloc[-1]:.0%}")

    cuts = pd.date_range(d["as_of"] - pd.DateOffset(months=12), periods=12, freq="ME")
    bt = methods.backtest_ibnr(d["claims"], d["exposure"], d["truth"], cuts, cfg.cl_avg_months, cfg.bf_max_age, cfg.max_lag)
    mae = bt.groupby("age")[["pct_error_cl", "pct_error_bf"]].agg(lambda x: np.abs(x).mean())
    print("\n== IBNR backtest: mean |error| of the ultimate by age, 12 historical data cuts ==")
    print(mae.rename(columns={"pct_error_cl": "chain_ladder", "pct_error_bf": "bornhuetter_ferguson"}).round(4).to_string())

    pm = res["completed_pmpm"]
    fs, fe = methods.forecast_sarimax(pm, cfg.horizon), methods.forecast_ets(pm, cfg.horizon)
    print(f"\n== {cfg.horizon}-month forecast of completed PMPM (95% interval) ==")
    fc = pd.DataFrame({"sarimax_mean": fs["mean"], "sarimax_lo": fs["lo"], "sarimax_hi": fs["hi"],
                       "ets_mean": fe["mean"].to_numpy(), "ets_lo": fe["lo"].to_numpy(), "ets_hi": fe["hi"].to_numpy()})
    fc.index = fc.index.strftime("%Y-%m")
    print(fc.round(0).to_string())
    bf = methods.backtest_forecasts(pm, cfg.backtest_months)
    print(f"holdout ({cfg.backtest_months} months): MAPE SARIMAX {bf['mape_sarimax']:.1%} (interval coverage "
          f"{bf['coverage_sarimax']:.0%}) | ETS {bf['mape_ets']:.1%} (coverage {bf['coverage_ets']:.0%})")

    print("\n== Checks ==")
    fnd = (checks.check_immature_months(res) + checks.check_factor_volatility(factors) + checks.check_negative_incrementals(tri)
           + checks.check_lag_shift(cum, factors) + checks.check_forecast_input(res))
    sp = data.generate(cfg.start, cfg.months, cfg.members, cfg.base_pmpm, cfg.annual_trend, cfg.max_lag, speedup=0.5, seed=cfg.seed)
    cum_sp = methods.cumulative(methods.build_triangle(sp["claims"], sp["as_of"], cfg.max_lag))
    fnd += [f for f in checks.check_lag_shift(cum_sp, methods.chain_ladder(cum_sp, cfg.cl_avg_months)[1])]
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")
    print("(the IBNR-004 line, if shown, comes from a second run with faster claims processing in the last 6 months)")

    tri.to_csv(OUT / "incremental_triangle.csv")
    factors.to_csv(OUT / "development_factors.csv", index=False)
    res.to_csv(OUT / "completed_pmpm.csv")
    bt.to_csv(OUT / "ibnr_backtest.csv", index=False)
    fc.to_csv(OUT / "pmpm_forecast.csv")
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    c = pd.DataFrame({"incurred_month_start": pd.to_datetime(["2025-01-01"] * 3 + ["2025-02-01"] * 2 + ["2025-03-01"]),
                      "paid_month_start": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-02-01", "2025-03-01", "2025-03-01"]),
                      "paid_amt": [50.0, 30.0, 20.0, 60.0, 30.0, 40.0]})
    cum = methods.cumulative(methods.build_triangle(c))
    cl, f = methods.chain_ladder(cum, None)
    exp = pd.Series(1.0, index=cl.index)
    bf = methods.bornhuetter_ferguson(cl, exp, pd.Series(120.0, index=cl.index))
    tests = {
        "triangle: Feb lag 2 is in the future (NaN), not 0": lambda: np.isnan(cum.loc["2025-02-01", 2]),
        "f0 = (80 + 90) / (50 + 60) = 1.5455": lambda: abs(f["factor"].iloc[0] - 170 / 110) < 1e-9,
        "f1 = 100 / 80 = 1.25": lambda: abs(f["factor"].iloc[1] - 1.25) < 1e-9,
        "Mar ultimate (CL) = 40 x 1.5455 x 1.25 = 77.27": lambda: abs(cl.loc["2025-03-01", "ultimate_amt"] - 40 * 170 / 110 * 1.25) < 1e-9,
        "Mar ultimate (BF) = 40 + 120 x (1 - 1/1.9318)": lambda: abs(bf.loc["2025-03-01"] - (40 + 120 * (1 - 1 / (170 / 110 * 1.25)))) < 1e-9,
        "Jan fully developed: IBNR 0": lambda: abs(cl.loc["2025-01-01", "ibnr_amt"]) < 1e-9,
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else (demo(Config()) or 0))
