"""
t15 — Operations: referral-to-care funnel, prior-authorization turnaround (CMS-0057-F targets), denials,
time to start of care. Synthetic data observed at a data cut, with the full truth kept for testing.

    python run.py              demo -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ops_lifecycle_prior_auth import checks, data, methods  # noqa: E402
from ops_lifecycle_prior_auth.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    d = data.generate(cfg.n_referrals, cfg.start, cfg.months, cfg.as_of, cfg.seed)
    obs, truth = d["observed"], d["truth"]
    print(f"{len(obs):,} referrals {cfg.start} .. {cfg.as_of}; data cut {cfg.as_of}\n")

    print("== STANDARD funnel: all referrals vs. referrals at least 60 days old ==")
    f_all, f_mat = methods.funnel(obs), methods.funnel(obs, cfg.as_of, min_age_days=60)
    print(f_all.merge(f_mat, on="stage", suffixes=("_all", "_60d_old"))[
        ["stage", "n_all", "conv_from_prev_all", "n_60d_old", "conv_from_prev_60d_old", "median_days_from_prev_60d_old"]].round(3).to_string(index=False))

    tts = methods.time_to_start(obs, cfg.as_of, cfg.report_days)
    true_days = (truth["soc_dt"] - truth["referral_dt"]).dt.days
    print("\n== Time from referral to start of care ==")
    for day in cfg.report_days:
        print(f"  P(started by day {day:>2}): KM {tts['by_day'][day]:.1%} | TRUE (full follow-up) {(true_days <= day).mean():.1%}")
    print(f"  half of ALL referrals start within {tts['median_days']:.0f} days (KM); among those who started, the median is "
          f"{methods.naive_median_days(obs):.0f} days - two different questions, label which one you report")
    dec = obs["referral_dt"] >= pd.Timestamp(cfg.as_of).replace(day=1)
    km_dec = methods.time_to_start(obs[dec], cfg.as_of, (30,))
    true_dec = (truth.loc[dec, "soc_dt"] - truth.loc[dec, "referral_dt"]).dt.days
    print(f"  most recent month's referrals (n={int(dec.sum())}, still open at the cut): naive 'started' share "
          f"{obs.loc[dec, 'soc_dt'].notna().mean():.0%} vs KM P(started by day 30) {km_dec['by_day'][30]:.0%} vs TRUE {(true_dec <= 30).mean():.0%}; "
          f"naive median among started {methods.naive_median_days(obs[dec]):.0f} days vs eventual {true_dec.median():.1f}")

    pa = methods.pa_metrics(obs, cfg.expedited_target_hours, cfg.standard_target_hours)
    print(f"\n== Prior authorization (targets: expedited {cfg.expedited_target_hours}h, standard {cfg.standard_target_hours}h per CMS-0057-F) ==")
    print(pa.round(3).to_string(index=False))
    by_payer = obs.dropna(subset=["pa_decision_ts"]).assign(
        hours=lambda x: (x["pa_decision_ts"] - x["pa_request_ts"]).dt.total_seconds() / 3600).groupby(
        ["payer_cd", "pa_priority_cd"])["hours"].median().unstack().round(1)
    print("median decision hours by payer:\n" + by_payer.to_string())

    coh = methods.cohort_conversion(obs, cfg.as_of, cfg.cohort_window_days)
    print(f"\n== ALTERNATIVE: monthly cohorts, started within {cfg.cohort_window_days} days (immature cohorts blank) ==")
    print(coh.assign(cohort=coh["cohort"].astype(str)).round(3).to_string(index=False))
    par = methods.denial_pareto(obs)
    print("\n== ALTERNATIVE: denial reasons (Pareto) ==")
    print(par.round(3).to_string(index=False))

    print("\n== Checks ==")
    fnd = (checks.check_turnaround(pa) + checks.check_immature_cohorts(coh) + checks.check_timestamp_order(obs)
           + checks.check_overturns(pa) + checks.check_open_cases(tts["censored_share"]))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    f_mat.to_csv(OUT / "funnel_mature.csv", index=False)
    tts["curve"].to_csv(OUT / "km_time_to_start.csv", index=False)
    pa.to_csv(OUT / "pa_metrics.csv", index=False)
    coh.to_csv(OUT / "cohort_conversion.csv", index=False)
    par.to_csv(OUT / "denial_pareto.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    km = methods.km_curve(np.array([2, 3, 4, 5]), np.array([1, 0, 1, 0]))
    pa = pd.DataFrame({"pa_request_ts": pd.to_datetime(["2025-01-01 08:00"] * 4),
                       "pa_decision_ts": pd.to_datetime(["2025-01-03 08:00", "2025-01-05 08:00", "2025-01-06 08:00", "2025-01-09 09:00"]),
                       "pa_priority_cd": ["expedited", "expedited", "standard", "standard"],
                       "pa_decision_cd": ["approved", "denied", "approved", "denied"], "appeal_flag": [0, 1, 0, 0],
                       "overturn_flag": [0, 1, 0, 0]})
    m = methods.pa_metrics(pa).set_index("priority")
    tests = {
        "KM: S(2) = 0.75, S(4) = 0.375": lambda: km["survival"].round(3).tolist() == [0.75, 0.375],
        "expedited: 48h within 72h, 96h not -> 50%": lambda: m.loc["expedited", "pct_within_target"] == 0.5,
        "standard: 120h within 168h, 193h not -> 50%": lambda: m.loc["standard", "pct_within_target"] == 0.5,
        "expedited overturn rate = 1/1": lambda: m.loc["expedited", "pct_overturned_of_appealed"] == 1.0,
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
