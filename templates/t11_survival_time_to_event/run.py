"""
t11 — Survival / time-to-event: KM, log-rank, Cox + PH diagnostics, PHReg, discrete-time hazard, immortal time.
Synthetic data with KNOWN hazard ratios.

    python run.py              demo -> outputs/
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

from survival_time_to_event import checks, data, methods  # noqa: E402
from survival_time_to_event.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    df = data.generate(cfg.n_members, cfg.max_follow_days, cfg.true_hr_program, cfg.acuity_hr_early, cfg.acuity_hr_late,
                       cfg.acuity_change_day, cfg.seed)
    print(f"Cohort: {len(df):,} discharged members; readmissions {int((df['event_cd'] == 1).sum())}, deaths "
          f"{int((df['event_cd'] == 2).sum())}, censored {int((df['event_cd'] == 0).sum())}")
    print(f"TRUE HRs: program {cfg.true_hr_program}, frailty 1.65/SD, age 1.16/10y (1.015/yr), caregiver 0.82, "
          f"high acuity {cfg.acuity_hr_early} (days 0-{cfg.acuity_change_day}) then {cfg.acuity_hr_late}\n")

    km = methods.km_table(df, "program_flag", cfg.report_days)
    print("== STANDARD: Kaplan-Meier cumulative readmission (1 - S(t)) ==")
    print(km.round(3).to_string(index=False))
    overall90 = methods.km_table(df, None, (90,))["cum_incidence"].iloc[0]
    print(f"naive 'readmitted by day 90' share (ignores censoring): {methods.naive_event_share(df, 90):.1%} vs KM {overall90:.1%}")
    lr = methods.logrank(df)
    print(f"log-rank program vs usual care: chi2 {lr['test_statistic']:.1f}, p = {lr['p_value']:.2g}")

    cph, hr = methods.cox_ph(df)
    ph = methods.ph_test(cph, df)
    print("\n== STANDARD: Cox PH (lifelines) + Schoenfeld PH test ==")
    print(hr.merge(ph.rename(columns={"p_value": "ph_p_value"})[["covariate", "ph_p_value"]], on="covariate").round(4).to_string(index=False))
    sm_hr = methods.cox_statsmodels(df)
    print(f"ALTERNATIVE statsmodels PHReg: max |HR difference| vs lifelines = {np.abs(sm_hr['hr'].to_numpy() - hr['hr'].to_numpy()).max():.1e}")
    dt = methods.discrete_time_hazard(df, interval_days=cfg.interval_days, max_days=cfg.max_follow_days)
    print("\n== ALTERNATIVE: discrete-time hazard (30-day periods); acuity effect allowed to change ==")
    print(dt.round(3).to_string(index=False))
    rm = methods.rmst_difference(df, cfg.rmst_horizon)
    print(f"\nRMST to day {cfg.rmst_horizon}: program {rm['rmst_treated']:.1f} vs usual care {rm['rmst_control']:.1f} "
          f"readmission-free days (+{rm['difference_days']:.1f})")

    it = data.immortal_time_cohort(seed=cfg.seed)
    naive = methods.cox_naive_ever_exposed(it)
    tv = methods.cox_time_varying(it)
    print("\n== Immortal time: program starts 0-45 days after discharge; TRUE program HR = 1.0 (it does nothing) ==")
    print(f"WRONG 'ever vs never' Cox HR {naive:.2f}   <- program members had to stay out of hospital until they started")
    print(f"RIGHT time-varying exposure  HR {tv['hr']:.2f} (95% CI {tv['ci_lo']:.2f} to {tv['ci_hi']:.2f})")

    print("\n== Checks ==")
    fnd = (checks.check_proportional_hazards(ph) + checks.check_censoring(df, cfg.rmst_horizon)
           + checks.check_events_per_variable(int(df["event_flag"].sum()), len(methods.COVARIATES))
           + checks.check_competing_events(df) + checks.check_immortal_time(it))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    km.to_csv(OUT / "km_cumulative_incidence.csv", index=False)
    hr.to_csv(OUT / "cox_hazard_ratios.csv", index=False)
    ph.to_csv(OUT / "ph_test.csv", index=False)
    dt.to_csv(OUT / "discrete_time_hazard.csv", index=False)
    pd.DataFrame([{"design": "ever_vs_never (wrong)", "hr": naive}, {"design": "time_varying", **tv}]).to_csv(
        OUT / "immortal_time.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    warnings.filterwarnings("ignore")
    # hand KM: events at days 2 and 4, censored at 3 and 5 -> S(2) = 3/4, S(4) = 3/4 x 1/2 = 0.375
    toy = pd.DataFrame({"member_id": list("ABCD"), "duration_days": [2, 3, 4, 5], "event_flag": [1, 0, 1, 0]})
    km = methods.km_table(toy, None, (2, 4))
    pp = methods.person_period(toy.assign(x=1), interval_days=2, max_days=6)
    tests = {
        "KM: 1 - S(2) = 0.25": lambda: abs(km["cum_incidence"].iloc[0] - 0.25) < 1e-12,
        "KM: 1 - S(4) = 0.625": lambda: abs(km["cum_incidence"].iloc[1] - 0.625) < 1e-12,
        "naive share by day 4 = 2/4 (understates 0.625)": lambda: methods.naive_event_share(toy, 4) == 0.5,
        "person-period: A (event day 2) = 1 row, event": lambda: pp[pp.member_id == "A"]["event_in_period"].tolist() == [1],
        "person-period: D (censored day 5) = 3 rows, no event": lambda: pp[pp.member_id == "D"]["event_in_period"].tolist() == [0, 0, 0],
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
