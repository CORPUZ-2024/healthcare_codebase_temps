"""
t10 — Causal impact evaluation: DiD + event study, PSM, IPW / AIPW, interrupted time series.
Synthetic data with KNOWN effects.

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

from causal_impact_evaluation import checks, data, methods  # noqa: E402
from causal_impact_evaluation.config import Config  # noqa: E402

OUT = HERE / "outputs"


def _ci(r: dict, k: str = "effect") -> str:
    return f"{r[k]:7.2f}  (95% CI {r['ci_lo']:.2f} to {r['ci_hi']:.2f})"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    fnd = []

    # --- A. Panel: DiD + event study -----------------------------------------------------------------
    pn = data.practice_panel(cfg.n_practices, cfg.n_months, cfg.adopt_month, cfg.did_effect, seed=cfg.seed)
    did = methods.did_twfe(pn["df"])
    es, p_pre = methods.event_study(pn["df"], window=cfg.event_window)
    bad = data.practice_panel(cfg.n_practices, cfg.n_months, cfg.adopt_month, cfg.did_effect, pre_trend=0.08, seed=cfg.seed)
    did_bad = methods.did_twfe(bad["df"])
    _, p_bad = methods.event_study(bad["df"], window=cfg.event_window)
    print(f"== A. Workflow rolled out in {cfg.n_practices // 2} of {cfg.n_practices} practices (ED visits / 1,000 MM; TRUE effect {pn['effect']}) ==")
    print(f"naive post-period comparison       {methods.naive_post_comparison(pn['df']):7.2f}   <- includes the pre-existing gap")
    print(f"STANDARD two-way FE DiD            {_ci(did)}  clusters {did['n_clusters']}")
    print(f"event-study pre-trend test         p = {p_pre:.2f}  (parallel trends hold)")
    print(es.round(2).to_string(index=False))
    verdict = "caught" if p_bad < 0.05 else "MISSED on this draw (it catches this drift ~94% of the time in simulation)"
    print(f"same design + treated drifting 0.08/month before adoption: pre-trend p = {p_bad:.3f} -> {verdict}; "
          f"DiD {did_bad['effect']:.2f} vs true {pn['effect']}  <- biased toward 0. A passed pre-trend test is not proof.")
    fnd += checks.check_pre_trends(p_bad) + checks.check_few_clusters(did["n_clusters"])

    # --- B. Cross-section: PSM, IPW, AIPW ------------------------------------------------------------
    cs = data.cross_section(cfg.n_members, cfg.seed)
    df = cs["df"]
    ps = methods.propensity_logit(df)
    psm, matched = methods.psm_att(df, ps, cfg.caliper_sd, seed=cfg.seed)
    att_matched_truth = float(matched.loc[matched["treated_flag"] == 1, "tau"].mean())
    ipw_ate, w_ate = methods.ipw(df, estimand="ate", trim=cfg.trim_pct)
    ipw_att, w_att = methods.ipw(df, estimand="att", trim=cfg.trim_pct)
    aipw = methods.aipw_ate(df, trim=cfg.trim_pct)
    print(f"\n== B. Referral program, {cfg.n_members:,} members ({df['treated_flag'].mean():.0%} referred); outcome = utilization index ==")
    print(f"TRUTH  ATE {cs['ate']:.2f} | ATT {cs['att']:.2f} | ATT among matched participants {att_matched_truth:.2f}")
    print(f"naive treated - untreated          {methods.naive_difference(df):7.2f}   <- wrong SIGN: sicker members are referred")
    print(f"STANDARD PSM (ATT, matched)        {psm['att']:7.2f}  (95% CI {psm['ci_lo']:.2f} to {psm['ci_hi']:.2f}); "
          f"{psm['share_treated_matched']:.0%} of participants matched")
    print(f"ALTERNATIVE IPW (ATE)              {_ci(ipw_ate)}")
    print(f"ALTERNATIVE IPW (ATT)              {_ci(ipw_att)}")
    print(f"ALTERNATIVE AIPW (ATE, doubly robust) {_ci(aipw)}")
    print(f"outcome regression only (linear)   {methods.outcome_regression_ate(df):7.2f}   <- misses the non-linear prior-use term")
    smd = methods.smd_table(df, methods.COVARIATES, weights=w_ate, matched=matched)
    print(smd.round(3).to_string(index=False))
    fnd += (checks.check_balance(smd, "smd_matched") + checks.check_balance(smd, "smd_weighted")
            + checks.check_overlap(ps, df["treated_flag"]) + checks.check_extreme_weights(w_att))

    # --- C. Interrupted time series ------------------------------------------------------------------
    its = data.its_series(cfg.its_months, cfg.its_break, cfg.its_level_change, cfg.its_slope_change, seed=cfg.seed)
    r = methods.its_segmented(its["df"], se_method=cfg.its_se_method, hac_lags=cfg.hac_lags)
    print(f"\n== C. Statewide policy at month {cfg.its_break} (no control group): TRUE level {its['level_change']}, slope {its['slope_change']} ==")
    print(f"ALTERNATIVE ITS level change {r['level_change']:.2f} ({r['level_lo']:.2f} to {r['level_hi']:.2f}); "
          f"slope change {r['slope_change']:.3f} ({r['slope_lo']:.3f} to {r['slope_hi']:.3f}) [{r['se_method']}]")
    cov_l = r["level_lo"] <= its["level_change"] <= r["level_hi"]
    cov_s = r["slope_lo"] <= its["slope_change"] <= r["slope_hi"]
    print(f"intervals cover the truth on this draw: level {'yes' if cov_l else 'NO'}, slope {'yes' if cov_s else 'NO'} "
          f"(GLSAR covers ~90% of draws, Newey-West ~75% - see README)")
    print(f"SE of level change: OLS {r['se_level_ols']:.2f} | Newey-West {r['se_level_hac']:.2f} | GLSAR {r['se_level_glsar']:.2f}; "
          f"AR(1) rho {r['rho']:.2f}, Durbin-Watson {r['durbin_watson']:.2f}")
    fnd += checks.check_its_autocorrelation(r["durbin_watson"], "hac") + checks.check_its_points(cfg.its_break, cfg.its_months - cfg.its_break)

    print("\n== Checks ==  (CAU-003 fires only if the drifting-trend pre-test rejects; CAU-006 shows why HAC was not used)")
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    pd.DataFrame([{"design": "DiD", "estimand": "effect on adopting practices", **did},
                  {"design": "PSM", "estimand": "ATT (matched)", "effect": psm["att"], **{k: psm[k] for k in ("se", "ci_lo", "ci_hi")}},
                  {"design": "IPW", **ipw_ate}, {"design": "IPW", **ipw_att}, {"design": "AIPW", **aipw},
                  {"design": "ITS", "estimand": "level change", "effect": r["level_change"], "ci_lo": r["level_lo"], "ci_hi": r["level_hi"]}]) \
        .to_csv(OUT / "effect_estimates.csv", index=False)
    es.to_csv(OUT / "event_study.csv", index=False)
    smd.to_csv(OUT / "balance_smd.csv", index=False)
    r["counterfactual"].to_csv(OUT / "its_counterfactual.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    warnings.filterwarnings("ignore")
    panel = pd.DataFrame({"practice_id": ["A", "A", "B", "B", "C", "C", "D", "D"], "month_idx": [0, 1] * 4,
                          "treated_flag": [1, 1, 1, 1, 0, 0, 0, 0], "post_flag": [0, 1] * 4,
                          "y": [10.0, 13.0, 12.0, 15.0, 10.0, 11.0, 14.0, 15.0]})
    did = methods.did_twfe(panel)
    tests = {
        "2x2 DiD = (3 + 3)/2 - (1 + 1)/2 = 2": lambda: abs(did["effect"] - 2.0) < 1e-9,
        "Durbin-Watson of alternating residuals = 3": lambda: abs(methods.durbin_watson(np.array([1.0, -1, 1, -1])) - 3) < 1e-12,
        "naive difference of a 2-row toy": lambda: methods.naive_difference(pd.DataFrame({"treated_flag": [1, 0], "outcome": [5.0, 2.0]})) == 3.0,
        "no-confounding data: AIPW ~ true effect -2": lambda: abs(methods.aipw_ate(_rct(), ["x"], ["x"])["effect"] + 2) < 0.2,
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


def _rct(n: int = 4_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = rng.integers(0, 2, n)
    return pd.DataFrame({"x": x, "treated_flag": t, "outcome": 3 * x - 2 * t + rng.normal(0, 1, n)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else (demo(Config()) or 0))
