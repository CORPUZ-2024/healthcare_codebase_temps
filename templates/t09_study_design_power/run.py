"""
t09 — Study design: power, MDE, clustering, randomization, SAP (no data needed).

    python run.py              demo -> outputs/ (power tables, randomization list, SAP_draft.md)
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from study_design_power import checks, methods  # noqa: E402
from study_design_power.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo_units(n: int = 400, seed: int = 9) -> pd.DataFrame:
    """FAKE enrollment list for the randomization demo: member_id, site_cd, risk_tier."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"member_id": [f"E{i:04d}" for i in range(1, n + 1)],
                         "site_cd": rng.choice(["NORTH", "SOUTH", "EAST"], n, p=[0.5, 0.3, 0.2]),
                         "risk_tier": rng.choice(["HIGH", "MEDIUM"], n, p=[0.4, 0.6])})


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    print("Scenario (FAKE): transitional-care program; primary endpoint 30-day readmission.\n")
    prim = methods.power_two_proportions(cfg.p_control, cfg.p_treat, cfg.alpha, cfg.power)
    print(f"== Primary: readmission {cfg.p_control:.0%} -> {cfg.p_treat:.0%} (STANDARD: statsmodels, Cohen's h {prim['effect_size_h']:.3f}) ==")
    print(f"n per arm {prim['n_per_arm']:,} (total {prim['n_total']:,}) for {cfg.power:.0%} power at alpha {cfg.alpha}")
    sim = methods.simulate_power(methods.sim_two_proportions(cfg.p_control, cfg.p_treat, prim["n_per_arm"]), cfg.n_sims, cfg.alpha, cfg.seed)
    print(f"ALTERNATIVE simulation at that n: power {sim['power']:.1%} +/- {1.96 * sim['mc_se']:.1%}")
    curve = methods.power_curve([400, 800, 1_200, 1_600, 2_000], lambda n: methods.power_two_proportions(
        cfg.p_control, cfg.p_treat, cfg.alpha, None, n)["power"])
    print(curve.assign(power=lambda d: d["power"].round(3)).to_string(index=False))
    mde = methods.mde_two_proportions(cfg.p_control, 800, cfg.alpha, cfg.power)
    print(f"MDE with only 800 per arm: {cfg.p_control:.0%} -> {mde:.1%} (a {1 - mde / cfg.p_control:.0%} relative reduction)")

    print(f"\n== Cluster randomization: practices of {cfg.cluster_size} patients, ICC {cfg.icc} ==")
    cl = methods.cluster_sample_size(prim["n_per_arm"], cfg.cluster_size, cfg.icc)
    print(f"design effect {cl['design_effect']:.2f} -> {cl['clusters_per_arm']} practices per arm ({cl['n_per_arm']:,} patients per arm)")
    ok = methods.simulate_power(methods.sim_cluster_proportions(cfg.p_control, cfg.p_treat, cl["clusters_per_arm"], cfg.cluster_size, cfg.icc),
                                cfg.n_sims, cfg.alpha, cfg.seed)
    naive_k = int(np.ceil(prim["n_per_arm"] / cfg.cluster_size))
    bad = methods.simulate_power(methods.sim_cluster_proportions(cfg.p_control, cfg.p_treat, naive_k, cfg.cluster_size, cfg.icc),
                                 cfg.n_sims, cfg.alpha, cfg.seed)
    print(f"simulated power: {ok['power']:.0%} with {cl['clusters_per_arm']} practices/arm vs. {bad['power']:.0%} if DEFF is ignored ({naive_k}/arm)")

    sec = methods.power_two_means(cfg.burden_delta, cfg.burden_sd, cfg.alpha, cfg.power)
    print(f"\n== Secondary: caregiver burden, difference {cfg.burden_delta} points (SD {cfg.burden_sd}) -> n per arm {sec['n_per_arm']} ==")

    sd_change = cfg.cost_cv * cfg.cost_pmpm * np.sqrt(2 * (1 - 0.5))
    analytic = methods.power_two_means(cfg.cost_effect_pct * cfg.cost_pmpm, sd_change, cfg.alpha, cfg.power)
    print(f"\n== Cost: {cfg.cost_effect_pct:.0%} lower PMPM on ${cfg.cost_pmpm:,.0f} (CV {cfg.cost_cv}), DiD on change scores ==")
    print(f"analytic (normal) n per arm: {analytic['n_per_arm']:,}")
    grid = []
    for n in (analytic["n_per_arm"] // 2, analytic["n_per_arm"], int(analytic["n_per_arm"] * 1.5)):
        s = methods.simulate_power(methods.sim_cost_did(cfg.cost_pmpm, cfg.cost_cv, cfg.cost_effect_pct, n), 400, cfg.alpha, cfg.seed)
        grid.append({"n_per_arm": n, "simulated_power": s["power"]})
        print(f"  simulated power at n={n:,}: {s['power']:.0%}")
    print("-> cost endpoints need thousands per arm; readmission is usually the better primary endpoint.")

    units = demo_units(400, cfg.seed)
    rnd = methods.stratified_block_randomize(units, ["site_cd", "risk_tier"], (2, 4), seed=cfg.seed)
    bal = rnd.groupby(["site_cd", "risk_tier", "arm"]).size().unstack()
    print("\n== Stratified permuted-block randomization (block sizes 2/4) ==")
    print(bal.to_string())

    attr = 0.10
    values = dict(
        study_title="Transitional care for high-risk discharges (FAKE)", version="0.1", date=str(dt.date.today()),
        author="Analytics", objective="Estimate the effect of a 30-day transitional-care program on 30-day all-cause readmission.",
        design_type="Pragmatic, cluster-randomized (primary-care practice), parallel, 1:1", unit_of_randomization="practice",
        strata="region x practice size", block_sizes="2 and 4", arms="usual care; transitional care", follow_up="30 days post-discharge (primary); 6 months (cost)",
        primary_endpoint="30-day readmission", primary_definition="any inpatient admission 1-30 days after index discharge (t01 definition)",
        primary_measure="risk difference", secondary_endpoint="caregiver burden", secondary_definition="FAKE 0-88 burden score at 30 days",
        secondary_measure="mean difference", exploratory_endpoint="6-month PMPM", exploratory_definition="paid claims PMPM, 6 months (t05/t06)",
        exploratory_measure="DiD in PMPM",
        sample_size_text=(f"To detect a reduction from {cfg.p_control:.0%} to {cfg.p_treat:.0%} (two-sided alpha {cfg.alpha}, power "
                          f"{cfg.power:.0%}) an individually randomized trial needs {prim['n_per_arm']:,} per arm. Randomizing practices "
                          f"of ~{cfg.cluster_size} patients with ICC {cfg.icc} inflates this by the design effect; simulation "
                          f"(cluster-level t test) gives {ok['power']:.0%} power at the planned size."),
        alpha=cfg.alpha, power=f"{cfg.power:.0%}", p_control=f"{cfg.p_control:.0%}", p_treat=f"{cfg.p_treat:.0%}",
        n_individual=f"{prim['n_per_arm']:,}", cluster_size=cfg.cluster_size, icc=cfg.icc, design_effect=f"{cl['design_effect']:.2f}",
        clusters_per_arm=cl["clusters_per_arm"], n_cluster=f"{cl['n_per_arm']:,}", attrition_pct=f"{attr:.0%}",
        n_final=f"{int(np.ceil(cl['n_per_arm'] / (1 - attr))):,}",
        primary_analysis="GEE logistic regression (exchangeable, practice clusters) with robust SE; sensitivity: cluster-level t test",
        secondary_analysis="linear mixed model (practice random intercept), baseline-adjusted (t12)",
        covariates="stratification factors, baseline risk score", estimand="intention-to-treat difference in 30-day readmission risk",
        missing_data="Primary endpoint from claims (complete for enrolled members). Burden: multiple imputation under MAR; tipping-point sensitivity.",
        multiplicity="One primary endpoint at alpha 0.05; secondary endpoints tested hierarchically.",
        subgroups="age 65+, prior-year admissions >= 2, caregiver present", interim="None planned.")
    sap = methods.render_sap(cfg.sap_template, values)
    (OUT / "SAP_draft.md").write_text(sap, encoding="utf-8")

    print("\n== Checks ==")
    fnd = (checks.check_underpowered(methods.power_two_proportions(cfg.p_control, cfg.p_treat, cfg.alpha, None, 800)["power"])
           + checks.check_cluster_design("practice", 1.0) + checks.check_few_clusters(cl["clusters_per_arm"])
           + checks.check_randomization_balance(rnd) + checks.check_multiplicity(1, False) + checks.check_skewed_outcome(cfg.cost_cv))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")
    print("(PWR-001 is for the 800-per-arm scenario; PWR-002 shows what happens if DEFF is left at 1.0)")

    curve.to_csv(OUT / "power_curve_primary.csv", index=False)
    pd.DataFrame(grid).to_csv(OUT / "power_cost_simulated.csv", index=False)
    rnd.to_csv(OUT / "randomization_list.csv", index=False)
    print(f"\nOutputs written to {OUT} (SAP_draft.md, randomization_list.csv, power tables)")


def selftest() -> int:
    tests = {
        "n per arm 20% vs 15% = 903 (Cohen's h)": lambda: methods.power_two_proportions(0.20, 0.15)["n_per_arm"] == 903,
        "n per arm d = 0.5 = 64": lambda: methods.power_two_means(5, 10)["n_per_arm"] == 64,
        "DEFF(40, 0.02) = 1.78": lambda: abs(methods.design_effect(40, 0.02) - 1.78) < 1e-12,
        "MDE at the planned n returns the planned p2": lambda: abs(methods.mde_two_proportions(
            0.20, methods.power_two_proportions(0.20, 0.15)["n_per_arm"]) - 0.15) < 0.001,
        "power at the solved n >= 80%": lambda: methods.power_two_proportions(0.20, 0.15, n_per_arm=903, power=None)["power"] >= 0.80,
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
