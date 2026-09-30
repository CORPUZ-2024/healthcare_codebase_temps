"""
t03 — Predictive risk stratification (6-month hospitalization risk) and acuity tiers.

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from predictive_risk_stratification import checks, data, methods  # noqa: E402
from predictive_risk_stratification.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    df = data.generate(cfg.n_per_cohort, cfg.seed)
    F = data.FEATURES
    train, test = methods.temporal_split(df, cfg.train_year, cfg.test_year)
    Xtr, ytr, Xte, yte = train[F], train["label_ip_6m"], test[F], test["label_ip_6m"]

    epv = methods.events_per_variable(ytr, len(F))
    findings = (checks.check_missing(df, F) + checks.check_leakage(F) + checks.check_epv(epv, cfg.min_epv)
                + checks.check_separation(methods.detect_separation(Xtr, ytr)))
    print(f"Train {cfg.train_year}: n={len(train):,}, events={ytr.sum()} (EPV {epv:.1f}) | "
          f"Test {cfg.test_year}: n={len(test):,}, events={yte.sum()}")
    if any(f.severity == "error" for f in findings):
        for f in findings:
            print(f)
        raise SystemExit("Stopping: fix pre-fit errors first.")

    lr = methods.fit_logistic(Xtr, ytr, cfg.logistic_C)
    gb = methods.fit_gradient_boosting(Xtr, ytr, seed=cfg.seed)
    p_lr, p_gb = methods.predict_proba_any(lr, Xte), methods.predict_proba_any(gb, Xte)
    res = pd.DataFrame({"STANDARD logistic": methods.evaluate(yte, p_lr, cfg.top_pct),
                        "ALTERNATIVE boosting": methods.evaluate(yte, p_gb, cfg.top_pct)}).round(3)
    print("\n== Out-of-time performance ==")
    print(res.to_string())

    cal = methods.calibration_slope_intercept(yte, p_lr)
    findings += checks.check_calibration_drift(cal)
    p_lr_recal = methods.recalibrate_intercept(p_lr, ytr.mean() * 0 + yte.mean())
    print(f"\nLogistic intercept on {cfg.test_year}: {cal['intercept']:+.2f} -> after intercept recalibration "
          f"mean pred {p_lr_recal.mean():.3f} vs observed {yte.mean():.3f}")
    print("\n== Calibration by decile (logistic) ==")
    print(methods.calibration_table(yte, p_lr).round(3).to_string(index=False))

    print("\n== Odds ratios per unit (logistic, top 8) ==")
    print(methods.odds_ratios(lr, F, Xtr).head(8).round(3).to_string(index=False))

    tiers = methods.tiers_by_capacity(test["member_id"], p_lr, cfg.tier_shares)
    tiers["label_ip_6m"] = yte.to_numpy()
    print("\n== Capacity tiers (STANDARD) ==")
    print(tiers.groupby("tier").agg(n=("risk", "size"), mean_risk=("risk", "mean"), observed=("label_ip_6m", "mean")).round(3).to_string())

    # continuous features only: a 0/1 column dominates k-means distances and splits on itself
    seg = methods.segments_kmeans(Xte[["age", "n_chronic", "prior_ip_cnt", "prior_ed_cnt", "pdc_pct",
                                       "hcbs_hours_per_week", "adi_decile"]], seed=cfg.seed)
    print(f"\n== k-means segments (ALTERNATIVE): K={seg['k']} by silhouette {({k: round(v, 3) for k, v in seg['silhouette_by_k'].items()})} ==")
    print(seg["profile"].to_string())

    # Rising risk: re-score the SAME members three months later. Simulated here: 10% of members
    # have a new ED visit and 3% a new admission since the last scoring run.
    rng = np.random.default_rng(cfg.seed)
    later = Xte.copy()
    later["prior_ed_cnt"] += (rng.random(len(later)) < 0.10).astype(int)
    later["prior_ip_cnt"] += (rng.random(len(later)) < 0.03).astype(int)
    now = methods.tiers_by_capacity(test["member_id"], methods.predict_proba_any(lr, later), cfg.tier_shares)
    rising = methods.rising_risk(tiers, now, cfg.rising_risk_increase)
    print(f"\n== Rising risk (re-scored 3 months later, +{cfg.rising_risk_increase:.0%}, not Tier 1): {len(rising)} members")

    print("\n== Checks ==")
    for f in findings:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    res.to_csv(OUT / "model_performance.csv")
    tiers.to_csv(OUT / "member_tiers.csv", index=False)
    seg["profile"].to_csv(OUT / "segment_profiles.csv")
    rising.to_csv(OUT / "rising_risk.csv", index=False)
    methods.odds_ratios(lr, F, Xtr).to_csv(OUT / "odds_ratios.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for p, name, color in ((p_lr, "logistic", "#2a78d6"), (p_gb, "boosting", "#eb6834")):
            t = methods.calibration_table(yte, p)
            ax.plot(t["predicted"], t["observed"], marker="o", markersize=4, linewidth=2, color=color, label=name)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color="#b0afab", linewidth=1, linestyle="--")
        ax.set_xlabel("mean predicted risk (decile)"); ax.set_ylabel("observed rate")
        ax.set_title("Calibration, out-of-time cohort", loc="left", fontsize=11)
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130); plt.close(fig)
    except ImportError:
        pass
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    df = data.generate(1_500, seed=3)
    tr, te = methods.temporal_split(df, 2023, 2024)
    lr = methods.fit_logistic(tr[data.FEATURES], tr["label_ip_6m"])
    p = methods.predict_proba_any(lr, te[data.FEATURES])
    ev = methods.evaluate(te["label_ip_6m"], p)
    t = methods.tiers_by_capacity(te["member_id"], p)
    tests = {
        "EPV = min(class) / predictors": lambda: methods.events_per_variable(pd.Series([1] * 30 + [0] * 70), 3) == 10,
        "AUC better than chance on synthetic data": lambda: ev["auc"] > 0.65,
        "Tier 1 is exactly 5% of members": lambda: (t["tier"] == 1).mean() == 0.05,
        "recalibration hits the target rate": lambda: abs(methods.recalibrate_intercept(p, 0.2).mean() - 0.2) < 1e-6,
        "separation detected": lambda: methods.detect_separation(pd.DataFrame({"x": [0, 0, 1, 1]}), pd.Series([0, 1, 1, 1])) == ["x"],
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
