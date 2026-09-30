"""
t05 — Total cost of care, PMPM, trend decomposition, MLR, episodes.

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from tcoc_pmpm_mlr import checks, data, methods, prep  # noqa: E402
from tcoc_pmpm_mlr.config import Config  # noqa: E402

OUT = HERE / "outputs"


def build(cfg: Config):
    """Generate data and return (claims with service_category + month, member_months by lob)."""
    u = data.generate_universe(cfg.n_members, cfg.seed, cfg.start, cfg.months)
    claims = methods.attach_month(prep.service_category_claim(prep.collapse_versions_latest(u["medical"])))
    end = (pd.Timestamp(cfg.start) + pd.DateOffset(months=cfg.months) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    mm = prep.member_months_daily(u["enrollment"], cfg.start, end, by=("member_id", "lob_cd"))
    return claims, mm


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    claims, mm_all = build(cfg)
    mm = methods.complete_months(claims, mm_all, cfg.runout_lag_months)
    claims = claims[claims["month"].isin(set(mm["month"]))]
    print(f"Analysis months: {mm['month'].min()} .. {mm['month'].max()} "
          f"(last {cfg.runout_lag_months} excluded for runout)")

    total = methods.pmpm_ratio_of_sums(claims, mm)
    by_lob = methods.pmpm_ratio_of_sums(claims, mm, by=["lob_cd"])
    by_cat = methods.pmpm_ratio_of_sums(claims, mm, by=["service_category"])
    print(f"\n== TCOC PMPM (STANDARD ratio of sums): ${total['pmpm'].iloc[0]:,.2f}")
    print(by_lob.round(2).to_string(index=False))
    print(by_cat.round(2).to_string(index=False))
    alt = methods.pmpm_mean_of_members(claims, mm, min_member_months=3)
    print(f"ALTERNATIVE mean-of-members PMPM: ${alt['mean']:,.2f} "
          f"(95% CI {alt['ci_lo']:,.0f}-{alt['ci_hi']:,.0f}; median ${alt['median']:,.2f}; n={alt['n_members']})")

    mc = methods.member_period_cost(claims)
    tp = methods.truncate_percentile(mc, cfg.truncation_pct)
    ta = methods.truncate_fixed_attachment(mc, cfg.attachment_amt)
    mmt = mm["member_months"].sum()
    print(f"\n== Truncation == untruncated ${mc.sum() / mmt:,.2f} | p{cfg.truncation_pct * 100:.0f} cap "
          f"${tp['cap_amt'].iloc[0]:,.0f} -> ${tp['truncated_amt'].sum() / mmt:,.2f} | "
          f"${cfg.attachment_amt:,.0f} attachment -> ${ta['truncated_amt'].sum() / mmt:,.2f}")

    y1 = claims[claims["month"].dt.year == 2023]
    y2 = claims[claims["month"].dt.year == 2024]
    u1 = methods.utilization_unit_cost(y1, mm[mm["month"].dt.year == 2023]).set_index("service_category")
    u2 = methods.utilization_unit_cost(y2, mm[mm["month"].dt.year == 2024]).set_index("service_category")
    rows = []
    for cat in u1.index.intersection(u2.index):
        a = methods.decompose_additive(u1.loc[cat], u2.loc[cat])
        g = methods.decompose_log(u1.loc[cat], u2.loc[cat])
        rows.append({"service_category": cat, **{k: round(v, 2) for k, v in a.items()},
                     "total_trend_pct": round(g["total_trend_pct"] * 100, 1),
                     "util_trend_pct": round(g["utilization_trend_pct"] * 100, 1),
                     "unit_cost_trend_pct": round(g["unit_cost_trend_pct"] * 100, 1)})
    decomp = pd.DataFrame(rows)
    print("\n== 2023 -> 2024 PMPM change: additive (STANDARD) and log trend (ALTERNATIVE) ==")
    print(decomp.to_string(index=False))

    premium = data.generate_premium(mm)
    prem = premium["premium_amt"].sum()
    reg = methods.mlr_regulatory(claims["paid_amt"].sum(), cfg.qi_expense_pct_premium * prem, prem,
                                 cfg.taxes_fees_pct_premium * prem, cfg.mlr_minimum)
    simple = methods.mlr_simple(claims["paid_amt"].sum(), prem)
    print(f"\n== MLR == regulatory {reg['mlr']:.1%} (min {reg['minimum']:.0%}, rebate/remittance "
          f"${reg['rebate_or_remittance']:,.0f}) | simple loss ratio {simple['loss_ratio']:.1%}")
    pm_prem = prem / mmt
    for fee_as in ("claims", "admin"):
        imp = methods.mlr_impact(total["pmpm"].iloc[0], pm_prem, cfg.program_savings_pmpm, cfg.program_fee_pmpm, fee_as)
        print(f"   program saves ${cfg.program_savings_pmpm}/fee ${cfg.program_fee_pmpm} PMPM, fee as {fee_as:6}: "
              f"MLR {imp['mlr_before']:.1%} -> {imp['mlr_after']:.1%} ({imp['mlr_change_pts']:+.1f} pts)")

    ep = methods.build_episodes_nonoverlap(claims, cfg.episode_pre_days, cfg.episode_post_days)
    eo = methods.build_episodes_overlap(claims, cfg.episode_pre_days, cfg.episode_post_days)
    print(f"\n== IP episodes ({cfg.episode_pre_days}d pre / {cfg.episode_post_days}d post) == "
          f"non-overlap {len(ep)} episodes, mean ${ep['episode_paid_amt'].mean():,.0f} | "
          f"overlap {len(eo)} episodes, summed ${eo['episode_paid_amt'].sum():,.0f} vs ${ep['episode_paid_amt'].sum():,.0f}")

    print("\n== Checks ==")
    for f in (checks.check_claims_without_exposure(claims, mm) + checks.check_high_cost_concentration(mc)
              + checks.check_small_exposure(by_lob)):
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    for name, df in {"pmpm_total": total, "pmpm_by_lob": by_lob, "pmpm_by_service_category": by_cat,
                     "trend_decomposition": decomp, "episodes_nonoverlap": ep}.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    one_mm = pd.DataFrame({"member_id": ["A", "B"], "month": pd.PeriodIndex(["2025-01", "2025-01"], freq="M"),
                           "member_months": [1.0, 0.5], "lob_cd": ["MCD", "MCD"]})
    one_c = pd.DataFrame({"member_id": ["A", "B"], "month": pd.PeriodIndex(["2025-01", "2025-01"], freq="M"),
                          "paid_amt": [300.0, 150.0]})
    base = {"util_per_1000": 100.0, "cost_per_unit": 1_200.0}
    cur = {"util_per_1000": 110.0, "cost_per_unit": 1_260.0}
    d = methods.decompose_additive(base, cur)
    tests = {
        "PMPM ratio of sums = 450 / 1.5 = 300": lambda: methods.pmpm_ratio_of_sums(one_c, one_mm)["pmpm"].iloc[0] == 300.0,
        "additive decomposition adds up": lambda: abs(d["utilization"] + d["unit_cost"] + d["interaction"] - d["pmpm_change"]) < 1e-9,
        "log trend: 1.10 x 1.05 - 1 = 15.5%": lambda: abs(methods.decompose_log(base, cur)["total_trend_pct"] - 0.155) < 1e-9,
        "MLR 0.80 -> remittance 5% of net premium": lambda: abs(methods.mlr_regulatory(80, 0, 100, 0)["rebate_or_remittance"] - 5) < 1e-9,
        "percentile cap never raises cost": lambda: (methods.truncate_percentile(pd.Series([1, 2, 3, 1000.0], index=list("abcd")))["truncated_amt"] <= [1, 2, 3, 1000]).all(),
        "fee as admin lowers MLR more than fee as claims": lambda: methods.mlr_impact(400, 500, 25, 15, "admin")["mlr_after"] < methods.mlr_impact(400, 500, 25, 15, "claims")["mlr_after"],
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
