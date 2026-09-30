"""
t06 — Program ROI / cost offset: matched DiD on PMPM, savings, ROI, break-even, tornado.
Synthetic data with a KNOWN effect; program economics are FAKE.

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from roi_cost_offset import checks, data, methods  # noqa: E402
from roi_cost_offset.config import Config  # noqa: E402

OUT = HERE / "outputs"


def true_att_matched(d: dict, matched: pd.DataFrame, post_months: int) -> float:
    """Truth for the matched participants (known only because the data are synthetic)."""
    p = d["panel"].merge(d["members"][["member_id", "index_month_idx"]], on="member_id")
    p = p[p["member_id"].isin(matched.loc[matched["treated_flag"] == 1, "member_id"])]
    p = p[(p["month_idx"] - p["index_month_idx"]).between(1, post_months)]
    return float((p["paid_cf_amt"] - p["paid_amt"]).sum() / p["member_months"].sum())


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    print("NOTE: synthetic data with a known 20% effect; program fee and costs are FAKE.\n")
    d = data.generate_panel(cfg.n_members, cfg.months, cfg.start, cfg.true_effect_pct, cfg.seed)
    mp = methods.member_periods(d["panel"], d["members"], cfg.pre_months, cfg.post_months, cfg.min_months_each_period)
    ps = methods.propensity_scores(mp)
    matched = methods.match_propensity(mp, ps, cfg.caliper_sd, cfg.exact_on, cfg.seed)
    bal = methods.balance_table(mp, matched)
    n_t = int(mp["treated_flag"].sum())
    print(f"== Matching == {n_t:,} participants, {len(mp) - n_t:,} potential controls -> {matched['pair_id'].nunique():,} pairs")
    print(bal.round(3).to_string(index=False))

    did = methods.did_pmpm(matched)
    breakeven = methods.roi(0.0, cfg.program_fee_pmpm, cfg.one_time_cost, cfg.participants,
                            cfg.months_in_program)["breakeven_savings_pmpm"]
    boot = methods.bootstrap_did(matched, cfg.n_boot, cfg.seed, breakeven_savings_pmpm=breakeven)
    tp = methods.two_part_did(matched)
    truth = true_att_matched(d, matched, cfg.post_months)
    naive = methods.naive_pre_post(mp)
    print("\n== Effect on PMPM (negative = savings) ==")
    print(f"TRUTH (synthetic only)            {-truth:9.1f}")
    print(f"naive participant pre/post        {naive:9.1f}   <- regression to the mean + trend; do not report")
    print(f"STANDARD matched DiD              {did['did_pmpm']:9.1f}   95% CI {did['ci_lo']:.1f} to {did['ci_hi']:.1f} (cluster-robust)")
    print(f"ALTERNATIVE pair bootstrap CI     {boot['did_pmpm']:9.1f}   95% CI {boot['ci_lo']:.1f} to {boot['ci_hi']:.1f}; "
          f"P(savings) = {boot['prob_savings']:.1%}, P(>= break-even {breakeven:.0f}) = {boot['prob_breakeven']:.1%}")
    print(f"ALTERNATIVE two-part (logit x Gamma) {tp['did_pmpm']:6.1f}   any-cost OR {tp['part1_odds_ratio']:.2f}, "
          f"cost-if-any ratio {tp['part2_cost_ratio']:.2f}")
    print(f"cells: participants {did['treated_pre']:.0f} -> {did['treated_post']:.0f}; controls {did['control_pre']:.0f} -> {did['control_post']:.0f}")
    pt = methods.pre_trend_test(d["panel"], matched, cfg.pre_months)
    print(f"pre-trend slope difference {pt['slope_diff_pmpm_per_month']:.1f} PMPM/month (p = {pt['p_value']:.2f})")

    base = dict(savings_pmpm=-did["did_pmpm"], program_fee_pmpm=cfg.program_fee_pmpm, one_time_cost=cfg.one_time_cost,
                participants=cfg.participants, months_in_program=cfg.months_in_program)
    r = methods.roi(**base)
    r_lo = methods.roi(**{**base, "savings_pmpm": -did["ci_hi"]})
    r_hi = methods.roi(**{**base, "savings_pmpm": -did["ci_lo"]})
    print(f"\n== ROI ({cfg.participants} participants x {cfg.months_in_program} months) ==")
    print(f"gross savings ${r['gross_savings']:,.0f} - program cost ${r['program_cost']:,.0f} = net ${r['net_savings']:,.0f}; "
          f"ROI {r['roi']:.2f} (CI {r_lo['roi']:.2f} to {r_hi['roi']:.2f}); break-even savings {r['breakeven_savings_pmpm']:.2f} PMPM")

    ranges = {"savings_pmpm": (-did["ci_hi"], -did["ci_lo"]), "program_fee_pmpm": (cfg.program_fee_pmpm * 0.8, cfg.program_fee_pmpm * 1.2),
              "months_in_program": (6, 18), "participants": (int(cfg.participants * 0.7), int(cfg.participants * 1.3)),
              "one_time_cost": (cfg.one_time_cost * 0.5, cfg.one_time_cost * 2)}
    tor = methods.tornado(base, ranges)
    print(f"\n== Tornado: net savings (base ${tor.attrs['base']:,.0f}) ==")
    width = tor["swing"].max()
    for row in tor.itertuples():
        bar = "#" * max(1, int(40 * row.swing / width))
        print(f"{row.input:18s} {row.metric_at_low:>13,.0f} .. {row.metric_at_high:>13,.0f}  {bar}")

    print("\n== Checks ==")
    for f in (checks.check_balance(bal) + checks.check_common_support(n_t, did["n_pairs"])
              + checks.check_pre_trends(pt) + checks.check_regression_to_mean(mp)
              + checks.check_effect_precision(did, r["breakeven_savings_pmpm"])):
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    bal.to_csv(OUT / "balance_table.csv", index=False)
    matched.to_csv(OUT / "matched_members.csv", index=False)
    pd.DataFrame([{"method": "naive_pre_post", "did_pmpm": naive}, {"method": "matched_did", **did},
                  {"method": "pair_bootstrap", **boot}, {"method": "two_part", **tp},
                  {"method": "TRUTH_synthetic", "did_pmpm": -truth}]).to_csv(OUT / "effect_estimates.csv", index=False)
    pd.DataFrame([{"scenario": "point", **r}, {"scenario": "ci_low_savings", **r_lo}, {"scenario": "ci_high_savings", **r_hi}]) \
        .to_csv(OUT / "roi.csv", index=False)
    tor.to_csv(OUT / "tornado.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    r = methods.roi(60.0, 25.0, 150.0, 100, 12)
    mp = pd.DataFrame({"member_id": list("ABCD"), "treated_flag": [1, 1, 0, 0], "pair_id": [0, 1, 0, 1],
                       "pre_paid_amt": [600.0, 1200.0, 300.0, 600.0], "pre_mm": [6.0] * 4,
                       "post_paid_amt": [300.0, 600.0, 360.0, 720.0], "post_mm": [6.0] * 4})
    mp["pre_pmpm"], mp["post_pmpm"] = mp["pre_paid_amt"] / 6, mp["post_paid_amt"] / 6
    did = methods.did_pmpm(mp)
    tor = methods.tornado(dict(savings_pmpm=60.0, program_fee_pmpm=25.0, one_time_cost=150.0, participants=100,
                               months_in_program=12), {"savings_pmpm": (30.0, 90.0), "one_time_cost": (100.0, 200.0)})
    tests = {
        "ROI: 72,000 gross - 45,000 cost = 27,000 net, ROI 0.60": lambda: (r["net_savings"], round(r["roi"], 2)) == (27000.0, 0.6),
        "break-even = fee 25 + 150/12 = 37.50 PMPM": lambda: r["breakeven_savings_pmpm"] == 37.5,
        "DiD = (75-150) - (90-75) = -90 PMPM": lambda: abs(did["did_pmpm"] - (-90.0)) < 1e-9,
        "naive pre/post = -75 PMPM (overstates savings)": lambda: abs(methods.naive_pre_post(mp) - (-75.0)) < 1e-9,
        "tornado: savings bar is widest": lambda: tor["input"].iloc[0] == "savings_pmpm",
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
