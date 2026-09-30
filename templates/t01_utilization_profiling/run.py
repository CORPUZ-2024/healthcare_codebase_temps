"""
t01 — Utilization profiling: per-1,000 rates, ALOS, readmissions, runout-aware trend.

    python run.py                        demo on synthetic data -> outputs/
    python run.py --selftest             plain-assert checks (no pytest needed)
    python run.py --synpuf-ip <csv>      profile DE-SynPUF inpatient claims
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from utilization_profiling import checks, data, methods, prep  # noqa: E402
from utilization_profiling.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    u = data.generate_universe(cfg.n_members, cfg.seed, cfg.start, cfg.months)
    claims = prep.service_category_claim(prep.collapse_versions_latest(u["medical"]))
    end = (pd.Timestamp(cfg.start) + pd.DateOffset(months=cfg.months) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    mm = prep.member_months_daily(u["enrollment"], cfg.start, end)
    incomplete_after = mm["month"].max() - cfg.runout_lag_months

    stays = methods.build_stays(claims, cfg.transfer_gap_days)
    events = methods.count_events(claims)
    complete = events[events["event_dt"].dt.to_period("M") <= incomplete_after]
    mm_c = mm[mm["month"] <= incomplete_after]
    util = methods.utilization_table(complete, mm_c, stays[stays["admit_dt"].dt.to_period("M") <= incomplete_after])
    print(f"== Utilization per 1,000 member-years ({mm_c['month'].min()}..{incomplete_after}) — STANDARD exact Poisson CI ==")
    print(util.round(1).to_string(index=False))

    me = complete[complete["event_type"] == "ED_VISIT"].groupby("member_id").size()
    boot = methods.rate_ci_cluster_bootstrap(me, mm_c.groupby("member_id")["member_months"].sum(), n_boot=cfg.n_boot)
    ed = util.set_index("event_type").loc["ED_VISIT"]
    print(f"ED visits: exact Poisson CI {ed['lo']:.0f}-{ed['hi']:.0f} vs ALTERNATIVE member-bootstrap "
          f"{boot['lo']:.0f}-{boot['hi']:.0f} (wider = overdispersion)")

    enr_end = u["enrollment"].groupby("member_id")["enroll_end_dt"].max()
    idx = methods.readmissions_per_index(stays, enr_end, cfg.readmit_window_days)
    rr = methods.readmission_rate(idx)
    rp = methods.readmissions_per_1000(idx[idx["eligible_flag"] == 1], mm_c["member_months"].sum())
    print(f"\n== {cfg.readmit_window_days}-day all-cause readmissions ==\nSTANDARD per index stay: "
          f"{rr['readmissions']}/{rr['index_stays']} = {rr['rate']:.1%} (95% CI {rr['lo']:.1%}-{rr['hi']:.1%})")
    print(f"ALTERNATIVE per 1,000 member-years: {rp['rate']:.1f} (95% CI {rp['lo']:.1f}-{rp['hi']:.1f})")

    cost = claims.groupby("member_id")["paid_amt"].sum().reindex(mm["member_id"].unique(), fill_value=0)
    print("\n== Annualized paid per member: distribution ==")
    print({k: round(v, 2) for k, v in methods.describe_distribution(cost).items()})

    freq = methods.frequent_ed_users(events, cfg.frequent_ed_threshold)
    print(f"\n== Frequent ED users (>= {cfg.frequent_ed_threshold} visits in 12 months): {len(freq)} members")

    monthly = methods.monthly_rates(events, mm, "ED_VISIT")
    r12 = methods.trend_rolling12(monthly, incomplete_after)
    yoy = methods.trend_yoy_same_month(monthly, incomplete_after)
    print("\n== ED trend: rolling-12 (STANDARD) and YoY same month (ALTERNATIVE), last 6 months ==")
    print(r12.join(yoy[["yoy_pct"]]).tail(6).round(3).to_string())

    print("\n== Checks ==")
    for f in (checks.check_ip_missing_dates(claims) + checks.check_same_day_transfers(stays)
              + checks.check_runout(monthly) + checks.check_small_denominator(rr["index_stays"])):
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    try:
        from utilization_profiling.charts import trend_chart
        p = trend_chart(r12, OUT / "ed_trend.png", "ED visits per 1,000 member-years (runout months blank)")
        print(f"\nChart: {p}")
    except ImportError:
        print("\n(matplotlib not installed: skipping chart)")

    util.to_csv(OUT / "utilization_per_1000.csv", index=False)
    idx.to_csv(OUT / "index_stays_readmissions.csv", index=False)
    freq.to_csv(OUT / "frequent_ed_users.csv", index=False)
    r12.join(yoy[["yoy_pct"]]).to_csv(OUT / "ed_monthly_trend.csv")
    print(f"Outputs written to {OUT}")


def synpuf_ip(path: str) -> None:
    ip = data.load_synpuf_inpatient(path)
    ip["service_category"] = "IP"
    stays = methods.build_stays(ip)
    print(f"{len(ip):,} SynPUF inpatient claims -> {len(stays):,} stays; ALOS {stays['los_days'].mean():.1f} days")
    idx = methods.readmissions_per_index(stays, stays.groupby("member_id")["discharge_dt"].max() + pd.Timedelta(days=365))
    print(methods.readmission_rate(idx))


def selftest() -> int:
    d = pd.to_datetime
    stays_claims = pd.DataFrame({
        "claim_id": ["a", "b", "c", "d"], "member_id": ["M1"] * 4, "service_category": ["IP"] * 4,
        "admit_dt": d(["2025-01-01", "2025-01-06", "2025-01-20", "2025-04-01"]),
        "discharge_dt": d(["2025-01-05", "2025-01-08", "2025-01-22", "2025-04-03"]), "paid_amt": [1.0] * 4})
    stays = methods.build_stays(stays_claims)
    idx = methods.readmissions_per_index(stays, pd.Series({"M1": d("2025-12-31")}), data_end=d("2025-12-31"))
    r = methods.rate_ci_poisson_exact(10, 1_200)
    tests = {
        "transfer (admit 1 day after discharge) merges into one stay": lambda: len(stays) == 3,
        "readmission 12 days after discharge is flagged": lambda: idx["readmit_flag"].tolist() == [1, 0, 0],
        "exact Poisson CI for 10 events / 1,200 MM = 48.0-183.9": lambda: (round(r["lo"], 1), round(r["hi"], 1)) == (48.0, 183.9),
        "describe: mean > median for skewed data": lambda: (lambda s: s["mean"] > s["median"])(methods.describe_distribution(pd.Series([1, 1, 1, 2, 50.0]))),
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
    ap.add_argument("--synpuf-ip", help="DE-SynPUF inpatient claims CSV")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    synpuf_ip(a.synpuf_ip) if a.synpuf_ip else demo(Config())
