"""
t00 — Claims foundation: raw claims + enrollment -> analytic-ready tables.

    python run.py                      demo on synthetic data -> outputs/
    python run.py --selftest           plain-assert checks (no pytest needed)
    python run.py --synpuf <csv>       load a DE-SynPUF carrier file and run the checks on it
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from claims_foundation import checks, data, methods  # noqa: E402
from claims_foundation.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    u = data.generate_universe(cfg.n_members, cfg.seed, cfg.start, cfg.months)
    print(f"Synthetic universe: " + ", ".join(f"{k}={len(v):,}" for k, v in u.items()))

    print("\n== Data-quality findings ==")
    findings = checks.run_all(u)
    for f in findings:
        print(f"[{f.severity.upper():5}] {f.check_id}: {f.message}\n        fix: {f.fix}")
    pd.DataFrame([f.__dict__ for f in findings]).to_csv(OUT / "dq_findings.csv", index=False)

    latest = methods.collapse_versions_latest(u["medical"])
    net = methods.collapse_versions_net(data.to_delta_feed(u["medical"]))
    print(f"\n== Version collapse ==\nraw lines {len(u['medical']):,} -> latest {len(latest):,}; "
          f"paid latest ${latest['paid_amt'].sum():,.2f} vs net-of-deltas ${net['paid_amt'].sum():,.2f}")

    mm_daily = methods.member_months_daily(u["enrollment"], cfg.period_start, cfg.period_end)
    mm_mid = methods.member_months_midmonth(u["enrollment"], cfg.period_start, cfg.period_end,
                                            anchor_day=cfg.midmonth_anchor_day)
    print(f"\n== Member months ==\ndaily-prorated {mm_daily['member_months'].sum():,.1f} vs "
          f"mid-month rule {mm_mid['member_months'].sum():,.1f}")

    cat = methods.service_category_claim(latest)
    by_cat = cat.groupby("service_category")["paid_amt"].sum().sort_values(ascending=False)
    print("\n== Paid by service category (claim-level hierarchy) ==")
    print(by_cat.round(0).to_string())

    rx = methods.net_pharmacy_drop_pairs(u["pharmacy"])
    print(f"\n== Pharmacy == {len(u['pharmacy']):,} rows -> {len(rx):,} fills after removing reversal pairs")

    latest.to_csv(OUT / "claim_line_latest.csv", index=False)
    mm_daily.to_csv(OUT / "member_month.csv", index=False)
    cat[["claim_id", "service_category"]].drop_duplicates().to_csv(OUT / "claim_service_category.csv", index=False)
    rx.to_csv(OUT / "rx_fill.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def synpuf(path: str) -> None:
    df = data.load_synpuf_carrier(path, nrows=200_000)
    print(f"Loaded {len(df):,} SynPUF carrier rows")
    for f in (checks.check_icd10_format(df["dx1_cd"]) + checks.check_npi_luhn(df["npi_id"])
              + checks.check_hcpcs_format(df["hcpcs_cd"])):
        print(f"[{f.severity}] {f.check_id}: {f.message}")


def selftest() -> int:
    u = data.generate_universe(150, seed=3, months=12)
    med, enr = u["medical"], u["enrollment"]
    latest = methods.collapse_versions_latest(med)
    one = pd.DataFrame({"member_id": ["A"], "enroll_start_dt": pd.to_datetime(["2025-01-17"]),
                        "enroll_end_dt": pd.to_datetime(["2025-02-28"])})
    tests = {
        "NPI Luhn example 1234567893": lambda: data.luhn_npi("123456789") == "1234567893",
        "latest-version output is unique on claim_id+line_seq": lambda: not latest.duplicated(["claim_id", "line_seq"]).any(),
        "replacement(latest) == delta(net) paid total": lambda: abs(
            latest["paid_amt"].sum() - methods.collapse_versions_net(data.to_delta_feed(med))["paid_amt"].sum()) < 0.05,
        "15/31 proration = 0.484": lambda: round(methods.member_months_daily(one, "2025-01-01", "2025-12-31")["member_months"].iloc[0], 3) == 0.484,
        "no member has > 1.0 member-month in a month": lambda: methods.member_months_daily(enr, "2023-01-01", "2023-12-31")["member_months"].max() <= 1.0 + 1e-9,
        "NDC 4-4-2 -> 11 digits": lambda: methods.normalize_ndc11("1234-5678-90") == "01234567890",
        "every claim gets exactly one category": lambda: methods.service_category_claim(latest).groupby("claim_id")["service_category"].nunique().max() == 1,
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
    ap.add_argument("--synpuf", help="path to a DE-SynPUF carrier claims CSV")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    synpuf(a.synpuf) if a.synpuf else demo(Config())
