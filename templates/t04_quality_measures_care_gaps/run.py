"""
t04 — HEDIS / Core Set style quality measures and care gaps (measure specs and value sets are FAKE).

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from quality_measures_care_gaps import checks, data, methods  # noqa: E402
from quality_measures_care_gaps.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    print("NOTE: measure specs and value sets are FAKE teaching versions of HEDIS / Core Set measures.\n")
    d = data.generate(cfg.n_members, cfg.measurement_year, cfg.seed, cfg.data_start)
    vs = data.load_value_sets(cfg.value_set_path)
    specs = data.load_measures(cfg.measures_dir)
    my = cfg.measurement_year

    rows, member_rows, gap_rows, hybrid_rows = [], [], [], []
    for mid, spec in specs.items():
        ml = methods.evaluate_measure(spec, d["members"], d["enrollment"], d["events"], vs, my)
        w = methods.rate_summary(ml, "wilson", cfg.z, cfg.min_denominator)
        j = methods.rate_summary(ml, "jeffreys", cfg.z, cfg.min_denominator)
        rows.append({**w, "jeffreys_lo": j["ci_lo"], "jeffreys_hi": j["ci_hi"], "method": "admin"})
        member_rows.append(ml)
        gap_rows.append(methods.care_gap_list(ml, spec, d["events"], vs, my))
        if spec["numerator"]["hybrid_sources"]:
            h, _ = methods.hybrid_rate(spec, ml, d["events"], vs, my, cfg.hybrid_sample_size, cfg.seed, cfg.z)
            hybrid_rows.append(h)
    rates = pd.DataFrame(rows)
    print(f"== MY{my} rates (STANDARD: administrative method, Wilson 95% CI; ALTERNATIVE: Jeffreys) ==")
    show = rates[["measure_id", "eligible_cnt", "exclusion_cnt", "denominator_cnt", "numerator_cnt", "rate",
                  "ci_lo", "ci_hi", "jeffreys_lo", "jeffreys_hi", "reportable_flag"]]
    print(show.round(3).to_string(index=False))

    print("\n== ALTERNATIVE numerator: hybrid method (systematic sample + chart review) ==")
    for h in hybrid_rows:
        a = rates.set_index("measure_id").loc[h["measure_id"]]
        print(f"{h['measure_id']}: admin {a['rate']:.1%} (n={a['denominator_cnt']}) vs hybrid {h['rate']:.1%} "
              f"(sample n={h['sample_size']}, CI {h['ci_lo']:.1%}-{h['ci_hi']:.1%}) - not comparable to admin benchmarks")

    print("\n== Small-denominator behaviour: the same measure for one small provider group (n=25) ==")
    bcs = member_rows[0]
    small = bcs[bcs["denominator_flag"] == 1].head(25)
    s = methods.rate_summary(small, "wilson", cfg.z, cfg.min_denominator)
    sj = methods.rate_summary(small, "jeffreys", cfg.z, cfg.min_denominator)
    print(f"rate {s['rate']:.0%}  Wilson {s['ci_lo']:.0%}-{s['ci_hi']:.0%}  Jeffreys {sj['ci_lo']:.0%}-{sj['ci_hi']:.0%}  {s['note']}")

    bench = data.load_core_set_rates(data.generate_fake_core_set(cfg.core_set_year, cfg.seed))
    print("\n== Benchmark position vs. (FAKE) Core Set state rates ==")
    bench_rows = []
    for mid, spec in specs.items():
        cd = spec.get("core_set_benchmark")
        if cd:
            b = methods.benchmark_position(float(rates.set_index("measure_id").loc[mid, "rate"]), bench, cd)
            bench_rows.append({"measure_id": mid, **b})
    print(pd.DataFrame(bench_rows).round(3).to_string(index=False))

    member_level = pd.concat(member_rows, ignore_index=True)
    gaps = pd.concat(gap_rows, ignore_index=True)
    print(f"\n== Care-gap outreach list: {len(gaps):,} open gaps ==")
    print(gaps.groupby("measure_id").size().to_string())
    print(gaps["note"].replace("", "(no prior service)").value_counts().to_string())

    print("\n== Checks ==")
    for f in (checks.check_code_format(d["events"]) + checks.check_overlapping_spans(d["enrollment"])
              + checks.check_runout(cfg.data_cut_dt, my, cfg.min_runout_days)
              + checks.check_value_sets_exist(specs, vs)
              + checks.check_small_denominators(rates, cfg.min_denominator)
              + checks.check_benchmark_vintage(bench, my)):
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    rates.to_csv(OUT / "measure_rates.csv", index=False)
    pd.DataFrame(hybrid_rows).to_csv(OUT / "hybrid_rates.csv", index=False)
    member_level.to_csv(OUT / "member_level_detail.csv", index=False)
    gaps.to_csv(OUT / "care_gap_outreach.csv", index=False)
    pd.DataFrame(bench_rows).to_csv(OUT / "benchmark_position.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    cfg = Config()
    vs = data.load_value_sets(cfg.value_set_path)
    spec = data.load_measures(cfg.measures_dir)["FAKE_BCS"]
    members = pd.DataFrame({"member_id": ["A", "B", "C", "D"], "sex_cd": ["F", "F", "F", "M"],
                            "birth_dt": pd.to_datetime(["1960-05-01"] * 4)})
    enr = pd.DataFrame({"member_id": ["A", "B", "C", "D"], "enroll_start_dt": pd.Timestamp("2023-01-01"),
                        "enroll_end_dt": pd.Timestamp("2025-12-31")})
    ev = pd.DataFrame({"member_id": ["A", "B", "C"], "event_dt": pd.to_datetime(["2023-10-01", "2023-09-30", "2025-03-01"]),
                       "code_system": ["CPT", "CPT", "HCPCS"], "code_cd": ["77067", "77067", "Q5001"],
                       "source_cd": "CLAIM"})
    ml = methods.evaluate_measure(spec, members, enr, ev, vs, 2025).set_index("member_id")
    e2 = pd.DataFrame({"member_id": ["X", "X"], "enroll_start_dt": pd.to_datetime(["2025-01-01", "2025-03-19"]),
                       "enroll_end_dt": pd.to_datetime(["2025-01-31", "2025-12-31"])})
    tests = {
        "mammogram on first day of the 27-month window counts": lambda: ml.loc["A", "status"] == "MET",
        "mammogram one day before the window is an open gap": lambda: ml.loc["B", "status"] == "OPEN_GAP",
        "hospice in MY excludes": lambda: ml.loc["C", "status"] == "EXCLUDED",
        "men are not in the BCS population": lambda: "D" not in ml.index,
        "45-day gap passes CE, 46-day gap fails": lambda: (
            methods.continuous_enrollment(e2, "2025-01-01", "2025-12-31", "2025-12-31")["ce_flag"].iloc[0] == 0
            and methods.continuous_enrollment(e2.assign(enroll_start_dt=pd.to_datetime(["2025-01-01", "2025-03-18"])),
                                              "2025-01-01", "2025-12-31", "2025-12-31")["ce_flag"].iloc[0] == 1),
        "Wilson 0/10 upper = 0.2775": lambda: abs(methods.ci_wilson(0, 10)[1] - 0.2775) < 1e-4,
        "Jeffreys 0/10 upper = 0.2172": lambda: abs(methods.ci_jeffreys(0, 10)[1] - 0.2172) < 1e-4,
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
