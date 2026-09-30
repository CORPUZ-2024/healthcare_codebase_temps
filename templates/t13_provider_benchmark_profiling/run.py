"""
t13 — Provider benchmarking: risk-adjusted O/E, funnel plot, EB shrinkage, mixed model, external benchmark.
Synthetic providers with KNOWN quality; the national benchmark file in the demo is FAKE.

    python run.py              demo -> outputs/ (incl. funnel_plot.png)
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

from provider_benchmark_profiling import checks, data, methods  # noqa: E402
from provider_benchmark_profiling.config import Config  # noqa: E402

OUT = HERE / "outputs"


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    d = data.generate(cfg.n_providers, cfg.provider_sd, cfg.seed)
    pts, truth = d["patients"], d["providers"].set_index("provider_id")
    exp_p, info = methods.risk_model(pts)
    oe = methods.funnel_flags(methods.oe_table(pts, exp_p), cfg.funnel_levels)
    eb = methods.eb_shrink_oe(oe).set_index("provider_id")
    mx = methods.mixed_logistic_ratios(pts).set_index("provider_id")
    print(f"{len(pts):,} patients at {cfg.n_providers} providers (volume {oe['n_cases'].min()}-{oe['n_cases'].max()}); "
          f"overall 30-day readmission {pts['readmit_flag'].mean():.1%}")
    print(f"risk model c-statistic {info['c_statistic']:.2f}; between-provider variance tau^2 {eb.attrs['tau2']:.3f} (shrinkage a = {eb.attrs['a']:.0f})\n")

    crude = eb["crude_rate"] / pts["readmit_flag"].mean()
    rows = []
    for name, est in (("crude rate / average", crude), ("O/E (STANDARD, raw)", eb["oe"]), ("EB-shrunk O/E (STANDARD)", eb["eb_oe"]),
                      ("mixed model P/E (ALTERNATIVE)", mx["pe_ratio"])):
        rows.append({"estimate": name, **methods.estimation_error(est.reindex(truth.index).clip(lower=0.01), truth["true_ratio"])})
    print("== Accuracy vs the TRUE provider ratio (synthetic only) ==")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n== Smallest and largest providers ==")
    show = eb.join(mx["pe_ratio"]).join(truth["true_ratio"]).sort_values("n_cases")
    cols = ["n_cases", "observed", "expected", "oe", "oe_lo", "oe_hi", "eb_oe", "reliability", "pe_ratio", "true_ratio", "flag_0.95"]
    print(pd.concat([show.head(4), show.tail(3)])[cols].round(2).to_string())

    fl = oe[[f"flag_{lv}" for lv in cfg.funnel_levels]]
    print(f"\nfunnel: {int((fl.iloc[:, 0] != 0).sum())} outside 95% limits, {int((fl.iloc[:, 1] != 0).sum())} outside 99.8%")
    try:
        methods.funnel_plot(oe, OUT / "funnel_plot.png", cfg.funnel_levels)
        print(f"funnel plot -> {OUT / 'funnel_plot.png'}")
    except ImportError:
        print("matplotlib not installed: funnel plot skipped")

    bench = data.load_hrrp(data.generate_fake_hrrp(seed=cfg.seed))
    top = eb["eb_oe"].idxmin()
    bp = methods.benchmark_percentile(float(eb.loc[top, "eb_oe"]), bench["excess_readmission_ratio"])
    print(f"\n== External benchmark (FAKE national HRRP-style file, {bp['n_reference']:,} hospitals reported) ==")
    print(f"best provider {top}: EB O/E {eb.loc[top, 'eb_oe']:.2f} -> national percentile {bp['percentile']:.0f} "
          f"(national median {bp['median']:.2f}); suppressed small hospitals: {int(bench['excess_readmission_ratio'].isna().sum())}")

    print("\n== Checks ==")
    fnd = (checks.check_small_volume(oe, cfg.min_cases) + checks.check_overdispersion(oe) + checks.check_risk_model(info)
           + checks.check_rank_disagreement(crude, eb["oe"])
           + checks.check_benchmark_vintage(bench["start_date"].min(), bench["end_date"].max(), cfg.measurement_start, cfg.measurement_end))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    show.to_csv(OUT / "provider_profile.csv")
    pd.DataFrame(rows).to_csv(OUT / "estimator_accuracy.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    pts = pd.DataFrame({"provider_id": ["A"] * 4 + ["B"] * 4, "readmit_flag": [1, 1, 0, 0, 1, 0, 0, 0]})
    oe = methods.oe_table(pts, pd.Series([0.25] * 8))
    lim = methods.funnel_limits(np.array([100.0]), (0.95,))
    eb = methods.eb_shrink_oe(pd.DataFrame({"observed": [0, 30, 60], "expected": [2.0, 30.0, 40.0]}))
    tests = {
        "O/E: A = 2/1, B = 1/1": lambda: oe["oe"].tolist() == [2.0, 1.0],
        "funnel at E=100: 95% limits ~0.81-1.20": lambda: abs(lim["lo_0.95"].iloc[0] - 0.81) < 0.02 and abs(lim["hi_0.95"].iloc[0] - 1.20) < 0.02,
        "EB pulls the small provider (O=0, E=2) furthest toward 1": lambda: (eb["eb_oe"] - eb["observed"] / eb["expected"]).abs().idxmax() == 0,
        "EB reliability rises with expected count": lambda: eb["reliability"].is_monotonic_increasing,
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
