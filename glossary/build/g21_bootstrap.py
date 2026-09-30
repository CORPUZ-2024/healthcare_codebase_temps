"""
G21 — Bootstrap confidence interval for a difference in mean cost (percentile and BCa)
======================================================================================
Copy this whole block into a file (e.g. boot.py) and run:  python boot.py
Requires: numpy, scipy   (pip install numpy scipy)
"""
import numpy as np
from scipy import stats


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for mean(a) - mean(b) by resampling each group: percentile (hand-rolled) and BCa (scipy).

    Healthcare context
    ------------------
    Cost differences are driven by a few very expensive people; the t-interval assumes the sampling
    distribution of the difference is symmetric. For a difference of MEANS between similar-sized groups
    with the same shape, skewness largely cancels and t, percentile and BCa intervals nearly agree (the
    self-test shows this). The bootstrap earns its keep with small or unbalanced groups, and for
    statistics without an easy SE: medians, ratios of means, ROI, percentiles. BCa corrects the
    percentile interval for bias and skewness.

    Returns dict: diff, pct_lo, pct_hi, bca_lo, bca_hi, t_lo, t_hi (Welch t for comparison).

    Steps
    -----
    1. Resample each group WITH replacement at its own size (stratified by group); recompute the difference.
    2. Percentile CI = alpha/2 and 1 - alpha/2 quantiles of the replicates.
    3. BCa via scipy.stats.bootstrap (bias correction + acceleration from the jackknife).

    Common mistakes
    ---------------
    - Resampling rows from the pooled data (group sizes change every replicate).
    - Resampling claims instead of MEMBERS when members have many claims (resample the independent unit).
    - Too few replicates for a 95% interval (use >= 2,000; 5,000+ for BCa).
    """
    rng = np.random.default_rng(seed)
    diff = float(a.mean() - b.mean())
    reps = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])   # step 1
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])                                                      # step 2
    bca = stats.bootstrap((a, b), lambda x, y, axis=-1: x.mean(axis=axis) - y.mean(axis=axis), vectorized=True,
                          n_resamples=n_boot, confidence_level=1 - alpha, method="BCa", random_state=seed)      # step 3
    w = stats.ttest_ind(a, b, equal_var=False)
    se = diff / w.statistic if w.statistic else float("nan")
    tq = stats.t.ppf(1 - alpha / 2, w.df)
    return {"diff": diff, "pct_lo": float(lo), "pct_hi": float(hi), "bca_lo": float(bca.confidence_interval.low),
            "bca_hi": float(bca.confidence_interval.high), "t_lo": diff - tq * se, "t_hi": diff + tq * se}


# ---------------------------------------------------------------------------
# Self-test: run `python boot.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(6)
    ctrl = rng.lognormal(8.0, 1.4, 300)
    prog = rng.lognormal(7.85, 1.4, 300)
    true = np.exp(7.85 + 1.4 ** 2 / 2) - np.exp(8.0 + 1.4 ** 2 / 2)
    r = bootstrap_mean_diff(prog, ctrl)
    print({k: round(v) for k, v in r.items()}, "true", round(true))
    cover = 0
    for s in range(60):                                         # quick coverage check of the percentile interval
        rr = np.random.default_rng(100 + s)
        x = bootstrap_mean_diff(rr.lognormal(7.85, 1.4, 300), rr.lognormal(8.0, 1.4, 300), n_boot=800, seed=s)
        cover += x["pct_lo"] <= true <= x["pct_hi"]
    checks = {
        "percentile CI contains the point estimate": r["pct_lo"] < r["diff"] < r["pct_hi"],
        "equal groups: BCa, percentile and t intervals agree within 10% of width (skewness cancels)":
            max(abs(r["bca_lo"] - r["t_lo"]), abs(r["bca_hi"] - r["t_hi"]), abs(r["pct_lo"] - r["t_lo"])) < 0.10 * (r["t_hi"] - r["t_lo"]),
        f"percentile coverage over 60 samples >= 85% (got {cover}/60)": cover >= 51,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
