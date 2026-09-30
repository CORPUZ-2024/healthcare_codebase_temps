"""
G24 — RADV-style stratified sample and overpayment extrapolation (lower confidence limit)
========================================================================================
Copy this whole block into a file (e.g. radv.py) and run:  python radv.py
Requires: numpy, pandas, scipy   (pip install numpy pandas scipy)
Audit methodology varies (CMS RADV, OIG audits use RAT-STATS); this shows the statistics, not the rules.
"""
import numpy as np
import pandas as pd
from scipy import stats


def stratified_sample(pop: pd.DataFrame, stratum: str, n_per_stratum: dict, seed: int = 0) -> pd.DataFrame:
    """Simple random sample without replacement within each stratum (e.g. by risk-adjusted payment level)."""
    rng = np.random.default_rng(seed)
    parts = [g.iloc[rng.choice(len(g), min(n_per_stratum[h], len(g)), replace=False)] for h, g in pop.groupby(stratum)]
    return pd.concat(parts)


def extrapolate(sample: pd.DataFrame, pop_counts: pd.Series, stratum: str, value: str, confidence: float = 0.90) -> dict:
    """Stratified estimate of the population total (e.g. overpayment) with SE and two-sided CI limits.

    Healthcare context
    ------------------
    Risk-adjustment data validation audits a sample of enrollees' medical records; payments for HCCs the
    records don't support are overpayments. The sample result is extrapolated to the contract. Audit
    practice (e.g. OIG) typically demands the LOWER limit of a two-sided 90% confidence interval, so the
    demand is less than the true total with high probability. Plans validate their own coding the same way.

    Returns dict: point_total, se, lower_limit, upper_limit.

    Steps
    -----
    1. Per stratum h: mean y_h, variance s_h^2, sample size n_h, population N_h.
    2. Total = sum N_h x mean_h; Var = sum N_h^2 x (1 - n_h/N_h) x s_h^2 / n_h (finite population correction).
    3. CI = total +/- t x SE (df = n - H); lower limit = the demand under the lower-limit convention.

    Common mistakes
    ---------------
    - Extrapolating a simple mean when the sample was stratified (weights ignored).
    - Dropping sampled records that couldn't be retrieved instead of treating them per the audit rules.
    - Ignoring the finite population correction when a stratum is heavily sampled.
    """
    g = sample.groupby(stratum)[value].agg(["mean", "var", "count"])                                   # step 1
    N = pop_counts.reindex(g.index)
    total = float((N * g["mean"]).sum())                                                                # step 2
    var = float((N ** 2 * (1 - g["count"] / N) * g["var"] / g["count"]).sum())
    se = np.sqrt(var)
    t = stats.t.ppf(1 - (1 - confidence) / 2, int(g["count"].sum() - len(g)))                         # step 3
    return {"point_total": total, "se": float(se), "lower_limit": total - t * se, "upper_limit": total + t * se}


# ---------------------------------------------------------------------------
# Self-test: run `python radv.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N = 20_000
    stratum = rng.choice(["low", "mid", "high"], N, p=[0.6, 0.3, 0.1])
    pay = np.select([stratum == "low", stratum == "mid"], [1_500, 4_000], 9_000) * rng.lognormal(0, 0.3, N)
    unsupported = rng.random(N) < np.select([stratum == "low", stratum == "mid"], [0.05, 0.10], 0.15)
    overpay = np.where(unsupported, pay * rng.uniform(0.2, 0.8, N), 0.0)
    pop = pd.DataFrame({"enrollee_id": np.arange(N), "stratum": stratum, "overpay": overpay})
    truth = overpay.sum()
    counts = pop.stratum.value_counts()
    hits, below = 0, 0
    for s in range(200):
        smp = stratified_sample(pop, "stratum", {"low": 70, "mid": 70, "high": 70}, seed=s)
        r = extrapolate(smp, counts, "stratum", "overpay")
        hits += r["lower_limit"] <= truth <= r["upper_limit"]
        below += r["lower_limit"] <= truth
    print(f"true overpayment {truth:,.0f}; last estimate {r['point_total']:,.0f} (LL {r['lower_limit']:,.0f})")
    checks = {
        f"90% CI covers the truth in ~90% of 200 audits (got {hits})": 165 <= hits <= 195,
        f"lower limit <= truth in ~95% of audits (got {below})": below >= 180,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
