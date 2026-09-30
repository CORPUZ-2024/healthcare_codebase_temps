"""
G14 — Sharp regression discontinuity (local linear, triangular kernel)
======================================================================
Copy this whole block into a file (e.g. rdd.py) and run:  python rdd.py
Requires: numpy, pandas, statsmodels   (pip install numpy pandas statsmodels)
For data-driven bandwidths and robust bias-corrected CIs use rdrobust (Python/R/Stata).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def rdd_estimate(df: pd.DataFrame, running: str, outcome: str, cutoff: float, bandwidth: float) -> dict:
    """Jump in the outcome at the cutoff, estimated with separate local lines on each side.

    Healthcare context
    ------------------
    Programs often enroll by a score threshold: acuity tier >= 7, risk score in the top 5%, age 65.
    People just above and just below the cutoff are alike except for the program, so the jump in
    outcomes at the cutoff is the program's effect FOR PEOPLE NEAR THE CUTOFF.

    Parameters
    ----------
    running : score that determines eligibility (treated if running >= cutoff)
    bandwidth : use observations within cutoff +/- bandwidth

    Returns dict: effect, se, ci_lo, ci_hi, n_left, n_right, density_ratio (counts just above / below).

    Steps
    -----
    1. Keep |running - cutoff| <= bandwidth; centre x = running - cutoff; D = x >= 0.
    2. Weighted least squares y ~ D + x + D:x with triangular weights 1 - |x|/h (robust SE).
    3. Effect = coefficient on D. Manipulation check: counts in the half-bandwidth on each side.

    Common mistakes
    ---------------
    - High-order global polynomials (they produce spurious jumps); stay local and linear.
    - Ignoring manipulation: if staff can nudge scores over the line, units just above differ.
    - Generalizing the effect to people far from the cutoff.
    """
    d = df[(df[running] - cutoff).abs() <= bandwidth].copy()                                       # step 1
    d["x"] = d[running] - cutoff
    d["D"] = (d.x >= 0).astype(float)
    X = sm.add_constant(pd.DataFrame({"D": d.D, "x": d.x, "Dx": d.D * d.x}))
    fit = sm.WLS(d[outcome], X, weights=1 - d.x.abs() / bandwidth).fit(cov_type="HC1")            # step 2
    b, se = float(fit.params["D"]), float(fit.bse["D"])
    half = bandwidth / 2
    above, below = ((d.x >= 0) & (d.x < half)).sum(), ((d.x < 0) & (d.x >= -half)).sum()           # step 3
    return {"effect": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "n_left": int((d.x < 0).sum()),
            "n_right": int((d.x >= 0).sum()), "density_ratio": float(above / below) if below else float("nan")}


# ---------------------------------------------------------------------------
# Self-test: run `python rdd.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 6_000
    score = rng.uniform(0, 10, n)
    y = 20 + 1.5 * score + 0.08 * score ** 2 - 4.0 * (score >= 7) + rng.normal(0, 2, n)     # program lowers ED visits by 4
    r = rdd_estimate(pd.DataFrame({"acuity": score, "ed_visits": y}), "acuity", "ed_visits", cutoff=7, bandwidth=1.5)
    none = rdd_estimate(pd.DataFrame({"acuity": score, "ed_visits": y - (-4.0) * (score >= 7)}), "acuity", "ed_visits", 7, 1.5)
    print({k: round(v, 3) for k, v in r.items()})
    checks = {
        "CI covers the true jump of -4": r["ci_lo"] <= -4.0 <= r["ci_hi"],
        "no jump when there is no program": none["ci_lo"] <= 0 <= none["ci_hi"],
        "no manipulation: density ratio ~ 1": 0.85 < r["density_ratio"] < 1.15,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
