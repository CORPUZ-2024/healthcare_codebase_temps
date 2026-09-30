"""
G13 — Synthetic control (one treated unit, many donors) with placebo inference
===============================================================================
Copy this whole block into a file (e.g. synth.py) and run:  python synth.py
Requires: numpy, pandas, scipy   (pip install numpy pandas scipy)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def synth_weights(treated_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    """Non-negative donor weights summing to 1 that best reproduce the treated unit's PRE-period path.

    Healthcare context
    ------------------
    A state (or county, or health system) adopts a policy; nobody else does. DiD needs a comparison
    group with parallel trends; synthetic control BUILDS one as a weighted average of untreated units
    that matches the treated unit before the policy. The post-period gap is the estimated effect.

    Parameters
    ----------
    treated_pre : (T0,) outcome of the treated unit before the policy
    donors_pre  : (T0, J) outcomes of J donor units before the policy

    Returns w : (J,) weights, w >= 0, sum(w) = 1 (SLSQP on the pre-period squared error).
    """
    j = donors_pre.shape[1]
    obj = lambda w: float(np.sum((treated_pre - donors_pre @ w) ** 2))  # noqa: E731
    res = minimize(obj, np.full(j, 1 / j), method="SLSQP", bounds=[(0, 1)] * j,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},), options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def synthetic_control(panel: pd.DataFrame, treated: str, t0: int) -> dict:
    """Effect path and placebo p-value.

    Parameters
    ----------
    panel : wide DataFrame, index = period (int), columns = units, values = outcome
    treated : column of the treated unit; t0 : first post-policy period

    Returns {'weights', 'gap' (treated - synthetic, all periods), 'avg_post_effect', 'rmspe_ratio', 'p_value'}.

    Steps
    -----
    1. Fit weights on pre-periods; synthetic = donors @ w for all periods; gap = treated - synthetic.
    2. Placebo: pretend each donor was treated (others as its donors); compute post/pre RMSPE ratios.
    3. p-value = share of units (incl. the treated) with a ratio >= the treated unit's ratio.

    Common mistakes
    ---------------
    - Donors affected by the same policy or by spillovers (contaminated comparison).
    - Poor pre-period fit (large pre RMSPE) and trusting the post gap anyway.
    - Reporting a p-value from 5 donors (the smallest possible p is 1/6).
    """
    pre, post = panel.index < t0, panel.index >= t0

    def fit(unit):
        donors = panel.drop(columns=unit)
        w = synth_weights(panel.loc[pre, unit].to_numpy(), donors.loc[pre].to_numpy())
        gap = panel[unit] - donors.to_numpy() @ w
        return w, gap, np.sqrt(np.mean(gap[post] ** 2)) / np.sqrt(np.mean(gap[pre] ** 2))

    w, gap, ratio = fit(treated)                                                                  # step 1
    ratios = [ratio] + [fit(u)[2] for u in panel.columns if u != treated]                          # step 2
    return {"weights": pd.Series(w, index=panel.drop(columns=treated).columns), "gap": gap,
            "avg_post_effect": float(gap[post].mean()), "rmspe_ratio": float(ratio),
            "p_value": float(np.mean(np.array(ratios) >= ratio))}                                  # step 3


# ---------------------------------------------------------------------------
# Self-test: run `python synth.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, J, t0, effect = 36, 15, 24, -5.0
    common = np.cumsum(rng.normal(0, 1, T))
    donors = pd.DataFrame({f"D{j}": 50 + j + common * (0.5 + 0.1 * j) + rng.normal(0, 0.4, T) for j in range(J)})
    truth_w = np.zeros(J); truth_w[[2, 5, 9]] = [0.5, 0.3, 0.2]
    treated = donors.to_numpy() @ truth_w + rng.normal(0, 0.3, T) + np.where(np.arange(T) >= t0, effect, 0)
    panel = donors.assign(TREATED=treated)
    r = synthetic_control(panel, "TREATED", t0)
    print(r["weights"].round(2)[r["weights"] > 0.01], "\n", {k: round(v, 3) for k, v in r.items() if k in ("avg_post_effect", "p_value")})
    checks = {
        "weights non-negative and sum to 1": (r["weights"] >= -1e-9).all() and abs(r["weights"].sum() - 1) < 1e-6,
        "donors D2/D5/D9 carry most weight": r["weights"][["D2", "D5", "D9"]].sum() > 0.8,
        "post effect ~ -5": abs(r["avg_post_effect"] - effect) < 1.0,
        "placebo p-value is the minimum possible (1/16)": abs(r["p_value"] - 1 / (J + 1)) < 1e-9,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
