"""
G15 — Instrumental variables: two-stage least squares with correct standard errors
===================================================================================
Copy this whole block into a file (e.g. iv.py) and run:  python iv.py
Requires: numpy   (pip install numpy)   Packaged alternative: linearmodels.IV2SLS.
"""
import numpy as np


def tsls(y: np.ndarray, x_endog: np.ndarray, z: np.ndarray, w: np.ndarray | None = None) -> dict:
    """2SLS for one endogenous regressor with instrument(s) z and exogenous controls w.

    Healthcare context
    ------------------
    When sicker people choose (or are chosen for) a treatment and severity isn't fully measured, OLS is
    biased. An instrument shifts treatment but affects the outcome ONLY through treatment - e.g.
    distance to the nearest specialty clinic, a provider's prescribing preference, a staggered rollout.
    Valid instruments are rare in operational data; say why yours qualifies.

    Returns dict: beta (effect of x), se (correct), naive_second_stage_se, first_stage_F.

    Steps
    -----
    1. First stage: regress x on [1, z, w]; F-statistic for the instruments (rule of thumb: F > 10).
    2. Second stage: regress y on [1, x_hat, w]; beta is the IV estimate.
    3. Standard errors: residuals must use the ACTUAL x (y - [1, x, w] b), not x_hat. Plugging the
       second-stage OLS residuals in gives wrong SEs (the classic hand-rolled 2SLS mistake).

    Common mistakes
    ---------------
    - Weak instruments (F < 10): 2SLS is biased toward OLS and CIs are wrong.
    - Instruments with a direct path to the outcome (exclusion restriction violated).
    - Reporting the naive second-stage SEs.
    """
    n = len(y)
    W = np.ones((n, 1)) if w is None else np.column_stack([np.ones(n), w])
    Z = np.column_stack([W, z])
    g, *_ = np.linalg.lstsq(Z, x_endog, rcond=None)                                           # step 1
    x_hat = Z @ g
    rss_full = np.sum((x_endog - x_hat) ** 2)
    gw, *_ = np.linalg.lstsq(W, x_endog, rcond=None)
    rss_restr = np.sum((x_endog - W @ gw) ** 2)
    q = Z.shape[1] - W.shape[1]
    f_stat = ((rss_restr - rss_full) / q) / (rss_full / (n - Z.shape[1]))
    X2 = np.column_stack([W[:, :1], x_hat, W[:, 1:]])                                           # step 2
    b, *_ = np.linalg.lstsq(X2, y, rcond=None)
    X_actual = np.column_stack([W[:, :1], x_endog, W[:, 1:]])
    resid = y - X_actual @ b                                                                    # step 3
    sigma2 = resid @ resid / (n - X2.shape[1])
    cov = sigma2 * np.linalg.inv(X2.T @ X2)
    naive_resid = y - X2 @ b
    naive_cov = (naive_resid @ naive_resid / (n - X2.shape[1])) * np.linalg.inv(X2.T @ X2)
    return {"beta": float(b[1]), "se": float(np.sqrt(cov[1, 1])), "naive_second_stage_se": float(np.sqrt(naive_cov[1, 1])),
            "first_stage_F": float(f_stat)}


# ---------------------------------------------------------------------------
# Self-test: run `python iv.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def simulate(n, rng):
        severity = rng.normal(0, 1, n)                                # unmeasured
        distance = rng.uniform(0, 50, n)                              # instrument: miles to the clinic
        age = rng.normal(0, 1, n)
        treat = (0.8 * severity - 0.04 * distance + 0.2 * age + rng.normal(0, 1, n) > -1).astype(float)
        cost = 10 - 3.0 * treat + 6.0 * severity + 1.0 * age + rng.normal(0, 2, n)     # true effect -3
        return cost, treat, distance, age

    rng = np.random.default_rng(2)
    cost, treat, distance, age = simulate(20_000, rng)
    ols = np.linalg.lstsq(np.column_stack([np.ones(len(cost)), treat, age]), cost, rcond=None)[0][1]
    r = tsls(cost, treat, distance, age)
    reps = [tsls(*simulate(2_000, rng)) for _ in range(200)]           # does the SE match the real sampling spread?
    sd_beta = float(np.std([x["beta"] for x in reps], ddof=1))
    mean_se = float(np.mean([x["se"] for x in reps]))
    print({k: round(v, 3) for k, v in r.items()}, "OLS", round(ols, 3), "| sampling SD", round(sd_beta, 3), "mean SE", round(mean_se, 3))
    checks = {
        "OLS is biased (sicker people treated)": ols > -1.0,
        "IV CI covers the true -3": abs(r["beta"] + 3.0) < 1.96 * r["se"],
        "strong instrument (F > 10)": r["first_stage_F"] > 10,
        "reported SE matches the sampling SD of beta across 200 samples (within 15%)": abs(mean_se / sd_beta - 1) < 0.15,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
