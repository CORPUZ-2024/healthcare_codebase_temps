"""
G20 — Two-part cost model: P(any cost) by logit x E[cost | cost > 0] by Gamma GLM (log link)
============================================================================================
Copy this whole block into a file (e.g. two_part.py) and run:  python two_part.py
Requires: numpy, pandas, statsmodels   (pip install numpy pandas statsmodels)   Used inside t06 (ROI) as an alternative.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def two_part_effect(df: pd.DataFrame, y: str, treat: str, covariates: list[str]) -> dict:
    """Average effect of ``treat`` on mean cost from a two-part model (recycled predictions).

    Healthcare context
    ------------------
    Annual cost has a spike at $0 (no use) and a long right tail. OLS on dollars is unbiased for the
    mean with enough data but noisy and can predict negative costs; log(cost) OLS needs a smearing
    retransformation and drops zeros. The two-part model handles both: part 1 = who has any cost,
    part 2 = how much, among users (Gamma, log link: effects are proportional).

    Returns dict: effect (mean cost difference), part1_or (odds ratio for any cost), part2_ratio (cost ratio among users),
    mean_treated_pred, mean_control_pred.

    Steps
    -----
    1. Part 1: logit any_cost ~ treat + covariates (all rows).
    2. Part 2: Gamma(log) GLM y ~ treat + covariates (rows with y > 0).
    3. Recycled predictions: E[y] = P(any) x E[y | any] for everyone with treat = 1 and with treat = 0; average the difference.

    Common mistakes
    ---------------
    - Reading the part-2 coefficient as the whole effect (it ignores changes in who has any cost).
    - log(y + 1) OLS and exponentiating (biased; the +1 is arbitrary).
    - Gaussian GLM with a log link on skewed costs (wrong variance; use Gamma).
    """
    d = df.assign(any_cost=(df[y] > 0).astype(float))
    rhs = " + ".join([treat] + covariates)
    p1 = smf.glm(f"any_cost ~ {rhs}", d, family=sm.families.Binomial()).fit()                          # step 1
    pos = d[d[y] > 0]
    p2 = smf.glm(f"{y} ~ {rhs}", pos, family=sm.families.Gamma(sm.families.links.Log())).fit()          # step 2
    e1 = p1.predict(d.assign(**{treat: 1})) * p2.predict(d.assign(**{treat: 1}))                        # step 3
    e0 = p1.predict(d.assign(**{treat: 0})) * p2.predict(d.assign(**{treat: 0}))
    return {"effect": float((e1 - e0).mean()), "part1_or": float(np.exp(p1.params[treat])),
            "part2_ratio": float(np.exp(p2.params[treat])), "mean_treated_pred": float(e1.mean()), "mean_control_pred": float(e0.mean())}


# ---------------------------------------------------------------------------
# Self-test: run `python two_part.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 20_000
    t = rng.integers(0, 2, n)
    risk = rng.normal(0, 1, n)
    p_any = 1 / (1 + np.exp(-(0.5 + 0.8 * risk - 0.3 * t)))
    mu = np.exp(8.5 + 0.5 * risk - 0.15 * t)                       # program: fewer users AND 14% lower cost per user
    y = np.where(rng.random(n) < p_any, rng.gamma(1.2, mu / 1.2), 0.0)
    df = pd.DataFrame({"cost": y, "program_flag": t, "risk": risk})
    true = (np.mean(1 / (1 + np.exp(-(0.5 + 0.8 * risk - 0.3))) * np.exp(8.5 + 0.5 * risk - 0.15))
            - np.mean(1 / (1 + np.exp(-(0.5 + 0.8 * risk))) * np.exp(8.5 + 0.5 * risk)))
    r = two_part_effect(df, "cost", "program_flag", ["risk"])
    print({k: round(v, 3) for k, v in r.items()}, "true effect", round(true, 1), "zeros", f"{(y == 0).mean():.0%}")
    checks = {
        "effect within 10% of the true mean difference": abs(r["effect"] / true - 1) < 0.10,
        "part 1 odds ratio ~ exp(-0.3) = 0.74": abs(r["part1_or"] - np.exp(-0.3)) < 0.05,
        "part 2 cost ratio ~ exp(-0.15) = 0.86": abs(r["part2_ratio"] - np.exp(-0.15)) < 0.04,
        "part 2 alone understates the effect": abs(r["effect"]) > abs((r["part2_ratio"] - 1) * y[y > 0].mean() * (y > 0).mean()),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
