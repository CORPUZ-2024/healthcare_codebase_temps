"""
G17 — Multiple imputation by chained equations (MICE) + Rubin's rules
=====================================================================
Copy this whole block into a file (e.g. mice.py) and run:  python mice.py
Requires: numpy, pandas, statsmodels   (pip install numpy pandas statsmodels)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.mice import MICE, MICEData


def mice_regression(df: pd.DataFrame, formula: str, n_imputations: int = 20, n_burnin: int = 10, seed: int = 0) -> pd.DataFrame:
    """Fit an OLS model on multiply imputed data and pool with Rubin's rules (statsmodels MICE).

    Healthcare context
    ------------------
    Survey items, lab values and social-risk fields are often missing. Dropping incomplete rows
    (complete-case analysis) is biased when missingness depends on OBSERVED values - e.g. sicker
    members skip the functional-status question and sicker members cost more. MICE imputes each
    missing variable from the others (INCLUDING the outcome), several times, and pools the results so
    the uncertainty from imputation is in the standard errors.

    Returns a table: term, coef, se, ci_lo, ci_hi, fmi (fraction of missing information).

    Steps
    -----
    1. MICEData: iteratively impute each variable with missing values from the others (predictive mean matching).
    2. Fit the analysis model on each completed data set.
    3. Rubin's rules: pooled coef = mean; total variance = within + (1 + 1/m) x between.

    Common mistakes
    ---------------
    - Leaving the outcome out of the imputation model (biases coefficients toward zero).
    - Imputing once and analysing as if the data were observed (SEs too small).
    - Using MI when data are missing NOT at random without a sensitivity analysis.
    """
    np.random.seed(seed)                                   # statsmodels' MICE uses the global numpy RNG
    imp = MICEData(df)                                                                       # step 1
    res = MICE(formula, sm.OLS, imp).fit(n_burnin=n_burnin, n_imputations=n_imputations)     # steps 2-3
    ci = res.conf_int()
    return pd.DataFrame({"term": res.exog_names, "coef": res.params, "se": res.bse, "ci_lo": ci[:, 0], "ci_hi": ci[:, 1],
                         "fmi": res.frac_miss_info})


# ---------------------------------------------------------------------------
# Self-test: run `python mice.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(4)
    n = 3_000
    frailty = rng.normal(0, 1, n)
    age = rng.normal(0, 1, n)
    cost = 5 + 2.0 * frailty + 1.0 * age + rng.normal(0, 1, n)            # true frailty effect 2.0
    miss = rng.random(n) < 1 / (1 + np.exp(-(cost - 7)))                 # frailty missing more often when cost is high (MAR on cost)
    df = pd.DataFrame({"cost": cost, "frailty": np.where(miss, np.nan, frailty), "age": age})
    cc = sm.OLS.from_formula("cost ~ frailty + age", df.dropna()).fit().params["frailty"]
    mi = mice_regression(df, "cost ~ frailty + age").set_index("term")
    print(f"missing frailty {miss.mean():.0%}; complete-case {cc:.3f}; MICE {mi.loc['frailty', 'coef']:.3f} (true 2.0)")
    checks = {
        "complete-case estimate is biased (missingness depends on the outcome)": abs(cc - 2.0) > 0.1,
        "MICE is closer to the truth": abs(mi.loc["frailty", "coef"] - 2.0) < abs(cc - 2.0),
        "MICE CI covers 2.0": mi.loc["frailty", "ci_lo"] <= 2.0 <= mi.loc["frailty", "ci_hi"],
        "fraction of missing information reported (> 0)": mi.loc["frailty", "fmi"] > 0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
