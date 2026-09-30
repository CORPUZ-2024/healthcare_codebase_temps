"""
Provider benchmarking: who is really better or worse, after case mix and chance?

    risk_model -> oe_table (indirect standardization) -> funnel_limits -> eb_shrink_oe   (STANDARD)
        vs  mixed_logistic_ratios (random-intercept logistic, predicted / expected)      (ALTERNATIVE)
    + eb_beta_binomial (unadjusted rates), benchmark_percentile, funnel_plot

Three sources of apparent differences between providers: case mix (who they treat), chance
(how many they treat) and quality. Risk adjustment removes the first; funnel limits and shrinkage
deal with the second; what is left is the signal.
"""
from __future__ import annotations

import re as re_
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

RISK_COVARIATES = ["age", "comorbidity_cnt", "prior_admit_flag"]

# ---------------------------------------------------------------------------
# 1. Risk model and O/E (STANDARD)
# ---------------------------------------------------------------------------

def risk_model(patients: pd.DataFrame, covariates: list[str] = RISK_COVARIATES, y: str = "readmit_flag") -> tuple[pd.Series, dict]:
    """Patient-level logistic model WITHOUT provider terms -> expected probability for each patient.

    Returns (expected probability Series, dict with c_statistic and calibration O/E overall).
    The model deliberately ignores the provider: E = what this patient would experience at an
    average provider.
    """
    fit = smf.glm(f"{y} ~ " + " + ".join(covariates), data=patients, family=sm.families.Binomial()).fit()
    p = fit.predict(patients)
    yv = patients[y].to_numpy()
    order = np.argsort(p.to_numpy())
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = yv.sum()
    c = (ranks[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yv) - n1))    # Mann-Whitney AUC
    return p, {"c_statistic": float(c), "overall_oe": float(yv.sum() / p.sum())}


def oe_table(patients: pd.DataFrame, expected: pd.Series, y: str = "readmit_flag", provider: str = "provider_id",
             alpha: float = 0.05) -> pd.DataFrame:
    """Indirect standardization: per provider observed, expected, O/E with exact Poisson CI, crude and
    risk-standardized rate (RSR = O/E x overall rate).

    Healthcare context
    ------------------
    O/E answers "did this provider have more events than expected FOR ITS OWN PATIENTS?". It is
    not a league table: O/Es from providers with different case mix are not directly comparable
    (indirect standardization), but each is a fair comparison with the average.

    >>> pts = pd.DataFrame({"provider_id": ["A"] * 4 + ["B"] * 4, "readmit_flag": [1, 1, 0, 0, 1, 0, 0, 0]})
    >>> oe_table(pts, pd.Series([0.25] * 8))[["provider_id", "observed", "expected", "oe"]].values.tolist()
    [['A', 2, 1.0, 2.0], ['B', 1, 1.0, 1.0]]

    Common mistakes
    ---------------
    - Ranking providers on crude rates (ranks sicker-patient providers as worse).
    - Ranking on O/E point estimates without intervals: a 12-case provider tops or bottoms every list.
    - Fitting the risk model WITH provider effects and then computing "expected" from it.
    """
    d = patients.assign(_e=expected.to_numpy())
    g = d.groupby(provider).agg(n_cases=(y, "size"), observed=(y, "sum"), expected=("_e", "sum")).reset_index()
    overall = d[y].mean()
    g["oe"] = g["observed"] / g["expected"]
    lo = stats.chi2.ppf(alpha / 2, 2 * g["observed"]) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (g["observed"] + 1)) / 2
    g["oe_lo"], g["oe_hi"] = np.nan_to_num(lo) / g["expected"], hi / g["expected"]
    g["crude_rate"] = g["observed"] / g["n_cases"]
    g["rsr"] = g["oe"] * overall
    g["observed"] = g["observed"].astype(int)
    return g


def funnel_limits(expected: np.ndarray, levels=(0.95, 0.998)) -> pd.DataFrame:
    """Exact Poisson control limits for O/E at each expected count (the funnel).

    Under 'no provider effect', O ~ Poisson(E); limits = Poisson quantiles / E. 95% ~ 2 SD ('warning'),
    99.8% ~ 3 SD ('alarm'). Returns expected, lo_<level>, hi_<level>.
    """
    e = np.asarray(expected, float)
    out = {"expected": e}
    for lv in levels:
        a = (1 - lv) / 2
        out[f"lo_{lv}"] = stats.poisson.ppf(a, e) / e
        out[f"hi_{lv}"] = stats.poisson.ppf(1 - a, e) / e
    return pd.DataFrame(out)


def funnel_flags(oe: pd.DataFrame, levels=(0.95, 0.998)) -> pd.DataFrame:
    """Add flag_<level> = -1 (better than limit), 0, +1 (worse) for each provider."""
    lim = funnel_limits(oe["expected"].to_numpy(), levels)
    out = oe.copy()
    for lv in levels:
        out[f"flag_{lv}"] = np.select([out["oe"] > lim[f"hi_{lv}"].to_numpy(), out["oe"] < lim[f"lo_{lv}"].to_numpy()], [1, -1], 0)
    return out


def eb_shrink_oe(oe: pd.DataFrame) -> pd.DataFrame:
    """STANDARD: empirical-Bayes (Poisson-gamma) shrinkage of O/E toward 1.

    Healthcare context
    ------------------
    A 12-case provider with 5 events has O/E ~2.5 - mostly luck. Shrinkage pulls each O/E toward 1
    in proportion to how little data it rests on: posterior mean = (O + a) / (E + a), where a is
    estimated from how much TRUE variation there is across providers. Reliability = E / (E + a).

    Steps
    -----
    1. Between-provider variance of true ratios: tau^2 = E-weighted var(O/E) - weighted mean of 1/E
       (observed spread minus Poisson noise; method of moments, floored at ~0).
    2. a = 1 / tau^2 (large a = little real variation = strong shrinkage).
    3. Posterior Gamma(O + a, E + a): mean and 95% interval; reliability E / (E + a).

    Same idea as the beta-binomial for unadjusted rates (eb_beta_binomial).
    """
    o, e = oe["observed"].to_numpy(float), oe["expected"].to_numpy(float)
    r = o / e
    w = e / e.sum()
    var_r = float(np.sum(w * (r - np.sum(w * r)) ** 2))
    sampling = float(np.sum(w / e))                    # expected Poisson noise in O/E, same weights (= n / sum E)
    tau2 = max(var_r - sampling, 1e-6)                                                         # step 1
    a = 1 / tau2                                                                               # step 2
    out = oe.copy()
    out["eb_oe"] = (o + a) / (e + a)                                                           # step 3
    out["eb_lo"] = stats.gamma.ppf(0.025, o + a, scale=1 / (e + a))
    out["eb_hi"] = stats.gamma.ppf(0.975, o + a, scale=1 / (e + a))
    out["reliability"] = e / (e + a)
    out.attrs["tau2"], out.attrs["a"] = tau2, a
    return out


def eb_beta_binomial(events: pd.Series, n: pd.Series) -> pd.DataFrame:
    """Beta-binomial EB for UNADJUSTED rates: fit Beta(alpha, beta) by maximum likelihood, then
    posterior mean (k + alpha) / (n + alpha + beta). Use when no risk adjustment is appropriate
    (process measures, e.g. % visits documented)."""
    from scipy.optimize import minimize

    k, m = events.to_numpy(float), n.to_numpy(float)

    def nll(x):
        al, be = np.exp(x)
        return -stats.betabinom.logpmf(k, m, al, be).sum()

    p0 = k.sum() / m.sum()
    res = minimize(nll, np.log([p0 * 20, (1 - p0) * 20]), method="Nelder-Mead")
    al, be = np.exp(res.x)
    return pd.DataFrame({"raw_rate": k / m, "eb_rate": (k + al) / (m + al + be), "alpha": al, "beta": be})


# ---------------------------------------------------------------------------
# 2. ALTERNATIVE: hierarchical (mixed-effects) logistic model
# ---------------------------------------------------------------------------

def mixed_logistic_ratios(patients: pd.DataFrame, covariates: list[str] = RISK_COVARIATES, y: str = "readmit_flag",
                          provider: str = "provider_id") -> pd.DataFrame:
    """ALTERNATIVE: random-intercept logistic regression (statsmodels BinomialBayesMixedGLM, variational
    Bayes) -> predicted / expected ratio per provider (the CMS hospital-measure construction).

    Trade-off
    ---------
    + Estimates risk adjustment and provider effects JOINTLY (case-mix coefficients aren't distorted by
      providers that treat sicker patients); shrinkage comes from the model; standard in CMS measures.
    - Heavier to fit and explain; variational estimates are approximate (use lme4/SAS for production);
      reliability depends on the random-effect variance being estimated well (few providers = unstable).

    Returns provider_id, re_mean (log-odds effect), pe_ratio = sum p(with effect) / sum p(effect = 0).
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = patients.copy()
    model = BinomialBayesMixedGLM.from_formula(f"{y} ~ " + " + ".join(covariates), {"prov": f"0 + C({provider})"}, d)
    with warnings.catch_warnings():                 # VB often reports non-convergence at default tolerances;
        warnings.simplefilter("ignore")             # estimates are stable here (tested vs. truth) but approximate
        fit = model.fit_vb()
    re = pd.Series(fit.vc_mean, index=[re_.search(r"\[(?:T\.)?(.+)\]", n).group(1) for n in fit.model.vc_names])
    fe = fit.fe_mean
    X = sm.add_constant(d[covariates].astype(float)).to_numpy()
    lp0 = X @ fe
    d["_u"] = d[provider].map(re).fillna(0.0).to_numpy()
    d["_p1"] = 1 / (1 + np.exp(-(lp0 + d["_u"])))
    d["_p0"] = 1 / (1 + np.exp(-lp0))
    g = d.groupby(provider).agg(re_mean=("_u", "first"), p1=("_p1", "sum"), p0=("_p0", "sum")).reset_index()
    g["pe_ratio"] = g["p1"] / g["p0"]
    return g[[provider, "re_mean", "pe_ratio"]]


# ---------------------------------------------------------------------------
# 3. Benchmarks and plots
# ---------------------------------------------------------------------------

def benchmark_percentile(value: float, reference: pd.Series) -> dict:
    """Where a value falls in an external distribution (e.g. national HRRP excess readmission ratios)."""
    r = reference.dropna().to_numpy()
    return {"n_reference": int(len(r)), "percentile": float(stats.percentileofscore(r, value, kind="mean")),
            "p25": float(np.percentile(r, 25)), "median": float(np.median(r)), "p75": float(np.percentile(r, 75))}


def estimation_error(estimates: pd.Series, truth: pd.Series) -> dict:
    """RMSE of log ratios vs. the truth and Spearman rank correlation (synthetic data only)."""
    le, lt = np.log(estimates.to_numpy(float)), np.log(truth.to_numpy(float))
    ok = np.isfinite(le)
    return {"rmse_log": float(np.sqrt(np.mean((le[ok] - lt[ok]) ** 2))),
            "spearman": float(stats.spearmanr(estimates, truth).statistic)}


def funnel_plot(oe: pd.DataFrame, path, levels=(0.95, 0.998), value_col: str = "oe") -> None:
    """Save a funnel plot (O/E vs expected with Poisson limits). Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(max(oe["expected"].min() * 0.8, 0.5), oe["expected"].max() * 1.05, 300)
    lim = funnel_limits(grid, levels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lv, ls in zip(levels, ("--", ":")):
        ax.plot(grid, lim[f"hi_{lv}"], "k" + ls, lw=1, label=f"{lv:.1%} limits")
        ax.plot(grid, lim[f"lo_{lv}"], "k" + ls, lw=1)
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.scatter(oe["expected"], oe[value_col], s=14)
    ax.set_xlabel("expected events")
    ax.set_ylabel("observed / expected")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
