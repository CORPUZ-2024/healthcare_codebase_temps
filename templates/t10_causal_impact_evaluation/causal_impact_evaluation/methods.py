"""
Causal impact evaluation for programs and policies.

    did_twfe + event_study            (STANDARD, panel)          vs  its_segmented, GLSAR SEs (ALTERNATIVE, one series)
    psm_att + smd_table               (STANDARD, cross-section)  vs  ipw / aipw_ate (ALTERNATIVE)
    + naive_difference (what NOT to report), propensity_logit, durbin_watson

Estimands: ATE = average effect if everyone were treated; ATT = average effect on those who were.
They differ when the effect is heterogeneous and treatment is targeted (sicker members referred).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

COVARIATES = ["age", "chronic_cnt", "risk_z", "prior_util", "rural_flag"]

# ---------------------------------------------------------------------------
# 1. Panel: difference-in-differences and event study (STANDARD)
# ---------------------------------------------------------------------------

def did_twfe(panel: pd.DataFrame, y: str = "y", unit: str = "practice_id", time: str = "month_idx", z: float = 1.96) -> dict:
    """STANDARD: two-way fixed-effects DiD, y ~ treated x post + unit FE + time FE, SEs clustered by unit.

    Healthcare context
    ------------------
    A workflow rolled out in some practices on one date. Unit fixed effects absorb permanent
    differences between practices (the treated ones start higher here); time fixed effects absorb
    seasonality and trends common to all. The treated x post coefficient is the effect - IF the
    groups would have moved in parallel without the change (check with event_study).

    Returns dict: effect, se, ci_lo, ci_hi, p_value, n_clusters.

    Common mistakes
    ---------------
    - Plain OLS standard errors: months within a practice are correlated; cluster by the unit that
      was assigned treatment.
    - Staggered adoption with TWFE: when units adopt at different dates and effects change over
      time, TWFE mixes in "forbidden comparisons" (Goodman-Bacon). Use a staggered-robust
      estimator (Callaway-Sant'Anna, Sun-Abraham) - not implemented here; one adoption date only.
    - Fewer than ~20 clusters: cluster-robust SEs are too small (checks.check_few_clusters).
    """
    d = panel.assign(treat_post=panel["treated_flag"] * panel["post_flag"])
    fit = smf.ols(f"{y} ~ treat_post + C({unit}) + C({time})", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[unit])[0]})
    b, se = float(fit.params["treat_post"]), float(fit.bse["treat_post"])
    return {"effect": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "p_value": float(fit.pvalues["treat_post"]),
            "n_clusters": int(d[unit].nunique())}


def event_study(panel: pd.DataFrame, y: str = "y", unit: str = "practice_id", time: str = "month_idx",
                ref: int = -1, window: tuple[int, int] = (-6, 6), z: float = 1.96) -> tuple[pd.DataFrame, float]:
    """Event-study DiD: treated x relative-month coefficients (reference month ``ref``), endpoints binned.

    Pre-period coefficients ("leads") near zero support parallel trends; the joint Wald test of all
    leads = 0 is the formal pre-trend test. Post coefficients show how the effect builds or fades.
    Relative months beyond ``window`` are pooled into the endpoint bins.

    Why bin: with one coefficient per month (23 leads here) and only 40 clusters, the cluster-robust
    joint test rejects parallel trends that DO hold (tested: test_event_study_unbinned_overrejects).

    Returns (DataFrame rel_month, coef, ci_lo, ci_hi; p-value of the joint pre-trend test).
    """
    d = panel.copy()
    d["rel_bin"] = d["rel_month"].clip(window[0], window[1])
    rels = sorted(r for r in d["rel_bin"].unique() if r != ref)
    names = []
    for r in rels:
        nm = f"ev_m{abs(r)}" if r < 0 else f"ev_p{r}"
        d[nm] = ((d["rel_bin"] == r) & (d["treated_flag"] == 1)).astype(int)
        names.append((r, nm))
    fit = smf.ols(f"{y} ~ " + " + ".join(n for _, n in names) + f" + C({unit}) + C({time})", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[unit])[0]})
    rows = [{"rel_month": r, "coef": float(fit.params[n]), "ci_lo": float(fit.params[n] - z * fit.bse[n]),
             "ci_hi": float(fit.params[n] + z * fit.bse[n])} for r, n in names]
    rows.append({"rel_month": ref, "coef": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    leads = [n for r, n in names if r < ref]
    p = float(fit.wald_test(", ".join(f"{n} = 0" for n in leads), scalar=True).pvalue) if leads else float("nan")
    return pd.DataFrame(rows).sort_values("rel_month").reset_index(drop=True), p


def naive_post_comparison(panel: pd.DataFrame, y: str = "y") -> float:
    """Treated minus control in the post period only. Biased by any pre-existing gap (don't report)."""
    post = panel[panel["post_flag"] == 1]
    return float(post.loc[post["treated_flag"] == 1, y].mean() - post.loc[post["treated_flag"] == 0, y].mean())


# ---------------------------------------------------------------------------
# 2. Cross-section: propensity score matching (STANDARD)
# ---------------------------------------------------------------------------

def propensity_logit(df: pd.DataFrame, covariates: list[str] = COVARIATES) -> pd.Series:
    """Logit of P(treated | X) from scikit-learn LogisticRegression (effectively unpenalized: C = 1e12;
    penalty=None is deprecated from scikit-learn 1.8), standardized X."""
    X = df[covariates].astype(float)
    Xs = (X - X.mean()) / X.std(ddof=0)
    m = LogisticRegression(C=1e12, max_iter=2_000).fit(Xs, df["treated_flag"])
    return pd.Series(m.decision_function(Xs), index=df.index, name="ps_logit")


def smd_table(df: pd.DataFrame, covariates: list[str], weights: pd.Series | None = None,
              matched: pd.DataFrame | None = None) -> pd.DataFrame:
    """Standardized mean differences: raw, and after matching and/or weighting.

    Denominator = pooled SD in the RAW sample (so it doesn't shrink as the sample changes).
    |SMD| < 0.1 is the usual balance threshold.
    """
    t, c = df["treated_flag"] == 1, df["treated_flag"] == 0
    rows = []
    for v in covariates:
        sd = np.sqrt((df.loc[t, v].var() + df.loc[c, v].var()) / 2) or 1.0
        row = {"covariate": v, "smd_raw": float((df.loc[t, v].mean() - df.loc[c, v].mean()) / sd)}
        if matched is not None:
            mt, mc = matched["treated_flag"] == 1, matched["treated_flag"] == 0
            row["smd_matched"] = float((matched.loc[mt, v].mean() - matched.loc[mc, v].mean()) / sd)
        if weights is not None:
            row["smd_weighted"] = float((np.average(df.loc[t, v], weights=weights[t]) -
                                         np.average(df.loc[c, v], weights=weights[c])) / sd)
        rows.append(row)
    return pd.DataFrame(rows)


def psm_att(df: pd.DataFrame, ps_logit: pd.Series, caliper_sd: float = 0.2, y: str = "outcome", seed: int = 0,
            z: float = 1.96) -> tuple[dict, pd.DataFrame]:
    """STANDARD: 1:1 nearest-neighbour matching on the logit PS, caliper 0.2 SD, without replacement -> ATT.

    Healthcare context
    ------------------
    Answers "what did the program do for the members who got it?" by pairing each participant with
    the most similar non-participant. Report balance (smd_table) before the effect.

    Returns (dict: att, se, ci_lo, ci_hi, n_pairs, share_treated_matched; matched rows with pair_id).
    SE from the paired differences - ignores uncertainty in the estimated PS and the matching itself
    (Abadie-Imbens SEs are more exact; bootstrap is NOT valid for NN matching).

    Common mistakes
    ---------------
    - Matching and then reporting an ATE: matching on treated members estimates the ATT.
    - Adding post-treatment variables (utilization after enrollment) to the PS model.
    - Tuning the PS model on the outcome ("p-hacking the match").
    """
    d = df.assign(ps_logit=ps_logit)
    cal = caliper_sd * float(d["ps_logit"].std())
    t_idx = np.random.default_rng(seed).permutation(d.index[d["treated_flag"] == 1].to_numpy())
    c = d[d["treated_flag"] == 0]
    c_ps, c_idx = c["ps_logit"].to_numpy(), c.index.to_numpy()
    used = np.zeros(len(c), bool)
    pairs = []
    for i in t_idx:
        dist = np.abs(c_ps - d.at[i, "ps_logit"])
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= cal:
            used[j] = True
            pairs.append((i, c_idx[j]))
    ti, ci = (np.array(x) for x in zip(*pairs))
    diff = d.loc[ti, y].to_numpy() - d.loc[ci, y].to_numpy()
    att, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
    matched = pd.concat([d.loc[ti].assign(pair_id=np.arange(len(ti))), d.loc[ci].assign(pair_id=np.arange(len(ci)))])
    return ({"att": att, "se": se, "ci_lo": att - z * se, "ci_hi": att + z * se, "n_pairs": int(len(ti)),
             "share_treated_matched": float(len(ti) / len(t_idx))}, matched.reset_index(drop=True))


def naive_difference(df: pd.DataFrame, y: str = "outcome") -> float:
    """Treated mean minus untreated mean, no adjustment. Confounded by who gets referred (don't report)."""
    return float(df.loc[df["treated_flag"] == 1, y].mean() - df.loc[df["treated_flag"] == 0, y].mean())


# ---------------------------------------------------------------------------
# 3. Weighting: IPW and doubly robust AIPW (ALTERNATIVE)
# ---------------------------------------------------------------------------

def _ps(df, covariates, trim):
    e = 1 / (1 + np.exp(-propensity_logit(df, covariates)))
    return e.clip(trim, 1 - trim)


def ipw(df: pd.DataFrame, covariates: list[str] = COVARIATES, estimand: str = "ate", y: str = "outcome",
        trim: float = 0.01, z: float = 1.96) -> tuple[dict, pd.Series]:
    """ALTERNATIVE: inverse-probability weighting (normalized / Hajek), ATE or ATT.

    Trade-off
    ---------
    + Keeps every member (matching discards unmatched ones); targets ATE or ATT explicitly.
    - Extreme weights when PS is near 0 or 1 -> high variance (trim, and check max weight).
    - Same assumption as PSM: no unmeasured confounding. SE here: robust (HC1) WLS, which ignores
      PS estimation - usually slightly conservative for the ATE.

    Weights: ATE  T/e + (1-T)/(1-e);  ATT  T + (1-T) e/(1-e).
    Returns (dict: effect, se, ci_lo, ci_hi, max_weight; weights).
    """
    e, t = _ps(df, covariates, trim), df["treated_flag"]
    w = t / e + (1 - t) / (1 - e) if estimand == "ate" else t + (1 - t) * e / (1 - e)
    fit = sm.WLS(df[y], sm.add_constant(t.astype(float)), weights=w).fit(cov_type="HC1")
    b, se = float(fit.params["treated_flag"]), float(fit.bse["treated_flag"])
    return {"effect": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "estimand": estimand,
            "max_weight_share": float(w.max() / w.sum())}, w


def aipw_ate(df: pd.DataFrame, ps_covariates: list[str] = COVARIATES, outcome_covariates: list[str] = COVARIATES,
             y: str = "outcome", trim: float = 0.01, z: float = 1.96) -> dict:
    """ALTERNATIVE: augmented IPW (doubly robust) ATE with influence-function SE.

    Trade-off
    ---------
    + Consistent if EITHER the propensity model OR the outcome model is right (double robustness);
      usually more efficient than IPW.
    - Two models to specify; if both are wrong it can be worse than either alone; extreme weights still hurt.

    psi_i = m1(x) - m0(x) + T (y - m1(x)) / e(x) - (1 - T)(y - m0(x)) / (1 - e(x));  ATE = mean(psi).
    Outcome models: OLS per arm on ``outcome_covariates``.
    """
    e, t, yv = _ps(df, ps_covariates, trim).to_numpy(), df["treated_flag"].to_numpy(), df[y].to_numpy()
    X = sm.add_constant(df[outcome_covariates].astype(float)).to_numpy()
    m1 = X @ np.linalg.lstsq(X[t == 1], yv[t == 1], rcond=None)[0]
    m0 = X @ np.linalg.lstsq(X[t == 0], yv[t == 0], rcond=None)[0]
    psi = m1 - m0 + t * (yv - m1) / e - (1 - t) * (yv - m0) / (1 - e)
    ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {"effect": ate, "se": se, "ci_lo": ate - z * se, "ci_hi": ate + z * se, "estimand": "ate"}


def outcome_regression_ate(df: pd.DataFrame, covariates: list[str] = COVARIATES, y: str = "outcome") -> float:
    """G-computation with per-arm OLS (no weighting) - biased when the outcome model is misspecified."""
    X = sm.add_constant(df[covariates].astype(float)).to_numpy()
    t, yv = df["treated_flag"].to_numpy(), df[y].to_numpy()
    b1 = np.linalg.lstsq(X[t == 1], yv[t == 1], rcond=None)[0]
    b0 = np.linalg.lstsq(X[t == 0], yv[t == 0], rcond=None)[0]
    return float((X @ b1 - X @ b0).mean())


# ---------------------------------------------------------------------------
# 4. Interrupted time series (ALTERNATIVE when there is no control group)
# ---------------------------------------------------------------------------

def its_segmented(series: pd.DataFrame, y: str = "y", se_method: str = "glsar", hac_lags: int = 3,
                  seasonal: bool = True, z: float = 1.96) -> dict:
    """ALTERNATIVE: segmented regression y ~ time + post + time_since (+ month-of-year) for one series.

    Trade-off
    ---------
    + Needs no comparison group: a policy that hit everyone at once (a statewide PA rule, a benefit change).
    + Separates an immediate level change from a change in slope.
    - Any other change at the same date is confounded with the policy (no control); needs >= ~8
      points on each side (more with seasonality).

    Standard errors (``se_method``)
    -------------------------------
    Monthly rates are autocorrelated, so plain OLS SEs are too small. The textbook fix is Newey-West
    (HAC), but on a 60-month series with AR(1) errors a coverage simulation gave only ~73-77% for a
    nominal 95% interval - no better than OLS. Modelling the AR(1) error directly (feasible GLS,
    statsmodels GLSAR) gave ~90%. Default ``"glsar"``; ``"hac"`` and ``"ols"`` are reported for comparison
    (test_its_glsar_covers_better_than_hac).

    Returns dict: level_change (+ lo/hi), slope_change (+ lo/hi), se_method, se_level_{ols,hac,glsar},
    rho (estimated AR(1)), durbin_watson (OLS residuals), counterfactual DataFrame.
    """
    d = series.copy()
    rhs = "month_idx + post_flag + time_since"
    if seasonal:
        d["moy"] = d["month_start"].dt.month
        rhs += " + C(moy)"
    ols = smf.ols(f"{y} ~ {rhs}", data=d).fit()
    names = list(ols.params.index)
    il, isl = names.index("post_flag"), names.index("time_since")
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
    gls = sm.GLSAR(ols.model.endog, ols.model.exog, rho=1).iterative_fit(maxiter=20)
    fits = {"ols": (ols.params.to_numpy(), ols.bse.to_numpy()), "hac": (hac.params, hac.bse), "glsar": (gls.params, gls.bse)}
    b, se = fits[se_method]
    lvl, slp, se_l, se_s = float(b[il]), float(b[isl]), float(se[il]), float(se[isl])
    cf = d.assign(post_flag=0, time_since=0)
    return {"level_change": lvl, "level_lo": lvl - z * se_l, "level_hi": lvl + z * se_l,
            "slope_change": slp, "slope_lo": slp - z * se_s, "slope_hi": slp + z * se_s, "se_method": se_method,
            "se_level_ols": float(fits["ols"][1][il]), "se_level_hac": float(fits["hac"][1][il]),
            "se_level_glsar": float(fits["glsar"][1][il]), "rho": float(np.ravel(gls.model.rho)[0]),
            "durbin_watson": durbin_watson(ols.resid.to_numpy()),
            "counterfactual": pd.DataFrame({"month_start": d["month_start"], "observed": d[y],
                                            "fitted": ols.fittedvalues, "no_policy": ols.predict(cf)})}


def durbin_watson(resid: np.ndarray) -> float:
    """DW statistic: ~2 = no autocorrelation, < 1.5 = positive autocorrelation (plain SEs too small).

    >>> round(durbin_watson(np.array([1.0, -1.0, 1.0, -1.0])), 2)
    3.0
    """
    return float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
