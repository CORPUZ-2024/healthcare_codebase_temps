"""
Patient- and caregiver-reported outcomes: score instruments correctly, then analyze change over time.

    score_instrument (YAML rules: reverse items, prorating)     (scoring - always)
    mixed_model_effect (MMRM-style linear mixed model)          (STANDARD)
        vs  ancova_change (completers)  and  gee_effect          (ALTERNATIVE)
    + responder_analysis (MCID), cronbach_alpha, floor_ceiling, severity_band
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------

def score_instrument(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Total score per row following the instrument's rules.

    Healthcare context
    ------------------
    Scores feed clinical decisions and program evaluations, so scoring must follow the instrument
    manual exactly: reverse positively worded items, prorate only when few items are missing, and
    leave the score missing otherwise (never treat a skipped item as 0).

    Steps
    -----
    1. Reverse items listed in ``reverse_items``: x -> item_max + item_min - x.
    2. Count missing items; if <= ``max_missing_items`` prorate: mean(answered) x number of items.
    3. Otherwise the total is NaN (unscorable).

    Returns DataFrame (same index): score, n_items_missing, prorated_flag.

    >>> spec = {"items": ["a", "b", "c"], "item_min": 0, "item_max": 4, "reverse_items": ["c"], "max_missing_items": 1}
    >>> score_instrument(pd.DataFrame({"a": [4, 4, None], "b": [2, None, None], "c": [0, 0, 1]}), spec)["score"].tolist()
    [10.0, 12.0, nan]

    Common mistakes
    ---------------
    - Summing without reversing (a caregiver who copies well scores as burdened).
    - fillna(0) before summing: understates scores for anyone who skipped an item.
    - Prorating a scale whose manual forbids it, or prorating when half the items are missing.
    """
    items = spec["items"]
    x = df[items].astype(float).copy()
    for it in spec.get("reverse_items", []):                                   # step 1
        x[it] = spec["item_max"] + spec["item_min"] - x[it]
    n_miss = x.isna().sum(axis=1)                                              # step 2
    score = x.mean(axis=1) * len(items)
    score = score.where(n_miss <= spec.get("max_missing_items", 0))            # step 3
    return pd.DataFrame({"score": score, "n_items_missing": n_miss, "prorated_flag": ((n_miss > 0) & score.notna()).astype(int)},
                        index=df.index)


def severity_band(score: pd.Series, spec: dict) -> pd.Series:
    """Label each score with the instrument's band (lower bounds inclusive); NaN stays NaN."""
    bands = sorted(spec["bands"], key=lambda b: b["min"])
    edges = [b["min"] for b in bands] + [np.inf]
    return pd.cut(score, bins=edges, right=False, labels=[b["label"] for b in bands])


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Internal consistency: k/(k-1) x (1 - sum item variances / variance of total). Complete rows only.

    >>> round(cronbach_alpha(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})), 3)
    1.0
    """
    x = items.dropna()
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def reversed_items(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Item columns with reverse-worded items flipped (for reliability and item analysis)."""
    x = df[spec["items"]].astype(float).copy()
    for it in spec.get("reverse_items", []):
        x[it] = spec["item_max"] + spec["item_min"] - x[it]
    return x


def floor_ceiling(score: pd.Series, spec: dict) -> dict:
    """Share of scores at the minimum and maximum possible total (> 15% limits sensitivity to change)."""
    k = len(spec["items"])
    s = score.dropna()
    return {"floor_share": float((s <= k * spec["item_min"]).mean()), "ceiling_share": float((s >= k * spec["item_max"]).mean())}


# ---------------------------------------------------------------------------
# 2. Change and responders
# ---------------------------------------------------------------------------

def to_wide(scored: pd.DataFrame, id_col: str = "caregiver_id", visit_col: str = "visit_month") -> pd.DataFrame:
    """One row per person: arm_flag, score_<visit> columns, change_<visit> vs. baseline."""
    w = scored.pivot_table(index=[id_col, "arm_flag"], columns=visit_col, values="score").reset_index()
    visits = [c for c in w.columns if isinstance(c, (int, np.integer))]
    w = w.rename(columns={v: f"score_{v}" for v in visits})
    base = f"score_{min(visits)}"
    for v in visits:
        if v != min(visits):
            w[f"change_{v}"] = w[f"score_{v}"] - w[base]
    return w


def responder_analysis(wide: pd.DataFrame, mcid: float, visit: int = 6, higher_is_worse: bool = True,
                       missing_as_nonresponder: bool = False) -> pd.DataFrame:
    """Share improving by at least the MCID, by arm.

    ``missing_as_nonresponder`` = True counts dropouts as non-responders (conservative, common in
    regulatory PRO analyses); False analyzes observed changes only.
    Returns arm_flag, n, responders, responder_rate, and the arm difference as an attribute.
    """
    ch = wide[f"change_{visit}"]
    improved = (-ch if higher_is_worse else ch) >= mcid
    d = wide.assign(responder=improved.astype(float).where(ch.notna(), 0.0 if missing_as_nonresponder else np.nan))
    d = d.dropna(subset=["responder"])
    out = d.groupby("arm_flag")["responder"].agg(n="size", responders="sum").reset_index()
    out["responder_rate"] = out["responders"] / out["n"]
    out.attrs["difference"] = float(out.loc[out["arm_flag"] == 1, "responder_rate"].iloc[0] -
                                    out.loc[out["arm_flag"] == 0, "responder_rate"].iloc[0])
    return out


# ---------------------------------------------------------------------------
# 3. Treatment effect over time
# ---------------------------------------------------------------------------

def _post_baseline_long(scored: pd.DataFrame, id_col: str = "caregiver_id") -> pd.DataFrame:
    base = scored[scored["visit_month"] == scored["visit_month"].min()][[id_col, "score"]].rename(columns={"score": "baseline"})
    post = scored[scored["visit_month"] > scored["visit_month"].min()].merge(base, on=id_col)
    return post.dropna(subset=["score", "baseline"])


def mixed_model_effect(scored: pd.DataFrame, visit: int = 6, id_col: str = "caregiver_id", z: float = 1.96) -> dict:
    """STANDARD: MMRM-style linear mixed model on post-baseline scores (statsmodels MixedLM, REML).

    score ~ baseline + C(visit) * arm, random intercept per person. Uses every observed visit, so a
    caregiver who dropped out after month 3 still informs the month-6 estimate through their month-3
    score: valid under missing-at-random dropout.

    Note: a true MMRM uses an UNSTRUCTURED within-person covariance (R ``mmrm``, SAS PROC MIXED);
    statsmodels' random intercept is compound symmetry. With 2 post-baseline visits the two are close.

    Returns dict: effect (arm difference at ``visit``), se, ci_lo, ci_hi, p_value, n_people, n_obs.

    Common mistakes
    ---------------
    - Analyzing only completers (see ancova_change) when dropout depends on how people were doing.
    - Putting baseline in the outcome vector AND as a covariate.
    - Reporting the arm main effect (the difference at the reference visit) instead of the target visit.
    """
    d = _post_baseline_long(scored, id_col)
    d["visit_cat"] = d["visit_month"].astype(str)
    ref = str(visit)
    fit = smf.mixedlm(f"score ~ baseline + C(visit_cat, Treatment('{ref}')) * arm_flag", d, groups=d[id_col]).fit(reml=True)
    b, se = float(fit.params["arm_flag"]), float(fit.bse["arm_flag"])
    return {"effect": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "p_value": float(fit.pvalues["arm_flag"]),
            "n_people": int(d[id_col].nunique()), "n_obs": int(len(d))}


def ancova_change(wide: pd.DataFrame, visit: int = 6, z: float = 1.96) -> dict:
    """ALTERNATIVE: ANCOVA on change at one visit, completers only: change ~ baseline + arm (HC1 SEs).

    Trade-off
    ---------
    + The simplest defensible analysis for a randomized 2-arm study with little dropout; baseline
      adjustment removes regression to the mean and gains power over a raw change comparison.
    - Uses only people observed at ``visit``: biased when dropout depends on earlier outcomes (MAR);
      discards information from partial completers.
    """
    d = wide.dropna(subset=[f"change_{visit}", "score_0"])
    fit = smf.ols(f"change_{visit} ~ score_0 + arm_flag", data=d).fit(cov_type="HC1")
    b, se = float(fit.params["arm_flag"]), float(fit.bse["arm_flag"])
    return {"effect": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "n_people": int(len(d))}


def gee_effect(scored: pd.DataFrame, visit: int = 6, id_col: str = "caregiver_id", z: float = 1.96) -> dict:
    """ALTERNATIVE: GEE (exchangeable working correlation), same mean model as the mixed model.

    Trade-off
    ---------
    + Population-average interpretation with robust SEs that don't depend on getting the correlation right.
    - Unweighted GEE is valid only if dropout is MISSING COMPLETELY at random; under MAR it is biased
      (weighted GEE with dropout weights fixes this - not implemented).
    """
    d = _post_baseline_long(scored, id_col)
    d["visit_cat"] = d["visit_month"].astype(str)
    fit = smf.gee(f"score ~ baseline + C(visit_cat, Treatment('{visit}')) * arm_flag", id_col, d,
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = float(fit.params["arm_flag"]), float(fit.bse["arm_flag"])
    return {"effect": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se}


def arm_mean_change(scored: pd.DataFrame, arm: int, visit: int = 6, id_col: str = "caregiver_id") -> dict:
    """Mean change from baseline in ONE arm at ``visit``: completers' raw mean vs. the mixed model's estimate.

    Healthcare context
    ------------------
    Single-arm programs report "caregivers improved by X points". Computed on completers, X is too
    good when people who were getting worse stopped answering. The mixed model estimate uses their
    earlier visits (MAR) and predicts at every enrolled person's baseline.

    Returns dict: completers_mean_change, completers_n, mixed_model_mean_change, enrolled_n.
    """
    w = to_wide(scored, id_col)
    wa = w[w["arm_flag"] == arm]
    d = _post_baseline_long(scored, id_col)
    d["visit_cat"] = d["visit_month"].astype(str)
    fit = smf.mixedlm(f"score ~ baseline + C(visit_cat, Treatment('{visit}')) * arm_flag", d, groups=d[id_col]).fit(reml=True)
    newx = wa[["score_0"]].dropna().rename(columns={"score_0": "baseline"}).assign(visit_cat=str(visit), arm_flag=arm)
    return {"completers_mean_change": float(wa[f"change_{visit}"].mean()), "completers_n": int(wa[f"change_{visit}"].notna().sum()),
            "mixed_model_mean_change": float((fit.predict(newx) - newx["baseline"]).mean()), "enrolled_n": int(len(newx))}
