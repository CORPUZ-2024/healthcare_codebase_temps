"""
Predictive risk stratification: who is likely to be hospitalized, and how to tier them for care.

    fit_logistic               (STANDARD)  vs  fit_gradient_boosting       (ALTERNATIVE)
    tiers_by_capacity          (STANDARD)  vs  segments_kmeans             (ALTERNATIVE)
    + events_per_variable, detect_separation, temporal_split, evaluate (AUC, Brier,
      calibration, PPV@top-k), calibration_table, recalibrate_intercept, rising_risk
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Pre-fit checks and splitting
# ---------------------------------------------------------------------------

def events_per_variable(y: pd.Series, n_predictors: int) -> float:
    """Events per variable (EPV) = number of the RARER outcome class / number of predictors.

    Rule of thumb: EPV >= 10 (>= 20 is safer). Below that, logistic coefficients are unstable and
    the model overfits. Run this BEFORE fitting.
    """
    events = int(min(y.sum(), len(y) - y.sum()))
    return events / max(n_predictors, 1)


def detect_separation(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Binary predictors that perfectly predict the outcome in one of their levels.

    A separated logistic model "converges" with enormous coefficients and meaningless p-values.
    Fix: drop/merge the level, or use penalized regression (the default here is L2-penalized).
    """
    out = []
    for c in X.columns:
        if X[c].nunique() == 2:
            tab = pd.crosstab(X[c], y)
            if (tab == 0).any().any():
                out.append(c)
    return out


def temporal_split(df: pd.DataFrame, train_year: int, test_year: int, time_col: str = "cohort_year"):
    """Train on an earlier cohort, test on a later one — the way the model will be used.

    A random split leaks future patterns into training and overstates performance.
    """
    return df[df[time_col] == train_year].copy(), df[df[time_col] == test_year].copy()


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------

def fit_logistic(X: pd.DataFrame, y: pd.Series, C: float = 1.0):
    """STANDARD: L2-penalized logistic regression on standardized features.

    Healthcare context
    ------------------
    Logistic regression is the default for clinical risk models: coefficients are explainable
    to clinicians ("each prior admission multiplies the odds by 1.6"), it is well calibrated when
    specified sensibly, and reviewers know how to audit it.

    Returns a fitted sklearn Pipeline (StandardScaler -> LogisticRegression). Use
    ``odds_ratios`` to explain it.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2_000))
    return model.fit(X, y)


def odds_ratios(model, feature_names: list[str], X: pd.DataFrame) -> pd.DataFrame:
    """Odds ratio per ONE-UNIT change in each raw feature (coefficient / feature SD, exponentiated)."""
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    beta_raw = lr.coef_[0] / scaler.scale_
    return (pd.DataFrame({"feature": feature_names, "coef_per_unit": beta_raw, "odds_ratio_per_unit": np.exp(beta_raw)})
            .sort_values("odds_ratio_per_unit", ascending=False).reset_index(drop=True))


def fit_gradient_boosting(X: pd.DataFrame, y: pd.Series, calibrate: bool = True, seed: int = 0):
    """ALTERNATIVE: histogram gradient boosting (sklearn's LightGBM-style trees) + isotonic calibration.

    Trade-off
    ---------
    + Finds thresholds and interactions automatically (e.g. "3+ ED visits", "heart failure AND
      lives alone"); usually higher AUC on claims features. XGBoost/LightGBM are drop-in
      equivalents if installed.
    - Harder to explain (use permutation importance / SHAP), easier to overfit, and raw
      probabilities are often miscalibrated — hence the isotonic step, fitted on a held-out
      slice of the training data.

    Returns a dict {"model", "calibrator" or None}; use ``predict_proba_any``.
    """
    rng = np.random.default_rng(seed)
    cal_mask = rng.random(len(X)) < 0.25 if calibrate else np.zeros(len(X), bool)
    # shallow trees + large leaves: with a few hundred events, deep trees memorize noise
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=2,
                                        min_samples_leaf=80, random_state=seed)
    gb.fit(X[~cal_mask], y[~cal_mask])
    iso = None
    if calibrate:
        iso = IsotonicRegression(out_of_bounds="clip").fit(gb.predict_proba(X[cal_mask])[:, 1], y[cal_mask])
    return {"model": gb, "calibrator": iso}


def predict_proba_any(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability from either model type."""
    if isinstance(model, dict):
        p = model["model"].predict_proba(X)[:, 1]
        return model["calibrator"].predict(p) if model["calibrator"] is not None else p
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------

def calibration_table(y: pd.Series, p: np.ndarray, groups: int = 10) -> pd.DataFrame:
    """Observed vs. mean predicted rate by decile of predicted risk (1 = lowest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["decile"] = pd.qcut(pd.Series(p).rank(method="first"), groups, labels=False) + 1
    t = df.groupby("decile").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")).reset_index()
    t["obs_to_pred"] = t["observed"] / t["predicted"]
    return t


def calibration_slope_intercept(y: pd.Series, p: np.ndarray) -> dict:
    """Fit logit(y) ~ a + b * logit(p). Perfect calibration: slope 1, intercept 0.
    Slope < 1 = predictions too extreme (overfit); intercept != 0 = overall level off (drift)."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1_000).fit(lp, np.asarray(y))
    return {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0])}


def evaluate(y: pd.Series, p: np.ndarray, top_pct: float = 0.05) -> dict:
    """AUC (discrimination), Brier (overall accuracy), calibration, and PPV/sensitivity if care
    managers can only reach the top ``top_pct`` of members."""
    y = np.asarray(y)
    k = max(1, int(round(len(p) * top_pct)))
    top = np.argsort(-p)[:k]
    ppv = y[top].mean()
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "base_rate": float(y.mean()), "mean_pred": float(p.mean()),
            f"ppv_top{int(top_pct * 100)}": float(ppv), f"sensitivity_top{int(top_pct * 100)}": float(y[top].sum() / y.sum()),
            f"lift_top{int(top_pct * 100)}": float(ppv / y.mean()), **calibration_slope_intercept(y, p)}


def recalibrate_intercept(p: np.ndarray, target_rate: float) -> np.ndarray:
    """Shift predictions on the logit scale so their mean matches a new base rate (drift fix).

    Use when the ranking is still good (AUC stable) but the level has moved (intercept != 0).
    """
    from scipy.optimize import brentq
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    f = lambda d: (1 / (1 + np.exp(-(lp + d)))).mean() - target_rate  # noqa: E731
    delta = brentq(f, -10, 10)
    return 1 / (1 + np.exp(-(lp + delta)))


# ---------------------------------------------------------------------------
# 3. Tiers and segments
# ---------------------------------------------------------------------------

def tiers_by_capacity(member_ids: pd.Series, p: np.ndarray, tier_shares=(0.05, 0.15)) -> pd.DataFrame:
    """STANDARD: tiers set by care-management CAPACITY on predicted risk rank.

    Tier 1 = top 5% (intensive), Tier 2 = next 15% (rising risk / telephonic), Tier 3 = everyone
    else. Capacity-based tiers guarantee the caseload fits the team, whatever the base rate.
    """
    r = pd.Series(p).rank(ascending=False, method="first") / len(p)
    t1, t2 = tier_shares[0], tier_shares[0] + tier_shares[1]
    tier = np.where(r <= t1, 1, np.where(r <= t2, 2, 3))
    return pd.DataFrame({"member_id": np.asarray(member_ids), "risk": p, "tier": tier})


def segments_kmeans(X: pd.DataFrame, k_range=range(2, 7), seed: int = 0) -> dict:
    """ALTERNATIVE: k-means segments on standardized features; K chosen by silhouette.

    Trade-off
    ---------
    + Describes WHO members are (e.g. "young, high ED, low adherence" vs "older, heart failure,
      lives alone"), which is what intervention design needs — a tier only says how risky.
    - Segments are not ordered by risk, change when features change, and K is a judgment call
      (silhouette is a guide). Label segments by profiling them, and re-check stability.

    Returns {"k", "labels", "silhouette_by_k", "profile"}.
    """
    Z = StandardScaler().fit_transform(X)
    scores, fits = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sample = np.random.default_rng(seed).choice(len(Z), size=min(3_000, len(Z)), replace=False)
        scores[k] = float(silhouette_score(Z[sample], km.labels_[sample]))
        fits[k] = km
    best = max(scores, key=scores.get)
    labels = fits[best].labels_
    profile = X.assign(segment=labels).groupby("segment").mean().round(2)
    profile["n"] = pd.Series(labels).value_counts().sort_index()
    return {"k": best, "labels": labels, "silhouette_by_k": scores, "profile": profile}


def rising_risk(prev: pd.DataFrame, curr: pd.DataFrame, min_increase: float = 0.05, max_tier: int = 3) -> pd.DataFrame:
    """Members whose predicted risk rose by >= ``min_increase`` (absolute) and who are not yet in
    an intensive tier — the classic 'rising risk' outreach list."""
    m = prev[["member_id", "risk"]].merge(curr[["member_id", "risk", "tier"]], on="member_id", suffixes=("_prev", "_curr"))
    m["risk_change"] = m["risk_curr"] - m["risk_prev"]
    return m[(m["risk_change"] >= min_increase) & (m["tier"] >= 2) & (m["tier"] <= max_tier)].sort_values(
        "risk_change", ascending=False).reset_index(drop=True)
