"""
Operational lifecycle analytics: referral-to-care funnel, prior-authorization (PA) turnaround,
denials, and time to start of care.

    funnel + km_curve (time to start of care, censoring-aware) + pa_metrics   (STANDARD)
        vs  cohort_conversion (monthly cohorts, mature only) + denial_pareto   (ALTERNATIVE)
    + naive_median_days (what NOT to report while cases are still open)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = [("referred", "referral_dt"), ("assessed", "assessment_dt"), ("pa_requested", "pa_request_ts"),
          ("pa_decided", "pa_decision_ts"), ("started_care", "soc_dt")]

# ---------------------------------------------------------------------------
# 1. Funnel
# ---------------------------------------------------------------------------

def funnel(df: pd.DataFrame, as_of=None, min_age_days: int | None = None) -> pd.DataFrame:
    """Counts, stage-to-stage conversion and median days per stage.

    Healthcare context
    ------------------
    Where do referrals stall - intake, authorization, staffing? The funnel is the operations team's
    daily view. With ``min_age_days`` only referrals at least that old are counted, so recent ones
    that simply haven't had time to progress don't depress the conversion rates.

    Returns stage, n, conv_from_prev, conv_from_referral, median_days_from_prev.

    Common mistakes
    ---------------
    - Mixing referrals from last week with last quarter: conversion looks like it's collapsing.
    - Medians of stage durations computed only on cases that finished the stage (see km_curve).
    """
    d = df
    if min_age_days is not None:
        d = df[df["referral_dt"] <= pd.Timestamp(as_of) - pd.Timedelta(days=min_age_days)]
    rows, prev_n, prev_col = [], None, None
    base = len(d)
    for name, col in STAGES:
        reached = d[col].notna()
        n = int(reached.sum())
        med = float((d.loc[reached, col] - d.loc[reached, prev_col]).dt.total_seconds().median() / 86_400) if prev_col else 0.0
        rows.append({"stage": name, "n": n, "conv_from_prev": n / prev_n if prev_n else 1.0, "conv_from_referral": n / base,
                     "median_days_from_prev": med})
        prev_n, prev_col = n, col
    known = d["retained_90d_flag"].notna()
    rows.append({"stage": "retained_90d", "n": int(d.loc[known, "retained_90d_flag"].sum()),
                 "conv_from_prev": float(d.loc[known, "retained_90d_flag"].mean()) if known.any() else np.nan,
                 "conv_from_referral": np.nan, "median_days_from_prev": 90.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Time to start of care (censoring-aware)
# ---------------------------------------------------------------------------

def km_curve(duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier product-limit estimate (no dependency). Returns time, at_risk, events, survival.

    >>> km_curve(np.array([2, 3, 4, 5]), np.array([1, 0, 1, 0]))["survival"].round(3).tolist()
    [0.75, 0.375]
    """
    d, e = np.asarray(duration, float), np.asarray(event, int)
    times = np.unique(d[e == 1])
    at_risk = np.array([(d >= t).sum() for t in times])
    events = np.array([((d == t) & (e == 1)).sum() for t in times])
    return pd.DataFrame({"time": times, "at_risk": at_risk, "events": events, "survival": np.cumprod(1 - events / at_risk)})


def time_to_start(df: pd.DataFrame, as_of, days=(7, 14, 30, 60)) -> dict:
    """STANDARD: KM probability of having started care by day d after referral; open referrals censored
    at the data cut. Returns {'curve': km table, 'by_day': {d: P(started by d)}, 'median_days', 'censored_share'}."""
    cut = pd.Timestamp(as_of)
    started = df["soc_dt"].notna().to_numpy()
    end = np.where(started, df["soc_dt"].fillna(cut), cut)
    dur = (pd.to_datetime(end) - df["referral_dt"]).dt.days.to_numpy().clip(min=0)
    curve = km_curve(dur, started.astype(int))

    def s_at(t):
        s = curve.loc[curve["time"] <= t, "survival"]
        return float(s.iloc[-1]) if len(s) else 1.0

    below = curve[curve["survival"] <= 0.5]
    return {"curve": curve, "by_day": {t: 1 - s_at(t) for t in days},
            "median_days": float(below["time"].iloc[0]) if len(below) else float("inf"),
            "censored_share": float((~started).mean())}


def naive_median_days(df: pd.DataFrame) -> float:
    """Median referral-to-start among those who started. Biased LOW while many are still open (don't report)."""
    s = df.dropna(subset=["soc_dt"])
    return float((s["soc_dt"] - s["referral_dt"]).dt.days.median())


# ---------------------------------------------------------------------------
# 3. Prior authorization (CMS-0057-F style metrics)
# ---------------------------------------------------------------------------

def pa_metrics(df: pd.DataFrame, expedited_target_hours: int = 72, standard_target_hours: int = 168) -> pd.DataFrame:
    """PA metrics by priority, shaped like the CMS-0057-F public-reporting set.

    Healthcare context
    ------------------
    The CMS Interoperability and Prior Authorization final rule (CMS-0057-F) sets PA decision timeframes
    of 72 hours (expedited) and 7 calendar days (standard) from 2026 for payers such as Medicare Advantage
    organizations and Medicaid/CHIP programs and plans, and requires impacted payers to post PA metrics
    publicly (share approved, denied, approved after appeal; average decision time). Which provisions
    apply depends on the line of business - verify for yours. Providers and agencies track the same
    numbers to find the payers and reasons that delay care.

    Returns priority, n_decided, pct_approved, pct_denied, pct_denials_appealed, pct_overturned_of_appealed,
    median_hours, mean_hours, pct_within_target.
    """
    d = df.dropna(subset=["pa_decision_ts"]).copy()
    d["hours"] = (d["pa_decision_ts"] - d["pa_request_ts"]).dt.total_seconds() / 3600
    d["target"] = np.where(d["pa_priority_cd"] == "expedited", expedited_target_hours, standard_target_hours)
    rows = []
    for pr, g in d.groupby("pa_priority_cd"):
        den = g["pa_decision_cd"] == "denied"
        app = g["appeal_flag"] == 1
        rows.append({"priority": pr, "n_decided": len(g), "pct_approved": float((~den).mean()), "pct_denied": float(den.mean()),
                     "pct_denials_appealed": float(app[den].mean()) if den.any() else np.nan,
                     "pct_overturned_of_appealed": float(g.loc[app, "overturn_flag"].mean()) if app.any() else np.nan,
                     "median_hours": float(g["hours"].median()), "mean_hours": float(g["hours"].mean()),
                     "pct_within_target": float((g["hours"] <= g["target"]).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. ALTERNATIVES: cohort tables and denial Pareto
# ---------------------------------------------------------------------------

def cohort_conversion(df: pd.DataFrame, as_of, window_days: int = 30) -> pd.DataFrame:
    """ALTERNATIVE: by referral month, share that started care within ``window_days``.

    Trade-off
    ---------
    + Simple, familiar, and exact for MATURE cohorts (every referral had the full window).
    - Recent cohorts are immature: shown as NaN here rather than as a falsely low rate. When most
      cohorts are immature (fast-changing operations), use the KM curve instead.
    """
    cut = pd.Timestamp(as_of)
    d = df.assign(cohort=df["referral_dt"].dt.to_period("M"))
    days = (d["soc_dt"] - d["referral_dt"]).dt.days
    d["started_in_window"] = (days <= window_days).astype(float)
    g = d.groupby("cohort")
    out = pd.DataFrame({"n_referrals": g.size(), "last_referral_dt": g["referral_dt"].max(),
                        "started_in_window": g["started_in_window"].mean()}).reset_index()
    out["mature_flag"] = (out["last_referral_dt"] + pd.Timedelta(days=window_days) <= cut).astype(int)
    out["conversion_rate"] = out["started_in_window"].where(out["mature_flag"] == 1)
    return out[["cohort", "n_referrals", "mature_flag", "conversion_rate"]]


def denial_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """ALTERNATIVE: denial reasons ranked with cumulative share (fix the top 2-3 first)."""
    r = df["denial_reason"].dropna().value_counts()
    return pd.DataFrame({"denial_reason": r.index, "n": r.to_numpy(), "share": (r / r.sum()).to_numpy(),
                         "cum_share": (r.cumsum() / r.sum()).to_numpy()})
