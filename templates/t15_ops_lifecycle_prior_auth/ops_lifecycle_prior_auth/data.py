"""
Synthetic referral-to-care lifecycle with prior authorization (home-care agency / payer operations).

One row per referral: referral -> intake assessment -> PA request -> PA decision (approve / deny;
denials may be appealed and overturned) -> start of care (SOC) -> still in service at 90 days.

What makes it realistic:
  * stage drop-off (never reached, declined, no staff) and skewed stage durations
  * PA priority (expedited ~20%) with decision times around the CMS-0057-F targets; a share miss them
  * denial reasons with a long tail (Pareto); ~40% of denials appealed, ~half of those overturned
  * a data cut (``as_of``): recent referrals haven't reached later stages YET (right-censoring) - the
    reason naive "median days to start" and fresh cohort conversion rates are biased
  * truth: every timestamp is generated first and then hidden if after the cut, so tests can compare
    estimates with what eventually happens

Public counterpart: payer prior-authorization metrics that CMS-0057-F requires impacted payers to post
publicly (first reports due March 31, 2026, for 2025 data) - see data/README.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DENIAL_REASONS = {"missing clinical documentation": 0.38, "not medically necessary": 0.22, "service not covered": 0.12,
                  "out-of-network provider": 0.10, "duplicate request": 0.07, "member not eligible": 0.06,
                  "incorrect codes": 0.05}


def generate(n: int = 5_000, start: str = "2025-01-01", months: int = 12, as_of: str = "2025-12-31",
             seed: int = 15) -> dict:
    """Returns {'observed': referrals as seen at the data cut, 'truth': the same with every future timestamp}.

    Columns: referral_id, referral_dt, assessment_dt, pa_request_ts, pa_priority_cd, pa_decision_ts,
    pa_decision_cd (approved/denied), denial_reason, appeal_flag, overturn_flag, soc_dt, retained_90d_flag
    (NaN when not yet knowable), payer_cd.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    horizon = (t0 + pd.DateOffset(months=months) - t0).days
    ref = t0 + pd.to_timedelta(rng.integers(0, horizon, n), unit="D")
    day = lambda x: pd.to_timedelta(np.round(x), unit="D")          # noqa: E731
    reached = rng.random(n) < 0.88
    assess = ref + day(rng.gamma(2.0, 1.5, n))
    pa_need = reached & (rng.random(n) < 0.95)
    pa_req = assess + pd.to_timedelta(rng.gamma(1.5, 16, n), unit="h")
    expedited = rng.random(n) < 0.2
    hours = np.where(expedited, rng.gamma(3.0, 16, n), rng.gamma(2.5, 45, n))      # some miss 72h / 168h
    pa_dec = pa_req + pd.to_timedelta(hours, unit="h")
    denied = rng.random(n) < 0.18
    reasons = rng.choice(list(DENIAL_REASONS), n, p=list(DENIAL_REASONS.values()))
    appeal = denied & (rng.random(n) < 0.4)
    overturn = appeal & (rng.random(n) < 0.5)
    approved_final = ~denied | overturn
    extra = np.where(overturn, rng.gamma(2.0, 7, n), 0.0)                               # appeal delay
    starts = reached & pa_need & approved_final & (rng.random(n) < 0.9)                 # staffing drop-off
    soc = pa_dec.normalize() + day(rng.gamma(2.2, 3.0, n) + extra)
    retained = rng.random(n) < 0.78

    truth = pd.DataFrame({
        "referral_id": [f"RF{i:05d}" for i in range(1, n + 1)], "referral_dt": ref.normalize(),
        "payer_cd": rng.choice(["MCD_A", "MCD_B", "MA_C"], n, p=[0.45, 0.35, 0.20]),
        "assessment_dt": pd.Series(assess.normalize()).where(reached),
        "pa_request_ts": pd.Series(pa_req).where(pa_need), "pa_priority_cd": np.where(expedited, "expedited", "standard"),
        "pa_decision_ts": pd.Series(pa_dec).where(pa_need),
        "pa_decision_cd": pd.Series(np.where(denied, "denied", "approved")).where(pa_need),
        "denial_reason": pd.Series(reasons).where(pa_need & denied),
        "appeal_flag": (pa_need & appeal).astype(int), "overturn_flag": (pa_need & overturn).astype(int),
        "soc_dt": pd.Series(soc).where(starts), "retained_90d_flag": pd.Series(retained.astype(float)).where(starts)})
    truth["pa_priority_cd"] = truth["pa_priority_cd"].where(pa_need)

    cut = pd.Timestamp(as_of)
    obs = truth.copy()
    for c in ("assessment_dt", "pa_request_ts", "pa_decision_ts", "soc_dt"):
        obs[c] = obs[c].where(obs[c] <= cut)
    obs.loc[obs["pa_decision_ts"].isna(), ["pa_decision_cd", "denial_reason"]] = np.nan
    obs.loc[obs["pa_decision_ts"].isna(), ["appeal_flag", "overturn_flag"]] = 0
    obs.loc[obs["pa_request_ts"].isna(), "pa_priority_cd"] = np.nan
    obs["retained_90d_flag"] = obs["retained_90d_flag"].where(obs["soc_dt"] + pd.Timedelta(days=90) <= cut)
    return {"observed": obs, "truth": truth, "as_of": cut}
