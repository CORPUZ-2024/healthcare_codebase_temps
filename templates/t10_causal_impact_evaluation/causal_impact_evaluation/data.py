"""
Synthetic data with KNOWN causal effects for the three evaluation designs.

  cross_section()  members referred to a program by risk -> confounded; heterogeneous effect, so the
                   ATT (effect on those treated) differs from the ATE (effect if everyone were treated)
  practice_panel() practices x months; half adopt a workflow at one date; optional pre-trend violation
  its_series()     a monthly population rate with a level and slope change at a policy date, seasonality
                   and AR(1) errors (so naive OLS standard errors are too small)

Real-data analogues: your own claims (t00/t05) for the first two; for ITS any monthly public series
around a policy date (e.g. state Medicaid policy changes + CMS monthly enrollment / utilization files).
Every generator returns its truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section(n: int = 6_000, seed: int = 10) -> dict:
    """Members with covariates, referral (treatment) and a continuous outcome.

    Outcome: 12-month acute-care utilization index (higher = worse). Treatment effect
    tau_i = -3 - 1.5 x risk_z (sicker members benefit more). Referral depends on risk, chronic
    conditions and prior use (confounding); the outcome depends on the same variables, with a
    non-linear prior-use term that a linear outcome model misses.

    Returns {'df': member_id, age, chronic_cnt, risk_z, prior_util, rural_flag, ps_true, treated_flag,
    outcome, tau; 'ate': float; 'att': float}.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 90, n)
    chronic = rng.poisson(0.6 + 0.03 * (age - 20))
    risk_z = rng.normal(0, 1, n) + 0.25 * (chronic - chronic.mean())
    prior = rng.gamma(2.0, 1.0 + 0.4 * np.clip(risk_z, -1, None))
    rural = (rng.random(n) < 0.25).astype(int)
    lin = -1.2 + 0.6 * risk_z + 0.25 * (chronic - 2) + 0.25 * (prior - 2) - 0.5 * rural
    ps = 1 / (1 + np.exp(-lin))
    t = (rng.random(n) < ps).astype(int)
    y0 = 20 + 4 * risk_z + 2 * chronic + 1.5 * prior + 0.4 * (prior - 2) ** 2 - 1.0 * rural + rng.normal(0, 5, n)
    tau = -3 - 1.5 * risk_z
    df = pd.DataFrame({"member_id": [f"K{i:05d}" for i in range(1, n + 1)], "age": age, "chronic_cnt": chronic,
                       "risk_z": risk_z.round(4), "prior_util": prior.round(4), "rural_flag": rural,
                       "ps_true": ps, "treated_flag": t, "outcome": y0 + t * tau, "tau": tau})
    return {"df": df, "ate": float(tau.mean()), "att": float(tau[t == 1].mean())}


def practice_panel(n_practices: int = 40, n_months: int = 36, adopt_month: int = 24, effect: float = -2.0,
                   pre_trend: float = 0.0, seed: int = 10) -> dict:
    """Practice x month ED visits per 1,000 member-months.

    y = practice level + common seasonal/trend month effect + effect x (treated & month >= adopt)
        + pre_trend x month (treated only; 0 = parallel trends hold) + noise.
    Returns {'df': practice_id, month_idx, month_start, treated_flag, post_flag, rel_month, y; 'effect': float}.
    """
    rng = np.random.default_rng(seed + 1)
    p = np.arange(n_practices)
    treated = (p < n_practices // 2).astype(int)
    level = rng.normal(55, 6, n_practices) + 3 * treated                 # treated practices start higher (fine for DiD)
    m = np.arange(n_months)
    month_fx = 0.08 * m + 2.5 * np.cos(2 * np.pi * m / 12)
    P, M = np.meshgrid(p, m, indexing="ij")
    post = (M >= adopt_month).astype(int)
    y = (level[P] + month_fx[M] + effect * treated[P] * post + pre_trend * treated[P] * M
         + rng.normal(0, 1.5, P.shape))
    df = pd.DataFrame({"practice_id": [f"PR{i:02d}" for i in P.ravel()], "month_idx": M.ravel(),
                       "month_start": pd.date_range("2023-01-01", periods=n_months, freq="MS")[M.ravel()],
                       "treated_flag": treated[P].ravel(), "post_flag": post.ravel(),
                       "rel_month": (M - adopt_month).ravel(), "y": y.ravel()})
    return {"df": df, "effect": float(effect)}


def its_series(n_months: int = 60, break_month: int = 36, level_change: float = -4.0, slope_change: float = -0.15,
               ar: float = 0.5, seed: int = 10) -> dict:
    """Monthly population rate (e.g. ED visits per 1,000) with a policy change at ``break_month``.

    Returns {'df': month_idx, month_start, y, post_flag, time_since; 'level_change', 'slope_change'}.
    """
    rng = np.random.default_rng(seed + 2)
    t = np.arange(n_months)
    e = np.zeros(n_months)
    shocks = rng.normal(0, 1.2, n_months)
    for i in range(n_months):
        e[i] = (ar * e[i - 1] if i else 0.0) + shocks[i]
    post = (t >= break_month).astype(int)
    since = np.where(post == 1, t - break_month, 0)
    y = 50 + 0.10 * t + 2.0 * np.cos(2 * np.pi * t / 12) + level_change * post + slope_change * since + e
    df = pd.DataFrame({"month_idx": t, "month_start": pd.date_range("2021-01-01", periods=n_months, freq="MS"),
                       "y": y, "post_flag": post, "time_since": since})
    return {"df": df, "level_change": level_change, "slope_change": slope_change}
