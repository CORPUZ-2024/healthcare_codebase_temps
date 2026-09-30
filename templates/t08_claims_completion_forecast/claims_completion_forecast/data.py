"""
Synthetic incurred-by-paid claims with a KNOWN ultimate cost per incurred month, and a loader
that turns any claim-line file (incurred date + paid date + amount) into this template's shape.

The generator bakes in what makes completion hard:
  * a lag pattern: ~30% paid in the service month, a tail out to 18 months
  * noise in the pattern month to month (Dirichlet), so factors from one row mislead
  * seasonality (winter high) and a 6% annual trend, membership growth
  * optional processing speed-up in the last months (``speedup``) - completion factors estimated on
    the old pattern then OVERSTATE IBNR (checks.check_lag_shift)

``truth`` holds the ultimate paid for every incurred month (what the triangle would show after
full runout), so IBNR methods can be scored.

Public test file
----------------
Any claims extract with a service (incurred) date and a paid / processed date. CMS DE-SynPUF
carrier and institutional files carry claim FROM/THRU dates; check whether your extract has a
processing date before using it for lags (data/README.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def lag_pattern(max_lag: int = 18, shape: float = 1.6, scale: float = 1.1) -> np.ndarray:
    """Share of an incurred month's ultimate paid in lag month 0..max_lag (discretized gamma)."""
    from scipy import stats
    edges = np.arange(max_lag + 2, dtype=float)
    p = np.diff(stats.gamma.cdf(edges, shape, scale=scale))
    return p / p.sum()


def generate(start: str = "2023-01-01", months: int = 36, members: int = 20_000, base_pmpm: float = 450.0,
             annual_trend: float = 0.06, max_lag: int = 18, speedup: float = 0.0, seed: int = 8) -> dict:
    """Paid-claims cells observed at the data cut, exposure, and the truth.

    Returns
    -------
    dict with
      claims : incurred_month_start, paid_month_start, paid_amt  (one row per incurred x paid month
               with payment observed by the data cut; aggregate it like claim lines)
      exposure : incurred_month_start, member_months
      truth : incurred_month_start, ultimate_paid_amt, member_months, true_pmpm
      as_of : last day of the last incurred month (the data cut)

    Steps
    -----
    1. Member-months grow 0.3%/month; true PMPM = base x trend x winter seasonality x noise.
    2. Ultimate per month = PMPM x member-months; split over lags by Dirichlet(400 x pattern).
       With ``speedup`` > 0 the last 6 incurred months pay faster (pattern shifted toward lag 0).
    3. Keep cells paid on or before the data cut.
    """
    rng = np.random.default_rng(seed)
    inc = pd.date_range(start, periods=months, freq="MS")
    t = np.arange(months)
    mm = members * 1.003 ** t                                                               # step 1
    season = 1 + 0.06 * np.cos(2 * np.pi * (inc.month.to_numpy() - 1) / 12)                # Jan peak
    pmpm = base_pmpm * (1 + annual_trend) ** (t / 12) * season * np.exp(rng.normal(0, 0.03, months))
    ultimate = pmpm * mm
    p = lag_pattern(max_lag)                                                                 # step 2
    fast = np.r_[p[0] + speedup * p[1], p[1] * (1 - speedup), p[2:]]
    rows = []
    for i in range(months):
        base = fast if (speedup and i >= months - 6) else p
        split = rng.dirichlet(400 * base) * ultimate[i]
        for lag in range(max_lag + 1):
            if i + lag < months:                                                             # step 3
                rows.append((inc[i], inc[i] + pd.DateOffset(months=lag), split[lag]))
    claims = pd.DataFrame(rows, columns=["incurred_month_start", "paid_month_start", "paid_amt"])
    claims["paid_amt"] = claims["paid_amt"].round(2)
    exposure = pd.DataFrame({"incurred_month_start": inc, "member_months": mm.round(1)})
    truth = exposure.assign(ultimate_paid_amt=ultimate, true_pmpm=pmpm)
    as_of = inc[-1] + pd.offsets.MonthEnd(0)
    return {"claims": claims, "exposure": exposure, "truth": truth, "as_of": as_of}


def load_claim_lines(path: str | Path, incurred_col: str, paid_col: str, amount_col: str,
                     date_format: str | None = None) -> pd.DataFrame:
    """Read a claim-line CSV and return incurred_month_start, paid_month_start, paid_amt.

    Works for any extract with a service date and a paid/processed date (e.g. t00's generator output:
    svc_from_dt / paid_dt / paid_amt). Rows with a paid date before the service date are dropped
    and counted - they are data errors, not negative lags.
    """
    df = pd.read_csv(path, usecols=[incurred_col, paid_col, amount_col], dtype=str)
    inc = pd.to_datetime(df[incurred_col], format=date_format, errors="coerce")
    paid = pd.to_datetime(df[paid_col], format=date_format, errors="coerce")
    out = pd.DataFrame({"incurred_month_start": inc.dt.to_period("M").dt.start_time,
                        "paid_month_start": paid.dt.to_period("M").dt.start_time,
                        "paid_amt": pd.to_numeric(df[amount_col], errors="coerce")})
    bad = out["paid_month_start"] < out["incurred_month_start"]
    if bad.any():
        print(f"load_claim_lines: dropped {int(bad.sum())} rows paid before service")
    return out[~bad & out.notna().all(axis=1)].reset_index(drop=True)
