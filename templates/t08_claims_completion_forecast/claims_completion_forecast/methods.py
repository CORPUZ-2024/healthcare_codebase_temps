"""
Claims completion (IBNR) and completed-PMPM forecasting.

    chain_ladder                 (STANDARD)  vs  bornhuetter_ferguson (immature months)  (ALTERNATIVE)
    forecast_sarimax             (STANDARD)  vs  forecast_ets (Holt-Winters / ETS)       (ALTERNATIVE)
    + build_triangle, age_to_age_factors, blend_cl_bf, backtest_ibnr, backtest_forecasts

Vocabulary
----------
incurred month  month of service           lag / age   paid month - incurred month (0 = same month)
cumulative paid paid to date at each lag   completion  paid to date / ultimate (= 1 / CDF)
ultimate        what the month will cost when every claim is paid     IBNR = ultimate - paid to date
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Triangle
# ---------------------------------------------------------------------------

def build_triangle(claims: pd.DataFrame, as_of=None, max_lag: int | None = None) -> pd.DataFrame:
    """Incremental paid triangle: rows = incurred month, columns = lag 0..max_lag.

    Healthcare context
    ------------------
    Every PMPM report on recent months is incomplete: claims for March keep arriving through
    the summer. The triangle lays out how much of each incurred month had been paid at each lag.
    Cells that could have been observed but had no payment are 0; cells in the future (below the
    diagonal on the right) are NaN - never fill them with 0.

    Parameters
    ----------
    claims : incurred_month_start, paid_month_start, paid_amt (claim lines or pre-aggregated cells)
    as_of : data cut; payments after it are dropped (lets you rebuild the triangle "as it was")

    Returns
    -------
    DataFrame indexed by incurred_month_start, integer lag columns, float cells.

    Example
    -------
    >>> c = pd.DataFrame({"incurred_month_start": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01"]),
    ...                   "paid_month_start": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-02-01"]),
    ...                   "paid_amt": [60.0, 40.0, 70.0]})
    >>> build_triangle(c).to_dict("index")[pd.Timestamp("2025-02-01")]
    {0: 70.0, 1: nan}

    Common mistakes
    ---------------
    - Lag in days / 30 instead of calendar months (a Jan 31 service paid Feb 1 is lag 1).
    - Filling future cells with 0 -> completion factors of 1.0 -> IBNR of zero.
    - Grouping by PAID month and calling it PMPM.
    """
    c = claims.copy()
    last = pd.Timestamp(as_of).to_period("M") if as_of is not None else c["paid_month_start"].max().to_period("M")
    ip, pp = c["incurred_month_start"].dt.to_period("M"), c["paid_month_start"].dt.to_period("M")
    c = c[(pp <= last) & (ip <= last)]
    ip, pp = ip[c.index], pp[c.index]
    c["lag"] = (pp.dt.year - ip.dt.year) * 12 + (pp.dt.month - ip.dt.month)
    max_lag = int(c["lag"].max()) if max_lag is None else max_lag
    c = c[c["lag"] <= max_lag]
    tri = c.pivot_table(index="incurred_month_start", columns="lag", values="paid_amt", aggfunc="sum")
    tri = tri.reindex(columns=range(max_lag + 1))
    months = pd.period_range(tri.index.min().to_period("M"), last, freq="M")
    tri = tri.reindex(months.start_time)
    tri.index.name = "incurred_month_start"
    for inc in tri.index:                                      # observable cells -> 0, future cells -> NaN
        age = (last - inc.to_period("M")).n
        tri.loc[inc, [k for k in tri.columns if k <= age]] = tri.loc[inc, [k for k in tri.columns if k <= age]].fillna(0.0)
        tri.loc[inc, [k for k in tri.columns if k > age]] = np.nan
    return tri.astype(float)


def cumulative(tri: pd.DataFrame) -> pd.DataFrame:
    """Cumulative paid by lag (NaN stays NaN)."""
    return tri.cumsum(axis=1, skipna=False)


# ---------------------------------------------------------------------------
# 2. Chain ladder (STANDARD)
# ---------------------------------------------------------------------------

def age_to_age_factors(cum: pd.DataFrame, n_avg: int | None = 12) -> pd.DataFrame:
    """Volume-weighted development factors f_k = sum C[k+1] / sum C[k] over the most recent
    ``n_avg`` incurred months that have both lags observed.

    Returns lag, factor, n_rows, cv_individual (spread of the single-row factors - how much to trust f_k).
    """
    rows = []
    for k in cum.columns[:-1]:
        both = cum[[k, k + 1]].dropna()
        both = both[both[k] > 0]
        if n_avg:
            both = both.tail(n_avg)
        f = both[k + 1].sum() / both[k].sum() if len(both) else 1.0
        ind = both[k + 1] / both[k]
        rows.append({"lag": int(k), "factor": float(f), "n_rows": int(len(both)),
                     "cv_individual": float(ind.std() / ind.mean()) if len(ind) > 1 else float("nan")})
    return pd.DataFrame(rows)


def chain_ladder(cum: pd.DataFrame, n_avg: int | None = 12, tail_factor: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STANDARD: chain-ladder ultimates and IBNR per incurred month.

    Healthcare context
    ------------------
    The actuarial default for health claims reserves and for "completing" recent PMPM. The
    development pattern observed on older months is applied to each month's paid-to-date.

    Steps
    -----
    1. Age-to-age factors per lag (volume-weighted, recent ``n_avg`` months).
    2. Cumulative development factor to ultimate CDF_k = product of f_j for j >= k, times the tail.
    3. For each incurred month at age a: ultimate = paid-to-date x CDF_a; completion = 1 / CDF_a;
       IBNR = ultimate - paid-to-date.

    Returns (per-month table: paid_to_date, age, cdf, completion_pct, ultimate_amt, ibnr_amt; factor table).

    Common mistakes
    ---------------
    - Trusting the chain ladder for the latest 1-2 months: ultimate = small paid x big factor, so one
      slow processing week swings the estimate. Use Bornhuetter-Ferguson there.
    - Averaging factors across a claims-system change (processing speed changed -> old factors wrong).
    """
    f = age_to_age_factors(cum, n_avg)                                                   # step 1
    cdf = np.r_[np.cumprod(f["factor"].to_numpy()[::-1])[::-1], 1.0] * tail_factor       # step 2
    ages = cum.notna().sum(axis=1) - 1
    ptd = cum.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else 0.0, axis=1)
    out = pd.DataFrame({"paid_to_date": ptd, "age": ages.astype(int)})                   # step 3
    out["cdf"] = cdf[out["age"].clip(upper=len(cdf) - 1)]
    out["completion_pct"] = 1 / out["cdf"]
    out["ultimate_amt"] = out["paid_to_date"] * out["cdf"]
    out["ibnr_amt"] = out["ultimate_amt"] - out["paid_to_date"]
    f["cdf_to_ultimate"] = cdf[:-1]
    return out, f


# ---------------------------------------------------------------------------
# 3. Bornhuetter-Ferguson (ALTERNATIVE) and the blend
# ---------------------------------------------------------------------------

def expected_pmpm_seasonal(completed: pd.Series, mature: pd.Series) -> pd.Series:
    """A-priori PMPM for BF: same month last year x year-over-year trend (median over mature pairs).

    ``completed`` = completed PMPM by incurred month; ``mature`` = boolean, months whose estimate is
    reliable (e.g. completion >= 95%). Months without a mature month 12 earlier get the trended mean.
    """
    yoy = (completed / completed.shift(12))[mature & mature.shift(12, fill_value=False)]
    trend = float(yoy.median()) if len(yoy) else 1.0
    prior = completed.where(mature).shift(12) * trend
    fallback = completed[mature].tail(12).mean() * trend
    return prior.fillna(fallback)


def bornhuetter_ferguson(cl: pd.DataFrame, exposure: pd.Series, expected_pmpm: pd.Series) -> pd.Series:
    """ALTERNATIVE: BF ultimate = paid-to-date + expected ultimate x (1 - completion).

    Trade-off
    ---------
    + For immature months (age 0-2) the chain ladder multiplies a small, noisy paid amount by a
      large factor. BF uses the development pattern only for the UNPAID share and takes the level
      from an a-priori PMPM (last year trended), so a slow week in claims processing barely moves it.
    - Anchored to the a-priori: if cost truly jumped this month, BF is slow to show it. Use the
      chain ladder once a month is mostly paid (the blend does this).

    Parameters
    ----------
    cl : output of chain_ladder (paid_to_date, completion_pct)
    exposure : member-months by incurred month (same index)
    expected_pmpm : a-priori PMPM by incurred month (e.g. expected_pmpm_seasonal)
    """
    expected_ult = expected_pmpm.reindex(cl.index) * exposure.reindex(cl.index)
    return cl["paid_to_date"] + expected_ult * (1 - cl["completion_pct"])


def blend_cl_bf(cl: pd.DataFrame, bf_ultimate: pd.Series, bf_max_age: int = 2) -> pd.DataFrame:
    """Chain ladder for mature months, BF for months with age <= bf_max_age (common reserving practice)."""
    out = cl.copy()
    use_bf = out["age"] <= bf_max_age
    out["method"] = np.where(use_bf, "BF", "CL")
    out["ultimate_amt"] = np.where(use_bf, bf_ultimate, out["ultimate_amt"])
    out["ibnr_amt"] = out["ultimate_amt"] - out["paid_to_date"]
    return out


def complete(claims: pd.DataFrame, exposure: pd.DataFrame, as_of, n_avg: int = 12, tail_factor: float = 1.0,
             bf_max_age: int = 2, max_lag: int | None = None) -> pd.DataFrame:
    """One call: triangle -> chain ladder -> BF -> blend -> completed PMPM per incurred month.

    Returns incurred_month_start index with paid_to_date, age, completion_pct, ultimate_cl, ultimate_bf,
    ultimate_amt (blend), ibnr_amt, method, member_months, paid_pmpm, completed_pmpm.
    """
    cum = cumulative(build_triangle(claims, as_of, max_lag))
    cl, _ = chain_ladder(cum, n_avg, tail_factor)
    mm = exposure.set_index("incurred_month_start")["member_months"].reindex(cl.index)
    cl_pmpm = cl["ultimate_amt"] / mm
    prior = expected_pmpm_seasonal(cl_pmpm, cl["completion_pct"] >= 0.95)
    bf = bornhuetter_ferguson(cl, mm, prior)
    out = blend_cl_bf(cl, bf, bf_max_age)
    out["ultimate_cl"], out["ultimate_bf"] = cl["ultimate_amt"], bf
    out["member_months"] = mm
    out["paid_pmpm"] = out["paid_to_date"] / mm
    out["completed_pmpm"] = out["ultimate_amt"] / mm
    return out


# ---------------------------------------------------------------------------
# 4. Forecast completed PMPM
# ---------------------------------------------------------------------------

def forecast_sarimax(pmpm: pd.Series, horizon: int = 12, alpha: float = 0.05,
                     order=(1, 0, 0), seasonal_order=(0, 1, 0, 12)) -> pd.DataFrame:
    """STANDARD: seasonal ARIMA on log completed PMPM (statsmodels SARIMAX) with drift.

    Healthcare context
    ------------------
    Budgets, rate filings and VBC benchmarks need next year's PMPM with a range. Log scale makes
    trend multiplicative (x% per year); seasonal differencing handles the winter peak; the constant
    after differencing is the annual trend.

    Returns forecast_month_start index with mean, lo, hi (back-transformed; mean uses the lognormal
    correction exp(mu + var/2), the interval is exp of the log-scale interval).

    Common mistakes
    ---------------
    - Forecasting on PAID (incomplete) PMPM: the last months look like a drop and the model extends it.
    - Reporting exp(mean log) as the mean (that is the median; slightly low).
    - Fitting a 12-month seasonal model to < 24 months of history.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = np.log(pmpm.astype(float))
    y.index = pd.DatetimeIndex(y.index, freq="MS")
    res = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend="c").fit(disp=False)
    fc = res.get_forecast(horizon)
    mu, var = fc.predicted_mean, fc.var_pred_mean
    ci = fc.conf_int(alpha=alpha)
    return pd.DataFrame({"mean": np.exp(mu + var / 2), "lo": np.exp(ci.iloc[:, 0]), "hi": np.exp(ci.iloc[:, 1])})


def forecast_ets(pmpm: pd.Series, horizon: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    """ALTERNATIVE: Holt-Winters exponential smoothing (statsmodels ETSModel) on log PMPM,
    additive damped trend + additive seasonality.

    Trade-off
    ---------
    + Adapts faster to level shifts (a new benefit, a membership mix change) because recent months
      get more weight; fewer modeling choices than ARIMA orders; damped trend avoids runaway
      extrapolation over long horizons.
    - Less principled intervals for short series; no easy way to add covariates (use SARIMAX exog).
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = np.log(pmpm.astype(float))
    y.index = pd.DatetimeIndex(y.index, freq="MS")
    res = ETSModel(y, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit(disp=False)
    pred = res.get_prediction(start=len(y), end=len(y) + horizon - 1).summary_frame(alpha=alpha)
    sd = (pred["pi_upper"] - pred["mean"]) / 1.959963984540054
    return pd.DataFrame({"mean": np.exp(pred["mean"] + sd ** 2 / 2), "lo": np.exp(pred["pi_lower"]),
                         "hi": np.exp(pred["pi_upper"])}, index=pred.index)


# ---------------------------------------------------------------------------
# 5. Backtests (score the methods against what actually happened)
# ---------------------------------------------------------------------------

def backtest_ibnr(claims: pd.DataFrame, exposure: pd.DataFrame, truth: pd.DataFrame, cutoffs, n_avg: int = 12,
                  bf_max_age: int = 2, max_lag: int | None = None) -> pd.DataFrame:
    """Rebuild the triangle as of each earlier cutoff, estimate ultimates, compare with the truth.

    Returns cutoff, incurred_month_start, age, pct_error_cl, pct_error_bf (estimate / truth - 1).
    """
    tr = truth.set_index("incurred_month_start")["ultimate_paid_amt"]
    rows = []
    for cut in cutoffs:
        r = complete(claims, exposure, cut, n_avg, bf_max_age=bf_max_age, max_lag=max_lag)
        r = r[r["age"] <= 6]
        rows.append(pd.DataFrame({"cutoff": pd.Timestamp(cut), "incurred_month_start": r.index, "age": r["age"].to_numpy(),
                                  "pct_error_cl": (r["ultimate_cl"] / tr.reindex(r.index) - 1).to_numpy(),
                                  "pct_error_bf": (r["ultimate_bf"] / tr.reindex(r.index) - 1).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def backtest_forecasts(pmpm: pd.Series, holdout: int = 6) -> dict:
    """Fit both forecasters on all but the last ``holdout`` months; MAPE on the held-out months."""
    train, test = pmpm.iloc[:-holdout], pmpm.iloc[-holdout:]
    out = {}
    for name, f in (("sarimax", forecast_sarimax), ("ets", forecast_ets)):
        fc = f(train, holdout)
        out[f"mape_{name}"] = float((fc["mean"].to_numpy() / test.to_numpy() - 1).__abs__().mean())
        out[f"coverage_{name}"] = float(((test.to_numpy() >= fc["lo"].to_numpy()) & (test.to_numpy() <= fc["hi"].to_numpy())).mean())
    return out
