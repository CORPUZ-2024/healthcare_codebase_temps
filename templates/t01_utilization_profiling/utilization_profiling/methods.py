"""
Utilization profiling: how much care a population uses, of what kind, and how it is changing.

    rate_ci_poisson_exact       (STANDARD)  vs  rate_ci_cluster_bootstrap  (ALTERNATIVE)
    readmissions_per_index      (STANDARD)  vs  readmissions_per_1000      (ALTERNATIVE)
    trend_rolling12             (STANDARD)  vs  trend_yoy_same_month       (ALTERNATIVE)
    + describe_distribution (percentiles, CV), build_stays (transfer-aware), frequent_ed_users
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Events: stays, ED visits, visits
# ---------------------------------------------------------------------------

def build_stays(claims: pd.DataFrame, transfer_gap_days: int = 1) -> pd.DataFrame:
    """Collapse inpatient claims into STAYS (acute-to-acute transfers become one stay).

    Healthcare context
    ------------------
    One hospitalization can generate several claims (interim bills, a transfer to another
    hospital). Counting claims overstates admissions and fakes "readmissions" on the day of
    transfer. Rule used here: an admit within ``transfer_gap_days`` of the previous discharge
    continues the same stay.

    Returns member_id, stay_id, admit_dt, discharge_dt, los_days, paid_amt, n_claims
    """
    ip = (claims[claims["service_category"] == "IP"]
          .groupby("claim_id", as_index=False)
          .agg(member_id=("member_id", "first"), admit_dt=("admit_dt", "first"),
               discharge_dt=("discharge_dt", "first"), paid_amt=("paid_amt", "sum"))
          .sort_values(["member_id", "admit_dt"]))
    prev_dis = ip.groupby("member_id")["discharge_dt"].shift()
    new_stay = prev_dis.isna() | (ip["admit_dt"] > prev_dis + pd.Timedelta(days=transfer_gap_days))
    ip["stay_seq"] = new_stay.groupby(ip["member_id"]).cumsum()
    stays = ip.groupby(["member_id", "stay_seq"], as_index=False).agg(
        admit_dt=("admit_dt", "min"), discharge_dt=("discharge_dt", "max"),
        paid_amt=("paid_amt", "sum"), n_claims=("claim_id", "nunique"))
    stays["los_days"] = (stays["discharge_dt"] - stays["admit_dt"]).dt.days.clip(lower=1)
    stays["stay_id"] = stays["member_id"] + "-" + stays["stay_seq"].astype(str)
    return stays.drop(columns="stay_seq")


def count_events(claims: pd.DataFrame) -> pd.DataFrame:
    """One row per event with member_id, event_type, event_dt.

    event_type: IP_ADMIT (from stays), ED_VISIT (distinct member+date of ED claims),
    OFFICE_VISIT (distinct member+date PROF/TELEHEALTH/PROF_HOME), HCBS_DAY (distinct
    member+date with HCBS services).
    """
    stays = build_stays(claims)
    parts = [stays.assign(event_type="IP_ADMIT", event_dt=stays["admit_dt"])[["member_id", "event_type", "event_dt"]]]
    def daily(cats, name):
        c = claims[claims["service_category"].isin(cats)][["member_id", "svc_from_dt"]].drop_duplicates()
        return c.rename(columns={"svc_from_dt": "event_dt"}).assign(event_type=name)
    parts += [daily(["ED"], "ED_VISIT"), daily(["PROF", "TELEHEALTH", "PROF_HOME"], "OFFICE_VISIT"),
              daily(["HCBS"], "HCBS_DAY")]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Rates per 1,000 with confidence intervals
# ---------------------------------------------------------------------------

def rate_ci_poisson_exact(events: int, member_months: float, per: float = 12_000, alpha: float = 0.05) -> dict:
    """STANDARD: rate per 1,000 member-years with an exact (Garwood) Poisson CI.

    Healthcare context
    ------------------
    "ED visits per 1,000" = events / member-months x 12,000. The exact Poisson interval is the
    textbook CI for an event rate and behaves well for small counts.

    Caveat: it assumes events are independent. They aren't — a few members generate many ED
    visits (overdispersion) — so this CI is too narrow for utilization. See the alternative.

    >>> r = rate_ci_poisson_exact(10, 1_200)
    >>> round(r["rate"], 1), round(r["lo"], 1), round(r["hi"], 1)
    (100.0, 48.0, 183.9)
    """
    lo = stats.chi2.ppf(alpha / 2, 2 * events) / 2 if events > 0 else 0.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / 2
    k = per / member_months
    return {"events": int(events), "member_months": float(member_months), "rate": float(events * k),
            "lo": float(lo * k), "hi": float(hi * k)}


def rate_ci_cluster_bootstrap(member_events: pd.Series, member_mm: pd.Series, per: float = 12_000,
                              n_boot: int = 2_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """ALTERNATIVE: resample MEMBERS (not events) to get a CI that respects overdispersion.

    Trade-off
    ---------
    + Honest width when a few members drive many events (typical for ED, IP, HCBS days).
    + No distributional assumption.
    - Slower; needs member-level data; unstable with very few members (< ~30).

    Parameters
    ----------
    member_events : events per member (index member_id; include zeros!)
    member_mm : member-months per member (same index)
    """
    ev = member_events.reindex(member_mm.index, fill_value=0).to_numpy(float)
    mm = member_mm.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ev), size=(n_boot, len(ev)))
    boots = ev[idx].sum(axis=1) / mm[idx].sum(axis=1) * per
    rate = ev.sum() / mm.sum() * per
    return {"events": int(ev.sum()), "member_months": float(mm.sum()), "rate": float(rate),
            "lo": float(np.quantile(boots, alpha / 2)), "hi": float(np.quantile(boots, 1 - alpha / 2))}


def utilization_table(events: pd.DataFrame, member_months: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Per-1,000 rates for every event type, exact Poisson CI, plus ALOS and days per 1,000 for IP."""
    mm_total = member_months["member_months"].sum()
    rows = []
    for et, g in events.groupby("event_type"):
        rows.append({"event_type": et, **rate_ci_poisson_exact(len(g), mm_total)})
    t = pd.DataFrame(rows)
    alos = stays["los_days"].mean() if len(stays) else np.nan
    t.loc[t["event_type"] == "IP_ADMIT", "alos_days"] = alos
    t.loc[t["event_type"] == "IP_ADMIT", "days_per_1000"] = stays["los_days"].sum() / mm_total * 12_000
    return t


# ---------------------------------------------------------------------------
# 3. Readmissions
# ---------------------------------------------------------------------------

def readmissions_per_index(stays: pd.DataFrame, enrollment_end: pd.Series, window_days: int = 30,
                           data_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """STANDARD: all-cause 30-day readmission rate per index discharge.

    Healthcare context
    ------------------
    The hospital-quality convention (CMS HRRP / HEDIS PCR shape): every eligible discharge is
    an index; it "has a readmission" if the member is admitted again 1-``window_days`` days
    after discharge. A readmission can itself be the next index.

    Eligibility used here (simplified from the official specs):
      * the member must be enrolled through discharge + window (else we can't observe it)
      * the discharge must be at least ``window_days`` before the end of the data
    Not modelled: planned-readmission exclusions, deaths, risk adjustment (see t13 for O/E).

    Returns one row per index stay with readmit_flag and days_to_readmit.
    """
    s = stays.sort_values(["member_id", "admit_dt"]).copy()
    s["next_admit_dt"] = s.groupby("member_id")["admit_dt"].shift(-1)
    s["days_to_readmit"] = (s["next_admit_dt"] - s["discharge_dt"]).dt.days
    s["readmit_flag"] = s["days_to_readmit"].between(1, window_days).astype(int)
    data_end = data_end or s["discharge_dt"].max()
    win_end = s["discharge_dt"] + pd.Timedelta(days=window_days)
    s["eligible_flag"] = ((win_end <= data_end) & (s["member_id"].map(enrollment_end) >= win_end)).astype(int)
    return s


def readmission_rate(index_stays: pd.DataFrame) -> dict:
    e = index_stays[index_stays["eligible_flag"] == 1]
    n, k = len(e), int(e["readmit_flag"].sum())
    lo, hi = _wilson(k, n)
    return {"index_stays": n, "readmissions": k, "rate": k / n if n else np.nan, "lo": float(lo), "hi": float(hi)}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def readmissions_per_1000(index_stays: pd.DataFrame, member_months: float) -> dict:
    """ALTERNATIVE: readmissions per 1,000 member-years (population-based).

    Trade-off
    ---------
    + Credits programs that prevent the FIRST admission too. A per-index rate can rise when a
      program keeps healthier people out of the hospital (the remaining admissions are sicker).
    + Natural for population-health and PMPM conversations.
    - Mixes admission frequency and readmission risk; not comparable to hospital quality
      measures.
    """
    k = int(index_stays["readmit_flag"].sum())
    return rate_ci_poisson_exact(k, member_months)


# ---------------------------------------------------------------------------
# 4. Trend
# ---------------------------------------------------------------------------

def monthly_rates(events: pd.DataFrame, member_months: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Monthly rate per 1,000 for one event type. Months with exposure but zero events are
    kept as 0; months without exposure are reindexed to NaN (never silently dropped)."""
    ev = events[events["event_type"] == event_type]
    num = ev.groupby(ev["event_dt"].dt.to_period("M")).size()
    den = member_months.groupby("month")["member_months"].sum()
    full = pd.period_range(den.index.min(), den.index.max(), freq="M")
    out = pd.DataFrame({"events": num.reindex(full, fill_value=0), "member_months": den.reindex(full)})
    out["rate_per_1000"] = out["events"] / out["member_months"] * 12_000
    out.index.name = "month"
    return out


def trend_rolling12(monthly: pd.DataFrame, incomplete_after: pd.Period | None = None) -> pd.DataFrame:
    """STANDARD: rolling-12-month rate (sum of events / sum of exposure over 12 months).

    + Removes seasonality and smooths small-number noise; the usual executive trend line.
    Months after ``incomplete_after`` (claims runout) are set to NaN, not plotted as a dip.
    """
    m = monthly.copy()
    if incomplete_after is not None:
        m.loc[m.index > incomplete_after, ["events", "member_months", "rate_per_1000"]] = np.nan
    r = m[["events", "member_months"]].rolling(12, min_periods=12).sum()
    m["r12_rate_per_1000"] = r["events"] / r["member_months"] * 12_000
    return m


def trend_yoy_same_month(monthly: pd.DataFrame, incomplete_after: pd.Period | None = None) -> pd.DataFrame:
    """ALTERNATIVE: compare each month with the same month last year (% change).

    Trade-off
    ---------
    + Reacts within a month (R12 lags by ~6 months), still controls for seasonality.
    - Noisy for small populations; a single unusual month last year distorts this year.
    """
    m = monthly.copy()
    if incomplete_after is not None:
        m.loc[m.index > incomplete_after, "rate_per_1000"] = np.nan
    m["yoy_pct"] = m["rate_per_1000"] / m["rate_per_1000"].shift(12) - 1
    return m


# ---------------------------------------------------------------------------
# 5. Descriptive distribution + high utilizers
# ---------------------------------------------------------------------------

def describe_distribution(values: pd.Series) -> dict:
    """Summary statistics used before any modelling: count, mean, median, percentiles, SD, CV.

    Cost and utilization are right-skewed: expect mean >> median and CV > 1.
    """
    v = values.dropna()
    q = v.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(q[0.5]), "sd": float(v.std(ddof=1)),
            "cv": float(v.std(ddof=1) / v.mean()) if v.mean() else np.nan,
            "p10": float(q[0.1]), "p25": float(q[0.25]), "p75": float(q[0.75]), "p90": float(q[0.9]),
            "p99": float(q[0.99]), "share_zero": float((v == 0).mean())}


def frequent_ed_users(events: pd.DataFrame, threshold: int = 4, window: str = "365D") -> pd.DataFrame:
    """Members with >= ``threshold`` ED visits in any rolling ``window`` — a common care-management
    referral rule. Returns member_id, max_visits_in_window."""
    ed = events[events["event_type"] == "ED_VISIT"].sort_values("event_dt")
    if ed.empty:
        return pd.DataFrame(columns=["member_id", "max_visits_in_window"])
    roll = (ed.set_index("event_dt").groupby("member_id")["event_type"]
            .rolling(window).count().rename("n").reset_index())
    top = roll.groupby("member_id")["n"].max().astype(int)
    return top[top >= threshold].rename("max_visits_in_window").reset_index()
