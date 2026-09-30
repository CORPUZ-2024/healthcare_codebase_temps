"""
Preparation steps COPIED from t00_claims_foundation (templates never import each other).
If you fix a bug here, sweep the same function in t00 and the other templates that copied it
(t01, t05). See docs/CONVENTIONS.md section 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_versions_latest(claims: pd.DataFrame) -> pd.DataFrame:
    """STANDARD: keep the latest version of each claim line; drop voided claims.

    Healthcare context
    ------------------
    Payers re-send a claim when it is adjusted (a *replacement* feed). If you sum every
    version you double count. The final adjudicated state is the highest ``adj_seq``.
    A void (status 'V') means the whole claim was cancelled and contributes $0.

    Parameters
    ----------
    claims : DataFrame with claim_id, line_seq, adj_seq, claim_status_cd, paid_amt, ...

    Returns
    -------
    DataFrame at grain claim_id x line_seq (unique), voids removed.

    Steps
    -----
    1. Find each claim's latest adj_seq (claim-level, because a void is claim-level).
    2. Keep only rows at that latest adj_seq.
    3. Drop claims whose latest status is 'V'.

    Common mistakes
    ---------------
    - Taking max(adj_seq) per *line*: a void row may carry fewer lines than the original,
      leaving orphaned original lines in the data.
    - Using this on a DELTA feed (see collapse_versions_net) — it keeps only the last delta.
    """
    latest = claims.groupby("claim_id")["adj_seq"].transform("max")
    out = claims[claims["adj_seq"] == latest]
    out = out[out["claim_status_cd"] != "V"]
    return out.reset_index(drop=True)


def _merge_overlaps(enr: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Merge overlapping/adjacent spans within group_cols so no day is counted twice."""
    e = enr.sort_values(group_cols + ["enroll_start_dt"]).copy()
    prev_end = e.groupby(group_cols)["enroll_end_dt"].transform(lambda s: s.shift().cummax())
    new_block = (prev_end.isna()) | (e["enroll_start_dt"] > prev_end + pd.Timedelta(days=1))
    e["block"] = new_block.groupby([e[c] for c in group_cols]).cumsum()
    return (e.groupby(group_cols + ["block"], as_index=False)
             .agg(enroll_start_dt=("enroll_start_dt", "min"), enroll_end_dt=("enroll_end_dt", "max"))
             .drop(columns="block"))


def member_months_daily(enrollment: pd.DataFrame, period_start: str, period_end: str,
                        by: tuple[str, ...] = ("member_id",)) -> pd.DataFrame:
    """STANDARD: prorated member-months = days enrolled in month / days in month.

    Healthcare context
    ------------------
    PMPM and per-1,000 rates divide by member-months. A member enrolled 15 of 31 days
    contributes 15/31 = 0.484, not 1.0. Overlapping spans (retro plan changes) are merged
    first so a day is never counted twice.

    Parameters
    ----------
    enrollment : member_id, enroll_start_dt, enroll_end_dt (inclusive) [+ columns in ``by``]
    period_start, period_end : 'YYYY-MM-DD', inclusive
    by : grouping columns kept on the output (member_id must be included)

    Returns
    -------
    DataFrame: *by, month (Period[M]), days_enrolled, member_months

    Steps
    -----
    1. Merge overlapping spans per ``by`` group.
    2. Clip spans to the analysis period.
    3. Expand to one row per (span, month touched).
    4. days_enrolled = overlap of span and month; member_months = days / days_in_month.

    Example
    -------
    >>> e = pd.DataFrame({"member_id": ["A"], "enroll_start_dt": pd.to_datetime(["2025-01-17"]),
    ...                   "enroll_end_dt": pd.to_datetime(["2025-02-28"])})
    >>> member_months_daily(e, "2025-01-01", "2025-12-31")["member_months"].round(3).tolist()
    [0.484, 1.0]

    Common mistakes
    ---------------
    - Counting 1.0 for any month touched (inflates the denominator, understates PMPM).
    - Treating enroll_end_dt as exclusive (drops the last day).
    - Summing overlapping spans (member_months > 1 in a month).
    """
    by = list(by)
    p0, p1 = pd.Timestamp(period_start), pd.Timestamp(period_end)
    e = _merge_overlaps(enrollment, by)
    e["s"] = e["enroll_start_dt"].clip(lower=p0)
    e["e"] = e["enroll_end_dt"].clip(upper=p1)
    e = e[e["s"] <= e["e"]]
    e["month"] = [pd.period_range(s, t, freq="M") for s, t in zip(e["s"], e["e"])]
    e = e.explode("month")
    m_start = e["month"].apply(lambda p: p.start_time.normalize())
    m_end = e["month"].apply(lambda p: p.end_time.normalize())
    days = (np.minimum(e["e"], m_end) - np.maximum(e["s"], m_start)).dt.days + 1
    e["days_enrolled"] = days
    e["member_months"] = days / e["month"].apply(lambda p: p.days_in_month)
    out = e.groupby(by + ["month"], as_index=False)[["days_enrolled", "member_months"]].sum()
    out["month"] = out["month"].astype("period[M]")
    return out


SERVICE_CATEGORY_ORDER = ["IP", "ED", "OP", "HCBS", "PROF_HOME", "TELEHEALTH", "PROF", "OTHER"]


def _line_category(df: pd.DataFrame) -> pd.Series:
    tob = df["tob_cd"].fillna("")
    rev = df["rev_cd"].fillna("")
    pos = df["pos_cd"].fillna("")
    hcpcs = df["hcpcs_cd"].fillna("")
    conds = [
        tob.str.startswith("11"),                                   # inpatient hospital bill type
        rev.str.match(r"^045\d") | hcpcs.isin(["99281", "99282", "99283", "99284", "99285"]),
        tob.str.startswith("13"),                                   # hospital outpatient
        hcpcs.isin(["T1019", "S5125", "S5130", "T1020"]),           # HCBS personal care / attendant
        pos.eq("12"),                                               # professional, in the home
        pos.isin(["02", "10"]),                                     # telehealth POS
        df["claim_type_cd"].eq("PROF"),
    ]
    return pd.Series(np.select(conds, SERVICE_CATEGORY_ORDER[:-1], default="OTHER"), index=df.index)


def service_category_claim(claims: pd.DataFrame) -> pd.DataFrame:
    """STANDARD: assign ONE category per claim using a priority hierarchy.

    Healthcare context
    ------------------
    Cost and utilization reports ("IP / ED / OP / professional / HCBS") need each claim in
    exactly one bucket. The usual rule: an inpatient bill type wins over everything, then an
    ED revenue code, then outpatient, and so on — so the lab line on an ED claim is ED cost.

    Steps
    -----
    1. Categorize every line.
    2. Rank lines by SERVICE_CATEGORY_ORDER; the claim takes its highest-priority line's category.
    3. Broadcast that category back to every line of the claim.
    """
    out = claims.copy()
    out["line_category"] = _line_category(out)
    rank = out["line_category"].map({c: i for i, c in enumerate(SERVICE_CATEGORY_ORDER)})
    out["service_category"] = out.assign(_r=rank).groupby("claim_id")["_r"].transform("min").map(
        dict(enumerate(SERVICE_CATEGORY_ORDER)))
    return out.drop(columns="line_category")
