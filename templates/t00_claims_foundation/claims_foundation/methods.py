"""
Claims foundation methods: raw claims + enrollment -> analytic-ready tables.

Each step has a STANDARD method (the industry default) and an ALTERNATIVE with a
stated trade-off. See README "Method choices".

    collapse_versions_latest   (STANDARD)  vs  collapse_versions_net   (ALTERNATIVE)
    member_months_daily        (STANDARD)  vs  member_months_midmonth (ALTERNATIVE)
    service_category_claim     (STANDARD)  vs  service_category_line  (ALTERNATIVE)
    net_pharmacy_drop_pairs    (STANDARD)  vs  net_pharmacy_signed    (ALTERNATIVE)
    normalize_ndc11            (single method — a format rule, not a modelling choice)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Claim version collapse
# ---------------------------------------------------------------------------

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


def collapse_versions_net(claims: pd.DataFrame, amount_cols: tuple[str, ...] = ("paid_amt",)) -> pd.DataFrame:
    """ALTERNATIVE: sum every version per claim line (for DELTA-style feeds).

    Trade-off
    ---------
    + Correct for feeds where adjustments are differences and voids are negative rows.
    + Keeps a full audit trail of amounts.
    - Non-amount fields (codes, dates) are taken from the latest version, so a coding change
      is only visible if you look at the history.
    - Wrong for replacement feeds (it double counts) — confirm the feed type with the payer's
      companion guide before choosing.

    Steps
    -----
    1. Sort by version; take non-amount columns from the last version of each line.
    2. Sum amount columns across all versions.
    3. Drop lines whose net paid is 0 AND whose last status is 'V' (fully voided).
    """
    keys = ["claim_id", "line_seq"]
    c = claims.sort_values(keys + ["adj_seq"])
    last = c.groupby(keys, as_index=False).last()
    sums = c.groupby(keys, as_index=False)[list(amount_cols)].sum()
    out = last.drop(columns=list(amount_cols)).merge(sums, on=keys)
    for col in amount_cols:
        out[col] = out[col].round(2)
    out = out[~((out["claim_status_cd"] == "V") & (out["paid_amt"].abs() < 0.005))]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Member months
# ---------------------------------------------------------------------------

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


def member_months_midmonth(enrollment: pd.DataFrame, period_start: str, period_end: str,
                           by: tuple[str, ...] = ("member_id",), anchor_day: int = 15) -> pd.DataFrame:
    """ALTERNATIVE: count a full member-month if enrolled on the anchor day (default the 15th).

    Trade-off
    ---------
    + Matches many capitation/premium contracts ("enrolled on the 15th = paid for the month"),
      so it reconciles to Finance's premium file.
    + Integer member-months; simpler to explain.
    - Biased when joins/terminations cluster around the anchor; short stays (< 1 month that
      miss the anchor) contribute 0 even though they may have claims -> PMPM overstated.

    Steps
    -----
    1. Merge overlapping spans. 2. For each month, anchor date = YYYY-MM-<anchor_day>.
    3. member_months = 1 if start <= anchor <= end, else 0.
    """
    by = list(by)
    e = _merge_overlaps(enrollment, by)
    months = pd.period_range(period_start, period_end, freq="M")
    rows = []
    for m in months:
        anchor = m.start_time + pd.Timedelta(days=anchor_day - 1)
        hit = e[(e["enroll_start_dt"] <= anchor) & (e["enroll_end_dt"] >= anchor)]
        if len(hit):
            rows.append(hit[by].assign(month=m, member_months=1.0))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=by + ["month", "member_months"])
    return out.groupby(by + ["month"], as_index=False)["member_months"].sum()


# ---------------------------------------------------------------------------
# 3. Service category
# ---------------------------------------------------------------------------
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


def service_category_line(claims: pd.DataFrame) -> pd.DataFrame:
    """ALTERNATIVE: categorize each LINE independently.

    Trade-off
    ---------
    + Shows what services were delivered (the lab on an ED claim counts as lab/OP).
    + Useful for fee-schedule and unit-cost work.
    - Totals by category will not match claim-level reports, and ED "visits" can't be
      counted from lines. Label outputs clearly.
    """
    out = claims.copy()
    out["service_category"] = _line_category(out)
    return out


# ---------------------------------------------------------------------------
# 4. Pharmacy reversals
# ---------------------------------------------------------------------------

def net_pharmacy_drop_pairs(rx: pd.DataFrame) -> pd.DataFrame:
    """STANDARD: remove each reversal AND the fill it reverses (same rx_claim_id).

    Needed for adherence (PDC) and days-supply logic, where a reversed fill never happened.
    """
    reversed_ids = set(rx.loc[rx["reversal_flag"] == 1, "rx_claim_id"])
    return rx[~rx["rx_claim_id"].isin(reversed_ids)].reset_index(drop=True)


def net_pharmacy_signed(rx: pd.DataFrame) -> pd.DataFrame:
    """ALTERNATIVE: keep every row and sum signed amounts per rx_claim_id.

    Trade-off: correct for DOLLARS (reconciles to the ledger, keeps the audit trail) but a
    fill with net $0 still looks like a fill — never use this output for adherence.
    """
    agg = rx.groupby("rx_claim_id", as_index=False).agg(
        member_id=("member_id", "first"), fill_dt=("fill_dt", "first"), ndc_cd=("ndc_cd", "first"),
        drug_class=("drug_class", "first"), days_supply=("days_supply", "first"),
        paid_amt=("paid_amt", "sum"), n_rows=("paid_amt", "size"))
    agg["paid_amt"] = agg["paid_amt"].round(2)
    return agg


# ---------------------------------------------------------------------------
# 5. NDC normalization
# ---------------------------------------------------------------------------

def normalize_ndc11(ndc: str | None) -> str | None:
    """Convert a 10-digit NDC (4-4-2, 5-3-2 or 5-4-1, hyphenated) to 11-digit 5-4-2.

    Healthcare context
    ------------------
    Labels print 10-digit NDCs in three segment layouts; claims and CMS files use 11 digits
    (5-4-2). The zero goes in front of whichever segment is short. Without hyphens a
    10-digit NDC is ambiguous — you cannot know where the zero goes.

    >>> normalize_ndc11("1234-5678-90"), normalize_ndc11("12345-678-90"), normalize_ndc11("12345-6789-0")
    ('01234567890', '12345067890', '12345678900')
    >>> normalize_ndc11("12345678901")
    '12345678901'
    """
    if ndc is None or (isinstance(ndc, float) and np.isnan(ndc)):
        return None
    s = str(ndc).strip()
    if "-" in s:
        a, b, c = s.split("-")
        return a.zfill(5) + b.zfill(4) + c.zfill(2)
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if len(digits) == 11 else None     # 10 digits w/o hyphens: ambiguous -> None
