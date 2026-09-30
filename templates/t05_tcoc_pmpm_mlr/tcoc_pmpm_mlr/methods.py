"""
Total cost of care (TCOC), PMPM, trend decomposition, medical loss ratio, episodes.

    pmpm_ratio_of_sums          (STANDARD)  vs  pmpm_mean_of_members       (ALTERNATIVE)
    truncate_percentile         (STANDARD)  vs  truncate_fixed_attachment  (ALTERNATIVE)
    decompose_additive          (STANDARD)  vs  decompose_log              (ALTERNATIVE)
    mlr_regulatory              (STANDARD)  vs  mlr_simple                 (ALTERNATIVE)
    build_episodes_nonoverlap   (STANDARD)  vs  build_episodes_overlap     (ALTERNATIVE)
    + mlr_impact (what a PMPM saving does to MLR), complete_months (runout guard)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Runout guard
# ---------------------------------------------------------------------------

def complete_months(claims: pd.DataFrame, member_months: pd.DataFrame, lag_months: int = 3) -> pd.DataFrame:
    """Drop the last ``lag_months`` months of exposure AND claims-month (claims still arriving).

    Healthcare context
    ------------------
    A service in month M is paid weeks-to-months later. The trailing months therefore look
    cheap. The simplest guard is to stop the analysis ``lag_months`` before the data cut;
    the better one is to complete them with factors (t08).

    Returns the member_months frame restricted to complete months; filter claims with the
    same ``month`` values.
    """
    last = member_months["month"].max()
    cutoff = last - lag_months
    return member_months[member_months["month"] <= cutoff].copy()


# ---------------------------------------------------------------------------
# 1. PMPM
# ---------------------------------------------------------------------------

def attach_month(claims: pd.DataFrame, date_col: str = "svc_from_dt") -> pd.DataFrame:
    """Add ``month`` (Period[M]) = incurred month (service date), the convention for TCOC."""
    out = claims.copy()
    out["month"] = out[date_col].dt.to_period("M")
    return out


def pmpm_ratio_of_sums(claims: pd.DataFrame, member_months: pd.DataFrame, by: list[str] | None = None,
                       amount_col: str = "paid_amt") -> pd.DataFrame:
    """STANDARD: PMPM = sum(paid) / sum(member_months) within each group.

    Healthcare context
    ------------------
    This is the actuarial/Finance definition. It weights every member by exposure, so a member
    enrolled 2 months counts 1/6 as much as a member enrolled 12. Report it by service
    category and period; add utilization per 1,000 and cost per unit to explain changes.

    Parameters
    ----------
    claims : must have member_id, month and ``amount_col`` (+ any ``by`` columns present on claims)
    member_months : member_id, month, member_months (+ ``by`` columns that live on members)
    by : grouping columns (e.g. ["lob_cd"], ["month"], ["service_category"])

    Returns
    -------
    DataFrame: *by, paid_amt, member_months, pmpm

    Steps
    -----
    1. Restrict claims to months present in member_months (no exposure = no PMPM); copy
       member-level group columns (e.g. lob_cd) from member_months onto claims.
    2. Sum paid by group; sum member-months by group (member-level groups only).
    3. PMPM = paid / member_months.

    Common mistakes
    ---------------
    - Grouping member-months by a CLAIM attribute (service_category): exposure does not split
      by service. Use total member-months as the denominator for every category.
    - Mixing incurred-month claims with paid-month exposure.
    """
    by = by or []
    c = claims[claims["month"].isin(set(member_months["month"]))]          # step 1
    if not by:
        paid, mm = c[amount_col].sum(), member_months["member_months"].sum()
        return pd.DataFrame({"paid_amt": [paid], "member_months": [mm], "pmpm": [paid / mm]})
    mm_keys = [b for b in by if b in member_months.columns]
    need = [k for k in mm_keys if k not in c.columns]      # member attributes (e.g. lob_cd) -> claims
    if need:
        attrs = member_months[["member_id", "month"] + need].drop_duplicates(["member_id", "month"])
        c = c.merge(attrs, on=["member_id", "month"], how="left")
    num = c.groupby(by, dropna=False)[amount_col].sum().rename("paid_amt").reset_index()   # step 2
    if mm_keys:
        den = member_months.groupby(mm_keys)["member_months"].sum().reset_index()
        if len(mm_keys) == len(by):          # every group has exposure; groups with no claims = $0
            out = den.merge(num, on=mm_keys, how="left").fillna({"paid_amt": 0.0})
        else:
            out = num.merge(den, on=mm_keys, how="left")
    else:                                    # claim-only groups (service_category): total exposure
        out = num.assign(member_months=member_months["member_months"].sum())
    out["pmpm"] = out["paid_amt"] / out["member_months"]                   # step 3
    return out[by + ["paid_amt", "member_months", "pmpm"]]


def pmpm_mean_of_members(claims: pd.DataFrame, member_months: pd.DataFrame, amount_col: str = "paid_amt",
                         min_member_months: float = 0.0) -> dict:
    """ALTERNATIVE: average of each member's own PMPM (member-weighted), with a 95% CI.

    Trade-off
    ---------
    + Gives a per-member distribution (median, percentiles, CI) — useful for clinical or
      program-level comparisons where each person counts once.
    - Short-enrolled members get a very noisy PMPM (one ED visit in a 0.3-month stay = huge
      PMPM) and pull the mean up. It will NOT reconcile to Finance. Use ``min_member_months``
      to exclude short stays and say so.

    Returns dict: mean, median, ci_lo, ci_hi, n_members
    """
    mm = member_months.groupby("member_id")["member_months"].sum()
    paid = claims[claims["month"].isin(set(member_months["month"]))].groupby("member_id")[amount_col].sum()
    per = (paid.reindex(mm.index, fill_value=0.0) / mm)[mm > min_member_months]
    se = per.std(ddof=1) / np.sqrt(len(per))
    return {"mean": float(per.mean()), "median": float(per.median()), "ci_lo": float(per.mean() - 1.96 * se),
            "ci_hi": float(per.mean() + 1.96 * se), "n_members": int(len(per))}


def utilization_unit_cost(claims: pd.DataFrame, member_months: pd.DataFrame, by: str = "service_category",
                          unit_col: str = "claim_id") -> pd.DataFrame:
    """PMPM = (units per 1,000 member-years) x (cost per unit) / 12,000.

    Units default to distinct claims. For IP use admissions (distinct claim_id on IP claims);
    for HCBS, units of service would be better (15-minute units) if available.
    """
    mm_total = member_months["member_months"].sum()
    g = claims.groupby(by).agg(paid_amt=("paid_amt", "sum"), units=(unit_col, "nunique")).reset_index()
    g["util_per_1000"] = g["units"] / mm_total * 12_000
    g["cost_per_unit"] = g["paid_amt"] / g["units"]
    g["pmpm"] = g["paid_amt"] / mm_total
    return g


# ---------------------------------------------------------------------------
# 2. High-cost claimant truncation
# ---------------------------------------------------------------------------

def member_period_cost(claims: pd.DataFrame, amount_col: str = "paid_amt") -> pd.Series:
    return claims.groupby("member_id")[amount_col].sum()


def truncate_percentile(member_cost: pd.Series, pct: float = 0.99) -> pd.DataFrame:
    """STANDARD: cap each member's period cost at the ``pct`` percentile of the population.

    Healthcare context
    ------------------
    A handful of catastrophic cases can swing a small population's PMPM by 10-20%. VBC
    benchmarks (e.g. MSSP) truncate member costs at a high percentile so one transplant
    does not decide whether the program "worked". Always report truncated AND untruncated.

    Returns member_id, cost_amt, cap_amt, truncated_amt, excess_amt
    """
    cap = float(member_cost.quantile(pct))
    out = member_cost.rename("cost_amt").to_frame()
    out["cap_amt"] = cap
    out["truncated_amt"] = out["cost_amt"].clip(upper=cap)
    out["excess_amt"] = out["cost_amt"] - out["truncated_amt"]
    return out.reset_index()


def truncate_fixed_attachment(member_cost: pd.Series, attachment: float = 250_000.0) -> pd.DataFrame:
    """ALTERNATIVE: cap at a fixed dollar attachment point (stop-loss / reinsurance style).

    Trade-off
    ---------
    + Matches how contracts and stop-loss policies are written ("specific attachment $250K");
      stable year to year, so trend isn't distorted by a moving cap.
    - In a small or low-cost population nothing may reach the attachment (no protection);
      a percentile cap always trims the tail.
    """
    out = member_cost.rename("cost_amt").to_frame()
    out["cap_amt"] = attachment
    out["truncated_amt"] = out["cost_amt"].clip(upper=attachment)
    out["excess_amt"] = out["cost_amt"] - out["truncated_amt"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# 3. Trend decomposition
# ---------------------------------------------------------------------------

def decompose_additive(base: dict, current: dict) -> dict:
    """STANDARD: split a PMPM change into utilization, unit cost, and interaction.

    Inputs are dicts with util_per_1000 and cost_per_unit (one service category).
    PMPM = U x C / 12,000.
      utilization effect = (U1 - U0) x C0 / 12,000
      unit-cost effect   = (C1 - C0) x U0 / 12,000
      interaction        = (U1 - U0) x (C1 - C0) / 12,000
    Sum of the three = PMPM1 - PMPM0 exactly.

    Common mistake: dropping the interaction, so the pieces don't add up to the total.
    """
    u0, c0, u1, c1 = base["util_per_1000"], base["cost_per_unit"], current["util_per_1000"], current["cost_per_unit"]
    return {"pmpm_change": (u1 * c1 - u0 * c0) / 12_000, "utilization": (u1 - u0) * c0 / 12_000,
            "unit_cost": (c1 - c0) * u0 / 12_000, "interaction": (u1 - u0) * (c1 - c0) / 12_000}


def decompose_log(base: dict, current: dict) -> dict:
    """ALTERNATIVE: multiplicative (log) decomposition — no interaction term.

    ln(PMPM1/PMPM0) = ln(U1/U0) + ln(C1/C0); each piece's share of the % change is its share
    of the log change.

    Trade-off
    ---------
    + Order-independent, no leftover interaction; trends compound correctly across years —
      the way actuaries quote "utilization trend 3%, unit-cost trend 4%".
    - Undefined when a category had zero units in either period.
    """
    u0, c0, u1, c1 = base["util_per_1000"], base["cost_per_unit"], current["util_per_1000"], current["cost_per_unit"]
    lu, lc = np.log(u1 / u0), np.log(c1 / c0)
    total = lu + lc
    return {"total_trend_pct": np.expm1(total), "utilization_trend_pct": np.expm1(lu),
            "unit_cost_trend_pct": np.expm1(lc), "utilization_share": lu / total if total else np.nan}


# ---------------------------------------------------------------------------
# 4. Medical loss ratio
# ---------------------------------------------------------------------------

def mlr_regulatory(incurred_claims: float, quality_improvement: float, premium: float,
                   taxes_fees: float, minimum: float = 0.85) -> dict:
    """STANDARD: MLR = (incurred claims + quality-improvement expense) / (premium - taxes & fees).

    Healthcare context
    ------------------
    This is the structure used by the ACA commercial rule, Medicare Advantage/Part D and the
    Medicaid managed care rule (42 CFR 438.8, 85% minimum). Plans below the minimum may owe
    rebates (commercial/MA) or remittances (some state Medicaid contracts). A VBC partner's
    services usually count as claims or QI — which is why "family-led care lowers MLR" is not
    automatic: it depends how the fee is classified.

    Simplifications (documented, not modelled): credibility adjustment for small plans,
    multi-year averaging, and the detailed QI/tax definitions in the regulation.
    """
    denom = premium - taxes_fees
    mlr = (incurred_claims + quality_improvement) / denom
    shortfall = max(0.0, minimum - mlr) * denom
    return {"mlr": mlr, "minimum": minimum, "meets_minimum": mlr >= minimum, "rebate_or_remittance": shortfall}


def mlr_simple(paid_claims: float, premium: float) -> dict:
    """ALTERNATIVE: paid claims / premium ("loss ratio" as used in quick internal views).

    Trade-off: easy and available monthly, but ignores QI expense, taxes/fees and IBNR, so it
    is usually LOWER than the regulatory MLR early in a year (claims not yet paid). Never
    quote it as "the MLR" to a health plan.
    """
    return {"loss_ratio": paid_claims / premium}


def mlr_impact(pmpm_claims: float, pmpm_premium: float, pmpm_savings: float, pmpm_program_fee: float,
               fee_counts_as: str = "claims") -> dict:
    """What happens to a plan's MLR when a program saves ``pmpm_savings`` and charges ``pmpm_program_fee``.

    fee_counts_as: "claims" (fee is a medical expense), "qi" (quality improvement — also in the
    numerator) or "admin" (outside the numerator). The classification changes the answer.
    """
    before = pmpm_claims / pmpm_premium
    numer = pmpm_claims - pmpm_savings + (pmpm_program_fee if fee_counts_as in ("claims", "qi") else 0.0)
    after = numer / pmpm_premium
    return {"mlr_before": before, "mlr_after": after, "mlr_change_pts": (after - before) * 100,
            "net_pmpm_to_plan": pmpm_savings - pmpm_program_fee}


# ---------------------------------------------------------------------------
# 5. Episodes (optional module)
# ---------------------------------------------------------------------------

def _episode_cost(triggers: pd.DataFrame, claims: pd.DataFrame, pre_days: int, post_days: int) -> pd.DataFrame:
    t = triggers.copy()
    t["win_start"] = t["anchor_dt"] - pd.Timedelta(days=pre_days)
    t["win_end"] = t["anchor_end_dt"] + pd.Timedelta(days=post_days)
    j = t.merge(claims[["member_id", "claim_id", "svc_from_dt", "paid_amt"]], on="member_id", how="left")
    inwin = j[(j["svc_from_dt"] >= j["win_start"]) & (j["svc_from_dt"] <= j["win_end"])]
    cost = inwin.groupby("episode_id").agg(episode_paid_amt=("paid_amt", "sum"), n_claims=("claim_id", "nunique"))
    return t.merge(cost, left_on="episode_id", right_index=True, how="left").fillna(
        {"episode_paid_amt": 0.0, "n_claims": 0})


def ip_triggers(claims: pd.DataFrame) -> pd.DataFrame:
    """One trigger per inpatient stay: member_id, trigger_claim_id, anchor_dt (admit), anchor_end_dt (discharge)."""
    ip = claims[claims["service_category"] == "IP"].drop_duplicates("claim_id")
    return (ip[["member_id", "claim_id", "admit_dt", "discharge_dt"]]
            .rename(columns={"claim_id": "trigger_claim_id", "admit_dt": "anchor_dt", "discharge_dt": "anchor_end_dt"})
            .sort_values(["member_id", "anchor_dt"]).reset_index(drop=True))


def build_episodes_nonoverlap(claims: pd.DataFrame, pre_days: int = 3, post_days: int = 30) -> pd.DataFrame:
    """STANDARD: inpatient-anchored episodes; a new trigger inside an open episode does NOT
    start another one (its cost belongs to the first episode) — the bundled-payment convention.

    Steps
    -----
    1. Triggers = IP stays. 2. Walk each member's triggers in date order; skip a trigger whose
    admit falls before the current episode's end (discharge + post_days). 3. Sum all claims
    with service dates in [admit - pre_days, discharge + post_days].
    """
    trig = ip_triggers(claims)
    keep, last_end = [], {}
    for r in trig.itertuples():
        end = last_end.get(r.member_id)
        if end is not None and r.anchor_dt <= end:
            continue
        keep.append(r.Index)
        last_end[r.member_id] = r.anchor_end_dt + pd.Timedelta(days=post_days)
    t = trig.loc[keep].copy()
    t["episode_id"] = [f"EP{i:06d}" for i in range(1, len(t) + 1)]
    return _episode_cost(t, claims, pre_days, post_days)


def build_episodes_overlap(claims: pd.DataFrame, pre_days: int = 3, post_days: int = 30) -> pd.DataFrame:
    """ALTERNATIVE: every IP stay starts its own episode (windows may overlap).

    Trade-off
    ---------
    + Each admission gets its own outcome window — the convention for readmission-style
      measures and for "cost following any admission" analyses.
    - Claims inside overlapping windows are counted in more than one episode, so episode
      costs cannot be summed into a total. Never add them up for a budget.
    """
    t = ip_triggers(claims)
    t["episode_id"] = [f"EO{i:06d}" for i in range(1, len(t) + 1)]
    return _episode_cost(t, claims, pre_days, post_days)
