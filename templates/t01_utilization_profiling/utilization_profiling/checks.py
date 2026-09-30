"""Checks specific to utilization work. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_ip_missing_dates(claims: pd.DataFrame) -> list[Finding]:
    ip = claims[claims["service_category"] == "IP"].drop_duplicates("claim_id")
    n = int((ip["admit_dt"].isna() | ip["discharge_dt"].isna()).sum())
    return [Finding("VAL-015", "error", f"{n} inpatient claims lack admit or discharge dates",
                    "Use svc_from/svc_to as fallback and flag, or exclude and report.", n)] if n else []


def check_same_day_transfers(stays_claims: pd.DataFrame) -> list[Finding]:
    """Stays assembled from >1 claim: expected for transfers/interim bills — disclose the count."""
    n = int((stays_claims["n_claims"] > 1).sum())
    return [Finding("ANL-020", "warn", f"{n} stays combine multiple claims (transfers or interim bills)",
                    "Confirm the transfer rule with the clinical team; report admissions from stays, not claims.", n)] if n else []


def check_runout(monthly: pd.DataFrame, months: int = 3, threshold: float = 0.7) -> list[Finding]:
    r = monthly["rate_per_1000"].dropna()
    if len(r) < months + 6:
        return []
    base = r.iloc[-(months + 6):-months].mean()
    low = r.iloc[-months:][r.iloc[-months:] < threshold * base]
    return [Finding("VAL-031", "warn", f"trailing months below {threshold:.0%} of baseline: {', '.join(map(str, low.index))}",
                    "Pass incomplete_after= to the trend functions (or complete with t08).", len(low))] if len(low) else []


def check_small_denominator(index_stays: int, minimum: int = 30) -> list[Finding]:
    return [Finding("VAL-042", "warn", f"only {index_stays} eligible index stays",
                    "Rates on < 30 index stays are unstable; pool periods or report counts only.", index_stays)] \
        if index_stays < minimum else []
