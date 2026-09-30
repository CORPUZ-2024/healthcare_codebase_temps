"""Checks specific to cost-of-care work. Each returns a list of Finding (empty = clean)."""
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


def check_claims_without_exposure(claims: pd.DataFrame, member_months: pd.DataFrame) -> list[Finding]:
    """Claims in a member-month with no exposure inflate PMPM (numerator without denominator)."""
    key = claims[["member_id", "month"]].drop_duplicates()
    m = key.merge(member_months[["member_id", "month"]].drop_duplicates(), how="left", indicator=True)
    n = int((m["_merge"] == "left_only").sum())
    return [Finding("VAL-041", "warn", f"{n} member-months have claims but no enrollment exposure",
                    "Report as unmatched spend; check retro-terminations and eligibility lag.", n)] if n else []


def check_high_cost_concentration(member_cost: pd.Series, top_pct: float = 0.01, share_warn: float = 0.20) -> list[Finding]:
    """If the top 1% of members drive > 20% of spend, report truncated and untruncated PMPM."""
    s = member_cost.sort_values(ascending=False)
    k = max(1, int(len(s) * top_pct))
    share = s.iloc[:k].sum() / s.sum() if s.sum() else 0.0
    if share > share_warn:
        return [Finding("ANL-009", "warn", f"top {top_pct:.0%} of members = {share:.0%} of spend",
                        "Show truncated and untruncated PMPM side by side.", k)]
    return []


def check_small_exposure(pmpm_table: pd.DataFrame, min_member_months: float = 1_200.0) -> list[Finding]:
    """PMPM on < ~100 member-years is too volatile to compare (credibility)."""
    small = pmpm_table[pmpm_table["member_months"] < min_member_months]
    n = len(small)
    return [Finding("VAL-042", "warn", f"{n} group(s) with fewer than {min_member_months:,.0f} member-months",
                    "Combine groups or years; don't rank on these.", n)] if n else []
