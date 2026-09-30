"""Risk-adjustment checks. Each returns a list of Finding (empty = clean)."""
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


def check_model_year(dx: pd.DataFrame, payment_year: int, prospective: bool = True) -> list[Finding]:
    """ANL-008: a prospective model scores payment year Y from diagnoses in Y-1."""
    expected = payment_year - 1 if prospective else payment_year
    bad = int((dx["svc_dt"].dt.year != expected).sum())
    return [Finding("ANL-008", "error", f"{bad} diagnoses are not from service year {expected}",
                    "Filter diagnoses to the model's data-collection year, or use the concurrent model.", bad)] if bad else []


def check_unacceptable_sources(dx: pd.DataFrame, acceptable: set[str]) -> list[Finding]:
    n = int((~dx["source_cd"].isin(acceptable)).sum())
    return [Finding("RA-010", "warn", f"{n} diagnosis rows come from non-acceptable sources (lab/DME/other)",
                    "Exclude them from scoring; route to suspect-gap review instead.", n)] if n else []


def check_unmapped_share(dx: pd.DataFrame, member_hcc: pd.DataFrame, warn_above: float = 0.9) -> list[Finding]:
    """Most dx codes don't map to a payment HCC; a share near 100% usually means a format problem
    (decimals, ICD-9, lower case)."""
    mapped_members = member_hcc["member_id"].nunique()
    members = dx["member_id"].nunique()
    share = 1 - mapped_members / members if members else 0.0
    return [Finding("RA-020", "warn", f"{share:.0%} of members with diagnoses have no payment HCC",
                    "Check dx format (no decimals, upper case) and code system (ICD-10).", members - mapped_members)] \
        if share > warn_above else []


def check_score_range(score: pd.Series, lo: float = 0.0, hi: float = 15.0) -> list[Finding]:
    n = int(((score < lo) | (score > hi)).sum())
    return [Finding("RA-030", "error", f"{n} scores outside [{lo}, {hi}]",
                    "Look for duplicated HCC rows or a coefficient table in the wrong units.", n)] if n else []
