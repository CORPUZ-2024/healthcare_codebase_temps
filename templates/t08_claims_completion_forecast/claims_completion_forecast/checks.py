"""Completion and forecasting checks. Each returns a list of Finding (empty = clean)."""
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


def check_immature_months(completed: pd.DataFrame, min_completion: float = 0.5) -> list[Finding]:
    """IBNR-001: months less than half paid get most of their ultimate from a factor, not from data."""
    imm = completed[(completed["completion_pct"] < min_completion) & (completed.get("method", "CL") == "CL")]
    return [Finding("IBNR-001", "warn", f"{len(imm)} month(s) under {min_completion:.0%} complete use the chain ladder",
                    "Use Bornhuetter-Ferguson for immature months (blend_cl_bf).", len(imm))] if len(imm) else []


def check_factor_volatility(factors: pd.DataFrame, cv_warn: float = 0.10) -> list[Finding]:
    """IBNR-002: widely varying single-month factors = unstable pattern; the average hides it."""
    bad = factors[factors["cv_individual"] > cv_warn]
    return [Finding("IBNR-002", "warn", f"lags {bad['lag'].tolist()} have factor CV > {cv_warn:.0%}",
                    "Use more months in the average, BF for affected ages, and show a range.", len(bad))] if len(bad) else []


def check_negative_incrementals(tri: pd.DataFrame) -> list[Finding]:
    """IBNR-003: negative cells (recoveries, voids, reprocessing) break multiplicative factors."""
    n = int((tri < 0).sum().sum())
    return [Finding("IBNR-003", "info", f"{n} negative incremental cells",
                    "Expected with recoveries; if large, develop gross paid and recoveries separately.", n)] if n else []


def check_lag_shift(cum: pd.DataFrame, factors: pd.DataFrame, recent: int = 3, tol: float = 0.05) -> list[Finding]:
    """IBNR-004: if recent months develop from lag 0 to 1 much less than the average, claims are being
    paid faster (system change, new clearinghouse) and the old factors overstate IBNR."""
    both = cum[[0, 1]].dropna()
    both = both[both[0] > 0].tail(recent)
    if both.empty:
        return []
    recent_f = float(both[1].sum() / both[0].sum())
    avg_f = float(factors.loc[factors["lag"] == 0, "factor"].iloc[0])
    rel = recent_f / avg_f - 1
    return [Finding("IBNR-004", "warn", f"last {recent} months' lag 0->1 factor is {rel:+.1%} vs. the average",
                    "Processing speed changed: shorten the averaging window or adjust factors; confirm with claims ops.",
                    recent)] if abs(rel) > tol else []


def check_forecast_input(completed: pd.DataFrame, min_months: int = 24, min_completion: float = 0.8) -> list[Finding]:
    """FC-001/FC-002: seasonal models need >= 2 years; never forecast from incomplete (paid) PMPM."""
    out = []
    if len(completed) < min_months:
        out.append(Finding("FC-001", "error", f"only {len(completed)} months of history for a 12-month seasonal model",
                           "Use a non-seasonal model or wait for 24+ months."))
    if "completed_pmpm" not in completed:
        out.append(Finding("FC-002", "error", "forecast input is not completed PMPM", "Complete the months first (complete())."))
    return out
