"""Pre-fit and post-fit checks. Each returns a list of Finding (empty = clean)."""
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


def check_epv(epv: float, minimum: float = 10.0) -> list[Finding]:
    return [Finding("ANL-003", "error", f"events per variable = {epv:.1f} (< {minimum})",
                    "Fewer predictors, more data (pool years), or a penalized model.")] if epv < minimum else []


def check_separation(cols: list[str]) -> list[Finding]:
    return [Finding("ANL-003b", "warn", f"possible separation in: {', '.join(cols)}",
                    "Merge rare levels or rely on the L2 penalty; don't interpret those coefficients.", len(cols))] if cols else []


def check_leakage(features: list[str], forbidden_substrings=("next", "future", "label", "outcome", "true_")) -> list[Finding]:
    bad = [f for f in features if any(s in f.lower() for s in forbidden_substrings)]
    return [Finding("ANL-005", "error", f"feature names suggest look-ahead leakage: {bad}",
                    "Features must come only from the lookback window.", len(bad))] if bad else []


def check_calibration_drift(cal: dict, tol: float = 0.2) -> list[Finding]:
    if abs(cal["intercept"]) > tol:
        return [Finding("ANL-006", "warn", f"calibration intercept {cal['intercept']:+.2f} on the test cohort",
                        "Base rate moved: recalibrate the intercept (recalibrate_intercept) before using probabilities.")]
    return []


def check_missing(df: pd.DataFrame, cols: list[str]) -> list[Finding]:
    n = int(df[cols].isna().any(axis=1).sum())
    return [Finding("DQ-010", "error", f"{n} rows with missing features",
                    "Impute explicitly (and add a missing-indicator) or exclude and report.", n)] if n else []
