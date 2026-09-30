"""Data-quality checks. Each returns a list of Finding (empty list = clean)."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Finding:
    check_id: str   # short stable id, e.g. "DQ-001"
    severity: str   # "error" (stop) | "warn" (proceed, but report)
    message: str    # what is wrong, with counts
    fix: str        # what the analyst should do


def check_unique_key(df: pd.DataFrame, key: str) -> list[Finding]:
    """The declared grain must be unique (one row per key)."""
    dup = int(df[key].duplicated().sum())
    if dup:
        return [Finding("DQ-001", "error", f"{dup} duplicate {key} values",
                        "De-duplicate or fix the grain before aggregating.")]
    return []
