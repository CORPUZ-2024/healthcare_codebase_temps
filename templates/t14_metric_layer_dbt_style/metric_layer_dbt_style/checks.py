"""Metric-layer and experiment checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_schema_tests(results: list[dict]) -> list[Finding]:
    """MET-001: any failing data test blocks publication of the metrics."""
    bad = [r for r in results if r["status"] == "FAIL"]
    return [Finding("MET-001", "error", f"{len(bad)} schema test(s) failed: " + "; ".join(f"{r['test']} ({r['failures']})" for r in bad),
                    "Fix the upstream data or the staging model; do not publish metrics built on failing tests.", len(bad))] if bad else []


def check_srm(srm: dict, alpha: float = 0.001) -> list[Finding]:
    """MET-002: sample ratio mismatch."""
    return [Finding("MET-002", "error", f"sample ratio mismatch: treatment share {srm['treatment_share']:.3f}, p = {srm['p_value']:.1e}",
                    "Find where units were lost (assignment logging, joins, eligibility filters) before reading any result.")] \
        if srm["p_value"] < alpha else []


def check_guardrails(readout: pd.DataFrame) -> list[Finding]:
    """MET-003: a guardrail metric cannot rule out unacceptable harm."""
    bad = readout[(readout["role"] == "guardrail") & readout["verdict"].str.startswith("FAIL")]
    return [Finding("MET-003", "error", f"guardrail(s) failed: {bad['metric'].tolist()}",
                    "Do not ship on the primary metric alone; investigate or run longer.", len(bad))] if len(bad) else []


def check_engine_parity(sql_table: pd.DataFrame, pandas_table: pd.DataFrame, tol: float = 1e-9) -> list[Finding]:
    """MET-004: the SQL and pandas engines disagree -> a definition drifted (NULLs, casting, filters)."""
    num = [c for c in sql_table.columns if pd.api.types.is_numeric_dtype(sql_table[c])]
    diff = float(np.nanmax(np.abs(sql_table[num].to_numpy(float) - pandas_table[num].to_numpy(float))))
    return [Finding("MET-004", "error", f"SQL vs pandas metric difference {diff:.3g}",
                    "Align NULL handling and casting; the YAML is the single source of truth.")] if diff > tol else []
