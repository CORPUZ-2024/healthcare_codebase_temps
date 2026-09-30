"""Benchmarking checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_small_volume(oe: pd.DataFrame, min_cases: int = 25) -> list[Finding]:
    """BEN-001: providers with few cases -> report with intervals, never rank."""
    n = int((oe["n_cases"] < min_cases).sum())
    return [Finding("BEN-001", "warn", f"{n} provider(s) with < {min_cases} cases",
                    "Show them with EB estimates and intervals; exclude from rankings / tiering.", n)] if n else []


def check_overdispersion(flags: pd.DataFrame, level: float = 0.95) -> list[Finding]:
    """BEN-002: far more providers outside the 95% funnel than 5% -> real between-provider variation
    (or an unmodelled risk factor). Plain Poisson limits then flag too many providers."""
    share = float((flags[f"flag_{level}"] != 0).mean())
    return [Finding("BEN-002", "info", f"{share:.0%} of providers outside the {level:.0%} limits (expect ~{1 - level:.0%} by chance)",
                    "Use EB-shrunk / mixed-model estimates or overdispersion-adjusted limits before acting on flags.")] \
        if share > 2 * (1 - level) else []


def check_risk_model(info: dict, min_c: float = 0.6, oe_tol: float = 0.02) -> list[Finding]:
    """BEN-003: weak discrimination or overall O/E away from 1 -> expected counts unreliable."""
    out = []
    if info["c_statistic"] < min_c:
        out.append(Finding("BEN-003", "warn", f"risk model c-statistic {info['c_statistic']:.2f}",
                           "Add clinical risk factors (HCCs, prior use); weak adjustment leaves case mix in the O/E."))
    if abs(info["overall_oe"] - 1) > oe_tol:
        out.append(Finding("BEN-003", "error", f"overall O/E {info['overall_oe']:.3f} (should be 1 in the development data)",
                           "The model was fit on different data; recalibrate the intercept."))
    return out


def check_rank_disagreement(crude: pd.Series, adjusted: pd.Series, min_rho: float = 0.8) -> list[Finding]:
    """BEN-004: crude and risk-adjusted rankings disagree -> case mix differs a lot between providers."""
    rho = float(stats.spearmanr(crude, adjusted).statistic)
    return [Finding("BEN-004", "info", f"crude vs risk-adjusted rank correlation {rho:.2f}",
                    "Never publish crude rankings for this measure.")] if rho < min_rho else []


def check_benchmark_vintage(bench_start, bench_end, own_start, own_end, max_gap_days: int = 180) -> list[Finding]:
    """ANL-013: external benchmark's performance period far from yours (or not overlapping)."""
    gap = abs((pd.Timestamp(own_end) - pd.Timestamp(bench_end)).days)
    return [Finding("ANL-013", "warn", f"benchmark period {pd.Timestamp(bench_start).date()}..{pd.Timestamp(bench_end).date()} "
                    f"vs yours {pd.Timestamp(own_start).date()}..{pd.Timestamp(own_end).date()} ({gap} days apart)",
                    "Use the release whose performance period matches yours, or state the lag and trend-adjust.")] \
        if gap > max_gap_days else []
