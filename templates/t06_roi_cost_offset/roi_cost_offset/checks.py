"""Program-evaluation checks. Each returns a list of Finding (empty = clean)."""
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


def check_balance(balance: pd.DataFrame, max_abs_smd: float = 0.1) -> list[Finding]:
    """ROI-001: after matching every covariate should have |SMD| < 0.1."""
    bad = balance[balance["smd_after"].abs() >= max_abs_smd]
    return [Finding("ROI-001", "error", f"{len(bad)} covariate(s) unbalanced after matching: {bad['covariate'].tolist()}",
                    "Tighten the caliper, add the covariate to exact matching, or adjust for it in the DiD model.",
                    len(bad))] if len(bad) else []


def check_common_support(n_treated: int, n_pairs: int, max_drop: float = 0.10) -> list[Finding]:
    """ROI-002: participants dropped for lack of a comparable control change WHO the result describes."""
    drop = 1 - n_pairs / n_treated if n_treated else 0.0
    return [Finding("ROI-002", "warn", f"{drop:.0%} of participants had no control inside the caliper",
                    "Report the effect as 'among matchable participants'; describe who was dropped.",
                    n_treated - n_pairs)] if drop > max_drop else []


def check_pre_trends(pre_trend: dict, alpha: float = 0.05) -> list[Finding]:
    """ROI-003: diverging pre-period slopes break the parallel-trends assumption behind DiD."""
    if pre_trend["p_value"] < alpha:
        return [Finding("ROI-003", "error",
                        f"pre-period slopes differ by {pre_trend['slope_diff_pmpm_per_month']:.1f} PMPM/month (p={pre_trend['p_value']:.3f})",
                        "Match on pre-period trajectory, shorten the pre window, or use an event-study / synthetic control (t10).")]
    return []


def check_regression_to_mean(mp: pd.DataFrame, ratio_warn: float = 1.2) -> list[Finding]:
    """ROI-004: participants' cost peaking just before enrollment means naive pre/post overstates savings.

    Ratio = PMPM in the last 3 pre months / PMPM in the earlier pre months (ratio of sums, participants).
    """
    t = mp[mp["treated_flag"] == 1]
    last3 = t["pre_last3_paid_amt"].sum() / t["pre_last3_mm"].sum()
    earlier = (t["pre_paid_amt"].sum() - t["pre_last3_paid_amt"].sum()) / (t["pre_mm"].sum() - t["pre_last3_mm"].sum())
    ratio = float(last3 / earlier) if earlier else 0.0
    return [Finding("ROI-004", "warn", f"participants' PMPM in the 3 months before enrollment is {ratio:.2f}x the earlier pre months",
                    "Never report participants' pre/post change as savings; use a matched comparison (DiD).",
                    len(t))] if ratio > ratio_warn else []


def check_effect_precision(did: dict, breakeven_savings_pmpm: float) -> list[Finding]:
    """ROI-005: savings CI includes $0, or includes values below break-even -> ROI sign is uncertain."""
    out = []
    lo_sav, hi_sav = -did["ci_hi"], -did["ci_lo"]              # savings = -DiD
    if lo_sav <= 0:
        out.append(Finding("ROI-005", "warn", f"95% CI for savings ({lo_sav:,.0f} to {hi_sav:,.0f} PMPM) includes $0",
                           "Report a range, not a point ROI; extend follow-up or pool cohorts (see t09 for power)."))
    elif lo_sav < breakeven_savings_pmpm:
        out.append(Finding("ROI-005", "info", f"lower CI savings {lo_sav:,.0f} PMPM is below break-even {breakeven_savings_pmpm:,.0f}",
                           "State the probability of breaking even (bootstrap) alongside the point ROI."))
    return out
