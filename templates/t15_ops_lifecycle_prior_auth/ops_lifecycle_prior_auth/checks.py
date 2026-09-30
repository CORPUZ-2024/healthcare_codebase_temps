"""Operations checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .methods import STAGES


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_turnaround(pa: pd.DataFrame, max_miss: float = 0.05) -> list[Finding]:
    """OPS-001: PA decisions missing the regulatory timeframe."""
    bad = pa[(1 - pa["pct_within_target"]) > max_miss]
    return [Finding("OPS-001", "warn", "; ".join(f"{r.priority}: {1 - r.pct_within_target:.0%} decided after target" for r in bad.itertuples()),
                    "Break down by payer and reason; escalate payers that miss CMS-0057-F timeframes.", len(bad))] if len(bad) else []


def check_immature_cohorts(cohorts: pd.DataFrame) -> list[Finding]:
    """OPS-002: cohorts without a full observation window must not be reported as conversion rates."""
    n = int((cohorts["mature_flag"] == 0).sum())
    return [Finding("OPS-002", "info", f"{n} referral cohort(s) still immature",
                    "Show them as 'in progress' or use the KM curve; never as low conversion.", n)] if n else []


def check_timestamp_order(df: pd.DataFrame) -> list[Finding]:
    """OPS-003: a later stage dated before an earlier one = data-entry or mapping error.

    Every (earlier, later) pair is compared, not just neighbours: a start-of-care date before the
    referral must be caught even when the stages in between are blank.
    """
    bad = pd.Series(False, index=df.index)
    cols = [c for _, c in STAGES]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            bad |= df[b].notna() & df[a].notna() & (df[b] < df[a].dt.normalize())
    n = int(bad.sum())
    return [Finding("OPS-003", "error", f"{n} referral(s) with stage dates out of order",
                    "Fix at the source system; exclude from duration metrics until fixed.", n)] if n else []


def check_overturns(pa: pd.DataFrame, max_rate: float = 0.25) -> list[Finding]:
    """OPS-004: many denials overturned on appeal = denials that shouldn't have happened (cost and delay)."""
    bad = pa[pa["pct_overturned_of_appealed"] > max_rate]
    return [Finding("OPS-004", "warn", "; ".join(f"{r.priority}: {r.pct_overturned_of_appealed:.0%} of appealed denials overturned"
                                                  for r in bad.itertuples()),
                    "Review the top denial reasons (Pareto) with the payer; documentation fixes often clear most.", len(bad))] \
        if len(bad) else []


def check_open_cases(censored_share: float, max_share: float = 0.10) -> list[Finding]:
    """OPS-005: many referrals still open -> medians among completed cases understate time to care."""
    return [Finding("OPS-005", "info", f"{censored_share:.0%} of referrals have not started care (yet or ever)",
                    "Report KM-based time to start (time_to_start), not the median among those who started.")] \
        if censored_share > max_share else []
