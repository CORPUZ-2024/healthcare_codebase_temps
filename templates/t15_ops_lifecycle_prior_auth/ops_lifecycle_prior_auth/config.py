"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 15
    n_referrals: int = 5_000
    start: str = "2025-01-01"
    months: int = 12
    as_of: str = "2025-12-31"            # data cut: later events are not visible yet
    expedited_target_hours: int = 72     # CMS-0057-F decision timeframes (impacted payers, from 2026)
    standard_target_hours: int = 168     # 7 calendar days
    retention_days: int = 90
    cohort_window_days: int = 30         # cohort table: started care within 30 days of referral
    report_days: tuple = (7, 14, 30, 60)
