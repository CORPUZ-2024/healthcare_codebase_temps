"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 11
    n_members: int = 3_000
    max_follow_days: int = 365
    true_hr_program: float = 0.75       # transitional-care program vs usual care
    acuity_hr_early: float = 2.5        # high acuity: HR in the first 30 days ...
    acuity_hr_late: float = 1.2         # ... and after day 30 (non-proportional on purpose)
    acuity_change_day: int = 30
    report_days: tuple = (30, 90, 180, 365)
    interval_days: int = 30             # discrete-time hazard period length
    rmst_horizon: int = 180
