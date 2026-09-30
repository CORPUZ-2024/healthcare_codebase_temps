"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_members: int = 1_500
    start: str = "2023-01-01"
    months: int = 24
    runout_lag_months: int = 3       # trailing months treated as incomplete
    readmit_window_days: int = 30
    transfer_gap_days: int = 1       # admit within 1 day of discharge = same stay
    frequent_ed_threshold: int = 4   # >= 4 ED visits in 12 months
    n_boot: int = 2_000
