"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_members: int = 1_000            # synthetic population for the demo run
    start: str = "2023-01-01"         # first month of the synthetic period
    months: int = 24                  # length of the synthetic period
    period_start: str = "2023-01-01"  # analysis window for member-months
    period_end: str = "2024-12-31"
    midmonth_anchor_day: int = 15     # ALTERNATIVE member-month rule
