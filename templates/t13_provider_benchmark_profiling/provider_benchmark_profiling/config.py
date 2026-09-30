"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 13
    n_providers: int = 80
    provider_sd: float = 0.25          # SD of true provider effects (log-odds)
    measurement_start: str = "2024-07-01"
    measurement_end: str = "2025-06-30"
    funnel_levels: tuple = (0.95, 0.998)
    min_cases: int = 25                # below this: report, don't rank
