"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_per_cohort: int = 6_000
    train_year: int = 2023
    test_year: int = 2024
    top_pct: float = 0.05          # care-management capacity for Tier 1
    tier_shares: tuple = (0.05, 0.15)
    min_epv: float = 10.0          # stop if events-per-variable is below this
    logistic_C: float = 1.0        # inverse L2 penalty strength
    rising_risk_increase: float = 0.05
