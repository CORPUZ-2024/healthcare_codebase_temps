"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 8
    start: str = "2023-01-01"
    months: int = 36                 # incurred months in the data; data cut = end of the last month
    members: int = 20_000
    base_pmpm: float = 450.0
    annual_trend: float = 0.06
    max_lag: int = 18                # months; later payments are ignored (tail factor 1.0)
    cl_avg_months: int = 12          # completion factors from the most recent 12 incurred months per lag
    tail_factor: float = 1.0
    bf_max_age: int = 2              # use Bornhuetter-Ferguson for months with <= 2 lag months observed
    horizon: int = 12                # forecast months
    backtest_months: int = 6
