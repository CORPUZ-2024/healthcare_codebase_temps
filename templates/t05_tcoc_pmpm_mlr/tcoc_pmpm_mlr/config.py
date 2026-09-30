"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    n_members: int = 1_500
    start: str = "2023-01-01"
    months: int = 24
    runout_lag_months: int = 3          # trailing months excluded as incomplete
    truncation_pct: float = 0.99        # STANDARD high-cost claimant cap
    attachment_amt: float = 100_000.0   # ALTERNATIVE fixed attachment (small synthetic population)
    mlr_minimum: float = 0.85           # Medicaid managed care / MA minimum
    qi_expense_pct_premium: float = 0.01
    taxes_fees_pct_premium: float = 0.02
    episode_pre_days: int = 3
    episode_post_days: int = 30
    program_savings_pmpm: float = 25.0  # scenario for mlr_impact
    program_fee_pmpm: float = 15.0
