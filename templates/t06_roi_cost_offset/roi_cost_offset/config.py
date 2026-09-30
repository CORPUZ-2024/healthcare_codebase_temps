"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 21
    n_members: int = 10_000
    start: str = "2024-01-01"
    months: int = 24
    true_effect_pct: float = 0.20       # synthetic truth: program cuts treated post-period cost by 20%
    pre_months: int = 6                 # months -6..-1 before the index (program start) month
    post_months: int = 6                # months +1..+6 after; the index month itself is excluded
    min_months_each_period: int = 4     # exposure needed in BOTH periods to be analysable
    caliper_sd: float = 0.2             # caliper = 0.2 x SD of the logit propensity score
    exact_on: tuple = ("lob_cd", "pre_last3_admit_flag")   # hard-match: LOB and the referral trigger
    n_boot: int = 1_000
    # ROI inputs (FAKE program economics)
    program_fee_pmpm: float = 150.0     # intensive care-management fee per participant month
    one_time_cost: float = 300.0        # enrollment + assessment per participant
    participants: int = 500
    months_in_program: int = 12
