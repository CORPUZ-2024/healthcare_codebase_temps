"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 10
    # cross-section (PSM / IPW / AIPW): members offered a program by referral
    n_members: int = 6_000
    # panel (DiD / event study): practices x months, half adopt at one date
    n_practices: int = 40
    n_months: int = 36
    adopt_month: int = 24
    did_effect: float = -2.0            # ED visits per 1,000 member-months
    # interrupted time series: a monthly population rate with a policy change
    its_months: int = 60
    its_break: int = 36
    its_level_change: float = -4.0
    its_slope_change: float = -0.15
    # method settings
    caliper_sd: float = 0.2
    trim_pct: float = 0.01              # IPW weight trimming (1st / 99th percentile)
    hac_lags: int = 3                   # Newey-West lags (ITS comparison SE)
    its_se_method: str = "glsar"        # AR(1) feasible GLS; see its_segmented for the coverage evidence
    event_window: tuple = (-6, 6)       # event-study bins; months beyond are pooled into the endpoints
