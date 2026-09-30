"""
Synthetic time-to-event data with KNOWN hazard ratios.

Scenario: members discharged home after a hospital stay; outcome = days to first readmission
(event_cd 1). Follow-up ends at death (event_cd 2 - a competing event, treated as censoring for the
readmission analysis; see glossary G16), disenrollment, or the study end (event_cd 0).

Hazard (Weibull, shape 0.8 -> risk highest right after discharge) x exp(linear predictor):
  program_flag (transitional care)  HR 0.75
  age per 10 years                  HR 1.16
  frailty_z                         HR 1.65 per SD
  caregiver_flag                    HR 0.82
  high_acuity_flag                  HR 2.5 for days 0-30, 1.2 after  <- NON-proportional on purpose

``immortal_time_cohort`` builds the classic design error: program start is DELAYED 0-45 days after
discharge, and members readmitted before their start date never become "program" members.

Public test data: CMS DE-SynPUF inpatient claims (admission / discharge dates -> readmission times)
and beneficiary death dates; see data/README.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIBULL_K, WEIBULL_LAMBDA = 0.8, 900.0


def _event_times(rng, lp_early, lp_late, change_day):
    """Invert a piecewise Weibull cumulative hazard: H(t) = exp(lp_early)(t/L)^k for t <= c, then + exp(lp_late)(...)."""
    e = rng.exponential(1.0, len(lp_early))
    hc = np.exp(lp_early) * (change_day / WEIBULL_LAMBDA) ** WEIBULL_K
    early = WEIBULL_LAMBDA * (e / np.exp(lp_early)) ** (1 / WEIBULL_K)
    late = WEIBULL_LAMBDA * ((e - hc) / np.exp(lp_late) + (change_day / WEIBULL_LAMBDA) ** WEIBULL_K) ** (1 / WEIBULL_K)
    return np.where(e <= hc, early, late)


def generate(n: int = 3_000, max_follow_days: int = 365, hr_program: float = 0.75, acuity_hr_early: float = 2.5,
             acuity_hr_late: float = 1.2, change_day: int = 30, seed: int = 11) -> pd.DataFrame:
    """One row per member: member_id, age, frailty_z, caregiver_flag, high_acuity_flag, program_flag,
    duration_days, event_cd (0 censored, 1 readmission, 2 death), event_flag (readmission)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 95, n)
    frailty = rng.normal(0, 1, n)
    caregiver = (rng.random(n) < 0.55).astype(int)
    acuity = (rng.random(n) < 0.3 + 0.1 * (frailty > 1)).astype(int)
    program = (rng.random(n) < 0.45).astype(int)                 # randomized-like assignment in the main cohort
    base = np.log(1.16) * (age - 70) / 10 + 0.5 * frailty + np.log(0.82) * caregiver + np.log(hr_program) * program
    t_read = _event_times(rng, base + np.log(acuity_hr_early) * acuity, base + np.log(acuity_hr_late) * acuity, change_day)
    t_death = rng.exponential(3_000 / np.exp(0.03 * (age - 70) + 0.3 * frailty))
    t_disenroll = rng.exponential(1_200, n)
    t_admin = rng.uniform(180, max_follow_days + 180, n).clip(max=max_follow_days)   # staggered entry
    t_cens = np.minimum(t_disenroll, t_admin)
    t = np.minimum.reduce([t_read, t_death, t_cens])
    code = np.select([t == t_read, t == t_death], [1, 2], 0)
    return pd.DataFrame({"member_id": [f"V{i:05d}" for i in range(1, n + 1)], "age": age, "frailty_z": frailty.round(4),
                         "caregiver_flag": caregiver, "high_acuity_flag": acuity, "program_flag": program,
                         "duration_days": np.ceil(t).astype(int).clip(min=1), "event_cd": code,
                         "event_flag": (code == 1).astype(int)})


def immortal_time_cohort(n: int = 4_000, hr_program: float = 1.0, max_start_day: int = 45, seed: int = 11) -> pd.DataFrame:
    """Delayed program start: intended start day 0-45; members readmitted before it never start.

    Columns: member_id, age, frailty_z, start_day (NaN = never started), ever_program_flag,
    duration_days, event_flag. With ``hr_program`` = 1 the program does nothing, yet 'ever vs never'
    comparisons make it look protective (the program group had to survive to its start date).
    """
    rng = np.random.default_rng(seed + 1)
    age = rng.integers(40, 95, n)
    frailty = rng.normal(0, 1, n)
    intends = rng.random(n) < 0.5
    start = np.where(intends, rng.integers(0, max_start_day + 1, n), np.nan)
    lp = np.log(1.16) * (age - 70) / 10 + 0.5 * frailty
    e = rng.exponential(1.0, n)
    # cumulative hazard with a multiplier hr after the start day
    h_start = np.where(intends, np.exp(lp) * (np.nan_to_num(start) / WEIBULL_LAMBDA) ** WEIBULL_K, np.inf)
    t_before = WEIBULL_LAMBDA * (e / np.exp(lp)) ** (1 / WEIBULL_K)
    t_after = WEIBULL_LAMBDA * ((e - h_start) / (np.exp(lp) * hr_program) + (np.nan_to_num(start) / WEIBULL_LAMBDA) ** WEIBULL_K) ** (1 / WEIBULL_K)
    t = np.where(e <= h_start, t_before, np.nan_to_num(t_after, nan=np.inf))
    cens = rng.uniform(180, 365, n)
    dur = np.minimum(t, cens)
    started = intends & (np.nan_to_num(start, nan=np.inf) < dur)
    return pd.DataFrame({"member_id": [f"W{i:05d}" for i in range(1, n + 1)], "age": age, "frailty_z": frailty.round(4),
                         "start_day": np.where(started, start, np.nan), "ever_program_flag": started.astype(int),
                         "duration_days": np.ceil(dur).astype(int).clip(min=1), "event_flag": (t <= cens).astype(int)})
