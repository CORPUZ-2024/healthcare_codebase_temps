"""
Synthetic member-period feature table with a future-hospitalization label.

Design (why it looks like this)
-------------------------------
* Features come from a 12-month LOOKBACK; the label is any inpatient admission in the next
  6 months (LOOKAHEAD). Nothing from the lookahead leaks into features.
* Two cohorts (``cohort_year`` 2023 and 2024) so you can train on one and test on the next —
  a temporal split, which is how the model will actually be used.
* The true risk has non-linear pieces (a threshold on ED visits, an interaction between heart
  failure and living alone), so gradient boosting has something to find that a main-effects
  logistic model misses.
* The 2024 cohort has a small shift (more telehealth, slightly lower baseline admissions) so
  calibration drift is visible.

Public test data
----------------
Build the same table from CMS DE-SynPUF (Beneficiary Summary for demographics and chronic
condition flags SP_CHF, SP_DIABETES, ...; inpatient claims for prior and next admissions):
https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
See data/README.md for the column recipe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["age", "female_flag", "dual_flag", "n_chronic", "chf_flag", "copd_flag", "diabetes_flag",
            "prior_ip_cnt", "prior_ed_cnt", "prior_cost_amt", "pdc_pct", "hcbs_hours_per_week",
            "lives_alone_flag", "adi_decile", "cg_program_flag", "telehealth_cnt"]


def generate(n_per_cohort: int = 4_000, seed: int = 42, cohorts: tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    """One row per member x cohort_year with FEATURES, ``label_ip_6m`` and ``true_prob``."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, year in enumerate(cohorts):
        n = n_per_cohort
        age = np.clip(rng.normal(55, 22, n), 1, 99)
        chf = rng.random(n) < 0.06 + 0.002 * np.maximum(age - 50, 0)
        copd = rng.random(n) < 0.08
        diab = rng.random(n) < 0.18
        n_chronic = chf + copd + diab + rng.poisson(1.1, n)
        prior_ip = rng.poisson(0.15 + 0.5 * chf + 0.3 * copd, n)
        prior_ed = rng.poisson(0.5 + 0.4 * n_chronic / 2 + 0.4 * prior_ip, n)
        lives_alone = rng.random(n) < 0.25
        adi = rng.integers(1, 11, n)
        cg = rng.random(n) < 0.3
        tele = rng.poisson(1.0 + 1.0 * k, n)                          # drift: more telehealth in 2024
        pdc = np.clip(rng.beta(6, 2, n) - 0.1 * (adi > 7), 0, 1)
        hcbs = np.where(rng.random(n) < 0.35, rng.gamma(2, 8, n), 0.0)
        prior_cost = np.round(rng.lognormal(7.8, 1.1, n) * (1 + prior_ip * 2 + n_chronic * 0.3), 2)
        logit = (-3.6 - 0.25 * k                                        # drift: lower baseline in 2024
                 + 0.02 * (age - 55) + 0.8 * chf + 0.6 * copd + 0.3 * diab
                 + 0.6 * np.minimum(prior_ip, 3)
                 + 1.1 * (prior_ed >= 3)                                 # threshold effect
                 + 1.0 * (chf & lives_alone)                             # interaction
                 - 1.8 * (pdc - 0.75) + 0.06 * adi - 0.4 * cg + 0.015 * hcbs)
        p = 1 / (1 + np.exp(-logit))
        frames.append(pd.DataFrame({
            "member_id": [f"M{i:06d}" for i in range(n)], "cohort_year": year,
            "age": age.round(0).astype(int), "female_flag": (rng.random(n) < 0.55).astype(int),
            "dual_flag": (rng.random(n) < 0.3).astype(int), "n_chronic": n_chronic,
            "chf_flag": chf.astype(int), "copd_flag": copd.astype(int), "diabetes_flag": diab.astype(int),
            "prior_ip_cnt": prior_ip, "prior_ed_cnt": prior_ed, "prior_cost_amt": prior_cost,
            "pdc_pct": pdc.round(3), "hcbs_hours_per_week": hcbs.round(1), "lives_alone_flag": lives_alone.astype(int),
            "adi_decile": adi, "cg_program_flag": cg.astype(int), "telehealth_cnt": tele,
            "true_prob": p, "label_ip_6m": (rng.random(n) < p).astype(int)}))
    return pd.concat(frames, ignore_index=True)
