"""
Synthetic patient-level outcomes across providers with KNOWN provider quality, and loaders for public
benchmark files (CMS Hospital Readmissions Reduction Program; Medicare Physician & Other Practitioners).

Generator: 80 hospitals (or home-health agencies) whose volumes range ~10x; each has a true quality
effect u_j (log-odds, SD 0.25) and its own case mix (some treat much sicker patients). Outcome: 30-day
readmission. What makes it realistic:
  * crude rates are confounded by case mix -> risk adjustment matters
  * small providers produce extreme observed rates by chance -> shrinkage matters
  * the truth per provider = expected events WITH its effect / expected events with u = 0, over its own
    patients (the quantity an O/E ratio is trying to estimate)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(n_providers: int = 80, provider_sd: float = 0.25, seed: int = 13) -> dict:
    """Returns {'patients': provider_id, age, comorbidity_cnt, prior_admit_flag, readmit_flag;
    'providers': provider_id, n_cases, case_mix_shift, true_effect, true_ratio}."""
    rng = np.random.default_rng(seed)
    vol = np.clip(np.round(np.exp(rng.normal(np.log(150), 0.9, n_providers))), 12, 1_500).astype(int)
    u = rng.normal(0, provider_sd, n_providers)
    mix = rng.normal(0, 1.0, n_providers)                          # provider case-mix shift (sicker populations)
    rows, truth = [], []
    for j in range(n_providers):
        n = vol[j]
        age = np.clip(rng.normal(74 + 3 * mix[j], 8, n), 40, 100)
        com = rng.poisson(np.clip(2.0 + 0.8 * mix[j], 0.2, None), n)
        prior = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 0.5 * mix[j])))).astype(int)
        lp0 = -2.2 + 0.02 * (age - 74) + 0.30 * com + 0.45 * prior
        p1, p0 = 1 / (1 + np.exp(-(lp0 + u[j]))), 1 / (1 + np.exp(-lp0))
        y = (rng.random(n) < p1).astype(int)
        pid = f"H{j + 1:03d}"
        rows.append(pd.DataFrame({"provider_id": pid, "age": age.round(1), "comorbidity_cnt": com, "prior_admit_flag": prior,
                                  "readmit_flag": y}))
        truth.append({"provider_id": pid, "n_cases": n, "case_mix_shift": mix[j], "true_effect": u[j],
                      "true_ratio": float(p1.sum() / p0.sum())})
    return {"patients": pd.concat(rows, ignore_index=True), "providers": pd.DataFrame(truth)}


# ---------------------------------------------------------------------------
# Public benchmark files
# ---------------------------------------------------------------------------
HRRP_COLS = {  # normalized header -> template column (verify against the current data dictionary)
    "facility name": "facility_name", "facility id": "facility_id", "state": "state", "measure name": "measure_name",
    "number of discharges": "n_discharges", "excess readmission ratio": "excess_readmission_ratio",
    "predicted readmission rate": "predicted_rate", "expected readmission rate": "expected_rate",
    "number of readmissions": "n_readmissions", "start date": "start_date", "end date": "end_date",
}


def load_hrrp(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CMS Hospital Readmissions Reduction Program file (provider-data.cms.gov).

    * Rates are percentages in the file -> divided by 100.
    * 'Too Few to Report' / 'N/A' become NaN (small hospitals are suppressed - never impute them as 0).
    * start_date / end_date = the PERFORMANCE period (usually 3 years ending ~2 years before the
      fiscal year) - compare with your own measurement period (checks.check_benchmark_vintage).
    """
    df = path.astype(str) if isinstance(path, pd.DataFrame) else pd.read_csv(path, dtype=str)
    ren = {c: HRRP_COLS[c.strip().lower()] for c in df.columns if c.strip().lower() in HRRP_COLS}
    df = df.rename(columns=ren)[list(ren.values())]
    for c in ("n_discharges", "excess_readmission_ratio", "predicted_rate", "expected_rate", "n_readmissions"):
        if c in df:
            df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    for c in ("predicted_rate", "expected_rate"):
        if c in df and df[c].max() > 1:
            df[c] = df[c] / 100
    for c in ("start_date", "end_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def generate_fake_hrrp(n: int = 2_500, start: str = "2021-07-01", end: str = "2024-06-30", seed: int = 13) -> pd.DataFrame:
    """FAKE national file with the public HRRP headers (demo/tests only): excess readmission ratios
    around 1.0, some suppressed small hospitals ('Too Few to Report')."""
    rng = np.random.default_rng(seed + 5)
    err = rng.normal(1.0, 0.08, n)
    disch = np.round(np.exp(rng.normal(np.log(300), 0.9, n))).astype(int)
    few = disch < 25
    return pd.DataFrame({"Facility Name": [f"FAKE HOSPITAL {i}" for i in range(n)], "Facility ID": [f"{i:06d}" for i in range(n)],
                         "State": "XX", "Measure Name": "READM-30-HF-HRRP",
                         "Number of Discharges": np.where(few, "Too Few to Report", disch.astype(str)),
                         "Excess Readmission Ratio": np.where(few, "N/A", np.round(err, 4).astype(str)),
                         "Predicted Readmission Rate": np.round(20 * err, 4).astype(str), "Expected Readmission Rate": "20.0",
                         "Number of Readmissions": "N/A", "Start Date": start, "End Date": end})
