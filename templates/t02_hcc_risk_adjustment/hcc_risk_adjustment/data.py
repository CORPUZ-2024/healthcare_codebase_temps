"""
Synthetic members, diagnoses (with source claim type), pharmacy fills and next-year cost,
plus loaders for the FAKE reference tables in ../reference/.

The cost the members incur in the payment year is generated FROM their true conditions, so
you can check how well a risk score predicts it (predictive ratios) and compare the standard
published-weights score with weights re-estimated on your own data.

Public test data
----------------
* CMS DE-SynPUF (Beneficiary Summary + claims) — real CMS layout, synthetic values. Diagnoses are
  ICD-9 (2008-2010), so they will NOT map with ICD-10 tables; useful for pipeline/volume testing.
  https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* Real mappings/coefficients (not test data, reference data): CMS Risk Adjustment page —
  https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(__file__).resolve().parents[1] / "reference"

# condition -> (prevalence, ICD-10 codes that can document it, "true" cost multiplier)
CONDITIONS = {   # multipliers are close to the FAKE V28 weights, so the published model is "roughly right"
    "DIAB":  (0.22, ["E119", "E1165", "E1122"], 0.18),
    "CHF":   (0.08, ["I5020", "I509"], 0.38),
    "COPD":  (0.09, ["J449"], 0.32),
    "ASTHMA": (0.08, ["J45909"], 0.15),   # real cost, but the V28-like model doesn't pay it
    "CKD":   (0.10, ["N1830", "N184", "N185"], 0.30),
    "MDD":   (0.14, ["F329", "F339"], 0.30),
    "SCHIZ": (0.02, ["F200"], 0.50),
    "CP":    (0.03, ["G809"], 0.40),
    "QUAD":  (0.005, ["G8250"], 1.10),
    "OBES":  (0.07, ["E6601"], 0.20),
    "CANCER": (0.02, ["C3490", "C7951"], 2.00),
}
RX_FOR = {"DIAB": "INSULIN_OR_METFORMIN", "CHF": "LOOP_DIURETIC", "COPD": "LAMA_LABA", "SCHIZ": "ANTIPSYCHOTIC"}
SOURCES = ["PROF_F2F", "PROF_F2F", "PROF_F2F", "OP", "IP", "LAB", "DME"]   # LAB/DME are not acceptable


def generate(n_members: int = 3_000, service_year: int = 2025, seed: int = 42,
             documentation_rate: float = 0.8) -> dict[str, pd.DataFrame]:
    """Return {'members', 'dx', 'rx', 'cost'}.

    members : member_id, age, sex_cd, dual_flag, disabled_flag
    dx      : member_id, dx_cd, svc_dt (in service_year), source_cd   (one row per coded dx)
    rx      : member_id, rx_class, fill_dt                               (drug evidence)
    cost    : member_id, payment_year, paid_amt                         (next-year cost)

    ``documentation_rate`` is the chance a true condition is coded on an acceptable claim —
    the rest appear only on LAB/DME claims or not at all (these are the "coding gaps").
    """
    rng = np.random.default_rng(seed)
    ids = [f"M{i:06d}" for i in range(1, n_members + 1)]
    age = np.clip(rng.normal(58, 20, n_members), 1, 99).astype(int)
    members = pd.DataFrame({"member_id": ids, "age": age, "sex_cd": rng.choice(["F", "M"], n_members),
                            "dual_flag": (rng.random(n_members) < 0.35).astype(int),
                            "disabled_flag": ((age < 65) & (rng.random(n_members) < 0.4)).astype(int)})
    dx_rows, rx_rows, true_mult = [], [], np.full(n_members, 0.30)
    for i, mid in enumerate(ids):
        age_boost = 1 + max(0, age[i] - 50) / 60
        for cond, (prev, codes, mult) in CONDITIONS.items():
            if rng.random() >= prev * age_boost:
                continue
            true_mult[i] += mult
            code = codes[min(int(rng.integers(0, len(codes))), len(codes) - 1)]
            if rng.random() < documentation_rate:
                src = rng.choice(SOURCES[:5])
            else:
                src = rng.choice(["LAB", "DME", None], p=[0.4, 0.2, 0.4])   # None = never coded
            for _ in range(int(rng.integers(1, 4))) if src else []:
                d = pd.Timestamp(f"{service_year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
                dx_rows.append((mid, code, d, src))
            if cond in RX_FOR and rng.random() < 0.7:
                rx_rows.append((mid, RX_FOR[cond], pd.Timestamp(f"{service_year}-03-01")))
        true_mult[i] += 0.19 * members.loc[i, "dual_flag"] + 0.004 * max(0, age[i] - 35)
        true_mult[i] += 0.25 * (rng.random() < 0.1)   # unmeasured frailty (e.g. needs daily personal care)
    dx = pd.DataFrame(dx_rows, columns=["member_id", "dx_cd", "svc_dt", "source_cd"])
    rx = pd.DataFrame(rx_rows, columns=["member_id", "rx_class", "fill_dt"])
    paid = 11_000 * true_mult * rng.lognormal(-0.25, 0.7, n_members)        # heavy right skew
    paid[rng.random(n_members) < 0.08] = 0.0                                # some members use nothing
    cost = pd.DataFrame({"member_id": ids, "payment_year": service_year + 1, "paid_amt": paid.round(2)})
    return {"members": members, "dx": dx, "rx": rx, "cost": cost}


def load_reference(ref_dir: str | Path = REF_DIR, prefix: str = "FAKE_") -> dict[str, pd.DataFrame]:
    """Load dx->HCC map, hierarchies, HCC coefficients and demographic factors.

    Codes are read as strings. Swap in real CMS tables by saving them with the same columns and
    calling ``load_reference(ref_dir, prefix="")``.
    """
    ref_dir = Path(ref_dir)
    rd = lambda name: pd.read_csv(ref_dir / f"{prefix}{name}.csv", dtype=str)  # noqa: E731
    coef = rd("hcc_coefficients")
    coef["coefficient"] = coef["coefficient"].astype(float)
    demo = rd("demographic_factors")
    demo["coefficient"] = demo["coefficient"].astype(float)
    return {"dx_map": rd("dx_to_hcc"), "hierarchy": rd("hierarchies"), "coef": coef, "demo": demo}
