"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 9
    alpha: float = 0.05
    power: float = 0.80
    # Primary endpoint (binary): 30-day readmission after discharge, usual care vs. transitional care
    p_control: float = 0.18
    p_treat: float = 0.14
    # Secondary endpoint (continuous): caregiver burden score (0-88 scale, FAKE instrument SD)
    burden_delta: float = 4.0
    burden_sd: float = 14.0
    # Cluster design: randomize primary-care practices
    cluster_size: int = 40
    icc: float = 0.02
    # Cost endpoint (skewed): 6-month PMPM
    cost_pmpm: float = 1_200.0
    cost_cv: float = 2.5
    cost_effect_pct: float = 0.10
    n_sims: int = 1_000
    sap_template: Path = HERE / "sap" / "sap_template.md"
