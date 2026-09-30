"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 7
    contracts_dir: Path = HERE / "contracts"
    # ACO performance year (synthetic, Medicare-FFS-like annual costs)
    n_beneficiaries: int = 12_000
    benchmark_pmpy: float = 12_000.0
    true_savings_pct: float = 0.035
    quality_score: float = 0.88
    # scenario grid
    grid_savings_pct: tuple = (-0.06, -0.03, -0.02, 0.0, 0.02, 0.03, 0.05, 0.08)
    grid_quality: tuple = (0.35, 0.70, 0.90)
    # Monte Carlo
    n_sims: int = 2_000
    trend_sd: float = 0.015          # benchmark trend miss (SD of log ratio, year level)
    # Medicaid sub-cap and care-management demos
    n_subcap_members: int = 1_500
    n_cm_participants: int = 400
    population_pmpm: float = 540.0     # whole-plan PMPM (t05-like), for the target-basis check
    cm_quality_met: dict = field(default_factory=lambda: {"follow_up_7d": True, "caregiver_assessment": True,
                                                          "med_reconciliation": False})
