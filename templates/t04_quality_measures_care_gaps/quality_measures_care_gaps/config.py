"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 11
    n_members: int = 8_000
    measurement_year: int = 2025         # MY: Jan 1 - Dec 31
    data_start: str = "2023-01-01"       # history needed for lookbacks (BCS reaches back 27 months)
    data_cut_dt: str = "2026-03-31"      # last date claims were loaded; >= 90 days after MY end
    min_runout_days: int = 90
    measures_dir: Path = HERE / "measures"
    value_set_path: Path = HERE / "reference" / "FAKE_value_sets.csv"
    z: float = 1.96                      # 95% interval
    min_denominator: int = 30            # Core Set / NCQA-style small-denominator rule
    hybrid_sample_size: int = 411        # NCQA hybrid systematic sample (before oversample)
    core_set_year: int = 2026            # Core Set FFY 2026 reports services in CY 2025
