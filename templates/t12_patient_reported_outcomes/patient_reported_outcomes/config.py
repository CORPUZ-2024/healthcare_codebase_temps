"""Every tunable parameter for this template, in one place."""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 12
    n_caregivers: int = 600
    visits: tuple = (0, 3, 6)             # months
    true_effect_6m: float = -4.0          # latent burden points, program vs usual care, change at 6 months
    item_missing_rate: float = 0.03
    instruments_dir: Path = HERE / "instruments"
    primary_instrument: str = "FAKE_CAREGIVER_BURDEN"
    secondary_instrument: str = "PHQ9"
