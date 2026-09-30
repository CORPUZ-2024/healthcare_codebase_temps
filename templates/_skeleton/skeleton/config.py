"""All tunable parameters live here, so run.py and tests read one place."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42          # random seed for synthetic data (reproducibility)
    n_members: int = 1_000  # synthetic population size for the demo run
