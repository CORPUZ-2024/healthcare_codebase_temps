import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictive_risk_stratification import data, methods  # noqa: E402


@pytest.fixture(scope="session")
def split():
    df = data.generate(n_per_cohort=4_000, seed=7)
    return methods.temporal_split(df, 2023, 2024)
