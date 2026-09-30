import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_design_power.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture
def units():
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame({"member_id": [f"U{i:04d}" for i in range(n)], "site_cd": rng.choice(["A", "B", "C"], n),
                         "risk_tier": rng.choice(["HIGH", "LOW"], n)})
