import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcc_risk_adjustment import data  # noqa: E402


@pytest.fixture(scope="session")
def ref():
    return data.load_reference()


@pytest.fixture(scope="session")
def synth():
    return data.generate(n_members=3_000, seed=4)


@pytest.fixture
def one_member():
    members = pd.DataFrame({"member_id": ["A"], "age": [70], "sex_cd": ["F"], "dual_flag": [1], "disabled_flag": [0]})
    dx = pd.DataFrame({"member_id": ["A"] * 5, "dx_cd": ["E11.22", "e119", "N185", "N1830", "I5020"],
                       "svc_dt": pd.to_datetime(["2025-02-01"] * 5),
                       "source_cd": ["PROF_F2F", "OP", "IP", "LAB", "IP"]})
    return members, dx
