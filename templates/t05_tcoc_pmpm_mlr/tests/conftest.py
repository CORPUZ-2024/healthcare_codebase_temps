import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcoc_pmpm_mlr import data, methods, prep  # noqa: E402


@pytest.fixture(scope="session")
def prepared():
    u = data.generate_universe(n_members=250, seed=5, months=12)
    claims = methods.attach_month(prep.service_category_claim(prep.collapse_versions_latest(u["medical"])))
    mm = prep.member_months_daily(u["enrollment"], "2023-01-01", "2023-12-31", by=("member_id", "lob_cd"))
    return claims, mm


@pytest.fixture
def tiny():
    mm = pd.DataFrame({"member_id": ["A", "B", "C"], "month": pd.PeriodIndex(["2025-01"] * 3, freq="M"),
                       "member_months": [1.0, 0.5, 0.1], "lob_cd": ["MCD", "MCD", "DUAL"]})
    c = pd.DataFrame({"member_id": ["A", "B", "C"], "month": pd.PeriodIndex(["2025-01"] * 3, freq="M"),
                      "paid_amt": [300.0, 150.0, 400.0], "service_category": ["PROF", "IP", "ED"]})
    return c, mm
