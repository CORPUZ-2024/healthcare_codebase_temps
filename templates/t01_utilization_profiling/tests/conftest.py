import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilization_profiling import data, methods, prep  # noqa: E402


@pytest.fixture(scope="session")
def prepared():
    u = data.generate_universe(n_members=300, seed=8, months=24)
    claims = prep.service_category_claim(prep.collapse_versions_latest(u["medical"]))
    mm = prep.member_months_daily(u["enrollment"], "2023-01-01", "2024-12-31")
    return u, claims, mm


def _ip_claims(rows):
    d = pd.to_datetime
    return pd.DataFrame([{"claim_id": c, "member_id": m, "service_category": "IP", "admit_dt": d(a),
                          "discharge_dt": d(b), "paid_amt": 1000.0} for c, m, a, b in rows])


@pytest.fixture
def ip_claims():
    """Factory: ip_claims([(claim_id, member_id, admit, discharge), ...]) -> IP claim rows."""
    return _ip_claims
