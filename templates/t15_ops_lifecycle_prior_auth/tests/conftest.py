import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops_lifecycle_prior_auth import data  # noqa: E402
from ops_lifecycle_prior_auth.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def sim(cfg):
    return data.generate(cfg.n_referrals, cfg.start, cfg.months, cfg.as_of, cfg.seed)
