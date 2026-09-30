import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_impact_evaluation import data, methods  # noqa: E402
from causal_impact_evaluation.config import Config  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def cs(cfg):
    return data.cross_section(cfg.n_members, cfg.seed)


@pytest.fixture(scope="session")
def psm(cs, cfg):
    df = cs["df"]
    return methods.psm_att(df, methods.propensity_logit(df), cfg.caliper_sd, seed=cfg.seed)
