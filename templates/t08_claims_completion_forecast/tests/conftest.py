import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims_completion_forecast import data, methods  # noqa: E402
from claims_completion_forecast.config import Config  # noqa: E402

warnings.filterwarnings("ignore", module="statsmodels")


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def synth(cfg):
    return data.generate(cfg.start, cfg.months, cfg.members, cfg.base_pmpm, cfg.annual_trend, cfg.max_lag, seed=cfg.seed)


@pytest.fixture(scope="session")
def completed(synth, cfg):
    r = methods.complete(synth["claims"], synth["exposure"], synth["as_of"], cfg.cl_avg_months, cfg.tail_factor,
                         cfg.bf_max_age, cfg.max_lag)
    return r.assign(true_ult=synth["truth"].set_index("incurred_month_start")["ultimate_paid_amt"],
                    true_pmpm=synth["truth"].set_index("incurred_month_start")["true_pmpm"])
