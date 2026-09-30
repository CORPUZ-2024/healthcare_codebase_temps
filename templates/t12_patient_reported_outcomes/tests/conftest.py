import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patient_reported_outcomes import data, methods  # noqa: E402
from patient_reported_outcomes.config import Config  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def ins(cfg):
    return data.load_instruments(cfg.instruments_dir)


@pytest.fixture(scope="session")
def trial(ins, cfg):
    return data.generate_trial(ins["FAKE_CAREGIVER_BURDEN"], ins["PHQ9"], cfg.n_caregivers, cfg.visits, cfg.true_effect_6m,
                               cfg.item_missing_rate, cfg.seed)


def scored(long, spec):
    return pd.concat([long[["caregiver_id", "arm_flag", "visit_month", "observed_flag"]], methods.score_instrument(long, spec)], axis=1)


@pytest.fixture(scope="session")
def sb(trial, ins):
    return scored(trial["long"], ins["FAKE_CAREGIVER_BURDEN"])
