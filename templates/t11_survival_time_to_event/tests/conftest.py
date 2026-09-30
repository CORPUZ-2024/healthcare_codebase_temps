import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survival_time_to_event import data, methods  # noqa: E402
from survival_time_to_event.config import Config  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def cohort(cfg):
    return data.generate(cfg.n_members, cfg.max_follow_days, cfg.true_hr_program, cfg.acuity_hr_early,
                         cfg.acuity_hr_late, cfg.acuity_change_day, cfg.seed)


@pytest.fixture(scope="session")
def cox(cohort):
    return methods.cox_ph(cohort)
