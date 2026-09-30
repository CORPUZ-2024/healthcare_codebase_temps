import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_measures_care_gaps import data, methods  # noqa: E402
from quality_measures_care_gaps.config import Config  # noqa: E402

MY = 2025


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def vs(cfg):
    return data.load_value_sets(cfg.value_set_path)


@pytest.fixture(scope="session")
def specs(cfg):
    return data.load_measures(cfg.measures_dir)


@pytest.fixture(scope="session")
def synth():
    return data.generate(n_members=3_000, my=MY, seed=5)


@pytest.fixture(scope="session")
def evaluated(synth, specs, vs):
    return {mid: methods.evaluate_measure(s, synth["members"], synth["enrollment"], synth["events"], vs, MY)
            for mid, s in specs.items()}
