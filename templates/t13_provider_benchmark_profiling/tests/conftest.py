import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provider_benchmark_profiling import data, methods  # noqa: E402
from provider_benchmark_profiling.config import Config  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def sim(cfg):
    return data.generate(cfg.n_providers, cfg.provider_sd, cfg.seed)


@pytest.fixture(scope="session")
def profile(sim, cfg):
    p, info = methods.risk_model(sim["patients"])
    oe = methods.funnel_flags(methods.oe_table(sim["patients"], p), cfg.funnel_levels)
    return {"oe": oe, "info": info, "eb": methods.eb_shrink_oe(oe).set_index("provider_id"),
            "mixed": methods.mixed_logistic_ratios(sim["patients"]).set_index("provider_id"),
            "truth": sim["providers"].set_index("provider_id")}
