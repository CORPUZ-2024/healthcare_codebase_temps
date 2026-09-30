import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roi_cost_offset import data, methods  # noqa: E402
from roi_cost_offset.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def synth(cfg):
    return data.generate_panel(cfg.n_members, cfg.months, cfg.start, cfg.true_effect_pct, cfg.seed)


@pytest.fixture(scope="session")
def mp(synth, cfg):
    return methods.member_periods(synth["panel"], synth["members"], cfg.pre_months, cfg.post_months,
                                  cfg.min_months_each_period)


@pytest.fixture(scope="session")
def matched(mp, cfg):
    return methods.match_propensity(mp, methods.propensity_scores(mp), cfg.caliper_sd, cfg.exact_on, cfg.seed)


@pytest.fixture(scope="session")
def truth(synth, matched, cfg):
    """True savings (negative DiD) for the matched participants - known because the data are synthetic."""
    p = synth["panel"].merge(synth["members"][["member_id", "index_month_idx"]], on="member_id")
    p = p[p["member_id"].isin(matched.loc[matched["treated_flag"] == 1, "member_id"])]
    p = p[(p["month_idx"] - p["index_month_idx"]).between(1, cfg.post_months)]
    return -float((p["paid_cf_amt"] - p["paid_amt"]).sum() / p["member_months"].sum())
