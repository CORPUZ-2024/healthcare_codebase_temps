import numpy as np
import pandas as pd

from causal_impact_evaluation import checks, methods


def test_balance_check():
    smd = pd.DataFrame({"covariate": ["a", "b"], "smd_matched": [0.02, 0.3]})
    assert checks.check_balance(smd, "smd_matched")[0].n_rows == 1
    assert not checks.check_balance(smd.assign(smd_matched=[0.02, 0.05]), "smd_matched")


def test_overlap_check(cs):
    ps = methods.propensity_logit(cs["df"])
    assert not checks.check_overlap(ps, cs["df"]["treated_flag"], 0.01, 0.99)
    extreme = pd.Series(np.r_[np.full(50, 6.0), np.zeros(50)])
    assert checks.check_overlap(extreme, pd.Series([1] * 100))[0].check_id == "CAU-002"


def test_pre_trend_and_cluster_checks():
    assert checks.check_pre_trends(0.01)[0].check_id == "CAU-003" and not checks.check_pre_trends(0.4)
    assert checks.check_few_clusters(12) and not checks.check_few_clusters(40)


def test_extreme_weight_check():
    assert checks.check_extreme_weights(pd.Series([1.0] * 99 + [50.0]))[0].check_id == "CAU-004"
    assert not checks.check_extreme_weights(pd.Series([1.0] * 1_000))


def test_its_checks():
    assert checks.check_its_autocorrelation(0.9, "hac") and not checks.check_its_autocorrelation(0.9, "glsar")
    assert checks.check_its_points(5, 20)[0].severity == "error" and not checks.check_its_points(24, 24)
