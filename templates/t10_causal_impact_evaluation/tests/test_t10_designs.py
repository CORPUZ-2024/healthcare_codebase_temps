"""Each design recovers its known effect; the failure modes it is meant to catch are demonstrated."""
import numpy as np
import pandas as pd
import pytest

from causal_impact_evaluation import data, methods


# --- A. DiD and event study ------------------------------------------------------------------------

def test_did_2x2_known_answer():
    panel = pd.DataFrame({"practice_id": list("AABBCCDD"), "month_idx": [0, 1] * 4, "treated_flag": [1] * 4 + [0] * 4,
                          "post_flag": [0, 1] * 4, "y": [10.0, 13.0, 12.0, 15.0, 10.0, 11.0, 14.0, 15.0]})
    assert methods.did_twfe(panel)["effect"] == pytest.approx(2.0)


def test_did_covers_truth_across_draws():
    hits = 0
    for s in range(30):
        pn = data.practice_panel(seed=100 + s)
        r = methods.did_twfe(pn["df"])
        hits += r["ci_lo"] <= pn["effect"] <= r["ci_hi"]
    assert hits >= 26                                                   # ~95% nominal


def test_naive_post_comparison_is_biased(cfg):
    pn = data.practice_panel(seed=cfg.seed)
    assert methods.naive_post_comparison(pn["df"]) > 0 > pn["effect"]    # treated practices start higher


def test_event_study_binned_false_alarm_rate_and_power():
    false_alarm = sum(methods.event_study(data.practice_panel(seed=200 + s)["df"])[1] < 0.05 for s in range(40))
    power = sum(methods.event_study(data.practice_panel(pre_trend=0.08, seed=200 + s)["df"])[1] < 0.05 for s in range(30))
    assert false_alarm <= 6 and power >= 24


def test_event_study_unbinned_overrejects():
    """One coefficient per month (23 leads, 40 clusters): the joint test rejects true parallel trends."""
    rej = sum(methods.event_study(data.practice_panel(seed=300 + s)["df"], window=(-24, 11))[1] < 0.05 for s in range(12))
    assert rej >= 6


def test_event_study_shape_and_reference(cfg):
    es, _ = methods.event_study(data.practice_panel(seed=cfg.seed)["df"], window=(-6, 6))
    assert es["rel_month"].tolist() == list(range(-6, 7))
    assert es.loc[es["rel_month"] == -1, "coef"].iloc[0] == 0.0
    assert es.loc[es["rel_month"] >= 0, "coef"].mean() == pytest.approx(-2.0, abs=0.5)


def test_pre_trend_violation_biases_did(cfg):
    ok = methods.did_twfe(data.practice_panel(seed=cfg.seed)["df"])["effect"]
    bad = methods.did_twfe(data.practice_panel(pre_trend=0.08, seed=cfg.seed)["df"])["effect"]
    assert abs(bad - (-2.0)) > 3 * abs(ok - (-2.0))


# --- B. PSM, IPW, AIPW -----------------------------------------------------------------------------

def test_att_differs_from_ate(cs):
    assert cs["att"] < cs["ate"] - 0.5                                   # sicker members referred AND benefit more


def test_naive_difference_has_the_wrong_sign(cs):
    assert methods.naive_difference(cs["df"]) > 0 > cs["ate"]


def test_psm_recovers_att_of_matched_and_balances(cs, psm):
    r, matched = psm
    truth = matched.loc[matched["treated_flag"] == 1, "tau"].mean()
    assert r["ci_lo"] <= truth <= r["ci_hi"]
    smd = methods.smd_table(cs["df"], methods.COVARIATES, matched=matched)
    assert (smd["smd_raw"].abs() > 0.2).sum() >= 3 and (smd["smd_matched"].abs() < 0.1).all()
    assert matched.groupby("pair_id")["treated_flag"].sum().eq(1).all()


def test_ipw_and_aipw_cover_the_ate(cs):
    i, w = methods.ipw(cs["df"])
    a = methods.aipw_ate(cs["df"])
    assert i["ci_lo"] <= cs["ate"] <= i["ci_hi"] and a["ci_lo"] <= cs["ate"] <= a["ci_hi"]
    assert a["se"] < i["se"]                                             # AIPW is more efficient
    smd = methods.smd_table(cs["df"], methods.COVARIATES, weights=w)
    assert (smd["smd_weighted"].abs() < 0.1).all()


def test_aipw_double_robustness(cs):
    """Wrong PS model (age only) + correct outcome model (with prior_util^2) -> AIPW still right;
    IPW with the same wrong PS is badly biased."""
    df = cs["df"].assign(prior_sq=(cs["df"]["prior_util"] - 2) ** 2)
    good_outcome = methods.COVARIATES + ["prior_sq"]
    a = methods.aipw_ate(df, ps_covariates=["age"], outcome_covariates=good_outcome)
    i, _ = methods.ipw(df, covariates=["age"])
    assert abs(a["effect"] - cs["ate"]) < 0.4
    assert abs(i["effect"] - cs["ate"]) > 3 * abs(a["effect"] - cs["ate"])


def test_aipw_coverage_across_draws():
    hits = 0
    for s in range(25):
        c = data.cross_section(3_000, 400 + s)
        r = methods.aipw_ate(c["df"])
        hits += r["ci_lo"] <= c["ate"] <= r["ci_hi"]
    assert hits >= 20


# --- C. ITS ----------------------------------------------------------------------------------------

def test_its_recovers_effects_on_average():
    lv, sl = [], []
    for s in range(40):
        r = methods.its_segmented(data.its_series(seed=600 + s)["df"])
        lv.append(r["level_change"]); sl.append(r["slope_change"])
    assert np.mean(lv) == pytest.approx(-4.0, abs=0.4) and np.mean(sl) == pytest.approx(-0.15, abs=0.03)


def test_its_glsar_covers_better_than_hac():
    """The documented reason for the GLSAR default: on 60 autocorrelated months HAC under-covers."""
    cov = {"glsar": 0, "hac": 0}
    for s in range(80):
        d = data.its_series(seed=700 + s)["df"]
        for m in cov:
            r = methods.its_segmented(d, se_method=m)
            cov[m] += r["level_lo"] <= -4.0 <= r["level_hi"]
    assert cov["glsar"] > cov["hac"] and cov["glsar"] >= 0.8 * 80


def test_its_counterfactual_and_dw(cfg):
    r = methods.its_segmented(data.its_series(seed=cfg.seed)["df"])
    cf = r["counterfactual"]
    pre = cf.index < cfg.its_break
    assert np.allclose(cf.loc[pre, "fitted"], cf.loc[pre, "no_policy"])  # identical before the policy
    assert r["durbin_watson"] < 1.5 and 0.2 < r["rho"] < 0.9              # AR(1) = 0.5 detected
