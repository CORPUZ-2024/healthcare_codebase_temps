"""Scoring rules, reliability, responders, and which longitudinal analysis survives informative dropout."""
import numpy as np
import pandas as pd
import pytest

from patient_reported_outcomes import checks, data, methods


def _score(long, spec):
    return pd.concat([long[["caregiver_id", "arm_flag", "visit_month", "observed_flag"]], methods.score_instrument(long, spec)], axis=1)


# --- scoring -----------------------------------------------------------------------------------------

def test_phq9_scoring_and_prorating(ins):
    phq = ins["PHQ9"]
    x = pd.DataFrame([{i: v for i, v in zip(phq["items"], [3, 3, 3, 2, 2, 2, 1, 1, None])}])
    r = methods.score_instrument(x, phq).iloc[0]
    assert r["score"] == pytest.approx(17 / 8 * 9) and r["prorated_flag"] == 1 and r["n_items_missing"] == 1


def test_reverse_items_are_flipped(ins):
    cb = ins["FAKE_CAREGIVER_BURDEN"]
    all_max = pd.DataFrame([{i: 4 for i in cb["items"]}])
    assert methods.score_instrument(all_max, cb)["score"].iloc[0] == 4 * 9      # 3 reverse items flip to 0


def test_never_fill_skipped_items_with_zero(ins):
    phq = ins["PHQ9"]
    x = pd.DataFrame([{i: 2 for i in phq["items"]}])
    x.loc[0, "phq1"] = None
    assert methods.score_instrument(x, phq)["score"].iloc[0] == 18 > x.fillna(0)[phq["items"]].sum(axis=1).iloc[0]


def test_bands(ins):
    b = methods.severity_band(pd.Series([0, 4.9, 5, 14.99, 15, 20, np.nan]), ins["PHQ9"]).astype(str).tolist()
    assert b == ["minimal", "minimal", "mild", "moderate", "moderately severe", "severe", "nan"]


def test_reliability_requires_reverse_scoring(trial, ins):
    cb = ins["FAKE_CAREGIVER_BURDEN"]
    base = trial["long"][trial["long"]["visit_month"] == 0]
    good = methods.cronbach_alpha(methods.reversed_items(base, cb))
    raw = methods.cronbach_alpha(base[cb["items"]])
    assert good > 0.8 and raw < 0.6
    assert checks.check_reliability(raw) and not checks.check_reliability(good)


def test_scores_track_the_latent_truth(trial, ins):
    """Scored complete data vs. observed data: prorating keeps the visit means unbiased."""
    cb = ins["FAKE_CAREGIVER_BURDEN"]
    comp = _score(trial["complete"].assign(observed_flag=1), cb)
    item_only = trial["complete"].copy()
    rng = np.random.default_rng(0)
    for c in cb["items"]:
        item_only.loc[rng.random(len(item_only)) < 0.03, c] = np.nan
    pro = _score(item_only.assign(observed_flag=1), cb)
    assert pro["score"].mean() == pytest.approx(comp["score"].mean(), abs=0.2)


# --- responders ------------------------------------------------------------------------------------

def test_responder_analysis_known_answer():
    w = pd.DataFrame({"arm_flag": [0, 0, 1, 1, 1], "change_6": [-5.0, 0.0, -4.0, -8.0, np.nan]})
    obs = methods.responder_analysis(w, mcid=4)
    assert obs.set_index("arm_flag")["responder_rate"].to_dict() == {0: 0.5, 1: 1.0}
    cons = methods.responder_analysis(w, mcid=4, missing_as_nonresponder=True)
    assert cons.set_index("arm_flag").loc[1, "responder_rate"] == pytest.approx(2 / 3)


# --- longitudinal effects -------------------------------------------------------------------------------

def test_between_arm_effect_all_methods_cover_truth(sb, trial):
    w = methods.to_wide(sb)
    for r in (methods.mixed_model_effect(sb), methods.ancova_change(w), methods.gee_effect(sb)):
        assert r["ci_lo"] <= trial["true_effect_6m"] <= r["ci_hi"]


def test_mixed_model_uses_partial_completers(sb):
    mm = methods.mixed_model_effect(sb)
    an = methods.ancova_change(methods.to_wide(sb))
    assert mm["n_people"] > an["n_people"]


def test_completers_overstate_within_arm_improvement(ins):
    """Across trials: caregivers getting worse drop out -> completers' mean change looks better than the truth;
    the mixed model (MAR) is much closer. This is the single-arm 'we improved burden by X points' trap."""
    cb, phq = ins["FAKE_CAREGIVER_BURDEN"], ins["PHQ9"]
    comp_bias, mm_bias = [], []
    for s in range(10):
        t = data.generate_trial(cb, phq, seed=200 + s)
        sc = _score(t["long"], cb)
        wc = methods.to_wide(_score(t["complete"].assign(observed_flag=1), cb))
        truth = wc.loc[wc["arm_flag"] == 1, "change_6"].mean()
        am = methods.arm_mean_change(sc, arm=1)
        comp_bias.append(am["completers_mean_change"] - truth)
        mm_bias.append(am["mixed_model_mean_change"] - truth)
    assert np.mean(comp_bias) < -0.3                     # completers look too good
    assert abs(np.mean(mm_bias)) < abs(np.mean(comp_bias)) / 2


def test_dropouts_were_getting_worse(ins):
    """Guard on the generator: dropout depends on WORSENING, so dropouts' true change is worse than
    completers'. (During the build, dropout on the LEVEL of burden left change scores almost unbiased.)"""
    t = data.generate_trial(ins["FAKE_CAREGIVER_BURDEN"], ins["PHQ9"], seed=200)
    wc = methods.to_wide(_score(t["complete"].assign(observed_flag=1), ins["FAKE_CAREGIVER_BURDEN"]))
    obs6 = t["long"][t["long"]["visit_month"] == 6].set_index("caregiver_id")["observed_flag"]
    wc["obs6"] = wc["caregiver_id"].map(obs6)
    by = wc.groupby("obs6")["change_6"].mean()
    assert by[0] > by[1] + 1.0


# --- checks ----------------------------------------------------------------------------------------------

def test_checks(ins, sb):
    assert not checks.check_license(ins["PHQ9"]) and not checks.check_license(ins["FAKE_CAREGIVER_BURDEN"])
    assert checks.check_license({"instrument_id": "ZBI", "license": "proprietary"})[0].check_id == "PRO-004"
    assert checks.check_differential_dropout(methods.to_wide(sb))[0].check_id == "PRO-002"
    assert checks.check_floor_ceiling({"floor_share": 0.3, "ceiling_share": 0.0})
    bad = pd.DataFrame({"score": [np.nan] * 3 + [1.0] * 7})
    assert checks.check_unscorable(bad, pd.Series([1] * 10))[0].check_id == "PRO-001"


def test_meps_loader(tmp_path):
    f = tmp_path / "meps.csv"
    pd.DataFrame({"DUPERSID": ["1001", "1002"], "PCS42": [45.2, -1], "MCS42": [50.1, 38.0], "PHQ242": [0, -9],
                  "K6SUM42": [3, 12], "OTHER": [1, 2]}).to_csv(f, index=False)
    m = data.load_meps_pro(f)
    assert list(m.columns) == ["person_id", "PCS42", "MCS42", "PHQ242", "K6SUM42"]
    assert np.isnan(m.loc[1, "PCS42"]) and np.isnan(m.loc[1, "PHQ242"]) and m.loc[0, "person_id"] == "1001"
