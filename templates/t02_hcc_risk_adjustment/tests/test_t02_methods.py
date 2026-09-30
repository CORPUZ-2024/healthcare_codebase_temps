import numpy as np
import pandas as pd
import pytest

from hcc_risk_adjustment import methods


def _score(members, dx, ref, v):
    acc = methods.filter_acceptable_sources(dx)
    hcc = methods.apply_hierarchies(methods.map_dx_to_hcc(acc, ref["dx_map"], v), ref["hierarchy"], v)
    X = methods.build_features(members, hcc, ref["coef"], v)
    return hcc, X, methods.score_published_weights(X, ref["coef"], ref["demo"], v)


def test_code_normalization_and_source_filter(one_member, ref):
    members, dx = one_member
    acc = methods.filter_acceptable_sources(dx)
    assert len(acc) == 4                                            # LAB row dropped
    hcc = methods.map_dx_to_hcc(acc, ref["dx_map"], "V24")
    assert set(hcc["hcc"]) == {"DIAB_CC", "DIAB_NC", "CKD5", "CHF"}  # 'E11.22' and 'e119' normalized


def test_hierarchy_and_interaction_known_answer(one_member, ref):
    members, dx = one_member
    hcc, X, s = _score(members, dx, ref, "V24")
    assert set(hcc["hcc"]) == {"DIAB_CC", "CKD5", "CHF"}
    assert X.loc["A", "DIAB_CHF"] == 1 and X.loc["A", "D4P"] == 0
    # F65_74 .310 + DUAL .180 + DIAB_CC .302 + CKD5 .815 + CHF .331 + DIAB_CHF .121
    assert s.iloc[0] == pytest.approx(0.310 + 0.180 + 0.302 + 0.815 + 0.331 + 0.121)


def test_version_differences(one_member, ref):
    members, dx = one_member
    dx2 = pd.concat([dx, dx.iloc[[0]].assign(dx_cd="J45909")])
    _, X24, _ = _score(members, dx2, ref, "V24")
    _, X28, _ = _score(members, dx2, ref, "V28")
    assert "ASTHMA" in X24.columns and "ASTHMA" not in X28.columns   # V28-like model doesn't pay it


def test_demographic_cells():
    assert [methods.demographic_variable(a, "F") for a in (5, 34, 35, 64, 65, 84, 85)] == \
           ["F0_34", "F0_34", "F35_64", "F35_64", "F65_74", "F75_84", "F85P"]


def test_blend_and_normalize():
    s = {"V24": pd.Series([1.0, 2.0]), "V28": pd.Series([2.0, 2.0])}
    assert methods.blend_versions(s, 2024).tolist() == pytest.approx([1.33, 2.0])
    assert methods.blend_versions(s, 2026).tolist() == [2.0, 2.0]
    assert methods.normalize_and_adjust(pd.Series([1.1]), 1.1, 0.059).iloc[0] == pytest.approx(0.941)


def test_empirical_weights_recover_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"A": rng.integers(0, 2, 4000), "B": rng.integers(0, 2, 4000)},
                     index=[f"m{i}" for i in range(4000)])
    cost = pd.Series(1_000 + 3_000 * X["A"] + 500 * X["B"] + rng.normal(0, 200, 4000), index=X.index)
    X["INTERCEPT"] = 1
    rel, w = methods.score_empirical_weights(X, cost, ridge=0.1)
    assert rel.mean() == pytest.approx(1.0)
    assert w["A"] / w["B"] == pytest.approx(6.0, rel=0.1)


def test_predictive_ratios_perfect_model():
    score = pd.Series(np.linspace(0.2, 3.0, 100), index=[f"m{i}" for i in range(100)])
    pr = methods.predictive_ratios(score, score * 10_000)
    assert pr["predictive_ratio"].tolist() == pytest.approx([1.0] * 10)
    assert pr.attrs["r2"] == pytest.approx(1.0)


def test_empirical_beats_published_out_of_sample(synth, ref):
    """On this synthetic population (with a condition the V28-like model doesn't pay), weights
    re-estimated on a training half predict the other half better."""
    acc = methods.filter_acceptable_sources(synth["dx"])
    hcc = methods.apply_hierarchies(methods.map_dx_to_hcc(acc, ref["dx_map"], "V28"), ref["hierarchy"], "V28")
    X = methods.build_features(synth["members"], hcc, ref["coef"], "V28")
    cost = synth["cost"].set_index("member_id")["paid_amt"]
    std = methods.score_published_weights(X, ref["coef"], ref["demo"], "V28")
    idx = X.index.to_numpy()
    tr = np.random.default_rng(1).random(len(idx)) < 0.5
    _, w = methods.score_empirical_weights(X.loc[idx[tr]], cost)
    emp = pd.Series(X.loc[idx[~tr]].astype(float).to_numpy() @ w.to_numpy(), index=idx[~tr])
    assert methods.predictive_ratios(emp, cost).attrs["r2"] > methods.predictive_ratios(std.loc[idx[~tr]], cost).attrs["r2"]


def test_suspect_gaps(ref):
    all_hcc = pd.DataFrame({"member_id": ["A", "B"], "hcc": ["CHF", "COPD"]})
    acc_hcc = pd.DataFrame({"member_id": ["B"], "hcc": ["COPD"]})
    rx = pd.DataFrame({"member_id": ["C", "B"], "rx_class": ["INSULIN_OR_METFORMIN", "LAMA_LABA"],
                       "fill_dt": pd.to_datetime(["2025-01-01"] * 2)})
    g = methods.suspect_gaps(all_hcc, acc_hcc, rx)
    assert set(map(tuple, g[["member_id", "suspected_hcc"]].to_numpy())) == {("A", "CHF"), ("C", "DIAB_CC")}


def test_explain_member_totals_to_score(one_member, ref):
    members, dx = one_member
    _, X, s = _score(members, dx, ref, "V28")
    e = methods.explain_member("A", X, ref["coef"], ref["demo"], "V28")
    assert e.iloc[-1]["coefficient"] == pytest.approx(s.iloc[0])
