import numpy as np
import pandas as pd
import pytest

from predictive_risk_stratification import checks, data, methods

F = data.FEATURES


def test_epv_and_separation():
    assert methods.events_per_variable(pd.Series([1] * 30 + [0] * 70), 3) == 10
    assert methods.events_per_variable(pd.Series([1] * 90 + [0] * 10), 2) == 5       # rarer class counts
    X = pd.DataFrame({"x": [0, 0, 1, 1, 0, 1], "z": [0, 1, 0, 1, 1, 0]})
    assert methods.detect_separation(X, pd.Series([0, 1, 1, 1, 0, 1])) == ["x"]      # x=1 -> always 1


def test_temporal_split_has_no_overlap(split):
    tr, te = split
    assert set(tr["cohort_year"]) == {2023} and set(te["cohort_year"]) == {2024}


def test_logistic_discriminates_and_is_calibrated(split):
    tr, te = split
    m = methods.fit_logistic(tr[F], tr["label_ip_6m"])
    ev = methods.evaluate(te["label_ip_6m"], methods.predict_proba_any(m, te[F]))
    assert ev["auc"] > 0.70
    assert 0.8 < ev["slope"] < 1.25


def test_boosting_runs_and_is_calibrated_by_isotonic(split):
    tr, te = split
    gb = methods.fit_gradient_boosting(tr[F], tr["label_ip_6m"])
    p = methods.predict_proba_any(gb, te[F])
    assert ((p >= 0) & (p <= 1)).all()
    assert methods.evaluate(te["label_ip_6m"], p)["auc"] > 0.65


def test_odds_ratios_recover_direction(split):
    tr, _ = split
    m = methods.fit_logistic(tr[F], tr["label_ip_6m"])
    orr = methods.odds_ratios(m, F, tr[F]).set_index("feature")["odds_ratio_per_unit"]
    assert orr["chf_flag"] > 1.5 and orr["prior_ip_cnt"] > 1.2 and orr["pdc_pct"] < 1


def test_evaluate_known_answer():
    y = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 10)
    p = np.where(y == 1, 0.9, 0.1)
    ev = methods.evaluate(y, p, top_pct=0.2)
    assert ev["auc"] == 1.0 and ev["ppv_top20"] == 1.0 and ev["sensitivity_top20"] == 1.0


def test_calibration_table_and_recalibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.4, 20_000)
    y = pd.Series((rng.random(20_000) < p).astype(int))
    t = methods.calibration_table(y, p)
    assert t["obs_to_pred"].between(0.8, 1.2).all()
    assert methods.recalibrate_intercept(p, 0.10).mean() == pytest.approx(0.10, abs=1e-6)


def test_capacity_tiers():
    t = methods.tiers_by_capacity(pd.Series([f"m{i}" for i in range(100)]), np.linspace(0, 1, 100))
    assert t["tier"].value_counts().to_dict() == {3: 80, 2: 15, 1: 5}
    assert t.loc[t["tier"] == 1, "risk"].min() > t.loc[t["tier"] == 2, "risk"].max()


def test_kmeans_finds_planted_clusters():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(np.vstack([rng.normal(0, 0.3, (200, 2)), rng.normal(4, 0.3, (200, 2)),
                                rng.normal([0, 4], 0.3, (200, 2))]), columns=["a", "b"])
    assert methods.segments_kmeans(X)["k"] == 3


def test_rising_risk():
    prev = pd.DataFrame({"member_id": ["a", "b", "c"], "risk": [0.10, 0.10, 0.50]})
    curr = pd.DataFrame({"member_id": ["a", "b", "c"], "risk": [0.20, 0.11, 0.70], "tier": [2, 3, 1]})
    assert methods.rising_risk(prev, curr, 0.05)["member_id"].tolist() == ["a"]


def test_checks():
    assert checks.check_epv(4.0)[0].check_id == "ANL-003"
    assert checks.check_leakage(["age", "next_6m_cost"])[0].check_id == "ANL-005"
    assert checks.check_calibration_drift({"slope": 1.0, "intercept": -0.5})
    assert not checks.check_leakage(F)
