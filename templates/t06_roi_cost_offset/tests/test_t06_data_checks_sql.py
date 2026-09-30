import numpy as np
import pandas as pd
import pytest

from roi_cost_offset import checks, data, methods


# --- data ------------------------------------------------------------------------------------

def test_synthetic_cost_shape_is_realistic(synth):
    p = synth["panel"]
    annual = p[p["month_idx"] < 12].groupby("member_id")["paid_amt"].sum()
    s = data.cost_distribution_summary(annual)
    assert 0.05 < s["share_zero"] < 0.25                 # MEPS: roughly 10-20% of people spend $0 in a year
    assert s["top1_share"] > 0.10                        # heavy tail
    assert s["mean"] > 2 * s["p50"]                      # right skew


def test_synthetic_selection_and_counterfactual(synth):
    m, p = synth["members"], synth["panel"]
    assert m.loc[m["treated_flag"] == 1, "chronic_cnt"].mean() > m.loc[m["treated_flag"] == 0, "chronic_cnt"].mean()
    ctrl = p[p["member_id"].isin(m.loc[m["treated_flag"] == 0, "member_id"])]
    assert (ctrl["paid_amt"] == ctrl["paid_cf_amt"]).all()   # the program touches participants only
    assert synth["truth_att_pmpm"] > 0


def test_meps_loader_csv(tmp_path):
    f = tmp_path / "h243.csv"
    pd.DataFrame({"DUPERSID": ["10001101", "10001102", "10001103"], "TOTEXP22": [0, 1500, 250000],
                  "PERWT22F": [1000.5, 0.0, 800.0], "AGE22X": [40, 5, 70], "SEX": [2, 1, 1],
                  "OTHER": [9, 9, 9]}).to_csv(f, index=False)
    df = data.load_meps_totexp(f, 2022)
    assert df["person_id"].tolist() == ["10001101", "10001103"]      # weight 0 = out of scope, dropped
    assert df["sex_cd"].tolist() == ["F", "M"] and df["totexp_amt"].iloc[1] == 250000


def test_cost_summary_weighted():
    s = data.cost_distribution_summary(pd.Series([0.0, 100.0, 100.0, 10_000.0]), pd.Series([2.0, 1.0, 1.0, 0.0]))
    assert s["share_zero"] == 0.5 and s["mean"] == 50.0               # zero-weight outlier ignored


# --- checks ----------------------------------------------------------------------------------

def test_balance_check():
    ok = pd.DataFrame({"covariate": ["a", "b"], "smd_after": [0.05, -0.09]})
    assert not checks.check_balance(ok)
    assert checks.check_balance(ok.assign(smd_after=[0.05, 0.25]))[0].n_rows == 1


def test_common_support_check():
    assert checks.check_common_support(100, 95) == []
    assert checks.check_common_support(100, 80)[0].n_rows == 20


def test_pre_trend_check():
    assert checks.check_pre_trends({"slope_diff_pmpm_per_month": 30.0, "p_value": 0.01})[0].check_id == "ROI-003"
    assert not checks.check_pre_trends({"slope_diff_pmpm_per_month": 1.0, "p_value": 0.6})


def test_regression_to_mean_check_fires_on_participants_only(mp):
    assert checks.check_regression_to_mean(mp)[0].check_id == "ROI-004"
    controls_as_if_treated = mp[mp["treated_flag"] == 0].assign(treated_flag=1)
    assert not checks.check_regression_to_mean(controls_as_if_treated)


def test_effect_precision_check():
    assert "includes $0" in checks.check_effect_precision({"ci_lo": -100.0, "ci_hi": 20.0}, 50.0)[0].message
    assert checks.check_effect_precision({"ci_lo": -300.0, "ci_hi": -40.0}, 50.0)[0].severity == "info"
    assert not checks.check_effect_precision({"ci_lo": -300.0, "ci_hi": -80.0}, 50.0)


# --- SQL twin --------------------------------------------------------------------------------

duckdb = pytest.importorskip("duckdb")
from roi_cost_offset.sqltwin import run_sql  # noqa: E402


def test_member_period_sql_matches_pandas(synth, mp, cfg):
    sql = run_sql("01_member_period_pmpm.sql", {"panel": synth["panel"], "members": synth["members"]},
                  pre_months=cfg.pre_months, post_months=cfg.post_months,
                  min_months=cfg.min_months_each_period).set_index("member_id")
    py = mp.set_index("member_id").sort_index()
    assert list(sql.index) == list(py.index)
    for c in ("pre_paid_amt", "pre_mm", "post_paid_amt", "post_pmpm"):
        assert np.allclose(sql[c].to_numpy(float), py[c].to_numpy(float)), c


def test_did_cells_sql_matches_pandas(matched):
    sql = run_sql("02_did_cells.sql", {"matched": matched[["member_id", "treated_flag", "pre_paid_amt", "pre_mm",
                                                          "post_paid_amt", "post_mm"]]}).iloc[0]
    assert sql["did_pmpm"] == pytest.approx(methods.did_pmpm(matched)["did_pmpm"])
