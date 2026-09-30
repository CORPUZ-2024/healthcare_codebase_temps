"""Known answers for alignment, DiD, ROI and tornado; recovery of the synthetic truth."""
import numpy as np
import pandas as pd
import pytest

from roi_cost_offset import data, methods


# --- alignment and exposure ------------------------------------------------------------------

def _tiny_panel():
    members = pd.DataFrame({"member_id": ["A", "B"], "index_month_idx": [3, 3], "treated_flag": [1, 0],
                            "age": [60, 61], "sex_cd": ["F", "M"], "lob_cd": "MCD", "chronic_cnt": [2, 2],
                            "risk_score": [1.0, 1.0]})
    rows = [("A", t, 100.0 * (t + 1)) for t in range(7)] + [("B", t, 50.0) for t in range(7) if t != 5]
    panel = pd.DataFrame(rows, columns=["member_id", "month_idx", "paid_amt"]).assign(member_months=1.0, ip_admit_flag=0)
    return panel, members


def test_member_periods_excludes_index_month_and_respects_exposure():
    panel, members = _tiny_panel()
    mp = methods.member_periods(panel, members, pre_months=3, post_months=3, min_months=2).set_index("member_id")
    assert mp.loc["A", "pre_paid_amt"] == 100 + 200 + 300           # months 0,1,2 (index month 3 excluded)
    assert mp.loc["A", "post_paid_amt"] == 500 + 600 + 700
    assert mp.loc["B", "post_mm"] == 2                               # month 5 missing -> 2 months, not $0
    assert mp.loc["B", "post_pmpm"] == 50.0
    strict = methods.member_periods(panel, members, 3, 3, min_months=3)
    assert set(strict["member_id"]) == {"A"}


# --- DiD -------------------------------------------------------------------------------------

def _pairs_frame():
    mp = pd.DataFrame({"member_id": list("ABCD"), "treated_flag": [1, 1, 0, 0], "pair_id": [0, 1, 0, 1],
                       "pre_paid_amt": [600.0, 1200.0, 300.0, 600.0], "pre_mm": [6.0, 6.0, 6.0, 3.0],
                       "post_paid_amt": [300.0, 600.0, 360.0, 720.0], "post_mm": [6.0, 3.0, 6.0, 6.0]})
    mp["pre_pmpm"], mp["post_pmpm"] = mp["pre_paid_amt"] / mp["pre_mm"], mp["post_paid_amt"] / mp["post_mm"]
    return mp


def test_did_regression_equals_ratio_of_sums():
    mp = _pairs_frame()
    d = methods.did_pmpm(mp)
    t_pre, t_post = 1800 / 12, 900 / 9
    c_pre, c_post = 900 / 9, 1080 / 12
    assert d["did_pmpm"] == pytest.approx((t_post - t_pre) - (c_post - c_pre))
    assert (d["treated_pre"], d["control_post"]) == pytest.approx((t_pre, c_post))


def test_bootstrap_point_equals_did_and_ci_contains_it(matched):
    d, b = methods.did_pmpm(matched), methods.bootstrap_did(matched, n_boot=300, seed=1, breakeven_savings_pmpm=0.0)
    assert b["did_pmpm"] == pytest.approx(d["did_pmpm"])
    assert b["ci_lo"] < b["did_pmpm"] < b["ci_hi"]
    assert b["prob_breakeven"] == pytest.approx(b["prob_savings"], abs=0.01)   # break-even 0 = any savings


def test_naive_overstates_and_matched_did_recovers_truth(mp, matched, truth):
    d = methods.did_pmpm(matched)
    naive = methods.naive_pre_post(mp)
    assert d["ci_lo"] < truth < d["ci_hi"]                         # CI covers the known effect
    assert d["ci_hi"] < 0                                          # and detects savings
    assert abs(naive - truth) > abs(d["did_pmpm"] - truth)         # naive pre/post is further off
    assert naive < truth                                           # ... in the "too good" direction


def test_unmatched_did_is_biased_by_selection(mp, matched, truth):
    everyone = mp.assign(pair_id=np.arange(len(mp)))
    crude = methods.did_pmpm(everyone)["did_pmpm"]
    assert abs(crude - truth) > abs(methods.did_pmpm(matched)["did_pmpm"] - truth)


def test_matching_on_cost_without_the_trigger_overstates_savings(mp, matched, truth, cfg):
    """Cost is a noisy proxy for the admission that triggered referral: leave the trigger out and the
    controls don't regress to the mean like participants do -> savings roughly doubled."""
    cov = [c for c in methods.COVARIATES if c != "pre_last3_admit_flag"]
    no_trigger = methods.match_propensity(mp, methods.propensity_scores(mp, cov), exact_on=("lob_cd",), seed=cfg.seed)
    bal = methods.balance_table(mp, no_trigger).set_index("covariate")
    assert abs(bal.loc["pre_last3_admit_flag", "smd_after"]) > 0.3          # trigger left unbalanced
    assert methods.did_pmpm(no_trigger)["did_pmpm"] < 1.5 * truth           # e.g. -403 vs its own truth -217
    assert abs(methods.did_pmpm(matched)["did_pmpm"] - truth) < abs(methods.did_pmpm(no_trigger)["did_pmpm"] - truth)


def test_no_effect_gives_ci_covering_zero(cfg):
    d = data.generate_panel(cfg.n_members, cfg.months, cfg.start, effect_pct=0.0, seed=cfg.seed)
    assert d["truth_att_pmpm"] == 0.0
    mp = methods.member_periods(d["panel"], d["members"])
    m = methods.match_propensity(mp, methods.propensity_scores(mp), exact_on=cfg.exact_on, seed=cfg.seed)
    r = methods.did_pmpm(m)
    assert r["ci_lo"] < 0 < r["ci_hi"]


def test_two_part_direction(matched):
    tp = methods.two_part_did(matched)
    assert tp["did_pmpm"] < 0 and tp["part2_cost_ratio"] < 1


# --- matching --------------------------------------------------------------------------------

def test_matching_is_one_to_one_exact_and_inside_caliper(mp, matched, cfg):
    assert matched.groupby("pair_id")["treated_flag"].agg(["sum", "count"]).eq([1, 2]).all().all()
    assert matched["member_id"].is_unique                           # without replacement
    for col in cfg.exact_on:
        assert (matched.groupby("pair_id")[col].nunique() == 1).all()
    caliper = cfg.caliper_sd * methods.propensity_scores(mp).std()
    gaps = matched.groupby("pair_id")["ps_logit"].agg(lambda s: s.max() - s.min())
    assert (gaps <= caliper + 1e-12).all()


def test_matching_balances_covariates(mp, matched):
    bal = methods.balance_table(mp, matched)
    assert (bal["smd_before"].abs() > 0.3).sum() >= 4               # participants really are different
    assert (bal["smd_after"].abs() < 0.1).all()


def test_pre_trends_parallel_after_matching(synth, matched):
    assert methods.pre_trend_test(synth["panel"], matched)["p_value"] > 0.05


# --- ROI and tornado -------------------------------------------------------------------------

def test_roi_break_even_is_zero_net():
    r = methods.roi(60.0, 25.0, 150.0, 100, 12)
    be = methods.roi(r["breakeven_savings_pmpm"], 25.0, 150.0, 100, 12)
    assert be["net_savings"] == pytest.approx(0.0) and be["roi"] == pytest.approx(0.0)
    assert r["savings_to_cost"] == pytest.approx(r["roi"] + 1)


def test_tornado_sorted_and_base_consistent():
    base = dict(savings_pmpm=200.0, program_fee_pmpm=150.0, one_time_cost=300.0, participants=500, months_in_program=12)
    t = methods.tornado(base, {"savings_pmpm": (50.0, 330.0), "program_fee_pmpm": (120.0, 180.0), "participants": (350, 650)})
    assert t["swing"].is_monotonic_decreasing and t["input"].iloc[0] == "savings_pmpm"
    assert t.attrs["base"] == methods.roi(**base)["net_savings"]
    fee = t.set_index("input").loc["program_fee_pmpm"]
    assert fee["metric_at_low"] > fee["metric_at_high"]              # higher fee -> lower net savings
