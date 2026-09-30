import pandas as pd
import pytest

from tcoc_pmpm_mlr import methods


# ---- PMPM --------------------------------------------------------------------------------
def test_ratio_of_sums_known_answer(tiny):
    c, mm = tiny
    assert methods.pmpm_ratio_of_sums(c, mm)["pmpm"].iloc[0] == pytest.approx(850 / 1.6)
    by_lob = methods.pmpm_ratio_of_sums(c, mm, by=["lob_cd"]).set_index("lob_cd")["pmpm"]
    assert by_lob["MCD"] == pytest.approx(450 / 1.5) and by_lob["DUAL"] == pytest.approx(4000)


def test_service_category_uses_total_exposure(tiny):
    c, mm = tiny
    out = methods.pmpm_ratio_of_sums(c, mm, by=["service_category"])
    assert (out["member_months"] == 1.6).all()
    assert out["pmpm"].sum() == pytest.approx(850 / 1.6)       # categories add up to total


def test_mean_of_members_overweights_short_stays(tiny):
    c, mm = tiny
    alt = methods.pmpm_mean_of_members(c, mm)
    assert alt["mean"] == pytest.approx((300 + 300 + 4000) / 3)
    assert alt["mean"] > methods.pmpm_ratio_of_sums(c, mm)["pmpm"].iloc[0]
    assert methods.pmpm_mean_of_members(c, mm, min_member_months=0.2)["n_members"] == 2


def test_lob_pmpm_reconciles_to_total(prepared):
    c, mm = prepared
    by = methods.pmpm_ratio_of_sums(c, mm, by=["lob_cd"])
    total = methods.pmpm_ratio_of_sums(c, mm)
    assert by["paid_amt"].sum() == pytest.approx(total["paid_amt"].iloc[0])
    assert by["member_months"].sum() == pytest.approx(total["member_months"].iloc[0])


def test_utilization_unit_cost_identity(prepared):
    c, mm = prepared
    uc = methods.utilization_unit_cost(c, mm)
    assert (uc["util_per_1000"] * uc["cost_per_unit"] / 12_000).round(6).tolist() == uc["pmpm"].round(6).tolist()


# ---- truncation --------------------------------------------------------------------------
def test_truncation_methods():
    cost = pd.Series([100.0] * 99 + [1_000_000.0], index=[f"M{i}" for i in range(100)])
    p = methods.truncate_percentile(cost, 0.99)
    f = methods.truncate_fixed_attachment(cost, 250_000)
    assert p["truncated_amt"].max() < 1_000_000
    assert f["truncated_amt"].max() == 250_000
    assert (p["cost_amt"] == p["truncated_amt"] + p["excess_amt"]).all()


def test_attachment_can_do_nothing_in_low_cost_population():
    cost = pd.Series([100.0, 200.0, 5_000.0], index=list("abc"))
    assert methods.truncate_fixed_attachment(cost, 250_000)["excess_amt"].sum() == 0
    assert methods.truncate_percentile(cost, 0.5)["excess_amt"].sum() > 0


# ---- decomposition ------------------------------------------------------------------------
def test_additive_and_log_decomposition():
    b, c = {"util_per_1000": 80.0, "cost_per_unit": 10_000.0}, {"util_per_1000": 72.0, "cost_per_unit": 11_000.0}
    a = methods.decompose_additive(b, c)
    assert a["utilization"] + a["unit_cost"] + a["interaction"] == pytest.approx(a["pmpm_change"])
    g = methods.decompose_log(b, c)
    assert (1 + g["utilization_trend_pct"]) * (1 + g["unit_cost_trend_pct"]) - 1 == pytest.approx(g["total_trend_pct"])
    assert g["utilization_trend_pct"] == pytest.approx(-0.10)


# ---- MLR -----------------------------------------------------------------------------------
def test_mlr_regulatory_known_answer():
    r = methods.mlr_regulatory(incurred_claims=8_000_000, quality_improvement=100_000, premium=10_000_000,
                               taxes_fees=300_000, minimum=0.85)
    assert r["mlr"] == pytest.approx(8_100_000 / 9_700_000)
    assert not r["meets_minimum"]
    assert r["rebate_or_remittance"] == pytest.approx((0.85 - 8_100_000 / 9_700_000) * 9_700_000)


def test_simple_ratio_lower_than_regulatory():
    assert methods.mlr_simple(80, 100)["loss_ratio"] < methods.mlr_regulatory(80, 1, 100, 3)["mlr"]


def test_mlr_impact_depends_on_fee_classification():
    claims = methods.mlr_impact(400, 500, 25, 15, "claims")
    admin = methods.mlr_impact(400, 500, 25, 15, "admin")
    assert claims["mlr_after"] == pytest.approx((400 - 25 + 15) / 500)
    assert admin["mlr_after"] == pytest.approx((400 - 25) / 500)
    assert claims["net_pmpm_to_plan"] == 10


# ---- episodes ------------------------------------------------------------------------------
def _ep_claims():
    d = pd.to_datetime
    return pd.DataFrame({
        "member_id": ["A"] * 5, "claim_id": ["IP1", "X1", "IP2", "X2", "X0"],
        "service_category": ["IP", "PROF", "IP", "PROF", "PROF"],
        "admit_dt": d(["2025-01-01", None, "2025-01-20", None, None]),
        "discharge_dt": d(["2025-01-05", None, "2025-01-25", None, None]),
        "svc_from_dt": d(["2025-01-01", "2025-01-10", "2025-01-20", "2025-02-20", "2024-12-30"]),
        "paid_amt": [10_000.0, 100.0, 8_000.0, 50.0, 25.0]})


def test_nonoverlap_episode_absorbs_readmission():
    ep = methods.build_episodes_nonoverlap(_ep_claims(), pre_days=3, post_days=30)
    assert len(ep) == 1
    # window 2024-12-29 .. 2025-02-04: IP1, X1, IP2, X0 (X2 is outside)
    assert ep["episode_paid_amt"].iloc[0] == pytest.approx(10_000 + 100 + 8_000 + 25)


def test_overlap_episodes_double_count():
    eo = methods.build_episodes_overlap(_ep_claims(), pre_days=3, post_days=30)
    ep = methods.build_episodes_nonoverlap(_ep_claims(), pre_days=3, post_days=30)
    assert len(eo) == 2
    assert eo["episode_paid_amt"].sum() > ep["episode_paid_amt"].sum()


def test_complete_months_drops_trailing(prepared):
    c, mm = prepared
    kept = methods.complete_months(c, mm, 3)
    assert kept["month"].max() == mm["month"].max() - 3
