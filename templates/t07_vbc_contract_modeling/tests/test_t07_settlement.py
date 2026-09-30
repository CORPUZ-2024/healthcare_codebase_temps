"""Known-answer settlements for each contract type; every reconciliation must close (ANL-015)."""
import numpy as np
import pandas as pd
import pytest

from vbc_contract_modeling import checks, data, methods

B, N = 10_000.0, 1_000


def _ss(c, actual_pmpy, q=0.9):
    return methods.reconcile_shared_savings(c, pd.Series([actual_pmpy] * N), B, q)


# --- shared savings ----------------------------------------------------------------------------

@pytest.mark.parametrize("actual, q, expected", [
    (9_500.0, 0.90, 337_500.0),       # 5% savings x 0.75 x 0.9
    (9_800.0, 0.90, 135_000.0),       # exactly at the 2% MSR: shares (>=), first dollar
    (9_801.0, 0.90, 0.0),             # 1.99%: inside the corridor
    (9_500.0, 0.30, 0.0),             # quality gate (0.40) missed: no savings
    (10_500.0, 0.90, -200_000.0),     # 5% loss; loss rate 1 - 0.675 floored at 0.40
    (10_500.0, 0.30, -375_000.0),     # low quality -> loss rate capped at 0.75
    (7_000.0, 1.00, 2_000_000.0),     # 30% savings x 0.75 = $2.25M, capped at 20% of benchmark = $2M
    (12_000.0, 0.90, -800_000.0),     # 20% loss x 0.40 = $800K; the 15% loss cap ($1.5M) does not bind
])
def test_enhanced_known_answers(enh, actual, q, expected):
    r = _ss(enh, actual, q)
    assert r["aco_amount"] == pytest.approx(expected)
    assert not checks.check_reconciliation_closes(r)


def test_loss_cap_binds(enh):
    r = _ss(enh, 16_000.0, 0.30)      # 60% loss x 0.75 = $4.5M > 15% cap x $10M = $1.5M
    assert r["aco_amount"] == pytest.approx(-1_500_000.0)


def test_one_sided_never_owes(basic):
    assert _ss(basic, 13_000.0)["aco_amount"] == 0.0
    assert _ss(basic, 9_500.0)["aco_amount"] == pytest.approx(500_000 * 0.40 * 0.9)


def test_first_dollar_vs_above_msr(enh):
    """At 3% savings with a 2% MSR, sharing only above the MSR pays one third as much."""
    first = _ss(enh, 9_700.0)["aco_amount"]
    above = _ss({**enh, "share_from_first_dollar": False}, 9_700.0)["aco_amount"]
    assert first == pytest.approx(3 * above)


def test_truncation_applies_to_actuals(enh):
    costs = pd.Series([200_000.0] + [5_000.0] * 99)
    r = methods.reconcile_shared_savings(enh, costs, 7_000.0, 0.9)
    assert r["truncation_amt_removed"] == 50_000.0
    assert r["actual_total"] == 150_000.0 + 495_000.0
    assert not checks.check_reconciliation_closes(r)


def test_truncated_mean_formula_and_inverse():
    rng = np.random.default_rng(3)
    x = data.sample_annual_costs(rng, 400_000, 12_000.0)
    assert data.truncated_mean(12_000.0, 150_000.0) == pytest.approx(np.minimum(x, 150_000).mean(), rel=0.01)
    raw = data.raw_mean_for_truncated(11_000.0, 150_000.0)
    assert raw > 11_000.0 and data.truncated_mean(raw, 150_000.0) == pytest.approx(11_000.0)


def test_no_true_savings_means_savings_rate_near_zero(enh):
    """Benchmark and actuals on the same truncated basis: no systematic phantom savings."""
    rates = [methods.reconcile_shared_savings(enh, data.generate_aco_year(12_000, B, 0.0, seed=s)["annual_cost_amt"], B, 0.9)
             ["savings_rate"] for s in range(20)]
    assert abs(np.mean(rates)) < 0.006                       # SE of the mean of 20 years ~ 0.004


# --- sub-cap -----------------------------------------------------------------------------------

def test_corridor_bands():
    bands = [dict(lo=0, hi=.03, provider_share=1), dict(lo=.03, hi=.10, provider_share=.5), dict(lo=.10, hi=9.99, provider_share=0)]
    assert methods.corridor_provider_share(0.02, bands) == pytest.approx(0.02)
    assert methods.corridor_provider_share(0.08, bands) == pytest.approx(0.03 + 0.025)
    assert methods.corridor_provider_share(-0.25, bands) == pytest.approx(-(0.03 + 0.035))


def test_subcap_known_answer(contracts):
    c = contracts["FAKE_MEDICAID_SUBCAP"]          # cap 410, attach 40,000, coins 0.8
    m = pd.DataFrame({"member_id": ["A", "B"], "member_months": [12.0, 12.0], "hcbs_cost_amt": [4_000.0, 50_000.0]})
    r = methods.reconcile_subcap(c, m)
    revenue, recovery = 410 * 24, 0.8 * 10_000
    margin = revenue - 54_000 + recovery
    assert (r["cap_revenue"], r["stoploss_recovery"], r["margin"]) == pytest.approx((revenue, recovery, margin))
    kept = methods.corridor_provider_share(margin / revenue, c["corridors"]) * revenue
    assert r["provider_result"] == pytest.approx(kept)
    assert not checks.check_reconciliation_closes(r)


def test_subcap_synthetic_closes(contracts, cfg):
    r = methods.reconcile_subcap(contracts["FAKE_MEDICAID_SUBCAP"], data.generate_subcap_members(cfg.n_subcap_members, seed=cfg.seed))
    assert not checks.check_reconciliation_closes(r)
    assert abs(r["provider_result"]) < abs(r["margin"])      # corridor shares the loss


# --- fee + upside ------------------------------------------------------------------------------

def _cm(actual_pmpm):
    return pd.DataFrame({"member_id": ["A"], "member_months": [12.0], "baseline_pmpm": [1_000.0], "actual_pmpm": [actual_pmpm]})


def test_fee_upside_known_answer(contracts):
    c = contracts["FAKE_CM_FEE_UPSIDE"]            # fee 150, trend 5%, MSR 2% (above-MSR only), 50%, cap 100% fees
    r = methods.reconcile_fee_upside(c, _cm(945.0), {"a": True, "b": True, "c": False})
    target, fees = 1_050 * 12, 150 * 12
    gross = target - 945 * 12                      # 10% savings
    upside = min((0.10 - 0.02) * target * 0.5, fees)
    assert (r["gross_savings"], r["upside"], r["vendor_revenue"]) == pytest.approx((gross, upside, fees + upside))
    assert not checks.check_reconciliation_closes(r)


def test_fee_upside_quality_gate_and_cap(contracts):
    c = contracts["FAKE_CM_FEE_UPSIDE"]
    missed = methods.reconcile_fee_upside(c, _cm(945.0), {"a": True, "b": False, "c": False})
    assert missed["upside"] == 0.0 and missed["fee_refund"] == pytest.approx(0.10 * 1_800)
    huge = methods.reconcile_fee_upside(c, _cm(400.0), {"a": True, "b": True})
    assert huge["upside"] == pytest.approx(huge["fees"])      # capped at 100% of fees


# --- grid, Monte Carlo, MSR ----------------------------------------------------------------------

def test_scenario_grid_shape_and_monotone(enh):
    g = methods.scenario_grid(enh, B, N, [-0.05, 0.0, 0.03, 0.05], [0.5, 0.9])
    assert len(g) == 8
    for _, grp in g.groupby("quality"):
        assert grp.sort_values("savings_pct")["aco_amount"].is_monotonic_increasing


def test_monte_carlo_matches_theory(basic):
    sims, s = methods.monte_carlo_shared_savings(basic, 12_000, B, 0.0, 0.9, n_sims=1_500, seed=2)
    assert abs(sims["savings_rate"].mean()) < 3 * s["sd_savings_rate"] / np.sqrt(len(sims)) + 1e-3
    from scipy import stats
    expected_p = stats.norm.sf(basic["msr_pct"] / s["sd_savings_rate"])
    assert s["p_shared_savings"] == pytest.approx(expected_p, abs=0.03)
    assert s["p_shared_losses"] == 0.0                         # one-sided


def test_msr_reduces_chance_payouts(enh):
    _, wide = methods.monte_carlo_shared_savings(enh, 12_000, B, 0.0, 0.9, n_sims=800, seed=4)
    _, none = methods.monte_carlo_shared_savings({**enh, "msr_pct": 0.0, "mlr_pct": 0.0}, 12_000, B, 0.0, 0.9, n_sims=800, seed=4)
    assert wide["p_shared_savings"] < none["p_shared_savings"]
    assert none["p_shared_savings"] == pytest.approx(0.5, abs=0.06)
