import pandas as pd
import pytest

from vbc_contract_modeling import checks, data, methods


def test_all_contracts_have_required_terms(contracts):
    assert all(not checks.check_contract_fields(c) for c in contracts.values())
    broken = {k: v for k, v in contracts["FAKE_MSSP_ENHANCED"].items() if k != "savings_cap_pct"}
    assert "savings_cap_pct" in checks.check_contract_fields(broken)[0].message
    assert checks.check_contract_fields({"contract_id": "X", "type": "bundle"})[0].check_id == "VBC-001"


def test_closure_check_catches_a_broken_identity(enh):
    r = methods.reconcile_shared_savings(enh, pd.Series([9_500.0] * 10), 10_000.0, 0.9)
    r["identities"].append(("tampered", 100.0, 250.0))
    f = checks.check_reconciliation_closes(r)
    assert f[0].check_id == "ANL-015" and "tampered" in f[0].message


def test_corridor_check(contracts):
    assert not checks.check_corridors(contracts["FAKE_MEDICAID_SUBCAP"]["corridors"])
    gap = [dict(lo=0, hi=.03, provider_share=1), dict(lo=.05, hi=1, provider_share=.5)]
    assert checks.check_corridors(gap)[0].check_id == "VBC-004"
    assert checks.check_corridors([dict(lo=0, hi=1, provider_share=1.5)])


def test_msr_vs_random_variation_check():
    assert checks.check_msr_vs_random_variation(0.02, 2_000, 1.7)[0].check_id == "VBC-002"    # small ACO
    assert not checks.check_msr_vs_random_variation(0.02, 50_000, 1.7)


def test_target_basis_check():
    assert checks.check_target_basis(1_400.0, 540.0)[0].check_id == "VBC-003"
    assert not checks.check_target_basis(560.0, 540.0)


def _puf(tmp_path):
    f = tmp_path / "mssp_puf.csv"
    pd.DataFrame({
        "ACO_ID": ["A1111", "A2222", "A3333"], "ACO_Name": ["One", "Two", "Three"], "Current_Track": ["BASIC", "ENHANCED", "BASIC"],
        "N_AB": ["10,000", "25,000", "8,000"],
        "ABtotBnchmk": ["$120,000,000", "$300,000,000", "$96,000,000"],
        "ABtotExp": ["$114,000,000", "$297,000,000", "$99,000,000"],
        "BnchmkMinExp": ["6,000,000", "3,000,000", "-3,000,000"],
        "MinSavPerc": ["2.5", "2.0", "3.1"], "FinalShareRate": ["45", "75", "40"], "QualScore": ["90.5", "88", "80"],
        "EarnSaveLoss": ["2,700,000", "0", "0"],
    }).to_csv(f, index=False)
    return f


def test_puf_loader_parses_money_and_percents(tmp_path):
    p = data.load_mssp_puf(_puf(tmp_path))
    assert p["benchmark_total_amt"].iloc[0] == 120_000_000
    assert p["msr_pct"].tolist() == pytest.approx([0.025, 0.02, 0.031])
    assert p["final_share_rate"].iloc[1] == 0.75 and p["quality_score"].iloc[0] == pytest.approx(0.905)
    assert p["aco_id"].iloc[2] == "A3333"


def test_reproduce_puf(tmp_path):
    r = methods.reproduce_mssp_puf(data.load_mssp_puf(_puf(tmp_path))).set_index("aco_id")
    assert r["gross_match_flag"].all()
    assert r.loc["A1111", "earned_recalc_amt"] == pytest.approx(6_000_000 * 0.45)      # 5% >= 2.5% MSR
    assert r.loc["A2222", "earned_recalc_amt"] == 0.0                                  # 1% < 2% MSR
    assert r["earned_match_flag"].all()
