"""Known-answer and standard-vs-alternative tests for claims_foundation.methods."""
import pandas as pd
import pytest

from claims_foundation import data, methods


def enr(rows):
    return pd.DataFrame(rows, columns=["member_id", "enroll_start_dt", "enroll_end_dt"]).assign(
        enroll_start_dt=lambda d: pd.to_datetime(d["enroll_start_dt"]),
        enroll_end_dt=lambda d: pd.to_datetime(d["enroll_end_dt"]))


# ---------- version collapse ----------------------------------------------------------
def test_latest_version_known_answer():
    c = pd.DataFrame({
        "claim_id": ["A", "A", "B", "B", "C", "C"], "line_seq": [1, 1, 1, 1, 1, 1],
        "adj_seq": [0, 1, 0, 1, 0, 0], "claim_status_cd": ["P", "P", "P", "V", "P", "P"],
        "paid_amt": [100.0, 80.0, 50.0, 50.0, 10.0, 20.0]})
    c.loc[5, "line_seq"] = 2
    out = methods.collapse_versions_latest(c)
    assert out.set_index(["claim_id", "line_seq"])["paid_amt"].to_dict() == {("A", 1): 80.0, ("C", 1): 10.0, ("C", 2): 20.0}


def test_latest_equals_net_of_delta_feed(universe):
    med = universe["medical"]
    latest = methods.collapse_versions_latest(med)
    net = methods.collapse_versions_net(data.to_delta_feed(med))
    assert latest["paid_amt"].sum() == pytest.approx(net["paid_amt"].sum(), abs=0.05)
    assert len(latest) == len(net)


def test_net_on_replacement_feed_double_counts(universe):
    """The trap: using the delta method on a replacement feed inflates paid."""
    med = universe["medical"]
    assert methods.collapse_versions_net(med)["paid_amt"].sum() > methods.collapse_versions_latest(med)["paid_amt"].sum()


# ---------- member months -------------------------------------------------------------
def test_member_months_partial_month():
    out = methods.member_months_daily(enr([["A", "2025-01-17", "2025-02-28"]]), "2025-01-01", "2025-12-31")
    assert out["member_months"].round(3).tolist() == [0.484, 1.0]
    assert out["days_enrolled"].tolist() == [15, 28]


def test_member_months_overlap_not_double_counted():
    e = enr([["A", "2025-01-01", "2025-03-31"], ["A", "2025-02-10", "2025-04-15"]])
    out = methods.member_months_daily(e, "2025-01-01", "2025-12-31")
    assert out["member_months"].max() == pytest.approx(1.0)
    assert out["member_months"].sum() == pytest.approx(3 + 15 / 30)


def test_member_months_adjacent_spans_merge():
    e = enr([["A", "2025-01-01", "2025-01-15"], ["A", "2025-01-16", "2025-01-31"]])
    assert methods.member_months_daily(e, "2025-01-01", "2025-01-31")["member_months"].tolist() == [1.0]


def test_midmonth_rule():
    e = enr([["A", "2025-01-10", "2025-01-20"], ["B", "2025-01-16", "2025-02-14"]])
    out = methods.member_months_midmonth(e, "2025-01-01", "2025-02-28")
    assert out.set_index("member_id")["member_months"].to_dict() == {"A": 1.0}   # B misses both anchors


def test_daily_vs_midmonth_close_on_population(universe):
    d = methods.member_months_daily(universe["enrollment"], "2023-01-01", "2023-12-31")["member_months"].sum()
    m = methods.member_months_midmonth(universe["enrollment"], "2023-01-01", "2023-12-31")["member_months"].sum()
    assert abs(d - m) / d < 0.02


# ---------- service category ----------------------------------------------------------
def test_claim_level_hierarchy_ip_wins():
    c = pd.DataFrame({"claim_id": ["X", "X"], "line_seq": [1, 2], "claim_type_cd": ["INST", "INST"],
                      "tob_cd": ["111", "111"], "rev_cd": ["0450", "0300"], "pos_cd": [None, None],
                      "hcpcs_cd": ["99285", "83036"]})
    assert set(methods.service_category_claim(c)["service_category"]) == {"IP"}
    assert list(methods.service_category_line(c)["service_category"]) == ["IP", "IP"]


def test_ed_claim_lab_line_differs_by_method():
    c = pd.DataFrame({"claim_id": ["E", "E"], "line_seq": [1, 2], "claim_type_cd": ["INST", "INST"],
                      "tob_cd": ["131", "131"], "rev_cd": ["0450", "0300"], "pos_cd": [None, None],
                      "hcpcs_cd": ["99285", "83036"]})
    assert set(methods.service_category_claim(c)["service_category"]) == {"ED"}
    assert list(methods.service_category_line(c)["service_category"]) == ["ED", "OP"]


# ---------- pharmacy + NDC -------------------------------------------------------------
def test_pharmacy_reversals():
    rx = pd.DataFrame({"rx_claim_id": ["R1", "R1", "R2"], "member_id": ["A"] * 3,
                       "fill_dt": pd.to_datetime(["2025-01-01"] * 3), "ndc_cd": ["00000000001"] * 3,
                       "drug_class": ["STATIN"] * 3, "days_supply": [30] * 3,
                       "paid_amt": [10.0, -10.0, 12.0], "reversal_flag": [0, 1, 0]})
    assert methods.net_pharmacy_drop_pairs(rx)["rx_claim_id"].tolist() == ["R2"]
    signed = methods.net_pharmacy_signed(rx).set_index("rx_claim_id")["paid_amt"]
    assert signed.to_dict() == {"R1": 0.0, "R2": 12.0}


@pytest.mark.parametrize("raw,expected", [
    ("1234-5678-90", "01234567890"), ("12345-678-90", "12345067890"),
    ("12345-6789-0", "12345678900"), ("12345678901", "12345678901"), ("1234567890", None), (None, None)])
def test_normalize_ndc11(raw, expected):
    assert methods.normalize_ndc11(raw) == expected
