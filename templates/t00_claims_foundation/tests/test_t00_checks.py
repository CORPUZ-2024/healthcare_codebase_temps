"""Each check must fire on an injected defect and stay quiet on clean data."""
import pandas as pd

from claims_foundation import checks, data


def ids(findings):
    return {f.check_id for f in findings}


def test_clean_synthetic_has_only_expected_findings(universe):
    # the generator deliberately includes multi-location NPIs, overlapping spans and runout
    assert ids(checks.run_all(universe)) <= {"VAL-005", "CLN-061", "VAL-031", "VAL-012"}


def test_npi_leading_zero_lost():
    good = data.luhn_npi("123456789")
    assert not checks.check_npi_luhn(pd.Series([good]))
    assert ids(checks.check_npi_luhn(pd.Series([good, "123456780", 1234567893 // 10])))== {"CLN-014"}


def test_icd9_flagged():
    assert not checks.check_icd10_format(pd.Series(["E119", "I10", "S72001A", "U071"]))
    assert checks.check_icd10_format(pd.Series(["4019", "25000"]))[0].n_rows == 2


def test_hcpcs_and_ndc_formats():
    assert not checks.check_hcpcs_format(pd.Series(["99213", "T1019", "0001U"]))
    assert checks.check_hcpcs_format(pd.Series(["99213-25", "9921"]))[0].n_rows == 2
    assert checks.check_ndc_format(pd.Series(["1234567890", 12345678901]))[0].n_rows == 1


def test_date_checks():
    c = pd.DataFrame({"svc_from_dt": pd.to_datetime(["2025-01-10", "2025-01-10"]),
                      "svc_to_dt": pd.to_datetime(["2025-01-09", "2025-01-10"]),
                      "paid_dt": pd.to_datetime(["2025-02-01", "2025-01-01"]),
                      "admit_dt": pd.NaT, "discharge_dt": pd.NaT})
    assert ids(checks.check_dates(c)) == {"VAL-011", "VAL-014"}


def test_grain_duplicate():
    df = pd.DataFrame({"claim_id": ["A", "A"], "line_seq": [1, 1], "adj_seq": [0, 0]})
    assert checks.check_grain_unique(df, ["claim_id", "line_seq", "adj_seq"])[0].n_rows == 1


def test_claims_outside_enrollment():
    enr = pd.DataFrame({"member_id": ["A"], "enroll_start_dt": pd.to_datetime(["2025-01-01"]),
                        "enroll_end_dt": pd.to_datetime(["2025-03-31"])})
    c = pd.DataFrame({"claim_id": ["1", "2"], "member_id": ["A", "A"],
                      "svc_from_dt": pd.to_datetime(["2025-02-01", "2025-05-01"])})
    assert checks.check_claims_outside_enrollment(c, enr)[0].n_rows == 1
