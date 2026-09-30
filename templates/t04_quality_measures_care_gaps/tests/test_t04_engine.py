"""Continuous enrollment and the measure engine on hand-built members with known answers."""
import pandas as pd
import pytest

from quality_measures_care_gaps import methods

MY = 2025


def _spans(*pairs, mid="A"):
    return pd.DataFrame({"member_id": mid, "enroll_start_dt": pd.to_datetime([p[0] for p in pairs]),
                         "enroll_end_dt": pd.to_datetime([p[1] for p in pairs])})


def _member(mid, sex="F", birth="1960-05-01"):
    return {"member_id": mid, "sex_cd": sex, "birth_dt": pd.Timestamp(birth)}


def _ev(mid, dt, system, code, source="CLAIM"):
    return {"member_id": mid, "event_dt": pd.Timestamp(dt), "code_system": system, "code_cd": code, "source_cd": source}


def _enrolled(members, start="2023-01-01"):
    return pd.DataFrame({"member_id": members["member_id"], "enroll_start_dt": pd.Timestamp(start),
                         "enroll_end_dt": pd.Timestamp("2025-12-31")})


@pytest.mark.parametrize("pairs, gap_cnt, max_gap, ce", [
    ([("2025-01-01", "2025-12-31")], 0, 0, 1),                                   # full year
    ([("2025-01-01", "2025-01-31"), ("2025-02-01", "2025-12-31")], 0, 0, 1),     # adjacent spans: no gap
    ([("2025-01-01", "2025-01-31"), ("2025-03-18", "2025-12-31")], 1, 45, 1),    # 45-day gap allowed
    ([("2025-01-01", "2025-01-31"), ("2025-03-19", "2025-12-31")], 1, 46, 0),    # 46 days fails
    ([("2025-02-01", "2025-12-31")], 1, 31, 1),                                  # a gap at the START counts
    ([("2025-01-01", "2025-03-31"), ("2025-04-11", "2025-06-30"), ("2025-07-11", "2025-12-31")], 2, 10, 0),
    ([("2025-01-01", "2025-12-30")], 1, 1, 0),                                   # not enrolled on the anchor
    ([("2025-01-01", "2025-08-31"), ("2025-03-01", "2025-06-30"), ("2025-09-01", "2025-12-31")], 0, 0, 1),  # nested
])
def test_continuous_enrollment_known_answers(pairs, gap_cnt, max_gap, ce):
    r = methods.continuous_enrollment(_spans(*pairs), "2025-01-01", "2025-12-31", "2025-12-31").iloc[0]
    assert (r["gap_cnt"], r["max_gap_days"], r["ce_flag"]) == (gap_cnt, max_gap, ce)


def test_member_with_no_span_is_not_ce():
    out = methods.continuous_enrollment(_spans(("2025-01-01", "2025-12-31")), "2025-01-01", "2025-12-31",
                                        "2025-12-31", members=pd.Series(["A", "Z"])).set_index("member_id")
    assert out.loc["Z", "ce_flag"] == 0 and out.loc["Z", "max_gap_days"] == 365


def test_two_year_ce_allows_one_gap_per_year(specs):
    spec = specs["FAKE_BCS"]                       # CE period = MY-1 and MY
    one_each = _spans(("2024-01-01", "2024-05-31"), ("2024-06-21", "2025-05-31"), ("2025-06-21", "2025-12-31"))
    two_in_one = _spans(("2024-01-01", "2024-03-31"), ("2024-04-21", "2024-07-31"), ("2024-08-21", "2025-12-31"))
    assert methods.ce_flags(one_each, spec, MY, pd.Series(["A"]))["ce_flag"].iloc[0] == 1
    assert methods.ce_flags(two_in_one, spec, MY, pd.Series(["A"]))["ce_flag"].iloc[0] == 0


def test_bcs_rules(specs, vs):
    members = pd.DataFrame([_member("IN_WINDOW"), _member("TOO_EARLY"), _member("HOSPICE"), _member("MASTECTOMY"),
                            _member("MALE", "M"), _member("AGE_51", birth="1974-01-01"),
                            _member("AGE_74", birth="1951-12-31"), _member("HOSPICE_LAST_YEAR")])
    ev = pd.DataFrame([_ev("IN_WINDOW", "2023-10-01", "CPT", "77067"), _ev("TOO_EARLY", "2023-09-30", "CPT", "77067"),
                       _ev("HOSPICE", "2025-06-01", "HCPCS", "Q5001"), _ev("HOSPICE", "2025-01-10", "CPT", "77067"),
                       _ev("MASTECTOMY", "2023-02-01", "ICD10CM", "z90.13"),
                       _ev("AGE_74", "2025-02-01", "CPT", "77067"),
                       _ev("HOSPICE_LAST_YEAR", "2024-06-01", "HCPCS", "Q5001")])
    ml = methods.evaluate_measure(specs["FAKE_BCS"], members, _enrolled(members), ev, vs, MY).set_index("member_id")
    assert ml.loc["IN_WINDOW", "status"] == "MET"                               # first day of the 27-month window
    assert ml.loc["TOO_EARLY", "status"] == "OPEN_GAP"                          # one day before it
    assert ml.loc["HOSPICE", "status"] == "EXCLUDED" and ml.loc["HOSPICE", "numerator_flag"] == 0
    assert ml.loc["MASTECTOMY", "exclusion_reason"] == "BILATERAL_MASTECTOMY"   # 'z90.13' normalized
    assert ml.loc["AGE_74", "status"] == "MET"                                  # 74 on Dec 31 is in
    assert ml.loc["HOSPICE_LAST_YEAR", "status"] == "OPEN_GAP"                  # outside the exclusion window
    assert "MALE" not in ml.index and "AGE_51" not in ml.index


def test_diabetes_needs_two_distinct_dates_and_admin_sources(specs, vs):
    members = pd.DataFrame([_member(m) for m in ("TWO", "ONE", "SAME_DAY", "LAB", "CHART")])
    dm = [_ev(m, d, "ICD10CM", "E119") for m in ("TWO", "LAB", "CHART") for d in ("2024-03-01", "2025-03-01")]
    dm += [_ev("ONE", "2025-03-01", "ICD10CM", "E119"),
           _ev("SAME_DAY", "2025-03-01", "ICD10CM", "E119"), _ev("SAME_DAY", "2025-03-01", "ICD10CM", "E1165")]
    num = [_ev("TWO", "2025-04-01", "CPT", "83036"), _ev("LAB", "2025-04-01", "LOINC", "4548-4", "LAB"),
           _ev("CHART", "2025-04-01", "LOINC", "4548-4", "CHART")]
    ml = methods.evaluate_measure(specs["FAKE_HBA1C_TEST"], members, _enrolled(members), pd.DataFrame(dm + num),
                                  vs, MY).set_index("member_id")
    assert ml.loc["ONE", "status"] == ml.loc["SAME_DAY", "status"] == "NO_DENOM_EVENT"
    assert ml.loc["TWO", "status"] == ml.loc["LAB", "status"] == "MET"          # lab feed = standard supplemental
    assert ml.loc["CHART", "status"] == "OPEN_GAP"                               # chart evidence only counts in hybrid


def test_join_on_code_system_not_code_alone(specs, vs):
    members = pd.DataFrame([_member("K", birth="2015-01-01")])
    ev = pd.DataFrame([_ev("K", "2025-05-01", "ICD10CM", "99393")])            # right string, wrong code system
    ml = methods.evaluate_measure(specs["FAKE_WCV"], members, _enrolled(members), ev, vs, MY)
    assert ml["status"].iloc[0] == "OPEN_GAP"


def test_funnel_identities_on_synthetic(evaluated):
    for ml in evaluated.values():
        s = methods.rate_summary(ml)
        assert s["denominator_cnt"] == s["eligible_cnt"] - s["exclusion_cnt"]
        assert s["numerator_cnt"] <= s["denominator_cnt"]
        assert (ml["gap_open_flag"] + ml["numerator_flag"] == ml["denominator_flag"]).all()
        assert set(ml["status"]) <= {"NOT_CE", "NO_DENOM_EVENT", "EXCLUDED", "MET", "OPEN_GAP"}


def test_synthetic_has_every_status(evaluated):
    seen = set().union(*(set(ml["status"]) for ml in evaluated.values()))
    assert seen == {"NOT_CE", "NO_DENOM_EVENT", "EXCLUDED", "MET", "OPEN_GAP"}
