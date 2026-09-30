import pandas as pd

from hcc_risk_adjustment import checks, methods


def test_model_year_mismatch():
    dx = pd.DataFrame({"svc_dt": pd.to_datetime(["2025-06-01", "2024-06-01"])})
    assert checks.check_model_year(dx, 2026)[0].n_rows == 1
    assert not checks.check_model_year(dx.iloc[[0]], 2026)


def test_unacceptable_sources(synth):
    f = checks.check_unacceptable_sources(synth["dx"], methods.ACCEPTABLE_SOURCES)
    assert f and f[0].check_id == "RA-010"


def test_unmapped_share_flags_format_problem(ref):
    dx = pd.DataFrame({"member_id": ["A", "B"], "dx_cd": ["4280", "25000"]})     # ICD-9
    hcc = methods.map_dx_to_hcc(dx, ref["dx_map"], "V28")
    assert checks.check_unmapped_share(dx, hcc)[0].check_id == "RA-020"


def test_score_range():
    assert checks.check_score_range(pd.Series([0.5, 40.0]))[0].n_rows == 1
