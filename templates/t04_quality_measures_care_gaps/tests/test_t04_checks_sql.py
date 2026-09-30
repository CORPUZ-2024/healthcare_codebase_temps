import pandas as pd
import pytest

from quality_measures_care_gaps import checks, data, methods

MY = 2025


def test_code_format_check(synth):
    assert checks.check_code_format(synth["events"])[0].check_id == "QM-001"
    assert not checks.check_code_format(methods.normalize_codes(synth["events"]))


def test_overlapping_spans_check(synth):
    assert checks.check_overlapping_spans(synth["enrollment"])[0].n_rows > 0
    clean = pd.DataFrame({"member_id": ["A", "A"], "enroll_start_dt": pd.to_datetime(["2025-01-01", "2025-02-01"]),
                          "enroll_end_dt": pd.to_datetime(["2025-01-31", "2025-12-31"])})
    assert not checks.check_overlapping_spans(clean)


def test_runout_check():
    assert checks.check_runout("2026-01-15", MY)[0].severity == "error"
    assert not checks.check_runout("2026-03-31", MY)


def test_missing_value_set_check(specs, vs):
    assert not checks.check_value_sets_exist(specs, vs)
    assert checks.check_value_sets_exist(specs, vs[vs["value_set_name"] != "HOSPICE"])[0].n_rows == 3


def test_small_denominator_check():
    s = pd.DataFrame({"measure_id": ["A", "B"], "denominator_cnt": [12, 400]})
    assert checks.check_small_denominators(s)[0].n_rows == 1


def test_benchmark_vintage_check():
    b = data.load_core_set_rates(data.generate_fake_core_set(2026))
    assert not checks.check_benchmark_vintage(b, MY)
    assert checks.check_benchmark_vintage(b, 2023)[0].check_id == "ANL-013"
    mixed = b.assign(methodology=["Administrative", "Hybrid"] * (len(b) // 2))
    assert any("methodolog" in f.message for f in checks.check_benchmark_vintage(mixed, MY))


duckdb = pytest.importorskip("duckdb")
from quality_measures_care_gaps.sqltwin import run_sql  # noqa: E402


def test_ce_sql_matches_pandas(synth):
    enr = synth["enrollment"]
    py = methods.continuous_enrollment(enr, "2025-01-01", "2025-12-31", "2025-12-31").set_index("member_id")
    sql = run_sql("01_continuous_enrollment.sql", {"enrollment": enr}, ce_start="2025-01-01", ce_end="2025-12-31",
                  anchor_dt="2025-12-31", allowable_gaps=1, max_gap_days=45).set_index("member_id")
    assert len(sql) == len(py) and py["ce_flag"].sum() > 0
    for c in ("gap_cnt", "max_gap_days", "anchor_enrolled_flag", "ce_flag"):
        assert (sql.loc[py.index, c].astype(int) == py[c]).all(), c


def test_rate_sql_matches_pandas(evaluated):
    ml = pd.concat(evaluated.values(), ignore_index=True)[["measure_id", "eligible_flag", "exclusion_flag",
                                                          "denominator_flag", "numerator_flag"]]
    sql = run_sql("02_measure_rate.sql", {"member_level": ml}).set_index("measure_id")
    for mid, m in evaluated.items():
        s = methods.rate_summary(m)
        assert sql.loc[mid, "denominator_cnt"] == s["denominator_cnt"]
        assert sql.loc[mid, "rate"] == pytest.approx(s["rate"])
