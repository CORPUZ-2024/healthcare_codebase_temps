"""Intervals (Wilson vs. Jeffreys), stability flag, hybrid method, care-gap list, benchmarks."""
import numpy as np
import pandas as pd
import pytest

from quality_measures_care_gaps import data, methods

MY = 2025


@pytest.mark.parametrize("k, n", [(0, 10), (3, 10), (81, 263), (30, 30), (1, 400)])
def test_intervals_match_statsmodels(k, n):
    sm = pytest.importorskip("statsmodels.stats.proportion")
    z = 1.959963984540054                                            # statsmodels uses alpha=0.05 exactly
    assert methods.ci_wilson(k, n, z) == pytest.approx(sm.proportion_confint(k, n, method="wilson"), abs=1e-9)
    lo, hi = methods.ci_jeffreys(k, n, z)
    slo, shi = sm.proportion_confint(k, n, method="jeffreys")
    if 0 < k < n:                                                    # statsmodels has no boundary rule
        assert (lo, hi) == pytest.approx((slo, shi), abs=1e-9)
    assert lo == (0.0 if k == 0 else pytest.approx(slo, abs=1e-9))
    assert hi == (1.0 if k == n else pytest.approx(shi, abs=1e-9))


def test_intervals_stay_in_unit_range_and_contain_p():
    for k, n in [(0, 5), (5, 5), (2, 7), (50, 100)]:
        for f in (methods.ci_wilson, methods.ci_jeffreys):
            lo, hi = f(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_coverage_near_nominal_and_worst_case_near_zero():
    """Average exact coverage over true rates 5-95%: both close to 95%. Near 0 on SMALL denominators
    (n = 30, 60) Wilson's worst-case coverage dips lower than Jeffreys' (~85% vs ~89%). This is the
    documented reason to switch - not a general win: at n = 100 the ordering flips."""
    grid = np.linspace(0.05, 0.95, 19)
    for m in ("wilson", "jeffreys"):
        avg = np.mean([methods.interval_coverage(60, p, m) for p in grid])
        assert 0.93 < avg < 0.97
    near0 = np.linspace(0.002, 0.10, 200)
    for n in (30, 60):
        w = min(methods.interval_coverage(n, p, "wilson") for p in near0)
        j = min(methods.interval_coverage(n, p, "jeffreys") for p in near0)
        assert w < j < 0.95


def test_stability_flag():
    ml = pd.DataFrame({"measure_id": "X", "eligible_flag": 1, "exclusion_flag": 0,
                       "denominator_flag": [1] * 29, "numerator_flag": [1] * 10 + [0] * 19})
    s = methods.rate_summary(ml)
    assert s["reportable_flag"] == 0 and s["note"].startswith("NR")
    assert methods.rate_summary(pd.concat([ml, ml.head(1)]))["reportable_flag"] == 1
    assert all(type(s[k]) is float for k in ("rate", "ci_lo", "ci_hi"))


def test_systematic_sample_size_and_determinism():
    ids = pd.Series([f"M{i:04d}" for i in range(1000)])
    a, b = methods.systematic_sample(ids, 411, seed=1), methods.systematic_sample(ids, 411, seed=1)
    assert len(a) == 411 and a.is_unique and a.equals(b)
    assert len(methods.systematic_sample(ids.head(100), 411)) == 100        # small denominator: take everyone


def test_hybrid_rate_at_least_admin_rate_on_the_same_sample(synth, specs, vs, evaluated):
    spec = specs["FAKE_HBA1C_TEST"]
    h, smp = methods.hybrid_rate(spec, evaluated["FAKE_HBA1C_TEST"], synth["events"], vs, MY, sample_size=150, seed=2)
    admin_on_sample = evaluated["FAKE_HBA1C_TEST"].set_index("member_id").loc[smp["member_id"], "numerator_flag"].mean()
    assert h["sample_size"] == 150 and h["denominator_cnt"] == 150
    assert h["rate"] > admin_on_sample                                      # chart review finds more
    assert (smp["numerator_flag"] >= smp["chart_flag"]).all()


def test_hybrid_refused_for_admin_only_measure(synth, specs, vs, evaluated):
    with pytest.raises(ValueError, match="administrative-only"):
        methods.hybrid_rate(specs["FAKE_WCV"], evaluated["FAKE_WCV"], synth["events"], vs, MY)


def test_care_gap_list_is_exactly_the_open_gaps(synth, specs, vs, evaluated):
    for mid, spec in specs.items():
        ml = evaluated[mid]
        gaps = methods.care_gap_list(ml, spec, synth["events"], vs, MY)
        assert set(gaps["member_id"]) == set(ml.loc[ml["gap_open_flag"] == 1, "member_id"])
        excluded = set(ml.loc[ml["status"].isin(["EXCLUDED", "NOT_CE"]), "member_id"])
        assert not excluded & set(gaps["member_id"])                      # never outreach to non-gaps


def test_care_gap_notes(specs, vs):
    members = pd.DataFrame({"member_id": ["OLD", "CHART"], "sex_cd": "F", "birth_dt": pd.Timestamp("1960-01-01")})
    enr = pd.DataFrame({"member_id": ["OLD", "CHART"], "enroll_start_dt": pd.Timestamp("2023-01-01"),
                        "enroll_end_dt": pd.Timestamp("2025-12-31")})
    dm = [{"member_id": m, "event_dt": pd.Timestamp(d), "code_system": "ICD10CM", "code_cd": "E119", "source_cd": "CLAIM"}
          for m in ("OLD", "CHART") for d in ("2024-02-01", "2025-02-01")]
    a1c = [{"member_id": "OLD", "event_dt": pd.Timestamp("2024-06-01"), "code_system": "CPT", "code_cd": "83036", "source_cd": "CLAIM"},
           {"member_id": "CHART", "event_dt": pd.Timestamp("2025-06-01"), "code_system": "LOINC", "code_cd": "4548-4", "source_cd": "CHART"}]
    ev = pd.DataFrame(dm + a1c)
    spec = specs["FAKE_HBA1C_TEST"]
    ml = methods.evaluate_measure(spec, members, enr, ev, vs, MY)
    g = methods.care_gap_list(ml, spec, ev, vs, MY).set_index("member_id")
    assert g.loc["OLD", "note"].startswith("overdue") and g.loc["OLD", "last_service_dt"] == pd.Timestamp("2024-06-01")
    assert g.loc["CHART", "note"].startswith("documented in chart only")
    assert g.loc["OLD", "close_by_dt"] == pd.Timestamp("2025-12-31")


def test_core_set_loader_converts_percentages(tmp_path):
    p = tmp_path / "core_set.csv"
    pd.DataFrame({"State": ["AL", "AK", "AZ"], "Measure Abbreviation": ["BCS-AD"] * 3, "FFY": ["2026"] * 3,
                  "Population": ["Medicaid"] * 3, "Methodology": ["Administrative"] * 3,
                  "State Rate": ["45.5", "", "60.0"], "Notes": ["", "Not reported", ""]}).to_csv(p, index=False)
    b = data.load_core_set_rates(p)
    assert b["state_rate"].tolist() == pytest.approx([0.455, 0.60])            # blank (not reported) dropped
    assert b["core_set_year"].iloc[0] == 2026


def test_benchmark_position():
    bench = pd.DataFrame({"measure_cd": "BCS-AD", "state_rate": np.linspace(0.30, 0.70, 41)})
    b = methods.benchmark_position(0.60, bench, "BCS-AD")
    assert b["median"] == pytest.approx(0.50) and b["quartile"] == "Q4 top" and 70 < b["percentile"] < 80
    assert methods.benchmark_position(0.5, bench, "NOPE")["n_states"] == 0
