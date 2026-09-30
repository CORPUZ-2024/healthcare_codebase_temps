"""Known answers, recovery of the true hazard ratios, and the design errors survival methods prevent."""
import numpy as np
import pandas as pd
import pytest

from survival_time_to_event import checks, data, methods

TOY = pd.DataFrame({"member_id": list("ABCD"), "duration_days": [2, 3, 4, 5], "event_flag": [1, 0, 1, 0]})


def test_km_hand_calculation():
    km = methods.km_table(TOY, None, (2, 4))
    assert km["cum_incidence"].tolist() == pytest.approx([0.25, 0.625])
    assert km["at_risk"].tolist() == [4, 2]


def test_naive_share_understates_when_censoring():
    assert methods.naive_event_share(TOY, 4) == 0.5 < methods.km_table(TOY, None, (4,))["cum_incidence"].iloc[0]


def test_km_monotone_and_program_lower(cohort, cfg):
    km = methods.km_table(cohort, "program_flag", cfg.report_days)
    for _, g in km.groupby("group"):
        assert g["cum_incidence"].is_monotonic_increasing
    at = km.pivot(index="day", columns="group", values="cum_incidence")
    assert (at[1] < at[0]).all()


def test_logrank_detects_program(cohort):
    assert methods.logrank(cohort)["p_value"] < 1e-4


def test_cox_recovers_true_hrs_on_average():
    """Across 20 cohorts the geometric-mean HRs match the generator (single cohorts vary: seed 11's program
    HR is 0.67, a 1.7-SE draw)."""
    hr = {"program_flag": [], "frailty_z": [], "age": []}
    for s in range(20):
        _, t = methods.cox_ph(data.generate(seed=100 + s))
        t = t.set_index("covariate")
        for k in hr:
            hr[k].append(np.log(t.loc[k, "hr"]))
    assert np.exp(np.mean(hr["program_flag"])) == pytest.approx(0.75, abs=0.03)
    assert np.exp(np.mean(hr["frailty_z"])) == pytest.approx(np.exp(0.5), abs=0.05)
    assert np.exp(np.mean(hr["age"])) == pytest.approx(1.16 ** 0.1, abs=0.003)


def test_ph_test_flags_only_the_nonproportional_covariate(cox, cohort):
    ph = methods.ph_test(cox[0], cohort)
    f = checks.check_proportional_hazards(ph)
    assert f and f[0].message.endswith("['high_acuity_flag']")
    assert ph.set_index("covariate").loc["program_flag", "p_value"] > 0.05


def test_statsmodels_phreg_matches_lifelines(cohort, cox):
    assert methods.cox_statsmodels(cohort)["hr"].to_numpy() == pytest.approx(cox[1]["hr"].to_numpy(), rel=1e-4)


def test_discrete_time_hazard_recovers_changing_acuity_effect(cohort, cfg):
    dt = methods.discrete_time_hazard(cohort, interval_days=cfg.interval_days, max_days=cfg.max_follow_days).set_index("covariate")
    early, late = dt.loc["high_acuity_flag_first_period"], dt.loc["high_acuity_flag_later"]
    assert early["ci_lo"] <= cfg.acuity_hr_early <= early["ci_hi"]
    assert late["ci_lo"] <= cfg.acuity_hr_late <= late["ci_hi"]
    assert early["odds_ratio"] > 1.5 * late["odds_ratio"]


def test_person_period_rows(cfg):
    pp = methods.person_period(TOY.assign(x=1), interval_days=2, max_days=6)
    assert pp.groupby("member_id").size().to_dict() == {"A": 1, "B": 2, "C": 2, "D": 3}
    assert pp["event_in_period"].sum() == TOY["event_flag"].sum()


def test_rmst_positive_for_program(cohort, cfg):
    r = methods.rmst_difference(cohort, cfg.rmst_horizon)
    assert 0 < r["difference_days"] < cfg.rmst_horizon and r["rmst_treated"] <= cfg.rmst_horizon


def test_immortal_time_bias_and_fix():
    it = data.immortal_time_cohort(n=4_000, hr_program=1.0, seed=3)
    naive = methods.cox_naive_ever_exposed(it)
    tv = methods.cox_time_varying(it)
    assert naive < 0.85                                      # a useless program looks protective
    assert tv["ci_lo"] <= 1.0 <= tv["ci_hi"]


def test_time_varying_cox_recovers_real_effect():
    tv = methods.cox_time_varying(data.immortal_time_cohort(n=6_000, hr_program=0.7, seed=4))
    assert tv["ci_lo"] <= 0.7 <= tv["ci_hi"]


def test_counting_process_person_time_is_preserved():
    it = data.immortal_time_cohort(n=500, seed=5)
    cp = methods.to_counting_process(it)
    person_time = (cp["stop"] - cp["start"]).groupby(cp["member_id"]).sum()
    assert (person_time == it.set_index("member_id")["duration_days"]).all()
    assert cp.groupby("member_id")["event_flag"].sum().eq(it.set_index("member_id")["event_flag"]).all()


def test_checks():
    ph = pd.DataFrame({"covariate": ["a", "b", "c", "d", "e"], "p_value": [0.02, 0.5, 0.004, 0.9, 0.3]})
    assert checks.check_proportional_hazards(ph)[0].n_rows == 1          # 0.02 is not < 0.01
    heavy = pd.DataFrame({"duration_days": [10] * 8 + [400] * 2, "event_flag": [0] * 10})
    assert checks.check_censoring(heavy, 180)[0].check_id == "SRV-002"
    assert checks.check_events_per_variable(30, 5)[0].severity == "error" and not checks.check_events_per_variable(200, 5)
    assert checks.check_competing_events(pd.DataFrame({"event_cd": [2] * 10 + [0] * 90}))
    assert checks.check_immortal_time(pd.DataFrame({"start_day": [0, 12, np.nan]}))[0].n_rows == 1
