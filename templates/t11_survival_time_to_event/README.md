# t11 — Survival and time-to-event analysis

"How long until readmission (or nursing-home placement, or death), and what changes that?" Members
are followed for different lengths of time: they disenroll, die, or the study ends. Survival methods
use each member's follow-up up to that point. This template covers:
* Kaplan–Meier curves and the log-rank test
* the Cox model with proportional-hazards diagnostics, plus two alternatives for when those diagnostics fail
* restricted mean survival time (RMST), the easiest summary to explain
* the most damaging design error in this area: **immortal time**

| | |
|---|---|
| **Workflow type** | F — Evidence (F3 time-to-event); C4 outcome definitions (readmission) |
| **Intent** | EXPLAIN, PREDICT |
| **Volume** | S. Demo: 3,000 discharged members followed ≤ 365 days; 4,000-member immortal-time cohort. |
| **Stack** | lifelines + statsmodels |
| **JD link** | "time-to-event and survival analysis", "outcome studies" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 14 tests, ~5 s
```

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Model | `cox_ph`: lifelines `CoxPHFitter` (Efron ties) + `ph_test` (Schoenfeld, rank time) | `cox_statsmodels`: statsmodels `PHReg` | lifelines isn't available or allowed. Same model: HRs match to < 1e-7 here. |
| Non-proportional hazards | — | `discrete_time_hazard`: pooled logistic on 30-day periods with period-specific effects | the PH test flags a covariate (here high acuity: HR 2.5 for the first 30 days, then 1.2). The Cox HR (1.41) is an average that depends on follow-up length; the discrete-time model recovers **2.40 → 1.18**. |
| Summary for non-statisticians | HR | `rmst_difference`: readmission-free days to day 180 | always show it next to the HR. It needs no PH assumption and reads as "program members spent 10.4 more days out of the hospital". |
| Exposure that starts after day 0 | — | `cox_time_varying`: counting-process rows, exposure switches on at the start date | **always**, when the program begins after the index date. |

## Findings the tests lock in

| Finding | Evidence |
|---|---|
| KM matches a hand calculation; the naive "events / members" share understates cumulative incidence | `test_km_hand_calculation`, `test_naive_share_understates_when_censoring` |
| Cox recovers the generator's HRs on average (20 cohorts: program 0.75, frailty e^0.5, age 1.16/10y). One cohort can land 1.7 SE away (seed 11: 0.67) | `test_cox_recovers_true_hrs_on_average` |
| The PH check flags **only** acuity once Bonferroni-adjusted. Unadjusted, frailty (truly PH) also "fails" at p = 0.024 | `test_ph_test_flags_only_the_nonproportional_covariate` |
| **Immortal time:** a program that does nothing (true HR 1.0) gets **HR 0.75** in an "ever vs. never" analysis; time-varying exposure gives 1.02 (CI 0.91–1.14) and recovers a real HR 0.7 | `test_immortal_time_bias_and_fix`, `test_time_varying_cox_recovers_real_effect` |

## Demo results (seed 11)

* 90-day readmission (KM): usual care 17.8%, program 11.9%. Log-rank p = 4.5e-9.
* Cox HR program 0.67 (CI 0.58–0.76); frailty 1.63 per SD; high acuity 1.41 (PH violated, p = 0.003).
* RMST to day 180: 160.2 vs. 149.9 readmission-free days (+10.4).
* Deaths (6.8%) are censored in the readmission analysis. The HRs are cause-specific (`SRV-004`, glossary G16).

## Test data

* **Synthetic (default):** `data.generate` (piecewise Weibull hazards with known HRs, a non-PH covariate, competing
  death, disenrollment and staggered study end) and `data.immortal_time_cohort` (delayed program start).
* **Public:** CMS DE-SynPUF inpatient claims plus beneficiary death dates. See [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t11_survival.py` | KM by hand; naive share bias; KM monotone and program lower; log-rank; Cox recovers truth across cohorts; PH test and Bonferroni check; PHReg = lifelines; discrete-time recovers the changing acuity effect; person-period expansion; RMST; immortal-time bias and fix; time-varying Cox recovers a real HR; counting-process rows preserve person-time and events; each check |

## Caveats

* **Competing risks:** censoring deaths gives cause-specific hazards. 1 − KM then *overstates* the absolute risk
  of readmission. Use Aalen–Johansen cumulative incidence (glossary G16) for absolute risks.
* **Informative censoring:** disenrollment related to health (e.g. moving to a nursing home) biases KM and Cox.
  Compare characteristics of censored and uncensored members.
* **Time zero** must be the same event for everyone (discharge date here). Eligibility and exposure are
  defined at time zero; anything later is time-varying.
* HRs are rate ratios, not risk ratios. Report absolute risks (KM or cumulative incidence) and RMST too.

## Explain it to Clinical and Finance

"Among members discharged home, 18% of usual-care members were readmitted within 90 days, against 12% in
the transitional-care program. After adjusting for age, frailty, caregiver support and acuity, program members
were readmitted at about two-thirds the rate. Over six months they spent about 10 more days out of the
hospital. One caution: members who started the program weeks after discharge must be counted as unexposed
until they started. Counted the wrong way, even a program that does nothing looks 25% better."

## Files

```
survival_time_to_event/  config.py  data.py  methods.py  checks.py
tests/  conftest.py  test_t11_survival.py
```
