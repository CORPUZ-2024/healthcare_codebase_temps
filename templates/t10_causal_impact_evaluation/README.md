# t10 — Causal impact evaluation

"Did the program cause the change?" This template covers the three designs a payer or health
system evaluation uses most:
* **difference-in-differences** with an event study, when some units adopt and others don't
* **propensity-score methods**, when members are selected into a program
* **interrupted time series**, when a policy hits everyone at once

Every generator carries a known effect, so each estimator is tested against the truth. The tests also
show the specific ways each design fails.

| | |
|---|---|
| **Workflow type** | F — Evidence (F2 quasi-experimental evaluation: DiD + parallel trends, PSM + SMD balance, ITS) |
| **Intent** | EXPLAIN |
| **Volume** | S. Demo: 40 practices × 36 months; 6,000 members; one 60-month series. |
| **Stack** | statsmodels + scikit-learn |
| **JD link** | "causal inference for program evaluation", "quasi-experimental methods" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 22 tests (incl. doctest); ~10 s, several are small simulations
```

## Method choices

| Design | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Units adopt at one date | `did_twfe`: y ~ treated×post + unit FE + time FE, SEs clustered by unit; `event_study` for pre-trends | `its_segmented` (one series, no control) | no comparison group exists (statewide policy). ITS can't separate the policy from anything else that changed that month. |
| Members selected into a program | `psm_att`: 1:1 NN on logit PS (scikit-learn), caliper 0.2 SD → **ATT among matched** + `smd_table` | `ipw` (ATE or ATT), `aipw_ate` (doubly robust) | you need the ATE, don't want to discard unmatched participants, or want protection against a misspecified model (AIPW is right if *either* the PS or the outcome model is right). |
| ITS standard errors | `se_method="glsar"`: AR(1) feasible GLS | Newey-West HAC, naive OLS (reported alongside) | see below. **This reverses the textbook default** on purpose. |

## Findings the tests lock in

| Finding | Evidence |
|---|---|
| Naive treated − untreated has the **wrong sign** (+4.9 vs. true ATE −3.0) | `test_naive_difference_has_the_wrong_sign` |
| ATT ≠ ATE when sicker members are referred and benefit more (−3.96 vs. −3.00) | `test_att_differs_from_ate` |
| PSM matches 87% of participants; the rest are the sickest, so PSM estimates the ATT **among matched** (−3.71 true, −3.51 estimated) | `test_psm_recovers_att_of_matched_and_balances` |
| AIPW is doubly robust: with a wrong PS model (age only) and a right outcome model it is within 0.4 of the truth, while IPW with the same PS is > 3× further off | `test_aipw_double_robustness` |
| An event study with **one coefficient per month** (23 leads, 40 clusters) rejects parallel trends that hold (~85% of draws); binning to ±6 months gives ~5% false alarms and ~94% power against a 0.08/month drift | `test_event_study_unbinned_overrejects`, `test_event_study_binned_false_alarm_rate_and_power` |
| A pre-trend violation biases DiD toward 0 (−0.45 vs. −2.0). **The pre-test can miss it**: it does in the demo draw (p = 0.18) | `test_pre_trend_violation_biases_did`; demo output |
| ITS on 60 autocorrelated months: Newey-West 95% intervals cover the truth only ~73–77% of the time (no better than OLS); GLSAR ~90% | `test_its_glsar_covers_better_than_hac` (coverage study of 300 draws during the build) |

## Demo results (seed 10)

* **DiD:** −1.89 (CI −2.19 to −1.59) vs. true −2.0. Event-study pre-trend p = 0.32.
* **PSM** −3.51 (ATT matched, true −3.71) · **IPW-ATE** −3.28 (CI −4.29 to −2.26) · **AIPW** −3.01
  (CI −3.54 to −2.48) vs. true ATE −3.00 · **IPW-ATT** −5.30 (wide: weights on controls that resemble the sickest participants).
* **ITS:** level −2.89 (CI −4.81 to −0.98) vs. true −4.0. Slope CI misses the truth on this draw (GLSAR ~90% coverage).
  SE of level: OLS 0.66, Newey-West 0.63, GLSAR 0.98 (Durbin-Watson 0.86).

## Test data

Synthetic only (`data.py`: `cross_section`, `practice_panel`, `its_series`), because causal methods can only be
*tested* where the truth is known. Real-data analogues: see [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t10_designs.py` | 2×2 DiD known answer; DiD coverage across draws; naive biases; event-study false-alarm rate, power, binning, shape; pre-trend violation bias; ATT vs. ATE; PSM balance and coverage; IPW/AIPW coverage and efficiency; AIPW double robustness and coverage across draws; ITS unbiased on average; GLSAR vs. HAC coverage; ITS counterfactual and AR detection |
| `test_t10_checks.py` | every check fires correctly |

## Caveats

* **No unmeasured confounding** is the untestable assumption behind PSM/IPW/AIPW. Balance on measured
  covariates says nothing about unmeasured ones. Run a sensitivity analysis (E-value) for real reports.
* **Staggered adoption:** `did_twfe` assumes one adoption date. With staggered dates and changing effects, TWFE
  is biased (Goodman-Bacon); use Callaway-Sant'Anna or Sun-Abraham (not implemented).
* **PSM SEs** ignore PS estimation and matching uncertainty (Abadie-Imbens is more exact; the naive bootstrap is invalid for NN matching).
* **IPW SEs** are robust WLS SEs that ignore PS estimation (usually conservative for the ATE).
* Synthetic control, regression discontinuity and instrumental variables are glossary entries (G13–G15).

## Explain it to Finance

"Comparing participants with everyone else makes the program look harmful, because we refer sicker
people. After matching each participant to a similar non-participant, the program lowers the
utilization index by about 3.5 points for the members it could be matched on. The doubly robust
estimate for the whole population is about 3.0. For the practice workflow, adopting practices fell
about 1.9 ED visits per 1,000 member-months more than similar practices that didn't adopt, and before
the rollout both groups were moving together. The statewide policy shows an immediate drop of about
3 visits per 1,000, but with no comparison group we can't rule out that something else changed that month."

## Files

```
causal_impact_evaluation/  config.py  data.py  methods.py  checks.py
tests/  conftest.py  test_t10_designs.py  test_t10_checks.py
```
