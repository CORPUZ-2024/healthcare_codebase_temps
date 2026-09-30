# HANDOFF — continue the build in VS Code

Written 2026-09-30 when work moved from the cloud session to the local repo.
Read this first, then `docs/build/BUILD_STATUS.md`, `docs/build/BUILD_NOTES.md` (what was learned building each
template), `IMPLEMENTATION_PLAN.txt` (scope) and `docs/CONVENTIONS.md` (rules). Paths are relative to the repo root.

## Decisions already made (by Nicole)

| Topic | Decision |
|---|---|
| Stub style | **Full implementation** — no stubs, no xfail placeholders |
| Methods | Every step: **STANDARD** (industry-default library/function) + **ALTERNATIVE** (second-most-useful approach), with the caveat/trade-off that makes you switch, in the docstring and the README "Method choices" table |
| READMEs | One per template (template shape: see `t00`–`t05`), plus `data/README.md` naming the **public test dataset** with URL, license and column mapping |
| Orchestrator output | `orchestrator/run_status/` (LATEST.md / latest.json / latest.html committed; history gitignored) |
| Scope | 16 templates + glossary (31 entries), phases P0–P7 |
| Languages | Python + SQL twin (DuckDB) where it teaches something |

## Where things stand

* **Done and green (244 tests):** P0 (repo, orchestrator, CI, docs), P1 (t00, t01, t05), P2 (t02, t03, t04), P3 (t06, t07, t08), P4: t09.
* **Next:** P4 — t10, t11, t12; then P5–P7. Build notes for t00–t09: `BUILD_NOTES.md` (this folder).
* The glossary is still the **mock-up** (2 of 31 entries). Its build script is `glossary/build/build_glossary.py`
  (entries list + FULL dict; each copy block is a standalone `.py` in `glossary/build/`).

## Setup (Windows, VS Code)

```powershell
cd C:\Users\16502\Documents\GitHub\healthcare_codebase_temps
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python orchestrator\run_all_tests.py          # expect 10 templates PASS
python -m pytest orchestrator                  # expect 12 passed
```
If `lifelines` fails to build, upgrade tooling first: `python -m pip install -U pip setuptools wheel`.

## How each template is built (copy this pattern)

1. `templates/tNN_<name>/` with `<name>/` package (unique name), `config.py`, `data.py`, `methods.py`, `checks.py`.
2. Claims-based templates **copy** `data.py` / `prep.py` from t05 or t01 (header says "COPIED from …"). Never import across templates.
3. `run.py` with `demo()` (prints + writes `outputs/`) and `selftest()` (plain asserts, prints `PASS`/`FAIL` lines).
4. `pytest.ini` (`testpaths = tests <pkg>`, `--doctest-modules`), `tests/conftest.py` (sys.path insert + fixtures;
   **do not** `from conftest import …` — breaks repo-root runs), test files named `test_tNN_*.py`.
5. `template.yaml` (flat keys only), `requirements.txt`, `README.md`, `data/README.md`, `outputs/.gitkeep`, `data/.gitkeep`.
6. Return plain `float`s from functions that return dicts (doctests print `np.float64(...)` otherwise).
7. Pass no pandas `Period` columns to DuckDB (convert to `month_start` timestamps).
8. Run `python -m pytest` in the folder, then `python orchestrator/run_all_tests.py --only tNN`, then the full run.

## Remaining templates — spec

| ID | Package | STANDARD | ALTERNATIVE (trade-off) | Public test data |
|---|---|---|---|---|
| t04 | `quality_measures_care_gaps` | HEDIS-style measure engine from YAML specs (continuous enrollment w/ 1 allowable gap ≤ 45 days, denominator, exclusions e.g. hospice, numerator lookback) + **Wilson** CI + stability flag (n < 30) + member-level open-gap list | **Jeffreys** interval (better coverage near 0/1, Bayesian flavor) and **hybrid-style** numerator from supplemental data (captures more, not comparable to admin-only) | Medicaid & CHIP Core Set state rates (data.medicaid.gov) as benchmarks; NCQA free Core Set VSDs for value sets |
| t06 | `roi_cost_offset` | Matched pre/post (propensity or exact match) → **DiD on PMPM** → gross savings, net of program cost → ROI, break-even PMPM, tornado sensitivity | **Two-part / Gamma GLM** cost model or **bootstrap CI** on savings (skewed costs) | synthetic; MEPS for cost distributions (meps.ahrq.gov) |
| t07 | `vbc_contract_modeling` | Contract YAMLs (MSSP-like shared savings: benchmark, MSR/MLR corridor, shared-savings rate, quality gate, stop-loss cap; Medicaid sub-cap; CM fee + upside) → **deterministic reconciliation** that must close (ANL-015) + scenario grid | **Monte Carlo** of savings/loss distribution (trend + random variation) → probability of hitting MSR, expected payout | **MSSP ACO PUF** (data.cms.gov "Shared Savings Program Accountable Care Organizations") — reproduce published savings for a few ACOs |
| t08 | `claims_completion_forecast` | **Chain-ladder** lag triangle → completion factors → IBNR; **SARIMAX** (statsmodels) forecast of completed PMPM with intervals | **Bornhuetter-Ferguson** for immature months (blends expected PMPM) and **Holt-Winters/ETS** forecast | DE-SynPUF (incurred vs paid dates) |
| t09 | `study_design_power` | statsmodels power (two proportions, two means, **cluster design effect** 1+(m−1)ICC), MDE, stratified block randomization, SAP generator (Markdown) | **Simulation-based power** (any design/test, incl. DiD) | none needed |
| t10 | `causal_impact_evaluation` | **DiD** (statsmodels OLS, cluster-robust SE) + **event-study** pre-trend test; **PSM** (sklearn logistic score, 1:1 NN caliper 0.2 SD logit) + SMD balance table | **IPW/AIPW** (doubly robust) and **ITS** segmented regression (Newey-West SE) | synthetic panel with known effect |
| t11 | `survival_time_to_event` | **lifelines** Kaplan–Meier, log-rank, **Cox PH** + Schoenfeld PH check | **statsmodels PHReg** / discrete-time logistic hazard (when lifelines unavailable or for time-varying effects); note competing risks → glossary G16 | synthetic; SynPUF for admissions/death dates |
| t12 | `patient_reported_outcomes` | Instrument scoring from YAML (reverse items, prorating ≤ 1 missing item), change scores, MCID responders, **MMRM**/linear mixed model (statsmodels MixedLM) | **ANCOVA on change** (baseline-adjusted) and GEE; public-domain instruments only (e.g. PHQ-9 is free; Zarit Burden Interview is licensed → FAKE instrument) | MEPS SF-12/VR-12 items |
| t13 | `provider_benchmark_profiling` | Indirect standardization **O/E** + **empirical-Bayes beta-binomial shrinkage** + **funnel plot** (95%/99.8%) + vintage-matched external benchmark (ANL-013) | **Mixed-effects logistic** (random intercept per provider) reliability-adjusted rates | Care Compare hospital readmission/HRRP files; Medicare Physician & Other Practitioners PUF |
| t14 | `metric_layer_dbt_style` | `models/staging|intermediate|marts/*.sql` run in order on DuckDB by `dbt_lite.py`; `metrics.yaml`; schema tests (not_null, unique, accepted_values, relationships); **experiment readout** (two-proportion z / Welch t with guardrail metrics) | Metrics computed in **pandas** from the same YAML (for when no warehouse), and **CUPED** variance reduction for the readout | synthetic |
| t15 | `ops_lifecycle_prior_auth` | Referral → assessment → authorization → start of care → 90-day retention **funnel** with conversion and median days per stage; PA turnaround (72h expedited / 7-day standard targets per CMS-0057-F), denial and overturn rates; **KM** for time-to-start-of-care | **Cohort (monthly) conversion tables** instead of KM when censoring is light; denial root-cause Pareto | payer PA metrics posted under CMS-0057-F (verify availability) |

## P6 — glossary (31 entries)

Entries, use cases and exclusion factors are already listed in `glossary/build/build_glossary.py` (`E` list).
For each entry: write `glossary/build/gNN_<slug>.py` (imports, fully documented function(s), `if __name__ == "__main__":`
self-test printing PASS/FAIL), add it to `FULL` with io/caveats/mistakes/sources, rebuild. Add
`glossary/test_glossary_blocks.py` that executes every block (orchestrator runs it only with an opt-in flag).
Remove the MOCK-UP banner when all 31 are full.

## P7 — finish

Final `IMPLEMENTATION_PLAN.txt` (replace the draft: simplified package layout, decisions above), `WORKFLOW_CATALOG.md`,
root README template table, full orchestrator run on Windows, tag `v1.0`.
