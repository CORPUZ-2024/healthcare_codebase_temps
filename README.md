# healthcare_codebase_temps

A generalist toolkit of **standalone, tested codebase templates** for staff-level healthcare
analytics — population health and risk, health economics and value-based care, clinical outcomes
and evidence generation, and cross-functional measurement. Each template runs on seeded synthetic
data out of the box, documents a public test dataset, and implements a **standard** method plus an
**alternative** with the trade-off that makes you switch.

> Build status: [`docs/build/BUILD_STATUS.md`](docs/build/BUILD_STATUS.md). Continuing work: [`docs/build/HANDOFF.md`](docs/build/HANDOFF.md).
> Build notes (every template, bugs found, decisions): [`docs/build/BUILD_NOTES.md`](docs/build/BUILD_NOTES.md).
> Scope: [`IMPLEMENTATION_PLAN.txt`](IMPLEMENTATION_PLAN.txt).

## Quick start (Windows / VS Code)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
run_all_tests.bat                      # every template's tests -> orchestrator\run_status\LATEST.md
cd templates\t05_tcoc_pmpm_mlr
python run.py                          # demo -> outputs\
python run.py --selftest               # checks without pytest
```

## Templates

| ID | Template | Area | Standard vs. alternative (headline) | Status |
|---|---|---|---|---|
| t00 | [claims_foundation](templates/t00_claims_foundation) | Data foundation | latest-version vs. net-of-deltas; daily vs. mid-month member-months | ✅ |
| t01 | [utilization_profiling](templates/t01_utilization_profiling) | Population health | exact Poisson vs. member-bootstrap CI; readmits per index vs. per 1,000 | ✅ |
| t02 | [hcc_risk_adjustment](templates/t02_hcc_risk_adjustment) | Risk adjustment | published (FAKE) weights vs. re-estimated weights | ✅ |
| t03 | [predictive_risk_stratification](templates/t03_predictive_risk_stratification) | Risk | logistic vs. gradient boosting; capacity tiers vs. k-means | ✅ |
| t04 | [quality_measures_care_gaps](templates/t04_quality_measures_care_gaps) | Quality | Wilson vs. Jeffreys; admin vs. hybrid | ✅ |
| t05 | [tcoc_pmpm_mlr](templates/t05_tcoc_pmpm_mlr) | Health economics | ratio-of-sums vs. mean-of-members PMPM; regulatory vs. simple MLR | ✅ |
| t06 | [roi_cost_offset](templates/t06_roi_cost_offset) | Health economics | matched DiD vs. two-part / bootstrap | ✅ |
| t07 | [vbc_contract_modeling](templates/t07_vbc_contract_modeling) | VBC | deterministic reconciliation vs. Monte Carlo | ✅ |
| t08 | [claims_completion_forecast](templates/t08_claims_completion_forecast) | Actuarial | chain-ladder + SARIMAX vs. Bornhuetter-Ferguson + ETS | ✅ |
| t09 | [study_design_power](templates/t09_study_design_power) | Evidence | analytic vs. simulation power | ✅ |
| t10 | [causal_impact_evaluation](templates/t10_causal_impact_evaluation) | Evidence | DiD + event study / PSM vs. AIPW / ITS (GLSAR) | ✅ |
| t11 | [survival_time_to_event](templates/t11_survival_time_to_event) | Evidence | lifelines KM/Cox vs. statsmodels PHReg / discrete-time hazard | ✅ |
| t12 | [patient_reported_outcomes](templates/t12_patient_reported_outcomes) | Evidence | mixed model (MMRM-style) vs. ANCOVA / GEE | ✅ |
| t13 | [provider_benchmark_profiling](templates/t13_provider_benchmark_profiling) | Performance | O/E + EB shrinkage vs. mixed model | ✅ |
| t14 | [metric_layer_dbt_style](templates/t14_metric_layer_dbt_style) | Cross-functional | SQL models on DuckDB vs. pandas; t-test vs. CUPED | ✅ |
| t15 | [ops_lifecycle_prior_auth](templates/t15_ops_lifecycle_prior_auth) | Operations | KM time-to-start vs. cohort tables; PA metrics + Pareto | ✅ |

Niche workflows that are snippets rather than templates: [`glossary/niche_workflows_glossary.html`](glossary/niche_workflows_glossary.html) (mock-up; 2 of 31 entries populated).

## Rules that keep it modular

* Templates never import each other or the orchestrator; shared helpers are **copied**.
* The orchestrator (`orchestrator/`) discovers templates by folder name and runs each in its own
  process; it imports only the standard library. Its tests enforce both rules.
* Only generated data is committed. Public files go in each template's `data/` (gitignored).

See [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md), [`docs/DEBUG_WITHOUT_AI.md`](docs/DEBUG_WITHOUT_AI.md),
[`docs/ADDING_A_TEMPLATE.md`](docs/ADDING_A_TEMPLATE.md).
