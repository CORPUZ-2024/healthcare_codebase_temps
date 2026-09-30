# healthcare_codebase_temps

A generalist toolkit of **standalone, tested codebase templates** for staff-level healthcare
analytics — population health and risk, health economics and value-based care, clinical outcomes
and evidence generation, and cross-functional measurement. Each template runs on seeded synthetic
data out of the box, documents a public test dataset, and implements a **standard** method plus an
**alternative** with the trade-off that makes you switch.

> Build status: see [`BUILD_STATUS.md`](BUILD_STATUS.md). Continuing work: [`HANDOFF.md`](HANDOFF.md).
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
| t06 | roi_cost_offset | Health economics | matched DiD vs. two-part / bootstrap | ⏳ |
| t07 | vbc_contract_modeling | VBC | deterministic reconciliation vs. Monte Carlo | ⏳ |
| t08 | claims_completion_forecast | Actuarial | chain-ladder + SARIMAX vs. Bornhuetter-Ferguson + ETS | ⏳ |
| t09 | study_design_power | Evidence | analytic vs. simulation power | ⏳ |
| t10 | causal_impact_evaluation | Evidence | DiD / PSM vs. AIPW / ITS | ⏳ |
| t11 | survival_time_to_event | Evidence | lifelines KM/Cox vs. statsmodels PHReg | ⏳ |
| t12 | patient_reported_outcomes | Evidence | MMRM vs. ANCOVA / GEE | ⏳ |
| t13 | provider_benchmark_profiling | Performance | O/E + EB shrinkage vs. mixed model | ⏳ |
| t14 | metric_layer_dbt_style | Cross-functional | SQL models on DuckDB vs. pandas; t-test vs. CUPED | ⏳ |
| t15 | ops_lifecycle_prior_auth | Operations | KM funnel vs. cohort tables | ⏳ |

Niche workflows that are snippets rather than templates: [`glossary/niche_workflows_glossary.html`](glossary/niche_workflows_glossary.html) (mock-up; 2 of 31 entries populated).

## Rules that keep it modular

* Templates never import each other or the orchestrator; shared helpers are **copied**.
* The orchestrator (`orchestrator/`) discovers templates by folder name and runs each in its own
  process; it imports only the standard library. Its tests enforce both rules.
* Only generated data is committed. Public files go in each template's `data/` (gitignored).

See [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md), [`docs/DEBUG_WITHOUT_AI.md`](docs/DEBUG_WITHOUT_AI.md),
[`docs/ADDING_A_TEMPLATE.md`](docs/ADDING_A_TEMPLATE.md).
