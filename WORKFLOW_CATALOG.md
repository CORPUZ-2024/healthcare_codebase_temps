# Workflow catalog

Every staff-level healthcare analytics workflow in scope, grouped by type, with its intent, typical data volume,
analytical stack, settings and where it lives in this repo: a **template** (`templates/tNN_*`) or a **glossary**
entry (`glossary/niche_workflows_glossary.html#GNN`). Generated from Section 1 of `IMPLEMENTATION_PLAN.txt`.

**Settings:** S1 PAYER · S2 VBC · S3 HOME (HCBS, caregiver programs, home health) · S4 HOSP · S5 PAC · S6 RX · S7 GOV · S8 HEOR  
**Volume:** XS < 10K rows · S 10K–1M · M 1M–50M · L > 50M (push to a warehouse)  
**Stacks:** AGG aggregation · INF inference · GLM regression · ML machine learning · CAU causal · SRV survival · LON longitudinal · ACT actuarial · RUL rule engines

## Type A — Data Foundation & Measurement Infrastructure

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| A1 | Claims + eligibility to analytic-ready tables | STANDARDIZE | M-L | AGG, RUL | all | [T00](templates/t00_claims_foundation) | ingest medical (837I/837P-shaped) + pharmacy claims > collapse adjustment/void versions > build member-month spans w/ partial-month proration > validate codes > analytic claim_line + member_month tables |
| A2 | Code-set normalization & validity (ICD-10-CM/PCS, CPT/HCPCS, NDC, POS, TOB, revenue code) | STANDARDIZE | S-M | RUL | all | [T00](templates/t00_claims_foundation) | format checks > effective-date validity > crosswalk to categories |
| A3 | Metric layer / single source of truth (shared metric definitions, dbt-style models + tests) | STANDARDIZE | M-L | AGG, RUL | S1 S2 S3 | [T14](templates/t14_metric_layer_dbt_style) | metric YAML > staging/intermediate/mart SQL > schema+data tests > metric catalog doc |
| A4 | Interop formats (X12 835/837, FHIR R4, HL7 v2) | STANDARDIZE | S-M | RUL | S1 S2 S4 | GLOSSARY ([G01](glossary/niche_workflows_glossary.html#G01), [G02](glossary/niche_workflows_glossary.html#G02)) |  |
| A5 | Record linkage / master patient index | STANDARDIZE | S-M | ML | S2 S3 S7 | GLOSSARY ([G03](glossary/niche_workflows_glossary.html#G03)) |  |

## Type B — Population Health & Risk

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| B1 | Utilization profiling (IP/ED/SNF/HH per 1,000, ALOS, 30-day readmission, runout-aware trend) | DESCRIBE, MONITOR | S-M | AGG | all | [T01](templates/t01_utilization_profiling) | claims > service-category grouper > per-1,000 rates > trend w/ runout flag > profile tables + charts |
| B2 | Cost-driver decomposition (service category, high-cost claimants, price vs. utilization vs. mix) | DESCRIBE, VALUE | S-M | AGG | S1 S2 | [T05](templates/t05_tcoc_pmpm_mlr) |  |
| B3 | Risk adjustment (CMS-HCC-style RAF: V24/V28 blend, normalization, coding-intensity, demographic factors) | MEASURE | S | RUL, AGG | S1 S2 | [T02](templates/t02_hcc_risk_adjustment) | dx rows > dx-to-HCC map > hierarchies > coefficients + demographics > normalize > member RAF + audit trail |
| B4 | Predictive risk (hospitalization, ED, adverse event) | PREDICT | S-M | GLM, ML | S1 S2 S3 | [T03](templates/t03_predictive_risk_stratification) | feature build (lookback) > label (lookahead) > temporal split > logistic baseline + GBM > calibration, AUC, PPV@top-k > scores |
| B5 | Acuity tiering & segmentation (rules + k-means, rising-risk) | PREDICT, DESCRIBE | S | ML, RUL | S2 S3 | [T03](templates/t03_predictive_risk_stratification) |  |
| B6 | Comorbidity & avoidable-utilization indices (Charlson/Elixhauser, AHRQ PQI, NYU ED, LACE) | MEASURE | S | RUL | S1 S2 S4 | GLOSSARY ([G05](glossary/niche_workflows_glossary.html#G05)–[G08](glossary/niche_workflows_glossary.html#G08)) |  |
| B7 | Provider attribution (plurality of visits) | STANDARDIZE | S | RUL | S2 | GLOSSARY ([G09](glossary/niche_workflows_glossary.html#G09)) |  |

## Type C — Quality & Clinical Measurement

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| C1 | HEDIS / Medicaid Core Set style measure rates (continuous enrollment, exclusions, Wilson CI, stability flag) | MEASURE | S-M | RUL, INF | S1 S2 S7 | [T04](templates/t04_quality_measures_care_gaps) |  |
| C2 | Member-level care-gap lists (open/closed gaps, outreach file) | MEASURE | S-M | RUL | S1 S2 S3 | [T04](templates/t04_quality_measures_care_gaps) |  |
| C3 | Part D adherence (PDC), opioid MME | MEASURE | S | RUL | S1 S6 | GLOSSARY ([G10](glossary/niche_workflows_glossary.html#G10), [G11](glossary/niche_workflows_glossary.html#G11)) |  |
| C4 | Clinical outcome definitions (30-day all-cause readmission, hospitalization-free days) | MEASURE | S | RUL | S2 S3 S4 | [T01](templates/t01_utilization_profiling) (defs), [T11](templates/t11_survival_time_to_event) (analysis) |  |

## Type D — Health Economics & Value-Based Care

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| D1 | Total cost of care / PMPM (proration, p99 truncation, runout) | MEASURE, VALUE | M | AGG | S1 S2 | [T05](templates/t05_tcoc_pmpm_mlr) |  |
| D2 | Medical loss ratio impact | VALUE | XS-S | AGG | S1 S2 | [T05](templates/t05_tcoc_pmpm_mlr) |  |
| D3 | Intervention ROI / cost offset / break-even / sensitivity | VALUE, EXPLAIN | S | AGG, CAU-lite | S2 S3 | [T06](templates/t06_roi_cost_offset) | cohort + matched comparison > pre/post PMPM > gross savings > program cost > ROI, break-even PMPM > tornado sensitivity |
| D4 | VBC contract modeling (shared savings/loss, MSR/MLR corridors, care-management fee, quality gate, stop-loss, scenario grid) | VALUE | XS-S | RUL, AGG | S1 S2 | [T07](templates/t07_vbc_contract_modeling) |  |
| D5 | Reconciliation replication (validate vs. MSSP ACO PUF) | VALUE, MEASURE | XS | AGG | S2 | [T07](templates/t07_vbc_contract_modeling) |  |
| D6 | Episode cost (trigger + look-back/forward windows) | MEASURE | M | RUL, AGG | S2 S4 S5 | [T05](templates/t05_tcoc_pmpm_mlr) (optional module) |  |
| D7 | Cost-effectiveness (ICER/QALY, Markov), budget impact | VALUE | XS | ACT-like | S8 | GLOSSARY ([G18](glossary/niche_workflows_glossary.html#G18), [G19](glossary/niche_workflows_glossary.html#G19)) |  |

## Type E — Actuarial & Financial Operations

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| E1 | Claims completion / IBNR (lag triangles, completion factors) | MEASURE | S-M | ACT | S1 S2 | [T08](templates/t08_claims_completion_forecast) |  |
| E2 | Trend & forecast (PMPM/utilization projection with intervals) | MONITOR, PREDICT | XS-S | ACT, GLM | S1 S2 | [T08](templates/t08_claims_completion_forecast) |  |
| E3 | Credibility weighting, stop-loss pricing, RADV sampling | VALUE | XS-S | ACT | S1 | GLOSSARY ([G22](glossary/niche_workflows_glossary.html#G22)–[G24](glossary/niche_workflows_glossary.html#G24)) |  |

## Type F — Evidence Generation / Outcomes Research

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| F1 | Study design (power/MDE, randomization, SAP skeleton) | DESIGN | XS | INF | S2 S3 S8 | [T09](templates/t09_study_design_power) |  |
| F2 | Quasi-experimental evaluation (DiD + parallel trends, PSM + SMD balance, interrupted time series) | EXPLAIN | S-M | CAU, GLM | S2 S3 S7 S8 | [T10](templates/t10_causal_impact_evaluation) |  |
| F3 | Time-to-event (Kaplan–Meier, log-rank, Cox PH, PH diagnostics) | EXPLAIN, PREDICT | S | SRV | S2 S3 S4 S8 | [T11](templates/t11_survival_time_to_event) |  |
| F4 | Patient-/caregiver-reported outcomes (instrument scoring, change scores, MCID responders, mixed models for repeated measures) | MEASURE, EXPLAIN | XS-S | LON, INF | S3 S8 | [T12](templates/t12_patient_reported_outcomes) |  |
| F5 | Advanced designs (synthetic control, RDD, IV, competing risks, multiple imputation, two-part cost models, bootstrap CIs) | EXPLAIN | S | CAU, SRV, GLM | S8 | GLOSSARY ([G13](glossary/niche_workflows_glossary.html#G13)–[G17](glossary/niche_workflows_glossary.html#G17), [G20](glossary/niche_workflows_glossary.html#G20), [G21](glossary/niche_workflows_glossary.html#G21)) |  |

## Type G — Performance, Operations & Cross-Functional Measurement

| ID | Workflow | Intent | Volume | Stack | Settings | Where | End-to-end |
|---|---|---|---|---|---|---|---|
| G1 | Provider / caregiver / panel benchmarking (O/E, empirical-Bayes shrinkage, funnel plot, external benchmark w/ vintage check) | MEASURE, MONITOR | S | INF, AGG | S1 S2 S3 | [T13](templates/t13_provider_benchmark_profiling) |  |
| G2 | Measurement framework & experiment readout (metric tree, guardrail metrics, A/B or pre/post readout) | DESIGN, MEASURE | XS-S | INF, RUL | S2 S3 | [T14](templates/t14_metric_layer_dbt_style) |  |
| G3 | Patient lifecycle funnel, prior-auth turnaround, denial / RCM KPIs | MONITOR | S | AGG, SRV-lite | S1 S2 S3 | [T15](templates/t15_ops_lifecycle_prior_auth) |  |
| G4 | EVV compliance, authorized-vs-delivered hours (home care) | MONITOR | S-M | RUL, AGG | S3 S7 | GLOSSARY ([G25](glossary/niche_workflows_glossary.html#G25)) |  |
| G5 | Medicaid churn / redetermination, dual-eligible identification | DESCRIBE | M | AGG | S1 S7 | GLOSSARY ([G26](glossary/niche_workflows_glossary.html#G26), [G27](glossary/niche_workflows_glossary.html#G27)) |  |
| G6 | AI/ML data readiness (drift PSI, subgroup calibration, model card) | MONITOR | S | INF | S1 S2 S3 | GLOSSARY ([G28](glossary/niche_workflows_glossary.html#G28), [G29](glossary/niche_workflows_glossary.html#G29)) |  |
| G7 | Publication privacy (small-cell suppression, de-identification) | STANDARDIZE | XS-S | RUL | all | GLOSSARY ([G30](glossary/niche_workflows_glossary.html#G30)) |  |
| G8 | Clinical NLP (note extraction with negation) | STANDARDIZE | S-M | ML | S2 S3 S4 | GLOSSARY ([G31](glossary/niche_workflows_glossary.html#G31)) |  |

