# Test data — t03 predictive risk stratification

## Default: synthetic (no download)
`predictive_risk_stratification/data.py` builds a member × cohort-year feature table (12-month
lookback) with a 6-month hospitalization label. The true model includes a threshold (3+ ED
visits) and an interaction (heart failure × lives alone); the 2024 cohort has a lower baseline
rate and more telehealth, so calibration drift shows up out-of-time.

## Public test data: CMS DE-SynPUF (optional)
* **Page:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* **Codebook:** https://www.cms.gov/files/document/de-10-codebook.pdf-0
* **Recipe:** Beneficiary Summary 2008 → age (`BENE_BIRTH_DT`), sex (`BENE_SEX_IDENT_CD`),
  chronic flags (`SP_CHF`, `SP_COPD`, `SP_DIABETES`, … coded 1 = yes, 2 = no); Inpatient Claims
  2008 → `prior_ip_cnt`; Outpatient claims with revenue centers 0450–0459 → `prior_ed_cnt`;
  label = any inpatient `CLM_ADMSN_DT` in the first 6 months of 2009. Train on 2008→2009,
  test on 2009→2010.
* **Caveat:** SynPUF perturbs relationships between variables, so expect a weaker AUC than on
  real claims. Use it to test the pipeline end to end.

Files in this folder are gitignored.
