# Test data — t11 survival / time-to-event

## Default: synthetic (no download)
`survival_time_to_event/data.py`:
* `generate()`: time to readmission after discharge, with known hazard ratios, one non-proportional covariate,
  competing death, disenrollment and staggered study end (`event_cd` 0/1/2).
* `immortal_time_cohort()`: program starting 0–45 days after discharge; `hr_program` sets the true effect.

## Public test file: CMS DE-SynPUF (optional)
* **Page:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* **License:** public, no DUA.
* **Build a readmission cohort:** from the Inpatient Claims file, take the admission and discharge dates
  (`CLM_ADMSN_DT`, `NCH_BENE_DSCHRG_DT`; verify the names in the codebook). Index = each discharge. Event = next
  admission. Censor at the end of the file year or at death (`BENE_DEATH_DT` in the Beneficiary Summary files).
  Then `duration_days` = event or censoring date − discharge date.
* **Caveat:** SynPUF is synthetic and 2008–2010. Use it to exercise the pipeline, not for real readmission rates.

Files in this folder are gitignored.
