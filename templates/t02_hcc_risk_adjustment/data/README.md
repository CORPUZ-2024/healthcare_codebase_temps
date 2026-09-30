# Test data — t02 HCC risk adjustment

## Default: synthetic (no download)
`hcc_risk_adjustment/data.py` generates members (age, sex, dual, disabled), ICD-10 diagnoses with
the **source** of each (face-to-face professional, outpatient, inpatient, lab, DME), drug-class
evidence, and next-year cost generated from each member's *true* conditions. ~20% of true
conditions are coded only on lab/DME claims or not at all — these become the suspect-gap list.

## Reference tables (in `../reference/`, all FAKE)
The mapping, hierarchy and coefficient tables are invented. Real tables:
* **CMS-HCC (Medicare Advantage)** model software, ICD-10 mappings, coefficients —
  https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
  Normalization factors and the coding-pattern adjustment: the annual **Rate Announcement** on the
  same site.
* **Medicaid:** CDPS+Rx — https://hwstudy.ucsd.edu/cdps/ (or the state's own model).

## Public test file (optional): CMS DE-SynPUF
* https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* Use Beneficiary Summary (age from `BENE_BIRTH_DT`, sex `BENE_SEX_IDENT_CD`, dual from
  `BENE_HMO_CVRAGE_TOT_MONS`/state buy-in months) and carrier/outpatient/inpatient diagnosis columns.
* **Caveat:** SynPUF diagnoses are **ICD-9**. They will not map with ICD-10 tables — you will see
  `RA-020` (nearly all members unmapped). That is the expected lesson; use SynPUF to test volume
  and joins, not scores.

Files in this folder are gitignored.
