# Test data — t01 utilization profiling

## Default: synthetic (no download)
`utilization_profiling/data.py` (copied from t00) generates claims with inpatient stays, ED visits,
office/telehealth visits, HCBS personal-care days, frailty-driven 30-day readmissions and a
3-month runout tail.

## Public test file: CMS DE-SynPUF Inpatient Claims (optional)
* **Source:** CMS, *2008–2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)*
* **Page:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* **Codebook:** https://www.cms.gov/files/document/de-10-codebook.pdf-0
* **License / DUA:** none
* **Download:** Sample 1 → *Inpatient Claims* → `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv`
* **Run:** `python run.py --synpuf-ip data/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv`

| SynPUF column | Template column |
|---|---|
| DESYNPUF_ID | member_id |
| CLM_ID | claim_id |
| CLM_ADMSN_DT | admit_dt |
| NCH_BENE_DSCHRG_DT | discharge_dt |
| CLM_PMT_AMT | paid_amt |
| ADMTNG_ICD9_DGNS_CD | dx1_cd |
| CLM_DRG_CD | drg_cd |

**What to expect:** SynPUF is synthetic — relationships between records were perturbed to protect
privacy, so readmission rates and LOS will not match published Medicare statistics. Use it to test
that the code survives real file structure and volume, not to learn about Medicare.

Files in this folder are gitignored.
