# Test data — t00 claims foundation

## Default: synthetic (no download)
`claims_foundation/data.py` generates enrollment, providers, medical and pharmacy claims with a
fixed seed. It includes the traps this template teaches: claim adjustments and voids, overlapping
enrollment spans, multi-location NPIs, pharmacy reversals and a claims-runout tail.

## Public test file: CMS DE-SynPUF (optional)
* **Source:** CMS, *2008–2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)*
* **Page:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* **Codebook:** https://www.cms.gov/files/document/de-10-codebook.pdf-0
* **License / DUA:** none — synthetic, free to download
* **Download:** Sample 1 → *Carrier Claims* (1A or 1B, zip → CSV). Save as
  `data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv`.
* **Run:** `python run.py --synpuf data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv`

| SynPUF column | Template column |
|---|---|
| DESYNPUF_ID | member_id |
| CLM_ID | claim_id |
| CLM_FROM_DT / CLM_THRU_DT | svc_from_dt / svc_to_dt |
| ICD9_DGNS_CD_1 / _2 | dx1_cd / dx2_cd |
| PRF_PHYSN_NPI_1 | npi_id |
| HCPCS_CD_1 | hcpcs_cd |
| LINE_NCH_PMT_AMT_1 / LINE_ALOWD_CHRG_AMT_1 | paid_amt / allowed_amt |

**What to expect:** SynPUF diagnoses are ICD-9 (2008–2010), so `CLN-052` fires on almost every row —
that is the lesson (code-system vs. service-date mismatch). SynPUF NPIs are synthetic and many fail
the Luhn check. The file has no adjustment versions, so version-collapse logic is a no-op on it.

Files in this folder are gitignored — never commit downloaded data.
