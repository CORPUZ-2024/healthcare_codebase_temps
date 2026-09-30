# Test data — t08 claims completion and forecast

## Default: synthetic (no download)
`claims_completion_forecast/data.py` → `generate()` builds incurred × paid cells from a known
ultimate cost per incurred month (the `truth` table). Only synthetic data can score IBNR methods,
because real ultimates are known only after full runout.

## Your own claim lines
`data.load_claim_lines(path, incurred_col, paid_col, amount_col)` works with any extract that has a
service (incurred) date and a paid (or check / processed) date, for example t00's synthetic claims
(`svc_from_dt`, `paid_dt`, `paid_amt`). Rows paid before the service date are dropped and counted.

## Public claims: CMS DE-SynPUF (check before relying on it)
* **Page:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
* **License:** public, no DUA.
* SynPUF claim files carry claim FROM / THRU (service) dates. Check the file layout in the codebook
  for a processing / payment date **before** using SynPUF for lag triangles. If there is none, SynPUF
  can't show payment lag, and the synthetic generator is the right test source.
* SynPUF is 2008–2010 and synthetic, so use it for volume and plumbing, not for real completion patterns.

Files in this folder are gitignored.
