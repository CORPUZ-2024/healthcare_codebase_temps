# Test data — t05 TCOC / PMPM / MLR

## Default: synthetic (no download)
`tcoc_pmpm_mlr/data.py` (copied from t00) generates enrollment and claims; `generate_premium`
adds a **FAKE** premium table (rates calibrated so the synthetic MLR lands near 85–90%).

## Public benchmark: CMS Medicare Geographic Variation PUF (optional)
* **Source:** CMS, *Medicare Geographic Variation — by National, State & County*
* **Page:** https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county
* **License:** public, no DUA
* **Use:** compare your per-capita (per member per year ÷ 12) FFS spend to state/county Medicare
  FFS averages. Loader: `data.load_geovar_puf(path)` (selects the per-capita payment columns by
  name pattern because column names change between releases).
* **Caveat:** GeoVar is Medicare FFS only (Parts A/B), standardized or actual payments — not
  comparable to Medicaid managed care or to Part D. Use it for order-of-magnitude checks.

## Public test file for the claims path: CMS DE-SynPUF
Same as t00 — see `../t00_claims_foundation/data/README.md`.

Files in this folder are gitignored.
