# Test data — t13 provider benchmarking

## Default: synthetic (no download)
`provider_benchmark_profiling/data.py` → `generate()`: patients nested in 80 providers with known quality
effects and different case mix. `generate_fake_hrrp()` builds a FAKE national file with the public HRRP headers.

## Public benchmark: CMS Hospital Readmissions Reduction Program (HRRP)
* **Source:** CMS Provider Data Catalog, *Hospital Readmissions Reduction Program*: https://data.cms.gov/provider-data/
  (search "Hospital Readmissions Reduction Program").
* **License:** public, no DUA.
* **Loader:** `data.load_hrrp(path)`

| File column (verify in the current data dictionary) | Template column |
|---|---|
| `Facility Name`, `Facility ID`, `State` | `facility_name`, `facility_id`, `state` |
| `Measure Name` (e.g. READM-30-HF-HRRP) | `measure_name` |
| `Number of Discharges`, `Number of Readmissions` | `n_discharges`, `n_readmissions` ("Too Few to Report" → NaN) |
| `Excess Readmission Ratio` | `excess_readmission_ratio` |
| `Predicted Readmission Rate`, `Expected Readmission Rate` (%) | `predicted_rate`, `expected_rate` (0–1) |
| `Start Date`, `End Date` (performance period) | `start_date`, `end_date` |

* **Caveats:** Medicare FFS, condition-specific cohorts (AMI, HF, pneumonia, COPD, CABG, THA/TKA). Hierarchical
  risk model with CMS's covariates; a 3-year performance period lagging the fiscal year. Use it for context and
  vintage-matched ranges, not as a direct comparator for your own O/E.

## Public volumes: Medicare Physician & Other Practitioners PUF (optional)
* **Source:** data.cms.gov, *Medicare Physician & Other Practitioners — by Provider and Service*.
* **Use:** clinician volumes and services for attribution sanity checks and for choosing a `min_cases`
  threshold that reflects real panel sizes. No outcomes, so it's not a quality benchmark.

Files in this folder are gitignored.
