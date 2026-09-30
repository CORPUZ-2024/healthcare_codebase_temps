# Test data — t04 quality measures and care gaps

## Default: synthetic (no download)
`quality_measures_care_gaps/data.py` generates members, enrollment spans (gaps, late starts, early
terminations, ~3% overlapping retro spans) and clinical events with a `source_cd` of CLAIM, LAB
(standard supplemental data) or CHART (found only by medical-record review). Member-level quality
data is never public, so the synthetic generator is the only member-level test source.

## Public benchmark: Medicaid & CHIP Core Set state rates
* **Source:** CMS / data.medicaid.gov, *Child and Adult Health Care Quality Measures* (state-level
  Core Set rates, one dataset per Core Set year). Search data.medicaid.gov for "Quality Measures".
  Program page: https://www.medicaid.gov/medicaid/quality-of-care/performance-measurement/adult-and-child-health-care-quality-measures
* **License:** public, no DUA.
* **Loader:** `data.load_core_set_rates(path)`. It matches headers case-insensitively and converts
  percentages to proportions:

| Public column (verify in your download) | Template column |
|---|---|
| `State` | `state` |
| `Measure Abbreviation` (e.g. `BCS-AD`, `WCV-CH`) | `measure_cd` |
| `FFY` or `Core Set Year` | `core_set_year` |
| `Population` (Medicaid, CHIP, Medicaid & CHIP) | `population` |
| `Methodology` (Administrative, Hybrid, ...) | `methodology` |
| `State Rate` (percent) | `state_rate` (0–1) |
| `Rate Definition` | `rate_definition` |

* **Caveats:** Core Set year *Y* mostly reflects services in calendar year *Y−1*. States report with
  different methodologies and populations, so filter before comparing (`check_benchmark_vintage`).
  Rates for the retired HbA1c *testing* indicator are not published; `FAKE_HBA1C_TEST` has no benchmark.

## Value sets
* NCQA HEDIS Value Set Directory: the Core Set subset is free after registration. https://www.ncqa.org/hedis/measures/
* VSAC (eCQM value sets, free UMLS account): https://vsac.nlm.nih.gov
* Format expected by the engine: see `../reference/README.md`.

Files in this folder are gitignored.
