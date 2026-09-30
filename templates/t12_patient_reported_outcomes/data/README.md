# Test data — t12 patient-reported outcomes

## Default: synthetic (no download)
`patient_reported_outcomes/data.py` → `generate_trial()`: a 600-caregiver, 3-visit trial with item-level
answers (FAKE burden scale + PHQ-9), skipped items, and missing-at-random dropout. `complete` holds the
data before missingness, which is the truth used by the tests.

## Public test file: AHRQ MEPS (optional)
* **Source:** Medical Expenditure Panel Survey, Full-Year Consolidated file (one per year).
  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
* **License:** public use, no DUA.
* **Loader:** `data.load_meps_pro(path)` keeps `DUPERSID`, SF-12 summary scores `PCS42` / `MCS42`, PHQ-2 `PHQ242`
  and K6 `K6SUM42`. Negative reserved codes (−1, −7, −8, −9) become NaN.
  **Verify the column names in your year's codebook**, since the round suffix and variable availability change.
* **Use:** population norms for PCS/MCS by age and sex, distribution checks (floor/ceiling), and a real
  single-timepoint dataset to exercise scoring pipelines. MEPS has no item-level PHQ-9 or caregiver burden data.

## Instruments
* **PHQ-9:** free to use (phqscreeners.com). Only item ids are stored here, not wording.
* **Caregiver burden:** the shipped scale is FAKE. The Zarit Burden Interview and similar scales are licensed
  (e.g. via Mapi Research Trust). Obtain a license before scoring them, then add a YAML spec with `license: licensed_on_file`.

Files in this folder are gitignored.
