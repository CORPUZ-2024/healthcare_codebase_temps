# Test data — t06 ROI and cost offset

## Default: synthetic (no download)
`roi_cost_offset/data.py` → `generate_panel` builds a member-month cost panel with program
participation, a known effect (`effect_pct`, default 20%), a counterfactual cost column
(`paid_cf_amt`) and the observed referral trigger (`ip_admit_flag`). Program evaluation needs a
truth to test against, and member-level program data is never public, so the synthetic panel is
the main test source.

## Public cost distribution: AHRQ MEPS Household Component (optional)
* **Source:** Agency for Healthcare Research and Quality, *Medical Expenditure Panel Survey*,
  Full-Year Consolidated Data File (one per year; e.g. HC-243 = 2022 — check the file number for
  the year you want on the download page).
* **Page:** https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp
* **License:** public use, no DUA. Cite AHRQ MEPS.
* **Formats:** Stata `.dta`, Excel `.xlsx`, SAS, ASCII. The loader reads `.dta`, `.xlsx` or `.csv`.
* **Loader:** `data.load_meps_totexp(path, year)`

| MEPS column | Template column |
|---|---|
| `DUPERSID` | `person_id` |
| `TOTEXPyy` (total expenditures, all payers) | `totexp_amt` |
| `PERWTyyF` (person weight; 0 = out of scope) | `weight` |
| `AGEyyX` | `age` |
| `SEX` (1 = male, 2 = female) | `sex_cd` |

* **Use:** `data.cost_distribution_summary(df["totexp_amt"], df["weight"])` gives the weighted share
  of $0, mean, p50/p90/p99, top-1% share and a lognormal fit of positive spend. Compare it with
  your population, or re-tune the generator (`risk` spread, `sigma`, non-user share).
* **Caveats:** MEPS is the U.S. civilian non-institutionalized population, all payers, **annual**
  person-level spend, not claims PMPM. Nursing-home residents are excluded, which trims the tail
  that Medicaid LTSS populations have. Use it for shape, not for levels.

Files in this folder are gitignored.
