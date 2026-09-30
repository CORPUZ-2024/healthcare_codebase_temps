# Test data — t07 VBC contract modeling

## Default: synthetic (no download)
`vbc_contract_modeling/data.py` generates one ACO performance year (beneficiary annual costs), Medicaid
HCBS sub-cap members, and care-management participants. All contract terms in `../contracts/` are FAKE.

## Public test file: MSSP ACO performance-year results
* **Source:** CMS, *Shared Savings Program Accountable Care Organizations (ACO) — Performance Year
  Financial and Quality Results* public use file (one file per performance year), on data.cms.gov
  (search "Shared Savings Program Accountable Care Organizations").
* **License:** public, no DUA.
* **Loader:** `data.load_mssp_puf(path)`. It strips `$`, `,` and `%`, and converts 0–100 percentages to 0–1.
* **Replication:** `methods.reproduce_mssp_puf(puf)` recomputes benchmark − expenditure and first-dollar
  earned savings, and flags mismatches.

| PUF column (verify in your year's data dictionary) | Template column |
|---|---|
| `ACO_ID`, `ACO_Name`, `Current_Track` | `aco_id`, `aco_name`, `track` |
| `N_AB` (assigned beneficiaries) | `n_ab` |
| `ABtotBnchmk`, `ABtotExp` | `benchmark_total_amt`, `expenditure_total_amt` |
| `BnchmkMinExp`, `GenSaveLoss`, `EarnSaveLoss` | `bnchmk_min_exp_amt`, `gen_save_loss_amt`, `earn_save_loss_amt` |
| `MinSavPerc`, `Sav_rate`, `FinalShareRate`, `QualScore` | `msr_pct`, `sav_rate`, `final_share_rate`, `quality_score` |

* **Expected mismatches** (not engine bugs): savings/loss caps, two-sided loss settlements, prior-savings
  adjustments, regional/benchmark adjustments already inside `ABtotBnchmk`, and any payment adjustments
  applied after the PUF figures. Read the PUF methodology notes; reproduce a handful of simple one-sided
  ACOs first.

Files in this folder are gitignored.
