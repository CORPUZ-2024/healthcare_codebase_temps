# Reference tables — ALL FAKE

Every file here starting with `FAKE_` is **invented** for teaching. Category names, codes-to-category
mappings, hierarchies and coefficients are shaped like the CMS-HCC model but are **not** CMS values.
Never use them for payment, bids, or anything shared outside a learning context.

## Where the real tables come from
* **CMS-HCC model software, mappings and coefficients (Medicare Advantage):** CMS *Risk Adjustment*
  page — https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
  (annual "model software/ICD-10 mappings" zip + the Rate Announcement for normalization factors).
* **Medicaid:** many states use CDPS+Rx (UC San Diego) — https://hwstudy.ucsd.edu/cdps/ — or a
  state-specific model; confirm with the plan.
* **ACA marketplace:** HHS-HCC (CCIIO) — different model, different hierarchies.

To use real tables: save them in this folder with the same columns as the FAKE files (or add a
column-rename step in `data.load_reference`) and pass `ref_dir=` / `prefix=""` in `Config`.

## Columns
| File | Columns |
|---|---|
| `FAKE_dx_to_hcc.csv` | dx_cd (no decimal), model_version, hcc, note |
| `FAKE_hierarchies.csv` | model_version, higher_hcc, drops_hcc |
| `FAKE_hcc_coefficients.csv` | model_version, variable, coefficient, label (HCCs, interactions, count variable) |
| `FAKE_demographic_factors.csv` | model_version, variable, coefficient (age/sex cells, DUAL, DISABLED) |
