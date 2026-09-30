# Reference tables — FAKE

`FAKE_value_sets.csv` is an **invented, tiny** value-set table. The codes are real, format-valid
CPT / HCPCS / ICD-10 / LOINC codes, but the *lists* are incomplete and are not NCQA value sets.
The measure specs in `../measures/` are likewise FAKE (each file says so in its first line).

## Where the real ones come from
* **HEDIS Value Set Directory (VSD)** — NCQA. Licensed; the **Medicaid & CHIP Core Set** subset is
  free to download after registering: https://www.ncqa.org/hedis/measures/ (look for "Value Set
  Directory" / "Core Set value sets").
* **Core Set technical specifications** (denominator, exclusions, continuous enrollment, lookbacks):
  https://www.medicaid.gov/medicaid/quality-of-care/performance-measurement/adult-and-child-health-care-quality-measures
* **eCQM value sets (VSAC)** — https://vsac.nlm.nih.gov (free UMLS account).

To use a real VSD: export the value sets you need to a CSV with columns
`value_set_name, code_system, code_cd` (codes **without** dots, upper case), keep the names used in
the YAML specs (or edit the YAML), and set `Config.value_set_path`.

| Column | Meaning |
|---|---|
| `value_set_name` | name referenced from `measures/*.yaml` |
| `code_system` | ICD10CM, ICD10PCS, CPT, CPT2 (Category II), HCPCS, LOINC |
| `code_cd` | code without dots, upper case (the engine normalizes event codes the same way) |
