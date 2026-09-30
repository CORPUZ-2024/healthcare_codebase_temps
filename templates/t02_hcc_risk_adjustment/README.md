# t02 — HCC risk adjustment

Turn diagnoses into a risk adjustment factor (RAF) the way CMS-HCC-style models do — acceptable
sources only, diagnosis → condition category (HCC), hierarchies, interactions, demographic cells,
version blending, normalization and coding-intensity adjustment — then check how well the score
predicts cost, compare it with weights re-estimated on your own population, and produce an audit
trail and a suspect-condition list for clinical review.

> **All mappings and coefficients in `reference/` are FAKE.** They are shaped like CMS-HCC so the
> mechanics are real, but the numbers are invented. See `reference/README.md` for the real sources.

| | |
|---|---|
| **Workflow type** | B — Population health & risk (B3 risk adjustment) |
| **Intent** | MEASURE |
| **Volume** | S. Demo: 3,000 members; real use ~30K–80K diagnosis rows per 10K members. |
| **Stack** | pandas, numpy (no ML library — the standard model is a published additive formula) |
| **JD link** | "own risk adjustment methodology (HCC/RAF) … aligned with payor and regulatory expectations" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 14 tests
```

## Workflow

1. **Filter** diagnoses to acceptable sources (IP, OP, face-to-face professional). Lab/DME out.
2. **Map** ICD-10 → HCC for each model version (codes normalized: upper case, no decimal).
3. **Hierarchies:** keep the most severe HCC in each family (CKD5 drops CKD4/CKD3).
4. **Features:** age/sex cell, DUAL, DISABLED, HCC flags, interactions (DIAB×CHF, CHF×COPD),
   count variable (4+ HCCs).
5. **Score** = Σ coefficients (STANDARD), per version.
6. **Blend** versions by payment year (PY2024 67/33 V24/V28, PY2025 33/67, PY2026 100% V28).
7. **Normalize and adjust:** ÷ normalization factor × (1 − coding-pattern adjustment, ≥ 5.9%).
8. **Validate:** predictive ratios by decile and R² on a held-out half; compare with the ALTERNATIVE.
9. **Operate:** audit trail per member; suspect-condition list for clinicians (not for coding).

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Score | `score_published_weights` — apply the payer's published coefficients | `score_empirical_weights` — re-estimate additive weights on your own cost data (ridge) | you need a score for **internal** stratification or evaluation in a population the payer model wasn't calibrated on (Medicaid children, home-care users). Never for revenue or bids; validate out of sample. |

The synthetic population includes a condition (ASTHMA) that the V28-like model does not pay, and
unmeasured frailty, so the re-estimated weights predict held-out cost better (R² ≈ 0.18 vs 0.06) —
while the published score is still the one the plan is paid on. Both facts matter in a
methodology discussion.

## Test data

* **Synthetic (default)** with a known "true" cost structure.
* **Reference tables:** FAKE here; real CMS-HCC / CDPS+Rx sources in `reference/README.md`.
* **Public claims:** DE-SynPUF (ICD-9 — see [`data/README.md`](data/README.md) for what to expect).

## Tests

| File | What it proves |
|---|---|
| `test_t02_methods.py` | code normalization; LAB exclusion; hierarchy + interaction known-answer score; version differences; demographic bands; blend and normalization arithmetic; empirical weights recover a known signal; predictive ratios = 1 for a perfect model; empirical beats published out-of-sample on this population; suspect gaps; audit trail sums to the score |
| `test_t02_checks.py` | model-year mismatch (ANL-008), unacceptable sources, ICD-9 detection, score range |

## Caveats

* **Prospective model timing:** diagnoses from year Y−1 pay year Y. Mixing years is a hard error.
* **Face-to-face and provider-type rules** are simplified to a source code here; real filtering
  uses bill type, CPT/HCPCS and provider specialty.
* **RADV:** every HCC must be supported by a medical record. The suspect list exists so clinicians
  can *assess* members; coding from drug evidence or lab orders is non-compliant.
* **Medicaid ≠ Medicare:** Medicaid plans usually use CDPS+Rx or a state model with different
  categories, pharmacy-based markers and no MA coding-pattern adjustment.
* Scores are relative. A mean below 1.0 on the synthetic data just reflects the FAKE calibration.

## Explain it to Finance

"Our members' average risk score for 2026 is 0.61 after the normalization factor and the
coding-intensity cut. The published model under-predicts our lowest-risk members and over-predicts
the top decile. A model fitted to our own data predicts cost better, which supports the argument
that the payer model misses cost drivers specific to home-based care — but payment is still based
on the published score."

## Files

```
hcc_risk_adjustment/  config.py  data.py  methods.py  checks.py
reference/  FAKE_dx_to_hcc.csv  FAKE_hierarchies.csv  FAKE_hcc_coefficients.csv  FAKE_demographic_factors.csv  README.md
tests/  conftest.py  test_t02_methods.py  test_t02_checks.py
```
