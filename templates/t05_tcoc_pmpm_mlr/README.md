# t05 — Total cost of care, PMPM, trend and medical loss ratio

Compute what a population costs per member per month (PMPM), what drives it (service category,
utilization vs. unit cost, a few high-cost members), what that means for a health plan's
medical loss ratio (MLR), and what an intervention's savings and fee do to that MLR. Includes an
optional inpatient-episode module.

| | |
|---|---|
| **Workflow type** | D — Health economics & VBC (D1 TCOC/PMPM, D2 MLR, D6 episodes) + B2 cost drivers |
| **Intent** | MEASURE, VALUE, DESCRIBE |
| **Volume** | M. Demo: 1,500 members × 24 months (~20K lines). SQL twin for warehouse push-down. |
| **Stack** | pandas (+ DuckDB SQL twin) |
| **JD link** | "total cost of care, cost offset, medical loss ratio impact", "cost drivers" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 21 tests
```

## Workflow

1. Prepare claims (latest version, service category) and prorated member-months by LOB
   (`prep.py`, copied from t00).
2. **Runout guard:** drop the last 3 months (`complete_months`); t08 shows how to complete them instead.
3. **PMPM** total, by LOB, by service category.
4. **High-cost claimants:** truncated vs. untruncated PMPM.
5. **Trend:** 2023 → 2024 change split into utilization and unit cost.
6. **MLR** (regulatory vs. simple) and **MLR impact** of a program's savings and fee.
7. **Episodes:** inpatient-anchored windows (3 days before, 30 after).

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| PMPM | `pmpm_ratio_of_sums` — Σpaid / Σmember-months (exposure-weighted) | `pmpm_mean_of_members` — mean of each member's PMPM, with CI | you need a per-person distribution or CI for a clinical comparison. Won't reconcile to Finance; short stays inflate it (see `test_mean_of_members_overweights_short_stays`). |
| High-cost truncation | `truncate_percentile` (p99) | `truncate_fixed_attachment` ($ attachment) | you're mirroring a stop-loss or contract with a fixed attachment point. In small or low-cost populations a fixed attachment may trim nothing. |
| Trend decomposition | `decompose_additive` — utilization + unit cost + interaction (adds to $ change) | `decompose_log` — multiplicative trend %, no interaction | you're talking to actuaries about annual trend rates, or compounding across years. Undefined with zero units. |
| MLR | `mlr_regulatory` — (claims + QI) / (premium − taxes & fees), vs. minimum | `mlr_simple` — paid / premium | quick monthly internal view only; never quote it to a plan as "the MLR". |
| Episodes | `build_episodes_nonoverlap` — readmission inside the window belongs to the first episode | `build_episodes_overlap` — every admission gets its own window | readmission-style outcome analysis. Episode costs then overlap and can't be summed. |

## Test data

* **Synthetic (default):** copied t00 generator + FAKE premium rates.
* **Public benchmark:** CMS Medicare Geographic Variation PUF (per-capita spend by state/county).
* **Public claims:** CMS DE-SynPUF (same as t00).
  Details and links: [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t05_methods.py` | PMPM known answers; categories add to the total; mean-of-members bias; truncation identities; decompositions add up; MLR formula and fee-classification effect; episode windows |
| `test_t05_checks_sql.py` | checks fire; DuckDB SQL PMPM = pandas; the **wrong join** anti-pattern (`sql/02_pmpm_wrong_join.sql`) understates PMPM |

## Caveats

* **Incurred vs. paid.** PMPM here is by service (incurred) month. Paid-month views are useful for
  cash but mix old and new services.
* **Exposure never splits by claim attributes.** Service-category PMPMs share the total denominator.
* **Truncation** thresholds in real contracts are set by the payer or program (MSSP truncates at
  a percentile of national spend); p99 of your own population is a stand-in.
* **MLR** here omits credibility adjustments, multi-year averaging and the regulation's detailed
  definitions of QI and deductible taxes. The MLR-impact scenario shows that *how the program fee
  is classified* (claims, QI or admin) matters as much as the savings.
* The synthetic population is small; PMPM by LOB has wide uncertainty (`check_small_exposure`).

## Explain it to Finance

"Total cost of care is $X per member per month, measured on service dates through <month> with the
last three months held out because claims are still arriving. Inpatient is the largest share. After
capping the top 1% of members, PMPM is $Y — both numbers are shown so one catastrophic case doesn't
decide the result. If our program saves $25 PMPM and costs $15 PMPM, the plan nets $10 PMPM; the
MLR effect depends on whether the plan books our fee as medical expense or administrative cost."

## Files

```
tcoc_pmpm_mlr/  config.py  data.py (copied from t00 + premium + GeoVar loader)  prep.py (copied from t00)
                methods.py  checks.py  sqltwin.py
sql/  01_pmpm_by_group.sql  02_pmpm_wrong_join.sql (anti-pattern, used in tests)
tests/  conftest.py  test_t05_methods.py  test_t05_checks_sql.py
```
