# t00 — Claims foundation

Turn raw payer data (medical claims, pharmacy claims, enrollment, providers) into the
analytic-ready tables every other analysis depends on: one row per final claim line, prorated
member-months, a service category per claim, and pharmacy fills net of reversals — plus the
data-quality checks that tell you what is wrong before you trust a number.

| | |
|---|---|
| **Workflow type** | A — Data foundation (A1 claims + eligibility → analytic-ready, A2 code-set validity) |
| **Intent** | STANDARDIZE |
| **Volume** | M (1M–50M lines). Demo: ~15K lines. Warehouse push-down pattern in `sql/`. |
| **Stack** | pandas + DuckDB SQL twin |
| **Settings** | payer, VBC, home-based care, hospital, post-acute, pharmacy, state Medicaid |
| **JD link** | "claims data structures and standard code sets (ICD-10, CPT, NDC)", "advanced SQL" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # synthetic demo -> outputs/
python run.py --selftest   # checks without pytest
python -m pytest           # full test suite (30 tests, ~3 s)
```

## Workflow

1. **Generate or load** claims, enrollment, providers, pharmacy (`data.py`).
2. **Check** grain, NPI validity, code formats, dates, overlaps, runout (`checks.run_all`).
3. **Collapse claim versions** to the final adjudicated state.
4. **Build member-months** with partial-month proration and overlap merging.
5. **Assign a service category** to each claim (IP > ED > OP > HCBS > home > telehealth > professional).
6. **Net pharmacy reversals.**
7. **Write** `outputs/claim_line_latest.csv`, `member_month.csv`, `claim_service_category.csv`,
   `rx_fill.csv`, `dq_findings.csv`.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Claim versions | `collapse_versions_latest` — keep highest `adj_seq` per claim, drop voids | `collapse_versions_net` — sum all versions | the payer sends a **delta** feed (adjustments as differences). Using the wrong one double counts — the test `test_net_on_replacement_feed_double_counts` shows it. |
| Member-months | `member_months_daily` — days enrolled / days in month | `member_months_midmonth` — full month if enrolled on the 15th | you must reconcile to a premium/capitation file that uses the anchor-day rule. |
| Service category | `service_category_claim` — one category per claim by hierarchy | `service_category_line` — each line on its own | you need unit-cost or fee-schedule work rather than visit/cost reporting. |
| Pharmacy reversals | `net_pharmacy_drop_pairs` — remove reversal + original | `net_pharmacy_signed` — sum signed $ | you're reconciling dollars to the ledger (never for adherence). |

## Data-quality checks (`checks.py`)

| ID | Catches | Severity |
|---|---|---|
| SCH-004 / VAL-005 | duplicate grain (e.g. one NPI, several practice locations → join fan-out) | error |
| CLN-014 | NPI not 10 digits / fails Luhn (usually lost leading zeros) | warn |
| CLN-052 | diagnosis not ICD-10-CM shaped (ICD-9, typos) | warn |
| CLN-053 / CLN-042 | invalid HCPCS/CPT / NDC format | warn |
| VAL-011 / 014 / 015 | impossible date orders | error |
| CLN-061 | overlapping enrollment spans | warn |
| CLN-023 | negative paid lines | warn |
| VAL-012 | claims outside every enrollment span | warn |
| VAL-031 | trailing months below 70% of baseline (claims runout) | warn |

## Test data

* **Synthetic (default):** seeded generator in `data.py`. Realism: right-skewed cost, Dec/Jan
  seasonality, ~3% replacement versions, ~1% voids, ~2% overlapping spans, Luhn-valid NPIs,
  string ZIPs with leading zeros, ~2% pharmacy reversals, 3-month runout tail.
* **Public:** CMS DE-SynPUF Sample 1 carrier claims — download steps and column mapping in
  [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `tests/test_t00_methods.py` | known answers (15/31 = 0.484 MM, overlaps, voids), standard vs. alternative behavior |
| `tests/test_t00_checks.py` | every check fires on an injected defect and stays quiet on clean data |
| `tests/test_t00_sql_twin.py` | DuckDB SQL gives the same results as pandas |
| doctests in `claims_foundation/` | docstring examples run |

## Caveats

* Code checks are **format** checks, not validity checks against the annual CMS code files
  (ICD-10 updates every Oct 1; HCPCS quarterly). For validity, join to the official code files.
* The service-category hierarchy is a common convention, not a standard. Write down the one you
  use; Finance and actuaries will ask.
* Claims outside enrollment should be reported as "unmatched spend", never silently dropped.
* The runout check flags a symptom; the fix (completion factors) is in `t08`.

## Explain it to Finance

"These are paid claims as of the latest adjudication, after removing voids and double-counted
adjustments, divided by the exact number of enrolled member-days converted to months. The last
three months are incomplete because claims are still coming in; we either exclude them or
estimate what's missing."

## Files

```
claims_foundation/
  config.py   data.py   checks.py   methods.py   sqltwin.py
sql/  01_claims_latest_version.sql  02_member_months.sql  03_service_category.sql
tests/  conftest.py  test_t00_methods.py  test_t00_checks.py  test_t00_sql_twin.py
```
