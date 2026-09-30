# t01 — Utilization profiling

Describe how much care a population uses and how that is changing: admissions, ED visits,
office/telehealth visits and HCBS days per 1,000 member-years with confidence intervals; average
length of stay; 30-day all-cause readmissions; frequent ED users; a runout-aware monthly trend;
and the cost distribution you should look at before any model.

| | |
|---|---|
| **Workflow type** | B — Population health (B1 utilization profiling) + C4 outcome definitions |
| **Intent** | DESCRIBE, MONITOR, MEASURE |
| **Volume** | M. Demo: 1,500 members × 24 months. SQL twins for ED rate and readmissions. |
| **Stack** | pandas, scipy.stats, matplotlib (+ DuckDB) |
| **JD link** | "identify utilization patterns, cost drivers", "readmission reduction", "payor claims" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/ (CSV + ed_trend.png)
python run.py --selftest
python -m pytest           # 18 tests
```

## Workflow

1. Prepare claims (latest version + service category) and member-months (`prep.py`, copied from t00).
2. **Stays:** collapse inpatient claims into stays; transfers (admit ≤ 1 day after discharge) merge.
3. **Events:** IP admits (from stays), ED visits, office visits, HCBS days (distinct member + date).
4. **Rates per 1,000** member-years with CIs, on complete months only.
5. **Readmissions:** 30-day all-cause per index stay, and per 1,000 member-years.
6. **Distribution** of paid per member (mean vs. median, percentiles, CV).
7. **Frequent ED users** (≥ 4 visits in any 12 months).
8. **Trend:** monthly rate, rolling-12 and YoY; runout months blanked.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| CI for a rate | `rate_ci_poisson_exact` — exact (Garwood) Poisson | `rate_ci_cluster_bootstrap` — resample members | events are concentrated in a few members (ED, IP, HCBS). The Poisson CI assumes independent events and is too narrow; the test shows the bootstrap is > 2× wider under overdispersion. |
| Readmissions | `readmissions_per_index` — % of eligible discharges readmitted in 1–30 days | `readmissions_per_1000` — readmissions per 1,000 member-years | you are evaluating a population program. Preventing first admissions can *raise* the per-index rate (remaining admissions are sicker) while total readmissions fall. |
| Trend | `trend_rolling12` — rolling 12-month rate | `trend_yoy_same_month` — this month vs. the same month last year | you need to see a change within months rather than a smoothed line (R12 lags ~6 months). |

## Test data

* **Synthetic (default):** copied t00 generator (frailty-driven readmissions, runout tail).
* **Public:** CMS DE-SynPUF Inpatient Claims — [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t01_methods.py` | transfers merge; the 30-day boundary (day 30 in, day 32 out); enrollment eligibility; Poisson CI known answer (0 events → upper 36.9); bootstrap wider under overdispersion; missing months reindexed not skipped; runout months blank; rolling-window frequent-ED rule |
| `test_t01_checks_sql.py` | checks fire; DuckDB ED rate and `LEAD()` readmission SQL match pandas |

## Caveats

* **Readmission here is simplified.** Official measures (CMS HRRP, HEDIS PCR) exclude planned
  readmissions, handle deaths and transfers in detail, and risk-adjust. This is an internal,
  all-cause, unadjusted rate. Say so on every chart.
* **Visits** are distinct member + date. Two ED claims on the same day are one visit; an ED visit
  that leads to admission is counted as ED here (many specs fold it into the admission).
* **Per 1,000 member-years** = events / member-months × 12,000.
* The CV of paid per member is ~1.8 — typical. Any model that assumes normal errors on cost will
  struggle (see t06 / glossary G20 two-part models).

## Explain it to Clinical and Finance

"Across the last 21 complete months, members had about 520 ED visits per 1,000 per year, and 17%
of hospital discharges were followed by another admission within 30 days. The ED range is wider
than a textbook interval because a small group of members accounts for many visits — 39 members
had four or more ED visits in a year, and they are the obvious outreach list."

## Files

```
utilization_profiling/  config.py  data.py (copied from t00 + SynPUF IP loader)  prep.py (copied)
                        methods.py  checks.py  charts.py  sqltwin.py
sql/  01_ed_per_1000_by_month.sql  02_readmissions.sql
tests/  conftest.py  test_t01_methods.py  test_t01_checks_sql.py
```
