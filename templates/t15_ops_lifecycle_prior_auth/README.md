# t15 — Operations lifecycle and prior authorization

The operations view of care: how many referrals become active patients, where they stall, how long it
takes, and how prior authorization (PA) affects it. This template builds:
* a referral → assessment → PA request → PA decision → start of care → 90-day retention funnel
* PA metrics against the CMS-0057-F decision timeframes, with denial and overturn rates
* censoring-aware time to start of care

Synthetic data are observed at a data cut, and the full future is kept as the truth. The tests check which
numbers stay honest while cases are still open.

| | |
|---|---|
| **Workflow type** | G — Cross-functional operations (G3 patient lifecycle funnel, PA turnaround, denial KPIs) |
| **Intent** | MONITOR |
| **Volume** | S. Demo: 5,000 referrals over 12 months. |
| **Stack** | pandas + numpy (own Kaplan–Meier; no survival library needed) |
| **JD link** | "operational analytics", "prior authorization turnaround", "referral-to-start-of-care" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 11 tests (incl. doctest)
```

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Stage conversion | `funnel(min_age_days=60)`: only referrals old enough to have progressed | — | always state the age cut-off; the all-referrals view understates conversion. |
| Time to start of care | `time_to_start`: Kaplan–Meier, open referrals censored at the data cut | `cohort_conversion`: monthly cohorts, share started within 30 days; immature cohorts left blank | most cohorts are mature and the audience wants a table. When operations change fast, most recent cohorts are immature, so use KM. |
| Denials | `pa_metrics` by priority (CMS-0057-F style) | `denial_pareto`: reasons ranked with cumulative share | deciding what to fix. Here the top 2 reasons are 59% of denials (missing documentation 39%). |

## Findings the tests lock in (seed 15)

| Finding | Evidence |
|---|---|
| KM time-to-start matches the eventual truth (P(start by 30 days) 62.8% vs. 62.8%) | `test_km_recovers_truth_despite_open_referrals` |
| **Most recent month:** naive "share started" is **34%**, KM says **63%** will start within 30 days, and the truth is **63%** | `test_recent_cohort_naive_share_is_badly_biased_km_is_not` |
| "Median days to start" answers two questions: 19 days for half of **all** referrals, 14 days among **those who started**. Label which one you report | demo output |
| PA: expedited median 41 h, standard 99 h. **16% / 18% miss the 72 h / 7-day targets** (`OPS-001`); 46–54% of appealed denials are overturned (`OPS-004`) | `test_pa_metrics_recover_generator`, `test_checks` |
| The timestamp-order check compares every stage pair (a build-time test showed neighbour-only comparisons miss errors when middle stages are blank) | `test_checks` |

## Test data

* **Synthetic (default):** `data.generate`: observed-at-cut referrals plus the full truth.
* **Public:** payer PA metrics required by CMS-0057-F (posted on each payer's website, first reports for 2025 data due
  March 31, 2026). See [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t15_ops.py` | KM by hand; KM recovers truth; recent-cohort naive bias vs. KM; funnel monotone and mature view; funnel by hand; PA metrics recover the generator; immature cohorts blank; Pareto order; every check (including the all-pairs order fix); observed data never shows the future |

## Caveats

* **Regulatory scope:** CMS-0057-F timeframes and reporting apply to specific payer types and start dates. The
  72 h / 7-day targets here are the rule's decision timeframes; verify which provisions apply to each payer and product.
* **Hours vs. calendar days:** "7 calendar days" is measured here as 168 hours from request to decision. Your contract or
  payer may count differently (business days, receipt of complete documentation).
* Never-starting referrals (declined, denied, no staff) are part of "all referrals". KM's "P(started by day d)" is over
  everyone; report the reasons for non-starts separately.
* Retention at 90 days is only known for starts at least 90 days before the cut.

## Explain it to Operations

"About 63% of referrals start care within 30 days; half start within 19 days. Don't judge last month's referrals by
who has started so far (34%). They're on the same track as everyone else; they just haven't had time. Authorization
is the biggest controllable delay. About one in six PA decisions comes after the regulatory deadline, and half of the
denials we appeal are overturned. Missing clinical documentation causes 39% of denials, so fixing the intake checklist
is the fastest win."

## Files

```
ops_lifecycle_prior_auth/  config.py  data.py  methods.py  checks.py
tests/  conftest.py  test_t15_ops.py
```
