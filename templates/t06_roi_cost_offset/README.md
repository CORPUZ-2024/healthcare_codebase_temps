# t06 — Program ROI and cost offset

Did a care-management (or caregiver-support, or disease-management) program lower medical cost
enough to pay for itself? This template builds a matched comparison group, estimates the savings
per participant month with difference-in-differences (DiD) on PMPM, and turns the result into
net savings, ROI and break-even PMPM. It also runs a tornado sensitivity showing which assumption
decides the answer.

The synthetic data have a **known** effect (20% lower cost for participants after enrollment).
It also contains the two biases that make vendor ROI slides wrong: **selection** (sicker members
are referred) and **regression to the mean** (referral follows an admission, and that cost spike
fades anyway). The tests check which methods recover the truth.

| | |
|---|---|
| **Workflow type** | D — Health economics & VBC (D3 intervention ROI / cost offset / break-even / sensitivity) |
| **Intent** | VALUE, EXPLAIN |
| **Volume** | S. Demo: 10,000 members × 24 months (~230K member-months), ~1,000 matched pairs. SQL twin for the pre/post summary. |
| **Stack** | pandas + statsmodels (+ DuckDB SQL twin) |
| **JD link** | "cost offset", "return on investment", "program evaluation" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 25 tests (incl. doctest)
```

## Workflow

1. **Align on the index month** (`member_periods`): program start for participants, a pseudo-index
   drawn the same way for everyone else. Pre = months −6..−1, post = +1..+6, and the index month is excluded.
   A member needs ≥ 4 months of exposure in each period.
2. **Propensity score** (logit) on age, sex, chronic conditions, risk, pre-period PMPM, last-3-month
   PMPM and the **referral trigger** (an admission in the last 3 months).
3. **Match** 1:1 on the logit score, caliper 0.2 SD, exact on LOB and the trigger → **balance table** (|SMD| < 0.1).
4. **Pre-trend test** on the matched pairs' monthly pre-period PMPM.
5. **DiD on PMPM** (standard), plus the **bootstrap** and **two-part** alternatives.
6. **ROI**: gross savings, program cost, net, ROI, break-even savings PMPM; the CI is carried through to ROI.
7. **Tornado**: each input at low/high, sorted by swing.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Comparison group | `match_propensity`: 1:1 nearest neighbour on logit PS, caliper 0.2 SD, exact on LOB + trigger | exact-only matching (put every covariate in `exact_on`) | you have few, categorical covariates and many controls. With continuous covariates, exact matching discards most participants. |
| Effect | `did_pmpm`: member-month-weighted DiD (= ratio-of-sums), cluster-robust SE by pair | `bootstrap_did`: percentile CI from resampled **pairs**, plus P(savings) and P(≥ break-even) | costs are skewed and the sample is small, or the decision needs a probability ("58% chance the program breaks even"). |
| Cost model | same DiD (a linear model on dollars) | `two_part_did`: logit P(any cost) × Gamma log-link E[PMPM \| cost > 0], recycled predictions | many $0 periods, or you need to separate "fewer members with any cost" from "lower cost per user". Model-based, and noisier here (−$119 against a true −$209). Harder to explain to Finance. |
| Naive pre/post | — | `naive_pre_post` (**do not report**) | never. It's included to show the bias (−$318 against a true −$209). |

## Demo results (seed 21)

| Estimate (PMPM, negative = savings) | Value | 95% CI |
|---|---|---|
| Truth (synthetic only) | −209 | |
| Naive participant pre/post | −318 | |
| **Matched DiD** | **−189** | −330 to −48 |
| Pair bootstrap | −189 | −336 to −59; P(savings) 99.7%, P(≥ break-even $175) 58% |
| Two-part model | −119 | |

With the FAKE economics (500 participants × 12 months, $150 PMPM fee + $300 one-time), net savings
are $82K and ROI is 0.08, with a CI of −0.73 to 0.88. The tornado shows the savings estimate
dominates everything else.

## Test data

* **Synthetic (default):** `data.generate_panel`, a member-month panel with a known effect and a counterfactual cost column.
* **Public cost distribution:** AHRQ **MEPS** Full-Year Consolidated file via `data.load_meps_totexp`.
  Use `cost_distribution_summary` to compare your plan's (or the synthetic) cost shape with a
  national survey: share of $0, top-1% share, lognormal fit.
  Details: [`data/README.md`](data/README.md).
* **Your claims:** build member × month paid with t05's `prep.py` (copy it, don't import it) and add the program start date.

## Tests

| File | What it proves |
|---|---|
| `test_t06_methods.py` | index-month alignment and exposure rules; DiD regression = ratio-of-sums; bootstrap point = DiD; **matched DiD CI covers the known truth and detects savings, naive pre/post is further off and too good, unmatched DiD and matching without the referral trigger are biased**; no effect → CI covers 0; matching is 1:1, exact, within caliper, and balances; parallel pre-trends; ROI break-even identity; tornado order |
| `test_t06_data_checks_sql.py` | synthetic cost shape (zeros, tail, skew); selection and counterfactual; MEPS loader and weighted summary; every check fires and stays quiet correctly; SQL pre/post summary and DiD cells = pandas |

## Caveats

* **Match on the referral trigger, not just on cost.** Matching fixes only what you match on. Leave
  the admission flag out and cost-based matching looks fine, but the trigger stays unbalanced
  (SMD ≈ 0.7). The DiD then roughly doubles the savings (−$403 against a true −$217 for those participants;
  `test_matching_on_cost_without_the_trigger_overstates_savings`). Use the real trigger
  (admission, ED visit, new diagnosis). Any trigger you *can't* observe biases the result the same way.
* **Engagement vs. enrollment.** Fees are often charged per *engaged* member; evaluate the same population you pay for.
* **Durability.** A 6-month DiD applied to 12 months of fees assumes the effect persists. Say so,
  or re-run with `post_months=12` once data allow.
* **Runout.** Post-period months must be complete (see t05 / t08). Recent months understate cost
  and flatter the program.
* **Dropped participants** (no control inside the caliper; 11% here): the effect describes matchable participants only.
* For staggered program start, event studies, AIPW and interrupted time series, see **t10**. For how
  many participants you need to detect a given saving, see **t09**.

## Explain it to Finance

"Participants cost $1,046 PMPM before the program and $837 after. But matched non-participants
with the same risk, the same recent hospitalization and the same LOB also changed, from $1,067 to
$1,046. The program's effect is the difference between those changes: about $189 PMPM saved, with
a 95% range of $48 to $330. At a $150 PMPM fee plus $300 per enrollment, break-even is $175 PMPM,
so the best estimate is a small net gain (ROI 0.08). The chance the program at least breaks even is
about 58%. The 'before vs. after' drop of $318 that a vendor would show is mostly members recovering
from the hospital stay that got them referred."

## Files

```
roi_cost_offset/  config.py  data.py (panel generator + MEPS loader)  methods.py  checks.py  sqltwin.py (copied from t05)
sql/    01_member_period_pmpm.sql  02_did_cells.sql
tests/  conftest.py  test_t06_methods.py  test_t06_data_checks_sql.py
```
