# t08 — Claims completion (IBNR) and PMPM forecast

Recent months always look cheap because their claims haven't all arrived yet. This template
builds the incurred-by-paid lag triangle, estimates what each month will cost once fully paid
(**ultimate**) and what is still outstanding (**IBNR**), and then forecasts the *completed* PMPM
12 months ahead with an interval. The synthetic data carry the true ultimate for every month, so
both steps are scored against it (backtests).

| | |
|---|---|
| **Workflow type** | E — Actuarial / forecasting (E1 completion / IBNR, E2 trend and forecast) |
| **Intent** | MEASURE, MONITOR, PREDICT |
| **Volume** | M. Demo: 36 incurred months × 19 lags from ~20K members (claim lines aggregate to the same shape). SQL twin for the triangle. |
| **Stack** | pandas + statsmodels (+ DuckDB SQL twin) |
| **JD link** | "claims completion and IBNR", "cost trend forecasting", "work with actuarial" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 26 tests (incl. doctest)
```

## Workflow

1. **Triangle** (`build_triangle`): incurred month × lag (calendar months). Observable-but-unpaid cells = 0,
   future cells = NaN. `as_of` rebuilds the triangle as it looked at any past data cut.
2. **Chain ladder** (`chain_ladder`): volume-weighted age-to-age factors over the latest 12 months →
   CDF to ultimate → completion %, ultimate, IBNR.
3. **Bornhuetter-Ferguson** for months with ≤ 2 lags observed (`expected_pmpm_seasonal` →
   `bornhuetter_ferguson` → `blend_cl_bf`). `complete()` does steps 1–3 in one call.
4. **Backtest IBNR** at 12 earlier data cuts against the true ultimates.
5. **Forecast** completed PMPM (SARIMAX and ETS) and backtest both on a 6-month holdout.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Ultimate / IBNR | `chain_ladder`: paid-to-date × CDF | `bornhuetter_ferguson`: paid + a-priori ultimate × (1 − completion); a-priori = same month last year × YoY trend | the month is immature (here age ≤ 2, < ~85% complete). Backtest error at age 0: **CL 5.1% vs. BF 3.1%**. Mature months: identical. BF is slow to show a real cost jump, so hand back to CL as the month matures. |
| Forecast | `forecast_sarimax`: log PMPM, AR(1) × seasonal difference + drift | `forecast_ets`: log PMPM, damped additive trend + additive seasonality | level shifts (benefit or mix changes) where recent months should dominate. Here ETS was less accurate (MAPE 4.5% vs. 3.3%), and its 95% interval covered only 2 of 6 holdout months, against 6 of 6 for SARIMAX. Check coverage before quoting ETS ranges. |

## Demo results (seed 8, data cut 2025-12-31)

| | Value |
|---|---|
| Last month's **paid** PMPM vs. true | $220 vs. $588 (understated by 63%) |
| Last month's completed PMPM (BF) | $587 (CL alone: $632) |
| Total IBNR estimate vs. truth | $14.94M vs. $15.30M (−2.4%) |
| Lag 0 → 1 factor | 1.91 (35% of a month is paid in the service month) |
| Faster-processing scenario | `IBNR-004` fires: recent lag-0→1 factor 16% below average, so old factors overstate IBNR |

## Test data

* **Synthetic (default):** `data.generate`: incurred × paid cells, exposure, and the **truth** (ultimate per
  month). `speedup=` simulates a claims-processing change.
* **Your claim lines / public files:** `data.load_claim_lines(path, incurred_col, paid_col, amount_col)`.
  Any extract with a service date and a paid date works, including t00's synthetic claims (`svc_from_dt`, `paid_dt`).
  See [`data/README.md`](data/README.md) for DE-SynPUF.

## Tests

| File | What it proves |
|---|---|
| `test_t08_completion.py` | 0 vs. NaN cells; `as_of` history; calendar-month lags; chain-ladder known answer; tail factor; BF bounds; paid PMPM understates; completed PMPM within 5% of truth; total IBNR within 5%; **BF beats CL at age 0 in the backtest**; processing speed-up biases IBNR up; loader drops paid-before-service rows |
| `test_t08_forecast_checks_sql.py` | forecast shape, intervals widen; winter peak + trend recovered; holdout MAPE < 8%; **forecasting paid (incomplete) PMPM drags the forecast down >10%**; each check fires; SQL triangle = pandas; paid-basis vs. incurred-basis PMPM |

## Caveats

* **Tail factor** is 1.0 (nothing paid after 18 months). Real health lines have a small tail, from
  coordination-of-benefits and provider disputes; get it from your actuary.
* **Pattern changes** (new claims system, clearinghouse, prompt-pay rules) invalidate old factors. Watch `IBNR-004`.
* **Large claims** distort single-month development. Develop them separately, or cap and add back.
* **Seasonal models** need ≥ 24 months. The ETS intervals here under-cover, so don't quote them without a backtest.
* This is **completion for analytics**, not a statutory reserve. Reserves add margins, provisions for
  adverse deviation, and actuarial sign-off.

## Explain it to Finance

"December looks like $220 PMPM because only about a third of December's claims have been paid so far.
Based on how past months filled in, and anchored to last December trended forward, December will land
at about $587. The outstanding amount across all months is about $15M, and in backtests this approach
was within 3% for the newest month. Next year's PMPM is forecast to run $490–630 by month, with the
usual January peak. The range widens the further out we go."

## Files

```
claims_completion_forecast/  config.py  data.py  methods.py  checks.py  sqltwin.py (copied from t05)
sql/    01_lag_triangle.sql  02_paid_vs_incurred_pmpm.sql
tests/  conftest.py  test_t08_completion.py  test_t08_forecast_checks_sql.py
```
