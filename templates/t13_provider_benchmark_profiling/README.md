# t13 — Provider benchmarking and profiling

Which hospitals, agencies, practices or care teams are really better or worse? Observed rates mix three
things:
* **case mix**: who they treat, which risk adjustment removes
* **chance**: how many they treat, which funnel limits and shrinkage handle
* **quality**: the signal left over

This template does indirect standardization (O/E), exact Poisson funnel limits, empirical-Bayes shrinkage and
a hierarchical logistic alternative. It also places a provider against a national file with a vintage check.
The synthetic providers have a known true quality, so every estimator is scored.

| | |
|---|---|
| **Workflow type** | G — Cross-functional performance (G1 provider/panel benchmarking: O/E, EB shrinkage, funnel, external benchmark with vintage check) |
| **Intent** | MEASURE, MONITOR |
| **Volume** | S. Demo: ~19,000 patients at 80 providers (12–863 cases each). |
| **Stack** | statsmodels + scipy (+ matplotlib for the funnel plot) |
| **JD link** | "provider performance benchmarking", "risk-adjusted comparisons" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/ (provider_profile.csv, funnel_plot.png)
python run.py --selftest
python -m pytest           # 13 tests (incl. doctest)
```

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Expected events | `risk_model`: patient-level logistic **without** provider terms | — | always fit without provider terms; expected = "this patient at an average provider". |
| Comparison | `oe_table` (exact Poisson CI) + `funnel_flags` (95% / 99.8% limits) | — | — |
| Small providers | `eb_shrink_oe`: Poisson-gamma empirical Bayes, (O + a)/(E + a), reliability E/(E + a) | `mixed_logistic_ratios`: random-intercept logistic (statsmodels variational Bayes), predicted/expected as in CMS hospital measures | you want case-mix coefficients estimated jointly with provider effects, or to match CMS construction. The two agreed here (RMSE 0.126 each). VB is approximate, so use lme4/SAS for production. |
| Unadjusted rates | — | `eb_beta_binomial` | process measures where risk adjustment isn't appropriate. |

## Findings the tests lock in (seed 13)

| Estimate vs. TRUE provider ratio | RMSE (log) | Spearman |
|---|---|---|
| crude rate / average | 0.594 | 0.57 |
| O/E (raw) | 0.499 | 0.78 |
| **EB-shrunk O/E** | **0.126** | 0.79 |
| **mixed model P/E** | **0.126** | 0.79 |

* **Crude rankings are wrong:** crude vs. risk-adjusted rank correlation is 0.67 (`BEN-004`), because some providers treat much sicker patients.
* **Shrinkage cuts error ~4×.** It moves small providers most: the 12-case provider with 0 events goes from O/E 0.00 to 0.96 (reliability 0.04).
* **Funnel:** 16 of 80 providers are outside the 95% limits; 5 are outside 99.8% (3 better, 2 worse; all 5 are truly different in the synthetic truth). Real variation exists (`BEN-002`), so act on EB/mixed estimates, not raw flags.
  With no true differences, the false-alarm rate is ≈ 5% / 0.2% (`test_funnel_false_alarm_rate_without_provider_effects`).

## External benchmark

`data.load_hrrp` reads CMS Hospital Readmissions Reduction Program files. Percentages become proportions,
and "Too Few to Report" becomes NaN, never 0. `benchmark_percentile` places a value in the national distribution.
`check_benchmark_vintage` (ANL-013) fires when the benchmark's **performance period** is more than 180 days from yours.
HRRP periods are 3 years ending ~2 years before the fiscal year, so the demo's FAKE file (2021-07 to 2024-06 vs.
your 2024-07 to 2025-06) is flagged.
**Your O/E and CMS's excess readmission ratio use different risk models and reference populations. Compare
positions and ranges, not decimals.**

## Test data

* **Synthetic (default):** `data.generate`: patients nested in providers with known effects and different case mix.
* **Public:** CMS HRRP (hospital readmissions) and the Medicare Physician & Other Practitioners PUF (volumes and
  services by clinician). See [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t13_benchmarking.py` | O/E and exact CI by hand; risk model calibrated; funnel narrows with volume and has the right false-alarm rate; EB and mixed beat raw O/E by > 2× and agree; risk adjustment beats crude ranking; shrinkage strongest for small providers; τ² in range; beta-binomial EB; HRRP loader (suppression, percentages, dates); every check; funnel PNG written |

## Caveats

* **Risk adjustment is only as good as its covariates** (c-statistic 0.68 here). Unmeasured severity looks like poor quality.
  Don't adjust for things the provider controls (that hides real differences).
* **Shrinkage pulls small good providers toward average too.** EB is conservative by design; report it with intervals and volume.
* **Multiple comparisons:** with 80 providers, expect ~4 outside 95% limits by chance alone. That's why the 99.8% limit exists.
* **Attribution:** patient-to-provider attribution rules (glossary G09) change the denominators; fix them before benchmarking.

## Explain it to Network Management

"Looking at raw readmission rates would put the hospitals that take the sickest patients at the bottom. After
adjusting for each hospital's patients and accounting for how few cases the small ones have, three large
hospitals are reliably better than expected (about 25-30% fewer readmissions), and two mid-sized hospitals are
reliably worse (about 40-70% more), even at the strict threshold. The small hospitals' rates are mostly noise,
and we don't rank them."

## Files

```
provider_benchmark_profiling/  config.py  data.py  methods.py  checks.py
tests/  conftest.py  test_t13_benchmarking.py
```
