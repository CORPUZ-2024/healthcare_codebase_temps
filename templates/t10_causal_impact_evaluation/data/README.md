# Test data — t10 causal impact evaluation

## Default: synthetic with known effects (no download)
`causal_impact_evaluation/data.py`:
* `cross_section()`: members referred by risk (confounded), heterogeneous effect; returns the true ATE and ATT.
* `practice_panel()`: practices × months, one adoption date; `pre_trend=` breaks parallel trends on purpose.
* `its_series()`: a monthly rate with a level and slope change, seasonality, AR(1) errors.

Causal estimators can only be *validated* where the truth is known, so the tests use these generators.

## Real-data analogues
* **Programs selected by referral / risk:** member-level covariates and outcomes from your claims (t00 → t05),
  with the program flag and enrollment date. Use t06 for the cost version (matched pre/post DiD on PMPM).
* **Practice or region rollouts:** practice × month utilization from claims (t01) with the rollout calendar.
* **Statewide policies (ITS):** a monthly public series spanning the policy date, e.g. CMS Medicaid & CHIP
  monthly enrollment or state-level utilization files on data.medicaid.gov / data.cms.gov. Check that nothing
  else changed at the same date (benefit changes, coding changes, the pandemic).

Files in this folder are gitignored.
