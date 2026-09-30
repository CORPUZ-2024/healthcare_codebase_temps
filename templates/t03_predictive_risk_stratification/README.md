# t03 — Predictive risk stratification

Predict which members will be hospitalized in the next 6 months, check that the predictions are
trustworthy out-of-time (discrimination and calibration), turn them into care-management tiers
that fit team capacity, describe member segments for intervention design, and flag rising-risk
members.

| | |
|---|---|
| **Workflow type** | B — Population health & risk (B4 predictive risk, B5 acuity tiers/segmentation) |
| **Intent** | PREDICT, DESCRIBE |
| **Volume** | S. Demo: 6,000 members per cohort. EPV (events per variable) is the real constraint. |
| **Stack** | scikit-learn (logistic, HistGradientBoosting, isotonic, k-means), scipy |
| **JD link** | "patient stratification and risk models (utilization, hospitalization, acuity tiers)" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/ (CSV + calibration.png)
python run.py --selftest
python -m pytest           # 11 tests
```

## Workflow

1. **Build features** from a 12-month lookback; label = admission in the next 6 months.
2. **Pre-fit checks:** missing values, leakage-looking names, EPV ≥ 10, separation.
3. **Temporal split:** train on 2023, test on 2024 (never a random split for a deployed model).
4. **Fit** logistic (STANDARD) and gradient boosting + isotonic calibration (ALTERNATIVE).
5. **Evaluate:** AUC, Brier, calibration slope/intercept and deciles, PPV / sensitivity / lift in
   the top 5% (what care managers can actually reach).
6. **Recalibrate** the intercept if the base rate drifted.
7. **Tier** by capacity (5% / 15% / 80%); **segment** with k-means; list **rising-risk** members.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Model | `fit_logistic` — L2 logistic on standardized features, explained with odds ratios | `fit_gradient_boosting` — shallow HistGradientBoosting + isotonic calibration | you have enough events (thousands) and strong non-linear signal. On this demo (~440 training events) logistic wins out-of-time (AUC 0.76 vs 0.74) — a common real-world result. XGBoost/LightGBM are drop-in swaps. |
| Grouping | `tiers_by_capacity` — tiers by predicted-risk rank sized to the team | `segments_kmeans` — k-means on standardized continuous features, K by silhouette | you're designing interventions and need to know *who* members are, not just how risky. Low silhouette (~0.15 here) means segments are soft — profile them before naming them. |

## Test data

* **Synthetic (default)** with a known true model and cohort drift.
* **Public:** CMS DE-SynPUF recipe in [`data/README.md`](data/README.md).

## Tests

`tests/test_t03_methods.py`: EPV and separation; temporal split; logistic AUC > 0.70 with a
calibration slope near 1; boosting output valid; odds ratios in the true direction;
evaluation known answers (perfect model → AUC 1, PPV 1); calibration and recalibration;
capacity tier sizes; k-means finds 3 planted clusters; rising-risk rule; checks fire.

## Caveats

* **AUC is not enough.** A model can rank well and still predict the wrong *level* after drift
  (intercept −0.22 here); recalibrate before using probabilities for ROI or staffing.
* **PPV at capacity** is what the care team feels: 27% of Tier 1 members were admitted, about 4.5×
  the base rate.
* **Leakage:** features must end before the prediction date. Claims runout means the last weeks of
  the lookback are incomplete in production — build training data with the same lag.
* **k-means with 0/1 columns** splits on the binary column; use continuous features or a mixed-type
  method (k-prototypes, Gower distance).
* **Fairness:** check calibration by subgroup (dual status, ADI decile, race/ethnicity where
  available) before deployment — see glossary G29.

## Explain it to Clinical

"The model ranks members by the chance of a hospital stay in the next six months. If the team can
reach the top 5%, about one in four of those members would otherwise be admitted — four and a half
times the average. The main drivers are heart failure, COPD, prior admissions and low medication
adherence. We checked it on a later year than it was built on and adjusted it for this year's lower
admission rate."

## Files

```
predictive_risk_stratification/  config.py  data.py  methods.py  checks.py
tests/  conftest.py  test_t03_methods.py
```
