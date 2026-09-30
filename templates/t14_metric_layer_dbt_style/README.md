# t14 — Metric layer (dbt-style) and experiment readout

Give every team the same numbers. Raw extracts go through version-controlled SQL models
(`staging → intermediate → marts`), are checked by data tests, and feed metrics defined **once** in
`metrics.yaml`. A small `dbt_lite.py` runs it all on DuckDB with no server; the same files move to real dbt
on Snowflake/BigQuery almost unchanged. On top sits an experiment readout for an outreach A/B test, with
guardrail metrics, a sample-ratio-mismatch check and CUPED variance reduction.

| | |
|---|---|
| **Workflow type** | G — Cross-functional (G2 measurement framework and experiment readout) |
| **Intent** | DESIGN, MEASURE |
| **Volume** | S. Demo: 20,000 members, ~45K outreach rows, 240K member-months. |
| **Stack** | DuckDB SQL + pandas + scipy + PyYAML |
| **JD link** | "metric definitions / semantic layer", "data quality tests", "A/B test readouts" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # build models, run tests, metrics, readout -> outputs/
python run.py --selftest
python -m pytest           # 18 tests (incl. doctests)
```

## Layout

```
models/staging/        stg_members  stg_assignments  stg_outreach  stg_claims_monthly   (read {{ source() }} only)
models/intermediate/   int_member_outreach  int_member_cost                             ({{ ref() }} staging)
models/marts/          fct_experiment_member                                            (no source() allowed)
models/schema.yml      not_null / unique / accepted_values / relationships tests
metrics.yaml           engagement_rate (primary), post_pmpm (secondary, CUPED covariate), complaint_rate and opt_out_rate (guardrails)
```

`dbt_lite.build` parses `{{ ref() }}` / `{{ source() }}`, sorts the models topologically (errors on unknown refs and cycles),
enforces the layering rules and creates one view per model. `dbt_lite.run_tests` turns each schema test into a
"failing rows" query.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Metric computation | `metric_table_sql`: SQL generated from `metrics.yaml`, run where the data live | `metric_table_pandas`: same YAML in pandas | no warehouse (notebook, extract, unit tests of definitions). `check_engine_parity` (MET-004) asserts both give identical numbers. |
| Effect estimate | `experiment_readout`: two-proportion z (proportions), Welch t (means) | `cuped_effect`: adjust the outcome for a **pre-assignment** covariate | noisy outcomes (cost) with a correlated pre-period value. Here CUPED cut post-PMPM variance by **67%** (CI −35.9..23.0 → −24.1..9.7). |
| Guardrails | fail if the **upper confidence bound** of harm exceeds `max_harm` (non-inferiority) | — | always. "p > 0.05 for harm" is not safety. A small-sample test shows a guardrail failing with p > 0.05. |

## Demo results (seed 14)

* Schema tests 11/11 PASS. SQL and pandas metrics identical. SRM p = 1.00.
* **Engagement 22.8% → 25.1% (+2.3 pp, CI +1.1 to +3.5, WIN)**; true lift +3 pp.
* Post PMPM −6.5 (no detectable effect; true −3% ≈ −23). CUPED −7.2 with a much narrower CI, still not significant at this n.
* Guardrails: complaints +0.09 pp (upper bound 0.37 pp < 0.5 pp margin → PASS). Opt-outs +0.44 pp: **p = 0.02 but PASS**, because the
  upper bound (0.82 pp) is inside the 1 pp margin. A difference can be statistically real and still acceptable.
* Defect run: the injected duplicate member, unknown LOB, orphan assignment and stray "holdout" arm are each caught by a named test.
  Losing 5% of treatment members upstream gives SRM p = 2e-4.

## Test data

Synthetic only (`data.generate`), because an experiment's truth and the injected defects must be known. See
[`data/README.md`](data/README.md) for mapping real extracts to the four `raw_*` sources.

## Tests

| File | What it proves |
|---|---|
| `test_t14_metric_layer.py` | build order and layering; unknown ref / cycle errors; clean data passes all tests; each injected defect caught; staging normalizes codes; unreached members stay in the ITT denominator; SQL = pandas and drift detection; readout recovers the true lift; two-proportion test = textbook; guardrail non-inferiority failure with p > 0.05; CUPED variance reduction and unbiasedness (12 null experiments); SRM detection |

## Caveats

* `dbt_lite` covers views, refs/sources, and four generic tests. There are no incremental models, macros, snapshots or
  docs site. Use real dbt when the project grows.
* **Peeking:** the readout assumes one pre-planned analysis at a fixed n (see t09 for power). Sequential looks need alpha spending.
* **Multiple metrics:** one primary metric decides. Secondary metrics are descriptive unless adjusted (Holm).
* CUPED needs a covariate measured **before** assignment; post-assignment covariates can absorb the effect.
* Randomization unit = member here. If outreach is assigned by care team, analyze by team (clusters, t09/t10).

## Explain it to the program team

"The new outreach script raised engagement from 22.8% to 25.1%, and we're confident the lift is between 1 and
3.5 points. We checked that it doesn't cause problems. Complaints and opt-outs rose slightly but stayed inside
the limits we agreed before the test. It's too early to see an effect on cost; even after using last year's
spending to cut the noise, a 3% cost change needs a larger or longer test. All numbers come from the same
tested definitions the dashboard uses."

## Files

```
metric_layer_dbt_style/  config.py  data.py  dbt_lite.py  methods.py  checks.py
models/  staging/  intermediate/  marts/  schema.yml
metrics.yaml
tests/   conftest.py  test_t14_metric_layer.py
```
