# Test data — t14 metric layer and experiment readout

## Default: synthetic (no download)
`metric_layer_dbt_style/data.py` → `generate()` returns four raw tables shaped like warehouse extracts, with messy
codes, string-typed columns and a known experiment effect. `inject_issues=True` adds the defects the schema tests
catch; `srm_drop=` silently loses treatment members (sample ratio mismatch).

## Using real extracts
Map your tables to the four sources the staging models read (or edit the staging SQL):

| Source | Needed columns | Typical origin |
|---|---|---|
| `raw_members` | `member_key`, `lob`, `enroll_start`, `age` | eligibility extract (t00) |
| `raw_assignments` | `member_key`, `variant`, `assigned_at` | experiment / randomization log |
| `raw_outreach` | `member_key`, `attempt_ts`, `outcome` | care-management or call-center system |
| `raw_claims_monthly` | `member_key`, `month_start`, `paid`, `period` (`pre`/`post`) | claims PMPM by member-month (t05; completed with t08) |

With a warehouse: point `dbt_lite.build` at a DuckDB connection that attaches your files, or copy `models/` and
`schema.yml` into a real dbt project (`ref()` / `source()` syntax is the same; declare the sources in a `sources.yml`).

No public dataset contains member-level randomized outreach experiments, so synthetic data is the test source.

Files in this folder are gitignored.
