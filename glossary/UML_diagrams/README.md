# UML diagrams

One draw.io UML class/component diagram per template, plus a toolkit overview. Each diagram shows
the code as it is today (top) and **where new models can be added and what they are for** (the
green dashed lanes at the bottom).

| File | What it shows |
|---|---|
| `00_toolkit_overview.drawio` | All 16 templates by area, the conceptual hand-offs between them, proposed new templates |
| `tNN_<name>.drawio` | That template's package: `run.py`, `Config`, `data.py`, `methods.py`, `checks.py` + `Finding`, extra modules, spec/SQL folders, and its extension points |
| `svg/*.svg` | The same diagrams exported to SVG so GitHub can display them (below) |

## Diagrams

Click an image to open it full size. The `.drawio` files are the editable sources.

### Toolkit overview

[![00_toolkit_overview](svg/00_toolkit_overview.svg)](svg/00_toolkit_overview.svg)

### Templates

<details><summary><b>t00_claims_foundation</b></summary>

[![t00_claims_foundation](svg/t00_claims_foundation.svg)](svg/t00_claims_foundation.svg)

</details>
<details><summary><b>t01_utilization_profiling</b></summary>

[![t01_utilization_profiling](svg/t01_utilization_profiling.svg)](svg/t01_utilization_profiling.svg)

</details>
<details><summary><b>t02_hcc_risk_adjustment</b></summary>

[![t02_hcc_risk_adjustment](svg/t02_hcc_risk_adjustment.svg)](svg/t02_hcc_risk_adjustment.svg)

</details>
<details><summary><b>t03_predictive_risk_stratification</b></summary>

[![t03_predictive_risk_stratification](svg/t03_predictive_risk_stratification.svg)](svg/t03_predictive_risk_stratification.svg)

</details>
<details><summary><b>t04_quality_measures_care_gaps</b></summary>

[![t04_quality_measures_care_gaps](svg/t04_quality_measures_care_gaps.svg)](svg/t04_quality_measures_care_gaps.svg)

</details>
<details><summary><b>t05_tcoc_pmpm_mlr</b></summary>

[![t05_tcoc_pmpm_mlr](svg/t05_tcoc_pmpm_mlr.svg)](svg/t05_tcoc_pmpm_mlr.svg)

</details>
<details><summary><b>t06_roi_cost_offset</b></summary>

[![t06_roi_cost_offset](svg/t06_roi_cost_offset.svg)](svg/t06_roi_cost_offset.svg)

</details>
<details><summary><b>t07_vbc_contract_modeling</b></summary>

[![t07_vbc_contract_modeling](svg/t07_vbc_contract_modeling.svg)](svg/t07_vbc_contract_modeling.svg)

</details>
<details><summary><b>t08_claims_completion_forecast</b></summary>

[![t08_claims_completion_forecast](svg/t08_claims_completion_forecast.svg)](svg/t08_claims_completion_forecast.svg)

</details>
<details><summary><b>t09_study_design_power</b></summary>

[![t09_study_design_power](svg/t09_study_design_power.svg)](svg/t09_study_design_power.svg)

</details>
<details><summary><b>t10_causal_impact_evaluation</b></summary>

[![t10_causal_impact_evaluation](svg/t10_causal_impact_evaluation.svg)](svg/t10_causal_impact_evaluation.svg)

</details>
<details><summary><b>t11_survival_time_to_event</b></summary>

[![t11_survival_time_to_event](svg/t11_survival_time_to_event.svg)](svg/t11_survival_time_to_event.svg)

</details>
<details><summary><b>t12_patient_reported_outcomes</b></summary>

[![t12_patient_reported_outcomes](svg/t12_patient_reported_outcomes.svg)](svg/t12_patient_reported_outcomes.svg)

</details>
<details><summary><b>t13_provider_benchmark_profiling</b></summary>

[![t13_provider_benchmark_profiling](svg/t13_provider_benchmark_profiling.svg)](svg/t13_provider_benchmark_profiling.svg)

</details>
<details><summary><b>t14_metric_layer_dbt_style</b></summary>

[![t14_metric_layer_dbt_style](svg/t14_metric_layer_dbt_style.svg)](svg/t14_metric_layer_dbt_style.svg)

</details>
<details><summary><b>t15_ops_lifecycle_prior_auth</b></summary>

[![t15_ops_lifecycle_prior_auth](svg/t15_ops_lifecycle_prior_auth.svg)](svg/t15_ops_lifecycle_prior_auth.svg)

</details>

## How to open

* **draw.io desktop**, or **app.diagrams.net** → *File → Open from → Device*.
* **VS Code**: the *Draw.io Integration* extension (`hediet.vscode-drawio`) opens `.drawio` files in an editor tab.
* **Claude Code**: the draw.io MCP server (`@drawio/mcp`) is registered at user scope. Restart the session to load its tools.

## How to read a template diagram

| Element | Meaning |
|---|---|
| Blue row `[S]` | Standard method: the README default |
| Orange row `[A]` | Alternative method: switch to it when the README's named caveat applies |
| Red row `[naive]` | Kept only to show the bias. Never report it |
| White row | Helper or shared step |
| Green dashed lane | **Extension point**. Its arrow points at the module the new code goes into |
| `«model»` / `«alternative»` / `«loader»` / `«check»` / `«spec»` / `«module»` | Kind of extension: new analysis / third method option / new data source / new assumption check / new YAML/CSV spec (no code) / new file in the package |
| *Pairs with* | Related template (`tNN`) or glossary entry (`gNN`) that already holds part of the method |

## Regenerating

The diagrams are generated, so they stay in step with the code:

```powershell
python glossary\UML_diagrams\build_uml_diagrams.py
```

* `build_uml_diagrams.py` reads each template's modules, public functions, `Config` and `Finding`
  fields, and spec folders with `ast`. It uses only the standard library.
* `extensions.py` holds the hand-curated parts: which functions are standard, alternative or naive
  (taken from each README's *Method choices* table), the extension points, and the proposed new
  templates. Edit this file to add or change an extension idea, then re-run the generator.
* If draw.io desktop is installed, the generator also exports `svg/` for the README (light theme,
  white background, without draw.io's PNG text fallbacks, so each file stays near 100 KB). It looks in the
  usual install paths. Set `DRAWIO=<path to draw.io executable>` if yours is elsewhere. Without it,
  the SVG step is skipped and the old `svg/` files go stale.

Manual edits made in draw.io are overwritten on regeneration. Put lasting changes in `extensions.py`.
