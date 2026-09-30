# Statistical Analysis Plan — $study_title

**Version:** $version  ·  **Date:** $date  ·  **Author:** $author
**Status:** DRAFT — to be finalized and signed before any outcome data are unblinded.

## 1. Objective
$objective

## 2. Design
* **Type:** $design_type
* **Unit of randomization:** $unit_of_randomization
* **Stratification:** $strata (permuted blocks of random size $block_sizes)
* **Arms:** $arms
* **Follow-up:** $follow_up

## 3. Endpoints
| Role | Endpoint | Definition | Summary measure |
|---|---|---|---|
| Primary | $primary_endpoint | $primary_definition | $primary_measure |
| Secondary | $secondary_endpoint | $secondary_definition | $secondary_measure |
| Exploratory | $exploratory_endpoint | $exploratory_definition | $exploratory_measure |

## 4. Sample size justification
$sample_size_text

| Quantity | Value |
|---|---|
| Two-sided alpha | $alpha |
| Power | $power |
| Assumed control rate | $p_control |
| Minimum clinically important rate (treatment) | $p_treat |
| n per arm (individual randomization) | $n_individual |
| Design effect (cluster size $cluster_size, ICC $icc) | $design_effect |
| Clusters per arm / patients per arm | $clusters_per_arm / $n_cluster |
| Inflated for $attrition_pct attrition | $n_final per arm |

## 5. Analysis methods
* **Primary:** $primary_analysis
* **Secondary:** $secondary_analysis
* **Covariates (pre-specified):** $covariates
* **Estimand:** $estimand

## 6. Missing data
$missing_data

## 7. Multiplicity
$multiplicity

## 8. Subgroups (pre-specified, interpreted as exploratory)
$subgroups

## 9. Interim analyses and stopping
$interim

## 10. Deviations
Any change to this plan after unblinding is reported as a deviation with its rationale.
