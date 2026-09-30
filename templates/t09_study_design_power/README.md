# t09 — Study design, power and statistical analysis plan

Before a program launches, answer three questions. How many people do we need to see the effect
that matters? How do we assign them fairly? What exactly will we analyze? This template computes
sample size, power and minimum detectable effect (MDE) analytically. It adjusts for cluster
randomization, checks every number by simulation, produces a stratified block randomization list,
and renders a Markdown statistical analysis plan (SAP).

| | |
|---|---|
| **Workflow type** | F — Evidence (F1 study design: power / MDE, randomization, SAP skeleton) |
| **Intent** | DESIGN |
| **Volume** | XS. No data needed. |
| **Stack** | statsmodels + scipy |
| **JD link** | "study design and power analysis", "randomization", "statistical analysis plans" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/ (SAP_draft.md, randomization_list.csv, power tables)
python run.py --selftest
python -m pytest           # 22 tests (incl. doctests)
```

## Workflow (FAKE scenario: transitional care after discharge, primary endpoint 30-day readmission)

1. **Primary endpoint power**: `power_two_proportions` (18% → 14%) → n per arm, power curve, MDE for the n you actually have.
2. **Check by simulation** (`simulate_power` + `sim_two_proportions`).
3. **Cluster randomization** (practices): `design_effect`, `cluster_sample_size`, then simulate with a
   cluster-level analysis (`sim_cluster_proportions`).
4. **Secondary** continuous endpoint (`power_two_means`) and a **skewed cost** endpoint (`sim_cost_did`).
5. **Randomize** (`stratified_block_randomize`: strata = site × risk tier, random blocks of 2/4).
6. **SAP** (`render_sap` fills `sap/sap_template.md`; refuses to render with a blank field).

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Power / sample size | statsmodels `NormalIndPower` (Cohen's h) / `TTestIndPower` | `simulate_power` with the planned analysis | clusters (especially < 8–10 per arm), skewed outcomes, DiD, unequal attrition, or any analysis without a closed form. Here the normal formula says 9,813 per arm for a 10% cost reduction, **but simulating the same analysis gives 68% power, not 80%.** |
| Clustering | DEFF = 1 + (m − 1) × ICC | simulation with beta-binomial clusters | few clusters, or unequal cluster sizes. Ignoring DEFF drops simulated power to **53%**; with it, 79%. |
| Allocation | stratified permuted blocks (random sizes 2/4) | — (simple randomization) | only very large trials; blocks guarantee balance within ±2 per stratum. |

## Demo results

| | Value |
|---|---|
| Readmission 18% → 14%, individual randomization | 1,314 per arm (simulated power 81%) |
| MDE with only 800 per arm | 18% → 12.9% (28% relative reduction) |
| Practices of 40, ICC 0.02 | DEFF 1.78 → 59 practices / 2,360 patients per arm (2,623 with 10% attrition) |
| Burden score, 4 points (SD 14) | 194 per arm |
| Cost, 10% of $1,200 PMPM, CV 2.5 | simulated power 68% at 9,813 and 85% at 14,719 per arm, so ~13,000–14,000 per arm. Use readmission as the primary endpoint. |

## Test data

None needed. The demo's enrollment list is generated in `run.py` (`demo_units`). See [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t09_power.py` | n equals the arcsine formula exactly and the pooled Fleiss formula within 1%; power ↔ n and MDE ↔ n are inverses; simulation matches analytic power and has a 5% type-I error; DEFF and cluster sizing; **ignoring clustering loses power**; the cluster simulator has the stated ICC; **normal formula is optimistic for skewed cost**; power curve monotone |
| `test_t09_randomization_sap_checks.py` | every stratum within ±2; complete blocks exactly balanced; deterministic per seed; balance check catches a bad list; SAP renders and refuses blanks; shipped template has every section; each check fires |

## Caveats

* The **effect size** is a decision, not a statistic. Use the smallest effect worth paying for (see t06's break-even), not the vendor's claim.
* **ICC** values come from prior data or published estimates. Try a range; the design is very sensitive to it.
* Binary endpoints with rare events (< 5%) or very small n: use exact methods or simulation.
* The SAP template is a skeleton. Add estimand details (ICH E9(R1)), a data-monitoring plan and sign-off for real studies.

## Explain it to Finance

"To show that the program cuts readmissions from 18% to 14%, we need about 1,300 patients per group,
or about 60 practices per group if we randomize practices instead. Practices' patients resemble each
other, so each practice counts for less. Trying to prove cost savings directly would need roughly
14,000 per group, because costs swing so much from person to person. So we measure readmissions
and translate them into dollars (t06)."

## Files

```
study_design_power/  config.py  methods.py  checks.py
sap/        sap_template.md
tests/      conftest.py  test_t09_power.py  test_t09_randomization_sap_checks.py
```
