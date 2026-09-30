# t12 — Patient- and caregiver-reported outcomes (PROs)

Score questionnaires exactly as their manuals say, check that the items hang together, and estimate
how a program changed scores over time when some people stop answering. Instruments are YAML specs:
item list, range, reverse-worded items, prorating rule, severity bands and MCID. One scoring function
serves them all. The demo is a caregiver-support trial with a FAKE 12-item burden scale (primary) and
the public-domain PHQ-9 (secondary) at baseline, 3 and 6 months.

> **Licensing:** only public-domain instruments (PHQ-9) or FAKE stand-ins are shipped. Licensed scales such
> as the Zarit Burden Interview must not be reproduced or scored without a license (`check_license`, PRO-004).
> Item **wording** is never stored, only item ids.

| | |
|---|---|
| **Workflow type** | F — Evidence (F4 patient-/caregiver-reported outcomes: scoring, change, MCID responders, mixed models) |
| **Intent** | MEASURE, EXPLAIN |
| **Volume** | XS. Demo: 600 caregivers × 3 visits × 21 items. |
| **Stack** | statsmodels + PyYAML |
| **JD link** | "patient-reported outcomes", "caregiver outcomes", "longitudinal mixed models" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 15 tests (incl. doctests), ~8 s
```

## Instruments (`instruments/*.yaml`)

| File | Items | Range | Reverse | Prorate | Bands | MCID |
|---|---|---|---|---|---|---|
| `PHQ9.yaml` (public domain) | 9 × 0–3 | 0–27 | — | ≤ 1 missing | 5/10/15/20 | 5 |
| `FAKE_CAREGIVER_BURDEN.yaml` | 12 × 0–4 | 0–48 | cb04, cb08, cb11 | ≤ 1 missing | 13/25/37 (FAKE) | 4 (FAKE) |

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Effect over time | `mixed_model_effect`: MMRM-style linear mixed model on post-baseline scores (baseline covariate, visit × arm, random intercept; statsmodels MixedLM) | `ancova_change`: change ~ baseline + arm on completers · `gee_effect`: exchangeable GEE | ANCOVA for a simple 2-arm study with little dropout; GEE for population-average effects when dropout is completely at random. Under MAR dropout both use less information, and unweighted GEE is formally biased. |
| Responders | `responder_analysis`, observed only | `missing_as_nonresponder=True` | the conservative convention (regulatory PRO labeling). |
| Single-arm "improvement" | `arm_mean_change`, mixed-model estimate | completers' raw mean | never report the completers' mean alone when dropout is related to how people are doing. |

## Findings the tests lock in

| Finding | Evidence |
|---|---|
| Skipped items are **prorated, never zero-filled**; two missing PHQ-9 items = unscorable | `test_phq9_scoring_and_prorating`, `test_never_fill_skipped_items_with_zero` |
| Cronbach's alpha 0.86 with reverse items flipped; **0.43 without**. A reverse-scoring error shows up as low reliability (PRO-005) | `test_reliability_requires_reverse_scoring` |
| Between-arm effect at month 6: mixed model, ANCOVA and GEE all cover the truth; the mixed model uses 516 people vs. 451 completers | `test_between_arm_effect_all_methods_cover_truth`, `test_mixed_model_uses_partial_completers` |
| **Completers overstate within-arm improvement** (mean bias < −0.3 points over 10 trials); the mixed model is at least 2× closer | `test_completers_overstate_within_arm_improvement` |

## Demo results (seed 12)

* True effect −4.11 points. Mixed model −4.46 (CI −5.51 to −3.40), ANCOVA −4.30, GEE −4.46.
* Program-arm improvement: completers −5.90, mixed model −5.68, true −5.54.
* Responders (≥ 4 points): 61% vs. 39% (+22%; +20% counting dropouts as non-responders).
* PHQ-9: −1.35 (CI −2.01 to −0.69). **Statistically clear but well below the 5-point MCID.**
* Month-6 scorable questionnaires: 83% program vs. 76% usual care, flagged by `PRO-002`.

## Build finding (documented in the data module)

The first dropout model made caregivers with a high burden **level** more likely to drop out. It barely
biased change scores: dropouts started higher and regressed toward the mean, so their change matched the
completers'. The realistic harmful mechanism is dropout among people who are **getting worse**. The generator
now uses that (still missing at random, since it depends on observed scores), and a test guards it.

## Test data

* **Synthetic (default):** `data.generate_trial`: item-level answers from latent trajectories, 3% skipped items,
  MAR dropout, and the complete data kept as the truth.
* **Public:** AHRQ MEPS SF-12 summary scores, PHQ-2 and K6 (`data.load_meps_pro`). See [`data/README.md`](data/README.md).

## Caveats

* statsmodels' random intercept is compound symmetry, not the unstructured covariance of a true MMRM
  (R `mmrm`, SAS PROC MIXED). With 2 post-baseline visits they are close; with more visits, use an unstructured model.
* MAR is untestable. Add a tipping-point / delta-adjusted sensitivity analysis for real reports.
* MCIDs depend on population and method (anchor vs. distribution). Cite the source for the one you use.
* Multiple instruments means multiple comparisons. Pre-specify one primary (see t09's SAP).

## Explain it to the program team

"Caregivers in the program reported about 4.5 points less burden at six months than those in usual care, on a
48-point scale, and 61% improved by a meaningful amount against 39%. More usual-care caregivers stopped
answering, mostly those who were struggling. So we used a model that learns from everyone's earlier answers;
looking only at people who finished would make the program arm look slightly better than it is. Depression
scores also fell, but by less than the change patients notice."

## Files

```
patient_reported_outcomes/  config.py  data.py  methods.py  checks.py
instruments/  PHQ9.yaml  FAKE_CAREGIVER_BURDEN.yaml
tests/  conftest.py  test_t12_pro.py
```
