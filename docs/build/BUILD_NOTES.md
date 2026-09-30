# BUILD NOTES — t00 to t09 (one complete list)

Everything learned while building the templates, in one place. It covers decisions, the bugs that
building and testing exposed and how they were fixed, the headline results that each template's
tests lock in, known limitations, and open follow-ups.
Companion files: [`BUILD_STATUS.md`](BUILD_STATUS.md) (phase table), [`HANDOFF.md`](HANDOFF.md)
(how to continue; specs for t10–t15), [`../CONVENTIONS.md`](../CONVENTIONS.md) (rules).

**Status at time of writing (2026-09-29):** 10 templates, **244 tests green** under the orchestrator
(plus 12 orchestrator self-tests), on Windows / Python 3.11.9 in `.venv`.

**Provenance.** t00, t01, t02, t03 and t05 were built in an earlier cloud session. Their notes below
come from their READMEs, tests and git history (`6c6cdb8` P0, `7c4ad40` P1, `4bf7fb1`, `9d371ad` P2
partial). The build-time bug log for those templates did not survive the handoff. t04 and t06–t09 were
built in the local session; their notes include the defects found during the build.

---

## A. Project-wide notes

### A1. Decisions (by Nicole) that every template follows
| Topic | Decision |
|---|---|
| Stub style | Full implementation. No stubs, no xfail placeholders. |
| Methods | Every analytical step has a **STANDARD** (industry default) and an **ALTERNATIVE**, with the caveat that makes you switch, in the docstring and in the README "Method choices" table. Tests compare the two. |
| READMEs | One per template, plus `data/README.md` naming the public test dataset (URL, license, column mapping). |
| Orchestrator output | `orchestrator/run_status/`: `LATEST.md`, `latest.json`, `latest.html` committed; history gitignored. |
| Scope | 16 templates + glossary (31 entries), phases P0–P7. |
| Languages | Python, plus a DuckDB SQL twin where it teaches something. |

### A2. Build rules (how every template is put together)
1. `templates/tNN_<name>/` with a uniquely named package holding `config.py`, `data.py`, `methods.py` and `checks.py`.
2. **Never import across templates.** Shared helpers are *copied*, with a header saying "COPIED from …"
   (`sqltwin.py` from t05 into t04, t06 and t08; `prep.py` from t00 into t01 and t05). `orchestrator/tests/test_orch_independence.py` enforces this.
3. `run.py` has `demo()` (prints and writes to `outputs/`) and `selftest()` (plain asserts, PASS/FAIL lines).
4. `pytest.ini` with `testpaths = tests <pkg>` and `--doctest-modules`. `tests/conftest.py` does the `sys.path` insert and fixtures.
   **Never `from conftest import …`**, because it breaks runs from the repo root (fixed in `4bf7fb1`). Test files are named `test_tNN_*.py`.
5. `template.yaml` has flat keys only, using values from the plan's vocabularies (types A–G, intents, stacks, settings, workflow codes).
6. Functions that return dicts return plain `float`/`int`, because doctests otherwise print `np.float64(...)`.
7. Don't pass pandas `Period` columns to DuckDB; convert them to `month_start` timestamps.
8. Verify in order: `python -m pytest` in the folder, then `python orchestrator/run_all_tests.py --only tNN`, then the full run.
9. Published or licensed tables (HCC coefficients, NCQA value sets, contract terms) ship as clearly labelled `FAKE_` toys that point to the real source.
10. **Every README number comes from an actual run.** After changing a generator or seed, re-run the demo and update the README.

### A3. Repo-level fixes and environment notes
| # | What | Fix |
|---|---|---|
| R1 | Tests failed when run from the repo root because of `from conftest import …` | Helpers moved into fixtures (`4bf7fb1`). |
| R2 | `orchestrator/tests/test_orch_discover_runner.py::test_report_files` failed on Windows: `read_text()` with no encoding decoded UTF-8 as cp1252 | Added `encoding="utf-8"` to every `read_text` in that test (commit `5e62a51`). |
| R3 | No venv on the local machine | `py -3.11 -m venv .venv` + `pip install -r requirements-dev.txt` (lifelines built without issue). |
| R4 | statsmodels convergence warnings cluttered t08's pytest output | `filterwarnings = ignore:::statsmodels` in t08's `pytest.ini` (template-local). |
| R5 | Git on Windows flips `run_all_tests.sh` to mode 644 in the working tree | Left uncommitted on purpose. Committing it would drop the executable bit that CI on Linux needs. |
| R6 | GitHub remote was empty (`origin/main: gone`) at the first local push | The first push created `main` with the full history. Nothing was overwritten. |
| R7 | Loose build docs at the repo root | `HANDOFF.md` and `BUILD_STATUS.md` moved to `docs/build/` (with this file). Links updated in `README.md` and inside both files. |

### A4. Lessons that repeated across templates (apply them to t10–t15)
* **Put the truth in the generator.** Every analytical template since t06 generates a known effect,
  ultimate or rate, and tests whether the STANDARD method recovers it. This caught three real
  defects (t06 confounding, t07 truncation basis, t04 interval claim) that plain unit tests would have missed.
* **Test the claim, not the hope.** Check each docstring claim (e.g. "Jeffreys has better coverage")
  numerically before shipping it. When the data disagree, narrow the claim.
* **Like with like.** Most bugs were basis mismatches: truncated vs. untruncated (t07), paid vs.
  completed PMPM (t08), admin vs. hybrid (t04), matched on cost vs. on the trigger (t06).
* **Selection → regression to the mean** shows up in t06 (naive pre/post), t07 (vendor target from the
  participants' own baseline) and t09 (why cost endpoints need huge n). Each template has a check for it.

---

## B. Per-template notes

### t00 — claims_foundation (P1) · 30 tests
* **Purpose:** raw claims → analysis-ready data: claim versions, member-months, service categories,
  pharmacy reversals, data-quality checks. Other templates copy its `data.py` / `prep.py`.
* **Standard vs. alternative:** latest version (`collapse_versions_latest`) vs. net of deltas;
  daily-prorated member-months vs. mid-month anchor; claim-level vs. line-level service category;
  drop reversal pairs vs. signed sum.
* **Locked in by tests:** using the net method on a replacement feed double counts
  (`test_net_on_replacement_feed_double_counts`); Luhn-valid NPIs; overlapping spans don't double count days.
* **Checks:** SCH-004/VAL-005 (duplicate grain, NPI fan-out), CLN-014 (NPI format/Luhn),
  CLN-052/053/042 (ICD-10, HCPCS, NDC format), VAL-011/012/014/015 (date order), CLN-061 (overlapping
  enrollment), CLN-023 (negative paid), VAL-031 (runout dip).
* **Limitations:** code checks are *format* checks, not validity against the annual CMS code files.
  The service-category hierarchy is a convention, not a standard.

### t01 — utilization_profiling (P1) · 18 tests
* **Purpose:** utilization per 1,000, ALOS, readmissions, trend.
* **Standard vs. alternative:** exact Poisson CI vs. member cluster bootstrap. Under overdispersion
  the bootstrap is > 2× wider (test). Readmissions per index vs. per 1,000 member-years. Rolling-12 vs.
  YoY same month.
* **Checks:** VAL-015 (missing IP dates), VAL-031 (trailing-month dip), VAL-042 (< 30 index stays), ANL-020 (stays built from several claims: transfers, interim bills).
* **Limitations:** readmission is simplified (no planned-readmission exclusion, deaths, transfers or
  risk adjustment). CV of paid per member is ~1.8.

### t02 — hcc_risk_adjustment (P2) · 14 tests
* **Purpose:** HCC-style RAF scores with FAKE coefficients: hierarchies, V24/V28 blend, normalization
  and coding-pattern adjustment, a suspect list, and a member audit trail.
* **Standard vs. alternative:** published (FAKE) weights vs. ridge re-estimated weights. On held-out
  data R² is ~0.18 vs. ~0.06, because the synthetic population has a condition the V28-like model doesn't
  pay, plus frailty. Revenue still uses the published score.
* **Checks:** ANL-008 (prospective model year), RA-010 (non-acceptable sources), RA-020 (unmapped
  share → format problem), RA-030 (scores outside a plausible range: duplicated HCCs or wrong units).
* **Limitations:** face-to-face rules are simplified to a source code. The suspect list is for
  clinical *assessment*, never coding (RADV). Medicaid uses CDPS+Rx or a state model.

### t03 — predictive_risk_stratification (P2) · 11 tests
* **Purpose:** predict next-period admission, calibrate, size tiers to care-team capacity, segment.
* **Standard vs. alternative:** L2 logistic vs. HistGradientBoosting + isotonic. Logistic wins out of
  time (AUC 0.76 vs. 0.74, ~440 training events). Capacity tiers vs. k-means (silhouette ~0.15,
  so the segments are soft).
* **Results:** PPV at capacity: 27% of Tier 1 admitted (~4.5× base rate). Intercept drift −0.22, so
  recalibrate before using probabilities.
* **Checks:** ANL-003 (events per variable too low), ANL-005 (look-ahead leakage in feature names),
  ANL-006 (calibration intercept drift), DQ-010 (missing features).
* **Limitations:** k-means on 0/1 columns splits on the binary. Fairness (calibration by subgroup) → glossary G29.

### t04 — quality_measures_care_gaps (P2) · 45 tests
* **Purpose:** HEDIS/Core-Set-shaped measures from YAML (FAKE BCS, HbA1c test, WCV) → rates with CIs,
  a stability flag, benchmark position, and a member-level care-gap outreach list.
* **Standard vs. alternative:** Wilson vs. Jeffreys CI; administrative method (claims + lab feed) vs.
  hybrid (systematic sample of 411 + chart review).
* **Results (seed 11, 8,000 members):** BCS 54.9% (CI 51.3–58.6%, n = 710), HbA1c 69.4% admin → 79.3%
  hybrid, WCV 51.3%. 1,596 open gaps, 61 of them flagged "documented in chart only, request the record, don't call".
* **Bugs / corrections during the build:**
  1. The first Jeffreys docstring claimed "better coverage near 0/1". The exact-coverage computation
     showed Jeffreys is *not* closer to 95% on average. It only has a higher **worst-case** coverage
     near 0 for small n (30–60: ~89% vs. ~85%), and at n = 100 the ordering flips. Docstring, README and
     test were narrowed to exactly that.
  2. A CE selftest expected a 45-day gap to fail. The arithmetic (Jan 31 → Mar 18 = 45 days) passes,
     so the test was fixed to 46 days.
  3. The diabetes denominator (229) was smaller than the hybrid sample (411), so the "alternative"
     never sampled. Prevalence was raised and the demo population doubled.
  4. A heredoc test file failed to parse and carried a stub importing from conftest (forbidden, R1).
     It was rewritten with the file tool and the stub dropped.
* **Verified against:** statsmodels `proportion_confint` (Wilson/Jeffreys to 1e-9, with the
  boundary rule noted); DuckDB CE gaps-and-islands SQL = pandas for every member.
* **Checks:** QM-001 (code format), QM-002 (overlapping spans), QM-010 (runout < 90 days), QM-020
  (denominator < 30), QM-030 (value set missing), ANL-013 (benchmark vintage / methodology mix).
* **Limitations:** FAKE specs omit frailty/advanced illness, palliative care, death and product-line CE
  rules. The hybrid ignores the oversample. Core Set year *Y* ≈ services in *Y − 1*.

### t05 — tcoc_pmpm_mlr (P1) · 21 tests
* **Purpose:** PMPM (total, by LOB, by service category), high-cost truncation, trend decomposition,
  MLR and MLR impact, inpatient episodes.
* **Standard vs. alternative:** ratio-of-sums vs. mean-of-members PMPM (mean-of-members overweights
  short stays: test); p99 vs. fixed-attachment truncation; additive vs. log decomposition; regulatory
  vs. simple MLR; non-overlapping vs. overlapping episodes.
* **Locked in by tests:** the SQL "wrong join" anti-pattern understates PMPM; categories sum to the total.
* **Checks:** VAL-041 (claims without exposure), ANL-009 (top 1% > 20% of spend), VAL-042 (small exposure).
* **Limitations:** MLR omits credibility and multi-year rules. How the program fee is classified (claims /
  QI / admin) moves MLR as much as the savings do.

### t06 — roi_cost_offset (P3) · 25 tests
* **Purpose:** program savings per participant month (matched DiD), ROI, break-even PMPM, tornado.
* **Standard vs. alternative:** 1:1 logit-PS match (caliper 0.2 SD, exact on LOB + referral trigger) →
  member-month-weighted DiD with pair-clustered SE vs. a pair bootstrap (with P(savings), P(≥ break-even))
  and a two-part logit × Gamma model.
* **Results (seed 21, 10,000 members):** truth −$209 PMPM; naive pre/post −$318; matched DiD −$189
  (CI −330 to −48); bootstrap P(≥ break-even $175) 58%; ROI 0.08 at a $150 fee + $300 one-time.
* **Bugs / corrections during the build:**
  1. **Unmeasured confounding in the first generator.** Referral followed a cost spike that matching on
     noisy cost couldn't see, and the matched DiD was −$470 vs. a true −$166. Fix: the trigger is an
     observed admission (`ip_admit_flag`) and matching is exact on it. A test now shows that leaving
     the trigger out roughly doubles the savings (−$403 vs. −$217) with the trigger SMD ≈ 0.7.
  2. The cost tail was too light (top 1% = 5% of spend, no zero years). Risk spread was widened and
     ~15% near-non-users added. Next, non-users got huge claims because the amount was scaled by
     1/P(any cost); it is now scaled by the *nominal* probability. Result: 13% zero-year, top 1% ≈ 14–17%.
  3. Estimates were too noisy to show anything (CI half-width ~$350 on a $160 effect). Per-month
     σ 1.3 → 1.0, effect 15% → 20%, n 5,000 → 10,000. Checked over 5 seeds: every CI covered the truth.
  4. The $25 PMPM fee was unrealistic for intensive care management and gave ROI 4.0. It is now $150 + $300.
  5. The regression-to-the-mean check compared last-3 vs. all-6 pre months, which diluted the spike (1.11×).
     It now compares last-3 vs. earlier months as a ratio of sums (1.26× participants vs. 0.95× controls).
  6. Leftover scaffolding in `bootstrap_did` and an unnecessary `predict` branch were removed.
* **Checks:** ROI-001 (balance), ROI-002 (common support, 11% dropped), ROI-003 (pre-trends),
  ROI-004 (regression to the mean), ROI-005 (CI includes $0 / below break-even).
* **Limitations:** the two-part model is noisy here (−$119 vs. truth −$209). No re-matching inside the
  bootstrap. Effect durability beyond 6 months is assumed.

### t07 — vbc_contract_modeling (P3) · 32 tests
* **Purpose:** settle FAKE contracts (MSSP BASIC-A / ENHANCED-shaped, Medicaid HCBS sub-cap with corridors
  and stop-loss, CM fee + upside) into waterfalls that **close to the dollar** (ANL-015); scenario grid;
  Monte Carlo; MSSP PUF reproduction.
* **Standard vs. alternative:** deterministic reconciliation vs. Monte Carlo (member-level variation + trend shock).
* **Results (seed 7):** observed savings 4.4% (true 3.5%). ENHANCED pays $4.17M of $6.33M; BASIC $2.23M.
  Sub-cap margin −9.7%, and the corridor moves $228K of the loss to the plan. The CM vendor "saved" $529K
  but cost $879K (payer −$350K; VBC-003: target from participants' own baseline, 2.6× plan PMPM).
  With zero true savings, ENHANCED still pays out 18% of years and bills 17%.
* **Bugs / corrections during the build:**
  1. **Truncation basis mismatch.** Actuals were truncated at $150K but the benchmark wasn't, which
     produced 8.2% "savings" from 3.5% true and 72% chance payouts at zero savings. Fix: the benchmark is
     on the truncated basis. The generator solves for the raw mean whose truncated mean hits the target
     (closed-form lognormal limited expected value + `brentq`). A test asserts no phantom savings over 20 seeds.
  2. The first selftest assumed a 5% loss owes (1 − 0.675) × loss. The contract's 40% loss-rate floor
     applies, so it owes $200K. The test was corrected; it is a good demonstration of the floor.
  3. Sub-cap catastrophic cases were too frequent (−19% margin). Now 0.4% of members at $45–90K.
  4. Display fixes: "$-0" for a zero refund; the MSR check prints 2 decimals (1.98% vs. 2.00%).
* **Checks:** ANL-015 (closure), VBC-001 (missing terms), VBC-002 (MSR below random variation),
  VBC-003 (target basis), VBC-004 (corridor bands).
* **Limitations:** no benchmark rebasing, regional adjustment, risk-score caps, variable one-sided MSR,
  sequestration or advance payments. The PUF column names need verifying against each year's dictionary.

### t08 — claims_completion_forecast (P3) · 26 tests
* **Purpose:** lag triangle → chain-ladder / Bornhuetter-Ferguson ultimates and IBNR → completed PMPM →
  12-month forecast with intervals; backtests of both.
* **Standard vs. alternative:** chain ladder vs. BF for age ≤ 2; SARIMAX (log, AR(1) × seasonal diff + drift)
  vs. ETS (damped additive).
* **Results (seed 8):** last month paid PMPM $220 vs. true $588 (−63%); BF $587, CL $632. IBNR $14.94M vs.
  true $15.30M (−2.4%). Backtest |error| at age 0: CL 5.1%, BF 3.1%. Forecast holdout MAPE: SARIMAX 3.3%
  (coverage 6/6), ETS 4.5% (coverage 2/6).
* **Bugs / corrections during the build:**
  1. A test expected a lag-2 column in a triangle cut at February. The column correctly doesn't exist yet,
     so the test was fixed (the triangle only has observable lags).
  2. statsmodels warnings: see R4.
* **Checks:** IBNR-001 (immature months on CL), IBNR-002 (factor volatility), IBNR-003 (negative cells),
  IBNR-004 (processing speed-up: fires at −16% in the `speedup=0.5` scenario), FC-001 (< 24 months), FC-002 (not completed PMPM).
* **Limitations:** tail factor 1.0; no large-claim handling. ETS intervals under-cover. This is analytic
  completion, not a statutory reserve. SynPUF may lack a paid/processed date (verify in the codebook).

### t09 — study_design_power (P4) · 22 tests
* **Purpose:** n / power / MDE (statsmodels), design effect and cluster sizing, simulation power for any
  design, stratified permuted-block randomization, Markdown SAP generator (`sap/sap_template.md`).
* **Standard vs. alternative:** analytic power vs. `simulate_power` with the planned analysis.
* **Results:** readmission 18% → 14% needs 1,314 per arm (simulated 81%). MDE at 800 per arm is 12.9%.
  Practices of 40, ICC 0.02 → DEFF 1.78 → 59 practices per arm; ignoring DEFF → 53% power. Cost 10% of
  $1,200 (CV 2.5): the normal formula says 9,813 per arm but simulation gives 68%, so it takes ~13–14K per arm.
* **Bugs / corrections during the build:** the README first said "~15,000 per arm". The simulation grid
  (68% at 9,813; 85% at 14,719) supports ~13–14K, so the text was corrected. A data-README link to t13
  (not built yet) was replaced with the public source itself.
* **Verified against:** the arcsine formula exactly and the pooled Fleiss formula within 1%; simulated
  type-I error 5%; the cluster simulator's ICC recovered by ANOVA.
* **Checks:** PWR-001 (underpowered), PWR-002 (cluster design without DEFF), PWR-003 (randomization
  imbalance), PWR-004 (multiplicity), PWR-005 (skewed outcome), PWR-006 (few clusters).
* **Limitations:** the SAP is a skeleton (add ICH E9(R1) estimands, DMC plan, sign-off). ICC must come from prior data.

---

## C. Check-ID registry (t00–t09)

| Family | IDs → template |
|---|---|
| Schema / cleaning | SCH-004, CLN-014, CLN-023, CLN-042, CLN-052, CLN-053, CLN-061 → t00 |
| Validation | VAL-005, VAL-011, VAL-012, VAL-014 → t00 · VAL-015, VAL-031 → t00, t01 · VAL-041 → t05 · VAL-042 → t01, t05 (same family, small denominators) |
| Analysis | ANL-003/005/006 → t03 · ANL-008 → t02 · ANL-009 → t05 · ANL-013 → t04 · ANL-015 → t07 · ANL-020 → t01 |
| Data quality | DQ-010 → t03 |
| Template-specific | RA-010/020/030 (t02) · QM-001/002/010/020/030 (t04) · ROI-001…005 (t06) · VBC-001…004 (t07) · IBNR-001…004, FC-001/002 (t08) · PWR-001…006 (t09) |

## D. Open follow-ups

1. **Verify public-file headers** against current downloads: Medicaid Core Set rates (t04), MSSP ACO PUF
   (t07), the MEPS FYC file number for the year (t06), SynPUF paid/processed date availability (t08).
   Loaders match headers leniently, but the column tables in each `data/README.md` are marked "verify".
2. t04: model the hybrid oversample and chart-found exclusions if hybrid rates are ever reported.
3. t06: re-match inside the bootstrap (propensity-model uncertainty); a longer post window once data allow.
4. t08: large-claim handling and a non-1.0 tail factor; don't quote ETS intervals without a backtest.
5. **Uncommitted:** t06, t07, t08, t09, these notes and the `docs/build/` move are not committed yet
   (last commit `5e62a51`). `run_all_tests.sh` mode change: see R5.
6. `project_specs/` (outside the repo) holds byte-identical copies of `IMPLEMENTATION_PLAN.txt` and
   `glossary/niche_workflows_glossary.html`. Keep them as the original drafts, or delete them. That's your call.
7. Next build: t10 causal (DiD event study, PSM, AIPW, ITS), t11 survival, t12 PROs; then P5, P6 glossary, P7.
   Specs are in `HANDOFF.md`.
