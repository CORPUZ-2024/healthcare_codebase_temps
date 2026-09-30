# t04 — Quality measures (HEDIS / Core Set style) and care gaps

Turn a measure specification into a rate a plan or state can report, and into the member-level
list of open care gaps that care managers actually work. Each measure is a YAML file
(population, continuous enrollment, event-based denominator, exclusions, numerator lookback,
allowed data sources). One engine runs them all. Rates come with confidence intervals, a
small-denominator flag and a position against published Medicaid Core Set state rates.

> **All measure specs and value sets here are FAKE.** They are shaped like HEDIS / Core Set measures
> but are simplified and incomplete. See [`reference/README.md`](reference/README.md) for the real sources.

| | |
|---|---|
| **Workflow type** | C — Quality & clinical measurement (C1 measure rates, C2 care-gap lists) |
| **Intent** | MEASURE |
| **Volume** | M. Demo: 8,000 members, ~30K events, 3 measures. SQL twin for continuous enrollment and the rate funnel. |
| **Stack** | pandas + scipy + PyYAML (+ DuckDB SQL twin) |
| **JD link** | "HEDIS / Medicaid Core Set measures", "care gap closure", "quality reporting" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 45 tests (incl. doctests)
```

## Workflow

1. Load specs (`measures/*.yaml`) and value sets (`reference/FAKE_value_sets.csv`).
2. **Tag events** with value sets: normalize codes (`e11.9` → `E119`) and join on *code system + code*.
3. **Population:** age on Dec 31 of the measurement year (MY), sex.
4. **Continuous enrollment:** whole CE period, ≤ 1 gap of ≤ 45 days **per year**, enrolled on Dec 31.
5. **Event-based eligibility** (e.g. diabetes on ≥ 2 distinct dates in MY-1/MY).
6. **Exclusions** (hospice in MY, bilateral mastectomy ever) → denominator.
7. **Numerator** inside the lookback, from the sources the method allows.
8. **Rate + CI + stability flag**, **hybrid** rate where the spec allows it, **benchmark position**.
9. **Care-gap list:** denominator members without a numerator event, with last service seen and a note.

Every member gets a `status` (`NOT_CE`, `NO_DENOM_EVENT`, `EXCLUDED`, `MET`, `OPEN_GAP`): the audit trail.

## Measures (FAKE specs)

| File | Shaped like | Population | Numerator window | Hybrid? |
|---|---|---|---|---|
| `FAKE_BCS.yaml` | HEDIS BCS-E / Core Set BCS-AD | women 52–74, CE MY-1 + MY | mammogram Oct 1 MY-2 → Dec 31 MY (27 months) | no |
| `FAKE_HBA1C_TEST.yaml` | retired HEDIS CDC HbA1c testing (current: GSD / HBD-AD) | 18–75 with diabetes on 2 dates | HbA1c test in MY; claims + lab feed | yes (chart) |
| `FAKE_WCV.yaml` | HEDIS WCV / Core Set WCV-CH | 3–21 | well-care visit in MY | no |

Add a measure by copying a YAML file. Offsets are months relative to Jan 1 of MY.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Rate CI | `ci_wilson` — Wilson score interval | `ci_jeffreys` — Beta(k+½, n−k+½) quantiles with the boundary rule | rates near 0% or 100% on small denominators (n ≈ 30–60): Wilson's worst-case coverage dips to ~85%, Jeffreys' to ~89% (`test_coverage_near_nominal_and_worst_case_near_zero`). Also for one-sided "below target" statements. Not better on average. Check your own n with `interval_coverage`. |
| Numerator data | `evaluate_measure` — **administrative**: claims + standard supplemental (lab feed) for the whole denominator | `hybrid_rate` — **hybrid**: systematic sample of 411, admin hits **or** medical-record (chart) hits | the spec allows hybrid and services are known to be missing from claims (point-of-care tests, bundled payments). Typically several points higher (here 69% → 79%). CI is on the sample, not the denominator. **Not comparable** to admin rates or admin benchmarks. |
| Small denominators | `rate_summary` flags `NR` when n < 30 | — | always on. Pool years or groups instead of ranking. |

## Test data

* **Synthetic (default):** `data.generate` — members, enrollment spans with realistic breaks and
  overlaps, events tagged CLAIM / LAB / CHART, some codes with dots/lower case.
* **Public benchmark:** Medicaid & CHIP **Core Set state rates** (data.medicaid.gov) via
  `data.load_core_set_rates`. The demo uses `generate_fake_core_set` with the same headers.
* **Value sets:** NCQA's free Core Set value-set directory (registration) or VSAC.
  Details and links: [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t04_engine.py` | CE known answers (45 vs 46 days, gap at start, adjacent and nested spans, anchor, one gap *per year*); BCS window edges, age on Dec 31, hospice-window, mastectomy with a dotted lower-case code; diabetes 2-date rule; LAB counts in admin, CHART doesn't; join on system + code; funnel identities |
| `test_t04_rates_gaps.py` | Wilson/Jeffreys equal statsmodels; stay in [0, 1]; exact coverage; stability flag; systematic sample; hybrid ≥ admin on the same sample; hybrid refused for admin-only measures; gap list = open gaps only; gap notes; Core Set loader (percent → proportion, blanks dropped); benchmark quartile |
| `test_t04_checks_sql.py` | every check fires on bad input and stays quiet on good; DuckDB CE SQL = pandas for every member; rate SQL = pandas |

## Caveats

* **FAKE specs.** Real HEDIS measures have more exclusions (frailty + advanced illness for 66+,
  palliative care, death), specific visit-type rules for event-based denominators, and
  product-line-specific CE rules. Use the NCQA technical specifications for anything reported.
* **Runout.** A rate computed in January understates the numerator. `check_runout` errors on
  < 90 days after MY end; label earlier runs "preliminary".
* **Hybrid** here ignores the oversample and chart-found exclusions.
* **Benchmarks:** Core Set year *Y* mostly reflects services in calendar year *Y−1* (`ANL-013`);
  compare admin to admin and Medicaid to Medicaid. State rates are not plan rates.
* **Gap lists are operational PHI.** Outputs are gitignored; don't paste them anywhere.

## Explain it to Finance

"For 2025, 55% of eligible women were screened for breast cancer (95% CI 51–59%). That is
upper-middle among states reporting to Medicaid. 710 women were in the measure after we removed
hospice and prior mastectomy, and 320 of them still have an open gap. Those are on the outreach
list. Our diabetes testing rate is 69% from claims and lab feeds. A chart review of 411 members
suggests the true rate is about 79%, so part of the 'gap' is missing data, not missing care.
Fixing the data feed is cheaper than calling members."

## Files

```
quality_measures_care_gaps/  config.py  data.py  methods.py  checks.py  sqltwin.py (copied from t05)
measures/   FAKE_BCS.yaml  FAKE_HBA1C_TEST.yaml  FAKE_WCV.yaml
reference/  FAKE_value_sets.csv  README.md
sql/        01_continuous_enrollment.sql  02_measure_rate.sql
tests/      conftest.py  test_t04_engine.py  test_t04_rates_gaps.py  test_t04_checks_sql.py
```
