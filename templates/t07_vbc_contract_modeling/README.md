# t07 — Value-based contract modeling

Model what a value-based contract actually pays, before you sign it and after the year closes.
Each contract is a YAML file of its terms. One engine settles it into a waterfall that has to
**close to the dollar**. A scenario grid shows where the corridor edges and caps bite, and a Monte
Carlo shows how often random variation alone produces a payout or a bill. The engine also
recomputes published MSSP ACO results from the public file.

> **All contract terms here are FAKE.** They are shaped like MSSP BASIC/ENHANCED, a Medicaid sub-capitation
> and a care-management vendor deal. Real MSSP rules: 42 CFR 425 and the CMS *Shared Savings and Losses and
> Assignment Methodology Specifications*.

| | |
|---|---|
| **Workflow type** | D — Health economics & VBC (D4 contract modeling, D5 reconciliation replication vs. MSSP PUF) |
| **Intent** | VALUE, MEASURE |
| **Volume** | XS. Demo: one ACO year (12,000 beneficiaries), 1,500 sub-cap members, 400 CM participants, 2,000 Monte Carlo years. |
| **Stack** | pandas + numpy + scipy + PyYAML |
| **JD link** | "value-based contract modeling", "shared savings / risk arrangements" |

## Quick start

```bash
pip install -r requirements.txt
python run.py              # demo -> outputs/
python run.py --selftest
python -m pytest           # 32 tests (incl. doctests)
```

## Contracts (`contracts/*.yaml`, FAKE)

| File | Type | Key terms |
|---|---|---|
| `FAKE_MSSP_BASIC_A.yaml` | `shared_savings`, one-sided | MSR 3%, sharing 40% × quality, cap 10% of benchmark, $150K truncation |
| `FAKE_MSSP_ENHANCED.yaml` | `shared_savings`, two-sided | MSR/MLR 2%, sharing 75% × quality, quality gate 0.40, loss rate 1 − sharing bounded 40–75%, caps 20% / 15% |
| `FAKE_MEDICAID_SUBCAP.yaml` | `subcap` | $410 PMPM HCBS cap; provider keeps 100% of margin within ±3%, 50% within 3–10%, 0% beyond; stop-loss 80% above $40K/member |
| `FAKE_CM_FEE_UPSIDE.yaml` | `fee_upside` | $150 PMPM fee; 50% of savings **above** a 2% threshold vs. baseline × 1.05; upside capped at 100% of fees; 2 of 3 quality measures or refund 10% of fees |

## Workflow

1. Load and validate contracts (`check_contract_fields`, `check_corridors`).
2. **Settle** each contract (`reconcile_shared_savings`, `reconcile_subcap`, `reconcile_fee_upside`) → waterfall + identities.
3. **Close** every reconciliation (`check_reconciliation_closes`, ANL-015): truncated = raw − truncation,
   gross = benchmark − actual, ACO + payer = gross, provider result = margin + settlement, and so on.
4. **Scenario grid**: true savings % × quality → ACO $ per beneficiary-year.
5. **Monte Carlo**: 2,000 simulated years of member-level random variation plus trend risk → P(shared savings),
   P(owing losses), expected payout, 5th–95th percentile.
6. **Replicate the MSSP PUF** (`reproduce_mssp_puf`) to prove the engine matches published results.

## Method choices

| Step | Standard (default) | Alternative | Switch when… |
|---|---|---|---|
| Settlement | `reconcile_*`: deterministic, closes to the dollar | `monte_carlo_shared_savings`: distribution of outcomes | deciding **whether** to take risk or which track to choose. The deterministic answer uses one expected cost; a real year is a draw. Here, with **zero** true savings, the two-sided contract still pays shared savings 18% of the time and bills losses 17% of the time. |
| What-if | `scenario_grid`: exact outcomes on a grid | Monte Carlo at each grid point | the grid shows *where* thresholds are; Monte Carlo shows *how likely* you are to land on each side. |
| Corridor sizing | contract MSR | `msr_for_confidence`: z × CV / √n | checking whether a proposed MSR covers random variation for your population size (`check_msr_vs_random_variation`). |

## Demo results (seed 7, 3.5% true savings, quality 0.88)

* **Observed savings rate 4.4%** (random variation around the true 3.5%). BASIC pays the ACO $2.23M;
  ENHANCED pays $4.17M of the $6.33M gross.
* **Sub-cap:** the cap is too low, and the margin is −9.7% after stop-loss. The corridor moves $228K of
  the $661K loss to the plan.
* **CM vendor:** 7.8% gross savings ($529K). But fees ($682K) plus upside ($197K) exceed it, so the
  **payer loses $350K net**. `VBC-003` flags the likely cause: the target is the participants' own
  high-cost baseline (2.6× the plan PMPM), so regression to the mean looks like savings (see t06).
* **Monte Carlo:** with 3.5% true savings, the ENHANCED ACO gets shared savings in 77% of years (BASIC
  60%, because of its higher 3% MSR).

## Test data

* **Synthetic (default):** ACO beneficiary-year costs, sub-cap members and CM participants (`data.py`).
  The benchmark is on the **truncated** basis: the generator solves for the raw mean whose truncated
  mean equals benchmark × (1 − savings), using the closed-form lognormal limited expected value.
* **Public:** CMS **MSSP ACO Performance Year Financial and Quality Results** PUF → `data.load_mssp_puf`,
  `methods.reproduce_mssp_puf`. Details: [`data/README.md`](data/README.md).

## Tests

| File | What it proves |
|---|---|
| `test_t07_settlement.py` | ENHANCED known answers (at/under the MSR, quality gate, loss-rate floor/ceiling, savings cap, loss cap); one-sided never owes; first-dollar vs. above-MSR = 3× at 3%/2%; truncation; truncated-mean formula vs. simulation; **no phantom savings with zero true savings**; corridor bands; sub-cap and fee+upside known answers, quality gate, upside cap; grid monotone; Monte Carlo matches normal theory; MSR cuts chance payouts |
| `test_t07_checks_puf.py` | every check fires correctly; closure check catches a tampered identity; PUF loader parses `$`, `,` and 0–100 percents; PUF reproduction matches gross and earned savings |

## Caveats

* **Truncated vs. untruncated.** The first version of this template compared truncated actuals with an
  untruncated benchmark. It produced 8% "savings" from nothing, and a zero-savings ACO cleared the MSR
  72% of the time. Both sides must be on the same basis.
* Real MSSP adds benchmark rebasing, regional adjustment, risk-score caps, prior-savings adjustment,
  variable one-sided MSR by population size, sequestration, and advance investment payments. None are modeled.
* **Sub-cap settlement order** (stop-loss before or after the corridor) varies by contract. Read yours.
* Monte Carlo assumes independent member costs from one lognormal family plus a single trend shock.
  Real years also have benchmark and quality uncertainty.

## Explain it to Finance

"Under the two-sided track we'd have earned $4.2M this year: 4.4% savings on a $144M benchmark,
shared at 66% because of our quality score. That number is one draw. If we changed nothing at all,
we'd still get a check about one year in six and owe a bill about one year in six. With our real
3.5% savings, we'd get a check in about three years out of four. The care-management vendor saved
$529K against its target but cost us $879K, and its target is inflated because members were enrolled
after a bad year. Renegotiate the target before renewing."

## Files

```
vbc_contract_modeling/  config.py  data.py  methods.py  checks.py
contracts/  FAKE_MSSP_BASIC_A.yaml  FAKE_MSSP_ENHANCED.yaml  FAKE_MEDICAID_SUBCAP.yaml  FAKE_CM_FEE_UPSIDE.yaml
tests/      conftest.py  test_t07_settlement.py  test_t07_checks_puf.py
```
