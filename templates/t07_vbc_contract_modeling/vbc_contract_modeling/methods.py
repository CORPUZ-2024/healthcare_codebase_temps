"""
Value-based contract settlement from YAML contracts.

    reconcile_* (deterministic, closes to the dollar)   (STANDARD)
        vs  monte_carlo_shared_savings (distribution of outcomes)   (ALTERNATIVE)
    + scenario_grid, msr_for_confidence, reproduce_mssp_puf

Contract types
--------------
shared_savings  ACO-style: benchmark vs. actual, MSR/MLR corridor, sharing rate x quality,
                quality gate, savings and loss caps, per-person truncation
subcap          Medicaid sub-capitation: fixed PMPM to a provider, risk corridors, stop-loss
fee_upside      vendor PMPM fee + share of savings above a threshold, quality gate, fees at risk

Every reconcile_* returns a dict with plain floats, a ``waterfall`` DataFrame (line, amount_usd)
and ``identities`` (name, lhs, rhs) that checks.check_reconciliation_closes verifies (ANL-015).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _identity(name: str, lhs: float, rhs: float) -> tuple[str, float, float]:
    return (name, float(lhs), float(rhs))


# ---------------------------------------------------------------------------
# 1. Shared savings / losses (ACO)
# ---------------------------------------------------------------------------

def shared_savings_core(c: dict, benchmark_pmpy, actual_pmpy, n, quality) -> dict:
    """Vectorized settlement arithmetic (numpy arrays or scalars broadcast together).

    Steps
    -----
    1. gross = (benchmark - actual) x n; savings rate = (benchmark - actual) / benchmark.
    2. Final sharing rate = max_sharing_rate x quality (if quality-scaled); 0 below the quality gate.
    3. Savings: if rate >= MSR, pay sharing rate x (all savings if first-dollar, else savings above
       MSR), capped at savings_cap_pct x benchmark total.
    4. Losses (two-sided only): if rate <= -MLR, owe loss rate x losses, loss rate =
       1 - max_sharing_rate x quality bounded to [loss_rate_min, loss_rate_max], capped at
       loss_cap_pct x benchmark total.
    5. ACO amount = payment - repayment; payer amount = gross - ACO amount.

    >>> c = dict(msr_pct=.02, mlr_pct=.02, share_from_first_dollar=True, max_sharing_rate=.75,
    ...          quality_scales_sharing=True, quality_gate=.4, savings_cap_pct=.2, two_sided=True,
    ...          loss_rate_min=.4, loss_rate_max=.75, loss_cap_pct=.15)
    >>> r = shared_savings_core(c, 10_000.0, 9_500.0, 1_000, 0.9)
    >>> float(r["gross"]), float(r["aco_amount"]), float(r["payer_amount"])
    (500000.0, 337500.0, 162500.0)
    """
    B, A, n, q = (np.asarray(x, dtype=float) for x in (benchmark_pmpy, actual_pmpy, n, quality))
    gross = (B - A) * n                                                                   # step 1
    rate = (B - A) / B
    share = c["max_sharing_rate"] * (q if c.get("quality_scales_sharing", True) else 1.0)  # step 2
    share = np.where(q < c.get("quality_gate", 0.0), 0.0, share)
    first = c.get("share_from_first_dollar", True)
    sav_basis = np.where(first, gross, (rate - c["msr_pct"]) * B * n)                     # step 3
    payment = np.where(rate >= c["msr_pct"], np.minimum(sav_basis * share, c["savings_cap_pct"] * B * n), 0.0)
    repayment = np.zeros_like(payment)
    loss_rate = np.zeros_like(np.asarray(share, dtype=float))
    if c.get("two_sided", False):                                                         # step 4
        loss_rate = np.clip(1 - c["max_sharing_rate"] * q, c["loss_rate_min"], c["loss_rate_max"])
        loss_basis = np.where(first, -gross, (-rate - c["mlr_pct"]) * B * n)
        repayment = np.where(rate <= -c["mlr_pct"], np.minimum(loss_basis * loss_rate, c["loss_cap_pct"] * B * n), 0.0)
    aco = payment - repayment                                                             # step 5
    outcome = np.select([payment > 0, repayment > 0, (rate >= c["msr_pct"]) & (share == 0)],
                        ["shared savings", "shared losses", "savings, quality gate failed"], default="within corridor")
    return {"gross": gross, "savings_rate": rate, "sharing_rate": share, "loss_rate": loss_rate,
            "payment": payment, "repayment": repayment, "aco_amount": aco, "payer_amount": gross - aco,
            "outcome": outcome}


def reconcile_shared_savings(c: dict, costs: pd.Series, benchmark_pmpy: float, quality: float) -> dict:
    """STANDARD: deterministic ACO settlement from beneficiary-level annual costs.

    Healthcare context
    ------------------
    In MSSP-style contracts the ACO shares savings only if spending beats the benchmark by more
    than the minimum savings rate (MSR) - a corridor that protects the payer from paying for random
    variation. Per-person costs are truncated first so one catastrophic case doesn't decide the year.
    The reconciliation must CLOSE: every dollar of gross savings belongs to the ACO or the payer.

    Parameters
    ----------
    costs : annual cost per assigned beneficiary (person-years; one row per beneficiary here)
    benchmark_pmpy : per-capita benchmark (already risk-adjusted and trended)
    quality : quality score 0-1

    Returns dict: n, benchmark_total, actual_raw_total, truncation_amt_removed, actual_total,
    gross_savings, savings_rate, sharing_rate, aco_amount, payer_amount, outcome, waterfall, identities.

    Common mistakes
    ---------------
    - Comparing untruncated actuals to a truncated benchmark (or the reverse).
    - Sharing only savings above the MSR when the contract says first dollar (or vice versa):
      at 3% savings and a 2% MSR that is a 3x difference in payout.
    - Reporting "savings" that fall inside the corridor as earned.
    """
    raw = float(costs.sum())
    trunc = costs.clip(upper=c.get("truncation_amt", np.inf))
    n = len(costs)
    actual_total = float(trunc.sum())
    B_total = benchmark_pmpy * n
    r = shared_savings_core(c, benchmark_pmpy, actual_total / n, n, quality)
    g = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
    wf = pd.DataFrame([
        ("1 Benchmark total", B_total), ("2 Actual expenditure, raw", raw),
        ("3 Less truncation above per-person cap", raw - actual_total), ("4 Actual expenditure, truncated", actual_total),
        ("5 Gross savings (+) / losses (-) = 1 - 4", g["gross"]),
        ("6 ACO shared savings (+) / shared losses owed (-)", g["aco_amount"]),
        ("7 Payer retained savings / absorbed losses = 5 - 6", g["payer_amount"]),
    ], columns=["line", "amount_usd"])
    ids = [_identity("truncated = raw - truncation", actual_total, raw - (raw - actual_total)),
           _identity("gross = benchmark - actual", g["gross"], B_total - actual_total),
           _identity("ACO + payer = gross", g["aco_amount"] + g["payer_amount"], g["gross"]),
           _identity("ACO = payment - repayment", g["aco_amount"], g["payment"] - g["repayment"])]
    return {"contract_id": c["contract_id"], "n": n, "benchmark_total": B_total, "actual_raw_total": raw,
            "truncation_amt_removed": raw - actual_total, "actual_total": actual_total, "gross_savings": g["gross"],
            "savings_rate": g["savings_rate"], "sharing_rate": g["sharing_rate"], "loss_rate": g["loss_rate"],
            "aco_amount": g["aco_amount"], "payer_amount": g["payer_amount"], "outcome": str(g["outcome"]),
            "waterfall": wf, "identities": ids}


# ---------------------------------------------------------------------------
# 2. Medicaid sub-capitation with corridors and stop-loss
# ---------------------------------------------------------------------------

def corridor_provider_share(margin_pct: float, corridors: list[dict]) -> float:
    """Share of the margin the provider keeps, integrated band by band on |margin %|.

    >>> bands = [dict(lo=0, hi=.03, provider_share=1), dict(lo=.03, hi=.10, provider_share=.5),
    ...          dict(lo=.10, hi=9.99, provider_share=0)]
    >>> round(corridor_provider_share(-0.06, bands), 4)   # keeps all of first 3%, half of next 3%
    -0.045
    """
    m = abs(margin_pct)
    kept = sum(b["provider_share"] * max(0.0, min(m, b["hi"]) - b["lo"]) for b in corridors)
    return float(np.sign(margin_pct) * kept)


def reconcile_subcap(c: dict, members: pd.DataFrame) -> dict:
    """STANDARD: sub-capitation settlement: cap revenue, stop-loss recovery, corridor settlement.

    Healthcare context
    ------------------
    A Medicaid plan pays a home-care agency a fixed PMPM for a service bundle. If services cost
    more, the agency loses money - until the corridor: beyond +/-3% it shares the result 50/50
    with the plan, beyond 10% the plan takes it all. Stop-loss reimburses most of any one member's
    cost above an attachment point, so a single 24-hour-care case can't sink a small agency.

    Parameters
    ----------
    members : member_id, member_months, hcbs_cost_amt (provider's annual cost for the bundle)

    Returns dict: cap_revenue, provider_cost, stoploss_recovery, margin, margin_pct,
    provider_result, plan_settlement_to_provider, plan_total_cost, waterfall, identities.

    Common mistakes
    ---------------
    - Applying the corridor to the margin BEFORE stop-loss (stop-loss settles first here - read the contract).
    - Treating the corridor share as a flat rate on the whole margin instead of band by band.
    """
    revenue = float(c["cap_pmpm"] * members["member_months"].sum())
    cost = float(members["hcbs_cost_amt"].sum())
    excess = (members["hcbs_cost_amt"] - c["stoploss_attachment_amt"]).clip(lower=0)
    recovery = float(c["stoploss_coinsurance"] * excess.sum())
    margin = revenue - (cost - recovery)
    mpct = margin / revenue if revenue else 0.0
    keeps = corridor_provider_share(mpct, c["corridors"]) * revenue
    settle = keeps - margin
    wf = pd.DataFrame([
        ("1 Capitation revenue (cap PMPM x member-months)", revenue), ("2 Provider cost of covered services", cost),
        ("3 Stop-loss recovery from plan", recovery), ("4 Margin before corridor = 1 - 2 + 3", margin),
        ("5 Corridor settlement: plan pays provider (+) / provider returns (-)", settle),
        ("6 Provider final result = 4 + 5", keeps), ("7 Plan total cost = 1 + 3 + 5", revenue + recovery + settle),
    ], columns=["line", "amount_usd"])
    ids = [_identity("margin = revenue - cost + recovery", margin, revenue - cost + recovery),
           _identity("provider result = margin + settlement", keeps, margin + settle),
           _identity("plan cost - provider result = provider cost", revenue + recovery + settle - keeps, cost)]
    return {"contract_id": c["contract_id"], "cap_revenue": revenue, "provider_cost": cost, "stoploss_recovery": recovery,
            "margin": float(margin), "margin_pct": float(mpct), "provider_result": float(keeps),
            "plan_settlement_to_provider": float(settle), "plan_total_cost": float(revenue + recovery + settle),
            "waterfall": wf, "identities": ids}


# ---------------------------------------------------------------------------
# 3. Care-management fee + upside
# ---------------------------------------------------------------------------

def reconcile_fee_upside(c: dict, participants: pd.DataFrame, quality_met: dict[str, bool]) -> dict:
    """STANDARD: vendor fee + upside settlement against a trended target.

    Healthcare context
    ------------------
    Care-management and caregiver-support vendors are often paid a PMPM fee plus a share of
    savings against a target (baseline PMPM trended forward), with part of the fee at risk on
    quality. The payer's real return is gross savings minus EVERYTHING paid to the vendor.

    Parameters
    ----------
    participants : member_id, member_months, baseline_pmpm, actual_pmpm
    quality_met : {measure name: met?}

    Returns dict: fees, target_total, actual_total, gross_savings, savings_rate, quality_gate_met,
    upside, fee_refund, vendor_revenue, payer_net_savings, waterfall, identities.

    Common mistakes
    ---------------
    - Target from the participants' OWN prior year when they were enrolled because of a bad year
      (regression to the mean pays the vendor; see t06 and checks.check_target_basis).
    - Forgetting that "savings" to the payer are net of fees.
    """
    mm = participants["member_months"]
    fees = float(c["fee_pmpm"] * mm.sum())
    target = float((participants["baseline_pmpm"] * (1 + c["target_trend_pct"]) * mm).sum())
    actual = float((participants["actual_pmpm"] * mm).sum())
    gross = target - actual
    rate = gross / target
    met = sum(bool(v) for v in quality_met.values()) >= c["quality_measures_required"]
    basis = gross if c.get("share_from_first_dollar", False) else (rate - c["msr_pct"]) * target
    upside = min(basis * c["upside_share"], c["upside_cap_pct_of_fees"] * fees) if (rate >= c["msr_pct"] and met) else 0.0
    refund = 0.0 if met else c["fees_at_risk_pct"] * fees
    vendor = fees + upside - refund
    wf = pd.DataFrame([
        ("1 Target (baseline x trend x member-months)", target), ("2 Actual cost", actual),
        ("3 Gross savings = 1 - 2", gross), ("4 Fees paid", fees), ("5 Upside earned", upside),
        ("6 Fees refunded (quality gate missed)", -refund if refund else 0.0), ("7 Vendor revenue = 4 + 5 + 6", vendor),
        ("8 Payer net savings = 3 - 7", gross - vendor),
    ], columns=["line", "amount_usd"])
    ids = [_identity("vendor revenue = fees + upside - refund", vendor, fees + upside - refund),
           _identity("vendor + payer net = gross", vendor + (gross - vendor), gross),
           _identity("gross = target - actual", gross, target - actual)]
    return {"contract_id": c["contract_id"], "fees": fees, "target_total": target, "actual_total": actual,
            "gross_savings": float(gross), "savings_rate": float(rate), "quality_gate_met": bool(met),
            "upside": float(upside), "fee_refund": float(refund), "vendor_revenue": float(vendor),
            "payer_net_savings": float(gross - vendor), "waterfall": wf, "identities": ids}


# ---------------------------------------------------------------------------
# 4. Scenario grid and Monte Carlo (shared savings)
# ---------------------------------------------------------------------------

def scenario_grid(c: dict, benchmark_pmpy: float, n: int, savings_pcts, qualities) -> pd.DataFrame:
    """Deterministic what-if table: every (true savings %, quality) pair, no random variation.

    Use it in negotiations: where are the corridor edges, how much does quality move the payout,
    what does a bad year cost under the loss cap.
    """
    s, q = np.meshgrid(np.asarray(savings_pcts, float), np.asarray(qualities, float), indexing="ij")
    r = shared_savings_core(c, benchmark_pmpy, benchmark_pmpy * (1 - s), n, q)
    return pd.DataFrame({"savings_pct": s.ravel(), "quality": q.ravel(), "outcome": r["outcome"].ravel(),
                         "gross_savings": r["gross"].ravel(), "aco_amount": r["aco_amount"].ravel(),
                         "payer_amount": r["payer_amount"].ravel(), "aco_pmpy": (r["aco_amount"] / n).ravel()})


def monte_carlo_shared_savings(c: dict, n: int, benchmark_pmpy: float, true_savings_pct: float, quality: float,
                               n_sims: int = 2_000, trend_sd: float = 0.015, seed: int = 0,
                               chunk: int = 100) -> tuple[pd.DataFrame, dict]:
    """ALTERNATIVE: distribution of settlement outcomes under random variation and trend risk.

    Trade-off
    ---------
    + The deterministic answer uses ONE expected cost; a real year draws n people's costs from a
      skewed distribution and the benchmark trend can miss. Monte Carlo gives P(clearing the MSR),
      P(owing losses) and the expected payout - the numbers for deciding whether to take risk.
    + Shows why the MSR exists: with no true savings an ACO still "saves" 2% some of the time.
    - Only as good as the cost distribution (here: 5% zeros + lognormal sigma 1.3) and trend_sd
      assumed; ignores quality uncertainty and benchmark rebasing across years.

    Steps
    -----
    1. For each simulation draw n annual costs (sample_annual_costs; TRUNCATED mean = benchmark x
       (1 - true savings)) and one trend shock exp(N(0, trend_sd)) that multiplies all of them.
    2. Truncate per person, average -> actual PMPY; settle with shared_savings_core.

    Returns (sims: sim, actual_pmpy, savings_rate, aco_amount, outcome; summary dict).
    """
    from .data import raw_mean_for_truncated, sample_annual_costs  # local import: methods usable without data

    rng = np.random.default_rng(seed)
    cap = c.get("truncation_amt", np.inf)
    raw = raw_mean_for_truncated(benchmark_pmpy * (1 - true_savings_pct), cap)   # benchmark is on the truncated basis
    pmpy = []
    for start in range(0, n_sims, chunk):                                                   # step 1
        k = min(chunk, n_sims - start)
        costs = sample_annual_costs(rng, k * n, raw).reshape(k, n)
        shock = np.exp(rng.normal(0.0, trend_sd, (k, 1)))
        pmpy.append(np.minimum(costs * shock, cap).mean(axis=1))                            # step 2
    A = np.concatenate(pmpy)
    r = shared_savings_core(c, benchmark_pmpy, A, n, quality)
    sims = pd.DataFrame({"sim": np.arange(n_sims), "actual_pmpy": A, "savings_rate": r["savings_rate"],
                         "aco_amount": r["aco_amount"], "outcome": r["outcome"]})
    a = sims["aco_amount"]
    summary = {"p_shared_savings": float((sims["outcome"] == "shared savings").mean()),
               "p_shared_losses": float((sims["outcome"] == "shared losses").mean()),
               "p_within_corridor": float((sims["outcome"] == "within corridor").mean()),
               "expected_aco_amount": float(a.mean()), "aco_p05": float(a.quantile(0.05)),
               "aco_p50": float(a.quantile(0.5)), "aco_p95": float(a.quantile(0.95)),
               "sd_savings_rate": float(sims["savings_rate"].std())}
    return sims, summary


def msr_for_confidence(cv: float, n: int, confidence: float = 0.90) -> float:
    """MSR that a population of n with per-person cost CV would clear by chance only (1 - confidence)
    of the time: z x CV / sqrt(n). The logic behind MSSP's one-sided MSR table (which uses a
    two-sided 90% interval).

    >>> round(msr_for_confidence(2.0, 10_000, 0.90), 4)
    0.0256
    """
    return float(stats.norm.ppf(confidence) * cv / np.sqrt(n))


# ---------------------------------------------------------------------------
# 5. Reproduce published MSSP results
# ---------------------------------------------------------------------------

def reproduce_mssp_puf(puf: pd.DataFrame, tol_pct: float = 0.005) -> pd.DataFrame:
    """Recompute generated and earned savings from the PUF's own benchmark and expenditure columns.

    Steps
    -----
    1. gross = benchmark total - expenditure total; compare with the published BnchmkMinExp / GenSaveLoss.
    2. savings rate = gross / benchmark total; earned = gross x FinalShareRate if rate >= MSR
       (first dollar), else 0 (one-sided view).
    3. Flag rows where |recomputed - published| > tol_pct of the benchmark.

    Mismatches are expected for: savings/loss caps, two-sided losses, prior-savings adjustments,
    and payment adjustments applied after the PUF figures - read the PUF notes before concluding
    your engine is wrong.
    """
    p = puf.copy()
    p["gross_recalc_amt"] = p["benchmark_total_amt"] - p["expenditure_total_amt"]
    pub = p["bnchmk_min_exp_amt"] if "bnchmk_min_exp_amt" in p else p.get("gen_save_loss_amt")
    p["savings_rate_recalc"] = p["gross_recalc_amt"] / p["benchmark_total_amt"]
    p["earned_recalc_amt"] = np.where((p["savings_rate_recalc"] >= p["msr_pct"]) & (p["gross_recalc_amt"] > 0),
                                      p["gross_recalc_amt"] * p["final_share_rate"], 0.0)
    tol = tol_pct * p["benchmark_total_amt"]
    p["gross_match_flag"] = ((p["gross_recalc_amt"] - pub).abs() <= tol).astype(int)
    if "earn_save_loss_amt" in p:
        p["earned_match_flag"] = ((p["earned_recalc_amt"] - p["earn_save_loss_amt"].clip(lower=0)).abs() <= tol).astype(int)
    return p
