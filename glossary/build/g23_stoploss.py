"""
G23 — Stop-loss pricing sketch: specific (per person) and aggregate attachment
===============================================================================
Copy this whole block into a file (e.g. stoploss.py) and run:  python stoploss.py
Requires: numpy, scipy   (pip install numpy scipy)   A SKETCH: real pricing needs actuarial certification.
"""
import numpy as np
from scipy import stats


def lognormal_excess(mean: float, sigma: float, attachment: float) -> float:
    """E[(X - d)+] for X lognormal with the given mean and log-SD: expected claims above the attachment d.

    Uses the limited expected value: E[X ^ d] = E[X] Phi((ln d - mu - s^2)/s) + d (1 - Phi((ln d - mu)/s)).
    """
    mu = np.log(mean) - sigma ** 2 / 2
    lev = mean * stats.norm.cdf((np.log(attachment) - mu - sigma ** 2) / sigma) + attachment * stats.norm.sf((np.log(attachment) - mu) / sigma)
    return float(mean - lev)


def specific_stoploss_premium(mean_cost: float, sigma: float, attachment: float, reimbursement: float = 1.0,
                              loading: float = 0.25) -> dict:
    """Per-member annual premium for SPECIFIC stop-loss: reimburse ``reimbursement`` x claims above ``attachment``.

    Healthcare context
    ------------------
    Self-funded employers, small risk-bearing provider groups and home-care agencies under sub-capitation
    (t07) buy stop-loss so one catastrophic case can't break them. Pure premium = expected reimbursed
    excess; gross premium adds expenses and margin (premium = pure / (1 - loading)).

    Returns dict: pure_pmpy, gross_pmpy, gross_pmpm, share_of_total_cost.

    Common mistakes
    ---------------
    - Trending the whole cost by 8% and assuming excess claims grow 8% too - they grow FASTER
      (leveraged trend: the attachment doesn't move). See the self-test.
    - Using an empirical tail from too few large claims (fit a distribution or pool years).
    - Ignoring the contract basis (incurred vs. paid, run-in / run-out months: 12/12, 12/15, 15/12).
    """
    pure = reimbursement * lognormal_excess(mean_cost, sigma, attachment)
    gross = pure / (1 - loading)
    return {"pure_pmpy": pure, "gross_pmpy": gross, "gross_pmpm": gross / 12, "share_of_total_cost": pure / mean_cost}


def aggregate_stoploss_cost(member_mean: float, sigma: float, n_members: int, attachment_pct: float = 1.25,
                            n_sims: int = 5_000, seed: int = 0) -> dict:
    """AGGREGATE stop-loss: expected payment when the group's total claims exceed attachment_pct x expected total."""
    rng = np.random.default_rng(seed)
    mu = np.log(member_mean) - sigma ** 2 / 2
    totals = np.array([rng.lognormal(mu, sigma, n_members).sum() for _ in range(n_sims)])
    attach = attachment_pct * member_mean * n_members
    excess = np.clip(totals - attach, 0, None)
    return {"p_hit": float((excess > 0).mean()), "expected_payment": float(excess.mean()),
            "expected_payment_pmpm": float(excess.mean() / n_members / 12)}


# ---------------------------------------------------------------------------
# Self-test: run `python stoploss.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mean, sigma, d = 6_000.0, 1.6, 100_000.0
    s = specific_stoploss_premium(mean, sigma, d)
    rng = np.random.default_rng(1)
    x = rng.lognormal(np.log(mean) - sigma ** 2 / 2, sigma, 2_000_000)
    sim_excess = np.clip(x - d, 0, None).mean()
    trended = lognormal_excess(mean * 1.08, sigma, d) / lognormal_excess(mean, sigma, d) - 1
    small = aggregate_stoploss_cost(mean, sigma, 200, seed=2)
    large = aggregate_stoploss_cost(mean, sigma, 5_000, n_sims=1_000, seed=3)
    print({k: round(v, 4) for k, v in s.items()}, f"| leveraged trend {trended:.1%} | agg p_hit 200 members {small['p_hit']:.1%}, 5,000 {large['p_hit']:.1%}")
    checks = {
        "closed-form excess matches 2M-draw simulation (within 3%)": abs(lognormal_excess(mean, sigma, d) / sim_excess - 1) < 0.03,
        "8% cost trend -> excess grows by more than 8% (leveraged trend)": trended > 0.08,
        "gross = pure / (1 - 25% loading)": abs(s["gross_pmpy"] - s["pure_pmpy"] / 0.75) < 1e-9,
        "aggregate attachment is hit far more often by a small group": small["p_hit"] > large["p_hit"],
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
