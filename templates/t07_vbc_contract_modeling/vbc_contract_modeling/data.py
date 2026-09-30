"""
Contract YAML loader, synthetic performance-year data for three contract types, and a loader
for the public MSSP ACO performance-year financial results file.

Everything generated is FAKE:
  * ACO beneficiaries: annual Medicare-FFS-like costs (~5% $0, lognormal tail) around a benchmark
  * Medicaid HCBS sub-cap members: partial-year enrollment, a few catastrophic members (stop-loss)
  * care-management participants: baseline and performance-period PMPM

Public test file
----------------
CMS Medicare Shared Savings Program — ACO Performance Year Financial and Quality Results PUF
(data.cms.gov). See data/README.md and ``load_mssp_puf``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_contract(path: str | Path) -> dict:
    """One contract YAML -> dict."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_contracts(contracts_dir: str | Path) -> dict[str, dict]:
    """All contracts in a folder keyed by contract_id (sorted, reproducible)."""
    cs = [load_contract(p) for p in sorted(Path(contracts_dir).glob("*.yaml"))]
    return {c["contract_id"]: c for c in cs}


def sample_annual_costs(rng: np.random.Generator, n: int, mean: float, zero_share: float = 0.05,
                        sigma: float = 1.3) -> np.ndarray:
    """Annual per-person cost: ``zero_share`` at $0, the rest lognormal with log-SD ``sigma``,
    scaled so the overall mean is ``mean`` (before any truncation)."""
    pos_mean = mean / (1 - zero_share)
    x = np.exp(np.log(pos_mean) - sigma ** 2 / 2 + sigma * rng.standard_normal(n))
    return np.where(rng.random(n) < zero_share, 0.0, x)


def truncated_mean(raw_mean: float, trunc: float, zero_share: float = 0.05, sigma: float = 1.3) -> float:
    """E[min(X, trunc)] for the sample_annual_costs distribution (closed-form lognormal limited
    expected value): E[X] Phi((ln T - mu - s^2)/s) + T (1 - Phi((ln T - mu)/s)), times (1 - zero share)."""
    from scipy import stats
    if not np.isfinite(trunc):
        return raw_mean
    m_pos = raw_mean / (1 - zero_share)
    mu = np.log(m_pos) - sigma ** 2 / 2
    lt = np.log(trunc)
    lev = m_pos * stats.norm.cdf((lt - mu - sigma ** 2) / sigma) + trunc * stats.norm.sf((lt - mu) / sigma)
    return float((1 - zero_share) * lev)


def raw_mean_for_truncated(target_truncated_mean: float, trunc: float, **kw) -> float:
    """Raw (untruncated) mean whose truncated mean equals the target - so a truncated benchmark and
    truncated actuals are compared like with like."""
    from scipy.optimize import brentq
    if not np.isfinite(trunc):
        return target_truncated_mean
    return float(brentq(lambda m: truncated_mean(m, trunc, **kw) - target_truncated_mean,
                        target_truncated_mean, target_truncated_mean * 10))


def generate_aco_year(n: int = 12_000, benchmark_pmpy: float = 12_000.0, true_savings_pct: float = 0.035,
                      seed: int = 7, truncation_amt: float = 150_000.0) -> pd.DataFrame:
    """Assigned beneficiaries for one performance year: bene_id, annual_cost_amt.

    The benchmark is on the TRUNCATED basis (as in MSSP): expected truncated cost per person =
    benchmark x (1 - true_savings_pct). The observed savings rate differs from the true one by
    random variation - exactly why contracts have an MSR.
    """
    rng = np.random.default_rng(seed)
    raw = raw_mean_for_truncated(benchmark_pmpy * (1 - true_savings_pct), truncation_amt)
    cost = sample_annual_costs(rng, n, raw)
    return pd.DataFrame({"bene_id": [f"B{i:06d}" for i in range(1, n + 1)], "annual_cost_amt": cost.round(2)})


def generate_subcap_members(n: int = 1_500, cap_pmpm: float = 410.0, cost_ratio: float = 1.02,
                            seed: int = 7) -> pd.DataFrame:
    """Medicaid HCBS sub-cap members: member_id, member_months (3-12), hcbs_cost_amt (annual).

    Expected cost = cost_ratio x cap x member_months (1.02 = the cap is 2% too low); ~0.4% of
    members are catastrophic (24-hour care), which is what stop-loss is for.
    """
    rng = np.random.default_rng(seed + 1)
    mm = np.where(rng.random(n) < 0.8, 12, rng.integers(3, 12, n)).astype(float)
    cost = sample_annual_costs(rng, n, 1.0, zero_share=0.08, sigma=0.9) * cost_ratio * cap_pmpm * mm
    cat = rng.random(n) < 0.004
    cost = np.where(cat, rng.uniform(45_000, 90_000, n) * mm / 12, cost)
    return pd.DataFrame({"member_id": [f"S{i:05d}" for i in range(1, n + 1)], "member_months": mm,
                         "hcbs_cost_amt": cost.round(2)})


def generate_cm_program(n: int = 400, baseline_pmpm: float = 1_450.0, true_savings_pct: float = 0.09,
                        seed: int = 7) -> pd.DataFrame:
    """Care-management participants: member_id, member_months, baseline_pmpm (prior year),
    actual_pmpm (performance year; trend 5% minus the true savings, plus noise)."""
    rng = np.random.default_rng(seed + 2)
    mm = np.where(rng.random(n) < 0.85, 12, rng.integers(4, 12, n)).astype(float)
    base = sample_annual_costs(rng, n, baseline_pmpm, zero_share=0.0, sigma=0.8)
    actual = base * 1.05 * (1 - true_savings_pct) * np.exp(rng.normal(0, 0.35, n) - 0.35 ** 2 / 2)
    return pd.DataFrame({"member_id": [f"C{i:05d}" for i in range(1, n + 1)], "member_months": mm,
                         "baseline_pmpm": base.round(2), "actual_pmpm": actual.round(2)})


# ---------------------------------------------------------------------------
# Public file: MSSP ACO performance-year results
# ---------------------------------------------------------------------------
MSSP_PUF_COLS = {  # PUF column -> template column (verify against the data dictionary of your year)
    "ACO_ID": "aco_id", "ACO_Name": "aco_name", "Current_Track": "track", "Track": "track", "N_AB": "n_ab",
    "ABtotBnchmk": "benchmark_total_amt", "ABtotExp": "expenditure_total_amt", "BnchmkMinExp": "bnchmk_min_exp_amt",
    "GenSaveLoss": "gen_save_loss_amt", "EarnSaveLoss": "earn_save_loss_amt", "MinSavPerc": "msr_pct",
    "Sav_rate": "sav_rate", "QualScore": "quality_score", "FinalShareRate": "final_share_rate",
}


def load_mssp_puf(path: str | Path) -> pd.DataFrame:
    """Load the MSSP ACO PUF, keep the reconciliation columns, parse numbers.

    Money columns can contain '$' and ','; percentage columns can be 0-100 or 0-1 depending on the
    year -> ``msr_pct`` and ``sav_rate`` are normalized to 0-1 when their max exceeds 1.
    """
    df = pd.read_csv(path, dtype=str)
    keep = {c: MSSP_PUF_COLS[c] for c in df.columns if c in MSSP_PUF_COLS}
    out = df[list(keep)].rename(columns=keep)
    out = out.loc[:, ~out.columns.duplicated()]
    for c in out.columns:
        if c not in ("aco_id", "aco_name", "track"):
            out[c] = pd.to_numeric(out[c].str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
    for c in ("msr_pct", "sav_rate", "final_share_rate", "quality_score"):
        if c in out and out[c].max() > 1:
            out[c] = out[c] / 100.0
    return out
