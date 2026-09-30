"""
Study design: sample size, power, minimum detectable effect, clustering, randomization, SAP.

    power_two_proportions / power_two_means / mde_*  (statsmodels, analytic)  (STANDARD)
        vs  simulate_power (any design + any analysis, by simulation)          (ALTERNATIVE)
    + design_effect / cluster_sample_size, stratified_block_randomize, render_sap

Rule of thumb: decide the analysis first, then compute power FOR THAT ANALYSIS. If no formula
matches the analysis (clusters + skewed cost + DiD), simulate.
"""
from __future__ import annotations

import math
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

# ---------------------------------------------------------------------------
# 1. Analytic power (STANDARD)
# ---------------------------------------------------------------------------

def power_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float | None = 0.80,
                          n_per_arm: float | None = None, ratio: float = 1.0) -> dict:
    """Sample size per arm (or power) to compare two proportions, two-sided z test on Cohen's h.

    Healthcare context
    ------------------
    Most program endpoints are rates: readmission, ED use, gap closure, retention. Give EITHER
    ``power`` (returns n) OR ``n_per_arm`` (returns power).

    >>> r = power_two_proportions(0.20, 0.15)
    >>> r["n_per_arm"]
    903

    Common mistakes
    ---------------
    - Using the effect you HOPE for rather than the smallest effect worth acting on.
    - Forgetting attrition: divide n by (1 - expected loss to follow-up).
    - Powering on a relative reduction ("20% fewer readmissions") without writing the absolute rates.
    """
    h = abs(proportion_effectsize(p1, p2))
    solver = NormalIndPower()
    if n_per_arm is None:
        n = solver.solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio, alternative="two-sided")
        return {"effect_size_h": float(h), "n_per_arm": int(math.ceil(n)), "n_total": int(math.ceil(n) + math.ceil(n * ratio)),
                "power": float(power)}
    pw = solver.power(effect_size=h, nobs1=n_per_arm, alpha=alpha, ratio=ratio, alternative="two-sided")
    return {"effect_size_h": float(h), "n_per_arm": int(n_per_arm), "power": float(pw)}


def power_two_means(delta: float, sd: float, alpha: float = 0.05, power: float | None = 0.80,
                    n_per_arm: float | None = None) -> dict:
    """Two-sample t test on means (Cohen's d = delta / sd). Returns n per arm or power.

    >>> power_two_means(5.0, 10.0)["n_per_arm"]
    64
    """
    d = delta / sd
    solver = TTestIndPower()
    if n_per_arm is None:
        n = solver.solve_power(effect_size=d, alpha=alpha, power=power, alternative="two-sided")
        return {"effect_size_d": float(d), "n_per_arm": int(math.ceil(n)), "n_total": 2 * int(math.ceil(n)), "power": float(power)}
    return {"effect_size_d": float(d), "n_per_arm": int(n_per_arm),
            "power": float(solver.power(effect_size=d, nobs1=n_per_arm, alpha=alpha, alternative="two-sided"))}


def mde_two_proportions(p1: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80, direction: int = -1) -> float:
    """Minimum detectable p2 given the baseline rate and the n you actually have.

    Invert Cohen's h: solve for h at the given n, then p2 = sin^2(asin(sqrt p1) + direction x h / 2).
    ``direction`` -1 = looking for a reduction.
    """
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, alternative="two-sided")
    return float(np.sin(np.arcsin(np.sqrt(p1)) + direction * h / 2) ** 2)


def mde_two_means(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable difference in means (in outcome units) for the n you have."""
    return float(TTestIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, alternative="two-sided") * sd)


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------

def design_effect(cluster_size: float, icc: float) -> float:
    """DEFF = 1 + (m - 1) x ICC: variance inflation when people in a cluster resemble each other.

    >>> design_effect(40, 0.02)
    1.78
    """
    return float(1 + (cluster_size - 1) * icc)


def cluster_sample_size(n_per_arm_individual: int, cluster_size: int, icc: float) -> dict:
    """Inflate an individually randomized n for cluster randomization (practices, care teams, counties).

    Returns design_effect, n_per_arm, clusters_per_arm (rounded up), effective_n_per_arm.

    Common mistakes
    ---------------
    - Randomizing practices but powering as if patients were randomized (DEFF ignored).
    - Fewer than ~6-8 clusters per arm: the normal approximation fails; use simulation and small-sample
      corrections (and consider a stepped-wedge or stratified design).
    """
    deff = design_effect(cluster_size, icc)
    n = int(math.ceil(n_per_arm_individual * deff))
    k = int(math.ceil(n / cluster_size))
    return {"design_effect": deff, "n_per_arm": k * cluster_size, "clusters_per_arm": k,
            "effective_n_per_arm": float(k * cluster_size / deff)}


# ---------------------------------------------------------------------------
# 3. Simulation-based power (ALTERNATIVE)
# ---------------------------------------------------------------------------

def simulate_power(simulate_once, n_sims: int = 1_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """ALTERNATIVE: power = share of simulated trials whose analysis rejects H0.

    Trade-off
    ---------
    + Works for ANY design and ANY analysis: clusters with few units, skewed costs, DiD, unequal
      attrition, the exact regression you will run. If you can simulate it, you can power it.
    + Forces you to write down the data-generating assumptions (a good SAP habit).
    - Monte Carlo error (+/- 1.4 points at 1,000 sims near 80%); slower; only as good as the
      assumptions you simulate.

    ``simulate_once(rng) -> p_value``. Returns power, mc_se, n_sims.
    """
    rng = np.random.default_rng(seed)
    p = np.array([simulate_once(rng) for _ in range(n_sims)])
    pw = float((p < alpha).mean())
    return {"power": pw, "mc_se": float(np.sqrt(pw * (1 - pw) / n_sims)), "n_sims": n_sims}


def sim_two_proportions(p1: float, p2: float, n_per_arm: int):
    """Simulator: two independent arms, chi-square-equivalent two-proportion z test (pooled)."""
    def once(rng):
        a, b = rng.binomial(n_per_arm, p1), rng.binomial(n_per_arm, p2)
        pool = (a + b) / (2 * n_per_arm)
        se = np.sqrt(pool * (1 - pool) * 2 / n_per_arm)
        z = (a - b) / n_per_arm / se if se > 0 else 0.0
        return float(2 * stats.norm.sf(abs(z)))
    return once


def sim_cluster_proportions(p1: float, p2: float, clusters_per_arm: int, cluster_size: int, icc: float):
    """Simulator: cluster-randomized binary outcome (beta-binomial clusters with the given ICC),
    analyzed CORRECTLY by a t test on cluster-level proportions."""
    def once(rng):
        means = []
        for p in (p1, p2):
            ab = (1 - icc) / icc                                  # beta(a, b) with a + b = (1 - ICC)/ICC gives that ICC
            cp = rng.beta(p * ab, (1 - p) * ab, clusters_per_arm)
            means.append(rng.binomial(cluster_size, cp) / cluster_size)
        return float(stats.ttest_ind(means[0], means[1]).pvalue)
    return once


def sim_cost_did(pmpm: float, cv: float, effect_pct: float, n_per_arm: int, pre_post_corr: float = 0.5):
    """Simulator: skewed cost (lognormal, CV ``cv``) measured pre and post; treated post x (1 - effect).
    Analysis = Welch t test on member-level change (post - pre) - i.e. a DiD on means."""
    s2 = np.log(1 + cv ** 2)
    mu = np.log(pmpm) - s2 / 2
    cov = np.array([[s2, pre_post_corr * s2], [pre_post_corr * s2, s2]])

    def once(rng):
        ch = []
        for eff in (0.0, effect_pct):
            z = rng.multivariate_normal([mu, mu], cov, n_per_arm)
            pre, post = np.exp(z[:, 0]), np.exp(z[:, 1]) * (1 - eff)
            ch.append(post - pre)
        return float(stats.ttest_ind(ch[1], ch[0], equal_var=False).pvalue)
    return once


def power_curve(n_grid, power_fn) -> pd.DataFrame:
    """Evaluate ``power_fn(n) -> power`` over a grid of n (for a chart or a table in the SAP)."""
    return pd.DataFrame({"n_per_arm": list(n_grid), "power": [float(power_fn(n)) for n in n_grid]})


# ---------------------------------------------------------------------------
# 4. Randomization
# ---------------------------------------------------------------------------

def stratified_block_randomize(units: pd.DataFrame, strata: list[str], block_sizes=(2, 4),
                               arms=("control", "treatment"), seed: int = 0) -> pd.DataFrame:
    """Permuted blocks of random size within each stratum (the standard for pragmatic trials).

    Healthcare context
    ------------------
    Simple coin flips can leave one arm sicker by chance, especially in small strata. Stratifying on
    the strongest predictors (site, risk tier) and using permuted blocks keeps arms balanced within
    every stratum at every point of enrollment; random block sizes stop staff guessing the next arm.

    Returns ``units`` plus arm, stratum_id, block_id, seq_in_stratum. Deterministic for a seed.

    Common mistakes
    ---------------
    - Too many strata for the sample (many strata with 1-2 people -> blocks never complete).
    - Fixed block size of 2 with unblinded staff: every second assignment is predictable.
    - Re-running randomization after seeing the result ("it wasn't balanced enough").
    """
    rng = np.random.default_rng(seed)
    k = len(arms)
    out = units.copy()
    out["stratum_id"] = out.groupby(strata, sort=True).ngroup()
    out["arm"], out["block_id"], out["seq_in_stratum"] = None, -1, -1
    block_counter = 0
    for sid, idx in out.groupby("stratum_id").groups.items():
        idx = list(idx)
        assign, blocks = [], []
        while len(assign) < len(idx):
            size = int(rng.choice([b for b in block_sizes if b % k == 0]))
            blk = list(np.repeat(arms, size // k))
            rng.shuffle(blk)
            assign += blk
            blocks += [block_counter] * size
            block_counter += 1
        out.loc[idx, "arm"] = assign[: len(idx)]
        out.loc[idx, "block_id"] = blocks[: len(idx)]
        out.loc[idx, "seq_in_stratum"] = range(len(idx))
    return out


# ---------------------------------------------------------------------------
# 5. Statistical analysis plan (Markdown)
# ---------------------------------------------------------------------------

def render_sap(template_path: str | Path, values: dict) -> str:
    """Fill the Markdown SAP template (``$name`` placeholders, stdlib string.Template).

    Raises KeyError naming any placeholder left unfilled - an SAP with a blank sample-size
    justification should not leave the building.
    """
    text = Path(template_path).read_text(encoding="utf-8")
    return Template(text).substitute(values)
