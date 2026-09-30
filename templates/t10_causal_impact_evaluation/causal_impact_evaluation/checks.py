"""Causal-design checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_balance(smd: pd.DataFrame, column: str, max_abs: float = 0.1) -> list[Finding]:
    """CAU-001: covariates still differ after matching/weighting."""
    bad = smd[smd[column].abs() >= max_abs]
    return [Finding("CAU-001", "error", f"{len(bad)} covariate(s) with |SMD| >= {max_abs} ({column}): {bad['covariate'].tolist()}",
                    "Refine the PS model (interactions, splines), tighten the caliper, or adjust in the outcome model.",
                    len(bad))] if len(bad) else []


def check_overlap(ps_logit: pd.Series, treated: pd.Series, lo: float = 0.05, hi: float = 0.95) -> list[Finding]:
    """CAU-002: propensity scores near 0 or 1 = people with no comparable counterpart (positivity)."""
    e = 1 / (1 + np.exp(-ps_logit))
    share = float(((e < lo) | (e > hi)).mean())
    t_share = float(((e > hi) & (treated == 1)).sum() / max(int(treated.sum()), 1))
    return [Finding("CAU-002", "warn", f"{share:.1%} of members have PS outside [{lo}, {hi}] ({t_share:.1%} of treated above {hi})",
                    "Restrict to the region of common support and say so; the estimand changes.", int(share * len(e)))] \
        if share > 0.05 else []


def check_pre_trends(p_value: float, alpha: float = 0.05) -> list[Finding]:
    """CAU-003: leads jointly non-zero -> parallel trends doubtful -> DiD biased."""
    return [Finding("CAU-003", "error", f"joint pre-trend test p = {p_value:.3f}",
                    "Find a better comparison group (matching on pre-trends), or use synthetic control / ITS with care.")] \
        if p_value < alpha else []


def check_extreme_weights(weights: pd.Series, max_share: float = 0.01) -> list[Finding]:
    """CAU-004: one observation carrying > 1% of total weight makes IPW unstable."""
    share = float(weights.max() / weights.sum())
    return [Finding("CAU-004", "warn", f"largest weight is {share:.1%} of the total",
                    "Trim/truncate weights, use overlap weights, or prefer AIPW / matching.")] if share > max_share else []


def check_few_clusters(n_clusters: int, minimum: int = 20) -> list[Finding]:
    """CAU-005: cluster-robust SEs are too small with few clusters."""
    return [Finding("CAU-005", "warn", f"only {n_clusters} clusters",
                    "Use wild-cluster bootstrap or small-sample corrections (CR2); report the number of clusters.")] \
        if n_clusters < minimum else []


def check_its_autocorrelation(durbin_watson: float, se_method: str) -> list[Finding]:
    """CAU-006: autocorrelated residuals with OLS or HAC standard errors on a short series."""
    return [Finding("CAU-006", "warn", f"Durbin-Watson {durbin_watson:.2f} with {se_method} SEs",
                    "Model the autocorrelation (se_method='glsar'); HAC under-covers on short series.")] \
        if durbin_watson < 1.5 and se_method != "glsar" else []


def check_its_points(n_pre: int, n_post: int, minimum: int = 8) -> list[Finding]:
    """CAU-007: too few points on one side of the break to separate level from slope."""
    return [Finding("CAU-007", "error", f"{n_pre} pre / {n_post} post points", "Collect more periods or use a comparison series.")] \
        if min(n_pre, n_post) < minimum else []
