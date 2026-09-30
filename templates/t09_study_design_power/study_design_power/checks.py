"""Design checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_underpowered(power: float, target: float = 0.80) -> list[Finding]:
    """PWR-001: the available sample gives less than the target power."""
    return [Finding("PWR-001", "warn", f"power {power:.0%} < {target:.0%} target",
                    "Report the MDE for the n you have, extend enrollment, or use a more sensitive endpoint.")] \
        if power < target else []


def check_cluster_design(randomization_unit: str, design_effect: float) -> list[Finding]:
    """PWR-002: cluster-randomized designs powered without a design effect are overpowered on paper."""
    clustered = randomization_unit.lower() not in ("individual", "patient", "member", "person")
    return [Finding("PWR-002", "error", f"randomizing by {randomization_unit} but design effect = {design_effect}",
                    "Apply 1 + (m - 1) x ICC (cluster_sample_size) or simulate.")] if clustered and design_effect <= 1.0 else []


def check_few_clusters(clusters_per_arm: int, minimum: int = 8) -> list[Finding]:
    """PWR-006: with few clusters, normal-theory power and tests are unreliable."""
    return [Finding("PWR-006", "warn", f"only {clusters_per_arm} clusters per arm",
                    "Simulate power with a small-sample test (t on cluster means) and consider more, smaller clusters.")] \
        if clusters_per_arm < minimum else []


def check_randomization_balance(assigned: pd.DataFrame, max_block: int = 4) -> list[Finding]:
    """PWR-003: permuted blocks guarantee |arm difference| <= max_block / 2 within every stratum."""
    counts = assigned.groupby(["stratum_id", "arm"]).size().unstack(fill_value=0)
    diff = counts.max(axis=1) - counts.min(axis=1)
    bad = diff[diff > max_block // 2]
    return [Finding("PWR-003", "error", f"{len(bad)} strata exceed the block imbalance bound",
                    "Randomization code is wrong - do not use this list.", len(bad))] if len(bad) else []


def check_multiplicity(n_primary_endpoints: int, alpha_adjusted: bool) -> list[Finding]:
    """PWR-004: several primary endpoints at alpha 0.05 each inflate the chance of a false win."""
    return [Finding("PWR-004", "warn", f"{n_primary_endpoints} primary endpoints without alpha adjustment",
                    "Pick one primary, or use Holm / hierarchical testing and power for the adjusted alpha.")] \
        if n_primary_endpoints > 1 and not alpha_adjusted else []


def check_skewed_outcome(cv: float, threshold: float = 1.5) -> list[Finding]:
    """PWR-005: very skewed outcomes (cost) make t-test formulas optimistic; simulate the planned analysis."""
    return [Finding("PWR-005", "info", f"outcome CV {cv:.1f} (skewed)",
                    "Use simulate_power with the real analysis (DiD / two-part / bootstrap); consider truncation.")] \
        if cv > threshold else []
