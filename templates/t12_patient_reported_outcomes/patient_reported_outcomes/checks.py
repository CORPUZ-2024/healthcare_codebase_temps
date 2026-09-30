"""PRO checks. Each returns a list of Finding (empty = clean)."""
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


def check_license(spec: dict) -> list[Finding]:
    """PRO-004: licensed instruments must not be reproduced or scored without a license."""
    ok = spec.get("license") in ("public_domain", "fake_teaching", "licensed_on_file")
    return [] if ok else [Finding("PRO-004", "error", f"{spec.get('instrument_id')}: license '{spec.get('license')}' not cleared",
                                  "Confirm the license (e.g. Mapi Research Trust) and set license: licensed_on_file.")]


def check_unscorable(scores: pd.DataFrame, observed: pd.Series, max_share: float = 0.05) -> list[Finding]:
    """PRO-001: questionnaires returned but unscorable (too many skipped items)."""
    returned = observed == 1
    share = float(scores.loc[returned, "score"].isna().mean()) if returned.any() else 0.0
    return [Finding("PRO-001", "warn", f"{share:.1%} of returned questionnaires are unscorable",
                    "Review skip patterns / item wording; never fill skipped items with 0.")] if share > max_share else []


def check_differential_dropout(wide: pd.DataFrame, visit: int = 6, max_gap: float = 0.05) -> list[Finding]:
    """PRO-002: different completion rates by arm signal informative dropout."""
    comp = wide.groupby("arm_flag")[f"score_{visit}"].apply(lambda s: s.notna().mean())
    gap = float(abs(comp.max() - comp.min()))
    return [Finding("PRO-002", "warn", f"completion at month {visit} differs by {gap:.0%} between arms ({comp.round(2).to_dict()})",
                    "Use a likelihood-based mixed model (MAR) and a tipping-point sensitivity analysis; report dropout reasons.")] \
        if gap > max_gap else []


def check_floor_ceiling(fc: dict, max_share: float = 0.15) -> list[Finding]:
    """PRO-003: many scores at the minimum or maximum -> instrument can't show change there."""
    worst = max(fc["floor_share"], fc["ceiling_share"])
    return [Finding("PRO-003", "warn", f"floor {fc['floor_share']:.0%} / ceiling {fc['ceiling_share']:.0%}",
                    "Consider a more sensitive instrument for this population.")] if worst > max_share else []


def check_reliability(alpha: float, minimum: float = 0.70) -> list[Finding]:
    """PRO-005: Cronbach's alpha below 0.70 -> items don't hang together (reverse-scoring error?)."""
    return [Finding("PRO-005", "error", f"Cronbach's alpha {alpha:.2f}",
                    "Check reverse-scored items first; then translation / administration problems.")] if alpha < minimum else []
