"""Contract-modeling checks. Each returns a list of Finding (empty = clean)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REQUIRED = {
    "shared_savings": ["msr_pct", "max_sharing_rate", "savings_cap_pct", "two_sided"],
    "subcap": ["cap_pmpm", "corridors", "stoploss_attachment_amt", "stoploss_coinsurance"],
    "fee_upside": ["fee_pmpm", "target_trend_pct", "msr_pct", "upside_share", "quality_measures_required"],
}


@dataclass
class Finding:
    check_id: str
    severity: str
    message: str
    fix: str
    n_rows: int = 0


def check_reconciliation_closes(result: dict, tol: float = 1.0) -> list[Finding]:
    """ANL-015: every identity in a settlement must hold to the dollar."""
    bad = [(n, a, b) for n, a, b in result["identities"] if abs(a - b) > tol]
    return [Finding("ANL-015", "error", f"{result['contract_id']}: {len(bad)} identity(ies) do not close: "
                    + "; ".join(f"{n} ({a:,.2f} vs {b:,.2f})" for n, a, b in bad),
                    "Find the line that is double counted or missing before sending anything.", len(bad))] if bad else []


def check_contract_fields(contract: dict) -> list[Finding]:
    """VBC-001: a missing term silently defaults to 0 (no cap, no MSR)."""
    t = contract.get("type")
    if t not in REQUIRED:
        return [Finding("VBC-001", "error", f"{contract.get('contract_id')}: unknown contract type {t!r}",
                        f"Use one of {sorted(REQUIRED)}.")]
    missing = [k for k in REQUIRED[t] if k not in contract]
    return [Finding("VBC-001", "error", f"{contract['contract_id']}: missing terms {missing}",
                    "Add every term from the executed contract; don't rely on defaults.", len(missing))] if missing else []


def check_corridors(corridors: list[dict]) -> list[Finding]:
    """VBC-004: corridor bands must start at 0, be contiguous, and have shares in [0, 1]."""
    b = sorted(corridors, key=lambda x: x["lo"])
    ok = b[0]["lo"] == 0 and all(np.isclose(x["hi"], y["lo"]) for x, y in zip(b, b[1:])) \
        and all(0 <= x["provider_share"] <= 1 for x in b)
    return [] if ok else [Finding("VBC-004", "error", "risk corridors have gaps/overlaps or shares outside [0, 1]",
                                  "Rewrite bands as contiguous [lo, hi) ranges starting at 0.")]


def check_msr_vs_random_variation(msr_pct: float, n: int, cv: float, confidence: float = 0.90) -> list[Finding]:
    """VBC-002: if chance alone moves spending by more than the MSR, payouts reward noise."""
    from scipy import stats
    needed = float(stats.norm.ppf(confidence) * cv / np.sqrt(n))
    return [Finding("VBC-002", "warn", f"MSR {msr_pct:.2%} is below the {confidence:.0%} random-variation band "
                    f"{needed:.2%} for n={n:,} (CV {cv:.1f})",
                    "Expect frequent shared savings (or losses) by chance; ask for a larger MSR or a bigger population.", n)] \
        if needed > msr_pct else []


def check_target_basis(participant_baseline_pmpm: float, population_pmpm: float, ratio_warn: float = 1.25) -> list[Finding]:
    """VBC-003: a target built from participants' own high-cost baseline year pays for regression to the mean."""
    r = participant_baseline_pmpm / population_pmpm if population_pmpm else 0.0
    return [Finding("VBC-003", "warn", f"participants' baseline PMPM is {r:.1f}x the population's",
                    "Set the target from a matched comparison group's trend (t06) or a risk-adjusted benchmark.")] \
        if r > ratio_warn else []
