"""Time-to-event checks. Each returns a list of Finding (empty = clean)."""
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


def check_proportional_hazards(ph: pd.DataFrame, alpha: float = 0.05) -> list[Finding]:
    """SRV-001: Schoenfeld test per covariate, Bonferroni-adjusted (k tests at alpha / k) so one
    chance p < 0.05 among five covariates doesn't send you chasing a non-problem."""
    thr = alpha / max(len(ph), 1)
    bad = ph[ph["p_value"] < thr]
    return [Finding("SRV-001", "warn", f"non-proportional hazards (p < {thr:.3f}): {bad['covariate'].tolist()}",
                    "Stratify on it, model time-specific effects (discrete_time_hazard), or report RMST.", len(bad))] if len(bad) else []


def check_censoring(df: pd.DataFrame, horizon: int, duration: str = "duration_days", event: str = "event_flag",
                    max_share: float = 0.6) -> list[Finding]:
    """SRV-002: most members censored before the horizon -> late estimates rest on few people."""
    share = float(((df[duration] < horizon) & (df[event] == 0)).mean())
    return [Finding("SRV-002", "warn", f"{share:.0%} censored before day {horizon}",
                    "Report estimates only where at-risk counts are adequate; show the number at risk.")] if share > max_share else []


def check_events_per_variable(n_events: int, n_covariates: int, minimum: float = 10.0) -> list[Finding]:
    """SRV-003: fewer than ~10 events per covariate -> unstable Cox estimates."""
    epv = n_events / max(n_covariates, 1)
    return [Finding("SRV-003", "error", f"{epv:.1f} events per covariate",
                    "Fewer covariates or more follow-up / pooled cohorts.")] if epv < minimum else []


def check_competing_events(df: pd.DataFrame, code_col: str = "event_cd", competing_code: int = 2,
                           max_share: float = 0.05) -> list[Finding]:
    """SRV-004: deaths censored for a readmission analysis -> 1 - KM overstates readmission risk."""
    share = float((df[code_col] == competing_code).mean())
    return [Finding("SRV-004", "info", f"{share:.1%} of members had the competing event (treated as censored)",
                    "Hazard ratios are cause-specific; for absolute risk use cumulative incidence "
                    "(Aalen-Johansen, glossary G16).")] if share > max_share else []


def check_immortal_time(df: pd.DataFrame, start_col: str = "start_day") -> list[Finding]:
    """SRV-005: exposure that begins after time zero, analysed as if present from day 0."""
    late = int((df[start_col] > 0).sum()) if start_col in df else 0
    return [Finding("SRV-005", "error", f"{late} exposed members start the program after time zero",
                    "Use a time-varying exposure (cox_time_varying) or a landmark design; never 'ever vs. never'.",
                    late)] if late else []
