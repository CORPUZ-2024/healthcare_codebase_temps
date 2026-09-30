"""Quality-measurement checks. Each returns a list of Finding (empty = clean)."""
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


def check_code_format(events: pd.DataFrame) -> list[Finding]:
    """QM-001: codes with dots, spaces or lower case silently fail a value-set join."""
    raw = events["code_cd"].astype(str)
    n = int((raw.str.contains(r"[.\s]", regex=True) | (raw != raw.str.upper())).sum())
    return [Finding("QM-001", "warn", f"{n} event codes contain dots/spaces/lower case",
                    "Normalize codes (methods.normalize_codes) before joining to value sets.", n)] if n else []


def check_overlapping_spans(enrollment: pd.DataFrame) -> list[Finding]:
    """QM-002: overlapping enrollment spans (retro plan changes) - CE logic must not double count."""
    e = enrollment.sort_values(["member_id", "enroll_start_dt"])
    prev_end = e.groupby("member_id")["enroll_end_dt"].transform(lambda s: s.cummax().shift())
    n = int(e.loc[e["enroll_start_dt"] <= prev_end, "member_id"].nunique())
    return [Finding("QM-002", "info", f"{n} members have overlapping enrollment spans",
                    "Expected with retro changes; CE uses a running max of end dates. Never sum span days.", n)] if n else []


def check_runout(data_cut_dt, my: int, min_days: int = 90) -> list[Finding]:
    """QM-010: rates run before claims runout understate the numerator (services not yet billed)."""
    days = (pd.Timestamp(data_cut_dt) - pd.Timestamp(f"{my}-12-31")).days
    return [Finding("QM-010", "error", f"data cut is only {days} days after the end of MY{my}",
                    f"Wait for >= {min_days} days of runout (HEDIS final rates use data through ~March-May), "
                    "or label the rate 'preliminary'.", 0)] if days < min_days else []


def check_value_sets_exist(specs: dict, value_sets: pd.DataFrame) -> list[Finding]:
    """QM-030: a value set named in a YAML spec but absent from the table gives a silent 0 numerator."""
    have = set(value_sets["value_set_name"])
    missing = set()
    for s in specs.values():
        names = list(s["numerator"]["value_sets"]) + [x["value_set"] for x in s.get("exclusions", [])]
        names += list((s.get("denominator_event") or {}).get("value_sets", []))
        missing |= {f"{s['measure_id']}:{n}" for n in names if n not in have}
    return [Finding("QM-030", "error", f"value sets not found: {sorted(missing)}",
                    "Fix the YAML name or load the value set.", len(missing))] if missing else []


def check_small_denominators(summaries: pd.DataFrame, min_n: int = 30) -> list[Finding]:
    """QM-020: rates on < 30 members swing by 3+ points per member; flag, don't rank."""
    small = summaries[summaries["denominator_cnt"] < min_n]
    return [Finding("QM-020", "warn", f"{len(small)} rate(s) with denominator < {min_n}: {small['measure_id'].tolist()}",
                    "Report 'NR' or pool years/groups; show the CI.", len(small))] if len(small) else []


def check_benchmark_vintage(bench: pd.DataFrame, my: int) -> list[Finding]:
    """ANL-013: Core Set year Y mostly reflects services in calendar year Y-1; methodology must match."""
    out = []
    years = set(bench["core_set_year"].dropna().astype(int)) if "core_set_year" in bench else set()
    if years and years != {my + 1}:
        out.append(Finding("ANL-013", "warn", f"benchmark Core Set year(s) {sorted(years)} vs. MY{my} (expected {my + 1})",
                           "Use the Core Set year whose measurement period matches your MY, or label the lag.", 0))
    if "methodology" in bench and bench["methodology"].nunique() > 1:
        out.append(Finding("ANL-013", "warn", "benchmark mixes methodologies: " + ", ".join(sorted(bench["methodology"].dropna().unique())),
                           "Compare admin rates with admin benchmarks (hybrid rates run higher).", 0))
    return out
