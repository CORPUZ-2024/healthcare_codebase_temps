"""
Synthetic members / enrollment / clinical events for quality measurement, plus loaders for the
measure YAML specs, the FAKE value-set table and the public Medicaid Core Set state rates.

Everything generated here is FAKE but shaped like payer data:
  * enrollment spans with realistic breaks: gaps of 10-120 days, late starts, early terminations,
    and ~3% overlapping spans (retro plan changes) that must NOT be counted as coverage twice
  * events carry a ``source_cd``: CLAIM (administrative), LAB (lab-results feed = standard
    supplemental data) or CHART (found only by medical-record review -> hybrid method)
  * diabetes coded on 1 date only for some members (fails the 2-date denominator rule)
  * a few diagnosis codes arrive with dots / lower case (``e11.9``) - the engine normalizes them
  * some mammograms fall just before the 27-month lookback (tests the window edge)

Public test / benchmark data
----------------------------
Medicaid & CHIP Core Set state-level rates (data.medicaid.gov) - see data/README.md and
``load_core_set_rates``. Member-level data is never public; use the synthetic generator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SOURCES = ("CLAIM", "LAB", "CHART")


# ---------------------------------------------------------------------------
# Specs and value sets
# ---------------------------------------------------------------------------

def load_measure(path: str | Path) -> dict:
    """Read one measure YAML into a dict and fill optional keys with defaults."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec.setdefault("denominator_event", None)
    spec.setdefault("exclusions", [])
    spec["population"].setdefault("sex", None)
    spec["numerator"].setdefault("hybrid_sources", [])
    return spec


def load_measures(measures_dir: str | Path) -> dict[str, dict]:
    """All ``*.yaml`` specs in a folder, keyed by measure_id (sorted for reproducible output)."""
    specs = [load_measure(p) for p in sorted(Path(measures_dir).glob("*.yaml"))]
    return {s["measure_id"]: s for s in specs}


def load_value_sets(path: str | Path) -> pd.DataFrame:
    """Value-set table: value_set_name, code_system, code_cd (strings, no dots, upper case)."""
    vs = pd.read_csv(path, dtype=str)
    vs["code_cd"] = vs["code_cd"].str.replace(".", "", regex=False).str.upper().str.strip()
    return vs


# ---------------------------------------------------------------------------
# Synthetic population
# ---------------------------------------------------------------------------

def generate_members(n: int = 4_000, anchor: str = "2025-12-31", seed: int = 11) -> pd.DataFrame:
    """member_id, birth_dt, sex_cd, lob_cd. Age mix is Medicaid-like (many children, some 65+ duals)."""
    rng = np.random.default_rng(seed)
    band = rng.choice(3, size=n, p=[0.35, 0.45, 0.20])
    age = np.where(band == 0, rng.integers(1, 22, n), np.where(band == 1, rng.integers(22, 65, n), rng.integers(65, 86, n)))
    birth = pd.Timestamp(anchor) - pd.to_timedelta(age * 365.25 + rng.integers(0, 365, n), unit="D")
    lob = np.where(age < 19, rng.choice(["MCD", "CHIP"], n, p=[0.8, 0.2]), np.where(age >= 65, "DUAL", "MCD"))
    return pd.DataFrame({"member_id": [f"Q{i:06d}" for i in range(1, n + 1)], "birth_dt": birth.normalize(),
                         "sex_cd": rng.choice(["F", "M"], n), "lob_cd": lob})


def generate_enrollment(members: pd.DataFrame, start: str = "2023-01-01", end: str = "2025-12-31",
                        seed: int = 11) -> pd.DataFrame:
    """Enrollment spans (inclusive dates). One row per span; breaks and overlaps are deliberate.

    Columns: member_id, plan_id, enroll_start_dt, enroll_end_dt
    """
    rng = np.random.default_rng(seed + 1)
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    horizon = (e0 - s0).days
    rows = []
    for mid in members["member_id"]:
        u = rng.random()
        if u < 0.70:                                           # enrolled throughout
            spans = [(s0, e0)]
        elif u < 0.82:                                         # one break of 10-120 days
            cut = s0 + pd.Timedelta(days=int(rng.integers(30, horizon - 130)))
            gap = int(rng.integers(10, 121))
            spans = [(s0, cut), (cut + pd.Timedelta(days=gap + 1), e0)]
        elif u < 0.91:                                         # joins late (often mid-MY)
            spans = [(s0 + pd.Timedelta(days=int(rng.integers(60, horizon - 30))), e0)]
        else:                                                  # leaves early
            spans = [(s0, s0 + pd.Timedelta(days=int(rng.integers(200, horizon))))]
        for a, b in spans:
            rows.append({"member_id": mid, "plan_id": "P01", "enroll_start_dt": a, "enroll_end_dt": b})
        if rng.random() < 0.03:                                # retro plan change -> overlapping span
            a = spans[0][0] + pd.Timedelta(days=int(rng.integers(5, 60)))
            rows.append({"member_id": mid, "plan_id": "P02", "enroll_start_dt": a,
                         "enroll_end_dt": min(a + pd.Timedelta(days=120), spans[0][1])})
    return pd.DataFrame(rows)


def _dates(rng, lo: str, hi: str, k: int) -> list[pd.Timestamp]:
    lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
    return [lo_t + pd.Timedelta(days=int(d)) for d in rng.integers(0, (hi_t - lo_t).days + 1, k)]


def generate_events(members: pd.DataFrame, my: int = 2025, seed: int = 11) -> pd.DataFrame:
    """Clinical events (one row per coded service / result) for the three FAKE measures.

    Columns: member_id, event_dt, code_system, code_cd, source_cd (CLAIM | LAB | CHART)

    Truth model (per member, independent): diabetes by age; A1c done 80% of diabetics
    (60% on a claim, 10% only in the lab feed, 10% only in the chart); mammogram 60% of women
    50+ (some dated just before the lookback); well-care visit 50% of ages 3-21; hospice 3% of 65+;
    bilateral mastectomy history 1.5% of women 40+.
    """
    rng = np.random.default_rng(seed + 2)
    my0, my1 = f"{my}-01-01", f"{my}-12-31"
    prior0 = f"{my - 1}-01-01"
    age = (pd.Timestamp(my1) - members["birth_dt"]).dt.days // 365.25
    rows = []

    def add(mid, dt, system, code, source="CLAIM"):
        rows.append((mid, dt, system, code, source))

    for m, a in zip(members.itertuples(index=False), age):
        mid = m.member_id
        for dt in _dates(rng, prior0, my1, int(rng.poisson(3))):          # background office visits
            add(mid, dt, "CPT", "99213")
        p_dm = 0.05 if a < 40 else (0.20 if a < 65 else 0.30)
        if a >= 18 and rng.random() < p_dm:
            n_dates = 1 if rng.random() < 0.15 else int(rng.integers(2, 6))
            code = rng.choice(["E119", "E1165", "E1122", "E109"], p=[0.6, 0.2, 0.15, 0.05])
            for dt in _dates(rng, prior0, my1, n_dates):
                raw = code if rng.random() > 0.04 else code[:3] + "." + code[3:].lower()  # 'E11.9'-style noise
                add(mid, dt, "ICD10CM", raw)
            u = rng.random()
            dt = _dates(rng, my0, my1, 1)[0]
            if u < 0.60:
                add(mid, dt, "CPT", "83036")
            elif u < 0.70:
                add(mid, dt, "LOINC", "4548-4", "LAB")
            elif u < 0.80:
                add(mid, dt, "LOINC", "4548-4", "CHART")
        if m.sex_cd == "F" and a >= 50 and rng.random() < 0.60:
            add(mid, _dates(rng, f"{my - 2}-07-01", my1, 1)[0], "CPT", rng.choice(["77067", "77066"]))
        if m.sex_cd == "F" and a >= 40 and rng.random() < 0.015:
            add(mid, _dates(rng, f"{my - 2}-01-01", my1, 1)[0], "ICD10CM", "Z9013")
        if 3 <= a <= 21 and rng.random() < 0.50:
            add(mid, _dates(rng, my0, my1, 1)[0], "CPT", "99393" if a < 12 else ("99394" if a < 18 else "99395"))
        if a >= 65 and rng.random() < 0.03:
            add(mid, _dates(rng, my0, my1, 1)[0], "HCPCS", "Q5001")
    ev = pd.DataFrame(rows, columns=["member_id", "event_dt", "code_system", "code_cd", "source_cd"])
    return ev.sort_values(["member_id", "event_dt"]).reset_index(drop=True)


def generate(n_members: int = 4_000, my: int = 2025, seed: int = 11, start: str = "2023-01-01") -> dict:
    """Convenience wrapper: {'members', 'enrollment', 'events'}."""
    members = generate_members(n_members, f"{my}-12-31", seed)
    return {"members": members, "enrollment": generate_enrollment(members, start, f"{my}-12-31", seed),
            "events": generate_events(members, my, seed)}


# ---------------------------------------------------------------------------
# Benchmarks: Medicaid & CHIP Core Set state rates (public) + FAKE stand-in
# ---------------------------------------------------------------------------
CORE_SET_COLS = {  # normalized header -> our name. Headers vary a little between releases.
    "state": "state", "measure abbreviation": "measure_cd", "ffy": "core_set_year",
    "core set year": "core_set_year", "population": "population", "methodology": "methodology",
    "state rate": "state_rate", "rate definition": "rate_definition",
}


def load_core_set_rates(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a Child/Adult Core Set state-rates CSV from data.medicaid.gov into a tidy frame.

    Steps
    -----
    1. Read as strings (or take an already-read DataFrame); match headers case-insensitively against CORE_SET_COLS.
    2. Parse ``state_rate``; rates are published as PERCENTAGES (52.3) -> divide by 100.
    3. Keep rows with a numeric rate (states that did not report, or were suppressed, are blank).

    Returns state, measure_cd, core_set_year, population, methodology, state_rate (0-1), ...

    Common mistakes
    ---------------
    - Comparing your 0-1 rate with a 0-100 benchmark (every plan looks terrible).
    - Ignoring ``methodology``/``population``: Admin vs. Hybrid and Medicaid-only vs. Medicaid+CHIP
      rates are not comparable.
    - Matching Core Set year to measurement year: FFY 2026 Core Set rates mostly reflect CY 2025
      services (check ANL-013).
    """
    df = path.astype(str) if isinstance(path, pd.DataFrame) else pd.read_csv(path, dtype=str)
    ren = {c: CORE_SET_COLS[c.strip().lower()] for c in df.columns if c.strip().lower() in CORE_SET_COLS}
    df = df.rename(columns=ren)[list(dict.fromkeys(ren.values()))]
    df["state_rate"] = pd.to_numeric(df["state_rate"], errors="coerce")
    if df["state_rate"].max() > 1.0:
        df["state_rate"] = df["state_rate"] / 100.0
    if "core_set_year" in df.columns:
        df["core_set_year"] = pd.to_numeric(df["core_set_year"], errors="coerce").astype("Int64")
    return df.dropna(subset=["state_rate"]).reset_index(drop=True)


def generate_fake_core_set(core_set_year: int = 2026, seed: int = 11) -> pd.DataFrame:
    """FAKE state rates with the same headers as the public file (for the demo and tests only)."""
    rng = np.random.default_rng(seed + 3)
    rows = []
    for cd, mu in (("BCS-AD", 50.0), ("WCV-CH", 48.0)):
        for i in range(45):
            rows.append({"State": f"State {i + 1:02d}", "Measure Abbreviation": cd, "FFY": str(core_set_year),
                         "Population": "Medicaid", "Methodology": "Administrative",
                         "State Rate": f"{np.clip(rng.normal(mu, 8.0), 15, 85):.1f}",
                         "Rate Definition": "FAKE rate"})
    return pd.DataFrame(rows)
