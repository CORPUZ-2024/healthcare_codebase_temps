"""
COPIED from t00_claims_foundation/claims_foundation/data.py (templates never import each other).
Premium generator added at the bottom for MLR.

Synthetic claims universe + loader for the public test file (CMS DE-SynPUF).

Everything here is FAKE but shaped like real payer data:
  * right-skewed cost with a long tail, December/January seasonality
  * claim adjustments (replacement versions) and voids on ~4% of claims
  * ~2% of members with overlapping enrollment spans (retro plan changes)
  * Luhn-valid NPIs, ZIP codes with leading zeros kept as strings
  * pharmacy claims with reversals
  * a declining trailing tail (claims runout) in the last 3 months
  * frailty-driven 30-day readmissions after inpatient stays

Public test dataset
-------------------
CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), Sample 1.
https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
No DUA required. See data/README.md. For TCOC benchmarking also see the CMS Geographic Variation PUF loader below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Small, REAL code lists (format-valid; used only to make the fake data look real)
# ---------------------------------------------------------------------------
ICD10_DX = {  # code (no decimal, as on claims) -> label
    "E119": "Type 2 diabetes w/o complications", "I10": "Essential hypertension",
    "I5020": "Systolic heart failure, unspecified", "J449": "COPD, unspecified",
    "F329": "Major depressive disorder, single episode", "N183": "CKD stage 3 (pre-FY2021 code)",
    "N1830": "CKD stage 3 unspecified", "G809": "Cerebral palsy, unspecified",
    "Q909": "Down syndrome, unspecified", "J45909": "Asthma, unspecified, uncomplicated",
    "Z0000": "General adult exam w/o abnormal findings", "R0602": "Shortness of breath",
    "S72001A": "Fracture of femoral neck, initial", "A419": "Sepsis, unspecified organism",
    "N390": "Urinary tract infection",
}
PROF_CODES = {  # HCPCS/CPT -> (label, typical allowed $)
    "99213": ("Office visit, est., low", 95), "99214": ("Office visit, est., moderate", 135),
    "99283": ("ED visit, moderate", 180), "99285": ("ED visit, high", 420),
    "99232": ("Subsequent hospital care", 110), "83036": ("Hemoglobin A1c", 13),
    "36415": ("Venipuncture", 3), "T1019": ("Personal care services, per 15 min", 6),
    "S5125": ("Attendant care, per 15 min", 5), "G0299": ("RN home health/hospice, 15 min", 45),
    "97110": ("Therapeutic exercise", 30), "90837": ("Psychotherapy, 60 min", 150),
}
DRUG_CLASSES = {"STATIN": 30, "RASA": 30, "DIABETES": 30, "INHALER": 30, "SSRI": 30, "ANTICONVULSANT": 30}
LOB = ["MCD", "MCD", "MCD", "DUAL", "MCR"]  # Medicaid-heavy mix (family-caregiver context)


def luhn_npi(base9: str) -> str:
    """Return a 10-digit NPI with a valid check digit.

    NPI check digits use the Luhn algorithm on the 9-digit base prefixed by '80840'.
    >>> luhn_npi("123456789")
    '1234567893'
    """
    digits = [int(c) for c in "80840" + base9]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:          # double every second digit starting from the right-most base digit
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return base9 + str((10 - total % 10) % 10)


def generate_providers(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """One row per provider NPI. Includes a few multi-location rows (join fan-out trap)."""
    rng = np.random.default_rng(seed)
    bases = rng.choice(np.arange(100_000_000, 199_999_999), size=n, replace=False)
    prov = pd.DataFrame({
        "npi_id": [luhn_npi(str(b)) for b in bases],
        "provider_type_cd": rng.choice(["IND", "ORG"], size=n, p=[0.8, 0.2]),
        "specialty_cd": rng.choice(["FAMILY_MED", "PEDIATRICS", "CARDIOLOGY", "HOSPITALIST",
                                    "EMERGENCY", "HOME_HEALTH", "BEHAVIORAL"], size=n),
        "practice_zip_cd": [f"{z:05d}" for z in rng.integers(1_000, 99_999, size=n)],
    })
    extra = prov.sample(3, random_state=seed).assign(practice_zip_cd=lambda d: "02139")  # multi-location
    return pd.concat([prov, extra], ignore_index=True)


def generate_enrollment(n_members: int = 1_000, start: str = "2023-01-01", months: int = 24,
                        seed: int = 42) -> pd.DataFrame:
    """Enrollment spans (one row per span). Partial months and overlaps are deliberate.

    Columns: member_id, lob_cd, plan_id, enroll_start_dt, enroll_end_dt (inclusive),
             birth_dt, sex_cd, zip_cd, cg_program_flag (enrolled in family-caregiver program)
    """
    rng = np.random.default_rng(seed)
    p_start = pd.Timestamp(start)
    p_end = p_start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    horizon = (p_end - p_start).days
    rows = []
    for i in range(1, n_members + 1):
        mid = f"M{i:06d}"
        # 60% enrolled all period; others join late / leave early on a random DAY (partial months)
        s = p_start if rng.random() < 0.7 else p_start + pd.Timedelta(days=int(rng.integers(0, horizon // 2)))
        e = p_end if rng.random() < 0.75 else s + pd.Timedelta(days=int(rng.integers(60, max(61, (p_end - s).days))))
        e = min(e, p_end)
        lob = rng.choice(LOB)
        age = int(rng.choice([rng.integers(1, 18), rng.integers(18, 65), rng.integers(65, 90)], p=[0.35, 0.4, 0.25]))
        birth = p_start - pd.Timedelta(days=int(age * 365.25 + rng.integers(0, 365)))
        base = dict(member_id=mid, lob_cd=lob, plan_id=f"P{rng.integers(1, 4):02d}", birth_dt=birth,
                    sex_cd=rng.choice(["F", "M"]), zip_cd=f"{int(rng.integers(900, 96_199)):05d}",
                    cg_program_flag=int(rng.random() < 0.3))
        rows.append({**base, "enroll_start_dt": s, "enroll_end_dt": e})
        if rng.random() < 0.02:  # retro plan change -> overlapping span
            ov_s = s + pd.Timedelta(days=int(rng.integers(10, 40)))
            rows.append({**base, "plan_id": "P09", "enroll_start_dt": ov_s,
                         "enroll_end_dt": min(ov_s + pd.Timedelta(days=90), p_end)})
    return pd.DataFrame(rows)


def generate_medical_claims(enrollment: pd.DataFrame, providers: pd.DataFrame, seed: int = 42,
                            runout_months: int = 3, as_of: str | None = None) -> pd.DataFrame:
    """Line-level medical claims (institutional + professional), replacement-style versions.

    Grain: claim_id x line_seq x adj_seq. adj_seq 0 = original; higher = replacement.
    claim_status_cd: 'P' paid, 'V' void (the whole claim is cancelled).
    Columns: claim_id, line_seq, adj_seq, claim_status_cd, member_id, claim_type_cd (INST/PROF),
             tob_cd (institutional type of bill), pos_cd (professional place of service),
             rev_cd, hcpcs_cd, dx1_cd, dx2_cd, admit_dt, discharge_dt, svc_from_dt, svc_to_dt,
             paid_dt, npi_id, allowed_amt, paid_amt
    """
    rng = np.random.default_rng(seed)
    npis = providers["npi_id"].drop_duplicates().to_numpy()
    dx_codes = np.array(list(ICD10_DX))
    prof_codes = np.array(list(PROF_CODES))
    rows, cid = [], 0
    p_end = enrollment["enroll_end_dt"].max()
    as_of = pd.Timestamp(as_of) if as_of else p_end
    for m in enrollment.drop_duplicates("member_id").itertuples(index=False):
        frailty = rng.lognormal(0, 0.9)                     # member-level driver -> long cost tail
        span_days = (m.enroll_end_dt - m.enroll_start_dt).days + 1
        n_events = rng.poisson(max(0.3, 0.012 * span_days * frailty))
        member_dx = rng.choice(dx_codes, size=3, replace=False)
        pending = []                                        # (service date, forced kind)
        for _ in range(n_events):
            svc = m.enroll_start_dt + pd.Timedelta(days=int(rng.integers(0, span_days)))
            if svc.month not in (12, 1) and rng.random() < 0.2:   # winter seasonality: redraw once,
                svc = m.enroll_start_dt + pd.Timedelta(days=int(rng.integers(0, span_days)))  # Dec/Jan kept
            pending.append((svc, None))
        while pending:
            svc, forced = pending.pop(0)
            if svc > m.enroll_end_dt:
                continue
            cid += 1
            kind = forced or rng.choice(["PROF", "PROF", "PROF", "ED", "IP", "HCBS", "OP"],
                                        p=[.33, .15, .1, .08, .03, .23, .08])
            lag = int(rng.gamma(2.0, 12 if kind != "IP" else 25))          # paid lag in days
            base = dict(claim_id=f"C{cid:08d}", member_id=m.member_id, dx1_cd=member_dx[0],
                        dx2_cd=rng.choice([member_dx[1], None]), npi_id=rng.choice(npis),
                        svc_from_dt=svc, paid_dt=svc + pd.Timedelta(days=lag), admit_dt=pd.NaT,
                        discharge_dt=pd.NaT, tob_cd=None, pos_cd=None, rev_cd=None)
            lines = []
            if kind == "IP":
                los = int(rng.integers(1, 9))
                base.update(claim_type_cd="INST", tob_cd="111", admit_dt=svc,
                            discharge_dt=svc + pd.Timedelta(days=los), svc_to_dt=svc + pd.Timedelta(days=los))
                amt = rng.lognormal(9.0, 0.5) * frailty ** 0.3
                if forced is None and rng.random() < min(0.35, 0.08 * frailty):   # frail members bounce back
                    pending.append((svc + pd.Timedelta(days=los + int(rng.integers(2, 31))), "IP"))
                lines = [("0120", None, amt * 0.7), ("0250", None, amt * 0.2), ("0300", None, amt * 0.1)]
            elif kind == "OP":
                base.update(claim_type_cd="INST", tob_cd="131", svc_to_dt=svc)
                lines = [("0510", "99214", rng.lognormal(5.5, 0.6)), ("0300", "83036", 13.0)]
            elif kind == "ED":
                base.update(claim_type_cd="INST", tob_cd="131", svc_to_dt=svc)
                lines = [("0450", rng.choice(["99283", "99285"]), rng.lognormal(6.5, 0.6))]
            elif kind == "HCBS":
                units = int(rng.integers(16, 120))            # 4-30 hours of personal care
                base.update(claim_type_cd="PROF", pos_cd="12", svc_to_dt=svc)  # POS 12 = home
                lines = [(None, rng.choice(["T1019", "S5125"]), units * 6.0)]
            else:
                code = rng.choice(prof_codes[:2])
                base.update(claim_type_cd="PROF", pos_cd=rng.choice(["11", "11", "02"]), svc_to_dt=svc)
                lines = [(None, code, PROF_CODES[code][1] * rng.uniform(0.8, 1.3))]
                if rng.random() < 0.2:
                    lines.append((None, "36415", 3.0))
            for ln, (rev, hcpcs, amt) in enumerate(lines, start=1):
                allowed = round(float(amt), 2)
                rows.append({**base, "line_seq": ln, "adj_seq": 0, "claim_status_cd": "P", "rev_cd": rev,
                             "hcpcs_cd": hcpcs, "allowed_amt": allowed, "paid_amt": round(allowed * 0.9, 2)})
    claims = pd.DataFrame(rows)

    # --- replacement versions (~3%) and voids (~1%) -----------------------------------------
    ids = claims["claim_id"].unique()
    adj_ids = rng.choice(ids, size=int(len(ids) * 0.03), replace=False)
    void_ids = rng.choice(np.setdiff1d(ids, adj_ids), size=int(len(ids) * 0.01), replace=False)
    adj = claims[claims["claim_id"].isin(adj_ids)].copy()
    adj["adj_seq"] = 1
    adj["paid_amt"] = (adj["paid_amt"] * rng.uniform(0.5, 1.2, len(adj))).round(2)
    adj["paid_dt"] = adj["paid_dt"] + pd.Timedelta(days=30)
    void = claims[claims["claim_id"].isin(void_ids)].copy()
    void["adj_seq"], void["claim_status_cd"] = 1, "V"
    void["paid_dt"] = void["paid_dt"] + pd.Timedelta(days=45)
    claims = pd.concat([claims, adj, void], ignore_index=True)

    # --- runout: claims not yet PAID by as_of are invisible (the trailing-month dip) -----------
    claims = claims[claims["paid_dt"] <= as_of + pd.Timedelta(days=0)]
    cols = ["claim_id", "line_seq", "adj_seq", "claim_status_cd", "member_id", "claim_type_cd", "tob_cd",
            "pos_cd", "rev_cd", "hcpcs_cd", "dx1_cd", "dx2_cd", "admit_dt", "discharge_dt", "svc_from_dt",
            "svc_to_dt", "paid_dt", "npi_id", "allowed_amt", "paid_amt"]
    return claims[cols].sort_values(["claim_id", "adj_seq", "line_seq"]).reset_index(drop=True)


def generate_pharmacy_claims(enrollment: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Pharmacy claims with reversals (reversal_flag=1 rows cancel an earlier fill).

    Columns: rx_claim_id, member_id, fill_dt, ndc_cd (11-digit string), drug_class,
             days_supply, qty, paid_amt, reversal_flag
    """
    rng = np.random.default_rng(seed + 1)
    rows, rid = [], 0
    for m in enrollment.drop_duplicates("member_id").itertuples(index=False):
        if rng.random() > 0.55:
            continue
        cls = rng.choice(list(DRUG_CLASSES))
        ndc = f"{int(rng.integers(0, 99_999)):05d}{int(rng.integers(0, 9_999)):04d}{int(rng.integers(0, 99)):02d}"
        d = m.enroll_start_dt + pd.Timedelta(days=int(rng.integers(0, 60)))
        adherence = rng.uniform(0.6, 1.05)
        while d <= m.enroll_end_dt:
            rid += 1
            ds = 30 if rng.random() < 0.85 else 90
            amt = round(float(rng.lognormal(3.2, 1.0)), 2)
            rows.append(dict(rx_claim_id=f"R{rid:08d}", member_id=m.member_id, fill_dt=d, ndc_cd=ndc,
                             drug_class=cls, days_supply=ds, qty=ds, paid_amt=amt, reversal_flag=0))
            if rng.random() < 0.02:                     # reversal of that fill (same id, negative $)
                rows.append(dict(rows[-1], paid_amt=-amt, reversal_flag=1, fill_dt=d))
            d = d + pd.Timedelta(days=int(ds / adherence))
    return pd.DataFrame(rows)


def generate_universe(n_members: int = 1_000, seed: int = 42, start: str = "2023-01-01",
                      months: int = 24) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: {'enrollment', 'providers', 'medical', 'pharmacy'}."""
    enr = generate_enrollment(n_members, start, months, seed)
    prov = generate_providers(seed=seed)
    return {"enrollment": enr, "providers": prov,
            "medical": generate_medical_claims(enr, prov, seed),
            "pharmacy": generate_pharmacy_claims(enr, seed)}


def to_delta_feed(claims: pd.DataFrame) -> pd.DataFrame:
    """Convert replacement-style versions into a DELTA-style feed.

    Some payers send each adjustment as a *difference* row (and a void as a full negative row)
    instead of a full replacement. Summing a delta feed gives the final amount; taking the
    latest version of a delta feed is WRONG. Used to show why you must know your feed type.
    """
    c = claims.sort_values(["claim_id", "line_seq", "adj_seq"]).copy()
    prev = c.groupby(["claim_id", "line_seq"])["paid_amt"].shift(1).fillna(0)
    c["paid_amt"] = np.where(c["claim_status_cd"] == "V", -prev, c["paid_amt"] - prev).round(2)
    return c


# ---------------------------------------------------------------------------
# Public test file loader
# ---------------------------------------------------------------------------
SYNPUF_CARRIER_MAP = {
    "DESYNPUF_ID": "member_id", "CLM_ID": "claim_id", "CLM_FROM_DT": "svc_from_dt",
    "CLM_THRU_DT": "svc_to_dt", "ICD9_DGNS_CD_1": "dx1_cd", "ICD9_DGNS_CD_2": "dx2_cd",
    "PRF_PHYSN_NPI_1": "npi_id", "HCPCS_CD_1": "hcpcs_cd", "LINE_NCH_PMT_AMT_1": "paid_amt",
    "LINE_ALOWD_CHRG_AMT_1": "allowed_amt",
}


def load_synpuf_carrier(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load DE-SynPUF Carrier Claims (Sample 1, file A or B) into this template's column names.

    Steps
    -----
    1. Read every column as string (IDs and codes must never become numbers).
    2. Keep and rename the columns in SYNPUF_CARRIER_MAP (line 1 of 13 only — the file is wide).
    3. Parse YYYYMMDD dates; cast amounts to float.
    4. Add the columns this template expects but SynPUF lacks (line_seq=1, adj_seq=0, status 'P').

    Caveat: SynPUF is 2008-2010, so diagnoses are ICD-9, not ICD-10. The ICD-10 format check
    will (correctly) flag them — use it to see what a code-system mismatch looks like.
    """
    df = pd.read_csv(path, dtype=str, nrows=nrows, usecols=lambda c: c in SYNPUF_CARRIER_MAP)
    df = df.rename(columns=SYNPUF_CARRIER_MAP)
    for c in ("svc_from_dt", "svc_to_dt"):
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    for c in ("paid_amt", "allowed_amt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.assign(line_seq=1, adj_seq=0, claim_status_cd="P", claim_type_cd="PROF", pos_cd=None,
                     tob_cd=None, rev_cd=None, paid_dt=df["svc_to_dt"])


# ---------------------------------------------------------------------------
# Premium (for MLR) — illustrative monthly capitation by line of business
# ---------------------------------------------------------------------------
PREMIUM_PMPM = {"MCD": 330.0, "DUAL": 325.0, "MCR": 285.0}   # FAKE: calibrated so synthetic MLR lands near 85-90%


def generate_premium(member_months: pd.DataFrame, lob_by_member: pd.Series | None = None,
                     rates: dict | None = None) -> pd.DataFrame:
    """Premium revenue per member-month = rate[lob] x member_months (prorated like exposure).

    ``member_months`` may already carry lob_cd; otherwise pass ``lob_by_member`` (index member_id).
    Returns member_id, month, lob_cd, premium_amt.
    """
    rates = rates or PREMIUM_PMPM
    p = (member_months.copy() if "lob_cd" in member_months.columns
         else member_months.merge(lob_by_member.rename("lob_cd"), left_on="member_id", right_index=True))
    p["premium_amt"] = (p["lob_cd"].map(rates) * p["member_months"]).round(2)
    return p[["member_id", "month", "lob_cd", "premium_amt"]]


def load_geovar_puf(path: str | Path) -> pd.DataFrame:
    """Load the CMS Medicare Geographic Variation PUF (state/county) for benchmarking PMPM.

    Source: https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county
    Keeps geography, year, beneficiary count and standardized per-capita spend columns whose
    names contain 'PYMT_PC' (per-capita payment). Column names change between releases, so
    this selects by pattern and prints what it kept.
    """
    df = pd.read_csv(path, dtype=str)
    keep = [c for c in df.columns if c.upper() in ("YEAR", "BENE_GEO_LVL", "BENE_GEO_DESC", "BENES_FFS_CNT")
            or "PYMT_PC" in c.upper()]
    out = df[keep].copy()
    for c in keep:
        if "PYMT_PC" in c.upper() or c.upper() == "BENES_FFS_CNT":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    print(f"load_geovar_puf kept {len(keep)} columns: {keep[:8]}{' ...' if len(keep) > 8 else ''}")
    return out
