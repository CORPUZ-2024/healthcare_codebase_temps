"""
Data-quality checks for claims, enrollment, providers and pharmacy.

Every check returns a list of Finding (empty = clean). Checks never modify data:
they tell you what is wrong and what to do about it. ``run_all`` runs the lot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

ICD10_CM_RE = re.compile(r"^[A-Z][0-9][0-9A-Z]([0-9A-Z]{1,4})?$")   # no decimal, 3-7 chars, letter first
HCPCS_RE = re.compile(r"^([0-9]{4}[0-9A-Z]|[A-V][0-9]{4})$")            # CPT (5 chars) or HCPCS Level II


@dataclass
class Finding:
    check_id: str
    severity: str      # "error" = fix before analysis; "warn" = analyse but disclose
    message: str
    fix: str
    n_rows: int = 0


def _f(cid, sev, n, msg, fix):
    return [Finding(cid, sev, msg.format(n=n), fix, int(n))] if n else []


def check_grain_unique(df: pd.DataFrame, keys: list[str], cid: str = "SCH-004") -> list[Finding]:
    """Declared grain must be unique. Duplicates double count every downstream sum."""
    n = int(df.duplicated(keys).sum())
    return _f(cid, "error", n, "{n} rows duplicate the grain " + "+".join(keys),
              "Find the extra key (version? location?) or de-duplicate before any join/sum.")


def check_npi_luhn(npis: pd.Series) -> list[Finding]:
    """NPIs are 10 digits with a Luhn check digit (prefix 80840). Also catches lost leading zeros."""
    def valid(x) -> bool:
        s = str(x)
        if not re.fullmatch(r"\d{10}", s):
            return False
        digits = [int(c) for c in "80840" + s]
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                d = d - 9 if d > 9 else d
            total += d
        return total % 10 == 0
    bad = npis.dropna().astype(str)
    n = int((~bad.map(valid)).sum())
    return _f("CLN-014", "warn", n, "{n} NPI values fail the 10-digit/Luhn check",
              "Read NPIs as strings (dtype=str). Invalid NPIs cannot be joined to NPPES.")


def check_icd10_format(codes: pd.Series, cid: str = "CLN-052") -> list[Finding]:
    """ICD-10-CM codes on claims: 3-7 chars, letter first, no decimal. ICD-9 (all digits / V/E) fails."""
    s = codes.dropna().astype(str).str.replace(".", "", regex=False).str.upper()
    n = int((~s.str.match(ICD10_CM_RE)).sum())
    return _f(cid, "warn", n, "{n} diagnosis codes are not ICD-10-CM shaped (ICD-9 or typos?)",
              "Check the service dates: ICD-10 applies to services on/after 2015-10-01. Map or exclude.")


def check_hcpcs_format(codes: pd.Series) -> list[Finding]:
    s = codes.dropna().astype(str).str.upper()
    n = int((~s.str.match(HCPCS_RE)).sum())
    return _f("CLN-053", "warn", n, "{n} HCPCS/CPT codes have an invalid format",
              "Strip modifiers into their own column; codes are exactly 5 characters.")


def check_ndc_format(ndcs: pd.Series) -> list[Finding]:
    s = ndcs.dropna().astype(str)
    n = int((~s.str.fullmatch(r"\d{11}")).sum())
    return _f("CLN-042", "warn", n, "{n} NDCs are not 11-digit strings",
              "Normalize with methods.normalize_ndc11; never store NDCs as numbers.")


def check_dates(claims: pd.DataFrame) -> list[Finding]:
    """svc_to >= svc_from; paid >= svc_from; discharge >= admit."""
    out = []
    out += _f("VAL-011", "error", int((claims["svc_to_dt"] < claims["svc_from_dt"]).sum()),
              "{n} lines end before they start", "Swap or null the dates; confirm with the source.")
    out += _f("VAL-014", "error", int((claims["paid_dt"] < claims["svc_from_dt"]).sum()),
              "{n} lines were paid before service", "Likely a date parsing issue (DD/MM vs MM/DD).")
    if "admit_dt" in claims:
        out += _f("VAL-015", "error", int((claims["discharge_dt"] < claims["admit_dt"]).sum()),
                  "{n} stays discharge before admission", "Fix dates before computing length of stay.")
    return out


def check_overlapping_spans(enr: pd.DataFrame) -> list[Finding]:
    """Overlapping enrollment spans for the same member double count member-months if not merged."""
    e = enr.sort_values(["member_id", "enroll_start_dt"])
    prev_end = e.groupby("member_id")["enroll_end_dt"].shift()
    n = int((e["enroll_start_dt"] <= prev_end).sum())
    return _f("CLN-061", "warn", n, "{n} enrollment spans overlap an earlier span for the same member",
              "Merge spans before member-months (methods._merge_overlaps does this); decide which plan wins.")


def check_negative_paid(claims: pd.DataFrame) -> list[Finding]:
    n = int((claims["paid_amt"] < 0).sum())
    return _f("CLN-023", "warn", n, "{n} lines have negative paid amounts",
              "Expected in delta feeds and reversals; unexpected in a collapsed replacement feed.")


def check_claims_outside_enrollment(claims: pd.DataFrame, enr: pd.DataFrame) -> list[Finding]:
    """A claim with a service date outside every enrollment span belongs to no member-month."""
    m = claims[["claim_id", "member_id", "svc_from_dt"]].drop_duplicates("claim_id").merge(
        enr[["member_id", "enroll_start_dt", "enroll_end_dt"]], on="member_id", how="left")
    m["inside"] = (m["svc_from_dt"] >= m["enroll_start_dt"]) & (m["svc_from_dt"] <= m["enroll_end_dt"])
    n = int((~m.groupby("claim_id")["inside"].any()).sum())
    return _f("VAL-012", "warn", n, "{n} claims fall outside every enrollment span",
              "Report them separately ('unmatched spend'); never silently drop them.")


def check_runout(claims: pd.DataFrame, months: int = 3, drop_threshold: float = 0.7) -> list[Finding]:
    """Trailing months with volume far below the prior average usually mean claims runout, not less care."""
    by_m = claims.groupby(claims["svc_from_dt"].dt.to_period("M"))["claim_id"].nunique().sort_index()
    if len(by_m) <= months + 3:
        return []
    base = by_m.iloc[-(months + 6):-months].mean()
    low = by_m.iloc[-months:][by_m.iloc[-months:] < drop_threshold * base]
    return _f("VAL-031", "warn", len(low),
              "{n} trailing month(s) below " + f"{drop_threshold:.0%} of baseline: " + ", ".join(map(str, low.index)),
              "Exclude or complete (IBNR, see t08) incomplete months before trending.")


def run_all(universe: dict[str, pd.DataFrame]) -> list[Finding]:
    """Run every check on a universe dict (enrollment, providers, medical, pharmacy)."""
    med, enr, prov, rx = (universe[k] for k in ("medical", "enrollment", "providers", "pharmacy"))
    out: list[Finding] = []
    out += check_grain_unique(med, ["claim_id", "line_seq", "adj_seq"])
    out += check_grain_unique(prov, ["npi_id"], cid="VAL-005")
    out += check_npi_luhn(med["npi_id"])
    out += check_icd10_format(pd.concat([med["dx1_cd"], med["dx2_cd"]]))
    out += check_hcpcs_format(med["hcpcs_cd"])
    out += check_ndc_format(rx["ndc_cd"])
    out += check_dates(med)
    out += check_overlapping_spans(enr)
    out += check_negative_paid(med)
    out += check_claims_outside_enrollment(med, enr)
    out += check_runout(med)
    return out
