"""
G27 — Dual-eligible identification from monthly Medicare-Medicaid dual status codes
===================================================================================
Copy this whole block into a file (e.g. dual_status.py) and run:  python dual_status.py
Requires: pandas   (pip install pandas)
VERIFY the code definitions against the current data dictionary (e.g. ResDAC's MBSF / MMA dual status
code documentation) - code sets have changed over time.
"""
import pandas as pd

FULL_BENEFIT = {"02": "QMB plus full Medicaid", "04": "SLMB plus full Medicaid", "08": "Other full-benefit dual"}
PARTIAL_BENEFIT = {"01": "QMB only", "03": "SLMB only", "05": "QDWI", "06": "QI"}


def classify_month(code) -> str:
    """'full' / 'partial' / 'non_dual_or_unknown' for one monthly dual status code (strings, zero-padded)."""
    c = str(code).strip().zfill(2) if pd.notna(code) else ""
    if c in FULL_BENEFIT:
        return "full"
    if c in PARTIAL_BENEFIT:
        return "partial"
    return "non_dual_or_unknown"


def annual_dual_status(monthly: pd.DataFrame, rule: str = "any_full") -> pd.DataFrame:
    """Person-year dual category from 12 monthly codes.

    Healthcare context
    ------------------
    Full-benefit duals get Medicare AND full Medicaid (including long-term services and supports);
    partial duals get help only with Medicare premiums / cost sharing. Costs, risk scores (the CMS-HCC
    model has separate full- and partial-dual segments), D-SNP eligibility and caregiver-program
    eligibility all depend on the distinction - and status changes month to month.

    Parameters
    ----------
    monthly : person_id, month, dual_status_cd (as delivered; read as STRING)
    rule : 'any_full' (full if any month is full, else partial if any partial) or 'majority'

    Returns person_id, full_months, partial_months, dual_category.

    Common mistakes
    ---------------
    - Reading codes as integers ('02' becomes 2 and '2' stops matching).
    - Using a single month (e.g. December) as the year's status.
    - Treating partial duals as Medicaid-covered for LTSS.
    """
    m = monthly.assign(cat=monthly.dual_status_cd.map(classify_month))
    g = m.groupby("person_id").cat
    out = pd.DataFrame({"full_months": g.apply(lambda s: int((s == "full").sum())),
                        "partial_months": g.apply(lambda s: int((s == "partial").sum()))})
    if rule == "any_full":
        out["dual_category"] = out.apply(lambda r: "full" if r.full_months else ("partial" if r.partial_months else "non_dual"), axis=1)
    else:
        out["dual_category"] = out.apply(lambda r: "full" if r.full_months >= 6 else ("partial" if r.full_months + r.partial_months >= 6 else "non_dual"), axis=1)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Self-test: run `python dual_status.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    months = range(1, 13)
    rows = [("A", m, "02") for m in months] + [("B", m, "01") for m in months] + \
           [("C", m, "00") for m in months] + [("D", m, "04" if m >= 10 else "03") for m in months] + [("E", m, 2) for m in months]
    monthly = pd.DataFrame(rows, columns=["person_id", "month", "dual_status_cd"])
    anyf = annual_dual_status(monthly).set_index("person_id")
    maj = annual_dual_status(monthly, "majority").set_index("person_id")
    print(anyf, "\n", maj)
    checks = {
        "A: QMB plus -> full": anyf.loc["A", "dual_category"] == "full",
        "B: QMB only -> partial": anyf.loc["B", "dual_category"] == "partial",
        "C: non-dual": anyf.loc["C", "dual_category"] == "non_dual",
        "D: 3 full months -> full under any_full, partial under majority": (anyf.loc["D", "dual_category"], maj.loc["D", "dual_category"]) == ("full", "partial"),
        "E: integer 2 re-padded to '02' -> full": anyf.loc["E", "dual_category"] == "full",
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
