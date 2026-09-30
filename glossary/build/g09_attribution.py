"""
G09 — Provider attribution by plurality of primary-care visits (two-step, MSSP-shaped)
=======================================================================================
Copy this whole block into a file (e.g. attribution.py) and run:  python attribution.py
Requires: pandas   (pip install pandas)
Contract rules differ (look-back, code lists, specialties, tie-breaks); replicate YOUR payer's file spec.
"""
import pandas as pd

PRIMARY_CARE_CODES = tuple([f"992{n:02d}" for n in range(1, 16)] + [f"993{n:02d}" for n in range(41, 51)]
                           + ["G0402", "G0438", "G0439"])      # office/home E&M + Medicare wellness (illustrative)
PCP_SPECIALTIES = {"FAMILY_MED", "INTERNAL_MED", "GERIATRICS", "PEDIATRICS", "GENERAL_PRACTICE", "NP", "PA"}


def attribute(visits: pd.DataFrame) -> pd.DataFrame:
    """Assign each member to one provider (TIN/NPI) by plurality of allowed charges for primary-care services.

    Healthcare context
    ------------------
    Value-based contracts pay a provider group for "its" members; attribution decides who those are.
    MSSP-style step-wise rule: (1) plurality among primary-care clinicians; (2) only if a member had NO
    primary-care-clinician visits, plurality among specialists. Denominators, PMPM and quality
    rates for the group all depend on this.

    Parameters
    ----------
    visits : member_id, npi_id, specialty_cd, hcpcs_cd, svc_dt, allowed_amt

    Returns member_id, npi_id, step (1 or 2), allowed_share, visit_cnt.

    Steps
    -----
    1. Keep primary-care service codes in the look-back window.
    2. Step 1 pool: PCP specialties; step 2 pool: everyone else, used only for members with no step-1 visits.
    3. Sum allowed $ by member x provider; pick the max; tie-break by most recent visit, then NPI.

    Common mistakes
    ---------------
    - Plurality of visit COUNTS when the contract says allowed charges (or vice versa).
    - Letting specialists win when the member also saw a PCP (skipping the step order).
    - Attributing on a window that doesn't match the contract (prospective vs. retrospective).
    """
    v = visits[visits.hcpcs_cd.isin(PRIMARY_CARE_CODES)].copy()                                  # step 1
    v["step"] = (~v.specialty_cd.isin(PCP_SPECIALTIES)).astype(int) + 1                          # step 2
    first = v.groupby("member_id")["step"].transform("min")
    v = v[v.step == first]
    g = v.groupby(["member_id", "npi_id", "step"]).agg(allowed=("allowed_amt", "sum"), visit_cnt=("svc_dt", "size"),
                                                        last_dt=("svc_dt", "max")).reset_index()
    g["allowed_share"] = g.allowed / g.groupby("member_id").allowed.transform("sum")
    g = g.sort_values(["member_id", "allowed", "last_dt", "npi_id"], ascending=[True, False, False, True])   # step 3
    return g.drop_duplicates("member_id")[["member_id", "npi_id", "step", "allowed_share", "visit_cnt"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-test: run `python attribution.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = pd.to_datetime
    visits = pd.DataFrame({
        "member_id": ["A", "A", "A", "B", "B", "C", "C", "D"],
        "npi_id": ["P1", "P1", "P2", "S1", "P3", "S1", "S2", "P4"],
        "specialty_cd": ["FAMILY_MED", "FAMILY_MED", "INTERNAL_MED", "CARDIOLOGY", "NP", "CARDIOLOGY", "ENDOCRINOLOGY", "FAMILY_MED"],
        "hcpcs_cd": ["99213", "99214", "99215", "99215", "99212", "99214", "99214", "93000"],
        "svc_dt": d(["2025-01-10", "2025-03-01", "2025-05-01", "2025-02-01", "2025-06-01", "2025-02-01", "2025-04-01", "2025-01-05"]),
        "allowed_amt": [95.0, 135.0, 180.0, 180.0, 60.0, 135.0, 135.0, 20.0]})
    r = attribute(visits).set_index("member_id")
    print(r)
    checks = {
        "A: P1 wins on allowed $ (230 vs 180) despite P2's pricier visit": r.loc["A", "npi_id"] == "P1",
        "B: the PCP (NP) wins step 1 even though the cardiologist has more $": r.loc["B", ["npi_id", "step"]].tolist() == ["P3", 1],
        "C: no PCP visits -> step 2; tie broken by most recent visit": r.loc["C", ["npi_id", "step"]].tolist() == ["S2", 2],
        "D: non-E&M service (ECG) does not attribute": "D" not in r.index,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
