"""
G08 — LACE readmission risk index (van Walraven et al., CMAJ 2010)
==================================================================
Copy this whole block into a file (e.g. lace.py) and run:  python lace.py
Requires: pandas   (pip install pandas)   Charlson score: see G05.
"""
import numpy as np
import pandas as pd


def lace_score(stays: pd.DataFrame) -> pd.DataFrame:
    """LACE = Length of stay + Acuity of admission + Comorbidity (Charlson) + ED visits in prior 6 months.

    Healthcare context
    ------------------
    A four-item bedside score for 30-day readmission or death after discharge. Transitional-care and
    home-health programs use it to decide who gets a follow-up call or visit. Score 0-19; >= 10 is the
    usual "high risk" cut-off.

    Parameters
    ----------
    stays : stay_id, los_days (int), emergent_admit_flag (0/1), charlson_score (int), ed_visits_6m (int)

    Returns the input with l_pts, a_pts, c_pts, e_pts, lace, high_risk_flag.

    Points
    ------
    L: <1 day 0, 1 day 1, 2 days 2, 3 days 3, 4-6 days 4, 7-13 days 5, >= 14 days 7
    A: emergent (unplanned) admission 3, else 0
    C: Charlson 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3, >= 4 -> 5
    E: ED visits in the 6 months before admission, capped at 4

    Common mistakes
    ---------------
    - Counting the index ED visit (the one that led to this admission) in E.
    - Charlson from diagnoses after discharge (use history up to the index stay).
    - Treating LACE as calibrated for every population; check observed rates by score band locally.
    """
    s = stays.copy()
    los = s.los_days
    s["l_pts"] = np.select([los < 1, los == 1, los == 2, los == 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
    s["a_pts"] = 3 * s.emergent_admit_flag
    cci = s.charlson_score
    s["c_pts"] = np.where(cci >= 4, 5, cci)
    s["e_pts"] = s.ed_visits_6m.clip(upper=4)
    s["lace"] = s[["l_pts", "a_pts", "c_pts", "e_pts"]].sum(axis=1)
    s["high_risk_flag"] = (s.lace >= 10).astype(int)
    return s


# ---------------------------------------------------------------------------
# Self-test: run `python lace.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    stays = pd.DataFrame({"stay_id": ["S1", "S2", "S3"], "los_days": [0, 5, 20], "emergent_admit_flag": [0, 1, 1],
                          "charlson_score": [0, 2, 6], "ed_visits_6m": [0, 1, 9]})
    r = lace_score(stays).set_index("stay_id")
    print(r)
    checks = {
        "S1: same-day elective, healthy = 0": r.loc["S1", "lace"] == 0,
        "S2: 4 + 3 + 2 + 1 = 10 -> high risk": r.loc["S2", ["lace", "high_risk_flag"]].tolist() == [10, 1],
        "S3: 7 + 3 + 5 + 4 = 19 (maximum)": r.loc["S3", "lace"] == 19,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
