"""
G07 — AHRQ Prevention Quality Indicator (PQI)-style avoidable admission rate
=============================================================================
Copy this whole block into a file (e.g. pqi.py) and run:  python pqi.py
Requires: pandas   (pip install pandas)
Shows the SHAPE of PQI #08 (heart failure admission rate). The code list and exclusions below are an
ABBREVIATED teaching version. For reporting use AHRQ's free QI software and technical specifications
(qualityindicators.ahrq.gov), which carry the full value sets, exclusions and risk adjustment.
"""
import pandas as pd

HF_PRINCIPAL_DX = ("I50", "I110", "I130", "I132")          # ABBREVIATED heart-failure principal dx prefixes
CARDIAC_PROC_EXCLUSION = "cardiac_procedure_flag"            # in AHRQ specs: a list of ICD-10-PCS cardiac procedures


def pqi_rate(admits: pd.DataFrame, population: pd.DataFrame, per: int = 100_000) -> pd.DataFrame:
    """Area- or plan-level PQI-style rate: qualifying admissions per 100,000 adults, by group.

    Healthcare context
    ------------------
    PQIs count hospitalizations that good outpatient care can often prevent (heart failure, COPD,
    diabetes complications). The denominator is the POPULATION (residents or members), not
    discharges: it measures the ambulatory system, not the hospital.

    Parameters
    ----------
    admits : admit_id, member_id, group_cd, age, principal_dx_cd, transfer_in_flag, cardiac_procedure_flag
    population : group_cd, adult_population (members 18+ or member-years)

    Returns group_cd, numerator, adult_population, rate_per_100k.

    Steps
    -----
    1. Numerator: age >= 18, principal diagnosis in the value set.
    2. Exclusions: transfers in from another facility (counted at the first hospital) and admissions
       with a cardiac procedure (a different clinical pathway).
    3. Rate = numerator / population x 100,000.

    Common mistakes
    ---------------
    - Using SECONDARY diagnoses (the PQI uses the principal diagnosis only).
    - Using discharges as the denominator (that's an inpatient quality indicator, not a PQI).
    - Comparing raw rates across areas with different age mix; AHRQ risk-adjusts by age and sex.
    """
    a = admits.assign(dx=admits.principal_dx_cd.str.replace(".", "", regex=False).str.upper())
    num = a[(a.age >= 18) & a.dx.str.startswith(HF_PRINCIPAL_DX)                                   # step 1
            & (a.transfer_in_flag == 0) & (a[CARDIAC_PROC_EXCLUSION] == 0)]                        # step 2
    out = population.merge(num.groupby("group_cd").size().rename("numerator").reset_index(), on="group_cd", how="left")
    out["numerator"] = out["numerator"].fillna(0).astype(int)
    out["rate_per_100k"] = out["numerator"] / out["adult_population"] * per                         # step 3
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python pqi.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    admits = pd.DataFrame({
        "admit_id": range(7), "member_id": list("ABCDEFG"), "group_cd": ["X"] * 5 + ["Y"] * 2,
        "age": [70, 80, 65, 17, 72, 60, 58],
        "principal_dx_cd": ["I50.9", "I11.0", "I50.22", "I50.9", "J44.1", "I13.0", "I50.9"],
        "transfer_in_flag": [0, 0, 1, 0, 0, 0, 0], "cardiac_procedure_flag": [0, 0, 0, 0, 0, 0, 1]})
    pop = pd.DataFrame({"group_cd": ["X", "Y"], "adult_population": [20_000, 10_000]})
    r = pqi_rate(admits, pop).set_index("group_cd")
    print(r)
    checks = {
        "X numerator: 2 (transfer, minor, COPD excluded)": r.loc["X", "numerator"] == 2,
        "Y numerator: 1 (cardiac procedure excluded)": r.loc["Y", "numerator"] == 1,
        "X rate = 2 / 20,000 x 100,000 = 10": r.loc["X", "rate_per_100k"] == 10.0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
