"""
G10 — Proportion of Days Covered (PDC), Stars-style medication adherence
========================================================================
Copy this whole block into a file (e.g. pdc.py) and run:  python pdc.py
Requires: pandas, numpy   (pip install pandas numpy)
"""
import pandas as pd
import numpy as np


def compute_pdc(
    fills: pd.DataFrame,
    period_start: str,
    period_end: str,
    min_fills: int = 2,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Compute Proportion of Days Covered (PDC) per member and drug class.

    Healthcare context
    ------------------
    PDC is the adherence metric behind the Medicare Part D Star Ratings
    adherence measures (diabetes meds, RAS antagonists, statins) and many
    Medicaid pharmacy programs. A member is "adherent" when PDC >= 0.80.
    Health plans use it to target pharmacist outreach; a VBC partner uses it
    to show that its care model keeps members on therapy.

    Parameters
    ----------
    fills : DataFrame
        One row per pharmacy claim (paid, not reversed). Required columns:
          member_id   str   member identifier
          drug_class  str   therapeutic class the measure is built on
          fill_dt     date  date of service (dispense date)
          days_supply int   days supply on the claim (> 0)
    period_start, period_end : str
        Measurement year boundaries, 'YYYY-MM-DD', inclusive.
    min_fills : int, default 2
        Denominator rule: member needs at least this many fills on
        DIFFERENT dates in the period to be measured.
    threshold : float, default 0.80
        Adherence cut point.

    Returns
    -------
    DataFrame: member_id, drug_class, index_dt, days_in_window,
               days_covered, pdc, adherent_flag

    Steps
    -----
    1. Keep fills inside the period; drop rows with days_supply <= 0.
    2. Denominator: keep member x class with >= min_fills distinct fill dates.
    3. Window = index date (first fill) through period_end.
    4. Walk fills in date order. If a fill arrives while the member still has
       supply on hand, SHIFT its start to the day after current supply ends
       (early refills are not double counted, but are not lost either).
    5. Truncate coverage at period_end. Count covered days.
    6. pdc = days_covered / days_in_window; adherent_flag = pdc >= threshold.

    Example
    -------
    Fills on Jan 1 (30d) and Feb 15 (30d), period Jan 1 - Mar 31 (90 days):
    covered = 30 + 30 = 60 -> pdc = 0.667 -> not adherent.

    Common mistakes
    ---------------
    - Summing days_supply / days_in_window: early refills push PDC above 1.
    - Using Jan 1 as the window start for everyone instead of the index date.
    - Forgetting reversals: a reversed claim left in the data looks like a fill.
    - Mixing classes: a statin fill should never cover a diabetes-med gap.
    """
    p_start, p_end = pd.Timestamp(period_start), pd.Timestamp(period_end)

    # Step 1 — restrict to the period and valid supply
    df = fills.copy()
    df["fill_dt"] = pd.to_datetime(df["fill_dt"])
    df = df[(df["fill_dt"] >= p_start) & (df["fill_dt"] <= p_end) & (df["days_supply"] > 0)]

    rows = []
    for (member, drug_class), grp in df.groupby(["member_id", "drug_class"]):
        # Step 2 — denominator rule: distinct fill dates
        if grp["fill_dt"].nunique() < min_fills:
            continue

        grp = grp.sort_values("fill_dt")
        index_dt = grp["fill_dt"].iloc[0]                    # Step 3
        days_in_window = (p_end - index_dt).days + 1

        covered = np.zeros(days_in_window, dtype=bool)        # one slot per day
        next_free = 0                                         # first uncovered day offset
        for fill_dt, supply in zip(grp["fill_dt"], grp["days_supply"]):
            start = max((fill_dt - index_dt).days, next_free)  # Step 4 — shift
            end = min(start + int(supply), days_in_window)     # Step 5 — truncate
            if start < end:
                covered[start:end] = True
            next_free = max(next_free, end)

        days_covered = int(covered.sum())
        pdc = days_covered / days_in_window                   # Step 6
        rows.append({
            "member_id": member, "drug_class": drug_class, "index_dt": index_dt.date(),
            "days_in_window": days_in_window, "days_covered": days_covered,
            "pdc": round(pdc, 4), "adherent_flag": pdc >= threshold,
        })

    return pd.DataFrame(rows, columns=["member_id", "drug_class", "index_dt",
                                       "days_in_window", "days_covered", "pdc", "adherent_flag"])


# ---------------------------------------------------------------------------
# Self-test: run `python pdc.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_fills = pd.DataFrame({
        "member_id":   ["A", "A", "A", "B", "B", "C"],
        "drug_class":  ["STATIN"] * 6,
        "fill_dt":     ["2025-01-01", "2025-01-25", "2025-03-01",   # A: early refills shift
                        "2025-01-01", "2025-02-15",                 # B: 15-day gap
                        "2025-01-10"],                              # C: 1 fill -> excluded
        "days_supply": [30, 30, 30, 30, 30, 30],
    })
    out = compute_pdc(test_fills, "2025-01-01", "2025-03-31").set_index("member_id")
    print(out, "\n")

    checks = {
        "A fully covered after shifting early refills (PDC = 1.0)": out.loc["A", "pdc"] == 1.0,
        "B has 60 of 90 days covered (PDC = 0.6667)":              out.loc["B", "pdc"] == 0.6667,
        "B is not adherent":                                       not out.loc["B", "adherent_flag"],
        "C excluded (only one fill)":                              "C" not in out.index,
        "PDC never exceeds 1":                                     (out["pdc"] <= 1).all(),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
