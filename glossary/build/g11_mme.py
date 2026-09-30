"""
G11 — Opioid morphine milligram equivalents (MME) per day
=========================================================
Copy this whole block into a file (e.g. mme.py) and run:  python mme.py
Requires: pandas   (pip install pandas)
Conversion factors below follow the CDC 2022 Clinical Practice Guideline table (VERIFY against the current
CDC table and, for NDC-level work, CDC's "Opioid NDC and Oral MME Conversion File").
"""
import pandas as pd

MME_FACTOR = {  # per mg of drug (fentanyl transdermal: per mcg/hr)
    "codeine": 0.15, "fentanyl_transdermal": 2.4, "hydrocodone": 1.0, "hydromorphone": 5.0, "methadone": 4.7,
    "morphine": 1.0, "oxycodone": 1.5, "oxymorphone": 3.0, "tapentadol": 0.4, "tramadol": 0.2,
}


def daily_mme(fills: pd.DataFrame) -> pd.DataFrame:
    """MME per day per member-day, summing overlapping prescriptions.

    Healthcare context
    ------------------
    Payers, Part D sponsors and state Medicaid programs monitor high-dose opioid use (the CDC guideline
    discusses reassessment at >= 50 MME/day and avoiding increases to >= 90 MME/day without justification).
    Per-fill MME/day = strength x (quantity / days supply) x conversion factor; a member's daily total
    sums every fill covering that day.

    Parameters
    ----------
    fills : member_id, fill_dt, ingredient (key of MME_FACTOR), strength_per_unit (mg or mcg/hr),
            quantity (units), days_supply

    Returns member_id, day, mme (one row per member per covered day).

    Steps
    -----
    1. Per fill: MME/day = strength_per_unit x quantity / days_supply x factor. Transdermal fentanyl:
       MME/day = patch strength (mcg/hr) x 2.4 - the number of patches per day does not enter.
    2. Expand each fill across the days it covers (fill_dt .. fill_dt + days_supply - 1).
    3. Sum by member x day.

    Common mistakes
    ---------------
    - Averaging MME over the whole year instead of per day (hides peaks).
    - Treating fentanyl patch quantity as tablets (strength is mcg/HOUR; one patch covers ~3 days).
    - Using the old CDC 2016 tiered methadone factors with the 2022 table (mixing versions).
    - Including buprenorphine for OUD (usually excluded from MME monitoring).
    """
    f = fills.copy()
    f["mme_per_day"] = f.strength_per_unit * f.quantity / f.days_supply * f.ingredient.map(MME_FACTOR)   # step 1
    patch = f.ingredient == "fentanyl_transdermal"            # a patch delivers mcg/HOUR continuously while worn:
    f.loc[patch, "mme_per_day"] = f.loc[patch, "strength_per_unit"] * MME_FACTOR["fentanyl_transdermal"]
    rows = []
    for r in f.itertuples(index=False):                                                                   # step 2
        for d in pd.date_range(r.fill_dt, periods=int(r.days_supply), freq="D"):
            rows.append((r.member_id, d, r.mme_per_day))
    daily = pd.DataFrame(rows, columns=["member_id", "day", "mme"])
    return daily.groupby(["member_id", "day"], as_index=False)["mme"].sum()                              # step 3


def high_dose_summary(daily: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
    """Per member: max daily MME, days at or above ``threshold``, and a flag."""
    g = daily.groupby("member_id").mme
    out = pd.DataFrame({"max_mme": g.max(), "days_at_or_above": daily[daily.mme >= threshold].groupby("member_id").size()})
    out["days_at_or_above"] = out["days_at_or_above"].fillna(0).astype(int)
    out["high_dose_flag"] = (out["days_at_or_above"] > 0).astype(int)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Self-test: run `python mme.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fills = pd.DataFrame({
        "member_id": ["A", "A", "B"], "fill_dt": pd.to_datetime(["2025-03-01", "2025-03-11", "2025-03-01"]),
        "ingredient": ["oxycodone", "hydrocodone", "fentanyl_transdermal"],
        "strength_per_unit": [10.0, 10.0, 25.0], "quantity": [120, 60, 10], "days_supply": [30, 30, 30]})
    daily = daily_mme(fills)
    s = high_dose_summary(daily).set_index("member_id")
    a = daily[daily.member_id == "A"].set_index("day").mme
    print(s)
    checks = {
        "oxycodone 10 mg x 4/day x 1.5 = 60 MME/day": a.loc["2025-03-01"] == 60.0,
        "overlap with hydrocodone 10 mg x 2/day adds 20 -> 80": a.loc["2025-03-15"] == 80.0,
        "A never reaches 90": s.loc["A", "high_dose_flag"] == 0,
        "fentanyl 25 mcg/hr patch x 2.4 = 60 MME/day (not 25 x 10/30 x 2.4 = 20)":
            abs(daily[daily.member_id == "B"].mme.iloc[0] - 60.0) < 1e-9,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
