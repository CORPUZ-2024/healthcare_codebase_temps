"""
G26 — Medicaid churn: enrollment spells, gaps and re-enrollment within N months
===============================================================================
Copy this whole block into a file (e.g. churn.py) and run:  python churn.py
Requires: pandas   (pip install pandas)   Real enrollment histories: T-MSIS Analytic Files (TAF, DUA required).
"""
import pandas as pd


def spells(monthly: pd.DataFrame) -> pd.DataFrame:
    """Collapse member-month enrollment flags into continuous spells.

    Parameters
    ----------
    monthly : member_id, month (Period[M] or first-of-month Timestamp), enrolled_flag (0/1)

    Returns member_id, spell_no, start_month, end_month, months.
    """
    m = monthly[monthly.enrolled_flag == 1].copy()
    m["month"] = pd.PeriodIndex(m["month"], freq="M")
    m = m.sort_values(["member_id", "month"])
    idx = m["month"].apply(lambda p: p.ordinal)
    new = idx.groupby(m.member_id).diff().ne(1)
    m["spell_no"] = new.groupby(m.member_id).cumsum()
    return (m.groupby(["member_id", "spell_no"]).month.agg(start_month="min", end_month="max", months="size").reset_index())


def churn_events(sp: pd.DataFrame, within_months: int = 3) -> pd.DataFrame:
    """Every disenrollment followed by re-enrollment within ``within_months`` = a churn event.

    Healthcare context
    ------------------
    "Churn" - losing Medicaid and regaining it within months, often for paperwork reasons at
    redetermination - interrupts care, raises cost (pent-up use after re-enrollment) and breaks
    continuous-enrollment rules for quality measures (G04). After the end of the pandemic continuous
    enrollment condition (2023 "unwinding"), states and plans track it closely.

    Returns member_id, exit_month, return_month, gap_months, churn_flag (gap <= within_months).

    Steps
    -----
    1. Pair each spell's end with the member's next spell start.
    2. gap = months between (exclusive); churn if 1 <= gap <= within_months.

    Common mistakes
    ---------------
    - Counting the last spell's end as an exit when it's just the end of the data (right-censored).
    - Using a gap of 0 (continuous) as churn after a plan switch - that's a transfer, not a loss.
    """
    s = sp.sort_values(["member_id", "start_month"]).copy()
    s["next_start"] = s.groupby("member_id").start_month.shift(-1)                                        # step 1
    s = s.dropna(subset=["next_start"])
    gap = s.apply(lambda r: r.next_start.ordinal - r.end_month.ordinal - 1, axis=1)
    return pd.DataFrame({"member_id": s.member_id, "exit_month": s.end_month, "return_month": s.next_start,
                         "gap_months": gap.astype(int), "churn_flag": gap.between(1, within_months).astype(int)}).reset_index(drop=True)   # step 2


# ---------------------------------------------------------------------------
# Self-test: run `python churn.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    months = pd.period_range("2024-01", "2024-12", freq="M")
    pattern = {"STAY": [1] * 12, "CHURN": [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], "LEAVE": [1] * 5 + [0] * 7,
               "LONG": [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}
    monthly = pd.DataFrame([(m, mo, f) for m, flags in pattern.items() for mo, f in zip(months, flags)],
                           columns=["member_id", "month", "enrolled_flag"])
    sp = spells(monthly)
    ev = churn_events(sp).set_index("member_id")
    print(sp, "\n", ev)
    checks = {
        "STAY has one 12-month spell": sp[sp.member_id == "STAY"].months.tolist() == [12],
        "CHURN: 2-month gap -> churn event": ev.loc["CHURN", ["gap_months", "churn_flag"]].tolist() == [2, 1],
        "LONG: 5-month gap -> re-enrollment but not churn (<= 3 months rule)": ev.loc["LONG", ["gap_months", "churn_flag"]].tolist() == [5, 0],
        "LEAVE: exit with no return is not a churn event (right-censored)": "LEAVE" not in ev.index,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
