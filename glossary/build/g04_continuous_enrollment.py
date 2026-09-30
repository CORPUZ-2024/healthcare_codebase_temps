"""
G04 — Continuous enrollment with an allowable gap (HEDIS style)
===============================================================
Copy this whole block into a file (e.g. ce.py) and run:  python ce.py
Requires: pandas, numpy   (pip install pandas numpy)   Full measure engine: templates/t04.
"""
import numpy as np
import pandas as pd


def continuous_enrollment(spans: pd.DataFrame, start: str, end: str, anchor: str | None = None,
                          allowable_gaps: int = 1, max_gap_days: int = 45) -> pd.DataFrame:
    """Continuous-enrollment flag per member for one period.

    Healthcare context
    ------------------
    Quality measures only hold a plan accountable for members it covered long enough: HEDIS allows
    one gap of up to 45 days per year and requires enrollment on the anchor date (often Dec 31).
    A gap at the START or END of the period counts. Overlapping spans (retro plan changes) must not
    create or hide gaps.

    Parameters
    ----------
    spans : member_id, enroll_start_dt, enroll_end_dt (inclusive dates)
    start, end : period 'YYYY-MM-DD' (inclusive); anchor : date the member must be enrolled on

    Returns member_id, gap_cnt, max_gap_days, anchor_flag, ce_flag.

    Steps
    -----
    1. Clip spans to the period; sort by start.
    2. Running max of earlier end dates (so an overlapping span never looks like a gap).
    3. Gap before each span = start - previous running max - 1 (first span: start - period start); trailing gap = period end - max end.
    4. CE = gap count <= allowable_gaps AND longest gap <= max_gap_days AND enrolled on the anchor.

    Common mistakes
    ---------------
    - Counting days enrolled instead of gaps (two 30-day gaps fail even though 60 days < 45 x 2).
    - Treating adjacent spans (Jan 31 -> Feb 1) as a gap, or overlapping spans as extra coverage.
    - Applying one gap to a two-year lookback (the rule is one gap per year).
    """
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    e = spans[(spans.enroll_end_dt >= s0) & (spans.enroll_start_dt <= s1)].copy()
    e["s"], e["e"] = e.enroll_start_dt.clip(lower=s0), e.enroll_end_dt.clip(upper=s1)       # step 1
    e = e.sort_values(["member_id", "s"])
    prev = e.groupby("member_id")["e"].transform(lambda x: x.cummax().shift())             # step 2
    e["gap"] = np.where(prev.isna(), (e.s - s0).dt.days, (e.s - prev).dt.days - 1).clip(min=0)  # step 3
    g = e.groupby("member_id")
    out = pd.DataFrame({"lead_gaps": g.gap.apply(lambda x: int((x > 0).sum())), "lead_max": g.gap.max(),
                        "trail": (s1 - g.e.max()).dt.days})
    a = pd.Timestamp(anchor) if anchor else s1
    out["anchor_flag"] = ((e.s <= a) & (e.e >= a)).groupby(e.member_id).any().astype(int)
    out["gap_cnt"] = out.lead_gaps + (out.trail > 0)
    out["max_gap_days"] = out[["lead_max", "trail"]].max(axis=1).astype(int)
    out["ce_flag"] = ((out.gap_cnt <= allowable_gaps) & (out.max_gap_days <= max_gap_days) & (out.anchor_flag == 1)).astype(int)  # step 4
    return out.reset_index()[["member_id", "gap_cnt", "max_gap_days", "anchor_flag", "ce_flag"]]


# ---------------------------------------------------------------------------
# Self-test: run `python ce.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = lambda *xs: pd.to_datetime(list(xs))  # noqa: E731
    spans = pd.DataFrame({
        "member_id": ["FULL", "ADJ", "ADJ", "G45", "G45", "G46", "G46", "LATE", "TWO", "TWO", "TWO", "OVL", "OVL"],
        "enroll_start_dt": d("2025-01-01", "2025-01-01", "2025-02-01", "2025-01-01", "2025-03-18", "2025-01-01", "2025-03-19",
                             "2025-02-01", "2025-01-01", "2025-04-11", "2025-07-11", "2025-01-01", "2025-03-01"),
        "enroll_end_dt": d("2025-12-31", "2025-01-31", "2025-12-31", "2025-01-31", "2025-12-31", "2025-01-31", "2025-12-31",
                           "2025-12-31", "2025-03-31", "2025-06-30", "2025-12-31", "2025-12-31", "2025-05-31")})
    r = continuous_enrollment(spans, "2025-01-01", "2025-12-31").set_index("member_id")
    print(r)
    checks = {
        "full year is CE": r.loc["FULL", "ce_flag"] == 1,
        "adjacent spans are not a gap": r.loc["ADJ", "gap_cnt"] == 0,
        "45-day gap allowed": r.loc["G45", ["max_gap_days", "ce_flag"]].tolist() == [45, 1],
        "46-day gap fails": r.loc["G46", "ce_flag"] == 0,
        "late start = 31-day gap at the start, still CE": r.loc["LATE", ["max_gap_days", "ce_flag"]].tolist() == [31, 1],
        "two gaps fail": r.loc["TWO", ["gap_cnt", "ce_flag"]].tolist() == [2, 0],
        "overlapping spans don't create a gap": r.loc["OVL", "gap_cnt"] == 0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
