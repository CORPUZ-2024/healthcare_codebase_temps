"""
G30 — Small-cell suppression with complementary suppression
===========================================================
Copy this whole block into a file (e.g. suppress.py) and run:  python suppress.py
Requires: pandas   (pip install pandas)
"""
import pandas as pd


def suppress_small_cells(
    table: pd.DataFrame,
    min_cell: int = 11,
    show_totals: bool = True,
    mask_text: str = "<11",
) -> pd.DataFrame:
    """Hide small counts in a crosstab so individuals cannot be re-identified.

    Healthcare context
    ------------------
    CMS's cell-size suppression policy prohibits publishing any cell with a
    count from 1 to 10 derived from CMS data (zero may be shown). Most DUAs
    and many state Medicaid agencies apply the same rule. If row totals are
    published, hiding one cell is not enough: total minus the visible cells
    reveals it. Complementary (secondary) suppression hides a second cell.

    Parameters
    ----------
    table : DataFrame
        Counts only (ints). Rows = one grouping (e.g. county), columns =
        another (e.g. age band). Do NOT include a totals row/column.
    min_cell : int, default 11
        Smallest count that may be shown. Counts 1..min_cell-1 are hidden.
    show_totals : bool, default True
        If True, row and column totals are appended AND complementary
        suppression is applied so they cannot be used to back-solve.
    mask_text : str
        Text shown in place of a hidden count.

    Returns
    -------
    DataFrame of strings, safe to publish.

    Steps
    -----
    1. Primary: mark every cell with 1 <= count < min_cell.
    2. Complementary (only if totals shown): for any row or column that has
       exactly ONE hidden cell, also hide the smallest remaining non-zero
       cell in that row/column. Repeat until nothing changes, because hiding
       a cell in a row can create a new single-hidden column.
    3. Render: hidden -> mask_text, everything else -> formatted number.
       Totals are always computed from the TRUE counts.

    Common mistakes
    ---------------
    - Suppressing cells but publishing totals (back-solvable).
    - Suppressing percentages but not the counts they came from, or showing
      a rate whose numerator is 1-10 (the rate reveals the count).
    - Checking cell size once, then filtering/re-aggregating the table.
    - Treating 0 as small: zero is allowed and hiding it can confuse readers.
    """
    counts = table.astype(int)
    hidden = (counts >= 1) & (counts < min_cell)          # Step 1 — primary

    if show_totals:                                       # Step 2 — complementary
        changed = True
        while changed:
            changed = False
            for axis_name, frame_iter in (("row", hidden.iterrows()),
                                          ("col", hidden.items())):
                for key, mask in frame_iter:
                    if mask.sum() != 1:
                        continue
                    vals = counts.loc[key] if axis_name == "row" else counts[key]
                    candidates = vals[(~mask) & (vals > 0)]
                    if candidates.empty:
                        continue
                    victim = candidates.idxmin()
                    if axis_name == "row":
                        hidden.loc[key, victim] = True
                    else:
                        hidden.loc[victim, key] = True
                    changed = True

    out = counts.map(lambda v: f"{v:,}").mask(hidden, mask_text)  # Step 3
    if show_totals:
        out["Total"] = counts.sum(axis=1).map(lambda v: f"{v:,}")
        out.loc["Total"] = [f"{v:,}" for v in counts.sum(axis=0)] + [f"{counts.values.sum():,}"]
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python suppress.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t = pd.DataFrame(
        {"0-17":  [120, 4, 0, 60, 8],
         "18-64": [310, 55, 40, 210, 70],
         "65+":   [95, 30, 12, 80, 25]},
        index=["County A", "County B", "County C", "County D", "County E"],
    )
    safe = suppress_small_cells(t)
    print(safe, "\n")

    hidden_cells = (safe.iloc[:-1, :-1] == "<11")
    checks = {
        "Primary: County B 0-17 (n=4) hidden":           safe.loc["County B", "0-17"] == "<11",
        "Zero is shown, not hidden":                     safe.loc["County C", "0-17"] == "0",
        "No row with totals has exactly 1 hidden cell":  (hidden_cells.sum(axis=1) != 1).all(),
        "No column has exactly 1 hidden cell":           (hidden_cells.sum(axis=0) != 1).all(),
        "Totals use true counts":                        safe.loc["Total", "Total"] == f"{t.values.sum():,}",
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
