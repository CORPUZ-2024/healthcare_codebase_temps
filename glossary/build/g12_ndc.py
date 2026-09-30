"""
G12 — NDC normalization: 10-digit (4-4-2 / 5-3-2 / 5-4-1) -> 11-digit 5-4-2
============================================================================
Copy this whole block into a file (e.g. ndc.py) and run:  python ndc.py
Requires: pandas   (pip install pandas)   Package/product lookup: FDA NDC Directory (free download).
"""
import re

import pandas as pd


def ndc_to_11(ndc: str) -> str | None:
    """Convert a hyphenated 10-digit NDC (or an 11-digit one) to the 11-digit 5-4-2 billing format.

    Healthcare context
    ------------------
    The FDA assigns 10-digit NDCs in three layouts (labeler-product-package): 4-4-2, 5-3-2, 5-4-1.
    Pharmacy claims (NCPDP) and HIPAA transactions use 11 digits (5-4-2), made by adding ONE leading
    zero to the short segment. Joining an FDA product file to claims fails until both are 11 digits.

    >>> ndc_to_11("1234-5678-90"), ndc_to_11("12345-678-90"), ndc_to_11("12345-6789-0")
    ('01234567890', '12345067890', '12345678900')

    Steps
    -----
    1. If already 11 digits (with or without hyphens), return the digits.
    2. With hyphens: pad the labeler to 5, product to 4, package to 2 (only one segment is short).
    3. A bare 10-digit string WITHOUT hyphens is ambiguous (which segment was short?) -> None; look it up
       in the FDA directory instead of guessing.

    Common mistakes
    ---------------
    - Reading NDCs as numbers (leading zeros vanish): always read as strings.
    - Padding a bare 10-digit NDC on the left (right answer only for the 4-4-2 layout).
    - Joining 10-digit FDA codes to 11-digit claim codes and concluding the drug "isn't in the file".
    """
    s = str(ndc).strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) == 11:                                                                          # step 1
        return digits
    parts = s.split("-")
    if len(parts) == 3 and len(digits) == 10:                                                      # step 2
        lab, prod, pkg = parts
        return lab.zfill(5) + prod.zfill(4) + pkg.zfill(2)
    return None                                                                                    # step 3


def format_11(ndc11: str) -> str:
    """'12345678901' -> '12345-6789-01' (for display and matching hyphenated references)."""
    return f"{ndc11[:5]}-{ndc11[5:9]}-{ndc11[9:]}"


def normalize_column(s: pd.Series) -> pd.DataFrame:
    """Vectorized helper: ndc_raw, ndc11, status ('ok' / 'ambiguous_10_digit' / 'invalid')."""
    out = pd.DataFrame({"ndc_raw": s.astype(str)})
    out["ndc11"] = out.ndc_raw.map(ndc_to_11)
    digits = out.ndc_raw.str.replace(r"\D", "", regex=True).str.len()
    out["status"] = "ok"
    out.loc[out.ndc11.isna() & (digits == 10), "status"] = "ambiguous_10_digit"
    out.loc[out.ndc11.isna() & (digits != 10), "status"] = "invalid"
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python ndc.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = normalize_column(pd.Series(["0002-3227-30", "50090-347-00", "12345-6789-0", "00002322730", "0002322730", "123"]))
    print(r)
    checks = {
        "4-4-2 pads labeler": r.ndc11[0] == "00002322730",
        "5-3-2 pads product": r.ndc11[1] == "50090034700",
        "5-4-1 pads package": r.ndc11[2] == "12345678900",
        "11-digit input passes through": r.ndc11[3] == "00002322730",
        "bare 10-digit is flagged ambiguous, not guessed": r.status[4] == "ambiguous_10_digit",
        "garbage flagged invalid": r.status[5] == "invalid",
        "display format": format_11("00002322730") == "00002-3227-30",
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
