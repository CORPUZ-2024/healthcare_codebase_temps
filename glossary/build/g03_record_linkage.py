"""
G03 — Fuzzy record linkage (member match across rosters)
=========================================================
Copy this whole block into a file (e.g. link.py) and run:  python link.py
Requires: pandas   (pip install pandas)   For production-scale probabilistic linkage see Splink / recordlinkage.
"""
import pandas as pd


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity in [0, 1] (1 = identical); rewards a shared prefix (typos cluster at the end).

    >>> round(jaro_winkler("MARTHA", "MARHTA"), 3)
    0.961
    """
    a, b = (a or "").upper(), (b or "").upper()
    if a == b:
        return 1.0 if a else 0.0
    if not a or not b:
        return 0.0
    rng = max(len(a), len(b)) // 2 - 1
    ma, mb = [False] * len(a), [False] * len(b)
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - rng), min(len(b), i + rng + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    sa = [c for c, f in zip(a, ma) if f]
    sb = [c for c, f in zip(b, mb) if f]
    t = sum(x != y for x, y in zip(sa, sb)) / 2
    jaro = (m / len(a) + m / len(b) + (m - t) / m) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * p * (1 - jaro)


def link_records(left: pd.DataFrame, right: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Match two person lists on name + date of birth + ZIP, blocking on birth date.

    Healthcare context
    ------------------
    Caregiver-program rosters, state files and payer eligibility rarely share an ID. Deterministic
    joins on exact name miss typos, nicknames and hyphenated names; fuzzy scoring inside a block
    (same birth date) keeps the comparison count small and the false-match rate low.

    Parameters
    ----------
    left, right : id, first_name, last_name, birth_dt (YYYY-MM-DD), zip_cd

    Returns
    -------
    left_id, right_id, score, match_flag (best right record per left record within its block).

    Steps
    -----
    1. Block: only compare pairs with the same birth_dt (also try month/day swapped in production).
    2. Score = 0.45 x JW(last) + 0.35 x JW(first) + 0.20 x (ZIP3 equal).
    3. Keep the best candidate per left record; match if score >= threshold. Review 0.80-0.90 by hand.

    Common mistakes
    ---------------
    - No blocking (n x m comparisons explode) or blocking on a field with typos (misses true matches).
    - One threshold with no clerical-review band.
    - Allowing one right record to match many left records without checking (1:1 constraint).
    """
    pairs = left.merge(right, on="birth_dt", suffixes=("_l", "_r"))                            # step 1
    pairs["score"] = [0.45 * jaro_winkler(a, b) + 0.35 * jaro_winkler(c, d) + 0.20 * (str(z1)[:3] == str(z2)[:3])  # step 2
                      for a, b, c, d, z1, z2 in zip(pairs.last_name_l, pairs.last_name_r, pairs.first_name_l,
                                                    pairs.first_name_r, pairs.zip_cd_l, pairs.zip_cd_r)]
    best = pairs.sort_values("score", ascending=False).drop_duplicates("id_l")                 # step 3
    out = best[["id_l", "id_r", "score"]].rename(columns={"id_l": "left_id", "id_r": "right_id"})
    out["match_flag"] = (out["score"] >= threshold).astype(int)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-test: run `python link.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    roster = pd.DataFrame({"id": ["CG1", "CG2", "CG3", "CG4"], "first_name": ["Martha", "Jon", "Maria", "Ann"],
                           "last_name": ["Nguyen", "Smith", "Garcia-Lopez", "Lee"],
                           "birth_dt": ["1950-02-03", "1961-07-19", "1972-11-30", "1980-01-01"],
                           "zip_cd": ["02139", "60614", "94110", "10001"]})
    elig = pd.DataFrame({"id": ["M9", "M8", "M7", "M6"], "first_name": ["Marhta", "John", "Maria", "Bob"],
                         "last_name": ["Nguyen", "Smith", "Garcia Lopez", "Lee"],
                         "birth_dt": ["1950-02-03", "1961-07-19", "1972-11-30", "1980-01-01"],
                         "zip_cd": ["02140", "60614", "94110", "10001"]})
    m = link_records(roster, elig).set_index("left_id")
    print(m)
    checks = {
        "JW('MARTHA','MARHTA') = 0.961 (textbook value)": round(jaro_winkler("MARTHA", "MARHTA"), 3) == 0.961,
        "typo in first name still matches": m.loc["CG1", "match_flag"] == 1,
        "Jon ~ John matches": m.loc["CG2", "match_flag"] == 1,
        "hyphen vs space in surname matches": m.loc["CG3", "match_flag"] == 1,
        "same surname + DOB, different first name does NOT match": m.loc["CG4", "match_flag"] == 0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
