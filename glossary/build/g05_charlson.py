"""
G05 — Charlson comorbidity index from ICD-10 codes (Quan 2005 coding algorithm, Charlson 1987 weights)
=====================================================================================================
Copy this whole block into a file (e.g. charlson.py) and run:  python charlson.py
Requires: pandas   (pip install pandas)
VERIFY the code lists against Quan et al., Med Care 2005;43:1130-9 (Table 1) before reporting. ICD-10-CM
adaptations exist (e.g. Glasheen et al. 2019); Elixhauser: use AHRQ's free Elixhauser Comorbidity Software Refined.
"""
import pandas as pd

# condition -> (weight, ICD-10 prefixes without dots). Prefix match: 'I50' matches I50, I509, I5020, ...
CHARLSON = {
    "mi": (1, ["I21", "I22", "I252"]),
    "chf": (1, ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"]),
    "pvd": (1, ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"]),
    "cvd": (1, ["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"]),
    "dementia": (1, ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]),
    "copd": (1, ["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63",
                 "J64", "J65", "J66", "J67", "J684", "J701", "J703"]),
    "rheum": (1, ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"]),
    "pud": (1, ["K25", "K26", "K27", "K28"]),
    "mild_liver": (1, ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74",
                       "K760", "K762", "K763", "K764", "K768", "K769", "Z944"]),
    "dm_uncomp": (1, [f"{e}{d}" for e in ("E10", "E11", "E12", "E13", "E14") for d in ("0", "1", "6", "8", "9")]),
    "dm_comp": (2, [f"{e}{d}" for e in ("E10", "E11", "E12", "E13", "E14") for d in ("2", "3", "4", "5", "7")]),
    "hemiplegia": (2, ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"]),
    "renal": (2, ["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054", "N055",
                  "N056", "N057", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"]),
    "cancer": (2, [f"C{n:02d}" for n in list(range(0, 27)) + list(range(30, 35)) + list(range(37, 42)) + [43]
                   + list(range(45, 59)) + list(range(60, 77)) + list(range(81, 86)) + [88] + list(range(90, 98))]),
    "severe_liver": (3, ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"]),
    "metastatic": (6, ["C77", "C78", "C79", "C80"]),
    "hiv_aids": (6, ["B20", "B21", "B22", "B24"]),
}
HIERARCHY = [("dm_comp", "dm_uncomp"), ("severe_liver", "mild_liver"), ("metastatic", "cancer")]   # (keep, drop)


def charlson(dx: pd.DataFrame) -> pd.DataFrame:
    """Charlson flags and weighted index per person from diagnosis codes in a lookback window.

    Healthcare context
    ------------------
    The most common comorbidity adjustment in outcome studies and a component of LACE (G08).
    Use diagnoses from a fixed lookback (typically 12 months before the index date), from claim types
    you trust (inpatient + face-to-face outpatient; exclude rule-out lab diagnoses).

    Parameters
    ----------
    dx : person_id, dx_cd (ICD-10, dots optional)

    Returns person_id, one 0/1 column per condition, cci (weighted sum after hierarchies).

    Steps
    -----
    1. Normalize codes (drop dots, upper case); prefix-match against each condition list.
    2. One flag per person x condition; apply hierarchies (complicated diabetes drops uncomplicated,
       severe liver drops mild, metastatic drops solid tumour) so nothing is counted twice.
    3. cci = sum of weight x flag.

    Common mistakes
    ---------------
    - Counting the same condition twice (both diabetes categories) - apply the hierarchy.
    - Using diagnoses AFTER the index date (the outcome leaks into the adjustment).
    - Mixing ICD-9 and ICD-10 without a crosswalk (every code before Oct 2015 is ICD-9 in US claims).
    """
    d = dx.assign(code=dx.dx_cd.astype(str).str.replace(".", "", regex=False).str.upper().str.strip())   # step 1
    people = pd.Index(d.person_id.unique(), name="person_id")
    flags = pd.DataFrame(index=people)
    for cond, (_, prefixes) in CHARLSON.items():
        hit = d.code.str.startswith(tuple(prefixes))
        flags[cond] = flags.index.isin(d.loc[hit, "person_id"]).astype(int)
    for keep, drop in HIERARCHY:                                                                         # step 2
        flags.loc[flags[keep] == 1, drop] = 0
    flags["cci"] = sum(flags[c] * w for c, (w, _) in CHARLSON.items())                                   # step 3
    return flags.reset_index()


# ---------------------------------------------------------------------------
# Self-test: run `python charlson.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dx = pd.DataFrame({"person_id": ["A", "A", "A", "B", "B", "B", "C", "D"],
                       "dx_cd": ["I50.22", "E11.9", "E11.22", "C34.90", "C78.00", "N18.4", "Z00.00", "b20"]})
    r = charlson(dx).set_index("person_id")
    print(r.loc[:, (r != 0).any()])
    checks = {
        "A: CHF 1 + complicated diabetes 2 (uncomplicated dropped) = 3": r.loc["A", "cci"] == 3 and r.loc["A", "dm_uncomp"] == 0,
        "B: metastatic 6 (lung cancer dropped) + renal 2 = 8": r.loc["B", "cci"] == 8 and r.loc["B", "cancer"] == 0,
        "C: routine exam = 0": r.loc["C", "cci"] == 0,
        "D: lower-case HIV code normalized -> 6": r.loc["D", "cci"] == 6,
        "dotted 'I50.22' matches the I50 prefix (CHF)": r.loc["A", "chf"] == 1,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
