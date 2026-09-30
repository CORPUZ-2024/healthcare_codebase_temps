"""
G06 — NYU ED algorithm: probability-weighted classification of ED visits
========================================================================
Copy this whole block into a file (e.g. nyu_ed.py) and run:  python nyu_ed.py
Requires: pandas   (pip install pandas)
The real lookup (diagnosis -> probabilities) comes from NYU Wagner (Billings et al.); an updated ICD-10
version is the "NYU EDA" by Johnston et al. (2017). The table below is a FAKE 4-row stand-in.
"""
import pandas as pd

CATEGORIES = ["nonemergent", "emergent_pc_treatable", "emergent_ed_care_preventable", "emergent_not_preventable"]
SPECIAL = ["injury", "mental_health", "alcohol", "drug", "unclassified"]

FAKE_LOOKUP = pd.DataFrame({  # dx_cd -> probability of each category (rows sum to 1); FAKE numbers
    "dx_cd": ["J069", "N390", "J45901", "I214"],
    "nonemergent": [0.66, 0.30, 0.10, 0.00], "emergent_pc_treatable": [0.28, 0.55, 0.20, 0.00],
    "emergent_ed_care_preventable": [0.00, 0.05, 0.60, 0.00], "emergent_not_preventable": [0.06, 0.10, 0.10, 1.00],
})


def classify_ed_visits(visits: pd.DataFrame, lookup: pd.DataFrame, special: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach NYU probabilities to ED visits by principal diagnosis and summarize.

    Healthcare context
    ------------------
    Payers and ACOs use the NYU algorithm to estimate how much ED use could have been handled in
    primary care. It does NOT classify each visit: it assigns each diagnosis a PROBABILITY of each
    category, so a population's "avoidable" share is a sum of probabilities, not a count.

    Parameters
    ----------
    visits : visit_id, member_id, dx_cd (principal diagnosis)
    lookup : dx_cd + the four category probability columns
    special : optional dx_cd -> special category (injury, mental_health, alcohol, drug)

    Returns (visit-level frame with probabilities and category='special'/'classified'/'unclassified').

    Steps
    -----
    1. Normalize codes; left-join the lookup on principal diagnosis.
    2. Special categories (injury, mental health, substance) take precedence; unmatched codes = unclassified.
    3. avoidable = nonemergent + emergent_pc_treatable (the usual "potentially avoidable" definition).

    Common mistakes
    ---------------
    - Rounding probabilities to a single category per visit (the algorithm's authors advise against it).
    - Ignoring the unclassified share (often 10-20%); report it.
    - Using a secondary diagnosis instead of the principal one.
    """
    v = visits.assign(dx_cd=visits.dx_cd.str.replace(".", "", regex=False).str.upper())                 # step 1
    out = v.merge(lookup, on="dx_cd", how="left")
    out["category"] = "classified"
    if special is not None:                                                                                # step 2
        out = out.merge(special.rename(columns={"category": "special_cd"}), on="dx_cd", how="left")
        is_special = out["special_cd"].notna()
        out.loc[is_special, CATEGORIES] = 0.0
        out.loc[is_special, "category"] = out.loc[is_special, "special_cd"]
        out = out.drop(columns="special_cd")
    unmatched = out[CATEGORIES].isna().all(axis=1)
    out.loc[unmatched, "category"] = "unclassified"
    out[CATEGORIES] = out[CATEGORIES].fillna(0.0)
    out["avoidable"] = out["nonemergent"] + out["emergent_pc_treatable"]                                  # step 3
    return out


def population_summary(classified: pd.DataFrame) -> pd.Series:
    """Expected share of visits in each category (sum of probabilities / visits), plus special and unclassified shares."""
    n = len(classified)
    s = classified[CATEGORIES + ["avoidable"]].sum() / n
    for c in SPECIAL:
        s[c] = (classified["category"] == c).mean()
    return s


# ---------------------------------------------------------------------------
# Self-test: run `python nyu_ed.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    visits = pd.DataFrame({"visit_id": range(6), "member_id": list("AABCDE"),
                           "dx_cd": ["J06.9", "J06.9", "N39.0", "I21.4", "S52.501A", "Z99.99"]})
    special = pd.DataFrame({"dx_cd": ["S52501A"], "category": ["injury"]})
    c = classify_ed_visits(visits, FAKE_LOOKUP, special)
    s = population_summary(c)
    print(c[["visit_id", "dx_cd", "category", "avoidable"]], "\n", s.round(3))
    checks = {
        "probabilities of classified visits sum to 1": (c.loc[c.category == "classified", CATEGORIES].sum(axis=1).round(9) == 1).all(),
        "URI visit avoidable = 0.66 + 0.28": abs(c.avoidable.iloc[0] - 0.94) < 1e-9,
        "injury routed to special category": c.category.iloc[4] == "injury",
        "unknown code -> unclassified": c.category.iloc[5] == "unclassified",
        "population avoidable = (0.94 x 2 + 0.85 + 0) / 6": abs(s["avoidable"] - (0.94 * 2 + 0.85) / 6) < 1e-9,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
