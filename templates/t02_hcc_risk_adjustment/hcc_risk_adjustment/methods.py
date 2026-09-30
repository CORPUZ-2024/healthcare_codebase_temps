"""
HCC-style risk adjustment: diagnoses -> condition categories -> risk adjustment factor (RAF).

    score_published_weights    (STANDARD)  vs  score_empirical_weights  (ALTERNATIVE)
    + filter_acceptable_sources, map_dx_to_hcc, apply_hierarchies, build_features,
      demographic_variable, blend_versions, normalize_and_adjust, predictive_ratios,
      suspect_gaps, explain_member
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACCEPTABLE_SOURCES = {"IP", "OP", "PROF_F2F"}          # inpatient, outpatient, face-to-face professional
INTERACTIONS = {"DIAB_CHF": ({"DIAB_NC", "DIAB_CC"}, {"CHF"}), "CHF_COPD": ({"CHF"}, {"COPD"})}
# Real CMS phase-in of the 2024 CMS-HCC model (V28): share of the score from each version.
BLEND_WEIGHTS = {2023: {"V24": 1.0, "V28": 0.0}, 2024: {"V24": 0.67, "V28": 0.33},
                 2025: {"V24": 0.33, "V28": 0.67}, 2026: {"V24": 0.0, "V28": 1.0}}


# ---------------------------------------------------------------------------
# 1. Diagnoses -> HCCs
# ---------------------------------------------------------------------------

def filter_acceptable_sources(dx: pd.DataFrame) -> pd.DataFrame:
    """Keep diagnoses from risk-adjustment-eligible encounters only.

    Healthcare context
    ------------------
    CMS accepts diagnoses from inpatient, outpatient and face-to-face professional encounters
    with acceptable provider types. Diagnoses on lab, DME or diagnostic-radiology claims do not
    count — a lab order "for diabetes" is not documentation that a clinician assessed diabetes.
    Using them inflates scores and fails a RADV audit.
    """
    return dx[dx["source_cd"].isin(ACCEPTABLE_SOURCES)].copy()


def map_dx_to_hcc(dx: pd.DataFrame, dx_map: pd.DataFrame, version: str) -> pd.DataFrame:
    """Map each member's diagnoses to HCCs for one model version. Returns member_id, hcc (unique).

    Steps: normalize codes (upper, no decimal) -> inner join to the version's map -> dedupe.
    Unmapped codes simply drop out (most ICD-10 codes are not in any payment HCC).
    """
    d = dx.assign(dx_cd=dx["dx_cd"].astype(str).str.replace(".", "", regex=False).str.upper().str.strip())
    m = dx_map[dx_map["model_version"] == version][["dx_cd", "hcc"]]
    return d.merge(m, on="dx_cd")[["member_id", "hcc"]].drop_duplicates().reset_index(drop=True)


def apply_hierarchies(member_hcc: pd.DataFrame, hierarchy: pd.DataFrame, version: str) -> pd.DataFrame:
    """Drop lower-severity HCCs when a higher one in the same family is present.

    Example: CKD stage 5 drops CKD 4 and CKD 3 — the member is paid once, at the most severe level.

    Steps: for each (higher, drops) rule, remove member rows of ``drops`` where the member also
    has ``higher``.
    """
    h = hierarchy[hierarchy["model_version"] == version]
    has = member_hcc.groupby("member_id")["hcc"].agg(set)
    drop_pairs = set()
    for member, hccs in has.items():
        for r in h.itertuples():
            if r.higher_hcc in hccs and r.drops_hcc in hccs:
                drop_pairs.add((member, r.drops_hcc))
    keep = [(m, c) not in drop_pairs for m, c in zip(member_hcc["member_id"], member_hcc["hcc"])]
    return member_hcc[keep].reset_index(drop=True)


def demographic_variable(age: int, sex: str) -> str:
    """Age/sex cell name, e.g. F35_64. Bands: 0-34, 35-64, 65-74, 75-84, 85+."""
    band = "0_34" if age < 35 else "35_64" if age < 65 else "65_74" if age < 75 else "75_84" if age < 85 else "85P"
    return f"{sex}{band}"


def build_features(members: pd.DataFrame, member_hcc: pd.DataFrame, coef: pd.DataFrame, version: str) -> pd.DataFrame:
    """Member x variable 0/1 matrix: demographic cell, DUAL, DISABLED, HCCs, interactions, D4P.

    Only variables that exist in ``coef`` for the version (+ demographics) are kept, so an HCC that
    a version does not pay (e.g. ASTHMA in V28-like) never gets a column.
    """
    paid = set(coef.loc[coef["model_version"] == version, "variable"])
    X = pd.DataFrame(index=members["member_id"])
    cells = [demographic_variable(a, s) for a, s in zip(members["age"], members["sex_cd"])]
    for c in sorted(set(cells)):
        X[c] = [int(x == c) for x in cells]
    X["DUAL"] = members["dual_flag"].to_numpy()
    X["DISABLED"] = members["disabled_flag"].to_numpy()
    hcc_sets = member_hcc[member_hcc["hcc"].isin(paid)].groupby("member_id")["hcc"].agg(set)
    for h in sorted(paid - set(INTERACTIONS) - {"D4P"}):
        X[h] = [int(h in hcc_sets.get(m, set())) for m in X.index]
    for name, (a, b) in INTERACTIONS.items():
        if name in paid:
            X[name] = [int(bool(hcc_sets.get(m, set()) & a) and bool(hcc_sets.get(m, set()) & b)) for m in X.index]
    if "D4P" in paid:
        X["D4P"] = [int(len(hcc_sets.get(m, set())) >= 4) for m in X.index]
    return X


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------

def score_published_weights(X: pd.DataFrame, coef: pd.DataFrame, demo: pd.DataFrame, version: str) -> pd.Series:
    """STANDARD: raw risk score = sum of published coefficients for every variable the member has.

    Healthcare context
    ------------------
    CMS-HCC is an ADDITIVE model with weights published by CMS each year. The analyst does not
    estimate anything — they apply the published weights so their score matches what the payer
    will pay on. A score of 1.0 = an average-cost beneficiary in the model's calibration data.

    Steps
    -----
    1. Look up a coefficient for every column of X (HCCs/interactions from ``coef``,
       demographics/DUAL/DISABLED from ``demo``); missing = 0.
    2. score = X @ weights.
    """
    w = pd.concat([coef[coef["model_version"] == version].set_index("variable")["coefficient"],
                   demo[demo["model_version"] == version].set_index("variable")["coefficient"]])
    weights = pd.Series({c: float(w.get(c, 0.0)) for c in X.columns})
    return (X.astype(float) @ weights).rename(f"raw_score_{version}")


def score_empirical_weights(X: pd.DataFrame, cost: pd.Series, ridge: float = 1.0) -> tuple[pd.Series, pd.Series]:
    """ALTERNATIVE: re-estimate additive weights on YOUR population (ridge-penalized least squares).

    Trade-off
    ---------
    + Fits your population's actual cost structure (e.g. Medicaid children, home-care users), which
      the Medicare-calibrated model was not built for. This is how custom and CDPS-style models
      are made. Useful for internal stratification and program evaluation.
    - Not what the payer pays on; can't be used for revenue. Overfits small samples (ridge helps);
      rewards whatever is coded in your data, including coding artefacts. Must be validated
      out-of-sample.

    Returns (relative_score normalized to mean 1.0, fitted weights in "relative cost" units).
    """
    Xf = X.astype(float).to_numpy()
    y = (cost.reindex(X.index).fillna(0.0) / cost.reindex(X.index).fillna(0.0).mean()).to_numpy()
    A = Xf.T @ Xf + ridge * np.eye(Xf.shape[1])
    beta = np.linalg.solve(A, Xf.T @ y)
    pred = Xf @ beta
    return pd.Series(pred / pred.mean(), index=X.index, name="empirical_score"), pd.Series(beta, index=X.columns)


def blend_versions(scores: dict[str, pd.Series], payment_year: int) -> pd.Series:
    """Blend version scores with the payment year's phase-in weights (BLEND_WEIGHTS).

    PY2024: 67% V24 + 33% V28; PY2025: 33% / 67%; PY2026: 100% V28.
    """
    w = BLEND_WEIGHTS[payment_year]
    return sum(scores[v] * w[v] for v in w if w[v] > 0).rename("blended_score")


def normalize_and_adjust(raw: pd.Series, normalization_factor: float, coding_intensity: float = 0.059) -> pd.Series:
    """Payment score = raw / normalization factor x (1 - coding-pattern adjustment).

    Both values are published in the CMS Rate Announcement for each payment year; 5.9% is the
    statutory minimum MA coding-pattern adjustment. The values used in this template's demo are
    illustrative.
    """
    return (raw / normalization_factor * (1 - coding_intensity)).rename("payment_score")


# ---------------------------------------------------------------------------
# 3. Validation & operations
# ---------------------------------------------------------------------------

def predictive_ratios(score: pd.Series, cost: pd.Series, groups: int = 10) -> pd.DataFrame:
    """Predicted / actual cost by score decile (1.0 = well calibrated), plus overall R^2.

    predicted cost = score x (total actual / total score), i.e. the score rescaled to dollars.
    Ratios > 1 in low deciles and < 1 in the top decile are the classic pattern of a model that
    under-predicts the sickest members.
    """
    df = pd.DataFrame({"score": score, "actual": cost.reindex(score.index).fillna(0.0)})
    df["pred"] = df["score"] * df["actual"].sum() / df["score"].sum()
    df["decile"] = pd.qcut(df["score"].rank(method="first"), groups, labels=False) + 1
    out = df.groupby("decile").agg(n=("score", "size"), mean_score=("score", "mean"),
                                   actual=("actual", "sum"), predicted=("pred", "sum"))
    out["predictive_ratio"] = out["predicted"] / out["actual"]
    ss_res = ((df["actual"] - df["pred"]) ** 2).sum()
    ss_tot = ((df["actual"] - df["actual"].mean()) ** 2).sum()
    out.attrs["r2"] = float(1 - ss_res / ss_tot)
    return out.reset_index()


def suspect_gaps(member_hcc_all_sources: pd.DataFrame, member_hcc_acceptable: pd.DataFrame,
                 rx: pd.DataFrame, rx_to_hcc: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """List members whose data SUGGEST a condition that isn't documented on an acceptable encounter.

    Two evidence types: (a) the HCC appears only on unacceptable sources (lab/DME);
    (b) a drug class implies the condition but no HCC is coded.

    COMPLIANCE: this is a list for a clinician to assess the member at a face-to-face visit. A code
    is only valid if the condition is assessed and documented. Never add codes from this list.
    """
    rx_to_hcc = rx_to_hcc or {"INSULIN_OR_METFORMIN": {"DIAB_NC", "DIAB_CC"}, "LOOP_DIURETIC": {"CHF"},
                              "LAMA_LABA": {"COPD"}, "ANTIPSYCHOTIC": {"SCHIZ"}}
    acc = member_hcc_acceptable.groupby("member_id")["hcc"].agg(set)
    rows = []
    for r in member_hcc_all_sources.itertuples():
        if r.hcc not in acc.get(r.member_id, set()):
            rows.append((r.member_id, r.hcc, "dx on non-acceptable source only"))
    for r in rx.drop_duplicates(["member_id", "rx_class"]).itertuples():
        fam = rx_to_hcc.get(r.rx_class, set())
        if fam and not (fam & acc.get(r.member_id, set())):
            rows.append((r.member_id, sorted(fam)[0], f"drug evidence: {r.rx_class}"))
    return pd.DataFrame(rows, columns=["member_id", "suspected_hcc", "evidence"]).drop_duplicates()


def explain_member(member_id: str, X: pd.DataFrame, coef: pd.DataFrame, demo: pd.DataFrame, version: str) -> pd.DataFrame:
    """Audit trail: every variable that contributes to one member's score, with its weight."""
    w = pd.concat([coef[coef["model_version"] == version].set_index("variable")["coefficient"],
                   demo[demo["model_version"] == version].set_index("variable")["coefficient"]])
    row = X.loc[member_id]
    on = row[row == 1].index
    out = pd.DataFrame({"variable": on, "coefficient": [float(w.get(v, 0.0)) for v in on]})
    return pd.concat([out, pd.DataFrame({"variable": ["TOTAL"], "coefficient": [out["coefficient"].sum()]})],
                     ignore_index=True)
