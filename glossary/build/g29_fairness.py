"""
G29 — Subgroup calibration and fairness audit for a risk model
===============================================================
Copy this whole block into a file (e.g. fairness.py) and run:  python fairness.py
Requires: numpy, pandas, statsmodels   (pip install numpy pandas statsmodels)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))) if 0 < n1 < len(y) else float("nan")


def subgroup_audit(df: pd.DataFrame, group: str, y: str = "outcome_flag", p: str = "pred_prob", threshold: float = 0.2,
                   max_oe_gap: float = 0.15, max_tpr_gap: float = 0.10) -> pd.DataFrame:
    """Per subgroup: calibration (O/E, calibration intercept and slope), discrimination (AUC), and
    who gets selected at the operating threshold (selection rate, TPR = sensitivity, PPV).

    Healthcare context
    ------------------
    Risk scores decide who gets care management. A model that is accurate overall can under-predict
    for a group - the widely cited case used past COST as the label, and because less money had been
    spent on Black patients with the same needs, they were under-selected (Obermeyer et al., Science 2019).
    Check every subgroup you can (dual status, language, area deprivation, race/ethnicity where
    available and lawful) for: is the predicted risk right (calibration) and are equally sick people
    equally likely to be selected (TPR)?

    Returns group, n, observed_rate, mean_pred, oe_ratio, cal_intercept, cal_slope, auc, selection_rate, tpr, ppv, flag.

    Steps
    -----
    1. Per group: O/E = observed events / sum of predictions (1 = calibrated in the large).
    2. Calibration intercept/slope: logistic regression of y on logit(p) (ideal 0 and 1).
    3. At the threshold: selection rate, TPR, PPV. Flag groups whose O/E or TPR differs from the overall
       value by more than the tolerances.

    Common mistakes
    ---------------
    - Checking only overall AUC (a group can be badly miscalibrated with the same AUC).
    - Dropping the protected attribute from the model and assuming that fixes bias (proxies remain).
    - Judging small subgroups without intervals.
    """
    rows = []
    overall_oe = df[y].sum() / df[p].sum()
    overall_tpr = ((df[p] >= threshold) & (df[y] == 1)).sum() / max(df[y].sum(), 1)
    for g, d in df.groupby(group):
        yy, pp = d[y].to_numpy(), d[p].clip(1e-6, 1 - 1e-6).to_numpy()
        lg = np.log(pp / (1 - pp))
        fit = sm.GLM(yy, sm.add_constant(lg), family=sm.families.Binomial()).fit()                     # step 2
        sel = pp >= threshold
        tpr = float((sel & (yy == 1)).sum() / max(yy.sum(), 1))                                          # step 3
        rows.append({"group": g, "n": len(d), "observed_rate": float(yy.mean()), "mean_pred": float(pp.mean()),
                     "oe_ratio": float(yy.sum() / pp.sum()), "cal_intercept": float(fit.params[0]), "cal_slope": float(fit.params[1]),
                     "auc": _auc(yy, pp), "selection_rate": float(sel.mean()), "tpr": tpr,
                     "ppv": float(yy[sel].mean()) if sel.any() else float("nan")})
    out = pd.DataFrame(rows)
    out["flag"] = ((out.oe_ratio - overall_oe).abs() > max_oe_gap) | ((out.tpr - overall_tpr).abs() > max_tpr_gap)
    return out


# ---------------------------------------------------------------------------
# Self-test: run `python fairness.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 30_000
    grp = rng.choice(["A", "B"], n, p=[0.75, 0.25])
    need = rng.normal(0, 1, n)
    true_p = 1 / (1 + np.exp(-(-2.0 + 1.0 * need)))
    y = (rng.random(n) < true_p).astype(int)
    # model learned from a biased label: under-predicts group B's risk (log-odds shifted down by 0.7)
    pred = 1 / (1 + np.exp(-(-2.0 + 1.0 * need - 0.7 * (grp == "B"))))
    r = subgroup_audit(pd.DataFrame({"g": grp, "outcome_flag": y, "pred_prob": pred}), "g").set_index("group")
    print(r.round(3))
    checks = {
        "group A calibrated (O/E ~ 1)": abs(r.loc["A", "oe_ratio"] - 1) < 0.08,
        "group B under-predicted (O/E > 1.3)": r.loc["B", "oe_ratio"] > 1.3,
        "B's calibration intercept ~ +0.7": abs(r.loc["B", "cal_intercept"] - 0.7) < 0.2,
        "similar AUC in both groups (ranking is fine within group)": abs(r.loc["A", "auc"] - r.loc["B", "auc"]) < 0.03,
        "B: equally sick members selected less often (lower TPR)": r.loc["B", "tpr"] < r.loc["A", "tpr"] - 0.1,
        "B flagged, A not": r.loc["B", "flag"] and not r.loc["A", "flag"],
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
