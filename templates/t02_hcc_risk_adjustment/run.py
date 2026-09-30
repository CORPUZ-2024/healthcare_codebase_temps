"""
t02 — HCC-style risk adjustment (ALL coefficients are FAKE; see reference/README.md).

    python run.py              demo on synthetic data -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hcc_risk_adjustment import checks, data, methods  # noqa: E402
from hcc_risk_adjustment.config import Config  # noqa: E402

OUT = HERE / "outputs"


def score_all(cfg: Config, d: dict, ref: dict):
    acc = methods.filter_acceptable_sources(d["dx"])
    feats, scores = {}, {}
    for v in ("V24", "V28"):
        hcc = methods.apply_hierarchies(methods.map_dx_to_hcc(acc, ref["dx_map"], v), ref["hierarchy"], v)
        feats[v] = methods.build_features(d["members"], hcc, ref["coef"], v)
        scores[v] = methods.score_published_weights(feats[v], ref["coef"], ref["demo"], v)
    return acc, feats, scores


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    print("NOTE: all HCC mappings and coefficients are FAKE teaching values.\n")
    d = data.generate(cfg.n_members, cfg.service_year, cfg.seed, cfg.documentation_rate)
    ref = data.load_reference()
    acc, feats, scores = score_all(cfg, d, ref)
    cost = d["cost"].set_index("member_id")["paid_amt"]

    raw = scores["V28"] if cfg.payment_year >= 2026 else methods.blend_versions(scores, cfg.payment_year)
    pay = methods.normalize_and_adjust(raw, cfg.normalization_factor, cfg.coding_intensity)
    blend = methods.blend_versions(scores, cfg.blend_year)
    print(f"== Scores == mean raw V24 {scores['V24'].mean():.3f} | V28 {scores['V28'].mean():.3f} | "
          f"PY{cfg.blend_year} blend {blend.mean():.3f} | PY{cfg.payment_year} payment score {pay.mean():.3f}")

    # STANDARD vs ALTERNATIVE, validated on a held-out half
    rng = np.random.default_rng(cfg.seed)
    ids = feats["V28"].index.to_numpy()
    train = rng.random(len(ids)) < cfg.train_frac
    emp_train, w = methods.score_empirical_weights(feats["V28"].loc[ids[train]], cost, cfg.ridge)
    Xtest = feats["V28"].loc[ids[~train]].astype(float)
    emp_test = pd.Series(Xtest.to_numpy() @ w.to_numpy(), index=Xtest.index)
    pr_std = methods.predictive_ratios(scores["V28"].loc[ids[~train]], cost)
    pr_emp = methods.predictive_ratios(emp_test, cost)
    print(f"\n== Held-out validation (n={len(Xtest):,}) ==\nSTANDARD published weights R^2 = {pr_std.attrs['r2']:.3f} | "
          f"ALTERNATIVE empirical weights R^2 = {pr_emp.attrs['r2']:.3f}")
    print(pd.DataFrame({"decile": pr_std["decile"], "PR_standard": pr_std["predictive_ratio"].round(2),
                        "PR_empirical": pr_emp["predictive_ratio"].round(2)}).to_string(index=False))

    all_hcc = methods.map_dx_to_hcc(d["dx"], ref["dx_map"], "V28")
    acc_hcc = methods.map_dx_to_hcc(acc, ref["dx_map"], "V28")
    gaps = methods.suspect_gaps(all_hcc, acc_hcc, d["rx"])
    print(f"\n== Suspect list for clinical review (NOT for coding): {len(gaps)} member-conditions ==")
    print(gaps["evidence"].str.split(":").str[0].value_counts().to_string())

    top = pay.idxmax()
    print(f"\n== Audit trail for highest-scoring member {top} (V28 raw) ==")
    print(methods.explain_member(top, feats["V28"], ref["coef"], ref["demo"], "V28").to_string(index=False))

    print("\n== Checks ==")
    for f in (checks.check_model_year(d["dx"], cfg.payment_year) + checks.check_unacceptable_sources(d["dx"], methods.ACCEPTABLE_SOURCES)
              + checks.check_unmapped_share(acc, acc_hcc) + checks.check_score_range(pay)):
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    pd.DataFrame({"raw_v24": scores["V24"], "raw_v28": scores["V28"], f"blend_py{cfg.blend_year}": blend,
                  f"payment_score_py{cfg.payment_year}": pay}).to_csv(OUT / "member_scores.csv")
    gaps.to_csv(OUT / "suspect_gaps_for_clinical_review.csv", index=False)
    pr_std.to_csv(OUT / "predictive_ratios_standard.csv", index=False)
    w.rename("relative_cost_weight").to_csv(OUT / "empirical_weights.csv")
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    ref = data.load_reference()
    members = pd.DataFrame({"member_id": ["A"], "age": [70], "sex_cd": ["F"], "dual_flag": [1], "disabled_flag": [0]})
    dx = pd.DataFrame({"member_id": ["A"] * 4, "dx_cd": ["E11.22", "E119", "N185", "N1830"],
                       "svc_dt": pd.to_datetime(["2025-02-01"] * 4), "source_cd": ["PROF_F2F", "OP", "IP", "LAB"]})
    acc = methods.filter_acceptable_sources(dx)
    hcc = methods.apply_hierarchies(methods.map_dx_to_hcc(acc, ref["dx_map"], "V24"), ref["hierarchy"], "V24")
    X = methods.build_features(members, hcc, ref["coef"], "V24")
    s = methods.score_published_weights(X, ref["coef"], ref["demo"], "V24")
    tests = {
        "LAB diagnosis excluded": lambda: len(acc) == 3,
        "hierarchy keeps DIAB_CC and CKD5 only": lambda: set(hcc["hcc"]) == {"DIAB_CC", "CKD5"},
        "score = F65_74 0.310 + DUAL 0.180 + DIAB_CC 0.302 + CKD5 0.815": lambda: abs(s.iloc[0] - 1.607) < 1e-9,
        "PY2025 blend = 0.33 V24 + 0.67 V28": lambda: abs(methods.blend_versions({"V24": pd.Series([1.0]), "V28": pd.Series([2.0])}, 2025).iloc[0] - 1.67) < 1e-9,
        "normalize: 1.05 / 1.05 x (1-0.059) = 0.941": lambda: abs(methods.normalize_and_adjust(pd.Series([1.05]), 1.05).iloc[0] - 0.941) < 1e-9,
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else (demo(Config()) or 0))
