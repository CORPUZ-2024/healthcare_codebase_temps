"""
t12 — Patient- and caregiver-reported outcomes: YAML scoring, reliability, change, MCID responders, mixed models.
Synthetic caregiver-support trial (PHQ-9 is public domain; the burden scale is FAKE).

    python run.py              demo -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from patient_reported_outcomes import checks, data, methods  # noqa: E402
from patient_reported_outcomes.config import Config  # noqa: E402

OUT = HERE / "outputs"


def scored_frame(long: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return pd.concat([long[["caregiver_id", "arm_flag", "visit_month", "observed_flag"]], methods.score_instrument(long, spec)], axis=1)


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    ins = data.load_instruments(cfg.instruments_dir)
    burden, phq = ins[cfg.primary_instrument], ins[cfg.secondary_instrument]
    trial = data.generate_trial(burden, phq, cfg.n_caregivers, cfg.visits, cfg.true_effect_6m, cfg.item_missing_rate, cfg.seed)
    long = trial["long"]
    sb, sp = scored_frame(long, burden), scored_frame(long, phq)
    wb = methods.to_wide(sb)
    base_items = methods.reversed_items(long[long["visit_month"] == 0], burden)
    alpha = methods.cronbach_alpha(base_items)
    fc = methods.floor_ceiling(sb["score"], burden)

    print(f"Trial: {cfg.n_caregivers} caregivers, 1:1, visits {cfg.visits} months. Primary: {burden['title']} (0-48). "
          f"Secondary: {phq['title']}.")
    print(f"TRUE program effect on burden change at month 6 (complete data): {trial['true_effect_6m']:.2f} points\n")
    print("== Scoring ==")
    print(f"questionnaires returned {int(long['observed_flag'].sum()):,}; prorated (1 item missing) {int(sb['prorated_flag'].sum())}; "
          f"unscorable (>1 missing) {int((sb['score'].isna() & (sb['observed_flag'] == 1)).sum())}")
    print(f"Cronbach's alpha (baseline, reverse items flipped) {alpha:.2f}; without flipping "
          f"{methods.cronbach_alpha(long.loc[long['visit_month'] == 0, burden['items']]):.2f}  <- why reverse scoring matters")
    print(methods.severity_band(sb.loc[sb["visit_month"] == 0, "score"], burden).value_counts().sort_index().to_string())
    print("completion by visit x arm:", long.groupby(["visit_month", "arm_flag"])["observed_flag"].mean().round(2).to_dict())

    mm = methods.mixed_model_effect(sb)
    an = methods.ancova_change(wb)
    ge = methods.gee_effect(sb)
    print("\n== Program effect on burden change at month 6 (negative = less burden) ==")
    print(f"STANDARD mixed model (MMRM-style, all visits)  {mm['effect']:.2f} (95% CI {mm['ci_lo']:.2f} to {mm['ci_hi']:.2f}), n = {mm['n_people']}")
    print(f"ALTERNATIVE ANCOVA on change (completers)       {an['effect']:.2f} (95% CI {an['ci_lo']:.2f} to {an['ci_hi']:.2f}), n = {an['n_people']}")
    print(f"ALTERNATIVE GEE (exchangeable)                  {ge['effect']:.2f} (95% CI {ge['ci_lo']:.2f} to {ge['ci_hi']:.2f})")
    am = methods.arm_mean_change(sb, arm=1)
    comp_truth = methods.to_wide(scored_frame(trial["complete"].assign(observed_flag=1), burden))
    true_arm = float(comp_truth.loc[comp_truth["arm_flag"] == 1, "change_6"].mean())
    print(f"\nProgram arm 'mean improvement' at month 6: completers {am['completers_mean_change']:.2f} (n={am['completers_n']}), "
          f"mixed model {am['mixed_model_mean_change']:.2f} (n={am['enrolled_n']}), TRUE {true_arm:.2f}")

    resp = methods.responder_analysis(wb, burden["mcid"])
    resp_c = methods.responder_analysis(wb, burden["mcid"], missing_as_nonresponder=True)
    print(f"\n== Responders (improved >= MCID {burden['mcid']} points) ==")
    print(resp.round(3).to_string(index=False))
    print(f"difference {resp.attrs['difference']:+.1%} observed-only; {resp_c.attrs['difference']:+.1%} counting dropouts as non-responders")
    phq_mm = methods.mixed_model_effect(sp)
    print(f"\nSecondary PHQ-9 effect at month 6: {phq_mm['effect']:.2f} (95% CI {phq_mm['ci_lo']:.2f} to {phq_mm['ci_hi']:.2f}); "
          f"MCID {phq['mcid']} points")

    print("\n== Checks ==")
    fnd = ([f for s in ins.values() for f in checks.check_license(s)] + checks.check_unscorable(sb, long["observed_flag"])
           + checks.check_differential_dropout(wb) + checks.check_floor_ceiling(fc) + checks.check_reliability(alpha))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")

    sb.to_csv(OUT / "burden_scores_long.csv", index=False)
    wb.to_csv(OUT / "burden_scores_wide.csv", index=False)
    pd.DataFrame([{"method": "mixed_model", **mm}, {"method": "ancova_completers", **an}, {"method": "gee", **ge},
                  {"method": "TRUTH", "effect": trial["true_effect_6m"]}]).to_csv(OUT / "effect_estimates.csv", index=False)
    resp.to_csv(OUT / "responders.csv", index=False)
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    ins = data.load_instruments(Config().instruments_dir)
    phq, cb = ins["PHQ9"], ins["FAKE_CAREGIVER_BURDEN"]
    one = pd.DataFrame([{**{i: 2 for i in phq["items"]}}])
    miss1 = one.copy()
    miss1.loc[0, "phq9"] = None
    miss2 = miss1.copy()
    miss2.loc[0, "phq8"] = None
    rev = pd.DataFrame([{**{i: 0 for i in cb["items"]}}])
    tests = {
        "PHQ-9 all 2s = 18": lambda: methods.score_instrument(one, phq)["score"].iloc[0] == 18,
        "PHQ-9 one item missing -> prorated 18": lambda: methods.score_instrument(miss1, phq)["score"].iloc[0] == 18,
        "PHQ-9 two items missing -> unscorable": lambda: pd.isna(methods.score_instrument(miss2, phq)["score"].iloc[0]),
        "burden all 0 answered: 3 reverse items count 4 each = 12": lambda: methods.score_instrument(rev, cb)["score"].iloc[0] == 12,
        "PHQ-9 18 = 'moderately severe'": lambda: str(methods.severity_band(pd.Series([18.0]), phq).iloc[0]) == "moderately severe",
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
