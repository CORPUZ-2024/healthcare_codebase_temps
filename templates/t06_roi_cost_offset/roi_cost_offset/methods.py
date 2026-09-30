"""
Program ROI / cost offset: matched pre/post, difference-in-differences on PMPM, ROI, sensitivity.

    match_propensity + did_pmpm (ratio of sums, cluster-robust SE)  (STANDARD)
        vs  bootstrap_did (pair bootstrap CI)  and  two_part_did (logit x Gamma GLM)  (ALTERNATIVE)
    + member_periods, balance_table, naive_pre_post (what NOT to report), roi, tornado

Reading order: member_periods -> propensity_scores -> match_propensity -> balance_table ->
did_pmpm -> roi -> tornado. Each function says which bias it removes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

COVARIATES = ["age", "female_flag", "chronic_cnt", "log_risk", "log_pre_pmpm", "log_pre_last3_pmpm",
              "pre_last3_admit_flag"]

# ---------------------------------------------------------------------------
# 1. Align every member on an index month and summarise pre / post
# ---------------------------------------------------------------------------

def member_periods(panel: pd.DataFrame, members: pd.DataFrame, pre_months: int = 6, post_months: int = 6,
                   min_months: int = 4) -> pd.DataFrame:
    """One row per member: pre- and post-period paid, exposure and PMPM around the index month.

    Healthcare context
    ------------------
    Participants start a program on different dates, so time is re-expressed relative to each
    member's index month (rel_month = month - index). Non-participants get a pseudo-index drawn
    the same way, so both groups are observed over the same relative window. The index month
    itself is excluded: it usually contains the event that triggered the referral.

    Parameters
    ----------
    panel : member_id, month_idx, member_months, paid_amt, ip_admit_flag
    members : member_id, index_month_idx, treated_flag, age, sex_cd, lob_cd, chronic_cnt, risk_score

    Returns
    -------
    member_id, treated_flag, lob_cd, covariates (see COVARIATES), pre_paid_amt, pre_mm, pre_pmpm,
    pre_last3_paid_amt, pre_last3_mm, pre_last3_pmpm, pre_last3_admit_flag, post_paid_amt, post_mm, post_pmpm - only members with >= ``min_months`` of
    exposure in BOTH periods.

    Steps
    -----
    1. rel_month = month_idx - index_month_idx.
    2. pre = rel -pre_months..-1, post = rel 1..post_months; last-3-months PMPM kept separately
       and whether an admission happened there (the usual referral trigger).
    3. Sum paid and member-months per member and period; PMPM = paid / member-months.
    4. Keep members with enough exposure in both periods; build model covariates.

    Common mistakes
    ---------------
    - Calendar pre/post (Jan-Jun vs Jul-Dec) when enrollment is rolling.
    - Keeping members who leave right after enrolling: their post PMPM rests on 1 month.
    - Filling missing months with $0 (disenrolled is not "cost nothing").
    """
    p = panel.merge(members[["member_id", "index_month_idx"]], on="member_id")
    p["rel"] = p["month_idx"] - p["index_month_idx"]                                       # step 1
    p["period"] = np.select([p["rel"].between(-pre_months, -1), p["rel"].between(1, post_months)],
                            ["pre", "post"], default="")                                     # step 2
    p = p[p["period"] != ""]
    agg = p.groupby(["member_id", "period"])[["paid_amt", "member_months"]].sum().unstack("period")   # step 3
    out = pd.DataFrame({"pre_paid_amt": agg[("paid_amt", "pre")], "pre_mm": agg[("member_months", "pre")],
                        "post_paid_amt": agg[("paid_amt", "post")], "post_mm": agg[("member_months", "post")]})
    last3 = p[p["rel"].between(-3, -1)].groupby("member_id")[["paid_amt", "member_months"]].sum()
    out["pre_last3_paid_amt"], out["pre_last3_mm"] = last3["paid_amt"], last3["member_months"]
    out["pre_last3_pmpm"] = last3["paid_amt"] / last3["member_months"]
    if "ip_admit_flag" in p:
        adm = p[p["rel"].between(-3, -1)].groupby("member_id")["ip_admit_flag"].max()
        out["pre_last3_admit_flag"] = adm.reindex(out.index).fillna(0).astype(int)
    out = out[(out["pre_mm"] >= min_months) & (out["post_mm"] >= min_months)].copy()          # step 4
    out["pre_pmpm"] = out["pre_paid_amt"] / out["pre_mm"]
    out["post_pmpm"] = out["post_paid_amt"] / out["post_mm"]
    out = out.reset_index().merge(members, on="member_id")
    out["female_flag"] = (out["sex_cd"] == "F").astype(int)
    out["log_risk"] = np.log(out["risk_score"])
    out["log_pre_pmpm"] = np.log1p(out["pre_pmpm"])
    out["log_pre_last3_pmpm"] = np.log1p(out["pre_last3_pmpm"].fillna(0))
    return out


# ---------------------------------------------------------------------------
# 2. Matching
# ---------------------------------------------------------------------------

def propensity_scores(mp: pd.DataFrame, covariates: list[str] = COVARIATES) -> pd.Series:
    """Logistic propensity score P(treated | covariates); returns the LOGIT of the score.

    Include pre-period cost (overall AND the last 3 months) and the referral trigger itself
    (an admission in the last 3 months): a control group without the same trigger won't regress
    to the mean the way participants do. Better still, match EXACTLY on the trigger.
    """
    X = sm.add_constant(mp[covariates].astype(float))
    fit = sm.Logit(mp["treated_flag"].astype(float), X).fit(disp=0)
    return pd.Series(X.to_numpy() @ fit.params.to_numpy(), index=mp.index, name="ps_logit")


def match_propensity(mp: pd.DataFrame, ps_logit: pd.Series, caliper_sd: float = 0.2,
                     exact_on: tuple = ("lob_cd",), seed: int = 0) -> pd.DataFrame:
    """STANDARD: 1:1 greedy nearest-neighbour match on the logit propensity score, without
    replacement, inside a caliper of ``caliper_sd`` x SD(logit), exact on ``exact_on`` columns.

    Healthcare context
    ------------------
    Program participants differ from everyone else (sicker, recently hospitalized). Comparing
    them to "all non-participants" credits the program with differences that were there before.
    Matching builds a comparison group that looked the same BEFORE the program.

    Returns
    -------
    The matched rows of ``mp`` plus pair_id and ps_logit. Treated members with no control inside
    the caliper are dropped (report how many: see checks.check_common_support).

    Steps
    -----
    1. Caliper = caliper_sd x SD of the logit score (Austin 2011 recommends 0.2).
    2. Visit treated members in random order (seeded); within the same exact-match cell pick the
       closest unused control; accept if |difference| <= caliper.

    Common mistakes
    ---------------
    - Matching on the probability instead of its logit (the caliper rule is on the logit).
    - Matching WITH replacement and then computing plain SEs (reused controls are correlated).
    - Forgetting to check balance afterwards - a matched sample is not automatically balanced.
    """
    df = mp.assign(ps_logit=ps_logit)
    caliper = caliper_sd * float(df["ps_logit"].std())                                      # step 1
    rng = np.random.default_rng(seed)
    keys = list(exact_on)
    pairs, pair_id = [], 0
    for _, cell in (df.groupby(keys) if keys else [(None, df)]):
        t = cell[cell["treated_flag"] == 1]
        c = cell[cell["treated_flag"] == 0]
        c_ps, c_idx = c["ps_logit"].to_numpy(), c.index.to_numpy()
        used = np.zeros(len(c), dtype=bool)
        for ti in rng.permutation(t.index.to_numpy()):                                      # step 2
            if used.all():
                break
            d = np.abs(c_ps - df.at[ti, "ps_logit"])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= caliper:
                used[j] = True
                pairs += [(ti, pair_id), (c_idx[j], pair_id)]
                pair_id += 1
    idx, pid = zip(*pairs) if pairs else ((), ())
    out = df.loc[list(idx)].copy()
    out["pair_id"] = list(pid)
    return out.reset_index(drop=True)


def balance_table(before: pd.DataFrame, after: pd.DataFrame, covariates: list[str] = COVARIATES) -> pd.DataFrame:
    """Standardized mean differences (SMD) before and after matching; |SMD| < 0.1 = balanced.

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2), using the
    BEFORE-matching variances in both columns so the denominator doesn't shrink with the sample.
    """
    rows = []
    for c in covariates:
        t0, c0 = before.loc[before["treated_flag"] == 1, c], before.loc[before["treated_flag"] == 0, c]
        t1, c1 = after.loc[after["treated_flag"] == 1, c], after.loc[after["treated_flag"] == 0, c]
        sd = np.sqrt((t0.var() + c0.var()) / 2) or 1.0
        rows.append({"covariate": c, "mean_treated": float(t1.mean()), "mean_control": float(c1.mean()),
                     "smd_before": float((t0.mean() - c0.mean()) / sd), "smd_after": float((t1.mean() - c1.mean()) / sd)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Effect estimates
# ---------------------------------------------------------------------------

def naive_pre_post(mp: pd.DataFrame) -> float:
    """Participants' post PMPM minus pre PMPM. Shown ONLY to demonstrate the bias.

    It attributes regression to the mean (the referral spike fading) and secular trend to the
    program. Vendors' "savings" slides are often this number.
    """
    t = mp[mp["treated_flag"] == 1]
    return float(t["post_paid_amt"].sum() / t["post_mm"].sum() - t["pre_paid_amt"].sum() / t["pre_mm"].sum())


def _long(mp: pd.DataFrame) -> pd.DataFrame:
    pre = mp.assign(post=0, pmpm=mp["pre_pmpm"], mm=mp["pre_mm"], paid=mp["pre_paid_amt"])
    post = mp.assign(post=1, pmpm=mp["post_pmpm"], mm=mp["post_mm"], paid=mp["post_paid_amt"])
    out = pd.concat([pre, post], ignore_index=True)
    out["tp"] = out["treated_flag"] * out["post"]
    return out


def did_pmpm(matched: pd.DataFrame, z: float = 1.96) -> dict:
    """STANDARD: difference-in-differences on PMPM in the matched sample.

    Healthcare context
    ------------------
    DiD = (post - pre) for participants MINUS (post - pre) for matched controls. The control
    change absorbs trend, seasonality and regression to the mean that both groups share.
    Negative = savings. Estimated as a weighted regression pmpm ~ treated + post + treated:post
    (weights = member-months, so the point estimate equals the ratio-of-sums DiD Finance can
    rebuild) with standard errors clustered by matched pair.

    Returns dict: did_pmpm, se, ci_lo, ci_hi, p_value, the four cell PMPMs, n_pairs.

    Common mistakes
    ---------------
    - Unweighted means of member PMPMs (a 1-month member counts as much as a 6-month one).
    - Ignoring pairing/clustering: two rows per member are not independent observations.
    - Reading DiD as causal without checking pre-period trends (checks.check_pre_trends).
    """
    lg = _long(matched)
    fit = smf.wls("pmpm ~ treated_flag + post + tp", data=lg, weights=lg["mm"]).fit(
        cov_type="cluster", cov_kwds={"groups": lg["pair_id"]})
    b, se = float(fit.params["tp"]), float(fit.bse["tp"])
    cells = lg.groupby(["treated_flag", "post"]).apply(lambda g: g["paid"].sum() / g["mm"].sum(), include_groups=False)
    return {"did_pmpm": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "p_value": float(fit.pvalues["tp"]),
            "treated_pre": float(cells[(1, 0)]), "treated_post": float(cells[(1, 1)]),
            "control_pre": float(cells[(0, 0)]), "control_post": float(cells[(0, 1)]),
            "n_pairs": int(matched["pair_id"].nunique())}


def bootstrap_did(matched: pd.DataFrame, n_boot: int = 1_000, seed: int = 0, alpha: float = 0.05,
                  breakeven_savings_pmpm: float | None = None) -> dict:
    """ALTERNATIVE: percentile bootstrap CI for the matched DiD, resampling PAIRS.

    Trade-off
    ---------
    + No normality assumption: cost DiDs are driven by a few catastrophic members, and the
      sandwich SE can be optimistic with a skewed, small sample. The bootstrap shows asymmetry.
    + Resampling pairs keeps the matched structure (and both periods of each member) together.
    - Slower; doesn't account for uncertainty in the propensity model unless you re-match
      inside each replicate (not done here - usually a small effect for 1:1 matching).

    + Gives decision probabilities directly: P(any savings) and P(savings >= break-even).

    Returns dict: did_pmpm (point), ci_lo, ci_hi, se_boot, n_boot, prob_savings (share of
    replicates with DiD < 0), prob_breakeven (share with savings >= ``breakeven_savings_pmpm``, if given).
    """
    s = matched.groupby(["pair_id", "treated_flag"])[["pre_paid_amt", "pre_mm", "post_paid_amt", "post_mm"]].sum()
    s = s.unstack("treated_flag")
    arr = {f"{k}_{t}": s[(k, t)].to_numpy() for k in ("pre_paid_amt", "pre_mm", "post_paid_amt", "post_mm") for t in (0, 1)}

    def did(ix):
        f = {k: v[ix].sum(axis=-1) for k, v in arr.items()}
        return ((f["post_paid_amt_1"] / f["post_mm_1"] - f["pre_paid_amt_1"] / f["pre_mm_1"])
                - (f["post_paid_amt_0"] / f["post_mm_0"] - f["pre_paid_amt_0"] / f["pre_mm_0"]))

    n = len(s)
    point = float(did(np.arange(n)))
    reps = did(np.random.default_rng(seed).integers(0, n, size=(n_boot, n)))
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return {"did_pmpm": point, "ci_lo": float(lo), "ci_hi": float(hi), "se_boot": float(reps.std(ddof=1)),
            "n_boot": n_boot, "prob_savings": float((reps < 0).mean()),
            "prob_breakeven": float((-reps >= breakeven_savings_pmpm).mean()) if breakeven_savings_pmpm is not None else float("nan")}


def two_part_did(matched: pd.DataFrame, covariates: tuple = ("age", "chronic_cnt", "log_risk")) -> dict:
    """ALTERNATIVE: two-part model DiD - P(any cost) by logit x E[PMPM | cost > 0] by Gamma GLM (log link).

    Trade-off
    ---------
    + Respects the cost shape: a mass at $0 and a long right tail (see MEPS). The Gamma log-link
      model treats effects as proportional, which is how programs usually act on cost.
    + Separates "fewer people with any cost" from "lower cost among users" (two effects reported).
    - The DiD in dollars comes from recycled predictions (model-based), not a simple difference;
      misspecification of either part biases it. Harder to explain to Finance.

    Steps
    -----
    1. Long data (member x period). Part 1: logit any_cost ~ treated + post + tp + covariates.
    2. Part 2: Gamma(log) GLM pmpm ~ same, on rows with cost > 0, weights = member-months.
    3. For participants' post rows predict E[pmpm] = P(any) x E[pmpm | any] with tp = 1 (observed)
       and tp = 0 (counterfactual); the member-month-weighted mean difference is the effect.

    Returns dict: did_pmpm, part1_odds_ratio, part2_cost_ratio.
    """
    lg = _long(matched)
    lg["any_cost"] = (lg["paid"] > 0).astype(float)
    rhs = "treated_flag + post + tp + " + " + ".join(covariates)
    p1 = smf.glm(f"any_cost ~ {rhs}", data=lg, family=sm.families.Binomial()).fit()                  # step 1
    pos = lg[lg["paid"] > 0]
    p2 = smf.glm(f"pmpm ~ {rhs}", data=pos, family=sm.families.Gamma(sm.families.links.Log()),
                 var_weights=pos["mm"]).fit()                                                            # step 2
    tp_rows = lg[(lg["treated_flag"] == 1) & (lg["post"] == 1)]                                          # step 3
    cf_rows = tp_rows.assign(tp=0)
    e1 = p1.predict(tp_rows) * p2.predict(tp_rows)
    e0 = p1.predict(cf_rows) * p2.predict(cf_rows)
    w = tp_rows["mm"]
    return {"did_pmpm": float(np.average(e1 - e0, weights=w)), "part1_odds_ratio": float(np.exp(p1.params["tp"])),
            "part2_cost_ratio": float(np.exp(p2.params["tp"]))}


def pre_trend_test(panel: pd.DataFrame, matched: pd.DataFrame, pre_months: int = 6) -> dict:
    """Parallel-trends diagnostic: do participants' and controls' monthly PMPM slopes differ BEFORE the index?

    Fits pmpm ~ rel + treated + rel:treated on matched members' pre-period months (-pre_months..-1),
    weighted by member-months, SEs clustered by pair. A significant rel:treated slope means the
    groups were already diverging and DiD will attribute that divergence to the program.

    Caveat: the months right before the index hold the referral trigger by design (both groups,
    because matching is exact on it); a spike shared by both groups does not bias the slope test.

    Returns dict: slope_diff_pmpm_per_month, se, p_value.
    """
    p = panel.merge(matched[["member_id", "pair_id", "treated_flag", "index_month_idx"]], on="member_id")
    p["rel"] = p["month_idx"] - p["index_month_idx"]
    p = p[p["rel"].between(-pre_months, -1)].assign(pmpm=lambda d: d["paid_amt"] / d["member_months"])
    fit = smf.wls("pmpm ~ rel * treated_flag", data=p, weights=p["member_months"]).fit(
        cov_type="cluster", cov_kwds={"groups": p["pair_id"]})
    k = "rel:treated_flag"
    return {"slope_diff_pmpm_per_month": float(fit.params[k]), "se": float(fit.bse[k]), "p_value": float(fit.pvalues[k])}


# ---------------------------------------------------------------------------
# 4. ROI and sensitivity
# ---------------------------------------------------------------------------

def roi(savings_pmpm: float, program_fee_pmpm: float, one_time_cost: float, participants: int,
        months_in_program: int) -> dict:
    """Gross savings, program cost, net savings, ROI and break-even savings PMPM.

    Healthcare context
    ------------------
    savings_pmpm is POSITIVE dollars saved per participant month (= -DiD). ROI = net / cost:
    0.5 means every $1 spent returns $1.50 gross ($0.50 net). Break-even savings PMPM is what the
    program must save to pay for itself: fee + one-time cost spread over the months in program.

    >>> r = roi(60.0, 25.0, 150.0, 100, 12)
    >>> r["gross_savings"], r["program_cost"], r["net_savings"], round(r["roi"], 3), r["breakeven_savings_pmpm"]
    (72000.0, 45000.0, 27000.0, 0.6, 37.5)

    Common mistakes
    ---------------
    - Quoting gross savings / cost as "ROI" (that's the savings-to-cost ratio, ROI + 1).
    - Applying a 6-month DiD to 12 months of program without saying so (effects can fade).
    - Using participants who ENROLLED rather than who ENGAGED (fees are often per engaged member).
    """
    member_months = participants * months_in_program
    gross = savings_pmpm * member_months
    cost = program_fee_pmpm * member_months + one_time_cost * participants
    net = gross - cost
    return {"gross_savings": float(gross), "program_cost": float(cost), "net_savings": float(net),
            "roi": float(net / cost) if cost else float("nan"),
            "savings_to_cost": float(gross / cost) if cost else float("nan"),
            "breakeven_savings_pmpm": float(program_fee_pmpm + one_time_cost / months_in_program)}


def tornado(base: dict, ranges: dict, metric: str = "net_savings") -> pd.DataFrame:
    """One-way sensitivity: vary each ROI input to its low/high value, others at base.

    ``base`` holds the keyword arguments of ``roi``; ``ranges`` maps some of them to (low, high).
    Returns input, low_value, high_value, metric_at_low, metric_at_high, swing - sorted by swing
    (the biggest bar goes on top of the tornado chart).
    """
    base_val = roi(**base)[metric]
    rows = []
    for k, (lo, hi) in ranges.items():
        at_lo, at_hi = roi(**{**base, k: lo})[metric], roi(**{**base, k: hi})[metric]
        rows.append({"input": k, "low_value": lo, "high_value": hi, "metric_at_low": at_lo, "metric_at_high": at_hi,
                     "swing": abs(at_hi - at_lo)})
    out = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    out.attrs["base"] = base_val
    return out
