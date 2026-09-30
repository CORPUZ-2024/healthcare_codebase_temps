"""
Synthetic member-month cost panel for program evaluation, with a KNOWN program effect,
plus a loader for the public MEPS file used to check the cost distribution.

The generator bakes in the three things that make naive ROI claims wrong:
  * selection     - sicker members (more chronic conditions, higher risk) are referred more often
  * regression to the mean - referral is triggered by an acute cost spike in the 3 months before
                    the index month; spikes fade on their own, program or not
  * secular trend - costs rise ~0.4% per month for everyone
and the cost shape of real claims: many $0 months, a lognormal right tail.

Truth: the program reduces a participant's post-index expected monthly cost by ``effect_pct``.
``truth_att_pmpm`` (dollars saved per participant month, from the counterfactual cost the
generator also draws) is returned so tests can check whether a method recovers it.

Input from real claims instead: build member x month paid_amt with t05's prep (member_months_daily
+ collapse_versions_latest, copied - never imported) and add the program enrollment date.

Public test file for the cost distribution
------------------------------------------
AHRQ Medical Expenditure Panel Survey (MEPS) Household Component, Full-Year Consolidated file
(total expenditures TOTEXPyy, person weight PERWTyyF). https://meps.ahrq.gov  - see data/README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_panel(n_members: int = 5_000, months: int = 24, start: str = "2024-01-01",
                   effect_pct: float = 0.15, seed: int = 21) -> dict:
    """Member-month panel with program participation.

    Returns
    -------
    dict with
      members : member_id, age, sex_cd, lob_cd, chronic_cnt, risk_score, treated_flag,
                index_month_idx (program start; a pseudo-index drawn the same way for non-participants)
      panel   : member_id, month_idx, month_start, member_months (1.0), paid_amt,
                paid_cf_amt (cost without the program; equals paid_amt for non-participants),
                ip_admit_flag (an acute episode - the referral trigger - starts this month)
      truth_att_pmpm : true dollars saved per participant month in months +1..+6 (float)

    Steps
    -----
    1. Member attributes; risk_score driven by age and chronic conditions.
    2. Acute episodes: a member-month starts one (an inpatient admission, ip_admit_flag) with
       p = 3% x risk; the episode multiplies expected cost x4, x2.5, x1.5 over 3 months.
    3. Candidate index month 8..15 for everyone; referral probability rises with risk and with an
       admission in the 3 months before the candidate month (-> regression to the mean). The
       admission is OBSERVED, so matching on it removes that bias; cost alone is too noisy a proxy.
    4. Monthly cost = Bernoulli(any cost) x lognormal amount x trend x spike; participants' post-index
       months get (1 - effect_pct) on the expected amount. The counterfactual uses the same draws.
    5. Attrition: ~15% disenroll at a random month (rows after it are dropped).
    """
    rng = np.random.default_rng(seed)
    n, T = n_members, months
    age = rng.integers(19, 90, n)                                                   # step 1
    chronic = rng.poisson(0.5 + 0.03 * (age - 19), n)
    risk = np.exp(0.12 * chronic + 0.005 * (age - 50) + rng.normal(0, 0.9, n))
    lob = np.where(age >= 65, rng.choice(["DUAL", "MCR"], n, p=[0.6, 0.4]), "MCD")

    spike = np.ones((n, T))                                                          # step 2
    starts = rng.random((n, T)) < np.minimum(0.03 * risk[:, None], 0.25)
    for lag, mult in enumerate((4.0, 2.5, 1.5)):
        shifted = np.zeros_like(starts)
        shifted[:, lag:] = starts[:, : T - lag]
        spike = np.where(shifted, np.maximum(spike, mult), spike)

    index = rng.integers(8, 16, n)                                                   # step 3
    rows = np.arange(n)
    recent_spike = np.max(np.stack([starts[rows, index - k] for k in (1, 2, 3)]), axis=0)
    logit = -3.4 + 0.35 * chronic + 0.8 * np.log(risk) + 1.6 * recent_spike
    treated = rng.random(n) < 1 / (1 + np.exp(-logit))

    t = np.arange(T)[None, :]                                                        # step 4
    base = 180.0 * risk[:, None] * (1.004 ** t) * spike
    user = np.where(rng.random(n) < 0.15, 0.03, 1.0)                               # ~15% near non-users
    p_nom = (1 - np.exp(-0.5 * np.sqrt(risk)))[:, None] * np.ones((1, T))           # P(any cost) for users
    p_any = p_nom * user[:, None]
    any_cost = rng.random((n, T)) < np.minimum(p_any * np.where(spike > 1, 1.6, 1.0), 0.98)
    sigma = 1.0
    amount = np.exp(np.log(base / p_nom) - sigma ** 2 / 2 + sigma * rng.standard_normal((n, T)))
    cf = np.where(any_cost, amount, 0.0)
    post = (t > index[:, None]) & treated[:, None]
    paid = np.where(post, cf * (1 - effect_pct), cf)

    last = np.where(rng.random(n) < 0.15, rng.integers(10, T, n), T - 1)             # step 5
    keep = t <= last[:, None]

    ids = np.array([f"R{i:06d}" for i in range(1, n + 1)])
    ii, tt = np.nonzero(keep)
    month_start = pd.date_range(start, periods=T, freq="MS")
    panel = pd.DataFrame({"member_id": ids[ii], "month_idx": tt, "month_start": month_start[tt],
                          "member_months": 1.0, "paid_amt": paid[ii, tt].round(2), "paid_cf_amt": cf[ii, tt].round(2),
                          "ip_admit_flag": starts[ii, tt].astype(int)})
    members = pd.DataFrame({"member_id": ids, "age": age, "sex_cd": rng.choice(["F", "M"], n), "lob_cd": lob,
                            "chronic_cnt": chronic, "risk_score": risk.round(3), "treated_flag": treated.astype(int),
                            "index_month_idx": index})
    rel = panel["month_idx"] - panel["member_id"].map(members.set_index("member_id")["index_month_idx"])
    tp = panel[panel["member_id"].isin(ids[treated]) & rel.between(1, 6)]
    truth = float((tp["paid_cf_amt"] - tp["paid_amt"]).sum() / tp["member_months"].sum())
    return {"members": members, "panel": panel, "truth_att_pmpm": truth}


# ---------------------------------------------------------------------------
# Public test file: MEPS Full-Year Consolidated (cost distribution)
# ---------------------------------------------------------------------------

def load_meps_totexp(path: str | Path, year: int) -> pd.DataFrame:
    """Load total annual expenditure, weight, age and sex from a MEPS Full-Year Consolidated file.

    Reads Stata (.dta), Excel (.xlsx) or CSV by extension. Column names carry the 2-digit year:
    TOTEXP22, PERWT22F, AGE22X for 2022. SEX is 1 = male, 2 = female.

    Returns person_id (DUPERSID), totexp_amt, weight, age, sex_cd - persons with weight 0 dropped
    (they are out of scope for the year).
    """
    yy = f"{year % 100:02d}"
    cols = ["DUPERSID", f"TOTEXP{yy}", f"PERWT{yy}F", f"AGE{yy}X", "SEX"]
    p = Path(path)
    if p.suffix.lower() == ".dta":
        df = pd.read_stata(p, columns=cols, convert_categoricals=False)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, usecols=cols)
    else:
        df = pd.read_csv(p, usecols=cols, dtype={"DUPERSID": str})
    out = pd.DataFrame({"person_id": df["DUPERSID"].astype(str), "totexp_amt": pd.to_numeric(df[f"TOTEXP{yy}"]),
                        "weight": pd.to_numeric(df[f"PERWT{yy}F"]), "age": pd.to_numeric(df[f"AGE{yy}X"]),
                        "sex_cd": df["SEX"].map({1: "M", 2: "F"})})
    return out[out["weight"] > 0].reset_index(drop=True)


def cost_distribution_summary(cost: pd.Series, weight: pd.Series | None = None) -> dict:
    """Share of zeros, weighted mean, percentiles, top-1% share and lognormal fit of the positives.

    Use it to check that synthetic (or your plan's) costs look like MEPS: a large zero mass and
    a tail where the top 1% carry ~20% of spend. Both facts are why the two-part model and the
    bootstrap exist.
    """
    w = pd.Series(1.0, index=cost.index) if weight is None else weight
    x, w = cost.to_numpy(float), w.to_numpy(float)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) / ws.sum()

    def q(p):
        return float(xs[np.searchsorted(cw, p)])

    top = xs[cw > 0.99]
    pos = x > 0
    lx = np.log(x[pos])
    mu = float(np.average(lx, weights=w[pos]))
    return {"share_zero": float(w[~pos].sum() / w.sum()), "mean": float(np.average(x, weights=w)),
            "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "top1_share": float((top * ws[cw > 0.99]).sum() / (xs * ws).sum()),
            "lognormal_mu": mu, "lognormal_sigma": float(np.sqrt(np.average((lx - mu) ** 2, weights=w[pos])))}
