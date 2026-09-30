"""
Time-to-event analysis: how long until readmission / placement / death, and what changes it.

    km_table + logrank + cox_ph (lifelines) + ph_test (Schoenfeld)       (STANDARD)
        vs  cox_statsmodels (PHReg)  and  discrete_time_hazard (time-varying effects)  (ALTERNATIVE)
    + naive_event_share (what NOT to report), rmst_difference, cox_time_varying (immortal time fix)

Censoring = we stopped watching before the event (disenrolled, study ended, died). Survival methods
use each member's follow-up up to that point; dropping or counting censored members as "no event" biases rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time

COVARIATES = ["program_flag", "age", "frailty_z", "caregiver_flag", "high_acuity_flag"]

# ---------------------------------------------------------------------------
# 1. Kaplan-Meier and log-rank (STANDARD descriptive)
# ---------------------------------------------------------------------------

def km_table(df: pd.DataFrame, group: str | None = "program_flag", days=(30, 90, 180, 365),
             duration: str = "duration_days", event: str = "event_flag") -> pd.DataFrame:
    """Cumulative incidence (1 - KM survival) with 95% CI at chosen days, by group.

    Healthcare context
    ------------------
    "What share of members are readmitted within 90 days?" - with members followed for different
    lengths of time. KM uses everyone while they are observed; the naive share (events / members)
    counts a member censored on day 20 as "not readmitted by day 90".

    Returns group, day, at_risk, cum_incidence, ci_lo, ci_hi, median_days.
    """
    rows = []
    groups = [(None, df)] if group is None else list(df.groupby(group))
    for g, d in groups:
        km = KaplanMeierFitter().fit(d[duration], d[event])
        ci = km.confidence_interval_survival_function_
        for t in days:
            s = float(km.survival_function_at_times(t).iloc[0])
            lo_s = float(np.interp(t, ci.index, ci.iloc[:, 0]))
            hi_s = float(np.interp(t, ci.index, ci.iloc[:, 1]))
            rows.append({"group": g, "day": t, "at_risk": int((d[duration] >= t).sum()), "cum_incidence": 1 - s,
                         "ci_lo": 1 - hi_s, "ci_hi": 1 - lo_s, "median_days": float(km.median_survival_time_)})
    return pd.DataFrame(rows)


def naive_event_share(df: pd.DataFrame, day: int, duration: str = "duration_days", event: str = "event_flag") -> float:
    """Events by ``day`` / all members - ignores censoring (biased LOW). Shown only to compare with KM."""
    return float(((df[duration] <= day) & (df[event] == 1)).mean())


def logrank(df: pd.DataFrame, group: str = "program_flag", duration: str = "duration_days", event: str = "event_flag") -> dict:
    """Log-rank test that the survival curves of two groups are equal (no covariate adjustment)."""
    a, b = df[df[group] == 1], df[df[group] == 0]
    r = logrank_test(a[duration], b[duration], a[event], b[event])
    return {"test_statistic": float(r.test_statistic), "p_value": float(r.p_value)}


# ---------------------------------------------------------------------------
# 2. Cox proportional hazards (STANDARD model)
# ---------------------------------------------------------------------------

def cox_ph(df: pd.DataFrame, covariates: list[str] = COVARIATES, duration: str = "duration_days",
           event: str = "event_flag", strata: list[str] | None = None) -> tuple[CoxPHFitter, pd.DataFrame]:
    """lifelines CoxPHFitter (Efron ties). Returns (fitted model, HR table: covariate, hr, ci_lo, ci_hi, p_value).

    Healthcare context
    ------------------
    Hazard ratio = relative rate of the event at any moment, adjusted for the other covariates.
    HR 0.75 for the program = 25% lower readmission rate at every point in follow-up - IF hazards
    are proportional (check with ph_test). Age is per year here; scale it for reporting.

    Common mistakes
    ---------------
    - Reading HR as a risk ratio ("25% fewer readmissions") - it's a rate ratio over time.
    - Ignoring non-proportional hazards: a covariate whose effect fades gets an averaged HR that
      depends on follow-up length.
    - Defining exposure using information from after time zero (immortal time; see cox_time_varying).
    """
    cph = CoxPHFitter()
    cph.fit(df[covariates + [duration, event] + (strata or [])], duration, event, strata=strata)
    s = cph.summary
    table = pd.DataFrame({"covariate": s.index, "hr": s["exp(coef)"].to_numpy(),
                          "ci_lo": s["exp(coef) lower 95%"].to_numpy(), "ci_hi": s["exp(coef) upper 95%"].to_numpy(),
                          "p_value": s["p"].to_numpy()})
    return cph, table


def ph_test(cph: CoxPHFitter, df: pd.DataFrame, covariates: list[str] = COVARIATES, duration: str = "duration_days",
            event: str = "event_flag") -> pd.DataFrame:
    """Schoenfeld-residual test of proportional hazards per covariate (rank-transformed time).

    A small p-value = the covariate's HR changes over follow-up. Fixes: stratify on it, add an
    interaction with time, or model the periods separately (discrete_time_hazard).
    """
    r = proportional_hazard_test(cph, df[covariates + [duration, event]], time_transform="rank")
    return pd.DataFrame({"covariate": [i if isinstance(i, str) else i[0] for i in r.summary.index],
                         "test_statistic": r.summary["test_statistic"].to_numpy(), "p_value": r.summary["p"].to_numpy()})


def rmst_difference(df: pd.DataFrame, horizon: int = 180, group: str = "program_flag", duration: str = "duration_days",
                    event: str = "event_flag") -> dict:
    """Restricted mean event-free days up to ``horizon``, by group, and the difference.

    Valid without proportional hazards and easy to explain: "program members spent X more days out
    of the hospital in the first 6 months".
    """
    out = {}
    for g in (1, 0):
        d = df[df[group] == g]
        out[g] = float(restricted_mean_survival_time(KaplanMeierFitter().fit(d[duration], d[event]), t=horizon))
    return {"rmst_treated": out[1], "rmst_control": out[0], "difference_days": out[1] - out[0], "horizon": horizon}


# ---------------------------------------------------------------------------
# 3. ALTERNATIVES: statsmodels PHReg, discrete-time hazard
# ---------------------------------------------------------------------------

def cox_statsmodels(df: pd.DataFrame, covariates: list[str] = COVARIATES, duration: str = "duration_days",
                    event: str = "event_flag") -> pd.DataFrame:
    """ALTERNATIVE: statsmodels PHReg (Efron ties) - same model, no lifelines dependency.

    Trade-off
    ---------
    + Available wherever statsmodels is (lifelines can be blocked in locked-down environments).
    + Supports strata and formula syntax; results match lifelines to ~1e-4.
    - No built-in Schoenfeld test or plotting helpers; fewer survival utilities (KM CIs, RMST).
    """
    fit = smf.phreg(f"{duration} ~ " + " + ".join(covariates), data=df, status=df[event].to_numpy(), ties="efron").fit()
    ci = np.exp(fit.conf_int())
    return pd.DataFrame({"covariate": covariates, "hr": np.exp(fit.params), "ci_lo": ci[:, 0], "ci_hi": ci[:, 1],
                         "p_value": fit.pvalues})


def person_period(df: pd.DataFrame, interval_days: int = 30, max_days: int = 365, duration: str = "duration_days",
                  event: str = "event_flag") -> pd.DataFrame:
    """Expand to one row per member per interval at risk; event_in_period = 1 in the interval of the event."""
    n_per = int(np.ceil(max_days / interval_days))
    rows = []
    for rec in df.itertuples(index=False):
        dur, ev = getattr(rec, duration), getattr(rec, event)
        last = min(int(np.ceil(dur / interval_days)), n_per)
        for k in range(1, last + 1):
            rows.append((rec.member_id, k, int(ev == 1 and k == last and dur <= max_days)))
    pp = pd.DataFrame(rows, columns=["member_id", "period", "event_in_period"])
    return pp.merge(df.drop(columns=[duration, event]), on="member_id")


def discrete_time_hazard(df: pd.DataFrame, covariates: list[str] = COVARIATES, interval_days: int = 30,
                         max_days: int = 365, time_varying: str | None = "high_acuity_flag") -> pd.DataFrame:
    """ALTERNATIVE: pooled logistic (discrete-time) hazard with period dummies, and optionally a
    separate effect of ``time_varying`` in period 1 vs. later periods.

    Trade-off
    ---------
    + Time-varying effects and time-varying covariates are just columns; any GLM tooling works;
      the odds ratio approximates the HR when per-period risk is small.
    - Coarser time (intervals); data grow with follow-up (member x period rows); the OR drifts from
      the HR when per-period risk is large.

    Returns covariate, odds_ratio, ci_lo, ci_hi (with ``<tv>_first_period`` and ``<tv>_later`` rows).
    """
    pp = person_period(df, interval_days, max_days)
    pp["first_period"] = (pp["period"] == 1).astype(int)
    covs = [c for c in covariates if c != time_varying]
    terms = covs + ["C(period)"]
    if time_varying:
        pp[f"{time_varying}_first_period"] = pp[time_varying] * pp["first_period"]
        pp[f"{time_varying}_later"] = pp[time_varying] * (1 - pp["first_period"])
        terms += [f"{time_varying}_first_period", f"{time_varying}_later"]
    fit = smf.glm("event_in_period ~ " + " + ".join(terms), data=pp, family=sm.families.Binomial()).fit()
    keep = [t for t in fit.params.index if not t.startswith("C(period)") and t != "Intercept"]
    ci = np.exp(fit.conf_int().loc[keep])
    return pd.DataFrame({"covariate": keep, "odds_ratio": np.exp(fit.params[keep]).to_numpy(),
                         "ci_lo": ci.iloc[:, 0].to_numpy(), "ci_hi": ci.iloc[:, 1].to_numpy()})


# ---------------------------------------------------------------------------
# 4. Immortal time: exposure that starts after time zero
# ---------------------------------------------------------------------------

def cox_naive_ever_exposed(df: pd.DataFrame, covariates=("age", "frailty_z")) -> float:
    """WRONG: HR for 'ever started the program', treating members as exposed from day 0. Returns the HR."""
    cph = CoxPHFitter().fit(df[list(covariates) + ["ever_program_flag", "duration_days", "event_flag"]], "duration_days", "event_flag")
    return float(cph.hazard_ratios_["ever_program_flag"])


def to_counting_process(df: pd.DataFrame) -> pd.DataFrame:
    """Split each member at their program start: rows (start, stop, program_exposed, event).

    Days before the start are UNEXPOSED person-time; only days after count as exposed.
    """
    rows = []
    for r in df.itertuples(index=False):
        s = r.start_day
        if np.isnan(s) or s <= 0:
            rows.append((r.member_id, 0, r.duration_days, int(not np.isnan(s)), r.event_flag, r.age, r.frailty_z))
        else:
            rows.append((r.member_id, 0, s, 0, 0, r.age, r.frailty_z))
            rows.append((r.member_id, s, r.duration_days, 1, r.event_flag, r.age, r.frailty_z))
    return pd.DataFrame(rows, columns=["member_id", "start", "stop", "program_exposed", "event_flag", "age", "frailty_z"])


def cox_time_varying(df: pd.DataFrame) -> dict:
    """RIGHT: Cox model with program exposure as a time-varying covariate (lifelines CoxTimeVaryingFitter)."""
    cp = to_counting_process(df)
    f = CoxTimeVaryingFitter().fit(cp.drop(columns="member_id").assign(id=pd.factorize(cp["member_id"])[0]),
                                   id_col="id", event_col="event_flag", start_col="start", stop_col="stop")
    s = f.summary.loc["program_exposed"]
    return {"hr": float(s["exp(coef)"]), "ci_lo": float(s["exp(coef) lower 95%"]), "ci_hi": float(s["exp(coef) upper 95%"])}
