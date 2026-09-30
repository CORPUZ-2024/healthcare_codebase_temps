"""
HEDIS / Core Set style quality measures from YAML specs, intervals, hybrid method, care gaps.

    evaluate_measure (admin sources)  (STANDARD)  vs  hybrid_rate (sample + chart review)  (ALTERNATIVE)
    ci_wilson                         (STANDARD)  vs  ci_jeffreys                          (ALTERNATIVE)
    + continuous_enrollment, rate_summary (stability flag), care_gap_list, benchmark_position,
      interval_coverage (exact coverage of an interval method, for choosing between the two)

Vocabulary (NCQA wording)
-------------------------
eligible population  age/sex + continuous enrollment + event-based criteria (e.g. diabetes dx)
exclusions           removed from the eligible population (hospice, bilateral mastectomy, ...)
denominator          eligible population minus exclusions
numerator            denominator members with a qualifying service in the lookback window
open care gap        denominator member without a numerator event
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 0. Helpers: codes, dates, age
# ---------------------------------------------------------------------------

def normalize_codes(events: pd.DataFrame) -> pd.DataFrame:
    """Strip dots/spaces and upper-case ``code_cd`` so 'e11.9' matches value-set code 'E119'.

    >>> normalize_codes(pd.DataFrame({"code_cd": ["e11.9", " Z00.129 "]}))["code_cd"].tolist()
    ['E119', 'Z00129']
    """
    out = events.copy()
    out["code_cd"] = out["code_cd"].astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
    return out


def window(my: int, start_offset_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Lookback window [Jan 1 of MY + offset months, Dec 31 of MY].

    >>> [str(d.date()) for d in window(2025, -15)]
    ['2023-10-01', '2025-12-31']
    """
    start = pd.Timestamp(f"{my}-01-01") + pd.DateOffset(months=int(start_offset_months))
    return start, pd.Timestamp(f"{my}-12-31")


def age_on(birth_dt: pd.Series, as_of: str | pd.Timestamp) -> pd.Series:
    """Age in completed years on ``as_of`` (HEDIS: as of Dec 31 of the measurement year).

    >>> age_on(pd.Series(pd.to_datetime(["1951-12-31", "1952-01-01"])), "2025-12-31").tolist()
    [74, 73]
    """
    t = pd.Timestamp(as_of)
    before_bday = (birth_dt.dt.month > t.month) | ((birth_dt.dt.month == t.month) & (birth_dt.dt.day > t.day))
    return (t.year - birth_dt.dt.year - before_bday.astype(int)).astype(int)


def tag_events(events: pd.DataFrame, value_sets: pd.DataFrame) -> pd.DataFrame:
    """Attach ``value_set_name`` to events by joining on (code_system, code_cd).

    Joining on code alone is a classic bug: the same string can be a valid code in two systems.
    One event can belong to several value sets, so rows can multiply - that is intended.
    """
    ev = normalize_codes(events)
    return ev.merge(value_sets[["value_set_name", "code_system", "code_cd"]], on=["code_system", "code_cd"])


# ---------------------------------------------------------------------------
# 1. Continuous enrollment
# ---------------------------------------------------------------------------

def continuous_enrollment(enrollment: pd.DataFrame, ce_start, ce_end, anchor_dt=None, allowable_gaps: int = 1,
                          max_gap_days: int = 45, members: pd.Series | None = None) -> pd.DataFrame:
    """Continuous-enrollment (CE) test for one period, HEDIS style.

    Healthcare context
    ------------------
    A plan is only accountable for a member it covered long enough to deliver the service.
    HEDIS: enrolled for the whole period with no more than ONE gap of up to 45 days per year,
    and enrolled on the anchor date (usually Dec 31). A gap at the start or end of the period
    counts as a gap. Overlapping spans (retro plan changes) must not hide or create gaps.

    Parameters
    ----------
    enrollment : member_id, enroll_start_dt, enroll_end_dt (inclusive dates)
    ce_start, ce_end : period to test (inclusive)
    anchor_dt : date the member must be enrolled on (None = no anchor rule)
    members : optional list of member_ids to report; members with no span in the period get
              one gap covering the whole period (ce_flag = 0)

    Returns
    -------
    DataFrame: member_id, gap_cnt, max_gap_days, anchor_enrolled_flag, ce_flag

    Steps
    -----
    1. Keep spans that touch the period and clip them to it.
    2. Sort by start; running max of end dates of earlier spans (so overlaps never look like gaps).
    3. Gap before each span = start - (previous running max end) - 1 day (first span: start - ce_start).
    4. Trailing gap = ce_end - max end.
    5. gap_cnt = number of positive gaps; CE if gap_cnt <= allowable_gaps, longest gap <= max_gap_days
       and enrolled on the anchor date.

    Example
    -------
    >>> e = pd.DataFrame({"member_id": ["A", "A", "B"],
    ...     "enroll_start_dt": pd.to_datetime(["2025-01-01", "2025-03-17", "2025-01-01"]),
    ...     "enroll_end_dt": pd.to_datetime(["2025-01-31", "2025-12-31", "2025-11-30"])})
    >>> continuous_enrollment(e, "2025-01-01", "2025-12-31", "2025-12-31").to_dict("records")[0]
    {'member_id': 'A', 'gap_cnt': 1, 'max_gap_days': 44, 'anchor_enrolled_flag': 1, 'ce_flag': 1}

    Common mistakes
    ---------------
    - Counting days enrolled instead of gaps (two 30-day gaps = 60 days missing = still not CE).
    - Summing overlapping spans: a retro plan change looks like >365 days of coverage.
    - Forgetting that "enrolled Feb 1 onward" is a 31-day gap at the START of the year.
    - Testing a two-year CE period as one block: the rule is one gap per year (see ``ce_flags``).
    """
    s0, s1 = pd.Timestamp(ce_start), pd.Timestamp(ce_end)
    e = enrollment[(enrollment["enroll_end_dt"] >= s0) & (enrollment["enroll_start_dt"] <= s1)]
    e = e.assign(s=e["enroll_start_dt"].clip(lower=s0), e=e["enroll_end_dt"].clip(upper=s1))       # step 1
    e = e.sort_values(["member_id", "s", "e"])
    prev_max = e.groupby("member_id")["e"].transform(lambda x: x.cummax().shift())                 # step 2
    gap = np.where(prev_max.isna(), (e["s"] - s0).dt.days, (e["s"] - prev_max).dt.days - 1)         # step 3
    e = e.assign(gap_days=np.maximum(gap, 0))
    g = e.groupby("member_id")
    out = pd.DataFrame({"lead_gaps": g["gap_days"].apply(lambda x: int((x > 0).sum())),
                        "lead_max": g["gap_days"].max(),
                        "trail": (s1 - g["e"].max()).dt.days})                                      # step 4
    if anchor_dt is not None:
        a = pd.Timestamp(anchor_dt)
        out["anchor_enrolled_flag"] = ((e["s"] <= a) & (e["e"] >= a)).groupby(e["member_id"]).any().astype(int)
    else:
        out["anchor_enrolled_flag"] = 1
    out["gap_cnt"] = out["lead_gaps"] + (out["trail"] > 0).astype(int)
    out["max_gap_days"] = out[["lead_max", "trail"]].max(axis=1).astype(int)
    out = out.reset_index()[["member_id", "gap_cnt", "max_gap_days", "anchor_enrolled_flag"]]
    if members is not None:
        missing = pd.Index(pd.unique(members)).difference(out["member_id"])
        out = pd.concat([out, pd.DataFrame({"member_id": missing, "gap_cnt": 1,
                                            "max_gap_days": (s1 - s0).days + 1, "anchor_enrolled_flag": 0})],
                        ignore_index=True)
    out["ce_flag"] = ((out["gap_cnt"] <= allowable_gaps) & (out["max_gap_days"] <= max_gap_days)
                      & (out["anchor_enrolled_flag"] == 1)).astype(int)                             # step 5
    return out.astype({"gap_cnt": int, "max_gap_days": int, "anchor_enrolled_flag": int})


def ce_flags(enrollment: pd.DataFrame, spec: dict, my: int, members: pd.Series) -> pd.DataFrame:
    """Apply the spec's CE rule year by year (one allowable gap PER YEAR), anchor on Dec 31 of MY.

    Returns member_id, gap_cnt (total), max_gap_days, anchor_enrolled_flag, ce_flag.
    """
    rule = spec["continuous_enrollment"]
    start, end = window(my, rule["start_offset_months"])
    parts, cursor = [], start
    while cursor <= end:
        seg_end = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        anchor = end if (seg_end == end and rule.get("anchor_enrolled", True)) else None
        parts.append(continuous_enrollment(enrollment, cursor, seg_end, anchor, rule["allowable_gaps"],
                                           rule["max_gap_days"], members))
        cursor = seg_end + pd.Timedelta(days=1)
    allp = pd.concat(parts)
    return allp.groupby("member_id", as_index=False).agg(
        gap_cnt=("gap_cnt", "sum"), max_gap_days=("max_gap_days", "max"),
        anchor_enrolled_flag=("anchor_enrolled_flag", "min"), ce_flag=("ce_flag", "min"))


# ---------------------------------------------------------------------------
# 2. Measure engine (STANDARD: administrative method)
# ---------------------------------------------------------------------------

def _hits(tagged: pd.DataFrame, value_sets: list[str], start, end, sources) -> pd.DataFrame:
    return tagged[tagged["value_set_name"].isin(value_sets) & tagged["event_dt"].between(start, end)
                  & tagged["source_cd"].isin(sources)]


def evaluate_measure(spec: dict, members: pd.DataFrame, enrollment: pd.DataFrame, events: pd.DataFrame,
                     value_sets: pd.DataFrame, my: int) -> pd.DataFrame:
    """STANDARD: administrative-method member-level results for one measure spec.

    Healthcare context
    ------------------
    HEDIS and Medicaid Core Set measures are rule chains: who is eligible, who is excluded, who
    got the service in the lookback window. The "administrative" method uses claims plus
    standard supplemental data (e.g. a lab-results feed) for the WHOLE eligible population.
    The member-level output is the audit trail AND the source of the care-gap list.

    Parameters
    ----------
    spec : dict from measures/*.yaml (see data.load_measure)
    members : member_id, birth_dt, sex_cd
    enrollment : member_id, enroll_start_dt, enroll_end_dt
    events : member_id, event_dt, code_system, code_cd, source_cd
    value_sets : value_set_name, code_system, code_cd
    my : measurement year

    Returns
    -------
    One row per member who meets the age/sex criteria:
    member_id, measure_id, age, ce_flag, gap_cnt, max_gap_days, denom_event_flag, eligible_flag,
    exclusion_flag, exclusion_reason, denominator_flag, numerator_flag, numerator_dt,
    gap_open_flag, status (NOT_CE | NO_DENOM_EVENT | EXCLUDED | MET | OPEN_GAP)

    Steps
    -----
    1. Tag events with value sets (normalized codes, joined on system + code).
    2. Age on Dec 31 of MY and sex -> population.
    3. Continuous enrollment, one allowable gap per year, enrolled on Dec 31.
    4. Event-based eligibility (e.g. diabetes on >= 2 distinct dates) from admin sources.
    5. Exclusions (any event in an exclusion value set inside its window) -> removed.
    6. Numerator: numerator value-set event inside the lookback, admin sources only.
    7. Open gap = in the denominator and not in the numerator.

    Common mistakes
    ---------------
    - Computing age on the service date or today instead of the anchor date.
    - Removing exclusions BEFORE continuous enrollment and reporting "excluded" counts that
      include members who were never eligible.
    - Counting chart-review (CHART) evidence in an administrative rate.
    - Letting exclusion evidence from outside its window exclude a member (hospice last year).
    """
    admin = list(spec["numerator"]["admin_sources"])
    tagged = tag_events(events, value_sets)                                                    # step 1
    anchor = f"{my}-12-31"
    pop = spec["population"]
    m = members.assign(age=age_on(members["birth_dt"], anchor))                                # step 2
    m = m[m["age"].between(pop["age_min"], pop["age_max"])]
    if pop.get("sex"):
        m = m[m["sex_cd"] == pop["sex"]]
    out = m[["member_id", "age"]].assign(measure_id=spec["measure_id"])
    out = out.merge(ce_flags(enrollment, spec, my, out["member_id"]), on="member_id", how="left")  # step 3

    de = spec.get("denominator_event")                                                         # step 4
    if de:
        s, e = window(my, de["start_offset_months"])
        h = _hits(tagged, de["value_sets"], s, e, admin)
        n_dates = h.groupby("member_id")["event_dt"].nunique()
        out["denom_event_flag"] = (out["member_id"].map(n_dates).fillna(0) >= de.get("min_distinct_dates", 1)).astype(int)
    else:
        out["denom_event_flag"] = 1
    out["eligible_flag"] = (out["ce_flag"] & out["denom_event_flag"]).astype(int)

    out["exclusion_reason"] = None                                                             # step 5
    for ex in spec.get("exclusions", []):
        s, e = window(my, ex["start_offset_months"])
        who = set(_hits(tagged, [ex["value_set"]], s, e, admin)["member_id"])
        hit = out["member_id"].isin(who) & (out["eligible_flag"] == 1) & out["exclusion_reason"].isna()
        out.loc[hit, "exclusion_reason"] = ex["value_set"]
    out["exclusion_flag"] = out["exclusion_reason"].notna().astype(int)
    out["denominator_flag"] = ((out["eligible_flag"] == 1) & (out["exclusion_flag"] == 0)).astype(int)

    num = spec["numerator"]                                                                    # step 6
    s, e = window(my, num["start_offset_months"])
    last = _hits(tagged, num["value_sets"], s, e, admin).groupby("member_id")["event_dt"].max()
    out["numerator_dt"] = out["member_id"].map(last)
    out["numerator_flag"] = ((out["denominator_flag"] == 1) & out["numerator_dt"].notna()).astype(int)
    out["gap_open_flag"] = ((out["denominator_flag"] == 1) & (out["numerator_flag"] == 0)).astype(int)  # step 7
    out["status"] = np.select(
        [out["ce_flag"] == 0, out["denom_event_flag"] == 0, out["exclusion_flag"] == 1, out["numerator_flag"] == 1],
        ["NOT_CE", "NO_DENOM_EVENT", "EXCLUDED", "MET"], default="OPEN_GAP")
    cols = ["member_id", "measure_id", "age", "ce_flag", "gap_cnt", "max_gap_days", "denom_event_flag",
            "eligible_flag", "exclusion_flag", "exclusion_reason", "denominator_flag", "numerator_flag",
            "numerator_dt", "gap_open_flag", "status"]
    return out[cols].sort_values("member_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Confidence intervals for a proportion
# ---------------------------------------------------------------------------

def ci_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """STANDARD: Wilson score interval for k successes out of n.

    Healthcare context
    ------------------
    Measure rates are proportions, often on small denominators (a provider group, a county).
    The textbook Wald interval p +/- z*sqrt(p(1-p)/n) collapses to zero width at 0% or 100% and
    under-covers for small n. Wilson stays inside [0, 1] and covers close to 95%.

    >>> [round(x, 4) for x in ci_wilson(0, 10)]
    [0.0, 0.2775]
    >>> [round(x, 4) for x in ci_wilson(81, 263)]
    [0.2553, 0.3662]

    Common mistakes
    ---------------
    - Using the Wald interval (and reporting "95% CI 0% to 0%").
    - Computing a CI on a hybrid rate with n = denominator instead of n = sample size.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def ci_jeffreys(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """ALTERNATIVE: Jeffreys interval = Beta(k + 0.5, n - k + 0.5) quantiles.

    Trade-off
    ---------
    + Near 0% or 100% on small denominators (n ~ 30-60: rare events, near-perfect measures)
      Wilson's worst-case coverage dips to ~85%; Jeffreys' dips less (~89%). Check your own n with
      ``interval_coverage`` - at n = 100 the ordering flips.
    + Equal-tailed, so one-sided statements ("below target") are more honest; Bayesian reading:
      a credible interval under the Jeffreys prior.
    - Not closer to 95% on average, less familiar to reviewers. Brown-Cai-DasGupta boundary rule
      applied: lower = 0 if k = 0, upper = 1 if k = n.

    >>> [round(x, 4) for x in ci_jeffreys(0, 10)]
    [0.0, 0.2172]
    """
    if n == 0:
        return (float("nan"), float("nan"))
    alpha = 2 * stats.norm.sf(z)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return (float(lo), float(hi))


CI_METHODS = {"wilson": ci_wilson, "jeffreys": ci_jeffreys}


def interval_coverage(n: int, p: float, method: str = "wilson", z: float = 1.96) -> float:
    """Exact coverage: probability that the interval from Binomial(n, p) data contains p.

    Use it to pick an interval for YOUR typical denominator and rate.
    >>> round(interval_coverage(50, 0.5, "wilson"), 3)
    0.935
    """
    f = CI_METHODS[method]
    ks = np.arange(n + 1)
    inside = np.array([f(int(k), n, z)[0] <= p <= f(int(k), n, z)[1] for k in ks])
    return float(stats.binom.pmf(ks, n, p)[inside].sum())


# ---------------------------------------------------------------------------
# 4. Rate summary with stability flag
# ---------------------------------------------------------------------------

def rate_summary(member_level: pd.DataFrame, interval: str = "wilson", z: float = 1.96,
                 min_denominator: int = 30) -> dict:
    """Counts, rate, CI and a reportability flag for one measure.

    Healthcare context
    ------------------
    Medicaid Core Set reporting and NCQA both suppress or flag rates with small denominators
    (commonly < 30). Report counts at every step so a reviewer can rebuild the funnel:
    eligible -> exclusions -> denominator -> numerator.

    Returns dict of plain ints/floats: eligible_cnt, exclusion_cnt, denominator_cnt, numerator_cnt,
    rate, ci_lo, ci_hi, interval, reportable_flag, note.
    """
    n = int(member_level["denominator_flag"].sum())
    k = int(member_level["numerator_flag"].sum())
    lo, hi = CI_METHODS[interval](k, n, z)
    ok = n >= min_denominator
    return {"measure_id": member_level["measure_id"].iloc[0] if len(member_level) else None,
            "eligible_cnt": int(member_level["eligible_flag"].sum()),
            "exclusion_cnt": int(member_level["exclusion_flag"].sum()),
            "denominator_cnt": n, "numerator_cnt": k, "rate": float(k / n) if n else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "interval": interval, "reportable_flag": int(ok),
            "note": "" if ok else f"NR: denominator {n} < {min_denominator} - do not publish or rank"}


# ---------------------------------------------------------------------------
# 5. ALTERNATIVE numerator: hybrid method (systematic sample + medical-record review)
# ---------------------------------------------------------------------------

def systematic_sample(ids: pd.Series, size: int, seed: int = 0) -> pd.Series:
    """NCQA-style systematic sample: sort, random start, every (N / size)-th member."""
    ids = pd.Series(sorted(pd.unique(ids)))
    if len(ids) <= size:
        return ids
    step = len(ids) / size
    start = np.random.default_rng(seed).uniform(0, step)
    return ids.iloc[np.floor(start + step * np.arange(size)).astype(int)].reset_index(drop=True)


def hybrid_rate(spec: dict, member_level: pd.DataFrame, events: pd.DataFrame, value_sets: pd.DataFrame,
                my: int, sample_size: int = 411, seed: int = 0, z: float = 1.96) -> tuple[dict, pd.DataFrame]:
    """ALTERNATIVE: hybrid-method rate - admin hits OR chart-review hits on a systematic sample.

    Trade-off
    ---------
    + Captures services that never produce a usable claim (point-of-care A1c, bundled payment,
      care before enrollment documented in the chart). Rates are typically several points higher.
    - Only allowed for measures whose spec permits it (``hybrid_sources`` non-empty); costs chart
      abstraction for every sampled member; the CI is based on the SAMPLE (n ~ 411), not the
      denominator; hybrid and admin rates are NOT comparable (check the Core Set ``methodology``).
    - Not modelled: oversample replacement of members excluded during chart review.

    Returns (summary dict like rate_summary + sample_size, sampled member-level frame).
    """
    hyb = list(spec["numerator"].get("hybrid_sources", []))
    if not hyb:
        raise ValueError(f"{spec['measure_id']} is administrative-only; the hybrid method is not allowed")
    den = member_level[member_level["denominator_flag"] == 1]
    sample_ids = systematic_sample(den["member_id"], sample_size, seed)
    smp = den[den["member_id"].isin(set(sample_ids))].copy()
    num = spec["numerator"]
    s, e = window(my, num["start_offset_months"])
    chart = _hits(tag_events(events, value_sets), num["value_sets"], s, e, hyb)
    smp["chart_flag"] = smp["member_id"].isin(set(chart["member_id"])).astype(int)
    smp["numerator_flag"] = ((smp["numerator_flag"] == 1) | (smp["chart_flag"] == 1)).astype(int)
    smp["gap_open_flag"] = 1 - smp["numerator_flag"]
    summ = rate_summary(smp, "wilson", z)
    summ.update(interval="wilson (sample)", sample_size=len(smp), method="hybrid")
    return summ, smp.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Care-gap list (the operational output)
# ---------------------------------------------------------------------------

def care_gap_list(member_level: pd.DataFrame, spec: dict, events: pd.DataFrame, value_sets: pd.DataFrame,
                  my: int) -> pd.DataFrame:
    """Member-level open gaps for outreach: who, which measure, what to do by when, last service.

    Healthcare context
    ------------------
    Care managers and provider groups work gap lists, not rates. Include the last qualifying
    service ever seen (any source, any date) so outreach can say "last mammogram Aug 2023" and
    so a gap already closed in the chart is caught before a member is called.

    Returns member_id, measure_id, measure_title, age, close_by_dt, needed, last_service_dt,
    last_service_source, note.

    Common mistakes
    ---------------
    - Sending members who are excluded (hospice) or not continuously enrolled - they are not gaps.
    - Using a list built on incomplete claims (run after runout, or refresh weekly).
    """
    g = member_level[member_level["gap_open_flag"] == 1][["member_id", "measure_id", "age"]].copy()
    num = spec["numerator"]
    s, e = window(my, num["start_offset_months"])
    any_hit = tag_events(events, value_sets)
    any_hit = any_hit[any_hit["value_set_name"].isin(num["value_sets"])].sort_values("event_dt")
    last = any_hit.groupby("member_id").tail(1).set_index("member_id")
    g["measure_title"] = spec["title"]
    g["close_by_dt"] = e
    g["needed"] = f"{'/'.join(num['value_sets'])} service dated {s.date()} to {e.date()}"
    g["last_service_dt"] = g["member_id"].map(last["event_dt"])
    g["last_service_source"] = g["member_id"].map(last["source_cd"])
    g["note"] = np.where(g["last_service_source"].isin(num.get("hybrid_sources", []))
                         & g["last_service_dt"].between(s, e),
                         "documented in chart only - request record / supplemental data, do not call member",
                         np.where(g["last_service_dt"].notna() & (g["last_service_dt"] < s), "overdue: last service before window", ""))
    cols = ["member_id", "measure_id", "measure_title", "age", "close_by_dt", "needed", "last_service_dt",
            "last_service_source", "note"]
    return g[cols].sort_values(["measure_id", "member_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Benchmark position
# ---------------------------------------------------------------------------

def benchmark_position(rate: float, bench: pd.DataFrame, measure_cd: str) -> dict:
    """Where a rate falls among published state rates for the same measure.

    Returns n_states, p25, median, p75, percentile (0-100), quartile ('Q1 bottom' .. 'Q4 top').
    Compare like with like: same Core Set year vintage, methodology and population (see checks).
    """
    r = bench.loc[bench["measure_cd"] == measure_cd, "state_rate"].dropna().to_numpy()
    if len(r) == 0:
        return {"measure_cd": measure_cd, "n_states": 0}
    p25, med, p75 = np.percentile(r, [25, 50, 75])
    q = "Q1 bottom" if rate < p25 else "Q2" if rate < med else "Q3" if rate < p75 else "Q4 top"
    return {"measure_cd": measure_cd, "n_states": int(len(r)), "p25": float(p25), "median": float(med),
            "p75": float(p75), "percentile": float(stats.percentileofscore(r, rate, kind="mean")), "quartile": q}
