"""
Synthetic RAW tables for an outreach A/B test, shaped like warehouse extracts (messy names, strings).

    raw_members         member_key, lob (mixed case, padded), enroll_start, age
    raw_assignments     member_key, variant ('Control' / 'Treatment '), assigned_at
    raw_outreach        member_key, attempt_ts, outcome   (0-4 attempts per member)
    raw_claims_monthly  member_key, month_start, paid, period ('pre' / 'post'), 6 + 6 months

Truth (for tests): engagement +3 pp, post-period PMPM -3%, complaints +0.2 pp.
``inject_issues=True`` adds the defects the schema tests exist to catch: a duplicated member, an
unknown LOB, an assignment for a member missing from eligibility, and a stray arm label.
Optional ``srm_drop`` silently loses a share of treatment members (a sample-ratio-mismatch bug).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(n: int = 20_000, engagement_control: float = 0.22, engagement_lift: float = 0.03,
             pmpm_effect_pct: float = -0.03, complaint_control: float = 0.010, complaint_lift: float = 0.002,
             inject_issues: bool = False, srm_drop: float = 0.0, seed: int = 14) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    keys = np.array([f"{i:07d}" for i in range(1, n + 1)])
    lob = rng.choice(["mcd", " MCD", "Dual", "MCR "], n, p=[0.5, 0.2, 0.2, 0.1])
    members = pd.DataFrame({"member_key": keys, "lob": lob, "enroll_start": "2024-01-01",
                            "age": rng.integers(18, 90, n).astype(str)})
    arm = rng.permutation(np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)])
    assign = pd.DataFrame({"member_key": keys, "variant": np.where(arm == 1, "Treatment ", "Control"),
                           "assigned_at": "2025-01-15"})

    p_eng = engagement_control + engagement_lift * arm
    p_cmp = complaint_control + complaint_lift * arm
    rows = []
    for k, e, c, a_n in zip(keys, rng.random(n) < p_eng, rng.random(n) < p_cmp, rng.integers(0, 5, n)):
        outs = list(rng.choice(["no_answer", "declined"], a_n))
        if e:
            outs.append("engaged")
        if c:
            outs.append("complaint")
        if rng.random() < 0.02:
            outs.append("opt_out")
        for j, o in enumerate(outs):
            rows.append((k, f"2025-01-{16 + j:02d}", o.upper() if rng.random() < 0.1 else o))
    outreach = pd.DataFrame(rows, columns=["member_key", "attempt_ts", "outcome"])

    base = np.exp(rng.normal(np.log(450), 1.0, n))                      # member-level cost propensity
    months = []
    for per, mult in (("pre", np.ones(n)), ("post", 1.03 * (1 + pmpm_effect_pct * arm))):
        for mth in range(6):
            paid = base * mult * np.exp(rng.normal(0, 0.8, n) - 0.32)
            start = pd.Timestamp("2024-07-01" if per == "pre" else "2025-02-01") + pd.DateOffset(months=mth)
            months.append(pd.DataFrame({"member_key": keys, "month_start": start.strftime("%Y-%m-%d"),
                                        "paid": paid.round(2), "period": per}))
    claims = pd.concat(months, ignore_index=True)

    if srm_drop:
        lost = keys[(arm == 1) & (rng.random(n) < srm_drop)]
        assign = assign[~assign["member_key"].isin(lost)]
    if inject_issues:
        members = pd.concat([members, members.iloc[[0]], pd.DataFrame(
            [{"member_key": "9999999", "lob": "COMMERCIAL", "enroll_start": "2024-01-01", "age": "40"}])], ignore_index=True)
        assign = pd.concat([assign, pd.DataFrame([{"member_key": "8888888", "variant": "control", "assigned_at": "2025-01-15"},
                                                  {"member_key": keys[5], "variant": "holdout", "assigned_at": "2025-01-15"}])],
                           ignore_index=True)
    return {"raw_members": members, "raw_assignments": assign, "raw_outreach": outreach, "raw_claims_monthly": claims}
