"""
G19 — Budget impact model (payer perspective, multi-year)
=========================================================
Copy this whole block into a file (e.g. budget_impact.py) and run:  python budget_impact.py
Requires: pandas   (pip install pandas)   Inputs in the self-test are FAKE.
"""
import pandas as pd


def budget_impact(members: float, member_growth: float, prevalence: float, uptake: list[float], cost_new: float,
                  cost_current: float, offset_per_treated: float = 0.0, years: int | None = None) -> pd.DataFrame:
    """Year-by-year net cost of covering a new program/treatment, total and per member per month (PMPM).

    Healthcare context
    ------------------
    Before a plan or state adds a benefit it asks "what will this cost our budget in years 1-3?", not
    "is it cost-effective?" (that's G18). Budget impact = eligible people x uptake x (new cost - displaced
    cost - offsets), spread over ALL members to express it as PMPM.

    Parameters
    ----------
    members : plan members in year 1; member_growth : annual growth rate
    prevalence : share of members eligible; uptake : share of eligible using the new option, by year
    cost_new, cost_current : annual cost per treated person with the new vs. current approach
    offset_per_treated : annual downstream savings per treated person (e.g. avoided admissions)

    Returns year, members, eligible, treated, gross_cost, offsets, net_cost, net_pmpm.

    Steps
    -----
    1. Members_y = members x (1 + growth)^(y-1); eligible = members_y x prevalence; treated = eligible x uptake_y.
    2. Gross = treated x (cost_new - cost_current); offsets = treated x offset_per_treated; net = gross - offsets.
    3. net PMPM = net / (members_y x 12).

    Common mistakes
    ---------------
    - Assuming 100% uptake in year 1.
    - Counting offsets that take years to appear (readmissions avoided) in year 1 at full size.
    - Dividing by treated members instead of all members when quoting PMPM to Finance.
    """
    years = years or len(uptake)
    rows = []
    for y in range(1, years + 1):
        m = members * (1 + member_growth) ** (y - 1)                                              # step 1
        elig = m * prevalence
        treated = elig * uptake[y - 1]
        gross = treated * (cost_new - cost_current)                                               # step 2
        off = treated * offset_per_treated
        rows.append({"year": y, "members": m, "eligible": elig, "treated": treated, "gross_cost": gross,
                     "offsets": off, "net_cost": gross - off, "net_pmpm": (gross - off) / (m * 12)})   # step 3
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test: run `python budget_impact.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = budget_impact(members=200_000, member_growth=0.02, prevalence=0.03, uptake=[0.10, 0.25, 0.40],
                      cost_new=2_400, cost_current=0, offset_per_treated=1_500)
    print(r.round(2))
    checks = {
        "year 1: 200,000 x 3% x 10% = 600 treated": abs(r.treated[0] - 600) < 1e-9,
        "year 1 net = 600 x (2,400 - 1,500) = 540,000": abs(r.net_cost[0] - 540_000) < 1e-6,
        "year 1 PMPM = 540,000 / (200,000 x 12) = $0.225": abs(r.net_pmpm[0] - 0.225) < 1e-12,
        "members grow 2% a year": abs(r.members[2] - 200_000 * 1.02 ** 2) < 1e-6,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
