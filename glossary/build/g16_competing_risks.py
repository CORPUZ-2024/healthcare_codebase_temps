"""
G16 — Competing risks: cumulative incidence (Aalen-Johansen) vs. the 1 - Kaplan-Meier mistake
==============================================================================================
Copy this whole block into a file (e.g. cif.py) and run:  python cif.py
Requires: numpy, pandas   (pip install numpy pandas)
Regression on the subdistribution hazard (Fine-Gray): R cmprsk / Python lifelines-extensions; cause-specific Cox: t11.
"""
import numpy as np
import pandas as pd


def cumulative_incidence(time: np.ndarray, event: np.ndarray, cause: int = 1) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence of ``cause`` when other causes compete.

    Healthcare context
    ------------------
    "Probability of nursing-home placement within 2 years" among frail older adults: many die first.
    Treating death as censoring and reporting 1 - KM assumes the dead would have gone on to be placed
    at the same rate - it OVERSTATES placement risk. The cumulative incidence function (CIF) counts
    death as what it is: an event that removes the possibility of placement.

    Parameters
    ----------
    time : follow-up time; event : 0 = censored, 1, 2, ... = cause codes

    Returns time, cif (for ``cause``), naive_1_minus_km (competing events treated as censored).

    Steps
    -----
    1. At each event time t: n at risk, d_cause, d_any.
    2. Overall survival S(t) = prod(1 - d_any / n)  (all causes).
    3. CIF(t) = sum over event times <= t of S(t-) x d_cause / n.

    Common mistakes
    ---------------
    - Reporting 1 - KM with competing events censored (always >= CIF).
    - Interpreting cause-specific hazard ratios as effects on absolute risk.
    """
    t, e = np.asarray(time, float), np.asarray(event, int)
    times = np.unique(t[e > 0])
    s_prev, cif, km_naive = 1.0, 0.0, 1.0
    rows = []
    for u in times:                                                                        # step 1
        n = (t >= u).sum()
        d_c = ((t == u) & (e == cause)).sum()
        d_any = ((t == u) & (e > 0)).sum()
        cif += s_prev * d_c / n                                                            # step 3
        km_naive *= 1 - d_c / n
        s_prev *= 1 - d_any / n                                                            # step 2
        rows.append((u, cif, 1 - km_naive))
    return pd.DataFrame(rows, columns=["time", "cif", "naive_1_minus_km"])


# ---------------------------------------------------------------------------
# Self-test: run `python cif.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 5_000
    t_place = rng.exponential(1 / 0.15, n)            # placement hazard 0.15/yr
    t_death = rng.exponential(1 / 0.25, n)            # death hazard 0.25/yr (competing)
    t = np.minimum(t_place, t_death)
    ev = np.where(t_place < t_death, 1, 2)
    r = cumulative_incidence(t, ev, cause=1)
    at2 = r[r.time <= 2].iloc[-1]
    true_cif2 = 0.15 / 0.40 * (1 - np.exp(-0.40 * 2))                  # closed form with constant hazards
    toy = cumulative_incidence(np.array([1, 2, 3, 4]), np.array([1, 2, 1, 0]))
    print(at2.round(3).to_dict(), "true", round(true_cif2, 3), "\n", toy)
    checks = {
        "CIF at 2 years matches the closed form": abs(at2.cif - true_cif2) < 0.015,
        "1 - KM overstates placement risk": at2.naive_1_minus_km > at2.cif + 0.02,
        "no censoring: CIF = observed share (toy: 2 of 4 placed)": abs(toy.cif.iloc[-1] - 0.5) < 1e-12,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
