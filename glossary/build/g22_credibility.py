"""
G22 — Limited-fluctuation credibility: blend a group's own experience with a manual rate
========================================================================================
Copy this whole block into a file (e.g. credibility.py) and run:  python credibility.py
Requires: scipy   (pip install scipy)   Results for rating/reserving need actuarial sign-off.
"""
import math

from scipy import stats


def full_credibility_standard(p: float = 0.90, k: float = 0.05, cv_severity: float | None = None) -> float:
    """Expected CLAIM COUNT needed for full credibility: observed within +/- k of the truth with probability p.

    Frequency only (Poisson claims): n_F = (z / k)^2, z = standard normal quantile at (1 + p) / 2.
    Pure premium (frequency x severity): n_F x (1 + CV_severity^2).

    >>> round(full_credibility_standard())
    1082
    """
    z = stats.norm.ppf((1 + p) / 2)
    n = (z / k) ** 2
    return n * (1 + cv_severity ** 2) if cv_severity is not None else n


def credibility_blend(observed: float, manual: float, n_claims: float, n_full: float) -> dict:
    """Square-root rule: Z = min(1, sqrt(n / n_full)); estimate = Z x observed + (1 - Z) x manual.

    Healthcare context
    ------------------
    A 400-member employer group or a small provider panel has a volatile PMPM. Credibility weighting
    gives its own experience a weight Z that grows with volume, and fills the rest from a manual
    (book-of-business or benchmark) rate. The same logic underlies setting small-group premiums and
    judging whether a small practice's cost trend is real.

    Returns dict: z, blended.

    Common mistakes
    ---------------
    - Using MEMBERS instead of expected CLAIMS in the standard (the rule is about claim counts).
    - Forgetting severity variation for dollar measures (health costs have CV 2-4, so n_full is 5-17x larger).
    - Blending with a manual rate from a different period or benefit design without trending/adjusting it.
    """
    z = min(1.0, math.sqrt(n_claims / n_full))
    return {"z": z, "blended": z * observed + (1 - z) * manual}


# ---------------------------------------------------------------------------
# Self-test: run `python credibility.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nf = full_credibility_standard()
    nf_pp = full_credibility_standard(cv_severity=2.5)
    small = credibility_blend(observed=620.0, manual=500.0, n_claims=nf / 4, n_full=nf)
    print(f"n_full (frequency) = {nf:.0f} claims; pure premium with CV 2.5 = {nf_pp:,.0f}; small group: {small}")
    checks = {
        "classic standard: 1,082 claims (p = 90%, k = 5%)": round(nf) == 1082,
        "severity CV 2.5 multiplies the standard by 7.25": abs(nf_pp / nf - 7.25) < 1e-9,
        "a quarter of the standard -> Z = 0.5": abs(small["z"] - 0.5) < 1e-12,
        "blend = 0.5 x 620 + 0.5 x 500 = 560": abs(small["blended"] - 560) < 1e-9,
        "Z capped at 1": credibility_blend(1, 0, 5 * nf, nf)["z"] == 1.0,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
