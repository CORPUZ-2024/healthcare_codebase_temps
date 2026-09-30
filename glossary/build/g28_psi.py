"""
G28 — Population stability index (PSI): has a model's input or score distribution drifted?
===========================================================================================
Copy this whole block into a file (e.g. psi.py) and run:  python psi.py
Requires: numpy, pandas   (pip install numpy pandas)
"""
import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> dict:
    """PSI = sum over bins (a - e) x ln(a / e), bins = quantiles of the EXPECTED (baseline) distribution.

    Healthcare context
    ------------------
    A risk model (t03) trained on last year's members is scored monthly. Benefit changes, Medicaid
    redetermination, a new clinic network or coding changes shift the inputs and scores. PSI is the
    standard one-number drift monitor; industry rules of thumb: < 0.10 stable, 0.10-0.25 watch,
    > 0.25 investigate / recalibrate. It says THAT something moved, not whether performance dropped
    (check calibration and PPV too - G29, t03).

    Returns dict: psi, status, table (bin, expected_share, actual_share, contribution).

    Steps
    -----
    1. Bin edges = baseline quantiles (so each baseline bin holds ~10%); outer edges open.
    2. Shares per bin in each sample; floor at eps (empty bins would give ln(0)).
    3. Contributions (a - e) ln(a / e) summed.

    Common mistakes
    ---------------
    - Re-computing bin edges on the new data (hides the drift).
    - Many bins on small samples (noise looks like drift); keep ~10 bins and >= a few hundred rows.
    - Treating PSI as symmetric in meaning: which population is the baseline matters for interpretation.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))                           # step 1
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)                                            # step 2
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    contrib = (a - e) * np.log(a / e)                                                               # step 3
    total = float(contrib.sum())
    status = "stable" if total < 0.10 else ("watch" if total <= 0.25 else "investigate")
    return {"psi": total, "status": status,
            "table": pd.DataFrame({"bin": range(len(e)), "expected_share": e, "actual_share": a, "contribution": contrib})}


# ---------------------------------------------------------------------------
# Self-test: run `python psi.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(8)
    base = rng.beta(2, 8, 20_000)                      # last year's risk scores
    same = rng.beta(2, 8, 20_000)
    mild = rng.beta(2.3, 8, 20_000)
    big = rng.beta(3, 6, 20_000)                       # sicker population after a network change
    r = {k: psi(base, v) for k, v in (("same", same), ("mild", mild), ("big", big))}
    print({k: (round(v["psi"], 3), v["status"]) for k, v in r.items()})
    checks = {
        "same distribution -> stable (~0)": r["same"]["status"] == "stable" and r["same"]["psi"] < 0.01,
        "small shift -> above the same-distribution PSI": r["mild"]["psi"] > r["same"]["psi"],
        "large shift -> investigate": r["big"]["status"] == "investigate",
        "baseline bins hold ~10% each": np.allclose(r["same"]["table"].expected_share, 0.1, atol=0.002),
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
