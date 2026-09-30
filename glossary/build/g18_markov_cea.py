"""
G18 — Cost-effectiveness: Markov cohort model -> costs, QALYs, ICER, net monetary benefit
=========================================================================================
Copy this whole block into a file (e.g. markov_cea.py) and run:  python markov_cea.py
Requires: numpy   (pip install numpy)   All inputs below are FAKE illustrative values.
"""
import numpy as np


def markov_cohort(P: np.ndarray, state_costs: np.ndarray, state_utils: np.ndarray, cycles: int, start: np.ndarray,
                  discount: float = 0.03, program_cost_per_cycle: float = 0.0, half_cycle: bool = True) -> dict:
    """Run a discrete-time Markov cohort model and return discounted total cost and QALYs per person.

    Healthcare context
    ------------------
    Health-technology assessment (and some state Medicaid and payer reviews) ask for cost per QALY:
    a cohort moves between health states each cycle; each state has a cost and a quality-of-life
    weight. Comparing two strategies gives the incremental cost-effectiveness ratio (ICER).

    Parameters
    ----------
    P : (S, S) transition matrix per cycle (rows sum to 1); state_costs, state_utils : (S,) per cycle
    start : (S,) starting distribution; program_cost_per_cycle : added for people alive (not in the last, absorbing state)

    Steps
    -----
    1. trace[t+1] = trace[t] @ P for t = 0..cycles-1.
    2. Per-cycle cost/QALY = trace x values; discount by 1 / (1 + r)^t.
    3. Half-cycle correction: weight the first and last cycle by 1/2 (transitions happen mid-cycle on average).

    Common mistakes
    ---------------
    - Rows of P not summing to 1 (people leak out of the model).
    - Converting annual probabilities to monthly by division (use rates: p_m = 1 - (1 - p_y)^(1/12)).
    - Discounting only costs, or discounting from cycle 1 instead of cycle 0.
    """
    assert np.allclose(P.sum(axis=1), 1), "transition rows must sum to 1"
    trace = [np.asarray(start, float)]
    for _ in range(cycles):                                                                     # step 1
        trace.append(trace[-1] @ P)
    trace = np.array(trace)
    alive = 1 - trace[:, -1]
    costs = trace @ state_costs + program_cost_per_cycle * alive
    qalys = trace @ state_utils
    d = 1 / (1 + discount) ** np.arange(cycles + 1)                                             # step 2
    w = np.ones(cycles + 1)
    if half_cycle:                                                                              # step 3
        w[0] = w[-1] = 0.5
    return {"cost": float(np.sum(costs * d * w)), "qaly": float(np.sum(qalys * d * w)), "trace": trace}


def icer(new: dict, old: dict, wtp: float = 100_000) -> dict:
    """Incremental cost, QALYs, ICER and net monetary benefit (NMB = wtp x dQALY - dCost)."""
    dc, dq = new["cost"] - old["cost"], new["qaly"] - old["qaly"]
    if dq <= 0 and dc >= 0:
        verdict = "dominated (costs more, no health gain)"
    elif dq >= 0 and dc <= 0:
        verdict = "dominant (saves money and gains health)"
    else:
        verdict = "cost-effective" if dc / dq <= wtp else "not cost-effective at this WTP"
    return {"d_cost": dc, "d_qaly": dq, "icer": dc / dq if dq else float("inf"), "nmb": wtp * dq - dc, "verdict": verdict}


# ---------------------------------------------------------------------------
# Self-test: run `python markov_cea.py`. Every line should end in PASS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # states: community, nursing home, dead (annual cycles) - caregiver support delays placement (FAKE numbers)
    usual = np.array([[0.85, 0.10, 0.05], [0.00, 0.80, 0.20], [0, 0, 1.0]])
    program = np.array([[0.89, 0.06, 0.05], [0.00, 0.80, 0.20], [0, 0, 1.0]])
    costs, utils, start = np.array([20_000, 90_000, 0]), np.array([0.70, 0.45, 0]), np.array([1.0, 0, 0])
    a = markov_cohort(usual, costs, utils, 10, start)
    b = markov_cohort(program, costs, utils, 10, start, program_cost_per_cycle=3_000)
    r = icer(b, a)
    one = markov_cohort(usual, costs, utils, 1, start, discount=0.0)
    print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    checks = {
        "trace rows sum to 1 every cycle": np.allclose(a["trace"].sum(axis=1), 1),
        "1 cycle, no discount, half-cycle: cost = 0.5 x 20,000 + 0.5 x (0.85 x 20,000 + 0.10 x 90,000)":
            abs(one["cost"] - (0.5 * 20_000 + 0.5 * (0.85 * 20_000 + 0.10 * 90_000))) < 1e-6,
        "program gains QALYs": r["d_qaly"] > 0,
        "delaying nursing-home placement pays for the program here (dominant)": r["verdict"].startswith("dominant"),
        "NMB = WTP x dQALY - dCost": abs(r["nmb"] - (100_000 * r["d_qaly"] - r["d_cost"])) < 1e-6,
    }
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    assert all(checks.values()), "One or more self-tests failed"
