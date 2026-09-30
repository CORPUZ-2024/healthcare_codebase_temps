"""Synthetic data generator + loader for the public test file.

Public test data: <name the source, URL, and which columns map to which> (see data/README.md).
"""
import numpy as np
import pandas as pd


def generate(n: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Return one row per member with a right-skewed annual cost.

    Steps
    -----
    1. Create member ids M000001...
    2. Draw cost from a lognormal (healthcare cost is right-skewed).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "member_id": [f"M{i:06d}" for i in range(1, n + 1)],
        "cost_amt": rng.lognormal(mean=7.5, sigma=1.2, size=n).round(2),
    })
