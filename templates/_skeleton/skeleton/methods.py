"""Analysis methods: a STANDARD approach and an ALTERNATIVE approach."""
import numpy as np
import pandas as pd


def mean_with_ci(x: pd.Series, z: float = 1.96) -> dict:
    """STANDARD: mean with a normal-approximation 95% CI.

    Caveat: with heavily skewed cost data and small n, the normal CI can be too narrow.
    """
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return {"estimate": m, "lo": m - z * se, "hi": m + z * se}


def mean_with_bootstrap_ci(x: pd.Series, n_boot: int = 2_000, seed: int = 0) -> dict:
    """ALTERNATIVE: percentile bootstrap CI — no normality assumption, slower."""
    rng = np.random.default_rng(seed)
    vals = x.to_numpy()
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return {"estimate": vals.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)}
