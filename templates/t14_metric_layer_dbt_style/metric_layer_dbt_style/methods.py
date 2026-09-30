"""
Metrics from ONE definition file, and an experiment readout with guardrails.

    metric_table_sql (generated from metrics.yaml, runs on DuckDB)   (STANDARD)
        vs  metric_table_pandas (same YAML, no warehouse)             (ALTERNATIVE)
    experiment_readout: two-proportion z / Welch t + guardrails        (STANDARD)
        vs  cuped_effect (pre-period covariate, lower variance)        (ALTERNATIVE)
    + srm_test (sample ratio mismatch), load_metrics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_metrics(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Metric tables
# ---------------------------------------------------------------------------

def metrics_sql(spec: dict) -> str:
    """Generate the SQL for every metric by group from metrics.yaml (what a semantic layer does).

    >>> print(metrics_sql({"model": "m", "group_by": "g", "metrics": [{"name": "r", "column": "c", "type": "mean"}]}))
    SELECT g, COUNT(*) AS n_units, AVG(CAST(c AS DOUBLE)) AS r FROM m GROUP BY g ORDER BY g
    """
    cols = ", ".join(f"AVG(CAST({m['column']} AS DOUBLE)) AS {m['name']}" for m in spec["metrics"])
    return f"SELECT {spec['group_by']}, COUNT(*) AS n_units, {cols} FROM {spec['model']} GROUP BY {spec['group_by']} ORDER BY {spec['group_by']}"


def metric_table_sql(con, spec: dict) -> pd.DataFrame:
    """STANDARD: run the generated SQL where the data live (DuckDB here; Snowflake/BigQuery unchanged)."""
    return con.execute(metrics_sql(spec)).df()


def metric_table_pandas(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """ALTERNATIVE: the same metrics from the same YAML, computed in pandas.

    Trade-off
    ---------
    + No warehouse needed (notebooks, small extracts, unit tests of the definitions themselves).
    - Pulls row-level data out of the governed layer; two engines can drift (NULL handling, integer
      division) - tests assert both give identical numbers.
    """
    g = df.groupby(spec["group_by"])
    out = g.size().rename("n_units").to_frame()
    for m in spec["metrics"]:
        out[m["name"]] = g[m["column"]].mean()
    return out.reset_index().sort_values(spec["group_by"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Experiment readout
# ---------------------------------------------------------------------------

def srm_test(counts: dict, expected_share: float = 0.5) -> dict:
    """Sample ratio mismatch: chi-square test that arm sizes match the design (e.g. 50/50).

    A tiny p-value (< 0.001) means members were lost or double counted in one arm (a logging or
    join bug); no effect estimate from such an experiment is trustworthy.

    >>> round(srm_test({"control": 5000, "treatment": 5000})["p_value"], 3)
    1.0
    """
    c, t = counts["control"], counts["treatment"]
    n = c + t
    exp = np.array([n * (1 - expected_share), n * expected_share])
    chi2 = float((((np.array([c, t]) - exp) ** 2) / exp).sum())
    return {"chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1)), "treatment_share": t / n}


def _diff(x_t: np.ndarray, x_c: np.ndarray, kind: str, alpha: float) -> dict:
    z = stats.norm.ppf(1 - alpha / 2)
    mt, mc = x_t.mean(), x_c.mean()
    if kind == "proportion":                                   # two-proportion z (unpooled SE for the CI)
        se = np.sqrt(mt * (1 - mt) / len(x_t) + mc * (1 - mc) / len(x_c))
        pool = (x_t.sum() + x_c.sum()) / (len(x_t) + len(x_c))
        se0 = np.sqrt(pool * (1 - pool) * (1 / len(x_t) + 1 / len(x_c)))
        p = 2 * stats.norm.sf(abs(mt - mc) / se0) if se0 > 0 else 1.0
    else:                                                      # Welch t
        r = stats.ttest_ind(x_t, x_c, equal_var=False)
        se, p = np.sqrt(x_t.var(ddof=1) / len(x_t) + x_c.var(ddof=1) / len(x_c)), float(r.pvalue)
    d = mt - mc
    return {"control": float(mc), "treatment": float(mt), "diff": float(d), "ci_lo": float(d - z * se),
            "ci_hi": float(d + z * se), "p_value": float(p), "se": float(se)}


def experiment_readout(df: pd.DataFrame, spec: dict, alpha: float = 0.05) -> pd.DataFrame:
    """STANDARD: effect per metric with CI and p-value; guardrails judged on the harm side.

    Healthcare context
    ------------------
    An outreach script that lifts engagement but raises complaints or opt-outs can cost more than it
    earns (complaints feed CAHPS / Star ratings; opt-outs shrink the reachable population). Guardrail
    rule here: FAIL if the upper confidence bound of the harm exceeds ``max_harm`` - i.e. we cannot
    rule out an unacceptable worsening (non-inferiority logic), not merely "p < 0.05 for harm".

    Returns one row per metric: name, role, control, treatment, diff, ci_lo, ci_hi, p_value, verdict.

    Common mistakes
    ---------------
    - Declaring a guardrail "fine" because its p-value is > 0.05 (absence of evidence).
    - Peeking daily and stopping at the first p < 0.05 (inflates false positives; pre-register the n).
    - Ignoring a sample ratio mismatch.
    """
    g, ct, tr = spec["group_by"], spec["control"], spec["treatment"]
    rows = []
    for m in spec["metrics"]:
        xt = df.loc[df[g] == tr, m["column"]].dropna().to_numpy(float)
        xc = df.loc[df[g] == ct, m["column"]].dropna().to_numpy(float)
        r = _diff(xt, xc, m["type"], alpha)
        good_sign = 1 if m.get("direction", "higher_is_better") == "higher_is_better" else -1
        if m["role"] == "guardrail":
            worst_harm = -good_sign * (r["ci_lo"] if good_sign == 1 else r["ci_hi"])      # upper bound of worsening
            verdict = "FAIL (cannot rule out harm)" if worst_harm > m["max_harm"] else "PASS"
        else:
            sig = r["p_value"] < alpha
            verdict = ("WIN" if good_sign * r["diff"] > 0 else "LOSS") if sig else "no detectable effect"
        rows.append({"metric": m["name"], "role": m["role"], **r, "verdict": verdict})
    return pd.DataFrame(rows)


def cuped_effect(df: pd.DataFrame, y: str, x: str, group: str = "arm_cd", treatment: str = "treatment",
                 control: str = "control", alpha: float = 0.05) -> dict:
    """ALTERNATIVE: CUPED - subtract the part of the outcome predicted by a PRE-experiment covariate.

    Trade-off
    ---------
    + Same unbiased effect, smaller variance: Var reduced by ~corr(y, x)^2. With noisy cost outcomes
      this is often the difference between "no detectable effect" and a clear result.
    - The covariate must be measured BEFORE assignment (a post-assignment covariate can absorb the effect);
      one more step to explain; no gain when pre and post are weakly correlated.

    theta = cov(y, x) / var(x) on the pooled data; y_adj = y - theta (x - mean x); Welch t on y_adj.
    Returns dict: diff, ci_lo, ci_hi, p_value, se, theta, variance_reduction.
    """
    d = df[[group, y, x]].dropna()
    theta = float(np.cov(d[y], d[x], ddof=1)[0, 1] / d[x].var(ddof=1))
    adj = d[y] - theta * (d[x] - d[x].mean())
    t, c = adj[d[group] == treatment].to_numpy(), adj[d[group] == control].to_numpy()
    r = _diff(t, c, "mean", alpha)
    raw = _diff(d.loc[d[group] == treatment, y].to_numpy(), d.loc[d[group] == control, y].to_numpy(), "mean", alpha)
    return {**{k: r[k] for k in ("diff", "ci_lo", "ci_hi", "p_value", "se")}, "theta": theta,
            "variance_reduction": float(1 - (r["se"] / raw["se"]) ** 2)}
