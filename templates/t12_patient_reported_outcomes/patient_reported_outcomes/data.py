"""
Instrument specs (YAML) and a synthetic caregiver-support trial with item-level responses.

Trial: family caregivers randomized 1:1 to a support program or usual care; the FAKE 12-item burden
scale (primary) and the PHQ-9 (secondary) at baseline, 3 and 6 months.

What makes it realistic (and testable):
  * item-level answers from a latent burden trajectory; three reverse-worded burden items
  * ~3% of items skipped at random (tests the prorating rule)
  * dropout that is MISSING AT RANDOM: caregivers whose burden has RISEN since baseline (at their last
    observed visit) are more likely to miss the next one -> completers look better than everyone;
    likelihood-based mixed models (which use the earlier visits) are not fooled. Dropout that depends
    only on the LEVEL of burden barely biases change scores (tested during the build).
  * the truth: the effect computed on the complete data before any missingness is imposed

Public test data: MEPS SF-12 / PHQ-2 / K6 items (AHRQ) - see data/README.md and ``load_meps_pro``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_instrument(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_instruments(folder: str | Path) -> dict[str, dict]:
    specs = [load_instrument(p) for p in sorted(Path(folder).glob("*.yaml"))]
    return {s["instrument_id"]: s for s in specs}


def _items_from_latent(rng, latent_total, spec, spread=0.8):
    """Item answers whose (reverse-corrected) sum tracks the latent total."""
    k, lo, hi = len(spec["items"]), spec["item_min"], spec["item_max"]
    offsets = np.linspace(-0.4, 0.4, k)
    out = {}
    for j, item in enumerate(spec["items"]):
        expected = latent_total / k + offsets[j]
        x = np.clip(np.rint(expected + rng.normal(0, spread, len(latent_total))), lo, hi)
        out[item] = (hi + lo - x) if item in spec.get("reverse_items", []) else x    # stored as ANSWERED
    return out


def generate_trial(burden_spec: dict, phq_spec: dict, n: int = 600, visits=(0, 3, 6), true_effect_6m: float = -4.0,
                   item_missing_rate: float = 0.03, seed: int = 12) -> dict:
    """Long data: caregiver_id, arm_flag, visit_month, <burden items>, <phq items>, observed_flag.

    Returns {'long': all rows incl. dropped visits (items NaN when observed_flag = 0),
             'complete': the same rows before any missingness, 'true_effect_6m': float (score scale)}.
    """
    rng = np.random.default_rng(seed)
    arm = (np.arange(n) % 2).astype(int)
    rng.shuffle(arm)
    base = rng.normal(28, 7, n)
    slope_re = rng.normal(0, 5.0, n)                              # people diverge: month 3 predicts month 6
    ctrl = {0: 0.0, 3: -1.0, 6: -1.5}
    trt = {0: 0.0, 3: -1.0 + true_effect_6m * 0.6, 6: -1.5 + true_effect_6m}
    rows, complete = [], []
    for v in visits:
        latent = base + np.where(arm == 1, trt[v], ctrl[v]) + slope_re * v / 6 + rng.normal(0, 2, n)
        latent = np.clip(latent, 0, 48)
        dep = np.clip(0.35 * latent - 1 + rng.normal(0, 2.5, n), 0, 27)
        items = {**_items_from_latent(rng, latent, burden_spec), **_items_from_latent(rng, dep, phq_spec, 0.6)}
        frame = pd.DataFrame({"caregiver_id": [f"G{i:04d}" for i in range(n)], "arm_flag": arm, "visit_month": v, **items})
        complete.append(frame)
    comp = pd.concat(complete, ignore_index=True)

    # --- missingness ---------------------------------------------------------------------------------
    long = comp.copy()
    item_cols = burden_spec["items"] + phq_spec["items"]
    mask = rng.random((len(long), len(item_cols))) < item_missing_rate
    long[item_cols] = long[item_cols].mask(mask)
    long["observed_flag"] = 1
    raw_total = comp[burden_spec["items"]].copy()
    for it in burden_spec.get("reverse_items", []):
        raw_total[it] = burden_spec["item_max"] + burden_spec["item_min"] - raw_total[it]
    comp_total = raw_total.sum(axis=1)
    prev_ok = np.ones(n, dtype=bool)
    base_score = comp_total[comp["visit_month"] == visits[0]].to_numpy()
    for prev, v in zip(visits[:-1], visits[1:]):                     # monotone MAR dropout: worsening since baseline
        prev_score = comp_total[comp["visit_month"] == prev].to_numpy()
        p = 1 / (1 + np.exp(-(-2.2 + 0.30 * (prev_score - base_score))))
        drop = prev_ok & (rng.random(n) < p)
        prev_ok = prev_ok & ~drop
        idx = long.index[long["visit_month"] == v]
        long.loc[idx[~prev_ok], item_cols] = np.nan
        long.loc[idx[~prev_ok], "observed_flag"] = 0

    c6 = comp_total[comp["visit_month"] == visits[-1]].to_numpy() - comp_total[comp["visit_month"] == visits[0]].to_numpy()
    true_eff = float(c6[arm == 1].mean() - c6[arm == 0].mean())
    return {"long": long, "complete": comp, "true_effect_6m": true_eff}


# ---------------------------------------------------------------------------
# Public file: MEPS PRO items
# ---------------------------------------------------------------------------
MEPS_PRO_COLS = ["DUPERSID", "PCS42", "MCS42", "PHQ242", "K6SUM42"]


def load_meps_pro(path: str | Path) -> pd.DataFrame:
    """Load SF-12 summary scores (PCS42, MCS42), PHQ-2 (PHQ242) and K6 (K6SUM42) from a MEPS FYC file.

    Negative values are MEPS reserved codes (-1 inapplicable, -7 refused, -8 don't know, -9 not
    ascertained) and become NaN - never average them. Column names carry the survey round ("42");
    verify against your year's codebook. Reads .dta, .xlsx or .csv.
    """
    p = Path(path)
    if p.suffix.lower() == ".dta":
        df = pd.read_stata(p, columns=[c for c in MEPS_PRO_COLS], convert_categoricals=False)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, dtype={"DUPERSID": str})
    df = df[[c for c in MEPS_PRO_COLS if c in df.columns]].copy()
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda s: s >= 0)
    return df.rename(columns={"DUPERSID": "person_id"})
