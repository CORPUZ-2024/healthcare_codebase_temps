"""Analytic power vs. textbook formulas and vs. simulation; clustering; skewed cost."""
import math

import numpy as np
import pytest
from scipy import stats

from study_design_power import methods


def test_two_proportions_matches_arcsine_formula():
    p1, p2 = 0.20, 0.15
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    assert methods.power_two_proportions(p1, p2)["n_per_arm"] == math.ceil(2 * (z / h) ** 2)


def test_two_proportions_close_to_pooled_normal_formula():
    """Classic pooled-variance formula (Fleiss, no continuity correction) agrees within 1%."""
    p1, p2 = 0.20, 0.15
    pbar = (p1 + p2) / 2
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
    assert methods.power_two_proportions(p1, p2)["n_per_arm"] == pytest.approx(n, rel=0.01)


def test_power_and_n_are_inverse():
    n = methods.power_two_proportions(0.18, 0.14)["n_per_arm"]
    assert methods.power_two_proportions(0.18, 0.14, power=None, n_per_arm=n)["power"] == pytest.approx(0.80, abs=0.002)
    assert methods.power_two_means(4, 14, power=None, n_per_arm=methods.power_two_means(4, 14)["n_per_arm"])["power"] >= 0.80


def test_mde_inverts_sample_size():
    n = methods.power_two_proportions(0.18, 0.14)["n_per_arm"]
    assert methods.mde_two_proportions(0.18, n) == pytest.approx(0.14, abs=0.001)
    assert methods.mde_two_means(14.0, methods.power_two_means(4.0, 14.0)["n_per_arm"]) == pytest.approx(4.0, abs=0.05)


def test_smaller_n_bigger_mde():
    assert methods.mde_two_proportions(0.18, 400) < methods.mde_two_proportions(0.18, 1_600)   # reductions: lower p2


def test_simulation_agrees_with_analytic(cfg):
    n = methods.power_two_proportions(cfg.p_control, cfg.p_treat)["n_per_arm"]
    s = methods.simulate_power(methods.sim_two_proportions(cfg.p_control, cfg.p_treat, n), 1_500, seed=3)
    assert s["power"] == pytest.approx(0.80, abs=3 * s["mc_se"] + 0.01)


def test_simulation_type_one_error(cfg):
    s = methods.simulate_power(methods.sim_two_proportions(0.18, 0.18, 800), 2_000, seed=4)
    assert s["power"] == pytest.approx(0.05, abs=0.015)


def test_design_effect_and_cluster_size():
    assert methods.design_effect(1, 0.3) == 1.0                  # clusters of 1 = individuals
    c = methods.cluster_sample_size(1_000, 40, 0.02)
    assert c["clusters_per_arm"] == math.ceil(1_000 * 1.78 / 40)
    assert c["effective_n_per_arm"] >= 1_000


def test_ignoring_clustering_loses_power(cfg):
    n = methods.power_two_proportions(cfg.p_control, cfg.p_treat)["n_per_arm"]
    right = methods.cluster_sample_size(n, cfg.cluster_size, cfg.icc)["clusters_per_arm"]
    naive = math.ceil(n / cfg.cluster_size)
    p_right = methods.simulate_power(methods.sim_cluster_proportions(cfg.p_control, cfg.p_treat, right, cfg.cluster_size, cfg.icc), 600, seed=5)
    p_naive = methods.simulate_power(methods.sim_cluster_proportions(cfg.p_control, cfg.p_treat, naive, cfg.cluster_size, cfg.icc), 600, seed=5)
    assert p_right["power"] > 0.72 and p_naive["power"] < 0.65


def test_cluster_simulator_has_the_stated_icc():
    """One-way ANOVA estimator of ICC on simulated clusters recovers the input."""
    rng = np.random.default_rng(0)
    icc, m, k, p = 0.05, 50, 2_000, 0.3
    ab = (1 - icc) / icc
    y = rng.binomial(m, rng.beta(p * ab, (1 - p) * ab, k)) / m
    msb = m * y.var(ddof=1)
    msw = (y * (1 - y)).mean() * m / (m - 1)
    assert (msb - msw) / (msb + (m - 1) * msw) == pytest.approx(icc, abs=0.01)


def test_skewed_cost_normal_formula_is_optimistic(cfg):
    """For CV 2.5 cost, the t-test formula claims 80% but simulation of the same analysis gives less."""
    sd_change = cfg.cost_cv * cfg.cost_pmpm * math.sqrt(2 * (1 - 0.5))
    n = methods.power_two_means(cfg.cost_effect_pct * cfg.cost_pmpm, sd_change)["n_per_arm"]
    s = methods.simulate_power(methods.sim_cost_did(cfg.cost_pmpm, cfg.cost_cv, cfg.cost_effect_pct, n), 300, seed=6)
    assert s["power"] < 0.76


def test_power_curve_monotone():
    c = methods.power_curve([100, 400, 900], lambda n: methods.power_two_means(4, 14, power=None, n_per_arm=n)["power"])
    assert c["power"].is_monotonic_increasing
