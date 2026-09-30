import pandas as pd
import pytest

from study_design_power import checks, methods


def test_block_randomization_balance_bound(units):
    r = methods.stratified_block_randomize(units, ["site_cd", "risk_tier"], (2, 4), seed=7)
    assert not checks.check_randomization_balance(r, max_block=4)
    counts = r.groupby(["stratum_id", "arm"]).size().unstack()
    assert ((counts["control"] - counts["treatment"]).abs() <= 2).all()
    assert r["arm"].notna().all() and len(r) == len(units)


def test_blocks_are_complete_and_balanced_inside(units):
    r = methods.stratified_block_randomize(units, ["site_cd"], (4,), seed=8)
    full = r.groupby("block_id").filter(lambda b: len(b) == 4)
    assert (full.groupby("block_id")["arm"].apply(lambda a: (a == "treatment").sum()) == 2).all()


def test_randomization_deterministic_and_seed_sensitive(units):
    a = methods.stratified_block_randomize(units, ["site_cd"], seed=1)["arm"]
    assert a.equals(methods.stratified_block_randomize(units, ["site_cd"], seed=1)["arm"])
    assert not a.equals(methods.stratified_block_randomize(units, ["site_cd"], seed=2)["arm"])


def test_balance_check_catches_bad_list(units):
    bad = units.assign(stratum_id=0, arm=["treatment"] * 200 + ["control"] * 100)
    assert checks.check_randomization_balance(bad)[0].check_id == "PWR-003"


def test_sap_renders_and_refuses_blanks(cfg, tmp_path):
    tpl = tmp_path / "t.md"
    tpl.write_text("# $study_title\nn = $n_final per arm\n", encoding="utf-8")
    assert methods.render_sap(tpl, {"study_title": "X", "n_final": "1,200"}) == "# X\nn = 1,200 per arm\n"
    with pytest.raises(KeyError, match="n_final"):
        methods.render_sap(tpl, {"study_title": "X"})


def test_shipped_sap_template_has_every_section(cfg):
    text = cfg.sap_template.read_text(encoding="utf-8")
    for h in ("Objective", "Design", "Endpoints", "Sample size justification", "Analysis methods", "Missing data",
              "Multiplicity", "Subgroups", "Interim"):
        assert h in text


def test_checks():
    assert checks.check_underpowered(0.6)[0].check_id == "PWR-001" and not checks.check_underpowered(0.85)
    assert checks.check_cluster_design("practice", 1.0)[0].severity == "error"
    assert not checks.check_cluster_design("patient", 1.0) and not checks.check_cluster_design("practice", 1.8)
    assert checks.check_few_clusters(5) and not checks.check_few_clusters(20)
    assert checks.check_multiplicity(3, False) and not checks.check_multiplicity(3, True)
    assert checks.check_skewed_outcome(2.5) and not checks.check_skewed_outcome(0.4)
