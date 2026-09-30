"""dbt_lite engine, schema tests, metric parity, experiment readout, CUPED, SRM."""
import numpy as np
import pandas as pd
import pytest

from metric_layer_dbt_style import checks, data, dbt_lite, methods

duckdb = pytest.importorskip("duckdb")


# --- engine --------------------------------------------------------------------------------------

def test_build_order_respects_refs_and_layers(built):
    o = built["order"]
    assert o.index("stg_outreach") < o.index("int_member_outreach") < o.index("fct_experiment_member")
    assert all(o.index(s) < o.index("fct_experiment_member") for s in o if s.startswith("stg_"))


def test_unknown_ref_and_cycle_are_errors():
    with pytest.raises(KeyError, match="nope"):
        dbt_lite.build_order({"a": {"refs": ["nope"], "layer": "marts"}})
    with pytest.raises(ValueError, match="cycle"):
        dbt_lite.build_order({"a": {"refs": ["b"], "layer": "marts"}, "b": {"refs": ["a"], "layer": "marts"}})


def test_layering_rule(tmp_path):
    (tmp_path / "marts").mkdir()
    (tmp_path / "marts" / "bad.sql").write_text("SELECT * FROM {{ source('raw_x') }}", encoding="utf-8")
    with pytest.raises(ValueError, match="marts must go through staging"):
        dbt_lite.build(duckdb.connect(), tmp_path, {"raw_x": pd.DataFrame({"a": [1]})})


def test_clean_data_passes_every_schema_test(built):
    assert all(t["status"] == "PASS" for t in built["tests"]) and len(built["tests"]) == 11


def test_injected_defects_are_caught(cfg):
    con = duckdb.connect()
    dbt_lite.build(con, cfg.models_dir, data.generate(3_000, inject_issues=True, seed=1))
    failed = {t["test"] for t in dbt_lite.run_tests(con, cfg.schema_yml) if t["status"] == "FAIL"}
    assert {"unique:stg_members.member_id", "accepted_values:stg_members.lob_cd", "unique:stg_assignments.member_id",
            "relationships:stg_assignments.member_id->stg_members.member_id", "accepted_values:stg_assignments.arm_cd"} <= failed


def test_staging_normalizes_messy_codes(built):
    lob = built["con"].execute("SELECT DISTINCT lob_cd FROM stg_members ORDER BY 1").df()["lob_cd"].tolist()
    assert lob == ["DUAL", "MCD", "MCR"]
    assert set(built["fct"]["arm_cd"]) == {"control", "treatment"}


def test_unreached_members_stay_in_the_denominator(built):
    """ITT: members with no outreach rows are in the mart with engaged_flag 0 (not dropped, not NULL)."""
    raw = built["raw"]
    reached = raw["raw_outreach"]["member_key"].nunique()
    assert len(built["fct"]) == len(raw["raw_assignments"]) > reached
    assert built["fct"]["engaged_flag"].notna().all()


# --- metrics -------------------------------------------------------------------------------------

def test_sql_and_pandas_engines_agree(built):
    s = methods.metric_table_sql(built["con"], built["spec"])
    p = methods.metric_table_pandas(built["fct"], built["spec"])
    assert not checks.check_engine_parity(s, p)
    assert s["n_units"].tolist() == [10_000, 10_000]


def test_parity_check_catches_a_drifted_definition(built):
    s = methods.metric_table_sql(built["con"], built["spec"])
    drift = s.copy()
    drift.loc[0, "engagement_rate"] += 0.001
    assert checks.check_engine_parity(s, drift)[0].check_id == "MET-004"


# --- readout -------------------------------------------------------------------------------------

def test_readout_detects_true_engagement_lift(built, cfg):
    ro = methods.experiment_readout(built["fct"], built["spec"]).set_index("metric")
    r = ro.loc["engagement_rate"]
    assert r["verdict"] == "WIN" and r["ci_lo"] <= cfg.engagement_lift <= r["ci_hi"]


def test_two_proportion_matches_textbook():
    df = pd.DataFrame({"arm_cd": ["treatment"] * 1000 + ["control"] * 1000, "y": [1] * 250 + [0] * 750 + [1] * 200 + [0] * 800})
    spec = {"group_by": "arm_cd", "control": "control", "treatment": "treatment",
            "metrics": [{"name": "r", "column": "y", "type": "proportion", "role": "primary"}]}
    r = methods.experiment_readout(df, spec).iloc[0]
    pool = 0.225
    z = 0.05 / np.sqrt(pool * (1 - pool) * 2 / 1000)
    from scipy import stats
    assert r["diff"] == pytest.approx(0.05) and r["p_value"] == pytest.approx(2 * stats.norm.sf(z))


def test_guardrail_noninferiority_logic():
    """Harm point estimate small but CI upper bound beyond the margin -> FAIL (can't rule out harm)."""
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({"arm_cd": ["treatment"] * n + ["control"] * n,
                       "c": np.r_[rng.random(n) < 0.012, rng.random(n) < 0.010].astype(int)})
    spec = {"group_by": "arm_cd", "control": "control", "treatment": "treatment",
            "metrics": [{"name": "complaints", "column": "c", "type": "proportion", "role": "guardrail",
                         "direction": "lower_is_better", "max_harm": 0.005}]}
    ro = methods.experiment_readout(df, spec)
    assert ro["verdict"].iloc[0].startswith("FAIL") and ro["p_value"].iloc[0] > 0.05
    assert checks.check_guardrails(ro)[0].check_id == "MET-003"


def test_cuped_reduces_variance_and_stays_unbiased(built, cfg):
    cu = methods.cuped_effect(built["fct"], "post_pmpm", "pre_pmpm")
    assert cu["variance_reduction"] > 0.4
    diffs = []
    for s in range(12):                                   # no true effect -> CUPED differences centre on 0
        con = duckdb.connect()
        dbt_lite.build(con, cfg.models_dir, data.generate(4_000, pmpm_effect_pct=0.0, seed=100 + s))
        fct = con.execute("SELECT * FROM fct_experiment_member").df()
        diffs.append(methods.cuped_effect(fct, "post_pmpm", "pre_pmpm")["diff"])
    assert abs(np.mean(diffs)) < 3 * np.std(diffs, ddof=1) / np.sqrt(len(diffs))


def test_srm(cfg):
    assert methods.srm_test({"control": 10_000, "treatment": 10_000})["p_value"] == pytest.approx(1.0)
    con = duckdb.connect()
    dbt_lite.build(con, cfg.models_dir, data.generate(cfg.n_members, srm_drop=0.05, seed=cfg.seed))
    counts = con.execute("SELECT arm_cd, COUNT(*) AS n FROM fct_experiment_member GROUP BY 1").df().set_index("arm_cd")["n"].to_dict()
    assert checks.check_srm(methods.srm_test(counts))[0].check_id == "MET-002"


def test_schema_check_finding():
    assert checks.check_schema_tests([{"test": "unique:x.id", "failures": 3, "status": "FAIL"}])[0].n_rows == 1
    assert not checks.check_schema_tests([{"test": "unique:x.id", "failures": 0, "status": "PASS"}])
