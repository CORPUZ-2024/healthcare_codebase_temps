"""
t14 — dbt-style metric layer on DuckDB + outreach A/B experiment readout (synthetic data).

    python run.py              build models, run schema tests, metrics, readout -> outputs/
    python run.py --selftest   plain-assert checks (no pytest needed)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from metric_layer_dbt_style import checks, data, dbt_lite, methods  # noqa: E402
from metric_layer_dbt_style.config import Config  # noqa: E402

OUT = HERE / "outputs"


def build_and_test(cfg: Config, raw: dict) -> tuple:
    con = duckdb.connect()
    order = dbt_lite.build(con, cfg.models_dir, raw)
    return con, order, dbt_lite.run_tests(con, cfg.schema_yml)


def demo(cfg: Config) -> None:
    OUT.mkdir(exist_ok=True)
    raw = data.generate(cfg.n_members, cfg.engagement_control, cfg.engagement_lift, cfg.pmpm_effect_pct,
                        cfg.complaint_control, cfg.complaint_lift, seed=cfg.seed)
    con, order, tests = build_and_test(cfg, raw)
    print("== dbt_lite build order ==\n  " + " -> ".join(order))
    print(f"== schema tests: {sum(t['status'] == 'PASS' for t in tests)}/{len(tests)} PASS ==")

    spec = methods.load_metrics(cfg.metrics_yml)
    sql_t = methods.metric_table_sql(con, spec)
    fct = con.execute(f"SELECT * FROM {spec['model']}").df()
    pd_t = methods.metric_table_pandas(fct, spec)
    print("\n== Metrics by arm (STANDARD: SQL generated from metrics.yaml) ==")
    print(sql_t.round(4).to_string(index=False))
    fnd = checks.check_engine_parity(sql_t, pd_t)
    print("ALTERNATIVE pandas engine: identical" if not fnd else "ALTERNATIVE pandas engine: MISMATCH")

    srm = methods.srm_test(fct[spec["group_by"]].value_counts().to_dict())
    ro = methods.experiment_readout(fct, spec, cfg.alpha)
    print(f"\n== Experiment readout (SRM p = {srm['p_value']:.2f}) ==")
    print(ro[["metric", "role", "control", "treatment", "diff", "ci_lo", "ci_hi", "p_value", "verdict"]].round(4).to_string(index=False))
    cu = methods.cuped_effect(fct, "post_pmpm", "pre_pmpm")
    raw_pm = ro.set_index("metric").loc["post_pmpm"]
    print(f"\nALTERNATIVE CUPED on post_pmpm (covariate pre_pmpm, theta {cu['theta']:.2f}): diff {cu['diff']:.1f} "
          f"(95% CI {cu['ci_lo']:.1f} to {cu['ci_hi']:.1f}, p = {cu['p_value']:.3f}); variance reduced {cu['variance_reduction']:.0%} "
          f"vs raw CI {raw_pm['ci_lo']:.1f} to {raw_pm['ci_hi']:.1f} (p = {raw_pm['p_value']:.3f})")

    print("\n== What the tests catch: the same pipeline on a raw extract with 4 injected defects ==")
    bad_raw = data.generate(4_000, inject_issues=True, seed=cfg.seed)
    _, _, bad_tests = build_and_test(cfg, bad_raw)
    for t in bad_tests:
        if t["status"] == "FAIL":
            print(f"  FAIL {t['test']}: {t['failures']} row(s)")
    lost = data.generate(cfg.n_members, srm_drop=0.05, seed=cfg.seed)
    con2, _, _ = build_and_test(cfg, lost)
    srm_bad = methods.srm_test(con2.execute(f"SELECT {spec['group_by']}, COUNT(*) FROM {spec['model']} GROUP BY 1").df()
                               .set_index(spec["group_by"]).iloc[:, 0].to_dict())
    print(f"  5% of treatment members silently lost upstream -> SRM p = {srm_bad['p_value']:.1e}")

    print("\n== Checks ==")
    fnd += (checks.check_schema_tests(tests) + checks.check_srm(srm, cfg.srm_alpha) + checks.check_guardrails(ro)
            + checks.check_schema_tests(bad_tests) + checks.check_srm(srm_bad, cfg.srm_alpha))
    for f in fnd:
        print(f"[{f.severity}] {f.check_id}: {f.message} -> {f.fix}")
    print("(the MET-001 / MET-002 lines come from the defect scenarios, not the main run)")

    pd.DataFrame(tests).to_csv(OUT / "schema_tests.csv", index=False)
    sql_t.to_csv(OUT / "metrics_by_arm.csv", index=False)
    ro.to_csv(OUT / "experiment_readout.csv", index=False)
    (OUT / "compiled_metrics.sql").write_text(methods.metrics_sql(spec) + "\n", encoding="utf-8")
    print(f"\nOutputs written to {OUT}")


def selftest() -> int:
    models = {"a": {"refs": [], "layer": "staging"}, "b": {"refs": ["a"], "layer": "intermediate"},
              "c": {"refs": ["a", "b"], "layer": "marts"}}
    con = duckdb.connect()
    con.register("t", pd.DataFrame({"id": [1, 2, 2, None], "v": ["x", "y", "z", "x"]}))
    con.execute("CREATE VIEW m AS SELECT * FROM t")
    tests = {
        "build order a -> b -> c": lambda: dbt_lite.build_order(models) == ["a", "b", "c"],
        "ref() compiles to the relation name": lambda: dbt_lite.compile_sql("SELECT * FROM {{ ref('stg_x') }}") == "SELECT * FROM stg_x",
        "unique test finds the duplicated id": lambda: con.execute(f"SELECT COUNT(*) FROM ({dbt_lite._test_sql('m', 'id', 'unique')[1]})").fetchone()[0] == 1,
        "not_null test finds the NULL": lambda: con.execute(f"SELECT COUNT(*) FROM ({dbt_lite._test_sql('m', 'id', 'not_null')[1]})").fetchone()[0] == 1,
        "SRM flags 5,000 vs 4,700": lambda: methods.srm_test({"control": 5000, "treatment": 4700})["p_value"] < 0.01,
    }
    ok = True
    for name, fn in tests.items():
        passed = bool(fn())
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + name)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else (demo(Config()) or 0))
