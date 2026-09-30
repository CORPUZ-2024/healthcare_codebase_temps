import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from metric_layer_dbt_style import data, dbt_lite, methods  # noqa: E402
from metric_layer_dbt_style.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def built(cfg):
    raw = data.generate(cfg.n_members, seed=cfg.seed)
    con = duckdb.connect()
    order = dbt_lite.build(con, cfg.models_dir, raw)
    spec = methods.load_metrics(cfg.metrics_yml)
    return {"con": con, "order": order, "raw": raw, "spec": spec,
            "fct": con.execute(f"SELECT * FROM {spec['model']}").df(), "tests": dbt_lite.run_tests(con, cfg.schema_yml)}
