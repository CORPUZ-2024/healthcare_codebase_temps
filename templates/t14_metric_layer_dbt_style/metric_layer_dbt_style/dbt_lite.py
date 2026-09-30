"""
dbt_lite: the smallest useful subset of dbt, on DuckDB, in one file.

    models/<layer>/<name>.sql     SELECT statements using {{ ref('model') }} and {{ source('table') }}
    models/schema.yml             generic tests per column: not_null, unique, accepted_values, relationships
    build(con, models_dir)        parse refs -> topological order -> CREATE OR REPLACE VIEW per model
    run_tests(con, schema_yml)    each test = a SQL query returning failing rows; 0 rows = pass

Why this shape: analysts get version-controlled, reviewable SQL with the SAME metric definitions for
every report, and data tests that fail loudly before a number reaches a dashboard. When a warehouse
and real dbt are available, these files move over almost unchanged (dbt uses the same Jinja calls).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REF = re.compile(r"\{\{\s*ref\(\s*['\"](\w+)['\"]\s*\)\s*\}\}")
SOURCE = re.compile(r"\{\{\s*source\(\s*['\"](\w+)['\"]\s*\)\s*\}\}")
LAYER_ORDER = {"staging": 0, "intermediate": 1, "marts": 2}


def discover(models_dir: str | Path) -> dict[str, dict]:
    """{model name: {'path', 'layer', 'sql', 'refs', 'sources'}} for every .sql file under models_dir."""
    out = {}
    for p in sorted(Path(models_dir).rglob("*.sql")):
        sql = p.read_text(encoding="utf-8")
        out[p.stem] = {"path": p, "layer": p.parent.name, "sql": sql, "refs": REF.findall(sql), "sources": SOURCE.findall(sql)}
    return out


def build_order(models: dict[str, dict]) -> list[str]:
    """Topological order of models by their refs (Kahn's algorithm). Raises on unknown refs or cycles.

    >>> build_order({"b": {"refs": ["a"], "layer": "marts"}, "a": {"refs": [], "layer": "staging"}})
    ['a', 'b']
    """
    missing = {r for m in models.values() for r in m["refs"] if r not in models}
    if missing:
        raise KeyError(f"ref() to unknown model(s): {sorted(missing)}")
    remaining = {k: set(v["refs"]) for k, v in models.items()}
    order = []
    while remaining:
        ready = sorted((k for k, deps in remaining.items() if not deps),
                       key=lambda k: (LAYER_ORDER.get(models[k]["layer"], 9), k))
        if not ready:
            raise ValueError(f"dependency cycle among: {sorted(remaining)}")
        for k in ready:
            order.append(k)
            del remaining[k]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def compile_sql(sql: str) -> str:
    """Replace {{ ref('x') }} and {{ source('x') }} with the bare relation name."""
    return SOURCE.sub(lambda m: m.group(1), REF.sub(lambda m: m.group(1), sql))


def build(con, models_dir: str | Path, sources: dict | None = None) -> list[str]:
    """Register ``sources`` (name -> DataFrame) and create one view per model in dependency order.

    Returns the build order. A layering rule is enforced: staging models may only read sources,
    marts may not read sources directly.
    """
    for name, df in (sources or {}).items():
        con.register(name, df)
    models = discover(models_dir)
    for name, m in models.items():
        if m["layer"] == "staging" and m["refs"]:
            raise ValueError(f"{name}: staging models read sources only")
        if m["layer"] == "marts" and m["sources"]:
            raise ValueError(f"{name}: marts must go through staging (no source())")
    order = build_order(models)
    for name in order:
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {compile_sql(models[name]['sql'])}")
    return order


def _test_sql(model: str, column: str, test) -> tuple[str, str]:
    """(test id, SQL returning the failing rows) for one generic test."""
    if test == "not_null":
        return f"not_null:{model}.{column}", f"SELECT * FROM {model} WHERE {column} IS NULL"
    if test == "unique":
        return f"unique:{model}.{column}", f"SELECT {column}, COUNT(*) AS n FROM {model} GROUP BY {column} HAVING COUNT(*) > 1"
    (kind, args), = test.items()
    if kind == "accepted_values":
        vals = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in args["values"])
        return f"accepted_values:{model}.{column}", f"SELECT * FROM {model} WHERE {column} NOT IN ({vals}) AND {column} IS NOT NULL"
    if kind == "relationships":
        return (f"relationships:{model}.{column}->{args['to']}.{args['field']}",
                f"SELECT c.* FROM {model} c LEFT JOIN {args['to']} p ON c.{column} = p.{args['field']} "
                f"WHERE c.{column} IS NOT NULL AND p.{args['field']} IS NULL")
    raise ValueError(f"unknown test {kind}")


def run_tests(con, schema_yml: str | Path) -> list[dict]:
    """Run every schema test; returns [{'test', 'failures', 'status'}] (status PASS / FAIL)."""
    spec = yaml.safe_load(Path(schema_yml).read_text(encoding="utf-8"))
    results = []
    for m in spec.get("models", []):
        for col in m.get("columns", []):
            for t in col.get("tests", []):
                tid, sql = _test_sql(m["name"], col["name"], t)
                n = con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
                results.append({"test": tid, "failures": int(n), "status": "PASS" if n == 0 else "FAIL"})
    return results
