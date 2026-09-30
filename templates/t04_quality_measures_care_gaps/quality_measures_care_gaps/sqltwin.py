"""COPIED from t05 (templates never import each other).
Run the SQL twin files in sql/ against pandas DataFrames using DuckDB (no server needed).

Usage::

    from quality_measures_care_gaps.sqltwin import run_sql
    run_sql("01_continuous_enrollment.sql", {"enrollment": enr_df}, ce_start="2025-01-01", ...)

DuckDB can query a pandas DataFrame directly once it is registered as a view. The same SQL
runs on Snowflake/BigQuery with the dialect notes written at the top of each file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def run_sql(filename: str, tables: dict[str, pd.DataFrame], **params) -> pd.DataFrame:
    """Execute sql/<filename> with ``tables`` registered as views and ``{params}`` substituted."""
    import duckdb  # imported here so the rest of the template works without duckdb installed

    sql = (SQL_DIR / filename).read_text(encoding="utf-8").format(**params)
    con = duckdb.connect()
    try:
        for name, df in tables.items():
            con.register(name, df)
        return con.execute(sql).df()
    finally:
        con.close()
