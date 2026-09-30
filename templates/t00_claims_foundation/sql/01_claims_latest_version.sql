-- SQL twin of methods.collapse_versions_latest  (DuckDB dialect)
-- Input table : claims  (claim_id, line_seq, adj_seq, claim_status_cd, ...)
-- Output      : one row per claim_id x line_seq, latest version, voided claims removed
-- Snowflake/BigQuery: identical except `SELECT * EXCLUDE (...)` is `SELECT * EXCEPT (...)` in BigQuery.
WITH ranked AS (
    SELECT
        *,
        MAX(adj_seq) OVER (PARTITION BY claim_id) AS latest_adj_seq   -- claim-level, not line-level
    FROM claims
)
SELECT * EXCLUDE (latest_adj_seq)
FROM ranked
WHERE adj_seq = latest_adj_seq
  AND claim_status_cd <> 'V'
ORDER BY claim_id, line_seq;
