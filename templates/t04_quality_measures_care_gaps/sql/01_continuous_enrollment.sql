-- SQL twin of methods.continuous_enrollment (one CE period). DuckDB dialect.
-- Parameters: {ce_start} {ce_end} {anchor_dt} {allowable_gaps} {max_gap_days}
-- Dialect notes: Snowflake/BigQuery -> DATEDIFF('day', a, b) / DATE_DIFF(b, a, DAY).
-- Gaps-and-islands with a running MAX(end) so overlapping spans never create or hide a gap.
WITH clipped AS (
    SELECT member_id,
           GREATEST(enroll_start_dt, TIMESTAMP '{ce_start}') AS s,
           LEAST(enroll_end_dt, TIMESTAMP '{ce_end}')        AS e
    FROM enrollment
    WHERE enroll_end_dt >= TIMESTAMP '{ce_start}' AND enroll_start_dt <= TIMESTAMP '{ce_end}'
),
ordered AS (
    SELECT *,
           MAX(e) OVER (PARTITION BY member_id ORDER BY s, e
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_e
    FROM clipped
),
gaps AS (
    SELECT member_id,
           CASE WHEN prev_max_e IS NULL THEN date_diff('day', TIMESTAMP '{ce_start}', s)
                ELSE date_diff('day', prev_max_e, s) - 1 END AS gap_days
    FROM ordered
    UNION ALL
    SELECT member_id, date_diff('day', MAX(e), TIMESTAMP '{ce_end}') FROM clipped GROUP BY member_id
),
per_member AS (
    SELECT member_id,
           SUM(CASE WHEN gap_days > 0 THEN 1 ELSE 0 END)::INTEGER AS gap_cnt,
           MAX(GREATEST(gap_days, 0))::INTEGER                   AS max_gap_days
    FROM gaps GROUP BY member_id
),
anchor AS (
    SELECT member_id, MAX(CASE WHEN s <= TIMESTAMP '{anchor_dt}' AND e >= TIMESTAMP '{anchor_dt}' THEN 1 ELSE 0 END) AS anchor_enrolled_flag
    FROM clipped GROUP BY member_id
)
SELECT p.member_id, p.gap_cnt, p.max_gap_days, a.anchor_enrolled_flag,
       CASE WHEN p.gap_cnt <= {allowable_gaps} AND p.max_gap_days <= {max_gap_days}
                 AND a.anchor_enrolled_flag = 1 THEN 1 ELSE 0 END AS ce_flag
FROM per_member p JOIN anchor a USING (member_id)
ORDER BY p.member_id
