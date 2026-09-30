-- SQL twin of methods.member_months_daily  (DuckDB dialect)
-- Input table : enrollment (member_id, enroll_start_dt, enroll_end_dt inclusive)
-- Params      : {period_start}, {period_end}  ('YYYY-MM-DD', substituted by sqltwin.run_sql)
-- Pattern     : "gaps and islands" to merge overlapping spans, then prorate by days in month.
WITH ordered AS (
    SELECT member_id, enroll_start_dt, enroll_end_dt,
           MAX(enroll_end_dt) OVER (
               PARTITION BY member_id ORDER BY enroll_start_dt, enroll_end_dt
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
    FROM enrollment
),
flagged AS (   -- 1 = this span starts a new island (no overlap/adjacency with anything earlier)
    SELECT *, CASE WHEN prev_max_end IS NULL
                     OR enroll_start_dt > prev_max_end + INTERVAL 1 DAY THEN 1 ELSE 0 END AS new_island
    FROM ordered
),
islands AS (
    SELECT *, SUM(new_island) OVER (PARTITION BY member_id ORDER BY enroll_start_dt, enroll_end_dt
                                    ROWS UNBOUNDED PRECEDING) AS island_id
    FROM flagged
),
merged AS (
    SELECT member_id,
           GREATEST(MIN(enroll_start_dt), DATE '{period_start}') AS s,
           LEAST(MAX(enroll_end_dt),   DATE '{period_end}')   AS e
    FROM islands
    GROUP BY member_id, island_id
),
months AS (
    SELECT CAST(m AS DATE) AS month_start, CAST(last_day(m) AS DATE) AS month_end
    FROM generate_series(DATE '{period_start}', DATE '{period_end}', INTERVAL 1 MONTH) AS t(m)
)
SELECT
    mg.member_id,
    mo.month_start,
    SUM(date_diff('day', GREATEST(mg.s, mo.month_start), LEAST(mg.e, mo.month_end)) + 1) AS days_enrolled,
    SUM(date_diff('day', GREATEST(mg.s, mo.month_start), LEAST(mg.e, mo.month_end)) + 1)
        / day(mo.month_end) AS member_months
FROM merged mg
JOIN months mo
  ON mg.s <= mo.month_end AND mg.e >= mo.month_start
WHERE mg.s <= mg.e
GROUP BY mg.member_id, mo.month_start, mo.month_end
ORDER BY mg.member_id, mo.month_start;
