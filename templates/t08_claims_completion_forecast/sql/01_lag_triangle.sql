-- SQL twin of methods.build_triangle (long format; pivot in the BI tool). DuckDB dialect.
-- Input: claims(incurred_month_start, paid_month_start, paid_amt).  Parameter: {as_of} (data cut date)
-- Dialect notes: Snowflake DATEDIFF('month', a, b); BigQuery DATE_DIFF(b, a, MONTH) - both count
-- calendar-month boundaries, which is exactly the lag definition (Jan 31 -> Feb 1 = lag 1).
SELECT incurred_month_start,
       date_diff('month', incurred_month_start, paid_month_start) AS lag,
       SUM(paid_amt)                                              AS paid_amt
FROM claims
WHERE paid_month_start <= DATE_TRUNC('month', TIMESTAMP '{as_of}')
GROUP BY 1, 2
ORDER BY 1, 2
