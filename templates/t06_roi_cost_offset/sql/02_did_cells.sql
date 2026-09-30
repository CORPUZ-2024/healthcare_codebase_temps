-- The four DiD cells as ratio-of-sums PMPM, from the matched member table. DuckDB dialect.
-- Input: matched(member_id, treated_flag, pre_paid_amt, pre_mm, post_paid_amt, post_mm)
WITH cells AS (
    SELECT treated_flag,
           SUM(pre_paid_amt) / SUM(pre_mm)   AS pre_pmpm,
           SUM(post_paid_amt) / SUM(post_mm) AS post_pmpm
    FROM matched GROUP BY treated_flag
)
SELECT t.pre_pmpm AS treated_pre, t.post_pmpm AS treated_post, c.pre_pmpm AS control_pre, c.post_pmpm AS control_post,
       (t.post_pmpm - t.pre_pmpm) - (c.post_pmpm - c.pre_pmpm) AS did_pmpm
FROM cells t CROSS JOIN cells c
WHERE t.treated_flag = 1 AND c.treated_flag = 0
