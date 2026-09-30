-- SQL twin of methods.member_periods (sums only). DuckDB dialect.
-- Inputs: panel(member_id, month_idx, member_months, paid_amt), members(member_id, index_month_idx, treated_flag)
-- Parameters: {pre_months} {post_months} {min_months}
-- The index month (rel = 0) is excluded on purpose: it holds the referral trigger.
WITH rel AS (
    SELECT p.member_id, m.treated_flag, p.month_idx - m.index_month_idx AS rel, p.member_months, p.paid_amt
    FROM panel p JOIN members m USING (member_id)
),
per AS (
    SELECT member_id, treated_flag,
           SUM(CASE WHEN rel BETWEEN -{pre_months} AND -1 THEN paid_amt END)      AS pre_paid_amt,
           SUM(CASE WHEN rel BETWEEN -{pre_months} AND -1 THEN member_months END) AS pre_mm,
           SUM(CASE WHEN rel BETWEEN 1 AND {post_months} THEN paid_amt END)       AS post_paid_amt,
           SUM(CASE WHEN rel BETWEEN 1 AND {post_months} THEN member_months END)  AS post_mm
    FROM rel GROUP BY member_id, treated_flag
)
SELECT *, pre_paid_amt / pre_mm AS pre_pmpm, post_paid_amt / post_mm AS post_pmpm
FROM per
WHERE pre_mm >= {min_months} AND post_mm >= {min_months}
ORDER BY member_id
