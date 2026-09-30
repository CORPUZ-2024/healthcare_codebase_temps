-- Intermediate: pre- and post-assignment PMPM per member (ratio of sums within member).
SELECT member_id,
       SUM(CASE WHEN period_cd = 'pre'  THEN paid_amt END) / NULLIF(SUM(CASE WHEN period_cd = 'pre'  THEN 1 END), 0) AS pre_pmpm,
       SUM(CASE WHEN period_cd = 'post' THEN paid_amt END) / NULLIF(SUM(CASE WHEN period_cd = 'post' THEN 1 END), 0) AS post_pmpm
FROM {{ ref('stg_claims_monthly') }}
GROUP BY member_id
