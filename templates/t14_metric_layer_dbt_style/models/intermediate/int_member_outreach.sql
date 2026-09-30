-- Intermediate: outreach outcomes rolled up to one row per member.
SELECT member_id,
       COUNT(*)                                                        AS attempt_cnt,
       MAX(CASE WHEN outcome_cd = 'engaged'   THEN 1 ELSE 0 END)       AS engaged_flag,
       MAX(CASE WHEN outcome_cd = 'complaint' THEN 1 ELSE 0 END)       AS complaint_flag,
       MAX(CASE WHEN outcome_cd = 'opt_out'   THEN 1 ELSE 0 END)       AS opt_out_flag
FROM {{ ref('stg_outreach') }}
GROUP BY member_id
