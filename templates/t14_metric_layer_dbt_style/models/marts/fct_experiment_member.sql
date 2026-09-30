-- Mart: one row per randomized member with every metric input. Members never reached get 0 flags,
-- not NULL (they are in the denominator of an intention-to-treat readout).
SELECT a.member_id,
       a.arm_cd,
       m.lob_cd,
       m.age,
       COALESCE(o.attempt_cnt, 0)    AS attempt_cnt,
       COALESCE(o.engaged_flag, 0)   AS engaged_flag,
       COALESCE(o.complaint_flag, 0) AS complaint_flag,
       COALESCE(o.opt_out_flag, 0)   AS opt_out_flag,
       c.pre_pmpm,
       c.post_pmpm
FROM {{ ref('stg_assignments') }} a
JOIN {{ ref('stg_members') }} m USING (member_id)
LEFT JOIN {{ ref('int_member_outreach') }} o USING (member_id)
LEFT JOIN {{ ref('int_member_cost') }} c USING (member_id)
