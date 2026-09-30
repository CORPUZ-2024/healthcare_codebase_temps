-- Rate funnel from the member-level table produced by methods.evaluate_measure. DuckDB dialect.
-- Input view: member_level (measure_id, eligible_flag, exclusion_flag, denominator_flag, numerator_flag)
SELECT measure_id,
       SUM(eligible_flag)                                        AS eligible_cnt,
       SUM(exclusion_flag)                                       AS exclusion_cnt,
       SUM(denominator_flag)                                     AS denominator_cnt,
       SUM(numerator_flag)                                       AS numerator_cnt,
       SUM(numerator_flag)::DOUBLE / NULLIF(SUM(denominator_flag), 0) AS rate
FROM member_level
GROUP BY measure_id
ORDER BY measure_id
