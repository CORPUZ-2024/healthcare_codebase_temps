-- SQL twin of methods.readmissions_per_index (DuckDB dialect)
-- Input : stays (member_id, stay_id, admit_dt, discharge_dt)
-- LEAD() looks at the member's NEXT admission — the core window-function pattern for readmissions.
SELECT
    member_id,
    stay_id,
    admit_dt,
    discharge_dt,
    LEAD(admit_dt) OVER (PARTITION BY member_id ORDER BY admit_dt)                       AS next_admit_dt,
    date_diff('day', discharge_dt,
              LEAD(admit_dt) OVER (PARTITION BY member_id ORDER BY admit_dt))              AS days_to_readmit,
    CASE WHEN date_diff('day', discharge_dt,
              LEAD(admit_dt) OVER (PARTITION BY member_id ORDER BY admit_dt)) BETWEEN 1 AND {window_days}
         THEN 1 ELSE 0 END                                                                 AS readmit_flag
FROM stays
ORDER BY member_id, admit_dt;
