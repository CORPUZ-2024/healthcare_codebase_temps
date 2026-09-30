-- Staging: outreach attempts (many per member).
SELECT CAST(member_key AS VARCHAR)            AS member_id,
       CAST(attempt_ts AS DATE)               AS attempt_dt,
       LOWER(TRIM(outcome))                   AS outcome_cd        -- engaged / no_answer / declined / complaint / opt_out
FROM {{ source('raw_outreach') }}
