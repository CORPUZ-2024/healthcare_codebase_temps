-- Staging: paid claims per member-month (already completed; see t08 for completion).
SELECT CAST(member_key AS VARCHAR)            AS member_id,
       CAST(month_start AS DATE)              AS month_start_dt,
       CAST(paid AS DOUBLE)                   AS paid_amt,
       CAST(period AS VARCHAR)                AS period_cd         -- pre / post relative to assignment
FROM {{ source('raw_claims_monthly') }}
