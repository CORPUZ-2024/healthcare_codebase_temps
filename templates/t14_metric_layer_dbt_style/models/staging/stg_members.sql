-- Staging: one row per member, typed and renamed. Source: raw_members (FAKE eligibility extract).
SELECT CAST(member_key AS VARCHAR)            AS member_id,
       UPPER(TRIM(lob))                       AS lob_cd,
       CAST(enroll_start AS DATE)             AS enroll_start_dt,
       CAST(age AS INTEGER)                   AS age
FROM {{ source('raw_members') }}
