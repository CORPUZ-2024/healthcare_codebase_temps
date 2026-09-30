-- Staging: experiment assignment (one row per member randomized).
SELECT CAST(member_key AS VARCHAR)            AS member_id,
       LOWER(TRIM(variant))                   AS arm_cd,
       CAST(assigned_at AS DATE)              AS assigned_dt
FROM {{ source('raw_assignments') }}
