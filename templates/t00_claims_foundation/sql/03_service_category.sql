-- SQL twin of methods.service_category_claim  (DuckDB dialect)
-- Input: claims (latest version). Output: claim_id, service_category (one per claim).
WITH lines AS (
    SELECT claim_id,
        CASE
            WHEN tob_cd LIKE '11%'                                         THEN 1  -- IP
            WHEN rev_cd LIKE '045%' OR hcpcs_cd IN ('99281','99282','99283','99284','99285') THEN 2  -- ED
            WHEN tob_cd LIKE '13%'                                         THEN 3  -- OP
            WHEN hcpcs_cd IN ('T1019','S5125','S5130','T1020')             THEN 4  -- HCBS
            WHEN pos_cd = '12'                                             THEN 5  -- PROF_HOME
            WHEN pos_cd IN ('02','10')                                     THEN 6  -- TELEHEALTH
            WHEN claim_type_cd = 'PROF'                                    THEN 7  -- PROF
            ELSE 8                                                                 -- OTHER
        END AS cat_rank
    FROM claims
)
SELECT claim_id,
       CASE MIN(cat_rank) WHEN 1 THEN 'IP' WHEN 2 THEN 'ED' WHEN 3 THEN 'OP' WHEN 4 THEN 'HCBS'
            WHEN 5 THEN 'PROF_HOME' WHEN 6 THEN 'TELEHEALTH' WHEN 7 THEN 'PROF' ELSE 'OTHER' END AS service_category
FROM lines
GROUP BY claim_id
ORDER BY claim_id;
