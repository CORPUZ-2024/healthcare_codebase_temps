-- SQL twin of methods.pmpm_ratio_of_sums(by=["lob_cd"])  (DuckDB dialect)
-- Inputs : claims (member_id, month_start DATE, paid_amt)
--          member_months (member_id, month_start DATE, lob_cd, member_months)
-- Output : lob_cd, paid_amt, member_months, pmpm
-- Key idea: aggregate numerator and denominator SEPARATELY, then join. Joining claims to
-- member_months first and summing member_months would count exposure once per claim.
WITH exposure AS (
    SELECT lob_cd, SUM(member_months) AS member_months
    FROM member_months
    GROUP BY lob_cd
),
spend AS (
    SELECT mm.lob_cd, SUM(c.paid_amt) AS paid_amt
    FROM claims c
    JOIN (SELECT DISTINCT member_id, month_start, lob_cd FROM member_months) mm
      ON c.member_id = mm.member_id AND c.month_start = mm.month_start
    GROUP BY mm.lob_cd
)
SELECT e.lob_cd,
       COALESCE(s.paid_amt, 0)                    AS paid_amt,
       e.member_months,
       COALESCE(s.paid_amt, 0) / e.member_months  AS pmpm
FROM exposure e
LEFT JOIN spend s USING (lob_cd)
ORDER BY e.lob_cd;
