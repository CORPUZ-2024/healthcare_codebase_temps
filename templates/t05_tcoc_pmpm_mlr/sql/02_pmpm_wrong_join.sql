-- ANTI-PATTERN kept on purpose (used by tests/test_t05_sql_twin.py to show the bug).
-- Joining claims to member_months and THEN summing member_months counts a member-month
-- once per claim line -> denominator inflated -> PMPM far too low.
SELECT mm.lob_cd,
       SUM(c.paid_amt)                          AS paid_amt,
       SUM(mm.member_months)                    AS member_months,
       SUM(c.paid_amt) / SUM(mm.member_months)  AS pmpm
FROM member_months mm
LEFT JOIN claims c
  ON c.member_id = mm.member_id AND c.month_start = mm.month_start
GROUP BY mm.lob_cd
ORDER BY mm.lob_cd;
