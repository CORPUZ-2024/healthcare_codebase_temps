-- Why "PMPM by paid month" and "PMPM by incurred month" disagree. DuckDB dialect.
-- Inputs: claims(incurred_month_start, paid_month_start, paid_amt), exposure(incurred_month_start, member_months)
WITH inc AS (SELECT incurred_month_start AS month_start, SUM(paid_amt) AS incurred_basis_paid FROM claims GROUP BY 1),
     pay AS (SELECT paid_month_start     AS month_start, SUM(paid_amt) AS paid_basis_paid     FROM claims GROUP BY 1)
SELECT e.incurred_month_start AS month_start, e.member_months,
       inc.incurred_basis_paid / e.member_months AS pmpm_incurred_basis,   -- low in recent months (runout)
       pay.paid_basis_paid     / e.member_months AS pmpm_paid_basis        -- smooth, but mixes service months
FROM exposure e
LEFT JOIN inc ON inc.month_start = e.incurred_month_start
LEFT JOIN pay ON pay.month_start = e.incurred_month_start
ORDER BY 1
