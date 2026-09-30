-- SQL twin of methods.monthly_rates(event_type="ED_VISIT")  (DuckDB dialect)
-- Inputs : claims (member_id, svc_from_dt, service_category)   -- latest versions, categorized
--          member_months (member_id, month_start DATE, member_months)
-- An ED "visit" = distinct member + service date on an ED claim.
WITH ed_visits AS (
    SELECT DISTINCT member_id, CAST(svc_from_dt AS DATE) AS visit_dt
    FROM claims
    WHERE service_category = 'ED'
),
num AS (
    SELECT date_trunc('month', visit_dt) AS month_start, COUNT(*) AS events
    FROM ed_visits
    GROUP BY 1
),
den AS (
    SELECT month_start, SUM(member_months) AS member_months
    FROM member_months
    GROUP BY 1
)
SELECT d.month_start,
       COALESCE(n.events, 0)                               AS events,
       d.member_months,
       COALESCE(n.events, 0) / d.member_months * 12000     AS rate_per_1000
FROM den d
LEFT JOIN num n ON n.month_start = d.month_start
ORDER BY d.month_start;
