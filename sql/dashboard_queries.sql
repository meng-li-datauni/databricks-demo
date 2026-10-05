-- Databricks SQL: queries for an AI/BI dashboard on a SQL warehouse.
-- Replace main.databricks_demo with your catalog.schema (dev target: main.<you>_databricks_demo).

-- 1. Revenue trend by week and category
SELECT date_trunc('WEEK', order_date) AS week, category, ROUND(SUM(revenue), 2) AS revenue
FROM main.databricks_demo.daily_sales_by_category
GROUP BY ALL
ORDER BY week, category;

-- 2. KPI tiles
SELECT
  ROUND(SUM(amount), 2)                                AS total_revenue,
  COUNT(DISTINCT order_id)                             AS orders,
  ROUND(SUM(amount) / COUNT(DISTINCT order_id), 2)     AS avg_order_value,
  ROUND(AVG(CASE WHEN coupon_code IS NOT NULL THEN 1 ELSE 0 END) * 100, 1) AS pct_with_coupon
FROM main.databricks_demo.orders_silver
WHERE status <> 'cancelled';

-- 3. Channel mix per month (PIVOT)
SELECT * FROM (
  SELECT date_format(order_date, 'yyyy-MM') AS month, channel, amount
  FROM main.databricks_demo.orders_silver
  WHERE status <> 'cancelled'
)
PIVOT (ROUND(SUM(amount), 2) FOR channel IN ('web', 'mobile', 'store'))
ORDER BY month;

-- 4. Top 3 customers per country (QUALIFY filters on a window function)
SELECT country, customer_id, tier, total_spend
FROM main.databricks_demo.customer_features
QUALIFY spend_rank_in_country <= 3
ORDER BY country, total_spend DESC;

-- 5. Data-quality summary
SELECT _dq_reason, COUNT(*) AS rows
FROM main.databricks_demo.orders_quarantine
GROUP BY ALL;

-- 6. Expectation results from the declarative pipeline's event log
SELECT
  timestamp,
  e.dataset,
  e.name AS expectation,
  e.passed_records,
  e.failed_records
FROM event_log(TABLE(main.databricks_demo_dlt.orders_silver)),
     LATERAL explode(from_json(details:flow_progress.data_quality.expectations,
       'array<struct<name:string,dataset:string,passed_records:bigint,failed_records:bigint>>')) AS t(e)
WHERE event_type = 'flow_progress'
ORDER BY timestamp DESC;

-- 7-8 change who sees what: run them to try fine-grained access control, then DROP them.

-- 7. A row filter: non-admins only see EU customers (Unity Catalog fine-grained access control)
CREATE OR REPLACE FUNCTION main.databricks_demo.eu_only(country STRING)
RETURN is_account_group_member('admins') OR country IN ('DE', 'FR', 'NL', 'SE');

ALTER TABLE main.databricks_demo.customers_silver SET ROW FILTER main.databricks_demo.eu_only ON (country);
-- ALTER TABLE main.databricks_demo.customers_silver DROP ROW FILTER;

-- 8. A column mask: hide emails from everyone outside the 'support' group
CREATE OR REPLACE FUNCTION main.databricks_demo.mask_email(email STRING)
RETURN CASE WHEN is_account_group_member('support') THEN email ELSE regexp_replace(email, '^[^@]+', '***') END;

ALTER TABLE main.databricks_demo.customers_silver ALTER COLUMN email SET MASK main.databricks_demo.mask_email;
-- ALTER TABLE main.databricks_demo.customers_silver ALTER COLUMN email DROP MASK;
