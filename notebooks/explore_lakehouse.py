# Databricks notebook source
# MAGIC %md
# MAGIC # Explore the demo lakehouse
# MAGIC
# MAGIC Interactive companion to the `lakehouse_etl` job. Run the job (or `python main.py` locally) first.
# MAGIC
# MAGIC Shows: **widgets**, `display()`, mixing **Python and SQL** cells, **parameterized SQL**,
# MAGIC Unity Catalog **lineage / metadata**, `dbutils.fs` on Volumes, and **time travel**.

# COMMAND ----------

# Widgets become notebook parameters (and job task parameters when scheduled).
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "databricks_demo", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
spark.sql(f"USE {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md ## What's in the schema?

# COMMAND ----------

display(spark.sql("SHOW TABLES"))

# COMMAND ----------

# Files in the Unity Catalog Volume (the landing zone written by the generator)
display(dbutils.fs.ls(f"/Volumes/{catalog}/{schema}/demo/landing/orders"))

# COMMAND ----------

# MAGIC %md ## SQL cells read widget values as named parameter markers

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT order_date, category, revenue, orders
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.daily_sales_by_category')
# MAGIC ORDER BY order_date, revenue DESC

# COMMAND ----------

# MAGIC %md ## Top customers per country (window function)

# COMMAND ----------

top = spark.sql(
    """
    SELECT country, customer_id, tier, total_spend, num_orders
    FROM customer_features
    QUALIFY spend_rank_in_country <= 3
    ORDER BY country, total_spend DESC
    """
)
display(top)

# COMMAND ----------

# MAGIC %md ## Data quality: what got quarantined, and why?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT _dq_reason, count(*) AS rows FROM orders_quarantine GROUP BY ALL

# COMMAND ----------

# MAGIC %md ## Delta: history, time travel, change feed

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY orders_silver

# COMMAND ----------

# Python API equivalent of VERSION AS OF.
# v0 = CREATE TABLE, v1-v2 = ADD CONSTRAINT, v3 = first MERGE from the silver stream.
after_first_merge = spark.read.option("versionAsOf", 3).table("orders_silver")
print("rows at version 3:", after_first_merge.count(), "| rows now:", spark.table("orders_silver").count())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Customers changed by the CDC feed (requires delta.enableChangeDataFeed, set in silver.py)
# MAGIC SELECT customer_id, email, country, _change_type, _commit_version
# MAGIC FROM table_changes('customers_silver', 1)
# MAGIC WHERE _change_type <> 'update_preimage'
# MAGIC ORDER BY customer_id, _commit_version

# COMMAND ----------

# MAGIC %md ## Unity Catalog governance

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Column comments and tags show up in Catalog Explorer and help search / AI assistants
# MAGIC ALTER TABLE customers_silver ALTER COLUMN email COMMENT 'PII: customer email address';
# MAGIC ALTER TABLE customers_silver ALTER COLUMN email SET TAGS ('pii' = 'email');

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Lineage is captured automatically; query it from the system tables
# MAGIC SELECT source_table_full_name, target_table_full_name, entity_type, event_time
# MAGIC FROM system.access.table_lineage
# MAGIC WHERE target_table_schema = :schema
# MAGIC ORDER BY event_time DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model predictions vs. actual tier
# MAGIC The model is registered in Unity Catalog as `<catalog>.<schema>.customer_tier_model` with alias `@champion`.

# COMMAND ----------

display(spark.table("customer_tier_predictions").groupBy("tier", "premium_prediction").count())
