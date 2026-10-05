"""Silver layer: typed, validated, deduplicated, upserted.

Patterns shown here:

* streaming **from** a Delta table (bronze) with ``foreachBatch``;
* Delta ``MERGE`` for idempotent upserts (re-delivered orders don't duplicate);
* SCD type 1 for customers, including CDC deletes;
* a quarantine table for rows that fail validation (instead of dropping them);
* table constraints (``NOT NULL``, ``CHECK``) and Change Data Feed.
"""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from databricks_demo.config import DemoConfig, build_parser
from databricks_demo.spark import get_spark

ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "unit_price",
    "amount",
    "status",
    "channel",
    "coupon_code",
    "order_ts",
    "order_date",
    "_ingested_at",
]
CUSTOMER_COLUMNS = ["customer_id", "name", "email", "country", "signup_date", "tier", "updated_at"]

ORDERS_DDL = """
CREATE TABLE {table} (
  order_id     STRING NOT NULL,
  customer_id  STRING NOT NULL,
  product_id   STRING,
  quantity     INT,
  unit_price   DECIMAL(10,2),
  amount       DECIMAL(12,2),
  status       STRING,
  channel      STRING,
  coupon_code  STRING,
  order_ts     TIMESTAMP,
  order_date   DATE,
  _ingested_at TIMESTAMP
) USING DELTA
COMMENT 'Validated, deduplicated orders'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
CUSTOMERS_DDL = """
CREATE TABLE {table} (
  customer_id STRING NOT NULL,
  name        STRING,
  email       STRING,
  country     STRING,
  signup_date DATE,
  tier        STRING,
  updated_at  TIMESTAMP
) USING DELTA
COMMENT 'Current customer state (SCD type 1)'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""


# ---------------------------------------------------------------- transforms
# Pure DataFrame -> DataFrame functions, unit-tested in tests/test_transforms.py.


def clean_orders(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return ``(valid, quarantine)``; quarantine rows carry a ``_dq_reason``."""
    if "coupon_code" not in df.columns:  # column only exists once a file contained it
        df = df.withColumn("coupon_code", F.lit(None).cast("string"))
    typed = (
        df.select(
            F.col("order_id").cast("string"),
            F.col("customer_id").cast("string"),
            F.col("product_id").cast("string"),
            F.col("quantity").cast("int"),
            F.col("unit_price").cast("decimal(10,2)"),
            F.lower("status").alias("status"),
            F.col("channel").cast("string"),
            F.col("coupon_code").cast("string"),
            F.to_timestamp("order_ts").alias("order_ts"),
            F.col("_ingested_at"),
        )
        .withColumn("amount", (F.col("quantity") * F.col("unit_price")).cast("decimal(12,2)"))
        .withColumn("order_date", F.to_date("order_ts"))
    )
    reason = (
        F.when(F.col("order_id").isNull(), "missing order_id")
        .when(F.col("customer_id").isNull(), "missing customer_id")
        .when(F.col("quantity").isNull() | (F.col("quantity") <= 0), "non-positive quantity")
        .when(F.col("order_ts").isNull(), "unparseable order_ts")
    )
    flagged = typed.withColumn("_dq_reason", reason)
    valid = flagged.filter(F.col("_dq_reason").isNull()).dropDuplicates(["order_id"]).select(*ORDER_COLUMNS)
    quarantine = flagged.filter(F.col("_dq_reason").isNotNull())
    return valid, quarantine


def latest_customer_changes(df: DataFrame) -> DataFrame:
    """Keep only the newest CDC record per customer (MERGE needs unique keys)."""
    newest_first = Window.partitionBy("customer_id").orderBy(F.col("updated_at").desc())
    return (
        df.withColumn("updated_at", F.to_timestamp("updated_at"))
        .withColumn("signup_date", F.to_date("signup_date"))
        .withColumn("_rn", F.row_number().over(newest_first))
        .filter("_rn = 1")
        .select(*CUSTOMER_COLUMNS, "operation")
    )


# ---------------------------------------------------------------- table writes


def _create_if_missing(spark: SparkSession, table: str, ddl: str, constraints: dict[str, str]) -> None:
    if spark.catalog.tableExists(table):
        return
    spark.sql(ddl.format(table=table))
    for name, check in constraints.items():
        spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({check})")


def _run_available_now(spark: SparkSession, cfg: DemoConfig, source: str, name: str, handler) -> None:
    query = (
        spark.readStream.table(cfg.table(source))
        .writeStream.foreachBatch(handler)
        .option("checkpointLocation", f"{cfg.checkpoint_dir}/{name}")
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()


def build_orders(spark: SparkSession, cfg: DemoConfig) -> None:
    target = cfg.table("orders_silver")
    quarantine = cfg.table("orders_quarantine")
    _create_if_missing(
        spark,
        target,
        ORDERS_DDL,
        {"positive_quantity": "quantity > 0", "valid_status": "status IN ('placed', 'shipped', 'cancelled')"},
    )

    def upsert(batch: DataFrame, batch_id: int) -> None:
        valid, bad = clean_orders(batch)
        (
            DeltaTable.forName(batch.sparkSession, target)
            .alias("t")
            .merge(valid.alias("s"), "t.order_id = s.order_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        bad.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(quarantine)

    _run_available_now(spark, cfg, "orders_bronze", "silver_orders", upsert)
    print(f"{target}: {spark.table(target).count()} rows")
    if spark.catalog.tableExists(quarantine):
        spark.table(quarantine).groupBy("_dq_reason").count().show(truncate=False)


def build_customers(spark: SparkSession, cfg: DemoConfig) -> None:
    target = cfg.table("customers_silver")
    _create_if_missing(spark, target, CUSTOMERS_DDL, {})

    def apply_cdc(batch: DataFrame, batch_id: int) -> None:
        changes = latest_customer_changes(batch)
        columns = {c: f"s.{c}" for c in CUSTOMER_COLUMNS}
        (
            DeltaTable.forName(batch.sparkSession, target)
            .alias("t")
            .merge(changes.alias("s"), "t.customer_id = s.customer_id")
            .whenMatchedDelete(condition="s.operation = 'DELETE'")
            .whenMatchedUpdate(condition="s.updated_at > t.updated_at", set=columns)
            .whenNotMatchedInsert(condition="s.operation <> 'DELETE'", values=columns)
            .execute()
        )

    _run_available_now(spark, cfg, "customers_bronze", "silver_customers", apply_cdc)
    print(f"{target}: {spark.table(target).count()} rows")


def build_products(spark: SparkSession, cfg: DemoConfig) -> None:
    target = cfg.table("products_silver")
    (
        spark.table(cfg.table("products_bronze"))
        .select("product_id", "name", "category", F.col("list_price").cast("decimal(10,2)"))
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target)
    )


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    build_products(spark, cfg)
    build_customers(spark, cfg)
    build_orders(spark, cfg)


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
