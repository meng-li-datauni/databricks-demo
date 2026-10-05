# Lakeflow Declarative Pipeline (formerly Delta Live Tables).
#
# The same bronze -> silver -> gold flow as src/databricks_demo, written
# declaratively: you define tables as queries, and the pipeline works out the
# dependency graph, checkpoints, retries, schema evolution and infrastructure.
#
# Only runs inside a Databricks pipeline (see resources/orders_pipeline.yml).
# `spark` is provided by the pipeline runtime.

import dlt
from pyspark.sql import functions as F

LANDING = spark.conf.get("demo.landing_path")  # noqa: F821 - set in the pipeline configuration


def auto_loader(source: str):
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{LANDING}/{source}")
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


# ------------------------------------------------------------------ bronze


@dlt.table(comment="Raw orders, ingested incrementally with Auto Loader")
def orders_bronze():
    return auto_loader("orders")


@dlt.table(comment="Raw customer change records (CDC feed)")
def customers_bronze():
    return auto_loader("customers")


@dlt.table(comment="Product reference data")
def products():
    return (
        spark.read.json(f"{LANDING}/products")  # noqa: F821
        .select("product_id", "name", "category", F.col("list_price").cast("decimal(10,2)"))
    )


# ------------------------------------------------------------------ silver
# Expectations are declarative data-quality rules. Their pass/fail counts are
# recorded in the pipeline event log and shown in the pipeline UI:
#   expect          -> keep the row, record the violation
#   expect_or_drop  -> drop the row
#   expect_or_fail  -> stop the update


@dlt.table(comment="Validated, deduplicated orders")
@dlt.expect("known_status", "status IN ('placed', 'shipped', 'cancelled')")
@dlt.expect_or_drop("has_customer", "customer_id IS NOT NULL")
@dlt.expect_or_drop("positive_quantity", "quantity > 0")
@dlt.expect_or_fail("has_order_id", "order_id IS NOT NULL")
def orders_silver():
    return (
        dlt.read_stream("orders_bronze")
        .select(
            "order_id",
            "customer_id",
            "product_id",
            F.col("quantity").cast("int"),
            F.col("unit_price").cast("decimal(10,2)"),
            (F.col("quantity") * F.col("unit_price")).cast("decimal(12,2)").alias("amount"),
            "status",
            "channel",
            F.to_timestamp("order_ts").alias("order_ts"),
        )
        .withColumn("order_date", F.to_date("order_ts"))
        # Bounded-state streaming dedup: duplicates arrive within a day of each other.
        .withWatermark("order_ts", "1 day")
        .dropDuplicatesWithinWatermark(["order_id"])
    )


@dlt.view(comment="Typed customer CDC records")
def customers_cdc():
    return (
        dlt.read_stream("customers_bronze")
        .withColumn("updated_at", F.to_timestamp("updated_at"))
        .withColumn("signup_date", F.to_date("signup_date"))
    )


# AUTO CDC builds a slowly changing dimension from a CDC feed, handling
# out-of-order records via `sequence_by`. SCD type 2 keeps full history with
# __START_AT / __END_AT columns (compare: the SCD type 1 MERGE in silver.py).
# On older runtimes this API is called dlt.apply_changes(...) with the same arguments.
dlt.create_streaming_table("customers_scd2", comment="Customer history (SCD type 2)")
dlt.create_auto_cdc_flow(
    target="customers_scd2",
    source="customers_cdc",
    keys=["customer_id"],
    sequence_by=F.col("updated_at"),
    apply_as_deletes=F.expr("operation = 'DELETE'"),
    except_column_list=["operation", "_source_file", "_rescued_data"],
    stored_as_scd_type=2,
)


# ------------------------------------------------------------------ gold
# Batch reads (dlt.read) make these materialized views: the pipeline keeps
# them up to date, incrementally where it can.


@dlt.table(comment="Daily revenue per category")
def daily_sales_by_category():
    return (
        dlt.read("orders_silver")
        .filter("status <> 'cancelled'")
        .join(dlt.read("products").select("product_id", "category"), "product_id", "left")
        .groupBy("order_date", "category")
        .agg(
            F.sum("amount").alias("revenue"),
            F.countDistinct("order_id").alias("orders"),
            F.countDistinct("customer_id").alias("customers"),
        )
    )


@dlt.table(comment="Revenue by country using the customer's *current* country")
def revenue_by_country():
    current_customers = dlt.read("customers_scd2").filter("__END_AT IS NULL")
    return (
        dlt.read("orders_silver")
        .filter("status <> 'cancelled'")
        .join(current_customers.select("customer_id", "country"), "customer_id")
        .groupBy("country")
        .agg(F.sum("amount").alias("revenue"), F.countDistinct("customer_id").alias("customers"))
    )
