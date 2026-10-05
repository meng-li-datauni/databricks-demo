"""Bronze layer: land raw files into Delta tables, incrementally and as-is.

On Databricks this uses **Auto Loader** (``cloudFiles``), which discovers new
files efficiently, infers the schema and evolves it when new columns appear.
Locally it falls back to Spark's built-in file streaming source with an
explicit schema. Either way, ``trigger(availableNow=True)`` processes whatever
is new and then stops, so a streaming query can run as a scheduled batch job.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from databricks_demo.config import DemoConfig, build_parser, is_databricks
from databricks_demo.spark import get_spark

# Only needed locally; Auto Loader infers (and evolves) these on Databricks.
ORDERS_SCHEMA = StructType(
    [
        StructField("order_id", StringType()),
        StructField("customer_id", StringType()),
        StructField("product_id", StringType()),
        StructField("quantity", LongType()),
        StructField("unit_price", DoubleType()),
        StructField("order_ts", StringType()),
        StructField("status", StringType()),
        StructField("channel", StringType()),
        StructField("coupon_code", StringType()),
    ]
)
CUSTOMERS_SCHEMA = StructType(
    [
        StructField("customer_id", StringType()),
        StructField("name", StringType()),
        StructField("email", StringType()),
        StructField("country", StringType()),
        StructField("signup_date", StringType()),
        StructField("tier", StringType()),
        StructField("operation", StringType()),
        StructField("updated_at", StringType()),
    ]
)


def read_landing_stream(spark: SparkSession, cfg: DemoConfig, source: str, schema: StructType) -> DataFrame:
    path = f"{cfg.landing_dir}/{source}"
    if is_databricks():
        reader = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation", f"{cfg.schema_dir}/bronze_{source}")
            .option("cloudFiles.inferColumnTypes", "true")
            # A new column stops the stream once; the restarted stream picks it up.
            # The job gives this task a retry for exactly that reason.
            .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        )
    else:
        reader = spark.readStream.format("json").schema(schema)
    return (
        reader.load(path)
        # File metadata column: lineage back to the exact source file.
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def ingest_stream(spark: SparkSession, cfg: DemoConfig, source: str, schema: StructType) -> None:
    target = cfg.table(f"{source}_bronze")
    query = (
        read_landing_stream(spark, cfg, source, schema)
        .writeStream.format("delta")
        .option("checkpointLocation", f"{cfg.checkpoint_dir}/bronze_{source}")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target)
    )
    query.awaitTermination()
    print(f"{target}: {spark.table(target).count()} rows")


def ingest_products(spark: SparkSession, cfg: DemoConfig) -> None:
    """Small reference data: a plain batch overwrite is fine."""
    target = cfg.table("products_bronze")
    (
        spark.read.json(f"{cfg.landing_dir}/products")
        .withColumn("_ingested_at", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target)
    )
    print(f"{target}: {spark.table(target).count()} rows")


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    ingest_products(spark, cfg)
    ingest_stream(spark, cfg, "customers", CUSTOMERS_SCHEMA)
    ingest_stream(spark, cfg, "orders", ORDERS_SCHEMA)


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
