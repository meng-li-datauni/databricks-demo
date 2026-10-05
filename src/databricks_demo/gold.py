"""Gold layer: business-level aggregates for BI dashboards and ML features."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from databricks_demo.config import DemoConfig, build_parser
from databricks_demo.spark import get_spark


def _completed(orders: DataFrame, products: DataFrame) -> DataFrame:
    return (
        orders.filter(F.col("status") != "cancelled")
        .join(products.select("product_id", "category"), "product_id", "left")
        .withColumn("category", F.coalesce("category", F.lit("Unknown")))
    )


def daily_sales_by_category(orders: DataFrame, products: DataFrame) -> DataFrame:
    return (
        _completed(orders, products)
        .groupBy("order_date", "category")
        .agg(
            F.sum("amount").alias("revenue"),
            F.countDistinct("order_id").alias("orders"),
            F.sum("quantity").alias("units"),
            F.countDistinct("customer_id").alias("customers"),
        )
    )


def customer_features(orders: DataFrame, products: DataFrame, customers: DataFrame) -> DataFrame:
    """One row per current customer: RFM-style behaviour features + ``tier`` label."""
    completed = _completed(orders, products)
    as_of = completed.agg(F.max("order_date").alias("as_of_date"))
    per_customer = completed.groupBy("customer_id").agg(
        F.countDistinct("order_id").alias("num_orders"),
        F.sum("amount").alias("total_spend"),
        F.countDistinct("category").alias("distinct_categories"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )
    spend_rank = Window.partitionBy("country").orderBy(F.col("total_spend").desc())
    return (
        customers.select("customer_id", "country", "signup_date", "tier")
        .join(per_customer, "customer_id", "left")
        .crossJoin(F.broadcast(as_of))
        .withColumn("num_orders", F.coalesce("num_orders", F.lit(0)))
        .withColumn("total_spend", F.coalesce("total_spend", F.lit(0)).cast("decimal(14,2)"))
        .withColumn("distinct_categories", F.coalesce("distinct_categories", F.lit(0)))
        .withColumn(
            "avg_order_value",
            F.when(F.col("num_orders") > 0, F.round(F.col("total_spend") / F.col("num_orders"), 2)).otherwise(0),
        )
        .withColumn(
            "days_since_last_order",
            F.datediff("as_of_date", F.coalesce("last_order_date", "signup_date")),
        )
        .withColumn("spend_rank_in_country", F.dense_rank().over(spend_rank))
        .drop("as_of_date")
    )


def _overwrite(df: DataFrame, table: str) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    orders = spark.table(cfg.table("orders_silver"))
    products = spark.table(cfg.table("products_silver"))
    customers = spark.table(cfg.table("customers_silver"))

    _overwrite(daily_sales_by_category(orders, products), cfg.table("daily_sales_by_category"))
    _overwrite(customer_features(orders, products, customers), cfg.table("customer_features"))

    spark.sql(
        f"""
        SELECT category, ROUND(SUM(revenue), 2) AS revenue, SUM(orders) AS orders
        FROM {cfg.table('daily_sales_by_category')}
        GROUP BY category ORDER BY revenue DESC
        """
    ).show()


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
