"""A guided tour of Delta Lake features on the tables the pipeline just built.

Covers: transaction log / history, time travel, Change Data Feed, schema
enforcement vs. evolution, DML (UPDATE/DELETE), RESTORE, OPTIMIZE + Z-ORDER
(or liquid clustering on Databricks), VACUUM and table details.
Destructive steps run against a scratch ``delta_playground`` table.
"""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from databricks_demo.config import DemoConfig, build_parser, is_databricks
from databricks_demo.spark import get_spark


def _section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def _current_version(spark: SparkSession, table: str) -> int:
    return DeltaTable.forName(spark, table).history(1).first()["version"]


def history_and_time_travel(spark: SparkSession, orders: str) -> None:
    _section("History: every write is a versioned commit in the _delta_log")
    history = spark.sql(f"DESCRIBE HISTORY {orders}")
    history.select("version", "timestamp", "operation").orderBy("version").show(truncate=False)

    _section("Time travel: query the table as of any earlier version")
    for row in history.select("version").orderBy("version").collect():
        n = spark.sql(f"SELECT count(*) AS n FROM {orders} VERSION AS OF {row.version}").first().n
        print(f"  version {row.version}: {n} rows")


def change_data_feed(spark: SparkSession, customers: str) -> None:
    _section("Change Data Feed: row-level inserts / updates / deletes")
    spark.sql(
        f"""
        SELECT _commit_version, _change_type, count(*) AS rows
        FROM table_changes('{customers}', 0)
        GROUP BY ALL ORDER BY _commit_version, _change_type
        """
    ).show()


def schema_enforcement_and_evolution(spark: SparkSession, playground: str) -> None:
    sample = spark.table(playground).limit(5).withColumn("loyalty_points", F.lit(100))

    _section("Schema enforcement: unexpected columns are rejected")
    try:
        sample.write.format("delta").mode("append").saveAsTable(playground)
    except Exception as e:  # AnalysisException; its class path differs across runtimes
        print(f"  rejected as expected: {str(e).splitlines()[0][:120]}")

    _section("Schema evolution: opt in with mergeSchema")
    sample.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(playground)
    print(f"  columns now: {spark.table(playground).columns}")


def dml_and_restore(spark: SparkSession, playground: str, baseline: int) -> None:
    _section("DML: UPDATE and DELETE are ACID transactions")
    spark.sql(f"UPDATE {playground} SET status = 'shipped' WHERE status = 'placed'")
    spark.sql(f"DELETE FROM {playground} WHERE channel = 'store'")
    print(f"  rows after DML: {spark.table(playground).count()}")

    _section(f"RESTORE: roll the whole table back to version {baseline}")
    spark.sql(f"RESTORE TABLE {playground} TO VERSION AS OF {baseline}")
    print(f"  rows after restore: {spark.table(playground).count()}")


def layout_optimization(spark: SparkSession, orders: str, playground: str) -> None:
    _section("OPTIMIZE: compact small files (+ Z-ORDER to co-locate by a column)")
    spark.sql(f"OPTIMIZE {orders} ZORDER BY (customer_id)").select("metrics.numFilesAdded", "metrics.numFilesRemoved").show()

    if is_databricks():
        _section("Liquid clustering (Databricks): replaces partitioning and Z-ORDER")
        spark.sql(f"ALTER TABLE {playground} CLUSTER BY (customer_id)")
        spark.sql(f"OPTIMIZE {playground}")
        spark.sql(f"DESCRIBE DETAIL {playground}").select("clusteringColumns").show(truncate=False)

    _section("VACUUM: list files older than the retention window (dry run)")
    spark.sql(f"VACUUM {orders} RETAIN 168 HOURS DRY RUN").show(5, truncate=False)

    _section("DESCRIBE DETAIL: physical table metadata")
    spark.sql(f"DESCRIBE DETAIL {orders}").select("format", "numFiles", "sizeInBytes", "properties").show(truncate=False)


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    orders = cfg.table("orders_silver")
    playground = cfg.table("delta_playground")

    history_and_time_travel(spark, orders)
    change_data_feed(spark, cfg.table("customers_silver"))

    spark.sql(f"CREATE OR REPLACE TABLE {playground} USING DELTA AS SELECT * FROM {orders} LIMIT 500")
    baseline = _current_version(spark, playground)
    schema_enforcement_and_evolution(spark, playground)
    dml_and_restore(spark, playground, baseline)
    layout_optimization(spark, orders, playground)


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
