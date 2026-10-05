"""SparkSession factory: Databricks runtime or local Delta-enabled Spark."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from databricks_demo.config import LOCAL_ROOT, is_databricks


def get_spark() -> SparkSession:
    if is_databricks():
        # Databricks already created the session (with Delta + Unity Catalog).
        return SparkSession.builder.getOrCreate()
    return _local_spark()


def _local_spark() -> SparkSession:
    from delta import configure_spark_with_delta_pip

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    # Python UDF workers (e.g. the MLflow scoring UDF) must use this venv, not system python.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    metastore = LOCAL_ROOT / "metastore_db"
    builder = (
        SparkSession.builder.appName("databricks-demo")
        .master("local[*]")
        # Delta Lake: the storage layer Databricks tables are built on.
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.sources.default", "delta")
        # A persistent (Derby) metastore so tables survive between runs.
        .config("spark.sql.warehouse.dir", str(LOCAL_ROOT / "warehouse"))
        .config(
            "spark.hadoop.javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={metastore};create=true",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .enableHiveSupport()
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    # Delta tables don't fit Hive's column-type model, so the local metastore logs
    # harmless warnings and stack traces on every schema change. Delta's own log is
    # the source of truth, so silence them.
    log4j = spark._jvm.org.apache.logging.log4j
    for logger in (
        "org.apache.hadoop.hive.metastore.HiveAlterHandler",
        "org.apache.spark.sql.hive.HiveExternalCatalog",
        "org.apache.hadoop.hive.conf.HiveConf",
    ):
        log4j.core.config.Configurator.setLevel(logger, log4j.Level.OFF)
    return spark
