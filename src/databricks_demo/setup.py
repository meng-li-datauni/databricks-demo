"""Create the schema (and, on Databricks, the Unity Catalog Volume) for the demo."""

from __future__ import annotations

from pyspark.sql import SparkSession

from databricks_demo.config import VOLUME_NAME, DemoConfig, build_parser, is_databricks
from databricks_demo.spark import get_spark


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.namespace} COMMENT 'Databricks demo lakehouse'")
    if is_databricks():
        # Volumes are governed storage for non-tabular files (landing zone, checkpoints).
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.namespace}.{VOLUME_NAME}")
        # Target schema of the Lakeflow Declarative Pipeline (resources/orders_pipeline.yml).
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.namespace}_dlt")
    print(f"Schema {cfg.namespace} ready; files under {cfg.volume_root}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
