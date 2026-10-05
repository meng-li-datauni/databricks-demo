"""Run the whole demo end to end on local Spark + Delta Lake.

    pip install -e ".[local,dev]"
    python main.py

Two batches are processed so the incremental parts are visible: the second run
of bronze/silver only picks up new files, MERGE applies customer updates and
deletes, and the new ``coupon_code`` column flows through via schema evolution.
"""

from databricks_demo import bronze, delta_features, generator, gold, setup, silver
from databricks_demo.config import DemoConfig, build_parser
from databricks_demo.spark import get_spark


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--batches", type=int, default=2, help="How many ~30-day batches to process")
    parser.add_argument("--skip-ml", action="store_true", help="Skip MLflow training and scoring")
    args = parser.parse_args()
    cfg = DemoConfig.from_args(args)
    spark = get_spark()

    setup.run(spark, cfg)
    for batch in range(args.batches):
        print(f"\n######## Batch {batch} ########")
        generator.run(cfg, batch)
        bronze.run(spark, cfg)
        silver.run(spark, cfg)
        gold.run(spark, cfg)

    delta_features.run(spark, cfg)

    if not args.skip_ml:
        from databricks_demo import ml

        ml.run(spark, cfg)


if __name__ == "__main__":
    main()
