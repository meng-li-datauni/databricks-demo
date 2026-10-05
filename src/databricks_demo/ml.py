"""MLflow: train, track, register and batch-score a model on gold features.

Predicts whether a customer is premium-tier from their purchasing behaviour.

* Tracking: params, metrics and the model artifact are logged to an MLflow run.
* Registry: on Databricks the model is registered in **Unity Catalog**
  (``catalog.schema.model``); locally in a SQLite-backed MLflow registry.
* Aliases: the new version gets the ``champion`` alias, and scoring loads
  ``models:/<name>@champion`` instead of a hard-coded version number.
* Batch inference: the model is wrapped as a Spark UDF and applied in parallel.
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from databricks_demo.config import LOCAL_ROOT, DemoConfig, build_parser, is_databricks
from databricks_demo.spark import get_spark

NUMERIC_FEATURES = ["num_orders", "total_spend", "avg_order_value", "distinct_categories", "days_since_last_order"]
CATEGORICAL_FEATURES = ["country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPERIMENT = "databricks_demo"
ALIAS = "champion"


def configure_mlflow(spark: SparkSession, cfg: DemoConfig) -> str:
    """Point MLflow at the right tracking server / registry; return the model name."""
    if is_databricks():
        mlflow.set_registry_uri("databricks-uc")
        user = spark.sql("SELECT current_user()").first()[0]
        mlflow.set_experiment(f"/Users/{user}/{EXPERIMENT}")
        return f"{cfg.catalog}.{cfg.schema}.customer_tier_model"

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{LOCAL_ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(LOCAL_ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return "customer_tier_model"


def _feature_frame(spark: SparkSession, cfg: DemoConfig):
    """Spark-side casts so training data and scoring data share one schema."""
    return spark.table(cfg.table("customer_features")).select(
        "customer_id",
        "tier",
        *[F.col(c).cast("double").alias(c) for c in NUMERIC_FEATURES],
        *CATEGORICAL_FEATURES,
    )


def train(spark: SparkSession, cfg: DemoConfig, model_name: str) -> str:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    data = _feature_frame(spark, cfg).toPandas()
    X, y = data[FEATURES], (data["tier"] == "premium").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("country", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
                    remainder="passthrough",
                ),
            ),
            ("clf", RandomForestClassifier(**params)),
        ]
    )

    with mlflow.start_run(run_name="rf_customer_tier") as run:
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(X_train))
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, proba >= 0.5),
            "f1": f1_score(y_test, proba >= 0.5),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)
        print(f"Run {run.info.run_id}: {', '.join(f'{k}={v:.3f}' for k, v in metrics.items())}")

        # Unity Catalog requires a signature (input/output schema) on every model.
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
            registered_model_name=model_name,
            # MLflow serializes with skops, which only loads types it was told to trust.
            # Safe here because we trained this model ourselves.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name = '{model_name}'"))
    client.set_registered_model_alias(model_name, ALIAS, str(version))
    print(f"Registered {model_name} v{version} as @{ALIAS}")
    return f"models:/{model_name}@{ALIAS}"


def batch_score(spark: SparkSession, cfg: DemoConfig, model_uri: str) -> None:
    predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")
    target = cfg.table("customer_tier_predictions")
    (
        _feature_frame(spark, cfg)
        .withColumn("premium_prediction", predict(F.struct(*FEATURES)).cast("int"))
        .withColumn("model_uri", F.lit(model_uri))
        .withColumn("scored_at", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target)
    )
    spark.sql(
        f"""
        SELECT tier, premium_prediction, count(*) AS customers
        FROM {target} GROUP BY ALL ORDER BY ALL
        """
    ).show()


def run(spark: SparkSession, cfg: DemoConfig) -> None:
    model_name = configure_mlflow(spark, cfg)
    model_uri = train(spark, cfg, model_name)
    batch_score(spark, cfg, model_uri)


def main(argv: list[str] | None = None) -> None:
    args = build_parser(__doc__).parse_args(argv)
    run(get_spark(), DemoConfig.from_args(args))


if __name__ == "__main__":
    main()
