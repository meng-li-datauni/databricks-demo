import os
import shutil

import pytest


@pytest.fixture(scope="session")
def spark():
    """Plain local Spark (no Delta needed): transforms are pure DataFrame functions."""
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        pytest.skip("Java is not installed; Spark tests need a JVM")
    pyspark = pytest.importorskip("pyspark")
    session = (
        pyspark.sql.SparkSession.builder.master("local[2]")
        .appName("databricks-demo-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
