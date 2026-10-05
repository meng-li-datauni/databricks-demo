from datetime import date, datetime
from decimal import Decimal

import pytest

pytest.importorskip("pyspark")
pytest.importorskip("delta")  # silver.py imports delta.tables at module level

from pyspark.sql import functions as F  # noqa: E402

from databricks_demo.gold import customer_features, daily_sales_by_category  # noqa: E402
from databricks_demo.silver import ORDER_COLUMNS, clean_orders, latest_customer_changes  # noqa: E402

RAW_ORDER_COLUMNS = ["order_id", "customer_id", "product_id", "quantity", "unit_price", "order_ts", "status", "channel"]


def _raw_orders(spark, rows):
    return spark.createDataFrame(rows, RAW_ORDER_COLUMNS).withColumn("_ingested_at", F.current_timestamp())


def test_clean_orders_splits_valid_and_quarantine(spark):
    raw = _raw_orders(
        spark,
        [
            ("O1", "C1", "P1", 2, 10.0, "2026-01-01T10:00:00", "Shipped", "web"),
            ("O1", "C1", "P1", 2, 10.0, "2026-01-01T10:00:00", "Shipped", "web"),  # duplicate
            ("O2", None, "P1", 1, 10.0, "2026-01-01T11:00:00", "placed", "web"),
            ("O3", "C2", "P2", -1, 5.0, "2026-01-02T09:00:00", "placed", "store"),
            ("O4", "C2", "P2", 1, 5.0, "not a timestamp", "placed", "store"),
        ],
    )
    valid, quarantine = clean_orders(raw)

    rows = valid.collect()
    assert valid.columns == ORDER_COLUMNS
    assert [r.order_id for r in rows] == ["O1"]
    assert rows[0].amount == Decimal("20.00")
    assert rows[0].status == "shipped"
    assert rows[0].order_date == date(2026, 1, 1)
    assert rows[0].coupon_code is None  # column added even though the source lacked it

    reasons = {r.order_id: r._dq_reason for r in quarantine.collect()}
    assert reasons == {"O2": "missing customer_id", "O3": "non-positive quantity", "O4": "unparseable order_ts"}


def test_latest_customer_changes_keeps_newest_per_key(spark):
    changes = spark.createDataFrame(
        [
            ("C1", "a", "old@x.com", "US", "2025-01-01", "standard", "UPSERT", "2026-01-01T00:00:00"),
            ("C1", "a", "new@x.com", "DE", "2025-01-01", "standard", "UPSERT", "2026-02-01T00:00:00"),
            ("C2", None, None, None, None, None, "DELETE", "2026-02-01T00:00:00"),
        ],
        ["customer_id", "name", "email", "country", "signup_date", "tier", "operation", "updated_at"],
    )
    result = {r.customer_id: r for r in latest_customer_changes(changes).collect()}

    assert result["C1"].email == "new@x.com"
    assert result["C1"].updated_at == datetime(2026, 2, 1)
    assert result["C2"].operation == "DELETE"


@pytest.fixture
def silver_tables(spark):
    orders = spark.createDataFrame(
        [
            ("O1", "C1", "P1", 2, Decimal("20.00"), "shipped", date(2026, 1, 1)),
            ("O2", "C1", "P2", 1, Decimal("5.00"), "shipped", date(2026, 1, 3)),
            ("O3", "C2", "P1", 1, Decimal("10.00"), "cancelled", date(2026, 1, 3)),
        ],
        "order_id string, customer_id string, product_id string, quantity int, amount decimal(12,2), "
        "status string, order_date date",
    )
    products = spark.createDataFrame([("P1", "Books"), ("P2", "Toys")], ["product_id", "category"])
    customers = spark.createDataFrame(
        [("C1", "US", date(2025, 6, 1), "premium"), ("C2", "US", date(2025, 12, 1), "standard")],
        ["customer_id", "country", "signup_date", "tier"],
    )
    return orders, products, customers


def test_daily_sales_excludes_cancelled(spark, silver_tables):
    orders, products, _ = silver_tables
    rows = {(r.order_date, r.category): r for r in daily_sales_by_category(orders, products).collect()}

    assert set(rows) == {(date(2026, 1, 1), "Books"), (date(2026, 1, 3), "Toys")}
    assert rows[(date(2026, 1, 1), "Books")].revenue == Decimal("20.00")


def test_customer_features(spark, silver_tables):
    features = {r.customer_id: r for r in customer_features(*silver_tables).collect()}

    c1, c2 = features["C1"], features["C2"]
    assert (c1.num_orders, c1.total_spend, c1.distinct_categories) == (2, Decimal("25.00"), 2)
    assert c1.avg_order_value == Decimal("12.50")
    assert c1.days_since_last_order == 0
    assert c1.spend_rank_in_country == 1
    # C2 only has a cancelled order: zero activity, recency measured from signup.
    assert (c2.num_orders, c2.total_spend) == (0, Decimal("0.00"))
    assert c2.days_since_last_order == (date(2026, 1, 3) - date(2025, 12, 1)).days
    assert c2.spend_rank_in_country == 2
