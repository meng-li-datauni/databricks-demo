import json

from databricks_demo import generator


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_batch_zero_writes_snapshot_files(tmp_path):
    written = generator.write_batch(tmp_path, 0)

    assert written["customers"] == generator.N_CUSTOMERS
    assert len(_read(tmp_path / "products" / "products.json")) == generator.N_PRODUCTS
    orders = _read(tmp_path / "orders" / "orders_000.json")
    assert len(orders) == written["orders"] >= generator.ORDERS_PER_BATCH
    assert not any("coupon_code" in o for o in orders), "coupon_code must only appear from batch 1"


def test_orders_contain_dirty_records():
    orders = generator.orders(0)
    ids = [o["order_id"] for o in orders]

    assert any(o["customer_id"] is None for o in orders)
    assert any(o["quantity"] <= 0 for o in orders)
    assert len(ids) > len(set(ids)), "expected exact duplicates"


def test_later_batches_add_new_column_and_cdc(tmp_path):
    generator.write_batch(tmp_path, 1)

    assert any("coupon_code" in o for o in _read(tmp_path / "orders" / "orders_001.json"))
    ops = {c["operation"] for c in _read(tmp_path / "customers" / "customers_001.json")}
    assert ops == {"UPSERT", "DELETE"}


def test_generation_is_deterministic():
    assert generator.orders(2) == generator.orders(2)
    assert generator.customer_changes(1) == generator.customer_changes(1)


def test_premium_customers_buy_more():
    tiers = {c["customer_id"]: c["tier"] for c in generator.customers()}
    spend = {"premium": 0.0, "standard": 0.0}
    for o in generator.orders(0):
        if o["customer_id"]:
            spend[tiers[o["customer_id"]]] += o["quantity"] * o["unit_price"]
    n_premium = sum(t == "premium" for t in tiers.values())
    per_premium = spend["premium"] / n_premium
    per_standard = spend["standard"] / (len(tiers) - n_premium)
    assert per_premium > 1.5 * per_standard, "the ML label needs a learnable signal"
