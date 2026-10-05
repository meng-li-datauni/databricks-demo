"""Generate synthetic e-commerce JSON files into the landing zone.

Plain Python (no Spark): on Databricks, UC Volumes are mounted as a regular
filesystem under ``/Volumes``, so ``open()`` just works.

Each batch represents ~30 days of activity:

* batch 0 writes products, the initial customer snapshot and the first orders;
* batch N > 0 writes more orders plus customer CDC records (updates, inserts,
  deletes), and orders start carrying a new ``coupon_code`` field to exercise
  schema evolution.

Orders deliberately contain dirty records (missing customer, negative quantity,
duplicates) so the silver layer and pipeline expectations have work to do.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from databricks_demo.config import DemoConfig, build_parser

SEED = 42
START = datetime(2026, 1, 1)
DAYS_PER_BATCH = 30
N_PRODUCTS = 40
N_CUSTOMERS = 200
ORDERS_PER_BATCH = 1500
NEW_CUSTOMERS_PER_BATCH = 10

CATEGORIES = {
    "Electronics": (50, 900),
    "Books": (8, 60),
    "Home": (15, 300),
    "Sports": (10, 250),
    "Toys": (5, 120),
    "Beauty": (6, 90),
}
COUNTRIES = ["US", "DE", "FR", "GB", "NL", "SE", "JP", "BR"]
STATUSES = ["placed", "shipped", "cancelled"]
STATUS_WEIGHTS = [0.20, 0.72, 0.08]
CHANNELS = ["web", "mobile", "store"]


def products() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        category = rng.choice(list(CATEGORIES))
        low, high = CATEGORIES[category]
        rows.append(
            {
                "product_id": f"P{i:03d}",
                "name": f"{category} item {i}",
                "category": category,
                "list_price": round(rng.uniform(low, high), 2),
            }
        )
    return rows


def customers() -> list[dict]:
    """Initial customer snapshot; ~25% are premium-tier (the ML label)."""
    rng = random.Random(SEED + 1)
    return [
        {
            "customer_id": f"C{i:04d}",
            "name": f"Customer {i}",
            "email": f"customer{i}@example.com",
            "country": rng.choice(COUNTRIES),
            "signup_date": (START - timedelta(days=rng.randint(30, 700))).date().isoformat(),
            "tier": "premium" if rng.random() < 0.25 else "standard",
            "operation": "UPSERT",
            "updated_at": START.isoformat(),
        }
        for i in range(1, N_CUSTOMERS + 1)
    ]


def customer_changes(batch: int) -> list[dict]:
    """CDC feed for a later batch: updates, brand-new customers and deletes."""
    rng = random.Random(SEED + 100 + batch)
    changed_at = (START + timedelta(days=batch * DAYS_PER_BATCH)).isoformat()
    base = customers()
    rows = []
    for customer in rng.sample(base, 15):
        rows.append(
            {
                **customer,
                "email": customer["email"].replace("@example.com", f"+b{batch}@example.com"),
                "country": rng.choice(COUNTRIES),
                "updated_at": changed_at,
            }
        )
    first_new = N_CUSTOMERS + (batch - 1) * NEW_CUSTOMERS_PER_BATCH + 1
    for i in range(first_new, first_new + NEW_CUSTOMERS_PER_BATCH):
        rows.append(
            {
                "customer_id": f"C{i:04d}",
                "name": f"Customer {i}",
                "email": f"customer{i}@example.com",
                "country": rng.choice(COUNTRIES),
                "signup_date": changed_at[:10],
                "tier": "standard",
                "operation": "UPSERT",
                "updated_at": changed_at,
            }
        )
    for customer in rng.sample(base, 3):
        rows.append({"customer_id": customer["customer_id"], "operation": "DELETE", "updated_at": changed_at})
    return rows


def orders(batch: int) -> list[dict]:
    rng = random.Random(SEED + 1000 + batch)
    catalog = products()
    people = customers()
    # Premium customers order more often, buy more, and favour pricier products;
    # a random per-customer activity level blurs the line so the model has to work.
    activity = random.Random(SEED + 2)
    customer_weights = [
        activity.lognormvariate(0, 0.6) * (1.8 if c["tier"] == "premium" else 1.0) for c in people
    ]
    premium_product_weights = [p["list_price"] ** 0.5 for p in catalog]
    window_start = START + timedelta(days=batch * DAYS_PER_BATCH)

    rows: list[dict] = []
    for n in range(ORDERS_PER_BATCH):
        customer = rng.choices(people, weights=customer_weights)[0]
        premium = customer["tier"] == "premium"
        product = rng.choices(catalog, weights=premium_product_weights)[0] if premium else rng.choice(catalog)
        order = {
            "order_id": f"O{batch:02d}{n:05d}",
            "customer_id": customer["customer_id"],
            "product_id": product["product_id"],
            "quantity": rng.randint(1, 4) if premium else rng.randint(1, 3),
            "unit_price": round(product["list_price"] * rng.uniform(0.85, 1.0), 2),
            "order_ts": (window_start + timedelta(seconds=rng.randint(0, DAYS_PER_BATCH * 86400 - 1))).isoformat(),
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0],
            "channel": rng.choice(CHANNELS),
        }
        if batch > 0 and rng.random() < 0.3:
            order["coupon_code"] = rng.choice(["WELCOME10", "SPRING25", "VIP50"])

        # Data-quality problems for the silver layer / pipeline expectations.
        roll = rng.random()
        if roll < 0.02:
            order["customer_id"] = None
        elif roll < 0.03:
            order["quantity"] = -order["quantity"]
        rows.append(order)
        if rng.random() < 0.02:
            rows.append(dict(order))  # exact duplicate, as from an at-least-once producer
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_batch(landing_dir: Path, batch: int) -> dict[str, int]:
    """Write one batch of landing files; idempotent for a given batch number."""
    order_rows = orders(batch)
    _write_jsonl(landing_dir / "orders" / f"orders_{batch:03d}.json", order_rows)
    if batch == 0:
        _write_jsonl(landing_dir / "products" / "products.json", products())
        customer_rows = customers()
    else:
        customer_rows = customer_changes(batch)
    _write_jsonl(landing_dir / "customers" / f"customers_{batch:03d}.json", customer_rows)
    return {"orders": len(order_rows), "customers": len(customer_rows)}


def run(cfg: DemoConfig, batch: int) -> None:
    written = write_batch(Path(cfg.landing_dir), batch)
    print(f"Batch {batch}: wrote {written} to {cfg.landing_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--batch", type=int, default=0, help="Which ~30-day batch to generate")
    args = parser.parse_args(argv)
    run(DemoConfig.from_args(args), args.batch)


if __name__ == "__main__":
    main()
