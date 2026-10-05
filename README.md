# Databricks Demo Lakehouse

A small, end-to-end project that touches most of the core Databricks concepts, using
synthetic e-commerce data (products, customers with CDC updates, orders with dirty records).
It runs **locally** on open-source Spark + Delta Lake, and **on Databricks** as an Asset Bundle.

```
landing JSON ──► bronze (raw Delta) ──► silver (clean, MERGE) ──► gold (aggregates) ──► MLflow model
   (Volume)       Auto Loader            quarantine, SCD1          BI + ML features       UC registry
                  └───────────── same flow, declaratively: Lakeflow Declarative Pipeline (SCD2) ─────┘
```

## Concept map

| Concept | Where |
|---|---|
| Medallion architecture (bronze / silver / gold) | `src/databricks_demo/{bronze,silver,gold}.py` |
| Unity Catalog: `catalog.schema.table`, Volumes | `config.py`, `setup.py` |
| Auto Loader (`cloudFiles`), schema inference & evolution | `bronze.py` |
| Structured Streaming, `trigger(availableNow=True)`, checkpoints | `bronze.py`, `silver.py` |
| Streaming from Delta + `foreachBatch` + `MERGE` (idempotent upsert) | `silver.py` |
| SCD type 1 with CDC deletes / SCD type 2 with AUTO CDC | `silver.py` / `pipelines/orders_pipeline.py` |
| Data quality: quarantine table, CHECK / NOT NULL constraints | `silver.py` |
| Window functions, broadcast join, RFM features | `silver.py`, `gold.py` |
| Delta: history, time travel, CDF, schema enforcement/evolution, UPDATE/DELETE, RESTORE, OPTIMIZE/Z-ORDER, liquid clustering, VACUUM | `delta_features.py` |
| Lakeflow Declarative Pipelines (DLT): streaming tables, materialized views, expectations | `pipelines/orders_pipeline.py` |
| MLflow: tracking, signature, UC model registry, aliases, Spark UDF batch scoring | `ml.py` |
| Lakeflow Jobs: task DAG, serverless environments, job parameters, retries, schedule | `resources/lakehouse_job.yml` |
| Databricks Asset Bundles: targets, variables, dev mode, wheel artifact | `databricks.yml` |
| Notebooks: widgets, `display`, `%sql`, parameter markers, `dbutils`, lineage system tables, tags | `notebooks/explore_lakehouse.py` |
| Databricks SQL: `QUALIFY`, `PIVOT`, `GROUP BY ALL`, pipeline event log, row filters, column masks | `sql/dashboard_queries.sql` |
| Unit testing transforms with local Spark | `tests/` |

## Run locally

Needs Python 3.10+ and **Java 17** (`sudo apt install openjdk-17-jre-headless`, or set `JAVA_HOME`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local,dev]"

pytest                 # unit tests (Spark tests skip when no JVM is found)
python main.py         # setup -> 2 batches of bronze/silver/gold -> Delta tour -> MLflow
mlflow ui --backend-store-uri sqlite:///.local/mlflow.db   # browse runs and the registered model
```

### Local explorer UI

```bash
pip install -e ".[local,ui]"
streamlit run ui/app.py      # http://localhost:8501
```

A stand-in for Catalog Explorer and the SQL editor, with these pages:

- **Overview**: KPIs and revenue charts
- **Tables**: preview, schema, Delta history with a time-travel slider, and the change feed
- **SQL**: a query editor with examples
- **Data quality**: quarantined rows and the reasons they were rejected
- **Model**: predictions vs. actual tier

The sidebar's **Generate & process batch N** button runs the next batch through bronze, silver and
gold, so you can watch MERGEs, history and the charts update. The local metastore allows
one Spark process at a time, so stop the UI before running `main.py`.

`main.py` processes two batches, so the second pass shows incremental behavior: only new
files are ingested, customer updates/deletes are MERGEd, and the new `coupon_code` column
flows through. Running it again is idempotent. To start from scratch, `rm -rf .local`.

Run a single step with its entry point, e.g. `python -m databricks_demo.generator --batch 2`
then `bronze`, `silver`, `gold` (these are also installed as console scripts).

## Deploy to Databricks

Needs a Unity Catalog–enabled workspace with serverless compute, and the
[Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html).

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle validate
databricks bundle deploy -t dev                       # builds the wheel, creates job + pipeline
databricks bundle run lakehouse_etl                   # batch 0
databricks bundle run lakehouse_etl --params batch=1  # next 30 days of data
```

- **dev** target: tables go to `main.<your_user>_databricks_demo`, resources are prefixed
  `[dev <user>]`, and the schedule is paused.
- **prod** target: `main.databricks_demo`, with the daily schedule active.
- Change the catalog with `--var catalog=<name>` if you can't write to `main`.
- With `batch=1` the `bronze` and `silver` tasks can fail once and then succeed on
  retry. That's expected: streams stop when Auto Loader sees the new `coupon_code`
  column and restart with the evolved schema.

After a run, open `notebooks/explore_lakehouse.py` (synced by the bundle) to explore, and
use `sql/dashboard_queries.sql` in the SQL editor or an AI/BI dashboard.

Clean up with `databricks bundle destroy -t dev` (tables in the schema are left in place;
`DROP SCHEMA main.<user>_databricks_demo CASCADE` removes them).

## Layout

```
databricks.yml               bundle definition (targets, variables, wheel artifact)
resources/                   job and pipeline definitions
src/databricks_demo/         the Python package (one module per step, each with a main())
pipelines/                   Lakeflow Declarative Pipeline source
notebooks/                   Databricks notebook (source format)
sql/                         Databricks SQL queries
tests/                       pytest unit tests
main.py                      local end-to-end runner
```
