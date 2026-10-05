"""Local lakehouse explorer: a stand-in for Catalog Explorer + SQL editor.

    pip install -e ".[local,ui]"
    streamlit run ui/app.py

Uses the same local Spark session/metastore as ``main.py``. The embedded Derby
metastore allows one process at a time, so don't run ``main.py`` while this is open;
use the sidebar to process new batches instead.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from databricks_demo import bronze, generator, gold, setup, silver
from databricks_demo.config import DemoConfig
from databricks_demo.spark import get_spark

st.set_page_config(page_title="Lakehouse Explorer", page_icon="🧱", layout="wide")

CFG = DemoConfig(catalog=None, schema="databricks_demo")
CATEGORIES = list(generator.CATEGORIES)  # fixed order: a category keeps its color everywhere
# Validated categorical palette (dataviz skill reference palette, slots 1-6), per theme.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
}


@st.cache_resource(show_spinner="Starting local Spark…")
def spark():
    return get_spark()


def sql(query: str) -> pd.DataFrame:
    return spark().sql(query).toPandas()


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:  # Streamlit < 1.46
        return "light"


def series_colors() -> alt.Scale:
    return alt.Scale(domain=CATEGORIES, range=PALETTE[theme()])


def single_color() -> str:
    return PALETTE[theme()][0]


def tables() -> list[str]:
    if not spark().catalog.databaseExists(CFG.schema):
        return []
    return sorted(t.name for t in spark().catalog.listTables(CFG.schema))


def landed_batches() -> int:
    return len(list(Path(CFG.landing_dir, "orders").glob("orders_*.json")))


# ------------------------------------------------------------------ sidebar


def sidebar() -> None:
    st.sidebar.header("Pipeline")
    batches = landed_batches()
    st.sidebar.caption(f"{batches} batch(es) landed in `{CFG.landing_dir}`")
    if st.sidebar.button(f"Generate & process batch {batches}", type="primary", width="stretch"):
        with st.sidebar.status(f"Batch {batches}", expanded=True) as status:
            setup.run(spark(), CFG)
            st.write("Generating landing files…")
            generator.run(CFG, batches)
            st.write("Bronze: Auto Loader-style incremental ingest…")
            bronze.run(spark(), CFG)
            st.write("Silver: validate, quarantine, MERGE…")
            silver.run(spark(), CFG)
            st.write("Gold: rebuild aggregates…")
            gold.run(spark(), CFG)
            status.update(label=f"Batch {batches} processed", state="complete")
        st.rerun()
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Also try**\n\n"
        "- MLflow UI: `mlflow ui --backend-store-uri sqlite:///.local/mlflow.db`\n"
        "- Spark UI: http://localhost:4040"
    )


# ------------------------------------------------------------------ overview


def revenue_trend_chart(df: pd.DataFrame) -> alt.LayerChart:
    """Weekly revenue per category: lines, hover crosshair + tooltip."""
    hover = alt.selection_point(fields=["week"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(df).encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
        y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="$,.0f", gridOpacity=0.4)),
        color=alt.Color("category:N", scale=series_colors(), legend=alt.Legend(orient="top", title=None)),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=70, filled=True, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    per_week = alt.Chart(df).transform_pivot("category", value="revenue", groupby=["week"]).encode(x="week:T")
    rule = per_week.mark_rule(color="gray", strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0))
    )
    # Invisible wide rules: a generous hit target per week that carries the tooltip.
    hit = per_week.mark_rule(strokeWidth=40, opacity=0).encode(
        tooltip=[alt.Tooltip("week:T", title="Week of", format="%b %d, %Y")]
        + [alt.Tooltip(f"{c}:Q", format="$,.0f") for c in CATEGORIES if c in set(df["category"])]
    ).add_params(hover)
    return alt.layer(lines, points, rule, hit).properties(height=320)


def horizontal_bars(df: pd.DataFrame, y: str, x: str, x_title: str, x_format: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(color=single_color(), cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=x_format, gridOpacity=0.4)),
            tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", title=x_title, format=x_format)],
        )
        .properties(height=alt.Step(30))
    )


def overview() -> None:
    kpis = sql(
        f"""
        SELECT CAST(SUM(amount) AS DOUBLE) AS revenue, COUNT(DISTINCT order_id) AS orders,
               COUNT(DISTINCT customer_id) AS customers,
               CAST(SUM(amount) / COUNT(DISTINCT order_id) AS DOUBLE) AS aov
        FROM {CFG.table('orders_silver')} WHERE status <> 'cancelled'
        """
    ).iloc[0]
    rejected = sql(f"SELECT COUNT(*) AS n FROM {CFG.table('orders_quarantine')}").iloc[0]["n"]
    cols = st.columns(5)
    cols[0].metric("Revenue", f"${kpis.revenue:,.0f}")
    cols[1].metric("Orders", f"{int(kpis.orders):,}")
    cols[2].metric("Avg order value", f"${kpis.aov:,.2f}")
    cols[3].metric("Active customers", f"{int(kpis.customers):,}")
    cols[4].metric("Rejected rows", f"{rejected:,}")

    trend = sql(
        f"""
        SELECT date_trunc('WEEK', order_date) AS week, category, CAST(SUM(revenue) AS DOUBLE) AS revenue
        FROM {CFG.table('daily_sales_by_category')} GROUP BY ALL ORDER BY week
        """
    )
    st.subheader("Weekly revenue by category")
    st.caption("Weeks start Monday; the first and last weeks of the data are partial.")
    st.altair_chart(revenue_trend_chart(trend), width="stretch")
    with st.expander("Data table"):
        st.dataframe(trend.pivot(index="week", columns="category", values="revenue"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue by category")
        by_cat = sql(
            f"""SELECT category, CAST(SUM(revenue) AS DOUBLE) AS revenue
            FROM {CFG.table('daily_sales_by_category')} GROUP BY ALL"""
        )
        st.altair_chart(horizontal_bars(by_cat, "category", "revenue", "Revenue", "$,.0f"), width="stretch")
    with right:
        st.subheader("Top customers")
        st.dataframe(
            sql(
                f"""SELECT customer_id, country, tier, total_spend, num_orders
                FROM {CFG.table('customer_features')} ORDER BY total_spend DESC LIMIT 10"""
            ),
            hide_index=True,
            width="stretch",
        )


# ------------------------------------------------------------------ table explorer


def table_explorer(names: list[str]) -> None:
    name = st.selectbox("Table", names, index=names.index("orders_silver") if "orders_silver" in names else 0)
    full = CFG.table(name)
    view = st.segmented_control(
        "View", ["Preview", "Schema & details", "History & time travel", "Change feed"],
        default="Preview", key="table_view", label_visibility="collapsed",
    ) or "Preview"

    if view == "Preview":
        st.caption(f"`{full}` · {spark().table(full).count():,} rows")
        st.dataframe(spark().table(full).limit(200).toPandas(), width="stretch")

    elif view == "Schema & details":
        st.dataframe(sql(f"DESCRIBE {full}"), hide_index=True, width="stretch")
        detail = sql(f"DESCRIBE DETAIL {full}").iloc[0]
        c = st.columns(3)
        c[0].metric("Files", detail["numFiles"])
        c[1].metric("Size", f"{detail['sizeInBytes'] / 1024:,.0f} KB")
        c[2].metric("Format", detail["format"])
        st.json(dict(detail["properties"] or {}))

    elif view == "History & time travel":
        history = sql(f"DESCRIBE HISTORY {full}")[["version", "timestamp", "operation", "operationParameters", "operationMetrics"]]
        st.dataframe(history, hide_index=True, width="stretch")
        versions = sorted(history["version"].tolist())
        if len(versions) > 1:
            version = st.select_slider("Query the table as of version", options=versions, value=versions[-1])
            snapshot = spark().sql(f"SELECT * FROM {full} VERSION AS OF {version}")
            st.caption(f"`SELECT * FROM {full} VERSION AS OF {version}` → {snapshot.count():,} rows")
            st.dataframe(snapshot.limit(200).toPandas(), width="stretch")

    elif view == "Change feed":
        props = sql(f"SHOW TBLPROPERTIES {full}")
        enabled = ((props["key"] == "delta.enableChangeDataFeed") & (props["value"] == "true")).any()
        if not enabled:
            st.info("Change Data Feed isn't enabled on this table (it is on `orders_silver` and `customers_silver`).")
        else:
            st.dataframe(
                sql(
                    f"""SELECT _commit_version, _change_type, count(*) AS rows
                    FROM table_changes('{full}', 0) GROUP BY ALL ORDER BY ALL"""
                ),
                hide_index=True,
            )
            change = st.selectbox("Show rows of type", ["update_postimage", "update_preimage", "insert", "delete"])
            st.dataframe(
                sql(f"SELECT * FROM table_changes('{full}', 0) WHERE _change_type = '{change}' LIMIT 200"),
                width="stretch",
            )


# ------------------------------------------------------------------ SQL editor

EXAMPLES = {
    # Databricks SQL can filter on the window directly with QUALIFY; OSS Spark 3.5 can't.
    "Top 3 customers per country (window function)": f"""SELECT country, customer_id, tier, total_spend
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY total_spend DESC) AS rn
  FROM {CFG.table('customer_features')}
)
WHERE rn <= 3
ORDER BY country, total_spend DESC""",
    "Channel mix per month (PIVOT)": f"""SELECT * FROM (
  SELECT date_format(order_date, 'yyyy-MM') AS month, channel, amount
  FROM {CFG.table('orders_silver')} WHERE status <> 'cancelled'
) PIVOT (ROUND(SUM(amount), 2) FOR channel IN ('web', 'mobile', 'store'))
ORDER BY month""",
    "Coupon usage": f"""SELECT coupon_code, COUNT(*) AS orders, ROUND(SUM(amount), 2) AS revenue
FROM {CFG.table('orders_silver')}
GROUP BY ALL ORDER BY orders DESC""",
    "Time travel": f"SELECT COUNT(*) AS rows FROM {CFG.table('orders_silver')} VERSION AS OF 3",
}


def sql_editor() -> None:
    example = st.selectbox("Start from an example", list(EXAMPLES))
    query = st.text_area("SQL", EXAMPLES[example], height=180, key=f"sql_{example}")
    if st.button("Run", type="primary"):
        try:
            st.session_state.sql_result = sql(query)
        except Exception as e:  # show Spark's message instead of a stack trace
            st.session_state.sql_result = str(e).split("\n\nJVM stacktrace")[0][:2000]
    result = st.session_state.get("sql_result")
    if isinstance(result, str):
        st.error(result)
    elif result is not None:
        st.caption(f"{len(result):,} rows")
        st.dataframe(result, width="stretch")


# ------------------------------------------------------------------ data quality & ML


def data_quality() -> None:
    reasons = sql(f"SELECT _dq_reason AS reason, COUNT(*) AS rows FROM {CFG.table('orders_quarantine')} GROUP BY ALL")
    st.subheader("Why rows were quarantined")
    st.altair_chart(horizontal_bars(reasons, "reason", "rows", "Rows", ",d"), width="stretch")
    st.subheader("Rejected rows")
    st.dataframe(spark().table(CFG.table("orders_quarantine")).limit(500).toPandas(), width="stretch")
    st.subheader("Constraints enforced on orders_silver")
    props = sql(f"SHOW TBLPROPERTIES {CFG.table('orders_silver')}")
    st.dataframe(props[props["key"].str.startswith("delta.constraints")], hide_index=True)


def model_results(names: list[str]) -> None:
    if "customer_tier_predictions" not in names:
        st.info("No predictions yet. Run `python main.py` (without `--skip-ml`) to train and score the model.")
        return
    preds = spark().table(CFG.table("customer_tier_predictions")).toPandas()
    actual = (preds["tier"] == "premium").astype(int)
    correct = (actual == preds["premium_prediction"]).mean()
    c = st.columns(2)
    c[0].metric("Customers scored", f"{len(preds):,}")
    c[1].metric("Accuracy (incl. training rows)", f"{correct:.1%}")
    st.caption(f"Scored with `{preds['model_uri'].iloc[0]}` (held-out test metrics are in MLflow)")
    st.subheader("Actual vs predicted")
    matrix = pd.crosstab(
        preds["tier"].rename("actual tier"),
        preds["premium_prediction"].map({1: "predicted premium", 0: "predicted standard"}),
    )
    st.dataframe(matrix.reset_index(), hide_index=True)
    st.subheader("Misclassified customers")
    st.dataframe(preds[actual != preds["premium_prediction"]], hide_index=True, width="stretch")
    st.caption("Runs, metrics and model versions: `mlflow ui --backend-store-uri sqlite:///.local/mlflow.db`")


# ------------------------------------------------------------------ main


PAGES = {
    "Overview": lambda names: overview(),
    "Tables": table_explorer,
    "SQL": lambda names: sql_editor(),
    "Data quality": lambda names: data_quality(),
    "Model": model_results,
}


def main() -> None:
    st.title("Lakehouse Explorer")
    sidebar()
    names = tables()
    if "orders_silver" not in names:
        st.info("No data yet. Click **Generate & process batch 0** in the sidebar (or run `python main.py`).")
        return
    # A segmented control (unlike st.tabs) keeps its selection across reruns and lets
    # us render only the active page, so Spark only runs that page's queries.
    page = st.segmented_control(
        "Page", list(PAGES), default="Overview", key="page", label_visibility="collapsed"
    )
    PAGES[page or "Overview"](names)


main()
