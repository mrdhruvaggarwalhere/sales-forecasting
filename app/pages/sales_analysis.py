"""
Sales Analysis Page.
Daily, Monthly, Yearly trends with Category, Store, and State breakdowns.
"""

import streamlit as st
import pandas as pd
from components.charts import line_chart, bar_chart, donut_chart


def render(df: pd.DataFrame):
    """Renders the Sales Analysis page."""
    st.markdown('<div class="page-title">Sales Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Granular breakdown of sales performance across time and segments</div>', unsafe_allow_html=True)

    if df.empty or "date" not in df.columns:
        st.warning("No data available for the selected filters.")
        return

    # ── Tabs ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Daily", "Monthly", "Yearly", "Category", "Store", "State"
    ])

    with tab1:
        daily = df.groupby("date")["sales_units"].sum().reset_index()
        fig = line_chart(daily, "date", "sales_units", "Daily Sales Volume")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "month" in df.columns and "year" in df.columns:
            monthly = df.groupby(["year", "month"])["sales_units"].sum().reset_index()
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            fig = bar_chart(monthly, "period", "sales_units", "Monthly Sales")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "year" in df.columns:
            yearly = df.groupby("year")["sales_units"].sum().reset_index()
            yearly["year"] = yearly["year"].astype(str)
            fig = bar_chart(yearly, "year", "sales_units", "Yearly Sales")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if "cat_id" in df.columns:
            cat = df.groupby("cat_id")["sales_units"].sum().reset_index()
            cat["cat_id"] = cat["cat_id"].astype(str)
            fig = donut_chart(cat, "sales_units", "cat_id", "Sales by Category")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if "store_id" in df.columns:
            store = df.groupby("store_id")["sales_units"].sum().sort_values(ascending=False).reset_index()
            store["store_id"] = store["store_id"].astype(str)
            fig = bar_chart(store, "store_id", "sales_units", "Store-wise Sales")
            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        if "state_id" in df.columns:
            state = df.groupby("state_id")["sales_units"].sum().reset_index()
            state["state_id"] = state["state_id"].astype(str)
            fig = donut_chart(state, "sales_units", "state_id", "Sales by State")
            st.plotly_chart(fig, use_container_width=True)
