"""
Product Analysis Page.
Best/worst selling products and department-level performance.
"""

import streamlit as st
import pandas as pd
from components.charts import bar_chart, donut_chart


def render(df: pd.DataFrame):
    """Renders the Product Analysis page."""
    st.markdown('<div class="page-title">Product Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Identify top performers and underperformers across your catalog</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab1, tab2, tab3 = st.tabs(["Best Selling", "Worst Selling", "Department Analysis"])

    with tab1:
        if "item_id" in df.columns and "revenue" in df.columns:
            top = df.groupby("item_id")["revenue"].sum().nlargest(15).reset_index()
            top["item_id"] = top["item_id"].astype(str)
            fig = bar_chart(top, "item_id", "revenue", "Top 15 Products by Revenue")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "item_id" in df.columns and "revenue" in df.columns:
            bottom = df.groupby("item_id")["revenue"].sum().nsmallest(15).reset_index()
            bottom["item_id"] = bottom["item_id"].astype(str)
            fig = bar_chart(bottom, "item_id", "revenue", "Bottom 15 Products by Revenue")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "dept_id" in df.columns and "revenue" in df.columns:
            dept = df.groupby("dept_id")["revenue"].sum().reset_index()
            dept["dept_id"] = dept["dept_id"].astype(str)
            col1, col2 = st.columns(2)
            with col1:
                fig = bar_chart(dept, "dept_id", "revenue", "Department Revenue")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = donut_chart(dept, "revenue", "dept_id", "Revenue Share by Dept")
                st.plotly_chart(fig, use_container_width=True)
