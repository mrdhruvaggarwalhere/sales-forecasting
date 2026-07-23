"""
Sales Analysis Page — Daily, Monthly, Yearly trends with Weekday and SNAP breakdowns.
"""

import streamlit as st
import pandas as pd
from components.charts import line_chart, bar_chart, donut_chart, PLOTLY_CONFIG
from components.kpi_cards import render_kpi_row


def render(df: pd.DataFrame):
    """Renders the Sales Analysis page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Sales Analysis</div>'
'<div class="page-subtitle">Granular breakdown of sales performance across time and segments</div>'
'</div>',
        unsafe_allow_html=True,
    )

    if df.empty or "sales" not in df.columns:
        st.warning("No sales data available.")
        return

    # ── Summary KPIs ──────────────────────────────────────────
    total = int(df["sales"].sum())
    avg = int(df["sales"].mean())
    peak = int(df["sales"].max())
    peak_date = df.loc[df["sales"].idxmax(), "date"].strftime("%Y-%m-%d") if "date" in df.columns else "N/A"

    render_kpi_row([
        {"label": "Total Sales", "value": f"{total:,}", "accent": "green", "icon": "📦"},
        {"label": "Avg Daily", "value": f"{avg:,}", "accent": "blue", "icon": "📊"},
        {"label": "Peak Day", "value": f"{peak:,}", "accent": "amber", "icon": "🔥", "subtext": peak_date},
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Daily Trend", "Monthly", "Yearly", "Weekday", "SNAP Days"
    ])

    with tab1:
        st.markdown('<div class="section-header">📈 Daily Sales Volume</div>', unsafe_allow_html=True)
        fig = line_chart(df, "date", "sales", "Daily Sales Over Time")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab2:
        if "month" in df.columns and "year" in df.columns:
            st.markdown('<div class="section-header">📅 Monthly Sales</div>', unsafe_allow_html=True)
            monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            fig = bar_chart(monthly, "period", "sales", "Monthly Total Sales")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab3:
        if "year" in df.columns:
            st.markdown('<div class="section-header">📊 Yearly Sales</div>', unsafe_allow_html=True)
            yearly = df.groupby("year")["sales"].sum().reset_index()
            yearly["year"] = yearly["year"].astype(str)
            fig = bar_chart(yearly, "year", "sales", "Annual Total Sales")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab4:
        if "weekday" in df.columns:
            st.markdown('<div class="section-header">📅 Sales by Day of Week</div>', unsafe_allow_html=True)
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_sales = df.groupby("weekday")["sales"].mean().reset_index()
            weekday_sales["weekday"] = pd.Categorical(weekday_sales["weekday"], categories=day_order, ordered=True)
            weekday_sales = weekday_sales.sort_values("weekday")

            col1, col2 = st.columns(2)
            with col1:
                fig = bar_chart(weekday_sales, "weekday", "sales", "Average Daily Sales by Weekday")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            with col2:
                fig = donut_chart(weekday_sales, "sales", "weekday", "Weekday Sales Distribution")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab5:
        snap_cols = [c for c in df.columns if c.startswith("snap_")]
        if snap_cols:
            st.markdown('<div class="section-header">🏛️ SNAP Days Impact</div>', unsafe_allow_html=True)
            st.caption("SNAP = Supplemental Nutrition Assistance Program — government benefits days that boost sales.")
            for snap_col in snap_cols:
                state = snap_col.replace("snap_", "").upper()
                snap_df = df.groupby(snap_col)["sales"].mean().reset_index()
                snap_df[snap_col] = snap_df[snap_col].map({0: f"Non-SNAP ({state})", 1: f"SNAP Day ({state})"})
                fig = bar_chart(snap_df, snap_col, "sales", f"Avg Sales: SNAP vs Non-SNAP — {state}")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
