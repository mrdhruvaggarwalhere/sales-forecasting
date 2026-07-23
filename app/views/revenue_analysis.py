"""
Revenue Analysis Page — Distribution, growth, rolling averages, and feature correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.charts import line_chart, bar_chart, histogram_chart, correlation_heatmap, multi_line_chart, PLOTLY_CONFIG
from components.kpi_cards import render_kpi_row


def render(df: pd.DataFrame, featured_df: pd.DataFrame = None):
    """Renders the Revenue Analysis page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Revenue Deep-Dive</div>'
'<div class="page-subtitle">Sales distribution, growth trends, rolling averages, and feature analysis</div>'
'</div>',
        unsafe_allow_html=True,
    )

    if df.empty or "sales" not in df.columns:
        st.warning("No sales data available.")
        return

    total = df["sales"].sum()
    avg = df["sales"].mean()
    std = df["sales"].std()
    peak = df["sales"].max()

    render_kpi_row([
        {"label": "Total Sales", "value": f"{total:,.0f}", "accent": "green", "icon": "📦"},
        {"label": "Daily Average", "value": f"{avg:,.0f}", "accent": "blue", "icon": "📊"},
        {"label": "Std Deviation", "value": f"{std:,.0f}", "accent": "amber", "icon": "📐"},
        {"label": "Peak Day", "value": f"{peak:,.0f}", "accent": "purple", "icon": "🔥"},
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Growth", "Rolling Averages", "Feature Correlation"])

    with tab1:
        st.markdown('<div class="section-header">📊 Sales Distribution</div>', unsafe_allow_html=True)
        fig = histogram_chart(df, "sales", "Daily Sales Distribution", nbins=50)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab2:
        st.markdown('<div class="section-header">📈 Month-over-Month Growth</div>', unsafe_allow_html=True)
        if "year" in df.columns and "month" in df.columns:
            monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            monthly["growth_pct"] = monthly["sales"].pct_change() * 100
            monthly = monthly.dropna(subset=["growth_pct"])
            fig = bar_chart(monthly, "period", "growth_pct", "Month-over-Month Sales Growth (%)")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab3:
        if featured_df is not None and not featured_df.empty:
            st.markdown('<div class="section-header">📉 Rolling Averages (7/14/28 day)</div>', unsafe_allow_html=True)
            rolling_cols = [c for c in featured_df.columns if "rolling_mean" in c]
            if rolling_cols and "date" in featured_df.columns:
                plot_df = featured_df[["date", "sales"] + rolling_cols].copy()
                fig = multi_line_chart(plot_df, "date", ["sales"] + rolling_cols, "Sales with Rolling Averages")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Featured data not available for rolling averages.")

    with tab4:
        if featured_df is not None and not featured_df.empty:
            st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
            numeric_cols = featured_df.select_dtypes(include=[np.number]).columns.tolist()
            interesting = [c for c in numeric_cols if c in [
                "sales", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI",
                "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"
            ]]
            if len(interesting) >= 3:
                fig = correlation_heatmap(featured_df, interesting, "Feature Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Featured data not available for correlation analysis.")
