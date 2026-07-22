"""
Home Page — Dashboard Overview.
Displays KPI cards, revenue trend, and sales trend.
"""

import streamlit as st
import pandas as pd
from components.kpi_cards import render_kpi_row
from components.charts import line_chart


def format_number(n: float) -> str:
    """Formats large numbers into human-readable strings."""
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:,.1f}M"
    elif abs(n) >= 1_000:
        return f"${n / 1_000:,.1f}K"
    return f"${n:,.0f}"


def render(df: pd.DataFrame, forecast_df: pd.DataFrame = None, metrics_df: pd.DataFrame = None):
    """Renders the Home page."""
    st.markdown('<div class="page-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time performance metrics and revenue trends</div>', unsafe_allow_html=True)

    # ── KPI Metrics ────────────────────────────────────────
    total_sales = int(df["sales_units"].sum()) if "sales_units" in df.columns else 0
    total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0
    total_orders = len(df) if not df.empty else 0

    # Forecast accuracy and best model (from metrics_df if available)
    best_model = "N/A"
    forecast_acc = "N/A"
    if metrics_df is not None and not metrics_df.empty:
        best_idx = metrics_df["RMSE"].idxmin() if "RMSE" in metrics_df.columns else None
        if best_idx is not None:
            best_model = metrics_df.loc[best_idx, "Model"]
            r2 = metrics_df.loc[best_idx, "R2 Score"] if "R2 Score" in metrics_df.columns else 0
            forecast_acc = f"{r2 * 100:.1f}%"

    render_kpi_row([
        {"label": "Total Sales", "value": f"{total_sales:,}", "delta": None, "accent": "blue"},
        {"label": "Total Revenue", "value": format_number(total_revenue), "delta": None, "accent": "green"},
        {"label": "Total Records", "value": f"{total_orders:,}", "delta": None, "accent": "amber"},
        {"label": "Forecast Accuracy (R²)", "value": forecast_acc, "accent": "purple"},
        {"label": "Best Model", "value": best_model, "accent": "blue"},
    ])

    st.markdown("", unsafe_allow_html=True)  # spacer

    # ── Revenue Trend ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Revenue Trend</div>', unsafe_allow_html=True)
        if "date" in df.columns and "revenue" in df.columns:
            daily_rev = df.groupby("date")["revenue"].sum().reset_index()
            fig = line_chart(daily_rev, x="date", y="revenue", title="Daily Revenue")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sales Trend</div>', unsafe_allow_html=True)
        if "date" in df.columns and "sales_units" in df.columns:
            daily_sales = df.groupby("date")["sales_units"].sum().reset_index()
            fig = line_chart(daily_sales, x="date", y="sales_units", title="Daily Sales Volume")
            st.plotly_chart(fig, use_container_width=True)
