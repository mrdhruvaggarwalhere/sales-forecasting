"""
Home Page — Executive Dashboard Overview.
Shows KPI cards, revenue trend, sales distribution, and model performance summary.
"""

import streamlit as st
import pandas as pd
from components.kpi_cards import render_kpi_row
from components.charts import line_chart, bar_chart, PLOTLY_CONFIG


def _format_number(n: float) -> str:
    """Formats large numbers into human-readable strings."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:,.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:,.1f}K"
    return f"{n:,.0f}"


def render(df: pd.DataFrame, eval_metrics: dict = None, plot_data: dict = None):
    """Renders the Home overview page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Executive Dashboard Overview</div>'
'<div class="page-subtitle">Real-time enterprise performance metrics and revenue intelligence</div>'
'</div>',
        unsafe_allow_html=True,
    )

    # ── KPI Calculations ──────────────────────────────────────
    total_sales = int(df["sales"].sum()) if "sales" in df.columns else 0
    total_days = len(df)
    avg_daily = int(df["sales"].mean()) if "sales" in df.columns else 0

    # Best model from evaluation_metrics.json
    best_model = "N/A"
    best_rmse = "N/A"
    if eval_metrics and isinstance(eval_metrics, dict):
        try:
            best_name = min(eval_metrics, key=lambda k: eval_metrics[k].get("RMSE", float("inf")))
            best_model = best_name
            best_rmse = f"{eval_metrics[best_name]['RMSE']:,.0f}"
        except Exception:
            pass

    render_kpi_row([
        {
            "label": "Total Sales Volume",
            "value": _format_number(total_sales),
            "delta": f"{total_days} days",
            "delta_positive": True,
            "accent": "green",
            "icon": "📦",
            "subtext": "units sold across all days",
        },
        {
            "label": "Avg Daily Sales",
            "value": _format_number(avg_daily),
            "accent": "blue",
            "icon": "📊",
            "subtext": "units per day",
        },
        {
            "label": "Data Points",
            "value": f"{total_days:,}",
            "accent": "amber",
            "icon": "⚡",
            "subtext": "daily records",
        },
        {
            "label": "Best Model",
            "value": best_model,
            "accent": "purple",
            "icon": "🏆",
            "subtext": f"RMSE: {best_rmse}",
        },
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    # ── Trend Charts ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📈 Daily Sales Trend</div>', unsafe_allow_html=True)
        if "date" in df.columns and "sales" in df.columns:
            fig = line_chart(df, x="date", y="sales", title="Daily Total Sales Volume")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        st.markdown('<div class="section-header">📅 Monthly Sales</div>', unsafe_allow_html=True)
        if "month" in df.columns and "year" in df.columns and "sales" in df.columns:
            monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            fig = bar_chart(monthly, "period", "sales", "Monthly Aggregated Sales")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Model Performance Preview ─────────────────────────────
    if eval_metrics and isinstance(eval_metrics, dict):
        st.markdown('<div class="section-header">⚡ Model Performance Overview</div>', unsafe_allow_html=True)
        metrics_rows = []
        for model_name, m in eval_metrics.items():
            if isinstance(m, dict):
                metrics_rows.append({
                    "Model": model_name,
                    "MAE": m.get("MAE", 0),
                    "RMSE": m.get("RMSE", 0),
                    "MAPE (%)": m.get("MAPE", 0),
                })
        if metrics_rows:
            metrics_table = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)
