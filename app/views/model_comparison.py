"""
Model Comparison Page — Side-by-side evaluation of all trained forecasting models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from components.charts import apply_theme, PLOTLY_CONFIG
from components.kpi_cards import render_kpi_row


def render(eval_metrics: dict = None, metrics_csv: pd.DataFrame = None):
    """Renders the Model Comparison page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Model Comparison</div>'
'<div class="page-subtitle">Side-by-side evaluation of all trained forecasting models</div>'
'</div>',
        unsafe_allow_html=True,
    )

    metrics_df = _build_metrics_df(eval_metrics, metrics_csv)

    if metrics_df.empty:
        st.info("No evaluation metrics found. Run `python main.py --train` to generate model metrics.")
        return

    best_idx = metrics_df["RMSE"].idxmin()
    best = metrics_df.loc[best_idx]

    render_kpi_row([
        {"label": "Best Model", "value": str(best["Model"]), "accent": "green", "icon": "🏆"},
        {"label": "RMSE", "value": f"{best['RMSE']:,.2f}", "accent": "blue", "icon": "📉"},
        {"label": "MAE", "value": f"{best['MAE']:,.2f}", "accent": "amber", "icon": "📐"},
        {"label": "MAPE", "value": f"{best.get('MAPE (%)', 0):.2f}%", "accent": "purple", "icon": "📊"},
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview Table", "RMSE", "MAE", "MAPE"])

    with tab1:
        st.markdown('<div class="section-header">📋 Full Metrics Comparison</div>', unsafe_allow_html=True)

        def highlight_best(row):
            is_best = row.name == best_idx
            return ["background-color: rgba(52,211,153,0.1);" if is_best else "" for _ in row]

        styled = metrics_df.style.apply(highlight_best, axis=1).format({
            "MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab2:
        fig = _metric_bar(metrics_df, "RMSE", "Root Mean Squared Error (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab3:
        fig = _metric_bar(metrics_df, "MAE", "Mean Absolute Error (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab4:
        if "MAPE (%)" in metrics_df.columns:
            fig = _metric_bar(metrics_df, "MAPE (%)", "Mean Absolute Percentage Error (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _build_metrics_df(eval_metrics, metrics_csv) -> pd.DataFrame:
    """Builds metrics DataFrame from JSON dict or CSV."""
    if eval_metrics and isinstance(eval_metrics, dict):
        rows = []
        for model_name, m in eval_metrics.items():
            if isinstance(m, dict):
                rows.append({
                    "Model": model_name,
                    "MAE": m.get("MAE", 0),
                    "RMSE": m.get("RMSE", 0),
                    "MAPE (%)": m.get("MAPE", 0),
                })
        if rows:
            return pd.DataFrame(rows)

    if metrics_csv is not None and not metrics_csv.empty:
        return metrics_csv

    return pd.DataFrame()


def _metric_bar(df: pd.DataFrame, col: str, title: str):
    """Horizontal bar chart comparing a metric across models."""
    sorted_df = df.sort_values(col, ascending=True)
    fig = px.bar(
        sorted_df, x=col, y="Model", orientation="h",
        title=title, color=col,
        color_continuous_scale=["#10B981", "#F59E0B", "#F43F5E"],
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig)
