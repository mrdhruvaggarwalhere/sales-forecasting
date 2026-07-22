"""
Model Comparison Page.
Displays evaluation metrics (MAE, RMSE, MAPE, R²) for all trained models
and highlights the best performer.
"""

import streamlit as st
import pandas as pd
from components.charts import model_comparison_chart, bar_chart
from components.kpi_cards import render_kpi_row


def render(metrics_df: pd.DataFrame = None):
    """Renders the Model Comparison page."""
    st.markdown('<div class="page-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Side-by-side evaluation of all trained forecasting models</div>', unsafe_allow_html=True)

    if metrics_df is None or metrics_df.empty:
        st.info("No evaluation metrics found. Please run the evaluation pipeline first.")
        return

    # ── Best Model Badge ───────────────────────────────────
    best_idx = metrics_df["RMSE"].idxmin() if "RMSE" in metrics_df.columns else None
    if best_idx is not None:
        best = metrics_df.loc[best_idx]
        render_kpi_row([
            {"label": "Best Model", "value": best["Model"], "accent": "green"},
            {"label": "RMSE", "value": f"{best['RMSE']:,.2f}", "accent": "blue"},
            {"label": "MAE", "value": f"{best['MAE']:,.2f}", "accent": "amber"},
            {"label": "R² Score", "value": f"{best['R2 Score']:.4f}", "accent": "purple"},
        ])

    st.markdown("", unsafe_allow_html=True)

    # ── Metric Tabs ────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "RMSE", "MAE", "MAPE", "R² Score"])

    with tab1:
        st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)

        # Highlight best row
        def highlight_best(row):
            is_best = row.name == best_idx
            return ["background-color: rgba(52,211,153,0.1);" if is_best else "" for _ in row]

        styled = metrics_df.style.apply(highlight_best, axis=1).format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE (%)": "{:.2f}",
            "R2 Score": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab2:
        fig = model_comparison_chart(metrics_df, "RMSE", "Root Mean Squared Error (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = model_comparison_chart(metrics_df, "MAE", "Mean Absolute Error (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if "MAPE (%)" in metrics_df.columns:
            fig = model_comparison_chart(metrics_df, "MAPE (%)", "Mean Absolute % Error (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if "R2 Score" in metrics_df.columns:
            df_sorted = metrics_df.sort_values("R2 Score", ascending=True)
            fig = model_comparison_chart(df_sorted, "R2 Score", "R² Score (Higher is Better)")
            st.plotly_chart(fig, use_container_width=True)
