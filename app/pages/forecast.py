"""
Forecast Page.
Displays historical vs forecast, multi-horizon forecasts, and confidence intervals.
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.charts import forecast_chart, line_chart


def render(df: pd.DataFrame, forecast_df: pd.DataFrame = None):
    """Renders the Forecast page."""
    st.markdown('<div class="page-title">Revenue Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-powered revenue predictions with confidence intervals</div>', unsafe_allow_html=True)

    if forecast_df is None or forecast_df.empty:
        st.info("No forecast data available. Please run the forecasting pipeline first.")
        return

    # Prepare historical aggregated data
    hist_daily = df.groupby("date")["revenue"].sum().reset_index() if not df.empty else pd.DataFrame()

    # Ensure forecast date column is datetime
    if "date" in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # ── Tabs for horizons ──────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Full Forecast", "7-Day", "30-Day", "60-Day", "90-Day"
    ])

    horizons = {
        "7-Day": 7,
        "30-Day": 30,
        "60-Day": 60,
        "90-Day": 90,
    }

    with tab1:
        st.markdown('<div class="section-header">Historical vs Full Forecast</div>', unsafe_allow_html=True)
        point = forecast_df["forecast"].values if "forecast" in forecast_df.columns else np.array([])
        lower = forecast_df["lower_ci"].values if "lower_ci" in forecast_df.columns else None
        upper = forecast_df["upper_ci"].values if "upper_ci" in forecast_df.columns else None
        dates = forecast_df["date"]

        fig = forecast_chart(hist_daily, dates, point, lower, upper, title="Historical vs Forecast Revenue")
        st.plotly_chart(fig, use_container_width=True)

    for tab, (label, days) in zip([tab2, tab3, tab4, tab5], horizons.items()):
        with tab:
            st.markdown(f'<div class="section-header">{label} Forecast</div>', unsafe_allow_html=True)
            subset = forecast_df.head(days)
            if not subset.empty:
                point = subset["forecast"].values
                lower = subset["lower_ci"].values if "lower_ci" in subset.columns else None
                upper = subset["upper_ci"].values if "upper_ci" in subset.columns else None

                fig = forecast_chart(hist_daily, subset["date"], point, lower, upper, title=f"{label} Revenue Forecast")
                st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Daily Forecast", f"${point.mean():,.0f}")
                col2.metric("Total Forecast", f"${point.sum():,.0f}")
                col3.metric("Peak Day", f"${point.max():,.0f}")
            else:
                st.warning(f"Not enough data for {label} forecast.")
