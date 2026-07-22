"""
app.py — Main Entry Point for the Sales Revenue Forecasting Dashboard.

Launch with:
    streamlit run app/app.py

This module:
1. Loads and injects the custom CSS.
2. Loads cached datasets (processed data, forecasts, evaluation metrics).
3. Renders the sidebar with navigation and filters.
4. Routes to the selected page module.
"""

import sys
import os
import streamlit as st
import pandas as pd
from pathlib import Path

# ── Ensure app/ is on the Python path ──────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

PROJECT_ROOT = APP_DIR.parent

# ── Page Configuration (must be FIRST Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Sales Revenue Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ────────────────────────────────────────────────────────────
css_path = APP_DIR / "styles.css"
if css_path.is_file():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Import Components & Pages ──────────────────────────────────────────────────
from components.sidebar import render_sidebar, apply_filters
from pages import home, sales_analysis, forecast, product_analysis
from pages import revenue_analysis, model_comparison, reports


# ── Cached Data Loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading processed data...")
def load_processed_data() -> pd.DataFrame:
    """Loads the feature-engineered dataset."""
    path = PROJECT_ROOT / "data" / "processed" / "featured_sales.csv"
    if path.is_file():
        df = pd.read_csv(path, nrows=500_000)  # safety cap for RAM
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading forecast data...")
def load_forecast_data() -> pd.DataFrame:
    """Loads the forecast output CSV."""
    path = PROJECT_ROOT / "data" / "predictions" / "forecast.csv"
    if path.is_file():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading model metrics...")
def load_metrics_data() -> pd.DataFrame:
    """Loads the model evaluation metrics CSV."""
    path = PROJECT_ROOT / "outputs" / "models" / "evaluation_metrics.csv"
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load data
    df = load_processed_data()
    forecast_df = load_forecast_data()
    metrics_df = load_metrics_data()

    # Render sidebar (navigation + filters)
    sidebar_state = render_sidebar(df)
    active_page = sidebar_state["page"]
    filters = sidebar_state["filters"]

    # Apply filters to the main dataset
    filtered_df = apply_filters(df, filters) if not df.empty else df

    # ── Page Router ────────────────────────────────────────
    if active_page == "Home":
        home.render(filtered_df, forecast_df, metrics_df)

    elif active_page == "Sales Analysis":
        sales_analysis.render(filtered_df)

    elif active_page == "Forecast":
        forecast.render(filtered_df, forecast_df)

    elif active_page == "Product Analysis":
        product_analysis.render(filtered_df)

    elif active_page == "Revenue Analysis":
        revenue_analysis.render(filtered_df)

    elif active_page == "Model Comparison":
        model_comparison.render(metrics_df)

    elif active_page == "Reports":
        reports.render(filtered_df, forecast_df, metrics_df)


if __name__ == "__main__":
    main()
