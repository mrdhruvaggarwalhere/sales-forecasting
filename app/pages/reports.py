"""
Reports Page.
Provides export/download functionality for forecasts, charts, and summaries.
"""

import streamlit as st
import pandas as pd
from io import BytesIO


def render(df: pd.DataFrame, forecast_df: pd.DataFrame = None, metrics_df: pd.DataFrame = None):
    """Renders the Reports page."""
    st.markdown('<div class="page-title">Reports & Exports</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Download forecasts, datasets, and model performance summaries</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ── Forecast CSV ───────────────────────────────────────
    with col1:
        st.markdown('<div class="section-header">📄 Forecast Data</div>', unsafe_allow_html=True)
        if forecast_df is not None and not forecast_df.empty:
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Forecast CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(forecast_df)} forecast records available.")
        else:
            st.info("No forecast data to export.")

    # ── Processed Data ─────────────────────────────────────
    with col2:
        st.markdown('<div class="section-header">📊 Processed Data</div>', unsafe_allow_html=True)
        if df is not None and not df.empty:
            # Export a sample (full dataset could be massive)
            sample = df.head(50_000)
            csv = sample.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Data Sample (50K rows)",
                data=csv,
                file_name="processed_data_sample.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"Showing first 50,000 of {len(df):,} records.")
        else:
            st.info("No processed data to export.")

    # ── Model Metrics ──────────────────────────────────────
    with col3:
        st.markdown('<div class="section-header">⚙️ Model Metrics</div>', unsafe_allow_html=True)
        if metrics_df is not None and not metrics_df.empty:
            csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Metrics CSV",
                data=csv,
                file_name="model_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(metrics_df)} models evaluated.")
        else:
            st.info("No metrics to export.")

    # ── Executive Summary ──────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Executive Summary</div>', unsafe_allow_html=True)

    summary_lines = []
    if df is not None and not df.empty:
        summary_lines.append(f"- **Total records analysed**: {len(df):,}")
        if "revenue" in df.columns:
            summary_lines.append(f"- **Total Revenue**: ${df['revenue'].sum():,.0f}")
        if "sales_units" in df.columns:
            summary_lines.append(f"- **Total Units Sold**: {df['sales_units'].sum():,}")

    if metrics_df is not None and not metrics_df.empty and "RMSE" in metrics_df.columns:
        best_idx = metrics_df["RMSE"].idxmin()
        best = metrics_df.loc[best_idx]
        summary_lines.append(f"- **Best Model**: {best['Model']} (RMSE: {best['RMSE']:,.2f})")

    if forecast_df is not None and not forecast_df.empty and "forecast" in forecast_df.columns:
        summary_lines.append(f"- **Forecast Horizon**: {len(forecast_df)} days")
        summary_lines.append(f"- **Avg Forecasted Revenue**: ${forecast_df['forecast'].mean():,.0f}/day")

    if summary_lines:
        st.markdown("\n".join(summary_lines))

        # Downloadable text summary
        summary_text = "\n".join(summary_lines).replace("**", "").replace("- ", "")
        st.download_button(
            label="⬇ Download Summary (.txt)",
            data=summary_text.encode("utf-8"),
            file_name="executive_summary.txt",
            mime="text/plain",
        )
    else:
        st.info("Run the pipeline to generate a summary.")
