"""
Reports Page — Download forecasts, datasets, and model performance summaries.
"""

import streamlit as st
import pandas as pd
import json


def render(df: pd.DataFrame, eval_metrics: dict = None, plot_data: dict = None, forecast_csv: pd.DataFrame = None):
    """Renders the Reports page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Reports &amp; Exports</div>'
'<div class="page-subtitle">Download forecasts, datasets, and model performance summaries</div>'
'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    # ── Daily Sales CSV ───────────────────────────────────────
    with col1:
        st.markdown('<div class="section-header">📊 Daily Sales Data</div>', unsafe_allow_html=True)
        if df is not None and not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Sales CSV",
                data=csv,
                file_name="daily_sales.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(df):,} daily records.")
        else:
            st.info("No sales data to export.")

    # ── Forecast Data ─────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-header">🔮 Forecast Data</div>', unsafe_allow_html=True)
        if forecast_csv is not None and not forecast_csv.empty:
            csv = forecast_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Forecast CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(forecast_csv)} forecast records.")
        elif plot_data and "predictions" in plot_data:
            # Export plot_data as JSON
            data_json = json.dumps(plot_data, indent=2).encode("utf-8")
            st.download_button(
                label="⬇ Download Forecast JSON",
                data=data_json,
                file_name="plot_data.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption(f"{len(plot_data.get('dates', []))} forecast points, {len(plot_data['predictions'])} models.")
        else:
            st.info("No forecast data to export.")

    # ── Model Metrics ─────────────────────────────────────────
    with col3:
        st.markdown('<div class="section-header">⚙️ Model Metrics</div>', unsafe_allow_html=True)
        if eval_metrics and isinstance(eval_metrics, dict):
            metrics_json = json.dumps(eval_metrics, indent=2).encode("utf-8")
            st.download_button(
                label="⬇ Download Metrics JSON",
                data=metrics_json,
                file_name="evaluation_metrics.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption(f"{len(eval_metrics)} models evaluated.")
        else:
            st.info("No metrics to export.")

    # ── Executive Summary ─────────────────────────────────────
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Executive Summary</div>', unsafe_allow_html=True)

    lines = []
    if df is not None and not df.empty:
        lines.append(f"- **Total daily records**: {len(df):,}")
        if "sales" in df.columns:
            lines.append(f"- **Total Sales Volume**: {df['sales'].sum():,} units")
            lines.append(f"- **Average Daily Sales**: {df['sales'].mean():,.0f} units")
        if "date" in df.columns:
            lines.append(f"- **Date Range**: {df['date'].min().date()} → {df['date'].max().date()}")

    if eval_metrics and isinstance(eval_metrics, dict):
        best_name = min(eval_metrics, key=lambda k: eval_metrics[k].get("RMSE", float("inf")))
        best_rmse = eval_metrics[best_name]["RMSE"]
        lines.append(f"- **Best Model**: {best_name} (RMSE: {best_rmse:,.2f})")

        for model, m in eval_metrics.items():
            lines.append(f"  - {model}: MAE={m.get('MAE', 'N/A')}, RMSE={m.get('RMSE', 'N/A')}, MAPE={m.get('MAPE', 'N/A')}%")

    if plot_data and "dates" in plot_data:
        lines.append(f"- **Forecast Horizon**: {len(plot_data['dates'])} days")

    if lines:
        st.markdown("\n".join(lines))
        summary_text = "\n".join(lines).replace("**", "").replace("- ", "")
        st.download_button(
            label="⬇ Download Summary (.txt)",
            data=summary_text.encode("utf-8"),
            file_name="executive_summary.txt",
            mime="text/plain",
        )
    else:
        st.info("Run the pipeline to generate a summary.")
