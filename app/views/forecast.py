"""
Forecast Page — Actual vs Predicted overlay for all models.
Uses plot_data.json (pre-computed or pipeline-generated).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.kpi_cards import render_kpi_row
from components.charts import apply_theme, PLOTLY_CONFIG, PALETTE


def render(df: pd.DataFrame, plot_data: dict = None, forecast_csv: pd.DataFrame = None):
    """Renders the Forecast page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Revenue Forecast Intelligence</div>'
'<div class="page-subtitle">Multi-model predictions with actual vs forecasted comparison</div>'
'</div>',
        unsafe_allow_html=True,
    )

    if not plot_data or "dates" not in plot_data:
        st.info("No forecast data available. Run `python main.py --train` to generate forecasts.")
        return

    dates = pd.to_datetime(plot_data["dates"])
    actuals = np.array(plot_data["actuals"])
    predictions = plot_data.get("predictions", {})

    if not predictions:
        st.warning("No model predictions found in plot data.")
        return

    # ── KPI Row ───────────────────────────────────────────────
    model_names = list(predictions.keys())
    render_kpi_row([
        {"label": "Forecast Horizon", "value": f"{len(dates)} days", "accent": "blue", "icon": "📅",
         "subtext": f"{dates.min().date()} to {dates.max().date()}"},
        {"label": "Avg Actual Sales", "value": f"{actuals.mean():,.0f}", "accent": "green", "icon": "📊",
         "subtext": "daily average"},
        {"label": "Models Compared", "value": str(len(model_names)), "accent": "purple", "icon": "🤖",
         "subtext": ", ".join(model_names[:3])},
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    # ── Tabs: All Models + Individual ─────────────────────────
    tab_names = ["All Models"] + model_names
    tabs = st.tabs(tab_names)

    colors = ["#10B981", "#F43F5E", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899"]

    with tabs[0]:
        st.markdown('<div class="section-header">📈 Actual vs All Model Forecasts</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=actuals, mode="lines", name="Actual",
            line=dict(color="#3B82F6", width=3),
        ))
        for idx, (name, preds) in enumerate(predictions.items()):
            pred_arr = np.array(preds)
            n = min(len(dates), len(pred_arr))
            fig.add_trace(go.Scatter(
                x=dates[:n], y=pred_arr[:n], mode="lines", name=name,
                line=dict(color=colors[idx % len(colors)], width=2, dash="dash"),
                opacity=0.8,
            ))
        fig.update_layout(title="30-Day Forecast: Actual vs All Models")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    for i, model_name in enumerate(model_names):
        with tabs[i + 1]:
            st.markdown(f'<div class="section-header">🔮 {model_name} — Forecast Detail</div>', unsafe_allow_html=True)
            pred_arr = np.array(predictions[model_name])
            n = min(len(dates), len(pred_arr))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates[:n], y=actuals[:n], mode="lines", name="Actual",
                line=dict(color="#3B82F6", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=dates[:n], y=pred_arr[:n], mode="lines+markers", name=model_name,
                line=dict(color=colors[i % len(colors)], width=2.5),
                marker=dict(size=5),
            ))
            fig.update_layout(title=f"{model_name} — Forecast vs Actual")
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            # Per-model error metrics
            mae = float(np.mean(np.abs(actuals[:n] - pred_arr[:n])))
            rmse = float(np.sqrt(np.mean((actuals[:n] - pred_arr[:n]) ** 2)))
            render_kpi_row([
                {"label": "MAE", "value": f"{mae:,.0f}", "accent": "amber", "icon": "📐"},
                {"label": "RMSE", "value": f"{rmse:,.0f}", "accent": "red", "icon": "📉"},
                {"label": "Avg Forecast", "value": f"{pred_arr[:n].mean():,.0f}", "accent": "blue", "icon": "📊"},
            ])
