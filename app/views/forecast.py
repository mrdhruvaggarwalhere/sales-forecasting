"""
Forecast Page - Actual vs predicted overlay for all models.
Uses plot_data.json for curves and official evaluation metrics for KPI cards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.kpi_cards import render_kpi_row
from components.charts import apply_theme, PLOTLY_CONFIG


def render(
    df: pd.DataFrame,
    plot_data: dict = None,
    forecast_csv: pd.DataFrame = None,
    eval_metrics: dict = None,
):
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

    model_names = list(predictions.keys())
    render_kpi_row([
        {
            "label": "Forecast Horizon",
            "value": f"{len(dates)} days",
            "accent": "blue",
            "subtext": f"{dates.min().date()} to {dates.max().date()}",
        },
        {
            "label": "Avg Actual Sales",
            "value": f"{actuals.mean():,.0f}",
            "accent": "green",
            "subtext": "daily average",
        },
        {
            "label": "Models Compared",
            "value": str(len(model_names)),
            "accent": "purple",
            "subtext": ", ".join(model_names[:3]),
        },
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    tab_names = ["All Models"] + model_names
    tabs = st.tabs(tab_names)
    colors = ["#10B981", "#F43F5E", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899"]

    with tabs[0]:
        st.markdown('<div class="section-header">Actual vs All Model Forecasts</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=actuals,
            mode="lines",
            name="Actual",
            line=dict(color="#3B82F6", width=3),
        ))

        for idx, (name, preds) in enumerate(predictions.items()):
            pred_arr = np.array(preds)
            n = min(len(dates), len(pred_arr))
            fig.add_trace(go.Scatter(
                x=dates[:n],
                y=pred_arr[:n],
                mode="lines",
                name=name,
                line=dict(color=colors[idx % len(colors)], width=2, dash="dash"),
                opacity=0.8,
            ))

        fig.update_layout(title="30-Day Forecast: Actual vs All Models")
        st.plotly_chart(apply_theme(fig), use_container_width=True, config=PLOTLY_CONFIG)

    for i, model_name in enumerate(model_names):
        with tabs[i + 1]:
            st.markdown(f'<div class="section-header">{model_name} - Forecast Detail</div>', unsafe_allow_html=True)
            pred_arr = np.array(predictions[model_name])
            n = min(len(dates), len(pred_arr))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates[:n],
                y=actuals[:n],
                mode="lines",
                name="Actual",
                line=dict(color="#3B82F6", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=dates[:n],
                y=pred_arr[:n],
                mode="lines+markers",
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2.5),
                marker=dict(size=5),
            ))

            fig.update_layout(title=f"{model_name} - Forecast vs Actual")
            st.plotly_chart(apply_theme(fig), use_container_width=True, config=PLOTLY_CONFIG)

            mae, rmse, mape = _resolve_model_metrics(eval_metrics, model_name, actuals[:n], pred_arr[:n])
            render_kpi_row([
                {"label": "MAE", "value": f"{mae:,.2f}", "accent": "amber"},
                {"label": "RMSE", "value": f"{rmse:,.2f}", "accent": "red"},
                {"label": "MAPE", "value": f"{mape:.2f}%", "accent": "blue"},
            ])


def _resolve_model_metrics(eval_metrics: dict, model_name: str, actuals: np.ndarray, preds: np.ndarray):
    """Uses official metrics when available, otherwise derives metrics from plotted data."""
    official = _get_official_metrics(eval_metrics, model_name)
    if official:
        return (
            float(official.get("MAE", 0)),
            float(official.get("RMSE", 0)),
            float(official.get("MAPE", official.get("MAPE (%)", 0))),
        )

    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    non_zero = actuals != 0
    mape = float(np.mean(np.abs((actuals[non_zero] - preds[non_zero]) / actuals[non_zero])) * 100) if np.any(non_zero) else 0.0
    return mae, rmse, mape


def _get_official_metrics(eval_metrics: dict, model_name: str) -> dict:
    """Looks up model metrics from the official evaluation source."""
    if not eval_metrics or not isinstance(eval_metrics, dict):
        return {}

    if model_name in eval_metrics and isinstance(eval_metrics[model_name], dict):
        return eval_metrics[model_name]

    normalized = model_name.replace(" ", "").lower()
    for name, metrics in eval_metrics.items():
        if name.replace(" ", "").lower() == normalized and isinstance(metrics, dict):
            return metrics

    return {}
