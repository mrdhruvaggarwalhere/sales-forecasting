"""
Product Analysis Page — Top products, daily trends, and per-product forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.charts import bar_chart, apply_theme, PALETTE, PLOTLY_CONFIG
from components.kpi_cards import render_kpi_row


def render(top_products: pd.DataFrame, top5_daily: pd.DataFrame, product_forecasts: dict):
    """Renders the Product Analysis page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">Product Analysis</div>'
'<div class="page-subtitle">Top-selling products, daily trends, and per-product forecast accuracy</div>'
'</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Top 50 Products", "Top 5 Daily Trends", "Product Forecasts"])

    with tab1:
        if top_products.empty:
            st.info("No top products data found.")
        else:
            st.markdown('<div class="section-header">🏆 Top 50 Products by Total Volume</div>', unsafe_allow_html=True)
            top1 = top_products.iloc[0]
            render_kpi_row([
                {"label": "#1 Product", "value": str(top1["item_id"]), "accent": "green", "icon": "🥇",
                 "subtext": f"{int(top1['total_volume']):,} units"},
                {"label": "Top 50 Total", "value": f"{int(top_products['total_volume'].sum()):,}", "accent": "blue", "icon": "📦"},
                {"label": "Avg Volume", "value": f"{int(top_products['total_volume'].mean()):,}", "accent": "amber", "icon": "📊"},
            ])
            st.markdown('<div style="margin-bottom:1rem;"></div>', unsafe_allow_html=True)
            top20 = top_products.head(20)
            fig = bar_chart(top20, "item_id", "total_volume", "Top 20 Products by Sales Volume")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown('<div class="section-header">📋 Full Top 50 Table</div>', unsafe_allow_html=True)
            st.dataframe(top_products, use_container_width=True, hide_index=True)

    with tab2:
        if top5_daily.empty:
            st.info("No daily product data found.")
        else:
            st.markdown('<div class="section-header">📈 Daily Sales — Top 5 Products</div>', unsafe_allow_html=True)
            product_cols = [c for c in top5_daily.columns if c.startswith("FOODS_") or c.startswith("HOBBIES_") or c.startswith("HOUSEHOLD_")]
            if product_cols and "date" in top5_daily.columns:
                fig = go.Figure()
                for i, col in enumerate(product_cols):
                    fig.add_trace(go.Scatter(
                        x=top5_daily["date"], y=top5_daily[col],
                        mode="lines", name=col,
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    ))
                fig.update_layout(title="Daily Sales Volume — Top 5 Products")
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab3:
        if not product_forecasts or "data" not in product_forecasts:
            st.info("No product forecast data found.")
        else:
            st.markdown('<div class="section-header">🔮 Per-Product Forecast vs Actual</div>', unsafe_allow_html=True)
            dates = pd.to_datetime(product_forecasts.get("dates", []))
            metrics = product_forecasts.get("metrics", {})
            data = product_forecasts.get("data", {})

            if metrics:
                metrics_rows = [{"Product": p, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE (%)": m["MAPE"]}
                                for p, m in metrics.items()]
                st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

            st.markdown('<div style="margin-bottom:1rem;"></div>', unsafe_allow_html=True)
            for idx, (prod_name, prod_data) in enumerate(data.items()):
                actuals = prod_data.get("actuals", [])
                preds = prod_data.get("predictions", [])
                n = min(len(dates), len(actuals), len(preds))
                if n > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates[:n], y=actuals[:n], mode="lines", name="Actual",
                                            line=dict(color="#3B82F6", width=2.5)))
                    fig.add_trace(go.Scatter(x=dates[:n], y=preds[:n], mode="lines+markers", name="Predicted",
                                            line=dict(color="#F59E0B", width=2, dash="dash"), marker=dict(size=4)))
                    fig.update_layout(title=f"{prod_name} — Forecast vs Actual")
                    fig = apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
