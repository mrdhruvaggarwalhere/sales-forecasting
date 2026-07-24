"""
Sustainability View — AI-Powered Order Consolidation & Environmental Impact Dashboard.

Tab 1: Overview & Eco KPIs
Tab 2: Shipment Optimization Timeline
Tab 3: Carbon & Logistics Cost Comparison
Tab 4: AI Sustainability Recommendations & Inventory Simulation
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from components.sustainability_cards import render_sustainability_kpi_row, render_recommendation_list
from components.optimization_charts import (
    shipment_timeline_chart,
    carbon_comparison_chart,
    savings_donut_chart,
    monthly_carbon_trend_chart,
    inventory_simulation_chart,
)
from components.charts import PLOTLY_CONFIG


def render(daily_df: pd.DataFrame = None, product_forecasts: Dict = None):
    """Renders the Sustainability Dashboard page."""
    st.markdown(
'<div class="page-title-container">'
'<div class="page-title">🌱 Social Impact &amp; Sustainability</div>'
'<div class="page-subtitle">AI-powered shipment consolidation, carbon reduction, and eco-logistics intelligence</div>'
'</div>',
        unsafe_allow_html=True,
    )

    # Load or run sustainability metrics
    sustainability_data = _load_sustainability_data(product_forecasts)
    kpis = sustainability_data.get("kpis", {})

    if not kpis or kpis.get("products_analyzed", 0) == 0:
        st.info("No forecast data available for sustainability analysis. Run the training pipeline first.")
        return

    # ── 6 Dashboard KPI Cards ──────────────────────────────────
    render_sustainability_kpi_row([
        {
            "label": "🌱 Carbon Saved",
            "value": f"{kpis.get('co2_saved_kg', 0):,.1f} kg",
            "icon": "🌍",
            "subtext": f"{sustainability_data.get('carbon_report', {}).get('reduction_pct', 0):.1f}% reduction",
            "accent": "green",
        },
        {
            "label": "🚚 Trips Reduced",
            "value": f"{kpis.get('trips_reduced', 0):,} trips",
            "icon": "🚛",
            "subtext": f"from {kpis.get('naive_trips', 0)} → {kpis.get('optimized_trips', 0)} trips",
            "accent": "emerald",
        },
        {
            "label": "⛽ Fuel Saved",
            "value": f"{kpis.get('fuel_saved_litres', 0):,.1f} L",
            "icon": "⛽",
            "subtext": "diesel fuel avoided",
            "accent": "amber",
        },
        {
            "label": "💰 Logistics Saved",
            "value": f"${kpis.get('cost_saved', 0):,.0f}",
            "icon": "💵",
            "subtext": "freight + fuel savings",
            "accent": "blue",
        },
        {
            "label": "📦 Optimized Orders",
            "value": f"{kpis.get('optimized_order_pct', 0):.1f}%",
            "icon": "📦",
            "subtext": "consolidated orders",
            "accent": "teal",
        },
        {
            "label": "♻ Eco Score",
            "value": f"{kpis.get('sustainability_score', 0)}/100",
            "icon": "🏆",
            "subtext": "sustainability index",
            "accent": "purple",
        },
    ])

    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    # ── 4 Main Tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Impact Overview",
        "🗓️ Optimization Timeline",
        "📉 Emissions Comparison",
        "💡 AI Recommendations",
    ])

    # ── Tab 1: Impact Overview ────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📈 Cumulative Carbon Reduction</div>', unsafe_allow_html=True)
            timeline = sustainability_data.get("inventory_timeline", {})
            fig1 = monthly_carbon_trend_chart(timeline)
            fig1.update_layout(
                title=None,
                height=340,
                margin=dict(l=35, r=15, t=10, b=80),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.24,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10, color="#CBD5E1"),
                ),
            )
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            st.markdown('<div class="section-header">🍩 Carbon Emissions Saved %</div>', unsafe_allow_html=True)
            reduction_pct = sustainability_data.get("carbon_report", {}).get("reduction_pct", 0)
            fig2 = savings_donut_chart(reduction_pct)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        # Summary Metrics Box
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.04);border:1px solid rgba(16,185,129,0.12);'
            f'border-radius:14px;padding:1.2rem;margin-top:1rem;">'
            f'<div style="font-size:0.95rem;font-weight:700;color:#F8FAFC;margin-bottom:0.4rem;">'
            f'🌱 Sustainability Impact Summary</div>'
            f'<div style="font-size:0.82rem;color:#94A3B8;line-height:1.6;">'
            f'Analyzing demand across <b>{kpis.get("products_analyzed", 0)} products</b> over a 30-day forecast horizon. '
            f'Without consolidation, <b>{kpis.get("naive_trips", 0)} separate shipment trips</b> would be required. '
            f'By intelligently bundling orders within a 3-day window, total shipments are reduced to '
            f'<b>{kpis.get("optimized_trips", 0)} consolidated trips</b> — saving '
            f'<b>{kpis.get("co2_saved_kg", 0):,.1f} kg of CO₂</b> and '
            f'<b>${kpis.get("cost_saved", 0):,.0f} in logistics expense</b>.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Tab 2: Optimization Timeline ──────────────────────────
    with tab2:
        st.markdown('<div class="section-header">🗓️ Shipment Order Consolidation Timeline</div>', unsafe_allow_html=True)
        st.caption("Green diamonds = AI-consolidated shipments. Red circles = individual un-consolidated reorders.")

        consolidated = sustainability_data.get("consolidated_shipments", [])
        events = sustainability_data.get("reorder_events", [])
        fig_gantt = shipment_timeline_chart(consolidated, events)
        st.plotly_chart(fig_gantt, use_container_width=True, config=PLOTLY_CONFIG)

        # Consolidated Shipments Table
        if consolidated:
            st.markdown('<div class="section-header">📋 Consolidated Shipment Details</div>', unsafe_allow_html=True)
            table_rows = []
            for s in consolidated:
                table_rows.append({
                    "Date": s["date"],
                    "Products Bundled": ", ".join(s["products"]),
                    "Total Units": f"{s['total_units']:,}",
                    "Products Count": s["products_consolidated"],
                    "Trips Saved": s["trips_saved"],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Tab 3: Emissions Comparison ───────────────────────────
    with tab3:
        st.markdown('<div class="section-header">📉 Current (Naive) vs Optimized Logistics Comparison</div>', unsafe_allow_html=True)

        report = sustainability_data.get("carbon_report", {})
        fig_comp = carbon_comparison_chart(report)
        st.plotly_chart(fig_comp, use_container_width=True, config=PLOTLY_CONFIG)

        # Detailed Breakdown Table
        naive = report.get("naive", {})
        opt = report.get("optimized", {})
        if naive and opt:
            st.markdown('<div class="section-header">📊 Logistics & Emission Metrics Matrix</div>', unsafe_allow_html=True)
            matrix = [
                {"Metric": "Shipment Trips", "Current (Naive)": f"{naive.get('trips', 0)}", "Optimized": f"{opt.get('trips', 0)}", "Saved": f"{kpis.get('trips_reduced', 0)}"},
                {"Metric": "Total Distance (km)", "Current (Naive)": f"{naive.get('distance_km', 0):,.1f}", "Optimized": f"{opt.get('distance_km', 0):,.1f}", "Saved": f"{naive.get('distance_km', 0) - opt.get('distance_km', 0):,.1f}"},
                {"Metric": "Fuel Consumed (L)", "Current (Naive)": f"{naive.get('fuel_litres', 0):,.1f}", "Optimized": f"{opt.get('fuel_litres', 0):,.1f}", "Saved": f"{kpis.get('fuel_saved_litres', 0):,.1f}"},
                {"Metric": "CO₂ Emissions (kg)", "Current (Naive)": f"{naive.get('co2_kg', 0):,.1f}", "Optimized": f"{opt.get('co2_kg', 0):,.1f}", "Saved": f"{kpis.get('co2_saved_kg', 0):,.1f}"},
                {"Metric": "Total Logistics Cost ($)", "Current (Naive)": f"${naive.get('total_cost', 0):,.2f}", "Optimized": f"${opt.get('total_cost', 0):,.2f}", "Saved": f"${kpis.get('cost_saved', 0):,.2f}"},
            ]
            st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)

    # ── Tab 4: AI Recommendations & Inventory Simulation ───────
    with tab4:
        col_rec, col_sim = st.columns([1, 1])

        with col_rec:
            st.markdown('<div class="section-header">🤖 Smart AI Recommendations</div>', unsafe_allow_html=True)
            recs = sustainability_data.get("recommendations", [])
            if recs:
                render_recommendation_list(recs)
            else:
                st.info("No recommendations generated.")

        with col_sim:
            st.markdown('<div class="section-header">📦 Inventory Level Simulation</div>', unsafe_allow_html=True)
            timeline = sustainability_data.get("inventory_timeline", {})
            if timeline:
                selected_prod = st.selectbox("Select Product to Inspect", list(timeline.keys()))
                snapshots = timeline[selected_prod]
                cfg = sustainability_data.get("config", {})
                thresh = int(cfg.get("reorder_threshold_pct", 0.25) * cfg.get("max_inventory_capacity", 2000))
                cap = cfg.get("max_inventory_capacity", 2000)

                fig_sim = inventory_simulation_chart(selected_prod, snapshots, thresh, cap)
                st.plotly_chart(fig_sim, use_container_width=True, config=PLOTLY_CONFIG)


def _load_sustainability_data(product_forecasts: Dict = None) -> Dict[str, Any]:
    """Loads pre-calculated sustainability results or runs analysis on the fly."""
    from src.sustainability.sustainability_metrics import run_sustainability_analysis

    return run_sustainability_analysis(product_forecasts)
