"""
Optimization Charts — Plotly visualizers for sustainability metrics.

Includes:
- Reorder Timeline / Shipment Gantt chart
- Carbon & Cost Comparison (Naive vs Optimized)
- Savings Donut Chart
- Monthly Carbon Trend
- Inventory Level Simulation Chart
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from components.charts import apply_theme, PLOTLY_CONFIG, PALETTE


def shipment_timeline_chart(consolidated_shipments: List[Dict], reorder_events: List[Dict]) -> go.Figure:
    """
    Gantt/timeline chart showing reorder events and consolidated shipments over time.
    """
    fig = go.Figure()

    # Plot raw reorder events (red markers for standalone)
    event_dates = [e["date"] for e in reorder_events]
    event_products = [e["product"] for e in reorder_events]
    event_qty = [e["quantity_needed"] for e in reorder_events]

    fig.add_trace(go.Scatter(
        x=event_dates,
        y=event_products,
        mode="markers",
        name="Individual Reorders",
        marker=dict(
            size=[min(30, max(10, q // 30)) for q in event_qty],
            color="#F43F5E",
            symbol="circle-open",
            line=dict(width=2, color="#F43F5E"),
        ),
        hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Qty Needed: %{text} units<extra></extra>",
        text=event_qty,
    ))

    # Plot consolidated shipments (green filled markers)
    cons_dates = []
    cons_products = []
    cons_units = []

    for ship in consolidated_shipments:
        for prod, qty in ship["quantities"].items():
            cons_dates.append(ship["date"])
            cons_products.append(prod)
            cons_units.append(qty)

    fig.add_trace(go.Scatter(
        x=cons_dates,
        y=cons_products,
        mode="markers",
        name="Consolidated Shipments",
        marker=dict(
            size=[min(30, max(12, u // 30)) for u in cons_units],
            color="#10B981",
            symbol="diamond",
        ),
        hovertemplate="<b>%{y}</b><br>Shipment Date: %{x}<br>Consolidated Qty: %{text} units<extra></extra>",
        text=cons_units,
    ))

    fig.update_layout(
        title="Shipment Optimization Timeline: Raw Reorders vs Consolidated Shipments",
        xaxis_title="Date",
        yaxis_title="Product",
        hovermode="closest",
    )
    return apply_theme(fig)


def carbon_comparison_chart(carbon_report: Dict) -> go.Figure:
    """
    Grouped bar chart comparing Naive vs Optimized across 4 dimensions:
    Trips, Distance (km), Fuel (L), CO2 (kg).
    """
    naive = carbon_report.get("naive", {})
    optimized = carbon_report.get("optimized", {})

    categories = ["Shipment Trips", "Fuel (Litres)", "CO₂ Emissions (kg)", "Logistics Cost ($)"]
    naive_vals = [
        naive.get("trips", 0),
        naive.get("fuel_litres", 0),
        naive.get("co2_kg", 0),
        naive.get("total_cost", 0),
    ]
    opt_vals = [
        optimized.get("trips", 0),
        optimized.get("fuel_litres", 0),
        optimized.get("co2_kg", 0),
        optimized.get("total_cost", 0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=naive_vals,
        name="Current (Naive)",
        marker_color="#F43F5E",
        text=[f"{v:,.0f}" for v in naive_vals],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=opt_vals,
        name="Optimized (AI Consolidated)",
        marker_color="#10B981",
        text=[f"{v:,.0f}" for v in opt_vals],
        textposition="auto",
    ))

    fig.update_layout(
        title="Logistics Metrics Comparison: Current vs Optimized",
        barmode="group",
    )
    return apply_theme(fig)


def savings_donut_chart(reduction_pct: float) -> go.Figure:
    """Donut chart highlighting carbon emissions reduced vs remaining."""
    saved_pct = max(0.0, min(100.0, reduction_pct))
    rem_pct = 100.0 - saved_pct

    fig = px.pie(
        values=[saved_pct, rem_pct],
        names=["CO₂ Reduced", "Remaining Emissions"],
        title="Carbon Emission Reduction %",
        hole=0.6,
        color_discrete_sequence=["#10B981", "#1E293B"],
    )
    fig.update_traces(textinfo="percent+label", textposition="inside")
    return apply_theme(fig)


def monthly_carbon_trend_chart(inventory_timeline: Dict) -> go.Figure:
    """
    Plots cumulative CO₂ saved over the 30-day horizon as orders are consolidated.
    """
    if not inventory_timeline:
        return go.Figure()

    # Pick first product's timeline for dates
    first_product = list(inventory_timeline.keys())[0]
    dates = [s["date"] for s in inventory_timeline[first_product]]

    # Estimate daily CO2 baseline vs optimized
    daily_naive_co2 = []
    daily_opt_co2 = []

    for day_idx in range(len(dates)):
        day_reorders = sum(
            1 for snapshots in inventory_timeline.values()
            if day_idx < len(snapshots) and snapshots[day_idx]["reorder_triggered"]
        )
        naive_co2 = day_reorders * (150 * 2 * 0.35 * 2.68)
        opt_co2 = (1 if day_reorders > 0 else 0) * (150 * 2 * 0.35 * 2.68)

        daily_naive_co2.append(naive_co2)
        daily_opt_co2.append(opt_co2)

    cum_naive = np.cumsum(daily_naive_co2)
    cum_opt = np.cumsum(daily_opt_co2)
    cum_saved = cum_naive - cum_opt

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=cum_naive, mode="lines", name="Cumulative CO₂ (Current)",
        line=dict(color="#F43F5E", width=2, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=cum_opt, mode="lines", name="Cumulative CO₂ (Optimized)",
        line=dict(color="#10B981", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=cum_saved, mode="lines", name="Cumulative CO₂ Saved",
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
        line=dict(color="#059669", width=2),
    ))

    fig.update_layout(
        title="30-Day Cumulative Carbon Emissions & Savings (kg CO₂)",
        xaxis_title="Date",
        yaxis_title="CO₂ Emissions (kg)",
    )
    return apply_theme(fig)


def inventory_simulation_chart(product: str, snapshots: List[Dict], threshold: int, capacity: int) -> go.Figure:
    """
    Plots a single product's inventory trajectory over 30 days,
    showing inventory consumption, reorder threshold line, and refill spikes.
    """
    dates = [s["date"] for s in snapshots]
    inv_levels = [s["inventory_level"] for s in snapshots]
    demands = [s["demand"] for s in snapshots]

    fig = go.Figure()

    # Inventory level line
    fig.add_trace(go.Scatter(
        x=dates, y=inv_levels, mode="lines+markers", name="Inventory Level",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=4),
    ))

    # Daily demand bars
    fig.add_trace(go.Bar(
        x=dates, y=demands, name="Daily Demand",
        marker_color="rgba(245,158,11,0.4)",
    ))

    # Reorder threshold line
    fig.add_shape(
        type="line", x0=dates[0], x1=dates[-1], y0=threshold, y1=threshold,
        line=dict(color="#F43F5E", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=dates[-1], y=threshold, text=f"Reorder Threshold ({threshold} u)",
        showarrow=False, yshift=10, font=dict(color="#F43F5E", size=10),
    )

    fig.update_layout(
        title=f"Inventory Simulation — {product} (Max Capacity: {capacity} u)",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
    )
    return apply_theme(fig)
