"""
Reusable Plotly Chart Functions for the Streamlit Dashboard.
All functions return styled Plotly figure objects with a ultra-sleek enterprise dark theme.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Ultra-Sleek Enterprise Chart Theme ──────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#94A3B8"),
    title_font=dict(size=15, color="#F8FAFC", family="Plus Jakarta Sans, sans-serif", weight="bold"),
    margin=dict(l=30, r=20, t=50, b=30),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#171E2C",
        font_size=12,
        font_family="Plus Jakarta Sans, sans-serif",
        bordercolor="rgba(255,255,255,0.15)"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#CBD5E1"),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)", 
        zerolinecolor="rgba(255,255,255,0.06)",
        showline=True,
        linecolor="rgba(255,255,255,0.1)"
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)", 
        zerolinecolor="rgba(255,255,255,0.06)",
        showline=True,
        linecolor="rgba(255,255,255,0.1)"
    ),
)

PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#06B6D4", "#EC4899"]

# Config to pass to st.plotly_chart to avoid WebGL issues
PLOTLY_CONFIG = {"displayModeBar": False}


def apply_theme(fig: go.Figure) -> go.Figure:
    """Applies the enterprise dark theme to any Plotly figure."""
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ── Line Charts ────────────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """Creates a styled line chart with smooth curves."""
    fig = px.line(df, x=x, y=y, title=title, color=color, color_discrete_sequence=PALETTE)
    fig.update_traces(line=dict(width=2.5))
    return apply_theme(fig)


def multi_line_chart(df: pd.DataFrame, x: str, y_cols: list, title: str) -> go.Figure:
    """Creates a multi-line chart from multiple y columns."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))
    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Bar Charts ─────────────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None, horizontal: bool = False) -> go.Figure:
    """Creates a styled bar chart with rounded corners."""
    if horizontal:
        fig = px.bar(df, x=y, y=x, title=title, color=color, orientation="h", color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color, color_discrete_sequence=PALETTE)
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return apply_theme(fig)


# ── Pie / Donut Charts ────────────────────────────────────────────────────────

def donut_chart(df: pd.DataFrame, values: str, names: str, title: str) -> go.Figure:
    """Creates a sleek donut chart."""
    fig = px.pie(df, values=values, names=names, title=title, hole=0.55, color_discrete_sequence=PALETTE)
    fig.update_traces(
        textposition="inside", 
        textinfo="percent+label",
        marker=dict(line=dict(color="#0B0E14", width=2))
    )
    return apply_theme(fig)


# ── Histogram ──────────────────────────────────────────────────────────────────

def histogram_chart(df: pd.DataFrame, x: str, title: str, nbins: int = 40) -> go.Figure:
    """Creates a styled histogram."""
    fig = px.histogram(df, x=x, title=title, nbins=nbins, color_discrete_sequence=PALETTE)
    fig.update_traces(opacity=0.85, marker_line_width=0)
    return apply_theme(fig)


# ── Scatter ────────────────────────────────────────────────────────────────────

def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """Creates a styled scatter plot."""
    fig = px.scatter(df, x=x, y=y, title=title, color=color, opacity=0.75, color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=7))
    return apply_theme(fig)


# ── Forecast Chart (with Confidence Interval) ────────────────────────────────

def forecast_chart(
    historical: pd.DataFrame,
    forecast_dates: pd.Series,
    point: np.ndarray,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    title: str = "Revenue Forecast Projection",
) -> go.Figure:
    """Creates a historical + forecast line chart with confidence bands."""
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["revenue"],
        mode="lines", name="Historical Sales",
        line=dict(color="#3B82F6", width=2.5),
    ))

    # Point forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=point,
        mode="lines+markers", name="Model Forecast",
        line=dict(color="#F59E0B", width=3, dash="dash"),
        marker=dict(size=6, symbol="circle"),
    ))

    # Confidence interval band
    if lower is not None and upper is not None and not np.isnan(lower).all():
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% Confidence Band",
        ))

    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Heatmap ────────────────────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, cols: list, title: str = "Feature Correlation Matrix") -> go.Figure:
    """Creates a correlation heatmap."""
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="Viridis",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Model Comparison Bar Chart ─────────────────────────────────────────────────

def model_comparison_chart(metrics_df: pd.DataFrame, metric_col: str, title: str) -> go.Figure:
    """Creates a horizontal bar chart comparing model performance metrics."""
    df_sorted = metrics_df.sort_values(metric_col, ascending=True)
    fig = px.bar(
        df_sorted, x=metric_col, y="Model", orientation="h",
        title=title, color=metric_col,
        color_continuous_scale=["#10B981", "#F59E0B", "#F43F5E"],
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig)
