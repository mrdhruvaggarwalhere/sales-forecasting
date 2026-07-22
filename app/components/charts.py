"""
Reusable Plotly Chart Functions for the Streamlit Dashboard.
All functions return styled Plotly figure objects with a consistent enterprise theme.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Enterprise Chart Theme ─────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#A3A8B8"),
    title_font=dict(size=14, color="#F0F2F6", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="#2D3044", zerolinecolor="#2D3044"),
    yaxis=dict(gridcolor="#2D3044", zerolinecolor="#2D3044"),
)

PALETTE = ["#4F8CFF", "#34D399", "#FBBF24", "#F87171", "#A78BFA", "#60A5FA", "#F472B6"]


def apply_theme(fig: go.Figure) -> go.Figure:
    """Applies the enterprise dark theme to any Plotly figure."""
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ── Line Charts ────────────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """Creates a styled line chart."""
    fig = px.line(df, x=x, y=y, title=title, color=color, color_discrete_sequence=PALETTE)
    return apply_theme(fig)


def multi_line_chart(df: pd.DataFrame, x: str, y_cols: list, title: str) -> go.Figure:
    """Creates a multi-line chart from multiple y columns."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Bar Charts ─────────────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None, horizontal: bool = False) -> go.Figure:
    """Creates a styled bar chart."""
    if horizontal:
        fig = px.bar(df, x=y, y=x, title=title, color=color, orientation="h", color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color, color_discrete_sequence=PALETTE)
    return apply_theme(fig)


# ── Pie / Donut Charts ────────────────────────────────────────────────────────

def donut_chart(df: pd.DataFrame, values: str, names: str, title: str) -> go.Figure:
    """Creates a donut chart."""
    fig = px.pie(df, values=values, names=names, title=title, hole=0.45, color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(fig)


# ── Histogram ──────────────────────────────────────────────────────────────────

def histogram_chart(df: pd.DataFrame, x: str, title: str, nbins: int = 40) -> go.Figure:
    """Creates a styled histogram."""
    fig = px.histogram(df, x=x, title=title, nbins=nbins, color_discrete_sequence=PALETTE)
    return apply_theme(fig)


# ── Scatter ────────────────────────────────────────────────────────────────────

def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """Creates a styled scatter plot."""
    fig = px.scatter(df, x=x, y=y, title=title, color=color, opacity=0.6, color_discrete_sequence=PALETTE)
    return apply_theme(fig)


# ── Forecast Chart (with Confidence Interval) ────────────────────────────────

def forecast_chart(
    historical: pd.DataFrame,
    forecast_dates: pd.Series,
    point: np.ndarray,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    title: str = "Revenue Forecast",
) -> go.Figure:
    """Creates a historical + forecast line chart with optional confidence bands."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["revenue"],
        mode="lines", name="Historical",
        line=dict(color="#4F8CFF", width=2),
    ))

    # Point forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=point,
        mode="lines+markers", name="Forecast",
        line=dict(color="#FBBF24", width=3, dash="dot"),
        marker=dict(size=5),
    ))

    # Confidence interval
    if lower is not None and upper is not None and not np.isnan(lower).all():
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_dates, forecast_dates[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(251,191,36,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        ))

    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Heatmap ────────────────────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, cols: list, title: str = "Feature Correlation") -> go.Figure:
    """Creates a correlation heatmap."""
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title=title)
    return apply_theme(fig)


# ── Model Comparison Bar Chart ─────────────────────────────────────────────────

def model_comparison_chart(metrics_df: pd.DataFrame, metric_col: str, title: str) -> go.Figure:
    """Creates a horizontal bar chart comparing model metrics."""
    df_sorted = metrics_df.sort_values(metric_col, ascending=True)
    fig = px.bar(
        df_sorted, x=metric_col, y="Model", orientation="h",
        title=title, color=metric_col,
        color_continuous_scale=["#34D399", "#FBBF24", "#F87171"],
    )
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig)
