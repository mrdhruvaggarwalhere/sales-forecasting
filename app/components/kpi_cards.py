"""
Reusable KPI Card Components for the Streamlit Dashboard.
Renders enterprise-style metric cards with accent bars, deltas, and icons.
"""

import streamlit as st


def render_kpi_card(label: str, value: str, delta: str = None, delta_positive: bool = True, accent: str = "blue"):
    """
    Renders a single KPI metric card using custom HTML/CSS.

    Args:
        label: The metric label (e.g., 'Total Revenue').
        value: The formatted metric value (e.g., '$1.2M').
        delta: Optional delta string (e.g., '+12.4%').
        delta_positive: If True, delta is green; otherwise red.
        accent: Accent color class ('blue', 'green', 'red', 'amber', 'purple').
    """
    delta_class = "positive" if delta_positive else "negative"
    delta_icon = "↑" if delta_positive else "↓"
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_icon} {delta}</div>'

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-accent {accent}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_row(metrics: list):
    """
    Renders a row of KPI cards.

    Args:
        metrics: List of dicts, each with keys: label, value, delta (optional),
                 delta_positive (optional, default True), accent (optional, default 'blue').
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_kpi_card(
                label=m.get("label", ""),
                value=m.get("value", ""),
                delta=m.get("delta"),
                delta_positive=m.get("delta_positive", True),
                accent=m.get("accent", "blue"),
            )
