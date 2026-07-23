"""
KPI Card Component — Premium glassmorphic metric cards.
Uses ONLY <div> elements (no <span>) because Streamlit's markdown
sanitizer strips nested <span> tags in certain contexts.
"""

import streamlit as st

DEFAULT_ICONS = {
    "blue": "💰", "green": "📈", "red": "📉", "amber": "⚡", "purple": "🎯",
}


def render_kpi_card(
    label: str,
    value: str,
    delta: str = None,
    delta_positive: bool = True,
    accent: str = "blue",
    icon: str = None,
    subtext: str = "vs previous period",
):
    """Renders a single glassmorphic KPI card using only div elements."""
    display_icon = icon or DEFAULT_ICONS.get(accent, "📊")

    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = (
            f'<div class="kpi-footer-row">'
            f'<div class="kpi-badge {cls}">{arrow} {delta}</div>'
            f'<div class="kpi-subtext">{subtext}</div>'
            f'</div>'
        )

    html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-accent {accent}"></div>'
        f'<div class="kpi-header-row">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-icon-wrapper">{display_icon}</div>'
        f'</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(metrics: list):
    """Renders a responsive row of KPI cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_kpi_card(
                label=m.get("label", ""),
                value=m.get("value", ""),
                delta=m.get("delta"),
                delta_positive=m.get("delta_positive", True),
                accent=m.get("accent", "blue"),
                icon=m.get("icon"),
                subtext=m.get("subtext", "vs prev. period"),
            )
