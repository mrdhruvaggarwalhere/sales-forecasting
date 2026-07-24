"""
Sustainability Cards — Eco-themed KPI and recommendation card components.
Uses green/eco color palette and div-only HTML (no span) for Streamlit compatibility.
"""

import streamlit as st
from typing import List, Dict


# ── Eco KPI Card ──────────────────────────────────────────────────────────────

def render_sustainability_kpi(label: str, value: str, icon: str, subtext: str = "", accent: str = "green"):
    """Renders a single eco-themed KPI card."""
    accent_colors = {
        "green": "linear-gradient(135deg, #059669 0%, #10B981 100%)",
        "blue": "linear-gradient(135deg, #2563EB 0%, #3B82F6 100%)",
        "amber": "linear-gradient(135deg, #D97706 0%, #F59E0B 100%)",
        "purple": "linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%)",
        "emerald": "linear-gradient(135deg, #047857 0%, #34D399 100%)",
        "teal": "linear-gradient(135deg, #0D9488 0%, #14B8A6 100%)",
    }
    bg = accent_colors.get(accent, accent_colors["green"])

    html = (
        f'<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);'
        f'border-radius:16px;padding:1.2rem;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;width:100%;height:3px;background:{bg};"></div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">'
        f'<div style="font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>'
        f'<div style="font-size:1.3rem;">{icon}</div>'
        f'</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.03em;margin-bottom:0.3rem;">{value}</div>'
        f'<div style="font-size:0.72rem;color:#64748B;font-weight:500;">{subtext}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_sustainability_kpi_row(metrics: List[Dict]):
    """Renders a row of eco-themed KPI cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_sustainability_kpi(
                label=m.get("label", ""),
                value=m.get("value", ""),
                icon=m.get("icon", "🌱"),
                subtext=m.get("subtext", ""),
                accent=m.get("accent", "green"),
            )


# ── Recommendation Card ──────────────────────────────────────────────────────

PRIORITY_STYLES = {
    "high": {"bg": "rgba(244,63,94,0.1)", "border": "rgba(244,63,94,0.3)", "color": "#F43F5E", "label": "HIGH"},
    "medium": {"bg": "rgba(245,158,11,0.1)", "border": "rgba(245,158,11,0.3)", "color": "#F59E0B", "label": "MEDIUM"},
    "low": {"bg": "rgba(59,130,246,0.1)", "border": "rgba(59,130,246,0.3)", "color": "#3B82F6", "label": "LOW"},
}

TYPE_LABELS = {
    "combine": "🔗 COMBINE ORDER",
    "delay": "⏳ DELAY ORDER",
    "advance": "⚡ ADVANCE ORDER",
    "reduce": "📦 OPTIMIZE CAPACITY",
    "avoid_overstock": "🌱 ENVIRONMENTAL IMPACT",
}


def render_recommendation_card(rec: Dict):
    """Renders a single recommendation card with priority badge and action type."""
    priority = rec.get("priority", "medium")
    rec_type = rec.get("type", "combine")
    style = PRIORITY_STYLES.get(priority, PRIORITY_STYLES["medium"])
    type_label = TYPE_LABELS.get(rec_type, "📋 RECOMMENDATION")
    icon = rec.get("icon", "📋")

    products_html = ""
    products = rec.get("products", [])
    if products:
        product_badges = "".join(
            f'<div style="display:inline-block;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);'
            f'border-radius:6px;padding:0.15rem 0.5rem;font-size:0.68rem;color:#3B82F6;font-weight:600;'
            f'margin-right:0.3rem;margin-top:0.3rem;">{p}</div>'
            for p in products[:5]
        )
        products_html = f'<div style="margin-top:0.5rem;">{product_badges}</div>'

    html = (
        f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
        f'border-radius:14px;padding:1.2rem;margin-bottom:0.8rem;border-left:3px solid {style["color"]};">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;">'
        f'<div style="font-size:1.1rem;">{icon}</div>'
        f'<div style="font-size:0.68rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;">{type_label}</div>'
        f'</div>'
        f'<div style="background:{style["bg"]};border:1px solid {style["border"]};color:{style["color"]};'
        f'font-size:0.62rem;font-weight:800;padding:0.15rem 0.5rem;border-radius:6px;letter-spacing:0.05em;">'
        f'{style["label"]} PRIORITY</div>'
        f'</div>'
        f'<div style="font-size:0.92rem;font-weight:700;color:#F8FAFC;margin-bottom:0.4rem;">{rec.get("title", "")}</div>'
        f'<div style="font-size:0.8rem;color:#94A3B8;line-height:1.5;">{rec.get("message", "")}</div>'
        f'{products_html}'
        f'<div style="margin-top:0.6rem;font-size:0.72rem;color:#10B981;font-weight:600;">'
        f'💡 {rec.get("estimated_savings", "")}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation_list(recommendations: List[Dict]):
    """Renders a list of recommendation cards, sorted by priority."""
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_recs = sorted(recommendations, key=lambda r: priority_order.get(r.get("priority", "low"), 2))

    for rec in sorted_recs:
        render_recommendation_card(rec)
