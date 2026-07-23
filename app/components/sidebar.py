"""
Sidebar Component — Navigation and filters for the dashboard.
Adapted for the aggregated daily sales dataset (no store/item/category columns).
"""

import streamlit as st
import pandas as pd


def render_sidebar(df: pd.DataFrame = None) -> dict:
    """Renders the sidebar with navigation and year filter."""
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────
        st.markdown(
'<div style="padding:0.5rem 0 1.2rem 0;display:flex;align-items:center;gap:0.75rem;">'
'<div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#3B82F6 0%,#6366F1 100%);display:flex;align-items:center;justify-content:center;font-size:1.25rem;box-shadow:0 0 15px rgba(59,130,246,0.4);">📈</div>'
'<div><div style="font-size:1.1rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;line-height:1.2;">SalesPulse</div>'
'<div style="font-size:0.72rem;font-weight:600;color:#3B82F6;letter-spacing:0.05em;text-transform:uppercase;">Enterprise Analytics</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;color:#64748B;'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Navigation</div>',
            unsafe_allow_html=True,
        )

        pages = {
            "🏠  Overview": "Home",
            "📊  Sales Analysis": "Sales Analysis",
            "🔮  Revenue Forecast": "Forecast",
            "📦  Product Breakdown": "Product Analysis",
            "💰  Revenue Deep-Dive": "Revenue Analysis",
            "⚡  Model Comparison": "Model Comparison",
            "📄  Executive Reports": "Reports",
        }

        selected_page = st.radio(
            "Navigate",
            options=list(pages.keys()),
            label_visibility="collapsed",
        )
        active_page = pages[selected_page]

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;color:#64748B;'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">Global Filters</div>',
            unsafe_allow_html=True,
        )

        filters = {}

        if df is not None and not df.empty:
            # Year filter
            if "year" in df.columns:
                years = ["All Years"] + sorted(df["year"].unique().tolist())
                selected_year = st.selectbox("Year", years)
                filters["year"] = None if selected_year == "All Years" else selected_year

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── System Status ─────────────────────────────────────
        st.markdown(
'<div style="padding:0.5rem;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);text-align:center;">'
'<div style="font-size:0.72rem;font-weight:600;color:#94A3B8;">System Status</div>'
'<div style="display:flex;align-items:center;justify-content:center;gap:0.4rem;font-size:0.7rem;color:#10B981;margin-top:0.2rem;">'
'<span style="width:6px;height:6px;border-radius:50%;background:#10B981;display:inline-block;"></span>'
'Pipeline Ready</div></div>',
            unsafe_allow_html=True,
        )

    return {"page": active_page, "filters": filters}
