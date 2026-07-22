"""
Revenue Analysis Page.
Revenue distribution, growth trends, and profitability insights.
"""

import streamlit as st
import pandas as pd
from components.charts import line_chart, histogram_chart, bar_chart
from components.kpi_cards import render_kpi_row


def render(df: pd.DataFrame):
    """Renders the Revenue Analysis page."""
    st.markdown('<div class="page-title">Revenue Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep-dive into revenue distribution, growth, and profitability</div>', unsafe_allow_html=True)

    if df.empty or "revenue" not in df.columns:
        st.warning("No revenue data available for the selected filters.")
        return

    # ── Summary KPIs ───────────────────────────────────────
    total_rev = df["revenue"].sum()
    avg_rev = df["revenue"].mean()
    max_rev = df["revenue"].max()

    render_kpi_row([
        {"label": "Total Revenue", "value": f"${total_rev:,.0f}", "accent": "green"},
        {"label": "Avg Revenue / Record", "value": f"${avg_rev:,.2f}", "accent": "blue"},
        {"label": "Peak Revenue (Single)", "value": f"${max_rev:,.2f}", "accent": "amber"},
    ])

    st.markdown("", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Growth", "Profitability"])

    with tab1:
        st.markdown('<div class="section-header">Revenue Distribution</div>', unsafe_allow_html=True)
        positive_rev = df[df["revenue"] > 0]
        fig = histogram_chart(positive_rev, "revenue", "Revenue Distribution (Log Scale)", nbins=50)
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Monthly Revenue Growth</div>', unsafe_allow_html=True)
        if "year" in df.columns and "month" in df.columns:
            monthly = df.groupby(["year", "month"])["revenue"].sum().reset_index()
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            monthly["growth_pct"] = monthly["revenue"].pct_change() * 100
            fig = bar_chart(monthly, "period", "growth_pct", "Month-over-Month Revenue Growth (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Revenue by Price Tier</div>', unsafe_allow_html=True)
        if "sell_price" in df.columns:
            df_copy = df.copy()
            df_copy["price_tier"] = pd.cut(
                df_copy["sell_price"], bins=[0, 2, 5, 10, 20, 100],
                labels=["$0-2", "$2-5", "$5-10", "$10-20", "$20+"]
            )
            tier = df_copy.groupby("price_tier", observed=True)["revenue"].sum().reset_index()
            fig = bar_chart(tier, "price_tier", "revenue", "Revenue by Price Tier")
            st.plotly_chart(fig, use_container_width=True)
