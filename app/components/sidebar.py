"""
Sidebar Component for the Streamlit Dashboard.
Handles navigation, filters (date, state, store, category, product), and branding.
"""

import streamlit as st
import pandas as pd


def render_sidebar(df: pd.DataFrame = None) -> dict:
    """
    Renders the sidebar with navigation and interactive filters.

    Args:
        df: The loaded dataframe used to populate filter options dynamically.

    Returns:
        dict: A dictionary containing the selected page and all active filter values.
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.5rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #F0F2F6; letter-spacing: -0.02em;">
                    📊 Revenue Forecast
                </div>
                <div style="font-size: 0.72rem; color: #6B7280; margin-top: 0.15rem;">
                    Sales Intelligence Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Navigation ───────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem; font-weight:600; color:#6B7280; '
            'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Navigation</div>',
            unsafe_allow_html=True,
        )

        pages = {
            "🏠  Home": "Home",
            "📈  Sales Analysis": "Sales Analysis",
            "🔮  Forecast": "Forecast",
            "📦  Product Analysis": "Product Analysis",
            "💰  Revenue Analysis": "Revenue Analysis",
            "⚙️  Model Comparison": "Model Comparison",
            "📄  Reports": "Reports",
        }

        selected_page = st.radio(
            "Navigate",
            options=list(pages.keys()),
            label_visibility="collapsed",
        )
        active_page = pages[selected_page]

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem; font-weight:600; color:#6B7280; '
            'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Filters</div>',
            unsafe_allow_html=True,
        )

        filters = {}

        if df is not None and not df.empty:
            # Date Range
            if "date" in df.columns:
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                filters["date_range"] = date_range

            # State
            if "state_id" in df.columns:
                states = ["All"] + sorted(df["state_id"].unique().tolist())
                filters["state"] = st.selectbox("State", states)

            # Store
            if "store_id" in df.columns:
                stores = ["All"] + sorted(df["store_id"].unique().tolist())
                filters["store"] = st.selectbox("Store", stores)

            # Category
            if "cat_id" in df.columns:
                cats = ["All"] + sorted(df["cat_id"].unique().tolist())
                filters["category"] = st.selectbox("Category", cats)

            # Department
            if "dept_id" in df.columns:
                depts = ["All"] + sorted(df["dept_id"].unique().tolist())
                filters["department"] = st.selectbox("Department", depts)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.68rem; color:#4B5563; text-align:center; padding:0.5rem 0;">'
            'Built with Streamlit &bull; v1.0</div>',
            unsafe_allow_html=True,
        )

    return {"page": active_page, "filters": filters}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Applies the sidebar filter selections to the dataframe.

    Args:
        df: The original dataframe.
        filters: Dictionary of filter values from render_sidebar().

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    filtered = df.copy()

    # Date range
    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    # Categorical filters
    for col_key, col_name in [("state", "state_id"), ("store", "store_id"),
                               ("category", "cat_id"), ("department", "dept_id")]:
        val = filters.get(col_key)
        if val and val != "All" and col_name in filtered.columns:
            filtered = filtered[filtered[col_name] == val]

    return filtered
