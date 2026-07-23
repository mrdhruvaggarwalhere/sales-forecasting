"""
Demo Data Generator for the Sales Revenue Forecasting System.
Generates realistic synthetic M5-style sales data so the project
can run out-of-the-box in seconds without downloading the full Kaggle dataset.

Usage:
    python scripts/generate_demo_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

np.random.seed(42)

# ── Configuration (optimized for fast demo execution ~30k rows) ─────────────────
STORES = ["CA_1", "CA_2", "TX_1", "WI_1"]
STATES = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX", "WI_1": "WI"}

CATEGORIES = {
    "FOODS": ["FOODS_1", "FOODS_2"],
    "HOUSEHOLD": ["HOUSEHOLD_1"],
    "HOBBIES": ["HOBBIES_1"],
}

ITEMS_PER_DEPT = 3  # 4 depts * 3 items = 12 items * 4 stores = 48 time series

EVENT_NAMES = [
    "SuperBowl", "ValentinesDay", "PresidentsDay", "StPatricksDay",
    "Easter", "MothersDay", "MemorialDay", "FathersDay",
    "IndependenceDay", "LaborDay", "Halloween", "Thanksgiving", "Christmas"
]

EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]

START_DATE = "2015-01-01"
END_DATE = "2016-05-22"

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def generate_items():
    """Generates a list of item dicts with category/department mappings."""
    items = []
    for cat_id, depts in CATEGORIES.items():
        for dept_id in depts:
            for i in range(1, ITEMS_PER_DEPT + 1):
                items.append({
                    "item_id": f"{dept_id}_{i:03d}",
                    "dept_id": dept_id,
                    "cat_id": cat_id,
                })
    return items


def generate_calendar():
    """Generates a calendar DataFrame covering the date range."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n = len(dates)

    cal = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": [d.year * 100 + d.week for d in dates],
        "weekday": dates.strftime("%A"),
        "wday": dates.dayofweek + 1,
        "month": dates.month,
        "year": dates.year,
        "event_name_1": [None] * n,
        "event_type_1": [None] * n,
        "event_name_2": [None] * n,
        "event_type_2": [None] * n,
    })

    # Assign 'd' column (d_1, d_2, ...)
    cal["d"] = [f"d_{i+1}" for i in range(n)]

    # Sprinkle random events (~5% of days)
    event_mask = np.random.random(n) < 0.05
    event_indices = np.where(event_mask)[0]
    for idx in event_indices:
        cal.loc[idx, "event_name_1"] = str(np.random.choice(EVENT_NAMES))
        cal.loc[idx, "event_type_1"] = str(np.random.choice(EVENT_TYPES))

    # Assign SNAP flags per state
    for state in ["CA", "TX", "WI"]:
        cal[f"snap_{state}"] = (cal["date"].dt.day <= 10).astype(int)

    return cal


def generate_sell_prices(items, stores, calendar):
    """Generates sell prices by store/item/week."""
    wm_yr_wks = calendar["wm_yr_wk"].unique()
    rows = []

    for store_id in stores:
        for item in items:
            base_price = np.round(np.random.uniform(1.50, 18.00), 2)
            for wk in wm_yr_wks:
                noise = np.random.uniform(-0.03, 0.03)
                price = max(0.50, round(base_price * (1 + noise), 2))
                rows.append({
                    "store_id": store_id,
                    "item_id": item["item_id"],
                    "wm_yr_wk": wk,
                    "sell_price": price,
                })

    return pd.DataFrame(rows)


def generate_sales(items, stores, calendar):
    """Generates wide-format sales data (one column per day)."""
    d_cols = calendar["d"].values.tolist()
    n_days = len(d_cols)

    dow_factor = np.array([1.0, 0.95, 0.9, 0.95, 1.1, 1.3, 1.2])
    day_indices = calendar["date"].dt.dayofweek.values

    rows = []
    for store_id in stores:
        state = STATES[store_id]
        for item in items:
            if item["cat_id"] == "FOODS":
                base = np.random.uniform(3, 15)
            elif item["cat_id"] == "HOUSEHOLD":
                base = np.random.uniform(1, 6)
            else:
                base = np.random.uniform(0.5, 4)

            seasonality = dow_factor[day_indices]
            noise_poisson = np.random.poisson(lam=max(1.0, base * 0.4), size=n_days) * 0.4
            noise_normal = np.random.normal(0, base * 0.2, size=n_days)

            sales = np.maximum(0, np.round(base * seasonality + noise_poisson + noise_normal)).astype(int)

            row = {
                "id": f"{item['item_id']}_{store_id}_evaluation",
                "item_id": item["item_id"],
                "dept_id": item["dept_id"],
                "cat_id": item["cat_id"],
                "store_id": store_id,
                "state_id": state,
            }
            for i, d in enumerate(d_cols):
                row[d] = sales[i]

            rows.append(row)

    return pd.DataFrame(rows)


def generate_all():
    """Master function — generates and saves all three raw datasets."""
    logger_header = "=" * 55
    print(logger_header)
    print("  Sales Revenue Forecasting - Fast Demo Data Generator")
    print(logger_header)

    os.makedirs(RAW_DIR, exist_ok=True)

    items = generate_items()
    print(f"[+] Items catalog created: {len(items)} products across {len(CATEGORIES)} categories")

    calendar = generate_calendar()
    cal_path = RAW_DIR / "calendar.csv"
    calendar.to_csv(cal_path, index=False)
    print(f"[+] Calendar generated -> {cal_path} ({len(calendar)} days)")

    prices = generate_sell_prices(items, STORES, calendar)
    prices_path = RAW_DIR / "sell_prices.csv"
    prices.to_csv(prices_path, index=False)
    print(f"[+] Sell prices generated -> {prices_path} ({len(prices):,} rows)")

    sales = generate_sales(items, STORES, calendar)
    sales_path = RAW_DIR / "sales_train_evaluation.csv"
    sales.to_csv(sales_path, index=False)
    print(f"[+] Sales dataset generated -> {sales_path} ({len(sales):,} series x {len(calendar)} days)")

    print("\n[OK] Fast demo dataset created successfully!\n")
    return True


if __name__ == "__main__":
    generate_all()
