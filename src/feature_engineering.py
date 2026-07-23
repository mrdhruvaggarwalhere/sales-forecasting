"""
Feature Engineering Module for the Sales Revenue Forecasting System.
Creates temporal, lag, rolling, trend, and categorical features while preventing data leakage.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.utils.data_utils import downcast_dtypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the dataframe by store, item, and date to prevent data leakage
    during lag and rolling calculations.
    """
    logger.info("Sorting data chronologically by store, item, and date...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['store_id', 'item_id', 'date']).reset_index(drop=True)
    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date-related features from the datetime column."""
    logger.info("Creating date features...")
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int32)
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(np.int8)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(np.int8)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(np.int8)

    return downcast_dtypes(df)


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """Creates lag features for a specific target grouped by item and store."""
    logger.info(f"Creating lag features {lags} for {target_col}...")

    # Group once and reuse
    grouped_target = df.groupby(['store_id', 'item_id'])[target_col]

    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = grouped_target.shift(lag)

    return downcast_dtypes(df)


def create_rolling_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Creates rolling mean and standard deviation features."""
    logger.info(f"Creating rolling features for {target_col}...")

    # Grouping object for reuse
    grouped = df.groupby(['store_id', 'item_id'])[target_col]

    df[f'{target_col}_rolling_mean_7'] = grouped.transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df[f'{target_col}_rolling_mean_14'] = grouped.transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df[f'{target_col}_rolling_mean_28'] = grouped.transform(lambda x: x.rolling(window=28, min_periods=1).mean())

    df[f'{target_col}_rolling_std_7'] = grouped.transform(lambda x: x.rolling(window=7, min_periods=1).std())
    df[f'{target_col}_rolling_std_28'] = grouped.transform(lambda x: x.rolling(window=28, min_periods=1).std())

    return downcast_dtypes(df)


def create_sales_trend_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Creates growth rate features based on the target column."""
    logger.info(f"Creating growth rate features for {target_col}...")

    # Group once and reuse
    grouped_target = df.groupby(['store_id', 'item_id'])[target_col]

    daily_lag = grouped_target.shift(1)
    weekly_lag = grouped_target.shift(7)
    monthly_lag = grouped_target.shift(28)

    # Using np.where to avoid division by zero
    df['daily_growth_rate'] = np.where(daily_lag == 0, 0, (df[target_col] - daily_lag) / daily_lag)
    df['weekly_growth_rate'] = np.where(weekly_lag == 0, 0, (df[target_col] - weekly_lag) / weekly_lag)
    df['monthly_growth_rate'] = np.where(monthly_lag == 0, 0, (df[target_col] - monthly_lag) / monthly_lag)

    # Replace inf/-inf with 0
    for col in ['daily_growth_rate', 'weekly_growth_rate', 'monthly_growth_rate']:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    return downcast_dtypes(df)


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on sell_price."""
    logger.info("Creating price features...")

    if 'sell_price' not in df.columns:
        logger.warning("'sell_price' column not found. Skipping price features.")
        return df

    grouped = df.groupby(['store_id', 'item_id'])['sell_price']

    # Rolling average price
    df['price_rolling_avg_7'] = grouped.transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    # Price change percentage vs lag 1
    price_lag_1 = grouped.shift(1)
    df['price_change_pct'] = np.where(price_lag_1 == 0, 0, (df['sell_price'] - price_lag_1) / price_lag_1)
    df['price_change_pct'] = df['price_change_pct'].replace([np.inf, -np.inf], 0)

    return downcast_dtypes(df)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label encodes categorical identifiers and event types."""
    logger.info("Encoding categorical features...")
    encoder = LabelEncoder()

    categorical_cols = ['store_id', 'item_id', 'state_id', 'dept_id', 'cat_id',
                        'event_name_1', 'event_type_1']

    for col in categorical_cols:
        if col in df.columns:
            # Handle potential NaNs in categorical columns
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = encoder.fit_transform(df[col])

    return downcast_dtypes(df)


def feature_engineering_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline.

    Args:
        input_path (str): Path to the processed sales CSV.
        output_path (str): Path to save the engineered CSV.

    Returns:
        pd.DataFrame: The feature-engineered dataframe.
    """
    logger.info("=" * 50)
    logger.info("Starting Feature Engineering Pipeline...")
    logger.info("=" * 50)

    try:
        # 1. Load Data
        logger.info(f"Loading preprocessed data from {input_path}...")
        df = pd.read_csv(input_path)

        # 2. Prevent Data Leakage (Crucial step)
        df = ensure_chronological_order(df)

        # 3. Revenue Feature check
        if 'revenue' not in df.columns:
            logger.info("Calculating Revenue feature...")
            if 'sales_units' in df.columns and 'sell_price' in df.columns:
                df['revenue'] = df['sales_units'] * df['sell_price']
            else:
                logger.error("Cannot calculate revenue: missing sales_units or sell_price.")
                raise ValueError("Missing columns for revenue calculation")

        # 4. Temporal Features
        df = create_date_features(df)

        # 5. Target Variable for Time Series (using revenue as target)
        target_col = 'revenue'

        # 6. Lag Features
        df = create_lag_features(df, target_col=target_col, lags=[1, 7, 14, 28])

        # 7. Rolling Features
        df = create_rolling_features(df, target_col=target_col)

        # 8. Sales Trend Features
        df = create_sales_trend_features(df, target_col=target_col)

        # 9. Price Features
        df = create_price_features(df)

        # 10. Categorical Encoding
        df = encode_categorical_features(df)

        # 11. Fill NaNs from lag/rolling features with 0
        logger.info("Filling NaN values generated by lag/rolling features with 0...")
        df = df.fillna(0)

        # 12. Save Data
        logger.info(f"Saving engineered data to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Feature Engineering Pipeline completed. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        raise


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    feature_engineering_pipeline(
        input_path=str(PROCESSED_DIR / "processed_sales.csv"),
        output_path=str(PROCESSED_DIR / "featured_sales.csv")
    )
