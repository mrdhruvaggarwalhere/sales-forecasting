"""
Data Preprocessing Package for the Sales Revenue Forecasting System.
Handles data ingestion, melting, merging, cleaning, downcasting, and validation for the M5 dataset.
"""

import os
import gc
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.data_utils import downcast_dtypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_melt_sales(file_path: str) -> pd.DataFrame:
    """
    Loads the wide-format sales data and melts it into a long format.

    Args:
        file_path (str): Path to sales_train_evaluation.csv

    Returns:
        pd.DataFrame: Melted sales dataframe.
    """
    try:
        logger.info(f"Loading and melting sales data from {file_path}...")
        df = pd.read_csv(file_path)

        # Identify identifier columns (non-day columns)
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

        # Melt dataframe
        df_melted = pd.melt(
            df,
            id_vars=id_vars,
            var_name='d',
            value_name='sales_units'
        )

        df_melted = downcast_dtypes(df_melted)
        logger.info(f"Successfully melted sales data. Shape: {df_melted.shape}")
        return df_melted

    except FileNotFoundError:
        logger.error(f"Sales data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error melting sales data: {e}")
        raise


def load_calendar(file_path: str) -> pd.DataFrame:
    """
    Loads and processes the calendar dataset.

    Args:
        file_path (str): Path to calendar.csv

    Returns:
        pd.DataFrame: Processed calendar dataframe.
    """
    try:
        logger.info(f"Loading calendar data from {file_path}...")
        cal = pd.read_csv(file_path)

        # Convert date to datetime format
        cal['date'] = pd.to_datetime(cal['date'])

        # Fill missing event names with 'No_Event'
        for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
            if col in cal.columns:
                cal[col] = cal[col].fillna('No_Event')

        cal = downcast_dtypes(cal)
        logger.info(f"Successfully loaded calendar data. Shape: {cal.shape}")
        return cal

    except Exception as e:
        logger.error(f"Error loading calendar data: {e}")
        raise


def load_prices(file_path: str) -> pd.DataFrame:
    """
    Loads and processes the sell prices dataset.

    Args:
        file_path (str): Path to sell_prices.csv

    Returns:
        pd.DataFrame: Processed prices dataframe.
    """
    try:
        logger.info(f"Loading prices data from {file_path}...")
        prices = pd.read_csv(file_path)
        prices = downcast_dtypes(prices)
        logger.info(f"Successfully loaded prices data. Shape: {prices.shape}")
        return prices

    except Exception as e:
        logger.error(f"Error loading prices data: {e}")
        raise


def merge_datasets(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the sales, calendar, and prices datasets into a single dataframe.

    Args:
        sales (pd.DataFrame): Melted sales dataframe.
        calendar (pd.DataFrame): Processed calendar dataframe.
        prices (pd.DataFrame): Processed prices dataframe.

    Returns:
        pd.DataFrame: Fully merged master dataframe.
    """
    try:
        logger.info("Merging sales with calendar...")
        # Merge on 'd' (day id)
        df = pd.merge(sales, calendar, on='d', how='left')

        # Free up memory
        del sales
        del calendar
        gc.collect()

        logger.info("Merging with sell prices...")
        # Merge on store, item, and week
        df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

        # Free up memory
        del prices
        gc.collect()

        logger.info(f"Datasets merged successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during dataset merging: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values, calculates target variable (revenue), and drops duplicates/redundant columns.

    Args:
        df (pd.DataFrame): The merged dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe ready for feature engineering.
    """
    try:
        logger.info("Starting data cleaning process...")

        # 1. Drop rows with NaN sell_price (item not yet launched in the store)
        initial_rows = len(df)
        df = df.dropna(subset=['sell_price']).copy()
        logger.info(f"Dropped {initial_rows - len(df)} rows where 'sell_price' was NaN.")

        # 2. Calculate Revenue (Target Variable)
        df['revenue'] = df['sales_units'] * df['sell_price']

        # 3. Handle duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Dropped {duplicates} duplicate rows.")

        # 4. Drop unnecessary columns
        columns_to_drop = ['d', 'wm_yr_wk', 'id']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Re-downcast to optimize newly created columns (like revenue)
        df = downcast_dtypes(df)

        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validates the final processed dataset to ensure integrity before saving.

    Args:
        df (pd.DataFrame): The final processed dataframe.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger.info("Validating processed dataset...")

    is_valid = True

    # Check for target variable
    if 'revenue' not in df.columns:
        logger.error("Validation Failed: 'revenue' column is missing.")
        is_valid = False

    # Check for nulls in crucial columns
    critical_cols = ['date', 'item_id', 'store_id', 'sales_units', 'sell_price']
    for col in critical_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            logger.warning(f"Validation Warning: {null_count} NaN values in '{col}'. Filling with defaults.")
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('Unknown')

    if is_valid:
        logger.info("Dataset validation passed.")

    return is_valid


def preprocess_pipeline(
    sales_path: str,
    calendar_path: str,
    prices_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Orchestrates the entire preprocessing pipeline.

    Args:
        sales_path (str): Path to raw sales CSV.
        calendar_path (str): Path to raw calendar CSV.
        prices_path (str): Path to raw prices CSV.
        output_path (str): Path to save the processed CSV.

    Returns:
        pd.DataFrame: The cleaned and validated dataframe.
    """
    logger.info("=" * 50)
    logger.info("Starting M5 Data Preprocessing Pipeline...")
    logger.info("=" * 50)

    try:
        # 1. Load Data
        sales = load_and_melt_sales(sales_path)
        calendar = load_calendar(calendar_path)
        prices = load_prices(prices_path)

        # 2. Merge Data
        merged_df = merge_datasets(sales, calendar, prices)

        # 3. Clean Data
        cleaned_df = clean_data(merged_df)

        # 4. Validate Data
        if validate_dataset(cleaned_df):
            # 5. Save Data
            logger.info(f"Saving processed data to {output_path}...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cleaned_df.to_csv(output_path, index=False)
            logger.info("[OK] Preprocessing Pipeline completed successfully.")
            return cleaned_df
        else:
            logger.error("Pipeline aborted due to validation failure.")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise
