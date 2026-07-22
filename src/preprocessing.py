"""
Data Preprocessing Module for the Sales Revenue Forecasting System.
Handles data ingestion, melting, merging, cleaning, downcasting, and validation for the M5 dataset.
"""

import os
import gc
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric data types to save memory.
    Crucial for the M5 dataset as melting creates over 50 million rows.
    
    Args:
        df (pd.DataFrame): The dataframe to compress.
        
    Returns:
        pd.DataFrame: The memory-optimized dataframe.
    """
    logger.info(f"Downcasting data types. Initial memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
        # Downcast object strings to category if cardinality is low
        elif col_type == object:
            if col not in ['date', 'd']:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
                    
    logger.info(f"Downcasting complete. Final memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    return df


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
        logger.info("Successfully loaded and melted sales data.")
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
        cal['event_name_1'] = cal['event_name_1'].fillna('No_Event')
        cal['event_type_1'] = cal['event_type_1'].fillna('No_Event')
        cal['event_name_2'] = cal['event_name_2'].fillna('No_Event')
        cal['event_type_2'] = cal['event_type_2'].fillna('No_Event')
        
        cal = downcast_dtypes(cal)
        logger.info("Successfully loaded calendar data.")
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
        logger.info("Successfully loaded prices data.")
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
        
        logger.info("Datasets merged successfully.")
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
        df.dropna(subset=['sell_price'], inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows where 'sell_price' was NaN.")
        
        # 2. Calculate Revenue (Target Variable)
        df['revenue'] = df['sales_units'] * df['sell_price']
        
        # 3. Handle duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df.drop_duplicates(inplace=True)
            logger.info(f"Dropped {duplicates} duplicate rows.")
            
        # 4. Drop unnecessary columns
        columns_to_drop = ['d', 'wm_yr_wk', 'id']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        # Re-downcast to optimize newly created columns (like revenue)
        df = downcast_dtypes(df)
        
        logger.info("Data cleaning completed successfully.")
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
        if df[col].isnull().any():
            logger.error(f"Validation Failed: NaN values found in critical column '{col}'.")
            is_valid = False
            
    if is_valid:
        logger.info("Dataset validation passed.")
        
    return is_valid


def preprocess_pipeline(
    sales_path: str,
    calendar_path: str,
    prices_path: str,
    output_path: str
) -> None:
    """
    Orchestrates the entire preprocessing pipeline.
    
    Args:
        sales_path (str): Path to raw sales CSV.
        calendar_path (str): Path to raw calendar CSV.
        prices_path (str): Path to raw prices CSV.
        output_path (str): Path to save the processed CSV.
    """
    logger.info("Starting M5 Data Preprocessing Pipeline...")
    
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
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as CSV (or Parquet/Feather for faster I/O in reality)
            cleaned_df.to_csv(output_path, index=False)
            logger.info("Data saved successfully. Preprocessing Pipeline completed.")
        else:
            logger.error("Pipeline aborted due to validation failure.")
            
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage (paths will be configured in config.py)
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    preprocess_pipeline(
        sales_path=str(RAW_DIR / "sales_train_evaluation.csv"),
        calendar_path=str(RAW_DIR / "calendar.csv"),
        prices_path=str(RAW_DIR / "sell_prices.csv"),
        output_path=str(PROCESSED_DIR / "processed_sales.csv")
    )
