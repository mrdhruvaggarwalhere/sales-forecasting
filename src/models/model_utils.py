"""
Model Utilities Module.
Contains reusable functions for loading data, splitting, aggregating, and saving/loading models.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Loads the engineered dataset and ensures the date column is properly formatted.
    
    Args:
        filepath (str): Path to the dataset.
        nrows (int): Optional number of rows to load (useful for fast testing).
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logger.info(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, nrows=nrows)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test chronologically to prevent data leakage.
    Never shuffles the data.
    
    Args:
        df (pd.DataFrame): The full dataframe.
        train_ratio (float): Percentage of data to use for training.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, test_df
    """
    logger.info(f"Performing chronological split (Train: {train_ratio*100}%, Test: {(1-train_ratio)*100}%)...")
    
    # We split based on unique dates to ensure the horizon is cut off cleanly
    unique_dates = df['date'].sort_values().unique()
    split_index = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_index]
    
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    logger.info(f"Training set ends at {train_df['date'].max().date()}, Testing set starts at {test_df['date'].min().date()}")
    return train_df, test_df


def aggregate_to_daily(df: pd.DataFrame, target_col: str = 'revenue') -> pd.DataFrame:
    """
    Aggregates the granular item-level data to total daily revenue.
    Required for univariate statistical models (ARIMA, Prophet).
    
    Args:
        df (pd.DataFrame): Granular dataframe.
        target_col (str): Column to aggregate.
        
    Returns:
        pd.DataFrame: Aggregated dataframe indexed by date.
    """
    logger.info("Aggregating data to daily total revenue...")
    agg_df = df.groupby('date')[target_col].sum().reset_index()
    agg_df.sort_values('date', inplace=True)
    return agg_df


def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model to disk using joblib.
    
    Args:
        model (Any): The trained ML/Stat model.
        filepath (str): Destination path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Loads a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model.
        
    Returns:
        Any: The loaded model.
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
