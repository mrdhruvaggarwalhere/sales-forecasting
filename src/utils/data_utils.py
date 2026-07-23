"""
Shared Data Utilities for the Sales Revenue Forecasting System.
Single source of truth for memory-optimization and common data operations.
"""

import logging
import pandas as pd
import numpy as np

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
    initial_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"Downcasting data types. Initial memory usage: {initial_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_datetime64_any_dtype(col_type) \
                and not isinstance(col_type, pd.CategoricalDtype):
            try:
                c_min = df[col].min()
                c_max = df[col].max()

                if pd.isna(c_min) or pd.isna(c_max):
                    continue

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                elif str(col_type)[:5] == 'float':
                    # Skip float16 — too imprecise for financial data
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except (ValueError, TypeError):
                continue

        # Downcast object strings to category if cardinality is low
        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_total > 0 and num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    final_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"Downcasting complete. Final memory usage: {final_mem:.2f} MB "
                f"(saved {initial_mem - final_mem:.2f} MB)")
    return df
