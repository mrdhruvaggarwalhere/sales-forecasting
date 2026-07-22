"""
Model Training Module.
Trains Linear Regression, Random Forest, XGBoost on granular data,
and ARIMA, SARIMA, Prophet on aggregated time-series data.
"""

import os
import logging
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Statistical Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from src.models.model_utils import load_dataset, chronological_split, aggregate_to_daily, save_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Machine Learning Models (Trained on granular item-level data)
# ---------------------------------------------------------

def prepare_ml_features(df: pd.DataFrame, target_col: str = 'revenue'):
    """Separates features (X) and target (y), dropping non-predictive/leakage columns."""
    # Drop date, string identifiers (if any remain), and the target itself
    cols_to_drop = ['date', target_col]
    
    # Also drop rows with NaNs caused by lags
    df = df.dropna()
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[target_col]
    return X, y, df['date'] # Return dates for alignment later if needed


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Multiple Linear Regression model."""
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Random Forest Regressor (with limits to prevent memory crashes)."""
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains an XGBoost Regressor."""
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# Statistical Models (Trained on aggregated daily revenue)
# ---------------------------------------------------------

def train_arima(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains an ARIMA model on daily aggregated revenue."""
    logger.info("Training ARIMA model...")
    # (p, d, q) order. Simple order for demonstration.
    model = ARIMA(df_agg[target_col].values, order=(5, 1, 0))
    fitted_model = model.fit()
    return fitted_model


def train_sarima(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains a SARIMA model (includes weekly seasonality) on daily aggregated revenue."""
    logger.info("Training SARIMA model...")
    # (p, d, q) x (P, D, Q, s). s=7 for weekly seasonality.
    model = SARIMAX(df_agg[target_col].values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fitted_model = model.fit(disp=False)
    return fitted_model


def train_prophet(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains a Facebook Prophet model on daily aggregated revenue."""
    logger.info("Training Prophet model...")
    # Prophet requires columns to be named 'ds' and 'y'
    prophet_df = df_agg[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    return model


# ---------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------

def run_training_pipeline(input_path: str, models_dir: str):
    """Orchestrates the training of all 6 models and saves them to disk."""
    logger.info("Starting Complete Model Training Pipeline...")
    
    # 1. Load Data
    # For demonstration/speed, we sample the data if it's too large, but normally load all.
    df = load_dataset(input_path, nrows=500000)
    
    # 2. Chronological Split
    train_df, _ = chronological_split(df, train_ratio=0.7)
    
    # 3. Prepare data for ML Models
    X_train, y_train, _ = prepare_ml_features(train_df, target_col='revenue')
    
    # 4. Prepare data for Statistical Models
    train_agg_df = aggregate_to_daily(train_df, target_col='revenue')
    
    # 5. Train and Save ML Models
    linreg_model = train_linear_regression(X_train, y_train)
    save_model(linreg_model, os.path.join(models_dir, 'linear_regression.pkl'))
    
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))
    
    # 6. Train and Save Statistical Models
    arima_model = train_arima(train_agg_df)
    save_model(arima_model, os.path.join(models_dir, 'arima.pkl'))
    
    sarima_model = train_sarima(train_agg_df)
    save_model(sarima_model, os.path.join(models_dir, 'sarima.pkl'))
    
    prophet_model = train_prophet(train_agg_df)
    save_model(prophet_model, os.path.join(models_dir, 'prophet.pkl'))
    
    logger.info("All models trained and saved successfully.")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "outputs" / "models"
    
    run_training_pipeline(
        input_path=str(PROCESSED_DIR / "featured_sales.csv"),
        models_dir=str(MODELS_DIR)
    )
