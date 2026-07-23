"""
Model Training Module.
Trains Linear Regression, Random Forest, XGBoost on granular data,
and optionally ARIMA, SARIMA, Prophet on aggregated time-series data.
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lazy imports for optional heavy dependencies
# ---------------------------------------------------------

def _import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        logger.warning("XGBoost not installed. Skipping XGBoost training.")
        return None


def _import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except ImportError:
        logger.warning("Prophet not installed. Skipping Prophet training.")
        return None


def _import_statsmodels():
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        return ARIMA, SARIMAX
    except ImportError:
        logger.warning("Statsmodels not installed. Skipping ARIMA/SARIMA training.")
        return None, None


# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------

def prepare_ml_features(df: pd.DataFrame, target_col: str = 'revenue'):
    """Separates features (X) and target (y), keeping strictly numeric columns."""
    df = df.dropna()
    dates = df['date'] if 'date' in df.columns else pd.Series(dtype='datetime64[ns]')

    cols_to_drop = ['date', target_col]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Ensure all features in X are numeric
    X = X.select_dtypes(include=[np.number])

    y = df[target_col]
    return X, y, dates


# ---------------------------------------------------------
# Machine Learning Models (Trained on granular item-level data)
# ---------------------------------------------------------

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Multiple Linear Regression model."""
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("[OK] Linear Regression trained.")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Random Forest Regressor (with limits to prevent memory crashes)."""
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    logger.info("[OK] Random Forest trained.")
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains an XGBoost Regressor."""
    xgb = _import_xgboost()
    if xgb is None:
        return None
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("[OK] XGBoost trained.")
    return model


# ---------------------------------------------------------
# Statistical Models (Trained on aggregated daily revenue)
# ---------------------------------------------------------

def train_arima(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains an ARIMA model on daily aggregated revenue."""
    ARIMA, _ = _import_statsmodels()
    if ARIMA is None:
        return None
    logger.info("Training ARIMA model...")
    try:
        model = ARIMA(df_agg[target_col].values, order=(5, 1, 0))
        fitted_model = model.fit()
        logger.info("[OK] ARIMA trained.")
        return fitted_model
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")
        return None


def train_sarima(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains a SARIMA model (includes weekly seasonality) on daily aggregated revenue."""
    _, SARIMAX = _import_statsmodels()
    if SARIMAX is None:
        return None
    logger.info("Training SARIMA model...")
    try:
        model = SARIMAX(df_agg[target_col].values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        fitted_model = model.fit(disp=False)
        logger.info("[OK] SARIMA trained.")
        return fitted_model
    except Exception as e:
        logger.warning(f"SARIMA training failed: {e}")
        return None


def train_prophet(df_agg: pd.DataFrame, target_col: str = 'revenue'):
    """Trains a Facebook Prophet model on daily aggregated revenue."""
    Prophet = _import_prophet()
    if Prophet is None:
        return None
    logger.info("Training Prophet model...")
    try:
        prophet_df = df_agg[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
        model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_df)
        logger.info("[OK] Prophet trained.")
        return model
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        return None


# ---------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------

def run_training_pipeline(input_path: str, models_dir: str):
    """Orchestrates the training of all models and saves them to disk."""
    from src.models.model_utils import load_dataset, chronological_split, aggregate_to_daily, save_model

    logger.info("=" * 50)
    logger.info("Starting Model Training Pipeline...")
    logger.info("=" * 50)

    # 1. Load Data
    df = load_dataset(input_path, nrows=500000)

    if df.empty:
        logger.error("No data loaded. Aborting training.")
        return

    # 2. Chronological Split
    train_df, _ = chronological_split(df, train_ratio=0.7)

    # 3. Prepare data for ML Models
    X_train, y_train, _ = prepare_ml_features(train_df, target_col='revenue')
    logger.info(f"ML training set: X={X_train.shape}, y={y_train.shape}")

    # 4. Prepare data for Statistical Models
    train_agg_df = aggregate_to_daily(train_df, target_col='revenue')

    os.makedirs(models_dir, exist_ok=True)
    trained_count = 0

    # 5. Train and Save ML Models
    linreg_model = train_linear_regression(X_train, y_train)
    save_model(linreg_model, os.path.join(models_dir, 'linear_regression.pkl'))
    trained_count += 1

    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    trained_count += 1

    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        save_model(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))
        trained_count += 1

    # 6. Train and Save Statistical Models
    arima_model = train_arima(train_agg_df)
    if arima_model is not None:
        save_model(arima_model, os.path.join(models_dir, 'arima.pkl'))
        trained_count += 1

    sarima_model = train_sarima(train_agg_df)
    if sarima_model is not None:
        save_model(sarima_model, os.path.join(models_dir, 'sarima.pkl'))
        trained_count += 1

    prophet_model = train_prophet(train_agg_df)
    if prophet_model is not None:
        save_model(prophet_model, os.path.join(models_dir, 'prophet.pkl'))
        trained_count += 1

    logger.info(f"[OK] Training complete. {trained_count} models trained and saved to {models_dir}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "outputs" / "models"

    run_training_pipeline(
        input_path=str(PROCESSED_DIR / "featured_sales.csv"),
        models_dir=str(MODELS_DIR)
    )
