"""
Unified Training & Evaluation Script for the Sales Revenue Forecasting System.
Uses pre-computed data from data/dataset/ — 70% train / 30% test chronological split.

Usage:
    python -m src.models.train_and_evaluate
    python main.py --train
"""

import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ─── Optional imports ─────────────────────────────────────────────────────────

def _import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")
        return None


def _import_statsmodels():
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        return ARIMA, SARIMAX
    except ImportError:
        logger.warning("statsmodels not installed — skipping ARIMA/SARIMA.")
        return None, None


# ─── Metrics ──────────────────────────────────────────────────────────────────

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates MAE, RMSE, MAPE, R² Score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    else:
        mape = 0.0

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2), "R2 Score": round(r2, 4)}


# ─── Data Loading & Splitting ────────────────────────────────────────────────

def load_featured_data() -> pd.DataFrame:
    """Loads featured_daily_sales.csv from data/dataset/."""
    path = DATASET_DIR / "featured_daily_sales.csv"
    logger.info(f"Loading featured data from {path}...")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """Splits data 70/30 by unique dates — no shuffling."""
    unique_dates = df["date"].sort_values().unique()
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    logger.info(f"Train: {len(train)} rows ({train['date'].min().date()} → {train['date'].max().date()})")
    logger.info(f"Test:  {len(test)} rows ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, test


def prepare_features(df: pd.DataFrame, target: str = "sales"):
    """Extracts numeric features (X) and target (y) from the dataframe."""
    drop_cols = ["date", "d", "weekday", target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target].values
    return X, y


# ─── Training ────────────────────────────────────────────────────────────────

def train_ml_models(X_train, y_train):
    """Trains LinearRegression, RandomForest, and optionally XGBoost."""
    models = {}

    # Linear Regression
    logger.info("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr
    logger.info("  ✓ Linear Regression trained.")

    # Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    logger.info("  ✓ Random Forest trained.")

    # XGBoost
    xgb = _import_xgboost()
    if xgb is not None:
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
        logger.info("  ✓ XGBoost trained.")

    return models


def train_statistical_models(train_series: np.ndarray):
    """Trains ARIMA and SARIMA on the daily sales time series."""
    models = {}
    ARIMA, SARIMAX = _import_statsmodels()

    if ARIMA is not None:
        logger.info("Training ARIMA(5,1,0)...")
        try:
            model = ARIMA(train_series, order=(5, 1, 0))
            fitted = model.fit()
            models["ARIMA"] = fitted
            logger.info("  ✓ ARIMA trained.")
        except Exception as e:
            logger.warning(f"  ✗ ARIMA failed: {e}")

    if SARIMAX is not None:
        logger.info("Training SARIMA(1,1,1)(1,1,1,7)...")
        try:
            model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted = model.fit(disp=False)
            models["SARIMA"] = fitted
            logger.info("  ✓ SARIMA trained.")
        except Exception as e:
            logger.warning(f"  ✗ SARIMA failed: {e}")

    return models


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_models(ml_models, stat_models, X_test, y_test, test_dates, test_steps):
    """Evaluates all models and returns metrics + predictions."""
    results = []
    predictions = {}

    # ML Models
    for name, model in ml_models.items():
        logger.info(f"Evaluating {name}...")
        try:
            preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds)
            metrics["Model"] = name
            results.append(metrics)
            predictions[name] = preds.tolist()
            logger.info(f"  ✓ {name}: RMSE={metrics['RMSE']}, R²={metrics['R2 Score']}")
        except Exception as e:
            logger.warning(f"  ✗ {name} evaluation failed: {e}")

    # Statistical Models
    for name, model in stat_models.items():
        logger.info(f"Evaluating {name}...")
        try:
            preds = model.forecast(steps=test_steps)
            min_len = min(len(y_test), len(preds))
            preds = preds[:min_len]
            y_true_slice = y_test[:min_len]
            metrics = calculate_metrics(y_true_slice, preds)
            metrics["Model"] = name
            results.append(metrics)
            predictions[name] = preds.tolist()
            logger.info(f"  ✓ {name}: RMSE={metrics['RMSE']}, R²={metrics['R2 Score']}")
        except Exception as e:
            logger.warning(f"  ✗ {name} evaluation failed: {e}")

    return results, predictions


# ─── Save Outputs ────────────────────────────────────────────────────────────

def save_outputs(metrics_list, predictions, test_dates, y_test, best_model_name):
    """Saves evaluation_metrics.csv, forecast.csv, and plot_data.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. evaluation_metrics.csv
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[["Model", "MAE", "RMSE", "MAPE (%)", "R2 Score"]]
    metrics_path = MODELS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION COMPARISON")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("-" * 60)
    print(f"  BEST MODEL: {best_model_name}")
    print("-" * 60 + "\n")

    # 2. forecast.csv (best model predictions for dashboard)
    if best_model_name in predictions:
        best_preds = np.array(predictions[best_model_name])
        n = min(len(test_dates), len(best_preds))
        forecast_df = pd.DataFrame({
            "date": test_dates[:n],
            "actual": y_test[:n],
            "forecast": best_preds[:n],
            "lower_ci": best_preds[:n] * 0.90,
            "upper_ci": best_preds[:n] * 1.10,
        })
        forecast_path = PREDICTIONS_DIR / "forecast.csv"
        forecast_df.to_csv(forecast_path, index=False)
        logger.info(f"✓ Forecast CSV saved to {forecast_path} ({n} rows)")

    # 3. plot_data.json (all model predictions for the forecast page)
    plot_data = {
        "dates": [str(d)[:10] for d in test_dates],
        "actuals": y_test.tolist(),
        "predictions": {}
    }
    for model_name, preds in predictions.items():
        min_len = min(len(test_dates), len(preds))
        plot_data["predictions"][model_name] = [round(p, 2) for p in preds[:min_len]]

    plot_path = PREDICTIONS_DIR / "plot_data.json"
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"✓ Plot data saved to {plot_path}")


def save_trained_models(ml_models, stat_models):
    """Saves all trained models to disk."""
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in {**ml_models, **stat_models}.items():
        safe_name = name.lower().replace(" ", "_")
        path = MODELS_DIR / f"{safe_name}.pkl"
        joblib.dump(model, path)
        logger.info(f"  Model saved: {path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline():
    """Runs the complete train-evaluate pipeline on data/dataset/."""
    logger.info("=" * 60)
    logger.info("  SALES FORECASTING — TRAIN & EVALUATE PIPELINE")
    logger.info("=" * 60)

    # 1. Load & Split
    df = load_featured_data()
    train_df, test_df = chronological_split(df, train_ratio=0.7)

    # 2. Prepare features
    X_train, y_train = prepare_features(train_df, target="sales")
    X_test, y_test = prepare_features(test_df, target="sales")
    test_dates = test_df["date"].values

    logger.info(f"Feature columns ({X_train.shape[1]}): {list(X_train.columns)}")

    # 3. Train ML models
    ml_models = train_ml_models(X_train, y_train)

    # 4. Train statistical models on the training sales series
    train_series = train_df["sales"].values
    stat_models = train_statistical_models(train_series)

    # 5. Evaluate all models
    test_steps = len(y_test)
    metrics_list, predictions = evaluate_models(
        ml_models, stat_models, X_test, y_test, test_dates, test_steps
    )

    if not metrics_list:
        logger.error("No models evaluated successfully. Aborting.")
        return

    # 6. Determine best model (lowest RMSE)
    metrics_df = pd.DataFrame(metrics_list)
    best_idx = metrics_df["RMSE"].idxmin()
    best_model_name = metrics_df.loc[best_idx, "Model"]

    # 7. Save everything
    save_outputs(metrics_list, predictions, test_dates, y_test, best_model_name)
    save_trained_models(ml_models, stat_models)

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
