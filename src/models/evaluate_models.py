"""
Model Evaluation Module.
Evaluates all trained models using standard metrics, generates a comparison table,
saves results to CSV, and creates forecast output for the dashboard.
"""

import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates standard regression metrics: MAE, RMSE, MAPE, R2 Score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE calculation — exclude zeros to avoid skewed explosion
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = 0.0

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R2 Score': round(r2, 4)
    }


def prepare_ml_test_features(df: pd.DataFrame, target_col: str = 'revenue'):
    """Prepares the test set features, returning features, true values, and dates."""
    df = df.dropna()
    dates = df['date'] if 'date' in df.columns else pd.Series(dtype='datetime64[ns]')

    cols_to_drop = ['date', target_col]
    X_test = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Strictly select numeric columns
    X_test = X_test.select_dtypes(include=[np.number])

    y_test = df[target_col]
    return X_test, y_test, dates


def plot_predictions(actual_df: pd.DataFrame, predictions_dict: dict, output_dir: str):
    """Generates an interactive Plotly chart comparing Actual vs Predicted daily revenue."""
    logger.info("Generating prediction plots...")

    fig = go.Figure()

    # Add Actuals
    fig.add_trace(go.Scatter(x=actual_df['date'], y=actual_df['revenue'],
                             mode='lines', name='Actual Revenue', line=dict(color='#3B82F6', width=3)))

    # Add all model predictions
    colors = ['#10B981', '#F43F5E', '#F59E0B', '#8B5CF6', '#06B6D4', '#EC4899']
    for idx, (model_name, pred_values) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(x=actual_df['date'], y=pred_values,
                                 mode='lines', name=model_name, opacity=0.7,
                                 line=dict(color=colors[idx % len(colors)])))

    fig.update_layout(
        title='Forecasted vs Actual Daily Revenue',
        xaxis_title='Date',
        yaxis_title='Total Revenue ($)',
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, 'forecast_comparison.html')
    fig.write_html(html_path)
    logger.info(f"Interactive plot saved to {html_path}")


def evaluate_all_models(input_path: str, models_dir: str, figures_dir: str):
    """
    Loads test data and all models, computes predictions, generates metric comparisons,
    saves evaluation_metrics.csv and forecast.csv for the dashboard.
    """
    from src.models.model_utils import load_dataset, chronological_split, aggregate_to_daily, load_model

    logger.info("=" * 50)
    logger.info("Starting Evaluation Pipeline...")
    logger.info("=" * 50)

    # 1. Load and Split Data
    df = load_dataset(input_path, nrows=500000)

    if df.empty:
        logger.error("No data loaded. Aborting evaluation.")
        return

    _, test_df = chronological_split(df, train_ratio=0.7)

    # 2. Get the Ground Truth (Daily Actuals)
    test_df_clean = test_df.dropna()
    if test_df_clean.empty:
        logger.error("Test set is empty after dropping NaNs. Aborting evaluation.")
        return

    actual_daily_df = aggregate_to_daily(test_df_clean, target_col='revenue')
    y_true_daily = actual_daily_df['revenue'].values
    test_steps = len(y_true_daily)

    predictions_daily = {}
    metrics_summary = []

    # 3. Evaluate ML Models (predict granular, then group by date and sum)
    X_test, y_test, test_dates = prepare_ml_test_features(test_df_clean, target_col='revenue')

    ml_models = ['linear_regression', 'random_forest', 'xgboost']
    for m_name in ml_models:
        model_path = os.path.join(models_dir, f"{m_name}.pkl")
        if os.path.exists(model_path):
            logger.info(f"Evaluating {m_name}...")
            try:
                model = load_model(model_path)

                # Predict at granular level
                preds = model.predict(X_test)

                # Aggregate predictions to daily totals
                temp_df = pd.DataFrame({'date': test_dates.values, 'pred': preds})
                pred_daily = temp_df.groupby('date')['pred'].sum().reset_index().sort_values('date')['pred'].values

                # Ensure lengths match
                min_len = min(len(y_true_daily), len(pred_daily))
                predictions_daily[m_name] = pred_daily[:min_len]

                metrics = calculate_metrics(y_true_daily[:min_len], pred_daily[:min_len])
                metrics['Model'] = m_name.replace('_', ' ').title()
                metrics_summary.append(metrics)
                logger.info(f"  [OK] {m_name}: RMSE={metrics['RMSE']}, R²={metrics['R2 Score']}")
            except Exception as e:
                logger.warning(f"  [X] Error evaluating {m_name}: {e}")
        else:
            logger.warning(f"Model file {model_path} not found.")

    # 4. Evaluate Statistical Models (forecast macro directly)
    stat_models = ['arima', 'sarima', 'prophet']
    for m_name in stat_models:
        model_path = os.path.join(models_dir, f"{m_name}.pkl")
        if os.path.exists(model_path):
            logger.info(f"Evaluating {m_name}...")
            try:
                model = load_model(model_path)

                if m_name in ['arima', 'sarima']:
                    pred_daily = model.forecast(steps=test_steps)
                elif m_name == 'prophet':
                    future = pd.DataFrame({'ds': actual_daily_df['date']})
                    forecast = model.predict(future)
                    pred_daily = forecast['yhat'].values

                min_len = min(len(y_true_daily), len(pred_daily))
                predictions_daily[m_name] = pred_daily[:min_len]

                metrics = calculate_metrics(y_true_daily[:min_len], pred_daily[:min_len])
                metrics['Model'] = m_name.replace('_', ' ').title()
                metrics_summary.append(metrics)
                logger.info(f"  [OK] {m_name}: RMSE={metrics['RMSE']}, R²={metrics['R2 Score']}")
            except Exception as e:
                logger.warning(f"  [X] Error evaluating {m_name}: {e}")
        else:
            logger.info(f"Model file {model_path} not found. Skipping.")

    # 5. Generate Comparison Table
    if not metrics_summary:
        logger.error("No models were evaluated successfully.")
        return

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df = metrics_df[['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R2 Score']]

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION COMPARISON")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    # 6. Determine the best model
    best_model_idx = metrics_df['RMSE'].idxmin()
    best_model_name = metrics_df.loc[best_model_idx, 'Model']
    best_rmse = metrics_df.loc[best_model_idx, 'RMSE']

    print("-" * 60)
    print(f"  BEST MODEL: {best_model_name.upper()} (Lowest RMSE: {best_rmse})")
    print("-" * 60 + "\n")

    # 7. Save evaluation_metrics.csv
    metrics_csv_path = os.path.join(models_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"[OK] Metrics saved to {metrics_csv_path}")

    # 8. Generate forecast.csv for the dashboard Forecast page
    _generate_forecast_csv(actual_daily_df, predictions_daily, best_model_name, models_dir, input_path)

    # 9. Generate Plots
    try:
        min_pred_len = min(len(v) for v in predictions_daily.values()) if predictions_daily else 0
        if min_pred_len > 0:
            plot_df = actual_daily_df.head(min_pred_len)
            plot_predictions(plot_df, {k: v[:min_pred_len] for k, v in predictions_daily.items()}, figures_dir)
    except Exception as e:
        logger.warning(f"Plot generation failed (non-critical): {e}")

    logger.info("[OK] Evaluation Pipeline completed.")


def _generate_forecast_csv(actual_daily_df, predictions_dict, best_model_name, models_dir, input_path):
    """Generates a forecast.csv file the Streamlit dashboard can read."""
    try:
        best_key = best_model_name.lower().replace(' ', '_')

        if best_key in predictions_dict:
            pred = predictions_dict[best_key]
        elif predictions_dict:
            first_key = list(predictions_dict.keys())[0]
            pred = predictions_dict[first_key]
        else:
            return

        n = min(len(actual_daily_df), len(pred))
        forecast_df = pd.DataFrame({
            'date': actual_daily_df['date'].values[:n],
            'forecast': pred[:n],
            'lower_ci': pred[:n] * 0.85,
            'upper_ci': pred[:n] * 1.15,
        })

        predictions_dir = Path(input_path).resolve().parent.parent / "data" / "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        forecast_path = predictions_dir / "forecast.csv"
        forecast_df.to_csv(forecast_path, index=False)
        logger.info(f"[OK] Forecast data saved to {forecast_path} ({len(forecast_df)} rows)")

    except Exception as e:
        logger.warning(f"Forecast CSV generation failed (non-critical): {e}")
