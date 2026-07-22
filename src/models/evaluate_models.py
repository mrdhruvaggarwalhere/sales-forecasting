"""
Model Evaluation Module.
Evaluates all trained models using standard metrics, generates a comparison table,
and visualizes Actual vs Predicted revenue using Plotly.
"""

import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.model_utils import load_dataset, chronological_split, aggregate_to_daily, load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates standard regression metrics: MAE, RMSE, MAPE, R2 Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE calculation with safeguard against division by zero
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R2 Score': round(r2, 4)
    }


def prepare_ml_test_features(df: pd.DataFrame, target_col: str = 'revenue'):
    """Prepares the test set features, returning features, true values, and dates for aggregation."""
    # Ensure NaNs are dropped from test set as well to match predictions
    df = df.dropna()
    cols_to_drop = ['date', target_col]
    X_test = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y_test = df[target_col]
    return X_test, y_test, df['date']


def plot_predictions(actual_df: pd.DataFrame, predictions_dict: dict, output_dir: str):
    """
    Generates an interactive Plotly chart comparing Actual vs Predicted daily revenue.
    """
    logger.info("Generating prediction plots...")
    
    fig = go.Figure()
    
    # Add Actuals
    fig.add_trace(go.Scatter(x=actual_df['date'], y=actual_df['revenue'], 
                             mode='lines', name='Actual Revenue', line=dict(color='black', width=3)))
    
    # Add all model predictions
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    for idx, (model_name, pred_values) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(x=actual_df['date'], y=pred_values, 
                                 mode='lines', name=model_name, opacity=0.7, 
                                 line=dict(color=colors[idx % len(colors)])))
        
    fig.update_layout(
        title='Forecasted vs Actual Daily Revenue',
        xaxis_title='Date',
        yaxis_title='Total Revenue ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, 'forecast_comparison.html')
    fig.write_html(html_path)
    logger.info(f"Interactive plot saved to {html_path}")


def evaluate_all_models(input_path: str, models_dir: str, figures_dir: str):
    """
    Loads test data and all models, computes predictions, aligns them to daily totals,
    generates a metric comparison table, and determines the best model.
    """
    logger.info("Starting Evaluation Pipeline...")
    
    # 1. Load and Split Data
    df = load_dataset(input_path, nrows=500000)
    _, test_df = chronological_split(df, train_ratio=0.7)
    
    # 2. Get the Ground Truth (Daily Actuals)
    # We must drop NaNs first so dates align perfectly with the ML models
    test_df_clean = test_df.dropna()
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
            model = load_model(model_path)
            
            # Predict at granular level
            preds = model.predict(X_test)
            
            # Create a temp dataframe to aggregate predictions to daily totals
            temp_df = pd.DataFrame({'date': test_dates.values, 'pred': preds})
            pred_daily = temp_df.groupby('date')['pred'].sum().reset_index().sort_values('date')['pred'].values
            
            predictions_daily[m_name] = pred_daily
            
            # Calculate metrics against daily actuals
            metrics = calculate_metrics(y_true_daily, pred_daily)
            metrics['Model'] = m_name.replace('_', ' ').title()
            metrics_summary.append(metrics)
        else:
            logger.warning(f"Model file {model_path} not found.")

    # 4. Evaluate Statistical Models (forecast macro directly)
    stat_models = ['arima', 'sarima', 'prophet']
    for m_name in stat_models:
        model_path = os.path.join(models_dir, f"{m_name}.pkl")
        if os.path.exists(model_path):
            logger.info(f"Evaluating {m_name}...")
            model = load_model(model_path)
            
            if m_name in ['arima', 'sarima']:
                # Statsmodels forecast method
                pred_daily = model.forecast(steps=test_steps)
            elif m_name == 'prophet':
                # Prophet requires a future dataframe
                future = pd.DataFrame({'ds': actual_daily_df['date']})
                forecast = model.predict(future)
                pred_daily = forecast['yhat'].values
                
            predictions_daily[m_name] = pred_daily
            
            metrics = calculate_metrics(y_true_daily, pred_daily)
            metrics['Model'] = m_name.replace('_', ' ').title()
            metrics_summary.append(metrics)
        else:
            logger.warning(f"Model file {model_path} not found.")

    # 5. Generate Comparison Table
    if not metrics_summary:
        logger.error("No models were evaluated successfully.")
        return
        
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df = metrics_df[['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R2 Score']]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION COMPARISON")
    print("="*50)
    print(metrics_df.to_markdown(index=False))
    
    # 6. Determine the best model
    best_model_idx = metrics_df['RMSE'].idxmin()
    best_model_name = metrics_df.loc[best_model_idx, 'Model']
    best_rmse = metrics_df.loc[best_model_idx, 'RMSE']
    
    print("-" * 50)
    print(f"🏆 BEST MODEL: {best_model_name.upper()} (Lowest RMSE: {best_rmse})")
    print("-" * 50 + "\n")
    
    # 7. Generate Plots
    plot_predictions(actual_daily_df, predictions_daily, figures_dir)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "outputs" / "models"
    FIGURES_DIR = BASE_DIR / "outputs" / "figures"
    
    evaluate_all_models(
        input_path=str(PROCESSED_DIR / "featured_sales.csv"),
        models_dir=str(MODELS_DIR),
        figures_dir=str(FIGURES_DIR)
    )
