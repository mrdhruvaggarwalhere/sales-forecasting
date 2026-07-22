"""
Database Orchestration Module.
High-level functions that connect the ML pipeline outputs to the database layer.
Intended to be called from main.py or the Streamlit dashboard.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.db_connection import get_engine, get_session
from src.database.models import Base
from src.database import crud

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def create_all_tables() -> None:
    """
    Creates all tables defined in models.py if they don't already exist.
    Uses SQLAlchemy's metadata.create_all method.
    """
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info("All database tables created (or already exist).")
    except Exception as exc:
        logger.exception("Failed to create tables: %s", exc)
        raise


def load_processed_data_to_db(processed_csv_path: str) -> None:
    """
    Loads the preprocessed CSV into the stores, products, and sales tables.

    Steps:
        1. Extract unique stores and insert them.
        2. Extract unique products and insert them.
        3. Batch-insert sales records.
    """
    logger.info("Loading processed data into the database...")
    session = get_session()

    try:
        df = pd.read_csv(processed_csv_path, nrows=100_000)  # cap for safety
        if "date" in df.columns:
            df.rename(columns={"date": "sale_date"}, inplace=True)
        df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date

        # ── Insert Stores ──────────────────────────────────
        if "store_id" in df.columns and "state_id" in df.columns:
            stores = df[["store_id", "state_id"]].drop_duplicates().to_dict(orient="records")
            try:
                crud.insert_stores(session, stores)
            except Exception:
                logger.warning("Stores may already exist — skipping.")
                session.rollback()

        # ── Insert Products ────────────────────────────────
        product_cols = [c for c in ["item_id", "dept_id", "cat_id"] if c in df.columns]
        if len(product_cols) == 3:
            products = df[product_cols].drop_duplicates().to_dict(orient="records")
            try:
                crud.insert_products(session, products)
            except Exception:
                logger.warning("Products may already exist — skipping.")
                session.rollback()

        # ── Insert Sales ───────────────────────────────────
        crud.insert_sales_batch(session, df, batch_size=5000)

        logger.info("Processed data loaded into database successfully.")
    except Exception as exc:
        logger.exception("Failed to load processed data: %s", exc)
        raise
    finally:
        session.close()


def load_metrics_to_db(metrics_csv_path: str) -> None:
    """
    Loads model evaluation metrics from CSV into the model_performance table.
    Automatically marks the best model (lowest RMSE).
    """
    logger.info("Loading model metrics into database...")
    session = get_session()

    try:
        df = pd.read_csv(metrics_csv_path)
        best_perf_id = None
        best_rmse = float("inf")

        for _, row in df.iterrows():
            metrics = {
                "model_name": row.get("Model", "Unknown"),
                "mae": row.get("MAE"),
                "rmse": row.get("RMSE"),
                "mape": row.get("MAPE (%)"),
                "r2_score": row.get("R2 Score"),
                "is_best": 0,
            }
            perf_id = crud.insert_model_performance(session, metrics)
            if metrics["rmse"] is not None and metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_perf_id = perf_id

        # Mark the best model
        if best_perf_id:
            crud.update_model_best_flag(session, best_perf_id)

        logger.info("Model metrics loaded successfully.")
    except Exception as exc:
        logger.exception("Failed to load metrics: %s", exc)
        raise
    finally:
        session.close()


def load_forecasts_to_db(forecast_csv_path: str, horizon_days: int = 7) -> None:
    """
    Loads forecast results from CSV into the forecasts table.
    """
    logger.info("Loading forecasts into database...")
    session = get_session()

    try:
        df = pd.read_csv(forecast_csv_path)
        if "date" in df.columns:
            df.rename(columns={"date": "forecast_date"}, inplace=True)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date

        rename_map = {"forecast": "forecast_value"}
        df.rename(columns=rename_map, inplace=True)

        if "horizon_days" not in df.columns:
            df["horizon_days"] = horizon_days

        # Get best model perf_id
        best_model = crud.get_best_model(session)
        perf_id = None
        if best_model:
            from src.database.models import ModelPerformance
            rec = session.query(ModelPerformance).filter(
                ModelPerformance.model_name == best_model["model_name"]
            ).first()
            if rec:
                perf_id = rec.perf_id

        cols = ["forecast_date", "forecast_value", "lower_ci", "upper_ci", "horizon_days"]
        forecast_df = df[[c for c in cols if c in df.columns]]
        crud.insert_forecasts(session, forecast_df, perf_id=perf_id)

        logger.info("Forecasts loaded successfully.")
    except Exception as exc:
        logger.exception("Failed to load forecasts: %s", exc)
        raise
    finally:
        session.close()


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # 1. Create tables
    create_all_tables()

    # 2. Load processed data
    processed_path = BASE_DIR / "data" / "processed" / "processed_sales.csv"
    if processed_path.is_file():
        load_processed_data_to_db(str(processed_path))

    # 3. Load metrics
    metrics_path = BASE_DIR / "outputs" / "models" / "evaluation_metrics.csv"
    if metrics_path.is_file():
        load_metrics_to_db(str(metrics_path))

    # 4. Load forecasts
    forecast_path = BASE_DIR / "data" / "predictions" / "forecast.csv"
    if forecast_path.is_file():
        load_forecasts_to_db(str(forecast_path))
