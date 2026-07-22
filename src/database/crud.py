"""
CRUD Operations Module.
Provides reusable functions for Insert, Retrieve, Update, and Delete operations
across all database tables with proper transaction management.
"""

import logging
from datetime import date
from typing import List, Optional, Dict, Any

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.models import Store, Product, Sale, ModelPerformance, Forecast, User

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# INSERT OPERATIONS
# ============================================================

def insert_stores(session: Session, stores_data: List[Dict[str, Any]]) -> int:
    """
    Bulk inserts store records.

    Args:
        session: Active SQLAlchemy session.
        stores_data: List of dicts with keys: store_id, state_id, store_name (optional).

    Returns:
        int: Number of records inserted.
    """
    try:
        objects = [Store(**row) for row in stores_data]
        session.bulk_save_objects(objects)
        session.commit()
        logger.info("Inserted %d store records.", len(objects))
        return len(objects)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to insert stores: %s", exc)
        raise


def insert_products(session: Session, products_data: List[Dict[str, Any]]) -> int:
    """
    Bulk inserts product records.

    Args:
        session: Active SQLAlchemy session.
        products_data: List of dicts with keys: item_id, dept_id, cat_id, item_name (optional).

    Returns:
        int: Number of records inserted.
    """
    try:
        objects = [Product(**row) for row in products_data]
        session.bulk_save_objects(objects)
        session.commit()
        logger.info("Inserted %d product records.", len(objects))
        return len(objects)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to insert products: %s", exc)
        raise


def insert_sales_batch(session: Session, sales_df: pd.DataFrame, batch_size: int = 5000) -> int:
    """
    Inserts processed sales data in batches using raw SQL for performance.
    Ideal for loading the massive M5 dataset without ORM overhead.

    Args:
        session: Active SQLAlchemy session.
        sales_df: DataFrame with columns matching the sales table.
        batch_size: Number of rows per batch insert.

    Returns:
        int: Total rows inserted.
    """
    required_cols = ["item_id", "store_id", "sale_date", "sales_units", "sell_price", "revenue"]
    for col in required_cols:
        if col not in sales_df.columns:
            raise ValueError(f"Missing required column: {col}")

    total = 0
    try:
        for start in range(0, len(sales_df), batch_size):
            batch = sales_df.iloc[start:start + batch_size]
            records = batch[required_cols].to_dict(orient="records")
            session.execute(
                text("""
                    INSERT INTO sales (item_id, store_id, sale_date, sales_units, sell_price, revenue)
                    VALUES (:item_id, :store_id, :sale_date, :sales_units, :sell_price, :revenue)
                    ON DUPLICATE KEY UPDATE
                        sales_units = VALUES(sales_units),
                        sell_price  = VALUES(sell_price),
                        revenue     = VALUES(revenue)
                """),
                records,
            )
            session.commit()
            total += len(records)
            logger.info("Inserted sales batch: %d / %d rows.", total, len(sales_df))

        logger.info("Completed inserting %d sales records.", total)
        return total
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to insert sales batch: %s", exc)
        raise


def insert_model_performance(session: Session, metrics: Dict[str, Any]) -> int:
    """
    Inserts a single model's evaluation metrics.

    Args:
        session: Active SQLAlchemy session.
        metrics: Dict with keys: model_name, mae, rmse, mape, r2_score, is_best.

    Returns:
        int: The perf_id of the inserted record.
    """
    try:
        record = ModelPerformance(**metrics)
        session.add(record)
        session.commit()
        logger.info("Inserted performance for model '%s' (perf_id=%d).", record.model_name, record.perf_id)
        return record.perf_id
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to insert model performance: %s", exc)
        raise


def insert_forecasts(session: Session, forecasts_df: pd.DataFrame, perf_id: int = None) -> int:
    """
    Bulk inserts forecast results.

    Args:
        session: Active SQLAlchemy session.
        forecasts_df: DataFrame with columns: forecast_date, forecast_value, lower_ci, upper_ci, horizon_days.
        perf_id: Optional model performance ID to link forecasts to a specific model.

    Returns:
        int: Number of records inserted.
    """
    try:
        records = forecasts_df.to_dict(orient="records")
        objects = [Forecast(perf_id=perf_id, **row) for row in records]
        session.bulk_save_objects(objects)
        session.commit()
        logger.info("Inserted %d forecast records.", len(objects))
        return len(objects)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to insert forecasts: %s", exc)
        raise


# ============================================================
# RETRIEVE OPERATIONS
# ============================================================

def get_sales_data(
    session: Session,
    store_id: Optional[str] = None,
    item_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 10000,
) -> pd.DataFrame:
    """
    Retrieves sales data with optional filters.

    Returns:
        pd.DataFrame: Filtered sales records.
    """
    try:
        query = session.query(Sale)

        if store_id:
            query = query.filter(Sale.store_id == store_id)
        if item_id:
            query = query.filter(Sale.item_id == item_id)
        if start_date:
            query = query.filter(Sale.sale_date >= start_date)
        if end_date:
            query = query.filter(Sale.sale_date <= end_date)

        query = query.order_by(Sale.sale_date.desc()).limit(limit)
        results = query.all()

        data = [{
            "sale_id": r.sale_id, "item_id": r.item_id, "store_id": r.store_id,
            "sale_date": r.sale_date, "sales_units": r.sales_units,
            "sell_price": float(r.sell_price), "revenue": float(r.revenue),
            "event_name": r.event_name, "event_type": r.event_type,
        } for r in results]

        logger.info("Retrieved %d sales records.", len(data))
        return pd.DataFrame(data)
    except Exception as exc:
        logger.exception("Failed to retrieve sales data: %s", exc)
        raise


def get_forecast_history(
    session: Session,
    horizon_days: Optional[int] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Retrieves forecast history with optional horizon filter.

    Returns:
        pd.DataFrame: Forecast records.
    """
    try:
        query = session.query(Forecast)
        if horizon_days:
            query = query.filter(Forecast.horizon_days == horizon_days)

        query = query.order_by(Forecast.forecast_date.desc()).limit(limit)
        results = query.all()

        data = [{
            "forecast_id": r.forecast_id, "forecast_date": r.forecast_date,
            "forecast_value": float(r.forecast_value),
            "lower_ci": float(r.lower_ci) if r.lower_ci else None,
            "upper_ci": float(r.upper_ci) if r.upper_ci else None,
            "horizon_days": r.horizon_days,
        } for r in results]

        logger.info("Retrieved %d forecast records.", len(data))
        return pd.DataFrame(data)
    except Exception as exc:
        logger.exception("Failed to retrieve forecast history: %s", exc)
        raise


def get_model_performance(session: Session) -> pd.DataFrame:
    """
    Retrieves all model evaluation metrics.

    Returns:
        pd.DataFrame: Model performance records.
    """
    try:
        results = session.query(ModelPerformance).order_by(ModelPerformance.rmse.asc()).all()
        data = [{
            "perf_id": r.perf_id, "Model": r.model_name,
            "MAE": float(r.mae) if r.mae else None,
            "RMSE": float(r.rmse) if r.rmse else None,
            "MAPE (%)": float(r.mape) if r.mape else None,
            "R2 Score": float(r.r2_score) if r.r2_score else None,
            "is_best": r.is_best, "trained_at": r.trained_at,
        } for r in results]

        logger.info("Retrieved %d model performance records.", len(data))
        return pd.DataFrame(data)
    except Exception as exc:
        logger.exception("Failed to retrieve model performance: %s", exc)
        raise


def get_best_model(session: Session) -> Optional[Dict[str, Any]]:
    """Returns the best-performing model record (is_best=1 or lowest RMSE)."""
    try:
        result = session.query(ModelPerformance).filter(ModelPerformance.is_best == 1).first()
        if result is None:
            result = session.query(ModelPerformance).order_by(ModelPerformance.rmse.asc()).first()
        if result:
            return {
                "model_name": result.model_name,
                "rmse": float(result.rmse) if result.rmse else None,
                "mae": float(result.mae) if result.mae else None,
                "r2_score": float(result.r2_score) if result.r2_score else None,
            }
        return None
    except Exception as exc:
        logger.exception("Failed to retrieve best model: %s", exc)
        raise


# ============================================================
# UPDATE OPERATIONS
# ============================================================

def update_model_best_flag(session: Session, perf_id: int) -> None:
    """
    Sets the is_best flag on the specified model and clears it from all others.

    Args:
        session: Active SQLAlchemy session.
        perf_id: The performance record to mark as best.
    """
    try:
        # Reset all
        session.query(ModelPerformance).update({ModelPerformance.is_best: 0})
        # Set the best
        session.query(ModelPerformance).filter(ModelPerformance.perf_id == perf_id).update(
            {ModelPerformance.is_best: 1}
        )
        session.commit()
        logger.info("Updated best model flag to perf_id=%d.", perf_id)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to update best model flag: %s", exc)
        raise


def update_sale_record(session: Session, sale_id: int, updates: Dict[str, Any]) -> None:
    """
    Updates a specific sale record.

    Args:
        session: Active SQLAlchemy session.
        sale_id: The sale record to update.
        updates: Dict of column-value pairs to update.
    """
    try:
        session.query(Sale).filter(Sale.sale_id == sale_id).update(updates)
        session.commit()
        logger.info("Updated sale_id=%d.", sale_id)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to update sale record: %s", exc)
        raise


# ============================================================
# DELETE OPERATIONS
# ============================================================

def delete_forecasts_by_horizon(session: Session, horizon_days: int) -> int:
    """
    Deletes all forecast records for a specific horizon.

    Returns:
        int: Number of records deleted.
    """
    try:
        count = session.query(Forecast).filter(Forecast.horizon_days == horizon_days).delete()
        session.commit()
        logger.info("Deleted %d forecast records for horizon=%d.", count, horizon_days)
        return count
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to delete forecasts: %s", exc)
        raise


def delete_all_model_performance(session: Session) -> int:
    """Deletes all model performance records (used before re-evaluation)."""
    try:
        count = session.query(ModelPerformance).delete()
        session.commit()
        logger.info("Deleted %d model performance records.", count)
        return count
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to delete model performance records: %s", exc)
        raise
