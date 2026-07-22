"""
SQLAlchemy ORM Models.
Maps the MySQL tables to Python classes for type-safe, Pythonic database access.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Date, DateTime,
    DECIMAL, TIMESTAMP, Enum, ForeignKey, Index, text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Store(Base):
    """Represents a retail store location."""
    __tablename__ = "stores"

    store_id = Column(String(20), primary_key=True)
    state_id = Column(String(10), nullable=False, index=True)
    store_name = Column(String(100), default=None)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    # Relationships
    sales = relationship("Sale", back_populates="store", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Store(store_id='{self.store_id}', state='{self.state_id}')>"


class Product(Base):
    """Represents a product (item) in the catalog."""
    __tablename__ = "products"

    item_id = Column(String(50), primary_key=True)
    dept_id = Column(String(30), nullable=False, index=True)
    cat_id = Column(String(30), nullable=False, index=True)
    item_name = Column(String(150), default=None)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    # Relationships
    sales = relationship("Sale", back_populates="product", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product(item_id='{self.item_id}', dept='{self.dept_id}')>"


class Sale(Base):
    """Represents a single daily sales record for one item in one store."""
    __tablename__ = "sales"

    sale_id = Column(BigInteger, primary_key=True, autoincrement=True)
    item_id = Column(String(50), ForeignKey("products.item_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    store_id = Column(String(20), ForeignKey("stores.store_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    sale_date = Column(Date, nullable=False, index=True)
    sales_units = Column(Integer, nullable=False, default=0)
    sell_price = Column(DECIMAL(10, 2), nullable=False, default=0.00)
    revenue = Column(DECIMAL(12, 2), nullable=False, default=0.00)
    event_name = Column(String(60), default="No_Event")
    event_type = Column(String(30), default="No_Event")
    is_snap = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    product = relationship("Product", back_populates="sales")
    store = relationship("Store", back_populates="sales")

    # Composite indexes
    __table_args__ = (
        Index("idx_sales_item_store", "item_id", "store_id"),
        Index("idx_sales_composite", "store_id", "item_id", "sale_date"),
    )

    def __repr__(self):
        return f"<Sale(id={self.sale_id}, item='{self.item_id}', store='{self.store_id}', date={self.sale_date})>"


class ModelPerformance(Base):
    """Stores evaluation metrics for each trained model."""
    __tablename__ = "model_performance"

    perf_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(60), nullable=False, index=True)
    mae = Column(DECIMAL(14, 4), default=None)
    rmse = Column(DECIMAL(14, 4), default=None)
    mape = Column(DECIMAL(10, 4), default=None)
    r2_score = Column(DECIMAL(10, 6), default=None)
    is_best = Column(Integer, default=0, index=True)
    trained_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    forecasts = relationship("Forecast", back_populates="model_performance", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ModelPerformance(model='{self.model_name}', rmse={self.rmse}, best={self.is_best})>"


class Forecast(Base):
    """Stores individual forecast data points."""
    __tablename__ = "forecasts"

    forecast_id = Column(BigInteger, primary_key=True, autoincrement=True)
    perf_id = Column(Integer, ForeignKey("model_performance.perf_id", ondelete="SET NULL", onupdate="CASCADE"), default=None)
    forecast_date = Column(Date, nullable=False, index=True)
    forecast_value = Column(DECIMAL(14, 2), nullable=False)
    lower_ci = Column(DECIMAL(14, 2), default=None)
    upper_ci = Column(DECIMAL(14, 2), default=None)
    horizon_days = Column(Integer, nullable=False, default=7, index=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    model_performance = relationship("ModelPerformance", back_populates="forecasts")

    def __repr__(self):
        return f"<Forecast(date={self.forecast_date}, value={self.forecast_value}, horizon={self.horizon_days})>"


class User(Base):
    """Optional user model for dashboard access control."""
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(120), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum("admin", "analyst", "viewer", name="user_role_enum"), default="viewer")
    is_active = Column(Integer, default=1)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"
