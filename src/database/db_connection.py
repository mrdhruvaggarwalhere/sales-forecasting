"""
Database Connection Module.
Manages MySQL connection pooling via SQLAlchemy engine and session factory.
"""

import os
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration from .env ────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "sales_forecasting")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Engine (singleton — connection pool) ───────────────────────────────────────
_engine = None


def get_engine():
    """
    Returns a singleton SQLAlchemy engine with connection pooling.

    Pool configuration:
        - pool_size: Maximum number of persistent connections.
        - max_overflow: Extra connections allowed beyond pool_size under load.
        - pool_recycle: Seconds before a connection is recycled (prevents MySQL timeout).
        - pool_pre_ping: Tests connections before use to avoid stale connections.
    """
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(
                DATABASE_URL,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("Database engine created successfully (%s@%s/%s).", DB_USER, DB_HOST, DB_NAME)
        except Exception as exc:
            logger.exception("Failed to create database engine: %s", exc)
            raise
    return _engine


# ── Session Factory ────────────────────────────────────────────────────────────
_SessionFactory = None


def get_session_factory():
    """Returns a session factory bound to the singleton engine."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def get_session() -> Session:
    """
    Creates and returns a new SQLAlchemy session.

    Usage:
        session = get_session()
        try:
            # work with session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    """
    factory = get_session_factory()
    return factory()


# ── Utility ────────────────────────────────────────────────────────────────────
def test_connection() -> bool:
    """Tests the database connection and returns True if successful."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection test: SUCCESS")
        return True
    except Exception as exc:
        logger.error("Database connection test: FAILED — %s", exc)
        return False
