-- ============================================================
-- Sales Revenue Forecasting System — Database Schema
-- Engine: MySQL 8.0+
-- ============================================================

CREATE DATABASE IF NOT EXISTS sales_forecasting
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE sales_forecasting;

-- ────────────────────────────────────────────────────────────
-- 1. STORES
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stores (
    store_id        VARCHAR(20)     NOT NULL,
    state_id        VARCHAR(10)     NOT NULL,
    store_name      VARCHAR(100)    DEFAULT NULL,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (store_id),
    INDEX idx_stores_state (state_id)
) ENGINE=InnoDB;

-- ────────────────────────────────────────────────────────────
-- 2. PRODUCTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    item_id         VARCHAR(50)     NOT NULL,
    dept_id         VARCHAR(30)     NOT NULL,
    cat_id          VARCHAR(30)     NOT NULL,
    item_name       VARCHAR(150)    DEFAULT NULL,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (item_id),
    INDEX idx_products_dept (dept_id),
    INDEX idx_products_cat (cat_id)
) ENGINE=InnoDB;

-- ────────────────────────────────────────────────────────────
-- 3. SALES  (fact table — the core transactional data)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sales (
    sale_id         BIGINT          NOT NULL AUTO_INCREMENT,
    item_id         VARCHAR(50)     NOT NULL,
    store_id        VARCHAR(20)     NOT NULL,
    sale_date       DATE            NOT NULL,
    sales_units     INT             NOT NULL DEFAULT 0,
    sell_price      DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    revenue         DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    event_name      VARCHAR(60)     DEFAULT 'No_Event',
    event_type      VARCHAR(30)     DEFAULT 'No_Event',
    is_snap         TINYINT(1)      DEFAULT 0,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (sale_id),
    FOREIGN KEY (item_id)  REFERENCES products(item_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (store_id) REFERENCES stores(store_id)  ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_sales_date (sale_date),
    INDEX idx_sales_item_store (item_id, store_id),
    INDEX idx_sales_composite (store_id, item_id, sale_date),
    UNIQUE KEY uq_sales_record (item_id, store_id, sale_date)
) ENGINE=InnoDB;

-- ────────────────────────────────────────────────────────────
-- 4. MODEL_PERFORMANCE  (evaluation metrics for every model)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_performance (
    perf_id         INT             NOT NULL AUTO_INCREMENT,
    model_name      VARCHAR(60)     NOT NULL,
    mae             DECIMAL(14, 4)  DEFAULT NULL,
    rmse            DECIMAL(14, 4)  DEFAULT NULL,
    mape            DECIMAL(10, 4)  DEFAULT NULL,
    r2_score        DECIMAL(10, 6)  DEFAULT NULL,
    is_best         TINYINT(1)      DEFAULT 0,
    trained_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (perf_id),
    INDEX idx_perf_model (model_name),
    INDEX idx_perf_best (is_best)
) ENGINE=InnoDB;

-- ────────────────────────────────────────────────────────────
-- 5. FORECASTS  (generated predictions)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     BIGINT          NOT NULL AUTO_INCREMENT,
    perf_id         INT             DEFAULT NULL,
    forecast_date   DATE            NOT NULL,
    forecast_value  DECIMAL(14, 2)  NOT NULL,
    lower_ci        DECIMAL(14, 2)  DEFAULT NULL,
    upper_ci        DECIMAL(14, 2)  DEFAULT NULL,
    horizon_days    INT             NOT NULL DEFAULT 7,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (forecast_id),
    FOREIGN KEY (perf_id) REFERENCES model_performance(perf_id) ON DELETE SET NULL ON UPDATE CASCADE,
    INDEX idx_forecast_date (forecast_date),
    INDEX idx_forecast_horizon (horizon_days)
) ENGINE=InnoDB;

-- ────────────────────────────────────────────────────────────
-- 6. USERS  (optional — for dashboard access control)
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id         INT             NOT NULL AUTO_INCREMENT,
    username        VARCHAR(50)     NOT NULL,
    email           VARCHAR(120)    NOT NULL,
    password_hash   VARCHAR(255)    NOT NULL,
    role            ENUM('admin', 'analyst', 'viewer') DEFAULT 'viewer',
    is_active       TINYINT(1)      DEFAULT 1,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id),
    UNIQUE KEY uq_users_username (username),
    UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB;
