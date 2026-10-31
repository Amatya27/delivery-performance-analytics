-- Star schema for the delivery analysis.
-- Two fact tables at different grains:
--   fact_orders       one row per order (delivery timings, totals, review)
--   fact_order_items  one row per item in an order (price, freight, distance)
-- Dimensions: date, customer, seller, product.

PRAGMA foreign_keys = ON;

CREATE TABLE dim_date (
    date_key     INTEGER PRIMARY KEY,   -- yyyymmdd
    date         TEXT NOT NULL,
    year         INTEGER,
    quarter      INTEGER,
    month        INTEGER,
    month_name   TEXT,
    day          INTEGER,
    weekday      TEXT,
    is_weekend   INTEGER
);

CREATE TABLE dim_customer (
    customer_id         TEXT PRIMARY KEY,
    customer_unique_id  TEXT,             -- same person can have several customer_ids
    zip_prefix          INTEGER,
    city                TEXT,
    state               TEXT,
    state_name          TEXT,
    lat                 REAL,
    lng                 REAL
);

CREATE TABLE dim_seller (
    seller_id    TEXT PRIMARY KEY,
    zip_prefix   INTEGER,
    city         TEXT,
    state        TEXT,
    state_name   TEXT,
    lat          REAL,
    lng          REAL
);

CREATE TABLE dim_product (
    product_id    TEXT PRIMARY KEY,
    category_pt   TEXT,
    category_en   TEXT,
    weight_g      REAL,
    length_cm     REAL,
    height_cm     REAL,
    width_cm      REAL,
    volume_cm3    REAL,
    photos_qty    REAL
);

CREATE TABLE fact_orders (
    order_id           TEXT PRIMARY KEY,
    customer_id        TEXT REFERENCES dim_customer(customer_id),
    purchase_date_key  INTEGER REFERENCES dim_date(date_key),
    order_status       TEXT,
    purchase_ts        TEXT,
    approved_ts        TEXT,
    carrier_ts         TEXT,
    delivered_ts       TEXT,
    estimated_date     TEXT,
    n_items            INTEGER,
    n_sellers          INTEGER,
    main_seller_id     TEXT REFERENCES dim_seller(seller_id),
    main_product_id    TEXT REFERENCES dim_product(product_id),
    main_category_en   TEXT,
    items_price        REAL,
    items_freight      REAL,
    total_weight_g     REAL,
    max_distance_km    REAL,
    payment_total      REAL,
    payment_type_main  TEXT,
    max_installments   INTEGER,
    payment_gap        REAL,     -- payment_total - (items_price + items_freight)
    payment_mismatch   INTEGER,  -- 1 if the gap is bigger than the tolerance
    review_score       INTEGER,
    approval_hours     REAL,
    handling_days      REAL,     -- approved -> handed to carrier
    transit_days       REAL,     -- carrier -> customer
    delivery_days      REAL,     -- purchase -> customer
    promised_days      REAL,     -- purchase -> estimated date shown to customer
    delay_days         REAL,     -- delivered date - estimated date (negative = early)
    is_late            INTEGER
);

CREATE TABLE fact_order_items (
    order_id             TEXT REFERENCES fact_orders(order_id),
    order_item_id        INTEGER,
    product_id           TEXT REFERENCES dim_product(product_id),
    seller_id            TEXT REFERENCES dim_seller(seller_id),
    shipping_limit_date  TEXT,
    price                REAL,
    freight_value        REAL,
    distance_km          REAL,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE INDEX idx_orders_customer ON fact_orders(customer_id);
CREATE INDEX idx_orders_date ON fact_orders(purchase_date_key);
CREATE INDEX idx_orders_seller ON fact_orders(main_seller_id);
CREATE INDEX idx_items_seller ON fact_order_items(seller_id);
