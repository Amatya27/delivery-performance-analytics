"""Export flat CSVs for Tableau Public into tableau/.

Tableau Public can't connect to SQLite, so I query the database here and save
one CSV per dashboard need. All of these come from the star schema views and
the model tables, nothing is recalculated by hand.

Usage:
    python src/export_tableau.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "delivery.db"
OUT = ROOT / "tableau"

ORDERS_SQL = """
SELECT
    o.order_id,
    date(o.purchase_ts)                       AS purchase_date,
    substr(o.purchase_ts, 1, 7) || '-01'      AS purchase_month,
    c.state                                   AS customer_state,
    c.state_name                              AS customer_state_name,
    c.city                                    AS customer_city,
    ROUND(c.lat, 4)                           AS customer_lat,
    ROUND(c.lng, 4)                           AS customer_lng,
    s.state                                   AS seller_state,
    s.state || ' -> ' || c.state              AS lane,
    o.main_category_en                        AS category,
    o.n_items,
    o.items_price,
    o.items_freight,
    ROUND(o.total_weight_g / 1000.0, 3)       AS weight_kg,
    o.max_distance_km                         AS distance_km,
    o.promised_days,
    o.delivery_days,
    o.handling_days,
    o.transit_days,
    o.delay_days,
    o.is_late,
    o.review_score,
    p.split                                   AS model_split,
    p.late_risk,
    p.flagged_high_risk
FROM fact_orders o
JOIN dim_customer c          ON c.customer_id = o.customer_id
LEFT JOIN dim_seller s       ON s.seller_id   = o.main_seller_id
LEFT JOIN model_predictions p ON p.order_id   = o.order_id
WHERE o.is_late IS NOT NULL
ORDER BY o.purchase_ts
"""

# nicer names for the feature importance chart
FEATURE_LABELS = {
    "promised_days": "Days promised to customer",
    "max_distance_km": "Seller to customer distance",
    "items_freight": "Freight paid",
    "items_price": "Order value",
    "total_weight_g": "Package weight",
    "n_items": "Number of items",
    "n_sellers": "Number of sellers",
    "freight_ratio": "Freight / order value",
    "same_state": "Same-state shipment",
    "purchase_hour": "Hour of purchase",
    "purchase_weekday": "Day of week",
    "max_installments": "Payment installments",
    "seller_prior_orders": "Seller's past delivered orders",
    "seller_prior_late_rate": "Seller's past late rate",
    "seller_prior_avg_days": "Seller's past avg delivery days",
    "state_recent_late_rate": "Late rate into state, last 14 days",
    "state_recent_avg_days": "Avg delivery days into state, last 14 days",
    "promise_slack_days": "Promise slack vs recent delivery times",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "main_category_en": "Product category",
    "payment_type_main": "Payment type",
}


def main():
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)

    exports = {
        "orders_delivery.csv": pd.read_sql(ORDERS_SQL, con),
        "monthly_kpis.csv": pd.read_sql("SELECT * FROM v_monthly_delivery_kpis", con),
        "state_performance.csv": pd.read_sql("SELECT * FROM v_state_performance", con),
        "lane_performance.csv": pd.read_sql(
            "SELECT origin_state || ' -> ' || destination_state AS lane, * FROM v_lane_performance", con
        ),
        "category_performance.csv": pd.read_sql("SELECT * FROM v_category_performance", con),
        "model_monitoring.csv": pd.read_sql("SELECT * FROM model_monitoring", con),
        "model_risk_deciles.csv": pd.read_sql("SELECT * FROM model_risk_deciles", con),
        "model_feature_importance.csv": pd.read_sql("SELECT * FROM model_feature_importance", con),
    }
    con.close()

    exports["monthly_kpis.csv"]["purchase_month"] += "-01"  # so Tableau reads it as a date
    exports["model_monitoring.csv"]["month"] += "-01"
    fi = exports["model_feature_importance.csv"]
    fi.insert(1, "label", fi["feature"].map(FEATURE_LABELS).fillna(fi["feature"]))

    for name, df in exports.items():
        df.to_csv(OUT / name, index=False)
        size_mb = (OUT / name).stat().st_size / 1e6
        print(f"  {name}: {len(df):,} rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
