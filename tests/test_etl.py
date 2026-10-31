"""Unit tests for the ETL helpers. They use tiny made-up frames, so they run
without downloading the dataset (that's what CI runs).

The last test checks the real database and is skipped if you haven't run
src/etl.py yet.

Run with:  pytest -q
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import etl  # noqa: E402


def ts(s):
    return pd.Timestamp(s)


def make_orders(rows):
    cols = ["order_id", "order_status", "purchase_ts", "approved_ts", "carrier_ts",
            "delivered_ts", "estimated_date"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[2:]:
        df[c] = pd.to_datetime(df[c])
    return df


def test_haversine_sao_paulo_to_rio():
    # city centers are roughly 360 km apart
    d = etl.haversine_km(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 350 < d < 370


def test_haversine_same_point_is_zero():
    assert etl.haversine_km(-10.0, -50.0, -10.0, -50.0) == pytest.approx(0.0)


def test_geolocation_drops_points_outside_brazil():
    geo = pd.DataFrame({
        "geolocation_zip_code_prefix": [1001, 1001, 1001, 2002],
        "geolocation_lat": [-23.55, -23.56, 40.0, -22.9],   # 40.0 is clearly not Brazil
        "geolocation_lng": [-46.63, -46.64, -3.7, -43.17],
    })
    zips, dropped = etl.clean_geolocation(geo)
    assert dropped == 1
    assert len(zips) == 2
    row = zips.set_index("zip_prefix").loc[1001]
    assert row["lat"] == pytest.approx(-23.555)


def test_delivered_on_estimated_day_is_on_time():
    orders = make_orders([
        ["a", "delivered", "2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00",
         "2018-01-10 15:00", "2018-01-10 00:00"],
    ])
    out = etl.add_delivery_metrics(orders)
    assert out.loc[0, "is_late"] == 0
    assert out.loc[0, "delay_days"] == 0
    assert out.loc[0, "promised_days"] == 9


def test_delivered_day_after_is_late():
    orders = make_orders([
        ["b", "delivered", "2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00",
         "2018-01-11 08:00", "2018-01-10 00:00"],
    ])
    out = etl.add_delivery_metrics(orders)
    assert out.loc[0, "is_late"] == 1
    assert out.loc[0, "delay_days"] == 1


def test_not_delivered_has_no_late_flag():
    orders = make_orders([
        ["c", "shipped", "2018-01-01 10:00", "2018-01-01 11:00", "2018-01-02 09:00",
         None, "2018-01-10 00:00"],
    ])
    out = etl.add_delivery_metrics(orders)
    assert pd.isna(out.loc[0, "is_late"])
    assert pd.isna(out.loc[0, "delay_days"])


def test_negative_stage_time_becomes_null():
    # carrier date before approval, this happens in the raw data
    orders = make_orders([
        ["d", "delivered", "2018-01-01 10:00", "2018-01-03 11:00", "2018-01-02 09:00",
         "2018-01-08 15:00", "2018-01-10 00:00"],
    ])
    out = etl.add_delivery_metrics(orders)
    assert np.isnan(out.loc[0, "handling_days"])
    assert out.loc[0, "transit_days"] > 0


def test_payment_reconciliation_flags_big_gaps_only():
    totals = pd.DataFrame(
        {"items_price": [100.0, 100.0, 50.0], "items_freight": [10.0, 10.0, 5.0]},
        index=pd.Index(["ok", "off", "nopay"], name="order_id"),
    )
    payments = pd.DataFrame({
        "order_id": ["ok", "off", "off"],
        "payment_type": ["credit_card", "credit_card", "voucher"],
        "payment_installments": [1, 3, 1],
        "payment_value": [110.40, 100.0, 25.0],   # "off" paid 125 for a 110 order
    })
    out = etl.reconcile_payments(totals, payments)
    assert out.loc["ok", "payment_mismatch"] == 0
    assert out.loc["off", "payment_mismatch"] == 1
    assert out.loc["off", "payment_gap"] == pytest.approx(15.0)
    assert out.loc["off", "payment_type_main"] == "credit_card"
    assert out.loc["nopay", "payment_mismatch"] == 1


def test_pick_one_review_keeps_latest():
    reviews = pd.DataFrame({
        "order_id": ["x", "x", "y"],
        "review_score": [1, 4, 5],
        "review_answer_timestamp": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-01-05"]),
    })
    out = etl.pick_one_review(reviews).set_index("order_id")
    assert out.loc["x", "review_score"] == 4
    assert len(out) == 2


@pytest.mark.skipif(not etl.DB_PATH.exists(), reason="run src/etl.py first")
def test_built_database_is_consistent():
    con = sqlite3.connect(etl.DB_PATH)
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731

    # one row per order, and every order points at a real customer
    assert q("SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM fact_orders") == 0
    assert q("""SELECT COUNT(*) FROM fact_orders o
                LEFT JOIN dim_customer c ON c.customer_id = o.customer_id
                WHERE c.customer_id IS NULL""") == 0
    # item totals in the order table match the item table
    assert q("""SELECT COUNT(*) FROM fact_orders o
                JOIN (SELECT order_id, ROUND(SUM(price), 2) p FROM fact_order_items GROUP BY order_id) i
                  ON i.order_id = o.order_id
                WHERE ABS(o.items_price - i.p) > 0.01""") == 0
    # late flag only exists for delivered orders
    assert q("SELECT COUNT(*) FROM fact_orders WHERE is_late IS NOT NULL AND order_status != 'delivered'") == 0
    # distances should be inside Brazil-sized numbers
    assert q("SELECT MAX(distance_km) FROM fact_order_items") < 5000
    con.close()
