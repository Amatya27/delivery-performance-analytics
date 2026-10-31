"""ETL: raw Olist CSVs -> cleaned star schema in SQLite (data/delivery.db).

Extract:   read the 9 raw CSVs from data/raw/
Transform: fix types, dedupe, compute delivery timings and distances,
           reconcile payments against item totals, one review per order
Load:      write dimension and fact tables (sql/schema.sql), then build
           the KPI views (sql/kpi_views.sql)

Also writes reports/data_quality.md so I can see what got dropped or flagged.

Usage:
    python src/etl.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "delivery.db"
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"

# rough bounding box for Brazil, a few geolocation rows are clearly wrong
# (points in Europe, the US, the ocean...)
BRAZIL_LAT = (-34.0, 5.5)
BRAZIL_LNG = (-74.0, -34.0)

# if the payments and the items+freight differ by more than this (BRL),
# I flag the order. small differences are usually rounding or vouchers
PAYMENT_TOLERANCE = 1.0

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapa", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceara", "DF": "Distrito Federal", "ES": "Espirito Santo", "GO": "Goias",
    "MA": "Maranhao", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Para", "PB": "Paraiba", "PR": "Parana",
    "PE": "Pernambuco", "PI": "Piaui", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondonia",
    "RR": "Roraima", "SC": "Santa Catarina", "SP": "Sao Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}


# ---------------------------------------------------------------------------
# small helpers (these are the parts covered by tests/test_etl.py)
# ---------------------------------------------------------------------------

def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km. Works on numbers or numpy/pandas arrays."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_geolocation(geo):
    """One lat/lng per zip prefix (median), ignoring points outside Brazil."""
    inside = (
        geo["geolocation_lat"].between(*BRAZIL_LAT)
        & geo["geolocation_lng"].between(*BRAZIL_LNG)
    )
    zips = (
        geo[inside]
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(columns={
            "geolocation_zip_code_prefix": "zip_prefix",
            "geolocation_lat": "lat",
            "geolocation_lng": "lng",
        })
    )
    return zips, int((~inside).sum())


def add_delivery_metrics(orders):
    """Add timing columns and the is_late flag to the orders table.

    is_late compares calendar dates: the estimated date has no time part, so
    an order delivered at 3pm on the estimated day still counts as on time.
    Negative stage durations (carrier date before approval etc.) are data
    entry problems, so I set them to NaN instead of keeping them.
    """
    df = orders.copy()
    day = pd.Timedelta(days=1)

    df["approval_hours"] = (df["approved_ts"] - df["purchase_ts"]) / pd.Timedelta(hours=1)
    df["handling_days"] = (df["carrier_ts"] - df["approved_ts"]) / day
    df["transit_days"] = (df["delivered_ts"] - df["carrier_ts"]) / day
    df["delivery_days"] = (df["delivered_ts"] - df["purchase_ts"]) / day
    df["promised_days"] = (df["estimated_date"].dt.normalize() - df["purchase_ts"].dt.normalize()) / day

    for col in ["approval_hours", "handling_days", "transit_days", "delivery_days"]:
        df.loc[df[col] < 0, col] = np.nan

    delivered = (df["order_status"] == "delivered") & df["delivered_ts"].notna()
    delay = (df["delivered_ts"].dt.normalize() - df["estimated_date"].dt.normalize()) / day
    df["delay_days"] = delay.where(delivered)
    df["is_late"] = (delay > 0).astype("Int64").where(delivered)

    for col in ["approval_hours", "handling_days", "transit_days", "delivery_days"]:
        df[col] = df[col].round(2)
    return df


def reconcile_payments(order_totals, payments, tol=PAYMENT_TOLERANCE):
    """Compare what the customer paid against items + freight for each order."""
    pay = payments.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
    )
    # main payment type = the one that covered the most money
    main_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
        .rename("payment_type_main")
    )
    out = order_totals.join(pay, how="left").join(main_type, how="left")
    out["payment_gap"] = (out["payment_total"] - (out["items_price"] + out["items_freight"])).round(2)
    out["payment_mismatch"] = (out["payment_gap"].abs() > tol).astype(int)
    out.loc[out["payment_total"].isna(), "payment_mismatch"] = 1
    return out


def pick_one_review(reviews):
    """Some orders have more than one review, keep the most recent answer."""
    return (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates("order_id", keep="last")
        [["order_id", "review_score"]]
    )


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def extract():
    def read(name, **kw):
        return pd.read_csv(RAW / name, **kw)

    raw = {
        "customers": read("olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": int}),
        "geolocation": read("olist_geolocation_dataset.csv"),
        "items": read("olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]),
        "payments": read("olist_order_payments_dataset.csv"),
        "reviews": read("olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"]),
        "orders": read(
            "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "products": read("olist_products_dataset.csv"),
        "sellers": read("olist_sellers_dataset.csv"),
        "translation": read("product_category_name_translation.csv"),
    }
    return raw


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def transform(raw):
    dq = {}  # numbers for the data quality report
    dq["raw_rows"] = {k: len(v) for k, v in raw.items()}

    zips, dq["geo_points_outside_brazil"] = clean_geolocation(raw["geolocation"])

    # --- dim_customer
    cust = raw["customers"].drop_duplicates("customer_id").rename(columns={
        "customer_zip_code_prefix": "zip_prefix",
        "customer_city": "city",
        "customer_state": "state",
    })
    cust = cust.merge(zips, on="zip_prefix", how="left")
    cust["state_name"] = cust["state"].map(STATE_NAMES)
    dq["customers_without_coords"] = int(cust["lat"].isna().sum())
    dim_customer = cust[["customer_id", "customer_unique_id", "zip_prefix", "city",
                         "state", "state_name", "lat", "lng"]]

    # --- dim_seller
    sell = raw["sellers"].drop_duplicates("seller_id").rename(columns={
        "seller_zip_code_prefix": "zip_prefix",
        "seller_city": "city",
        "seller_state": "state",
    })
    sell = sell.merge(zips, on="zip_prefix", how="left")
    sell["state_name"] = sell["state"].map(STATE_NAMES)
    dq["sellers_without_coords"] = int(sell["lat"].isna().sum())
    dim_seller = sell[["seller_id", "zip_prefix", "city", "state", "state_name", "lat", "lng"]]

    # --- dim_product
    prod = raw["products"].drop_duplicates("product_id").merge(
        raw["translation"], on="product_category_name", how="left"
    )
    prod = prod.rename(columns={
        "product_category_name": "category_pt",
        "product_category_name_english": "category_en",
        "product_weight_g": "weight_g",
        "product_length_cm": "length_cm",
        "product_height_cm": "height_cm",
        "product_width_cm": "width_cm",
        "product_photos_qty": "photos_qty",
    })
    # a couple of categories have no english name in the translation file
    prod["category_en"] = prod["category_en"].fillna(prod["category_pt"]).fillna("unknown")
    prod["volume_cm3"] = prod["length_cm"] * prod["height_cm"] * prod["width_cm"]
    dim_product = prod[["product_id", "category_pt", "category_en", "weight_g", "length_cm",
                        "height_cm", "width_cm", "volume_cm3", "photos_qty"]]

    # --- fact_order_items (item grain) with seller -> customer distance
    orders = raw["orders"].drop_duplicates("order_id").rename(columns={
        "order_purchase_timestamp": "purchase_ts",
        "order_approved_at": "approved_ts",
        "order_delivered_carrier_date": "carrier_ts",
        "order_delivered_customer_date": "delivered_ts",
        "order_estimated_delivery_date": "estimated_date",
    })
    items = raw["items"].drop_duplicates(["order_id", "order_item_id"])
    items = items.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    items = items.merge(dim_customer[["customer_id", "lat", "lng"]], on="customer_id", how="left")
    items = items.merge(
        dim_seller[["seller_id", "lat", "lng"]].rename(columns={"lat": "s_lat", "lng": "s_lng"}),
        on="seller_id", how="left",
    )
    items["distance_km"] = haversine_km(items["s_lat"], items["s_lng"], items["lat"], items["lng"]).round(1)
    items = items.merge(dim_product[["product_id", "category_en", "weight_g"]], on="product_id", how="left")

    fact_items = items[["order_id", "order_item_id", "product_id", "seller_id",
                        "shipping_limit_date", "price", "freight_value", "distance_km"]]

    # --- roll items up to order level
    by_order = items.groupby("order_id")
    totals = pd.DataFrame({
        "n_items": by_order.size(),
        "n_sellers": by_order["seller_id"].nunique(),
        "items_price": by_order["price"].sum().round(2),
        "items_freight": by_order["freight_value"].sum().round(2),
        "total_weight_g": by_order["weight_g"].sum(min_count=1),
        "max_distance_km": by_order["distance_km"].max(),
    })
    # the most expensive item decides the "main" seller and category
    main = (
        items.sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")
        .set_index("order_id")[["seller_id", "product_id", "category_en"]]
        .rename(columns={"seller_id": "main_seller_id", "product_id": "main_product_id",
                         "category_en": "main_category_en"})
    )
    totals = totals.join(main)

    recon = reconcile_payments(totals, raw["payments"])
    dq["orders_with_items"] = len(recon)
    dq["payment_mismatches"] = int(recon["payment_mismatch"].sum())
    inst = (recon["payment_mismatch"] == 1) & (recon["max_installments"] > 1)
    dq["payment_mismatch_installments"] = int(inst.sum())
    dq["installment_mismatch_overpaid"] = int((inst & (recon["payment_gap"] > 0)).sum())
    dq["orders_missing_payment"] = int(recon["payment_total"].isna().sum())

    # --- fact_orders (order grain)
    orders = add_delivery_metrics(orders)
    dq["orders_without_items"] = int((~orders["order_id"].isin(recon.index)).sum())
    dq["delivered_missing_date"] = int(
        ((orders["order_status"] == "delivered") & orders["delivered_ts"].isna()).sum()
    )
    dq["negative_handling_or_transit"] = int(
        (((orders["carrier_ts"] - orders["approved_ts"]) < pd.Timedelta(0))
         | ((orders["delivered_ts"] - orders["carrier_ts"]) < pd.Timedelta(0))).sum()
    )

    reviews = pick_one_review(raw["reviews"])
    dq["orders_with_multiple_reviews"] = int(raw["reviews"]["order_id"].duplicated().sum())

    fact = orders.merge(recon, left_on="order_id", right_index=True, how="left")
    fact = fact.merge(reviews, on="order_id", how="left")
    fact["purchase_date_key"] = fact["purchase_ts"].dt.strftime("%Y%m%d").astype(int)

    fact_orders = fact[[
        "order_id", "customer_id", "purchase_date_key", "order_status",
        "purchase_ts", "approved_ts", "carrier_ts", "delivered_ts", "estimated_date",
        "n_items", "n_sellers", "main_seller_id", "main_product_id", "main_category_en",
        "items_price", "items_freight", "total_weight_g", "max_distance_km",
        "payment_total", "payment_type_main", "max_installments", "payment_gap",
        "payment_mismatch", "review_score", "approval_hours", "handling_days",
        "transit_days", "delivery_days", "promised_days", "delay_days", "is_late",
    ]]

    # --- dim_date, every day between the first and last purchase
    days = pd.date_range(fact["purchase_ts"].min().normalize(), fact["purchase_ts"].max().normalize())
    dim_date = pd.DataFrame({
        "date_key": days.strftime("%Y%m%d").astype(int),
        "date": days.strftime("%Y-%m-%d"),
        "year": days.year,
        "quarter": days.quarter,
        "month": days.month,
        "month_name": days.strftime("%b"),
        "day": days.day,
        "weekday": days.strftime("%a"),
        "is_weekend": (days.day_of_week >= 5).astype(int),
    })

    tables = {
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_seller": dim_seller,
        "dim_product": dim_product,
        "fact_orders": fact_orders,
        "fact_order_items": fact_items,
    }
    return tables, dq


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load(tables, db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch every run, it only takes a minute

    con = sqlite3.connect(db_path)
    con.executescript((SQL_DIR / "schema.sql").read_text())
    for name, df in tables.items():
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_sql(name, con, if_exists="append", index=False)
    con.executescript((SQL_DIR / "kpi_views.sql").read_text())
    con.commit()

    counts = {name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in tables}
    con.close()
    return counts


def write_dq_report(dq, counts):
    REPORTS.mkdir(exist_ok=True)
    lines = ["# Data quality report", "", "Generated by `src/etl.py`.", "", "## Raw rows", ""]
    lines += [f"- {k}: {v:,}" for k, v in dq["raw_rows"].items()]
    lines += ["", "## Loaded rows", ""]
    lines += [f"- {k}: {v:,}" for k, v in counts.items()]
    lines += [
        "",
        "## Checks",
        "",
        f"- geolocation points outside Brazil (dropped): {dq['geo_points_outside_brazil']:,}",
        f"- customers with no coordinates for their zip prefix: {dq['customers_without_coords']:,}",
        f"- sellers with no coordinates for their zip prefix: {dq['sellers_without_coords']:,}",
        f"- orders with no items (mostly canceled/unavailable): {dq['orders_without_items']:,}",
        f"- delivered orders missing a delivery date: {dq['delivered_missing_date']:,}",
        f"- orders with a negative handling or transit time (set to null): {dq['negative_handling_or_transit']:,}",
        f"- orders with more than one review (kept latest): {dq['orders_with_multiple_reviews']:,}",
        "",
        "## Payment reconciliation",
        "",
        f"Compared payment total vs items + freight for {dq['orders_with_items']:,} orders "
        f"(tolerance {PAYMENT_TOLERANCE} BRL).",
        "",
        f"- mismatched orders: {dq['payment_mismatches']:,}",
        f"- of those, paid in installments: {dq['payment_mismatch_installments']:,}, "
        f"and {dq['installment_mismatch_overpaid']:,} of them paid more than the item total. "
        "My guess is installment interest, the dataset doesn't say.",
        f"- orders with items but no payment record: {dq['orders_missing_payment']:,}",
        "",
    ]
    (REPORTS / "data_quality.md").write_text("\n".join(lines))


def main():
    print("extracting ...")
    raw = extract()
    print("transforming ...")
    tables, dq = transform(raw)
    print("loading into", DB_PATH.relative_to(ROOT), "...")
    counts = load(tables)
    write_dq_report(dq, counts)
    for name, n in counts.items():
        print(f"  {name}: {n:,} rows")
    print("data quality report -> reports/data_quality.md")


if __name__ == "__main__":
    main()
