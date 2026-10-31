"""Download the Olist public e-commerce dataset into data/raw/.

The files come from Olist's own GitHub repo (MIT license):
https://github.com/olist/work-at-olist-data

Usage:
    python src/get_data.py
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists() and target.stat().st_size > 0:
            print(f"already have {name}, skipping")
            continue
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(BASE_URL + name, target)
        except Exception as err:
            # don't leave a half-written file behind, the next run would skip it
            print(f"  failed: {err}")
            if target.exists():
                target.unlink()
            sys.exit(1)
    print("done, files are in", RAW_DIR)


if __name__ == "__main__":
    main()
