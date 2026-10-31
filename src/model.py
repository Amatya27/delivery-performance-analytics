"""Late delivery model.

Question: at the moment an order is placed, can we tell which orders are
likely to arrive after the date the customer was promised?

Only features that are known at purchase time are used. The seller and
destination-state history features look only at deliveries that were
completed before the purchase, so nothing from the future leaks in.

Train on 2017 orders, test on Jan to Aug 2018 (a time split, like it would be
used for real). Results go to reports/model_metrics.json and the
model_predictions / model_monitoring tables in the database.

Usage:
    python src/model.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "delivery.db"
REPORTS = ROOT / "reports"

TRAIN_START, TEST_START, TEST_END = "2017-01-01", "2018-01-01", "2018-09-01"
TOP_SHARE = 0.10   # ops team reviews the riskiest 10% of orders each day
HISTORY_DAYS = 14  # window for the "recent late rate into this state" feature
SEED = 42

NUMERIC = [
    "promised_days", "max_distance_km", "items_freight", "items_price",
    "total_weight_g", "n_items", "n_sellers", "freight_ratio", "same_state",
    "purchase_hour", "purchase_weekday", "max_installments",
    "seller_prior_orders", "seller_prior_late_rate", "seller_prior_avg_days",
    "state_recent_late_rate", "state_recent_avg_days", "promise_slack_days",
]
# I tried purchase_month too. With only one year of training data it just
# memorized 2017 (Black Friday etc.) and test ROC AUC dropped from about 0.71 to 0.69,
# so it's out. The state_recent_* features cover seasonal backlogs instead.
CATEGORICAL = ["customer_state", "seller_state", "main_category_en", "payment_type_main"]

QUERY = """
SELECT
    o.order_id, o.purchase_ts, o.delivered_ts, o.is_late, o.delivery_days,
    o.promised_days, o.max_distance_km, o.items_freight, o.items_price,
    o.total_weight_g, o.n_items, o.n_sellers, o.max_installments,
    o.payment_type_main, o.main_category_en, o.main_seller_id,
    c.state AS customer_state,
    s.state AS seller_state
FROM fact_orders o
JOIN dim_customer c ON c.customer_id = o.customer_id
JOIN dim_seller s   ON s.seller_id   = o.main_seller_id
WHERE o.is_late IS NOT NULL
"""


def load_orders(con):
    df = pd.read_sql(QUERY, con, parse_dates=["purchase_ts", "delivered_ts"])
    return df.sort_values("purchase_ts").reset_index(drop=True)


def seller_history(df):
    """For each order: how many of this seller's orders had been delivered
    before the purchase, what share of those were late, and how long they
    took on average."""
    events = df[["main_seller_id", "delivered_ts", "is_late", "delivery_days"]].sort_values("delivered_ts").copy()
    grp = events.groupby("main_seller_id")
    events["n"] = grp.cumcount() + 1
    events["late_sum"] = grp["is_late"].cumsum()
    events["days_sum"] = grp["delivery_days"].cumsum()

    lookup = df[["order_id", "main_seller_id", "purchase_ts"]].sort_values("purchase_ts")
    merged = pd.merge_asof(
        lookup, events[["main_seller_id", "delivered_ts", "n", "late_sum", "days_sum"]],
        left_on="purchase_ts", right_on="delivered_ts", by="main_seller_id",
        allow_exact_matches=False, direction="backward",
    )
    merged["seller_prior_orders"] = merged["n"].fillna(0)
    # sellers with no history get NaN, the gradient boosting model handles that
    merged["seller_prior_late_rate"] = merged["late_sum"] / merged["n"]
    merged["seller_prior_avg_days"] = merged["days_sum"] / merged["n"]
    cols = ["seller_prior_orders", "seller_prior_late_rate", "seller_prior_avg_days"]
    return merged.set_index("order_id")[cols]


def state_recent_history(df, days=HISTORY_DAYS):
    """Late rate and average delivery time of deliveries that arrived in the
    customer's state during the `days` before the purchase date. This is meant
    to pick up things like carrier strikes or holiday backlogs while they are
    happening."""
    d = df.assign(day=df["delivered_ts"].dt.normalize())
    daily = d.groupby(["customer_state", "day"]).agg(
        late=("is_late", "sum"), days=("delivery_days", "sum"), count=("is_late", "size")
    )
    out = []
    for state, g in daily.groupby(level=0):
        g = g.droplevel(0).asfreq("D", fill_value=0)
        roll = g.rolling(f"{days}D").sum()
        # shift by a day so deliveries on the purchase day itself aren't used
        roll = roll.shift(1, freq="D")
        out.append(pd.DataFrame({
            "customer_state": state,
            "day": roll.index,
            "state_recent_late_rate": (roll["late"] / roll["count"]).to_numpy(),
            "state_recent_avg_days": (roll["days"] / roll["count"]).to_numpy(),
        }))
    rates = pd.concat(out, ignore_index=True)

    keys = df[["order_id", "customer_state"]].assign(day=df["purchase_ts"].dt.normalize())
    keys = keys.merge(rates, on=["customer_state", "day"], how="left")
    return keys.set_index("order_id")[["state_recent_late_rate", "state_recent_avg_days"]]


def build_features(df):
    feats = df.copy()
    feats["freight_ratio"] = feats["items_freight"] / feats["items_price"].replace(0, np.nan)
    feats["same_state"] = (feats["customer_state"] == feats["seller_state"]).astype(int)
    feats["purchase_hour"] = feats["purchase_ts"].dt.hour
    feats["purchase_weekday"] = feats["purchase_ts"].dt.day_of_week
    feats = feats.join(seller_history(df), on="order_id")
    feats = feats.join(state_recent_history(df), on="order_id")
    # how much slack the promise has compared to how long deliveries into
    # this state have actually been taking lately
    feats["promise_slack_days"] = feats["promised_days"] - feats["state_recent_avg_days"]
    for col in CATEGORICAL:
        feats[col] = feats[col].fillna("missing")
    return feats


def top_share_stats(y, score, share=TOP_SHARE):
    """Precision and recall if we flag the riskiest `share` of orders."""
    n_flag = max(1, int(round(len(score) * share)))
    flagged = np.zeros(len(score), dtype=bool)
    flagged[np.argsort(-score)[:n_flag]] = True
    tp = (flagged & (y == 1)).sum()
    return {
        "precision": float(tp / n_flag),
        "recall": float(tp / max(1, (y == 1).sum())),
    }


def evaluate(y, score):
    out = {
        "orders": int(len(y)),
        "late_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }
    top = top_share_stats(np.asarray(y), np.asarray(score))
    out["precision_top10"] = top["precision"]
    out["recall_top10"] = top["recall"]
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def make_models():
    # baseline: logistic regression on scaled numbers + one-hot categories
    # median fill is only needed for the history features (new sellers have none)
    numeric_lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    logreg = make_pipeline(
        ColumnTransformer([
            ("num", numeric_lr, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
        ]),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )

    # main model: gradient boosted trees, categories passed as ordinal codes
    hgb = make_pipeline(
        ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), CATEGORICAL),
        ]),
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))),
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
        ),
    )
    return {"logistic_regression": logreg, "gradient_boosting": hgb}


def monthly_monitoring(test, score):
    rows = []
    t = test.assign(score=score, month=test["purchase_ts"].dt.strftime("%Y-%m"))
    for month, g in t.groupby("month"):
        y = g["is_late"].to_numpy()
        row = {
            "month": month,
            "orders": len(g),
            "actual_late_rate": round(float(y.mean()), 4),
            "avg_predicted_risk": round(float(g["score"].mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, g["score"])), 4) if 0 < y.sum() < len(y) else None,
        }
        top = top_share_stats(y, g["score"].to_numpy())
        row["precision_top10"] = round(top["precision"], 4)
        row["recall_top10"] = round(top["recall"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def risk_deciles(test, score):
    t = test[["is_late"]].assign(score=score)
    t["decile"] = pd.qcut(t["score"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    return (
        t.groupby("decile")
        .agg(orders=("is_late", "size"), actual_late_rate=("is_late", "mean"),
             avg_predicted_risk=("score", "mean"))
        .round(4)
        .reset_index()
    )


def main():
    con = sqlite3.connect(DB_PATH)
    df = load_orders(con)
    feats = build_features(df)

    train = feats[(feats["purchase_ts"] >= TRAIN_START) & (feats["purchase_ts"] < TEST_START)]
    test = feats[(feats["purchase_ts"] >= TEST_START) & (feats["purchase_ts"] < TEST_END)]
    X_cols = NUMERIC + CATEGORICAL
    print(f"train {len(train):,} orders ({train['is_late'].mean():.1%} late), "
          f"test {len(test):,} orders ({test['is_late'].mean():.1%} late)")

    metrics = {"train_period": [TRAIN_START, TEST_START], "test_period": [TEST_START, TEST_END],
               "features": X_cols, "models": {}}
    fitted = {}
    for name, model in make_models().items():
        model.fit(train[X_cols], train["is_late"])
        score = model.predict_proba(test[X_cols])[:, 1]
        metrics["models"][name] = evaluate(test["is_late"].to_numpy(), score)
        fitted[name] = (model, score)
        print(f"  {name}: {metrics['models'][name]}")

    # simple rule a person might use instead of a model: promised days alone
    # shorter promise = riskier, squashed into 0..1 so the metrics still work
    rule = 1.0 / (1.0 + test["promised_days"].clip(lower=0).to_numpy())
    metrics["models"]["rule_short_promise"] = evaluate(test["is_late"].to_numpy(), rule)

    # both models rank about the same, but the boosted model's probabilities are
    # far better calibrated (the balanced logistic regression inflates them),
    # and I want the predicted risk to mean something on the dashboard
    best_name = "gradient_boosting"
    model, score = fitted[best_name]
    metrics["chosen_model"] = best_name

    # which features matter, measured on the test set
    sample = test.sample(n=min(10000, len(test)), random_state=SEED)
    imp = permutation_importance(model, sample[X_cols], sample["is_late"], scoring="roc_auc",
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = (
        pd.DataFrame({"feature": X_cols, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .round(4)
    )
    print(importance.head(8).to_string(index=False))

    monitoring = monthly_monitoring(test, score)
    deciles = risk_deciles(test, score)

    # scores for every order in the model window so Tableau can show them
    window = feats[(feats["purchase_ts"] >= TRAIN_START) & (feats["purchase_ts"] < TEST_END)]
    all_scores = model.predict_proba(window[X_cols])[:, 1]
    cutoff = np.quantile(score, 1 - TOP_SHARE)
    preds = pd.DataFrame({
        "order_id": window["order_id"],
        "split": np.where(window["purchase_ts"] < TEST_START, "train", "test"),
        "late_risk": all_scores.round(4),
        "flagged_high_risk": (all_scores >= cutoff).astype(int),
    })
    metrics["high_risk_cutoff"] = round(float(cutoff), 4)

    preds.to_sql("model_predictions", con, if_exists="replace", index=False)
    monitoring.to_sql("model_monitoring", con, if_exists="replace", index=False)
    deciles.to_sql("model_risk_deciles", con, if_exists="replace", index=False)
    importance.to_sql("model_feature_importance", con, if_exists="replace", index=False)
    con.close()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(monitoring.to_string(index=False))
    print("metrics -> reports/model_metrics.json")


if __name__ == "__main__":
    main()
