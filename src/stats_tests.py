"""A few statistical checks on the delivered orders.

1. Do late orders get worse reviews? (Mann-Whitney U, since review scores
   are 1 to 5 and not normal, plus a bootstrap CI for the difference in means)
2. Is the share of 1 and 2 star reviews different? (chi-square test)
3. Does distance go with delivery time? (Spearman correlation)

Writes reports/stats_summary.json.

Usage:
    python src/stats_tests.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "delivery.db"
REPORTS = ROOT / "reports"
SEED = 42


def bootstrap_mean_diff(a, b, n_boot=2000, seed=SEED):
    """95% bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT is_late, review_score, delivery_days, max_distance_km
        FROM fact_orders
        WHERE is_late IS NOT NULL
        """,
        con,
    )
    con.close()

    rev = df.dropna(subset=["review_score"])
    late = rev.loc[rev["is_late"] == 1, "review_score"].to_numpy()
    on_time = rev.loc[rev["is_late"] == 0, "review_score"].to_numpy()

    u = stats.mannwhitneyu(late, on_time, alternative="less")
    ci = bootstrap_mean_diff(late, on_time)

    low = rev["review_score"] <= 2
    table = pd.crosstab(rev["is_late"], low)
    chi2, chi_p, _, _ = stats.chi2_contingency(table)

    dist = df.dropna(subset=["max_distance_km", "delivery_days"])
    rho, rho_p = stats.spearmanr(dist["max_distance_km"], dist["delivery_days"])

    summary = {
        "orders_with_review": int(len(rev)),
        "late_orders": int(len(late)),
        "on_time_orders": int(len(on_time)),
        "mean_review_late": round(float(late.mean()), 3),
        "mean_review_on_time": round(float(on_time.mean()), 3),
        "mean_diff_ci95": [round(ci[0], 3), round(ci[1], 3)],
        "mann_whitney_p": float(u.pvalue),
        "low_review_share_late": round(float((late <= 2).mean()), 4),
        "low_review_share_on_time": round(float((on_time <= 2).mean()), 4),
        "chi2": round(float(chi2), 1),
        "chi2_p": float(chi_p),
        "spearman_distance_vs_delivery_days": round(float(rho), 3),
        "spearman_p": float(rho_p),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "stats_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
